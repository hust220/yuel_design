# Third-party imports
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from tqdm import tqdm

# Local imports
from src.noise import GammaNetwork, PredefinedNoiseSchedule
from src.e3former import E3former

def expand_to_nodes(array, target):
    """
    Expands the array to match the target's node dimension.
    
    Args:
        array: Input tensor (scalar, [1], or [N])
        target: Target tensor with shape [N, ...]
        
    Returns:
        Tensor with shape [N] matching the target's node dimension
    """
    if array.dim() == 0:  # scalar
        return array.expand(target.size(0))
    elif array.dim() == 1 and array.size(0) == 1:  # [1] -> expand to [N]
        return array.expand(target.size(0))
    else:  # already the right shape
        return array

def sigma(gamma, target_tensor):
    """Computes sigma given gamma."""
    sigma_val = expand_to_nodes(torch.sqrt(torch.sigmoid(gamma)), target_tensor)
    if sigma_val.dim() == 1:
        sigma_val = sigma_val[:, None]
    return sigma_val

def alpha(gamma, target_tensor):
    """Computes alpha given gamma."""
    alpha_val = expand_to_nodes(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    if alpha_val.dim() == 1:
        alpha_val = alpha_val[:, None]
    return alpha_val

def sigma_and_alpha_t_given_s(gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = expand_to_nodes(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)),
        target_tensor
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = expand_to_nodes(alpha_t_given_s, target_tensor)
    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

class ContModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Extract parameters with defaults
        timesteps = kwargs.get('diffusion_steps', 100)
        noise_schedule = kwargs.get('diffusion_noise_schedule', 'polynomial_2')
        noise_precision = kwargs.get('diffusion_noise_precision', 1e-5)
        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.T = timesteps
        self.t_weight_power = kwargs.get('t_weight_power', 0.0)

        self.n_blocks = kwargs['n_blocks']

        hidden_nf = kwargs['hidden_nf']
        in_node_features = kwargs['in_node_features']
        in_edge_features = kwargs['in_edge_features']
        
        # E3former initialization
        self.e3former = E3former(
            seq_input_dim=in_node_features + 2,  # +2 for mask and time
            z_input_dim=in_edge_features,
            c_m=hidden_nf,
            c_z=hidden_nf,
            c_hidden_seq_att=kwargs.get('c_hidden_seq_att', hidden_nf // 2),
            c_hidden_opm=kwargs.get('c_hidden_opm', hidden_nf // 2),
            c_hidden_mul=kwargs.get('c_hidden_mul', hidden_nf // 2),
            c_hidden_pair_att=kwargs.get('c_hidden_pair_att', hidden_nf // 2),
            no_heads_seq=kwargs.get('no_heads_seq', 8),
            no_heads_pair=kwargs.get('no_heads_pair', 4),
            no_blocks=self.n_blocks,
            transition_n=kwargs.get('transition_n', 4),
            blocks_per_ckpt=kwargs.get('blocks_per_ckpt', self.n_blocks),
            eps=kwargs.get('eps', 1e-8),
        )
        
        self.chunk_size = kwargs.get('chunk_size', None)     

    def sample_weighted_t(self, device):
        # Compute weights and probabilities
        t_weights = torch.arange(1, self.T + 1, dtype=torch.float32, device=device) ** self.t_weight_power
        t_probs = t_weights / t_weights.sum()
        
        # Use torch.multinomial for weighted sampling - only sample one time step
        t_index = torch.multinomial(t_probs, 1, replacement=True)
        t_int = (t_index + 1).float()  # +1 because index starts from 0 but t starts from 1
        return t_int

    def compute_fix_mean(self, x0, fix_mask):
        """Compute fixed atoms mean for each sample in batch (vectorized)
        
        Args:
            x0: (B, N, 3) - coordinates
            fix_mask: (B, N) - fix mask (1 for fixed atoms, 0 for sampling atoms)
        
        Returns:
            (B, 1, 3) - fixed atoms mean for each sample
        """
        fix_mask_float = (fix_mask == 1).unsqueeze(-1).float()  # (B, N, 1)
        fix_sum = (x0 * fix_mask_float).sum(dim=1, keepdim=True)  # (B, 1, 3)
        fix_count = fix_mask_float.sum(dim=1, keepdim=True)  # (B, 1, 1)
        
        # If no fixed atoms, use all atoms mean; otherwise use fixed atoms mean
        has_fix = (fix_count > 0).squeeze(-1)  # (B, 1)
        fix_mean = torch.where(
            has_fix.unsqueeze(-1),  # (B, 1, 1)
            fix_sum / fix_count.clamp(min=1),
            x0.mean(dim=1, keepdim=True)
        )  # (B, 1, 3)
        
        return fix_mean

    def model_predict(self, xt, t, data):
        """Helper function to run E3former forward pass
        
        Args:
            xt: [batch_size, n, 3] - noisy coordinates
            t: scalar - time step
            data: dict with batched tensors (batch_size, n, ...)
        
        Returns:
            [batch_size, n, 3] - predicted coordinate update
        """
        h = data['h']  # [batch_size, n, h_dim]
        mask = data['mask']  # [batch_size, n]
        z = data['z']  # [batch_size, n, n, z_dim]
        seq_mask = data['seq_mask']  # [batch_size, n]
        pair_mask = data['pair_mask']  # [batch_size, n, n]

        batch_size, n_nodes = h.shape[0], h.shape[1]
        mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, n, 1]
        t_expanded = t.expand(batch_size, n_nodes, 1).to(h.device)  # [batch_size, n, 1]
        seq = torch.cat([h, mask_expanded, t_expanded], dim=-1)  # [batch_size, n, h_dim + 2]
        
        x_in = xt.clone()  # Save input coordinates
        
        # E3former forward: forward(seq, x, z, seq_mask, pair_mask, chunk_size) -> x
        x_out = self.e3former.forward(
            seq=seq,
            x=x_in,
            z=z,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            chunk_size=self.chunk_size,
        )
        
        # Return coordinate update
        dx = x_out - x_in
        
        return dx

    def forward(self, data, training=None):
        """Forward pass with batched data (B, N, 3)"""
        x0 = data['x']  # (B, N, 3)
        seq_mask = data['seq_mask']  # (B, N)
        pair_mask = data['pair_mask']  # (B, N, N)
        fix_mask = data['mask']  # (B, N)
        batch_size = x0.size(0)
        
        fix_mask_2d = fix_mask.unsqueeze(-1)  # (B, N, 1)
        sample_mask = (1.0 - fix_mask_2d) * seq_mask.unsqueeze(-1)
        
        # Compute fixed atoms mean and center coordinates
        fix_mean = self.compute_fix_mean(x0, fix_mask)  # (B, 1, 3)
        x0 = x0 - fix_mean  # (B, N, 3)
        
        # Sample t for the batch (same t for all samples)
        t_int = self.sample_weighted_t(x0.device)
        t = t_int / self.T
        
        # Compute alpha_t and sigma_t and expand to (B, N, 1)
        gamma_t = self.gamma(t)  # scalar
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t)).expand(batch_size, x0.size(1), 1)
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t)).expand(batch_size, x0.size(1), 1)
        
        # Sample noise
        eps_t = torch.randn_like(x0) * sample_mask
        
        # Sample z_t given x
        xt = alpha_t * x0 + sigma_t * eps_t
        
        # Predict noise (model predicts coordinate update, interpret as noise)
        eps_hat = self.model_predict(xt, t, data) * sample_mask
        
        # Compute loss (only for valid nodes using mask)
        loss = ((eps_t - eps_hat) ** 2).sum() / sample_mask.sum()

        return {'loss': loss}

    @torch.no_grad()
    def sample_chain(self, data, keep_frames=None):
        """Unified sample_chain method that handles unbatched data (single sample)"""
        x0 = data['x']  # [N, 3]
        seq_mask = data['seq_mask']  # [N] - valid atoms mask (all 1s for single sample)
        pair_mask = data['pair_mask']  # [N, N] - valid pairs mask (all 1s for single sample)
        fix_mask = data['mask']  # [N] - fixed atoms mask
        fix_mask_2d = fix_mask[:, None]  # [N, 1]
        sample_mask = (1.0 - fix_mask_2d) * seq_mask.unsqueeze(-1)  # [N, 1]
        fix_mean = x0[fix_mask == 1].mean(dim=0, keepdim=True)
        x0 = x0 - fix_mean
        x = x0 * fix_mask_2d + torch.randn_like(x0) * sample_mask

        data_batch = {k: v.unsqueeze(0) if v.dim() > 0 else v for k, v in data.items()}

        chain = []

        # Sample p(z_s | z_t) - treat as single graph
        for s_step in tqdm(reversed(range(0, self.T)), desc="Diffusion sampling", total=self.T):
            # For single graph, keep time steps as scalars [1] for consistency
            s_val = torch.tensor([s_step], dtype=torch.float32, device=x.device)  # [1]
            t_val = s_val + 1
            
            s = s_val / self.T  # [1] - normalized time step
            t = t_val / self.T  # [1] - normalized time step

            gamma_t = expand_to_nodes(self.gamma(t), x)
            alpha_t = alpha(gamma_t, x)
            x = alpha_t * x0 * fix_mask_2d + x * sample_mask # [N, 3]

            x = self.sample_p_zs_given_zt(s, t, x, data_batch)
            # x = x - x.mean(dim=0, keepdim=True)

            chain.append(x + fix_mean)

        return chain[-1], torch.stack(chain, dim=0)

    def sample_p_zs_given_zt(self, s, t, xt, data_batch):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only linker features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = sigma_and_alpha_t_given_s(gamma_t, gamma_s, xt)
        sigma_s = sigma(gamma_s, target_tensor=xt)
        sigma_t = sigma(gamma_t, target_tensor=xt)

        # Neural net prediction
        eps_hat = self.model_predict(xt.unsqueeze(0), t.unsqueeze(0), data_batch).squeeze(0)
        # print(eps_hat[5, 1])

        # Compute mu for p(z_s | z_t)
        alpha_t_given_s = alpha_t_given_s[:, None]  
        sigma2_t_given_s = sigma2_t_given_s[:, None]
        mu = xt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat # [N, F]

        # Compute sigma for p(z_s | z_t)
        # sigma_t_given_s: [N], sigma_s: [N, 1], sigma_t: [N, 1]
        sigma_t_given_s = sigma_t_given_s[:, None]  # [N] -> [N, 1] for broadcasting
        sigma_sampling = sigma_t_given_s * sigma_s / sigma_t # [N, 1]
        eps = torch.randn_like(mu)# [N, F]
        # eps: [N, F], sigma_sampling: [N, F], mu: [N, F]
        xt = mu + sigma_sampling * eps # [N, F]

        return xt

    








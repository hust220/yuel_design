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

class LigandModel(torch.nn.Module):
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
        
        # E3former block
        # seq_input_dim includes in_node_features + receptor_mask (1) + time (1)
        # z_input_dim includes in_edge_features + time (1)
        self.e3former = E3former(
            seq_input_dim=in_node_features + 2,
            z_input_dim=in_edge_features + 1,
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
            blocks_per_ckpt=kwargs.get('blocks_per_ckpt', 4),
            inf=kwargs.get('inf', 1e9),
            eps=kwargs.get('eps', 1e-10),
        )
        
        self.chunk_size = kwargs.get('chunk_size', 4)   

        self.device = kwargs.get('device', 'cuda')     

    def sample_weighted_t(self, device):
        # Compute weights and probabilities
        t_weights = torch.arange(1, self.T + 1, dtype=torch.float32, device=device) ** self.t_weight_power
        t_probs = t_weights / t_weights.sum()
        
        # Use torch.multinomial for weighted sampling - only sample one time step
        t_index = torch.multinomial(t_probs, 1, replacement=True)
        t_int = (t_index + 1).float()  # +1 because index starts from 0 but t starts from 1
        return t_int

    def compute_receptor_mean(self, x0, receptor_mask):
        """Compute receptor mean for each sample in batch (vectorized)
        
        Args:
            x0: (B, N, 3) - coordinates
            receptor_mask: (B, N) - receptor mask (1 for receptor, 0 for ligand)
        
        Returns:
            (B, 1, 3) - receptor mean for each sample
        """
        receptor_mask_float = (receptor_mask == 1).unsqueeze(-1).float()  # (B, N, 1)
        receptor_sum = (x0 * receptor_mask_float).sum(dim=1, keepdim=True)  # (B, 1, 3)
        receptor_count = receptor_mask_float.sum(dim=1, keepdim=True)  # (B, 1, 1)
        
        # If no receptor atoms, use all atoms mean; otherwise use receptor mean
        has_receptor = (receptor_count > 0).squeeze(-1)  # (B, 1)
        receptor_mean = torch.where(
            has_receptor.unsqueeze(-1),  # (B, 1, 1)
            receptor_sum / receptor_count.clamp(min=1),
            x0.mean(dim=1, keepdim=True)
        )  # (B, 1, 3)
        
        return receptor_mean

    def _egnn_forward(self, xt, t, data):
        """Helper function to run E3former forward pass with batched data
        
        Args:
            xt: (B, N, 3) - noisy coordinates
            t: scalar - time step
            data: dict with batched tensors (B, N, ...)
        
        Returns:
            (B, N, 3) - predicted noise
        """
        seq = data['seq']  # (B, N, seq_dim)
        z = data['z']  # (B, N, N, z_dim)
        seq_mask = data['seq_mask']  # (B, N)
        pair_mask = data['pair_mask']  # (B, N, N)
        receptor_mask = data['mask']  # (B, N)
        
        batch_size, n_nodes = seq.shape[0], seq.shape[1]
        
        # Add time step to seq: (B, N, seq_dim) -> (B, N, seq_dim + 2)
        t_expanded = t.expand(batch_size, n_nodes, 1)  # (B, N, 1)
        seq_with_t = torch.cat([seq, receptor_mask.unsqueeze(-1).float(), t_expanded], dim=-1)
        
        # Add time step to z: (B, N, N, z_dim) -> (B, N, N, z_dim + 1)
        t_pair = t.expand(batch_size, n_nodes, n_nodes, 1)  # (B, N, N, 1)
        z_with_t = torch.cat([z, t_pair], dim=-1)
        
        # Run E3former (already expects batched input)
        x_out = self.e3former(seq_with_t, xt, z_with_t, seq_mask, pair_mask, self.chunk_size)
        
        return x_out - xt

    def forward(self, data, training=None):
        """Forward pass with batched data (B, N, 3)"""
        x0 = data['x']  # (B, N, 3)
        receptor_mask = data['mask']  # (B, N)
        batch_size = x0.size(0)
        
        receptor_mask_2d = receptor_mask.unsqueeze(-1)  # (B, N, 1)
        sample_mask = 1.0 - receptor_mask_2d
        
        # Compute receptor mean and center coordinates
        receptor_mean = self.compute_receptor_mean(x0, receptor_mask)  # (B, 1, 3)
        x0 = x0 - receptor_mean  # (B, N, 3)
        
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
        
        # Predict noise with batched forward
        eps_hat = self._egnn_forward(xt, t, data)
        
        # Compute loss (only for valid nodes using mask)
        loss = ((eps_t - eps_hat) ** 2).sum() / sample_mask.sum()

        return {'loss': loss}

    @torch.no_grad()
    def sample_chain(self, data, keep_frames=None):
        """Unified sample_chain method that handles raw data"""
        x0 = data['x']
        receptor_mask = data['mask']  # [N]
        receptor_mask_2d = receptor_mask[:, None]  # [N, 1]
        sample_mask = 1.0 - receptor_mask_2d
        receptor_mean = x0[receptor_mask == 1].mean(dim=0, keepdim=True)
        x0 = x0 - receptor_mean
        x = x0 * receptor_mask_2d + torch.randn_like(x0) * sample_mask

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
            x = alpha_t * x0 * receptor_mask_2d + x * sample_mask # [N, 3]

            x = self.sample_p_zs_given_zt(s, t, x, data_batch)
            # x = x - x.mean(dim=0, keepdim=True)

            chain.append(x + receptor_mean)

        # x = self.sample_p_xh_given_z0(x, data_batch)
        # chain.append(x + receptor_mean)

        return chain[-1], torch.stack(chain, dim=0)

    def sample_p_zs_given_zt(self, s, t, xt, data_batch):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only linker features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = sigma_and_alpha_t_given_s(gamma_t, gamma_s, xt)
        sigma_s = sigma(gamma_s, target_tensor=xt)
        sigma_t = sigma(gamma_t, target_tensor=xt)

        # Neural net prediction
        eps_hat = self._egnn_forward(xt.unsqueeze(0), t.unsqueeze(0), data_batch).squeeze(0)

        # Compute mu for p(z_s | z_t)
        alpha_t_given_s = alpha_t_given_s[:, None]  
        sigma2_t_given_s = sigma2_t_given_s[:, None]
        mu = xt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat # [N, F]

        # Compute sigma for p(z_s | z_t)
        # sigma_t_given_s: [N], sigma_s: [N, 1], sigma_t: [N, 1]
        sigma_t_given_s = sigma_t_given_s[:, None]  # [N] -> [N, 1] for broadcasting
        sigma_sampling = sigma_t_given_s * sigma_s / sigma_t # [N, 1]
        mask = data_batch['mask'].view(-1, 1)
        eps = torch.randn_like(mu) * (1.0 - mask) # [N, F]
        # eps: [N, F], sigma_sampling: [N, F], mu: [N, F]
        xt = mu + sigma_sampling * eps # [N, F]

        return xt

    def sample_p_xh_given_z0(self, xt, data_batch):
        """Samples x ~ p(x|z0). Samples only linker features and coords"""
        zeros = torch.zeros(1, device=xt.device)  # [1] for single graph
        gamma_0 = self.gamma(zeros) # [1]

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = torch.exp(0.5 * gamma_0)  # [1]
        
        # Neural net prediction
        eps_hat = self._egnn_forward(xt.unsqueeze(0), zeros.unsqueeze(0), data_batch).squeeze(0)

        mu = self.compute_x_pred(eps_t=eps_hat, xt=xt, gamma_t=gamma_0)
        mask = data_batch['mask'].view(-1, 1)
        eps = torch.randn_like(mu) * (1.0 - mask)
        xt = mu + sigma_x * eps

        return xt

    def compute_x_pred(self, eps_t, xt, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = sigma(gamma_t, target_tensor=eps_t)
        alpha_t = alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (xt - sigma_t * eps_t)
        return x_pred
    








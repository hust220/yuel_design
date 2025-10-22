# Standard library imports
import math

# Third-party imports
import torch
import torch.nn.functional as F

# Local imports
from . import utils
from .egnn import EGNN
from .noise import GammaNetwork, PredefinedNoiseSchedule


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

class EDM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Get dataset configuration parameters from kwargs
        in_node_nf = kwargs.get('num_node_features')
        node_attr_nf = kwargs.get('num_node_attr_features')
        edge_feat_nf = kwargs.get('num_edge_features')
        bidirectional = kwargs.get('bidirectional')
        n_dims = kwargs.get('n_dims')
        low_memory = kwargs.get('low_memory', False)
        
        # Dynamics parameters
        device = kwargs.get('device')

        # Create EGNN
        self.egnn = EGNN(
            in_node_nf=in_node_nf,  # original node features
            node_attr_nf=node_attr_nf + 1,  # +1 for time (added in _egnn_forward)
            hidden_nf=kwargs.get('nf', 64),
            inv_sublayers=kwargs.get('inv_sublayers', 1),
            out_node_nf=in_node_nf,  # output positions + features
            edge_feat_nf=edge_feat_nf,  # no time added to edge features
            activation=kwargs.get('activation', 'silu'),
            n_layers=kwargs.get('n_layers', 16),
            attention=kwargs.get('attention', True),
            tanh=kwargs.get('tanh', False),
            norm_constant=kwargs.get('norm_constant', 0),
            normalization_factor=kwargs.get('normalization_factor', 1),
            aggregation_method=kwargs.get('aggregation_method', 'sum'),
            bidirectional=bidirectional,
            low_memory=low_memory,
        )
        
        # Extract parameters with defaults
        timesteps = kwargs.get('diffusion_steps', 100)
        noise_schedule = kwargs.get('diffusion_noise_schedule', 'polynomial_2')
        noise_precision = kwargs.get('diffusion_noise_precision', 1e-5)
        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.t_weight_power = kwargs.get('t_weight_power', 0.0)
        self.coord_scale_factor = kwargs.get('coord_scale_factor', 0.2)
        self.data_augmentation = kwargs.get('data_augmentation', False)

    def sample_weighted_t(self, device):
        """使用加权采样来采样时间步 t,让更大的 t 值有更高的概率 - 单图版本"""
        # 动态计算权重和概率
        t_weights = torch.arange(1, self.T + 1, dtype=torch.float32, device=device) ** self.t_weight_power
        t_probs = t_weights / t_weights.sum()
        
        # 使用 torch.multinomial 进行加权采样 - 只采样一个时间步
        t_index = torch.multinomial(t_probs, 1, replacement=True)
        t_int = (t_index + 1).float()  # +1 因为索引从0开始，但t从1开始
        return t_int

    def _egnn_forward(self, z, t, graph):
        """Helper function to run EGNN forward pass"""
        # Add time step to node_attr for flattened data
        # t shape: [1], z shape: [N, F], node_mask shape: [N]
        N, F = z.shape
        t_expanded = t.expand(N, 1)  # [N, 1]
        
        # Add time step to node_attr instead of directly to features
        node_attr_with_time = torch.cat([graph.ndata['node_attr'], t_expanded], dim=1)

        free_mask = 1.0 - graph.ndata['anchor_mask']

        # Neural net prediction
        h_out, x_out = self.egnn.forward(
            h=z[:, self.n_dims:],  # features only (after coordinates)
            x=z[:, :self.n_dims],  # positions
            edge_index=graph.edge_index,  # Use custom graph edge_index directly
            edge_attr=graph.edata['edge_attr'],
            node_attr=node_attr_with_time,  # node_attr with time step added
            node_mask=graph.ndata['node_mask'],
            free_mask=free_mask,
            edge_mask=graph.edata['edge_mask'],
        )
        
        eps_hat = torch.cat([x_out, h_out], dim=1)
        return eps_hat # [N, F]

    def forward(self, graph, training=None):
        """Unified forward method that handles both raw data and preprocessed data"""
        x = graph.ndata['positions']
        h = graph.ndata['one_hot']

        # Calculate free_mask as complement of anchor_mask
        free_mask = 1.0 - graph.ndata['anchor_mask']

        # Remove center of mass from protein atoms (anchor_mask)
        # x: [N, 3], anchor_mask: [N], node_mask: [N]
        com_mask = graph.ndata['anchor_mask']  # anchor atoms
        N = com_mask.sum()  # number of anchor atoms
        if N < 1e-5:
            mean_pos = torch.zeros(3, device=x.device)
        else:
            # Calculate center of mass: sum(x * mask) / sum(mask)
            # x: [N, 3], com_mask: [N] -> x_masked: [N, 3]
            x_masked = x * com_mask[:, None]  # [N, 3] * [N, 1] -> [N, 3]
            mean_pos = x_masked.sum(dim=0) / N  # [3]
        
        # Subtract center of mass from all atoms
        # x: [N, 3], mean_pos: [3], node_mask: [N] -> [N, 3]
        x = x - mean_pos[None, :]
        x = x * self.coord_scale_factor

        if training and self.data_augmentation:
            x = utils.random_rotation(x)

        # Concatenation - treat as single graph [N, F]
        xh = torch.cat([x, h], dim=1)

        # Sample t with weighted sampling (favoring larger t values)
        # For single graph, sample one time step
        t_int = self.sample_weighted_t(x.device)
        t = t_int / self.T
        gamma_t = expand_to_nodes(self.gamma(t), x)
        alpha_t = alpha(gamma_t, x)
        sigma_t = sigma(gamma_t, x)

        # Sample noise - treat as single graph
        eps_t = torch.randn_like(xh) * free_mask[:, None]

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps_t

        eps_t_hat = self._egnn_forward(z_t, t, graph)
        error_t = ((eps_t - eps_t_hat) ** 2).sum()

        # Computing L2-loss for t>0
        normalization = (self.n_dims + self.in_node_nf) * graph.ndata['node_mask'].sum()
        l2_loss = error_t / normalization
        l2_loss = l2_loss.mean()

        return {'loss': l2_loss}

    @torch.no_grad()
    def sample_chain(self, graph, keep_frames=None):
        """Unified sample_chain method that handles raw data"""
        # Calculate free_mask as complement of anchor_mask
        anchor_mask = graph.ndata['anchor_mask']
        free_mask = 1.0 - graph.ndata['anchor_mask']
        
        # Raw data format - apply preprocessing
        x = graph.ndata['positions']
        h = graph.ndata['one_hot']

        x, mean_pos = utils.remove_partial_mean_with_mask(x, graph.ndata['node_mask'], anchor_mask)

        x = x * self.coord_scale_factor

        # Concatenation - treat as single graph [N, F]
        xh = torch.cat([x, h], dim=1)

        # Initial linker sampling from N(0, I) - treat as single graph
        z = torch.randn_like(xh) * free_mask[:, None]
        z = xh * anchor_mask[:, None] + z * free_mask[:, None]

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Sample p(z_s | z_t) - treat as single graph
        for s_step in reversed(range(0, self.T)):
            # For single graph, keep time steps as scalars [1] for consistency
            s_val = torch.tensor([s_step], dtype=torch.float32, device=z.device)  # [1]
            t_val = s_val + 1
            
            s = s_val / self.T  # [1] - normalized time step
            t = t_val / self.T  # [1] - normalized time step

            gamma_t = expand_to_nodes(self.gamma(t), x)
            alpha_t = alpha(gamma_t, x)
            # z = alpha_t * (xh * cond['anchor_mask'] + z * cond['free_mask'])
            z = alpha_t * xh * anchor_mask[:, None] + z * free_mask[:, None]

            # Sample z_s given z_t
            z = self.sample_p_zs_given_zt(s, t, z, graph)
            # if sum of anchor_mask is 0, then recenter the chain
            if anchor_mask.sum() < 1e-5:
                z = z - z.mean(dim=0, keepdim=True)

            write_index = (s_step * keep_frames) // self.T
            chain[write_index] = z

        # Finally sample p(x, h | z_0)
        gamma_0 = expand_to_nodes(self.gamma(torch.zeros(1, device=z.device)), x)
        alpha_0 = alpha(gamma_0, x)
        z = alpha_0 * xh * anchor_mask + z * free_mask

        x, h = self.sample_p_xh_given_z0(z, graph)
        chain[0] = torch.cat([x, h], dim=1)  # Use dim=1 for flattened data [N, F]

        # For flattened data, chain shape is [T_keep, N, F], so use 3 indices
        chain[:, :, :3] = chain[:, :, :3] / self.coord_scale_factor
        chain[:, :, :3] = chain[:, :, :3] + mean_pos

        return chain

    def sample_p_zs_given_zt(self, s, t, z_t, graph):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only linker features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s = sigma(gamma_s, target_tensor=z_t)
        sigma_t = sigma(gamma_t, target_tensor=z_t)

        # Neural net prediction
        eps_hat = self._egnn_forward(z_t, t, graph)
        free_mask = 1.0 - graph.ndata['anchor_mask']
        eps_hat = eps_hat * free_mask[:, None]  # [N, F] * [N, 1] for proper broadcasting

        # Compute mu for p(z_s | z_t)
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat

        # Compute sigma for p(z_s | z_t)
        sigma_sampling = sigma_t_given_s * sigma_s / sigma_t

        # Sample z_s given the parameters derived from zt
        eps = torch.randn_like(mu) * free_mask[:, None]
        z_s = mu + sigma_sampling * eps
        # z_s = z_t * anchor_mask + z_s * free_mask 

        return z_s

    def sample_p_xh_given_z0(self, z_0, graph):
        """Samples x ~ p(x|z0). Samples only linker features and coords"""
        zeros = torch.zeros(1, device=z_0.device)  # [1] for single graph
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = torch.exp(0.5 * gamma_0)  # Will be expanded inline if needed
        
        # Neural net prediction
        eps_hat = self._egnn_forward(z_0, zeros, graph)
        free_mask = 1.0 - graph.ndata['anchor_mask']
        eps_hat = eps_hat * free_mask[:, None]

        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        eps = torch.randn_like(mu_x) * free_mask[:, None]
        xh = mu_x + sigma_x[:, None] * eps
        # xh = z_0 * anchor_mask + xh * free_mask

        # For flattened data [N, F], use dim=1 for slicing
        x, h = xh[:, :self.n_dims], xh[:, self.n_dims:]
        h = F.one_hot(torch.argmax(h, dim=1), self.in_node_nf) * graph.ndata['node_mask'][:, None]

        return x, h

    def compute_x_pred(self, eps_t, z_t, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = sigma(gamma_t, target_tensor=eps_t)
        alpha_t = alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred
    








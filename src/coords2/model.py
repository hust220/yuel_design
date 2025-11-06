# Third-party imports
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from tqdm import tqdm

# Local imports
from src.noise import GammaNetwork, PredefinedNoiseSchedule
from src.e3attention import E3AttentionBlock

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

class CoordsModel(torch.nn.Module):
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
        
        # 1. 节点特征嵌入层 (h: in_dim -> hidden_nf)
        self.embedding_node = nn.Sequential(
            Linear(in_node_features + 1, hidden_nf),
            nn.SiLU(),
            Linear(hidden_nf, hidden_nf)
        )
        
        # 2. 边特征嵌入层 (e: in_dim -> hidden_nf)
        # 注意：e 的输出维度必须与 GNN 层期望的 e 的维度 (hidden_nf) 一致。
        self.embedding_edge = nn.Sequential(
            Linear(in_edge_features, hidden_nf),
            nn.SiLU(),
            Linear(hidden_nf, hidden_nf)
        )
        
        # 3. GNN 块列表
        self.blocks = nn.ModuleList([
            # 使用我们之前定义的注意力层
            E3AttentionBlock(hidden_nf=hidden_nf)
            for _ in range(self.n_blocks)
        ])        

    def sample_weighted_t(self, device):
        """使用加权采样来采样时间步 t,让更大的 t 值有更高的概率 - 单图版本"""
        # 动态计算权重和概率
        t_weights = torch.arange(1, self.T + 1, dtype=torch.float32, device=device) ** self.t_weight_power
        t_probs = t_weights / t_weights.sum()
        
        # 使用 torch.multinomial 进行加权采样 - 只采样一个时间步
        t_index = torch.multinomial(t_probs, 1, replacement=True)
        t_int = (t_index + 1).float()  # +1 因为索引从0开始，但t从1开始
        return t_int

    def _egnn_forward(self, xt, t, graph):
        """Helper function to run EGNN forward pass"""
        # Add time step to node_attr for flattened data
        # t shape: [1], z shape: [N, F], node_mask shape: [N]

        h = graph.ndata['h'] # [n_nodes, n_node_features]
        edge_index = graph.edge_index # [2, n_edges]
        edge_attr = graph.edata['edge_attr'] # [n_edges, n_edge_features]

        t_expanded = t.expand(h.shape[0], 1).to(h.device)  # [n_nodes, 1]
        h = torch.cat([h, t_expanded], dim=1) # [n_nodes, n_node_features + 1]

        edge_dist_one_hot = torch.nn.functional.one_hot(graph.edata['edge_dist'], num_classes=13).float()
        # ligand_bonds_one_hot = torch.nn.functional.one_hot(graph.edata['ligand_bonds'], num_classes=5).float()
        # edge_attr = torch.cat([edge_dist_one_hot, ligand_bonds_one_hot, edge_attr], dim=1) # [n_edges, n_edge_features + 12]
        edge_attr = torch.cat([edge_dist_one_hot, edge_attr], dim=1) # [n_edges, n_edge_features + 12]
        
        h = self.embedding_node(h)
        e = self.embedding_edge(edge_attr)
        x = xt
        for block in self.blocks:
            h, x = block(h, x, e, edge_index)

        return x - xt

    def forward(self, graph, training=None):
        """Unified forward method that handles both raw data and preprocessed data"""
        x0 = graph.ndata['x']

        # Sample t with weighted sampling (favoring larger t values)
        # For single graph, sample one time step
        t_int = self.sample_weighted_t(x0.device)
        t = t_int / self.T
        gamma_t = expand_to_nodes(self.gamma(t), x0)
        alpha_t = alpha(gamma_t, x0)
        sigma_t = sigma(gamma_t, x0)

        # Sample noise - treat as single graph
        eps_t = torch.randn_like(x0)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        xt = alpha_t * x0 + sigma_t * eps_t

        eps_t_hat = self._egnn_forward(xt, t, graph)
        loss = ((eps_t - eps_t_hat) ** 2).sum(dim=1).mean()
        rmsd = torch.sqrt(loss)

        return {'loss': loss, 'rmsd': rmsd}

    @torch.no_grad()
    def sample_chain(self, graph, keep_frames=None):
        """Unified sample_chain method that handles raw data"""
        h = graph.ndata['h']
        x = torch.randn(h.shape[0], 3, device=h.device)

        chain = []

        # Sample p(z_s | z_t) - treat as single graph
        for s_step in tqdm(reversed(range(0, self.T)), desc="Diffusion sampling", total=self.T):
            # For single graph, keep time steps as scalars [1] for consistency
            s_val = torch.tensor([s_step], dtype=torch.float32, device=x.device)  # [1]
            t_val = s_val + 1
            
            s = s_val / self.T  # [1] - normalized time step
            t = t_val / self.T  # [1] - normalized time step

            x = self.sample_p_zs_given_zt(s, t, x, graph)
            x = x - x.mean(dim=0, keepdim=True)

            chain.append(x)

        x = self.sample_p_xh_given_z0(x, graph)
        chain.append(x)

        return chain[-1], torch.stack(chain, dim=0)

    def sample_p_zs_given_zt(self, s, t, xt, graph):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only linker features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = sigma_and_alpha_t_given_s(gamma_t, gamma_s, xt)
        sigma_s = sigma(gamma_s, target_tensor=xt)
        sigma_t = sigma(gamma_t, target_tensor=xt)

        # Neural net prediction
        eps_hat = self._egnn_forward(xt, t, graph)

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

    def sample_p_xh_given_z0(self, xt, graph):
        """Samples x ~ p(x|z0). Samples only linker features and coords"""
        zeros = torch.zeros(1, device=xt.device)  # [1] for single graph
        gamma_0 = self.gamma(zeros) # [1]

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = torch.exp(0.5 * gamma_0)  # [1]
        
        # Neural net prediction
        eps_hat = self._egnn_forward(xt, zeros, graph)

        mu = self.compute_x_pred(eps_t=eps_hat, xt=xt, gamma_t=gamma_0)
        eps = torch.randn_like(mu)
        xt = mu + sigma_x * eps

        return xt

    def compute_x_pred(self, eps_t, xt, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = sigma(gamma_t, target_tensor=eps_t)
        alpha_t = alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (xt - sigma_t * eps_t)
        return x_pred
    








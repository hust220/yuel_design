import torch
import torch.nn as nn
from torch.nn import Linear, SiLU, Sequential as Seq

def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[col] - x[row]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_diff = coord_diff / (radial + 1e-8) ** 0.5
    return radial, coord_diff

def normalize_l2(x, dim=-1, eps=1e-8):
    """L2 normalize a tensor along the specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension along which to normalize
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized tensor with same shape as input
    """
    # Manual L2 norm calculation: sqrt(sum(x^2))
    norm = torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=True) + eps)
    return x / norm

def segment_sum(data, segment_ids, num_segments, aggregation: str = 'sum'):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
        Added 'max' aggregation.
    """
    if aggregation == 'sum':
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids_expanded, data)
        return result
    elif aggregation == 'mean':
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids_expanded, data)

        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids_expanded, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
        return result
    elif aggregation == 'max':
        # For 'max' aggregation, initialize with negative infinity for robust max finding
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, float('-inf'))  # Initialize with -inf
        segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        
        # Use scatter_reduce with 'amax' mode for max aggregation
        # Note: torch.scatter_reduce was introduced in PyTorch 2.0
        # For older versions, you might need to use torch_scatter library or a manual loop/scatter_max.
        result = torch.scatter_reduce(
            result,             # The output tensor
            0,                  # Dimension along which to scatter (segments)
            segment_ids_expanded, # Indices to scatter along the given dimension
            data,               # Values to scatter
            reduce='amax',      # Reduction operation (maximum)
            include_self=False  # Do not include initial value in reduction, unless it's valid data
        )
        # A common practice is to filter out -inf values if a segment was empty.
        # This isn't strictly necessary if `segment_ids` are guaranteed to cover all `num_segments`.
        return result
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation}")
    
    return result


class GNNBlock(nn.Module):
    def __init__(self, hidden_nf):
        """
        Args:
            hidden_nf (int): 隐藏特征维度 (d)。假设节点特征和边特征维度相等。
        """
        super().__init__()
        self.hidden_nf = hidden_nf

        # Node embeddings for attention (single layer like standard Transformer)
        self.node_query_embedding = Linear(hidden_nf, hidden_nf)
        self.node_key_embedding = Linear(hidden_nf, hidden_nf)
        self.node_value_embedding = Linear(hidden_nf, hidden_nf)
        
        # Edge embeddings for attention (single layer like standard Transformer)
        self.edge_query_embedding = Linear(hidden_nf, hidden_nf)
        self.edge_key_embedding = Linear(hidden_nf, hidden_nf)
        self.edge_value_embedding = Linear(hidden_nf, hidden_nf)

        self.norm_h = nn.LayerNorm(hidden_nf)
        self.norm_e = nn.LayerNorm(hidden_nf)
        
        # Feed-Forward Network (FFN) for nodes and edges
        # Standard FFN: Linear -> ReLU/GELU -> Linear
        # Typically expands to 4x hidden dimension, then back to hidden_nf
        ffn_expansion = 4
        self.ffn_h = nn.Sequential(
            Linear(hidden_nf, hidden_nf * ffn_expansion),
            nn.ReLU(),  # or nn.GELU()
            Linear(hidden_nf * ffn_expansion, hidden_nf)
        )
        self.ffn_e = nn.Sequential(
            Linear(hidden_nf, hidden_nf * ffn_expansion),
            nn.ReLU(),  # or nn.GELU()
            Linear(hidden_nf * ffn_expansion, hidden_nf)
        )
        
    def forward(self, h, e, edge_index):
        row, col = edge_index

        # layer norm
        h_norm = self.norm_h(h) 
        e_norm = self.norm_e(e) 

        # update nodes
        h_q = normalize_l2(self.node_query_embedding(h_norm), dim=1) 
        e_k = normalize_l2(self.edge_key_embedding(e_norm), dim=1) 
        e_v = self.edge_value_embedding(e_norm) 
        e_att = torch.einsum('ij,ij->i', h_q[row], e_k).unsqueeze(-1) 
        e_v = e_att * e_v
        h_update = segment_sum(e_v, row, h.size(0), 'sum') # 消息传递结果

        # update edges
        e_q = normalize_l2(self.edge_query_embedding(e_norm), dim=1) # (n_edges, d)
        h_k = normalize_l2(self.node_key_embedding(h_norm), dim=1) # (n_nodes, d)
        h_v = self.node_value_embedding(h_norm) # (n_nodes, d)
        h_att_row = torch.einsum('ij,ij->i', e_q, h_k[row]).unsqueeze(-1) # (n_edges, 1)
        h_att_col = torch.einsum('ij,ij->i', e_q, h_k[col]).unsqueeze(-1) # (n_edges, 1)
        h_v_row = h_att_row * h_v[row] # (n_edges, d)
        h_v_col = h_att_col * h_v[col] # (n_edges, d)
        e_update = h_v_row + h_v_col # (n_edges, d)

        h_attn_out = h + h_update
        e_attn_out = e + e_update

        h_ffn_in = self.norm_h(h_attn_out)
        e_ffn_in = self.norm_e(e_attn_out)

        h_ffn_out = self.ffn_h(h_ffn_in)
        e_ffn_out = self.ffn_e(e_ffn_in)

        # residual connection
        h_final = h_attn_out + h_ffn_out 
        e_final = e_attn_out + e_ffn_out

        return h_final, e_final

class E3GNN(nn.Module):
    """
    E3GNN 模块，包含多个 E3AttentionBlock 和统一的坐标更新。
    """
    def __init__(self, kwargs):
        super().__init__()
        self.hidden_nf = kwargs['hidden_nf'] # 隐藏特征维度
        self.n_blocks = kwargs['n_blocks'] # E3AttentionBlock 的数量
        self.in_node_features = kwargs['in_node_features'] # 输入节点特征维度
        self.in_edge_features = kwargs['in_edge_features'] # 输入边特征维度
        
        # 节点特征嵌入
        self.node_embedding = nn.Sequential(
            Linear(self.in_node_features, self.hidden_nf),
            nn.SiLU(),
            Linear(self.hidden_nf, self.hidden_nf)
        )
        # 边特征嵌入
        self.edge_embedding = nn.Sequential(
            Linear(self.in_edge_features + 1, self.hidden_nf),
            nn.SiLU(),
            Linear(self.hidden_nf, self.hidden_nf)
        )
        
        # 创建 n_blocks 个 E3AttentionBlock
        self.blocks = nn.ModuleList([
            GNNBlock(hidden_nf=self.hidden_nf)
            for _ in range(self.n_blocks)
        ])
        
        # 坐标更新 MLP (Equivariant part)
        # 输入维度: 加权消息 e_v (d)
        self.coord_mlp = Seq(
            Linear(self.hidden_nf, self.hidden_nf),
            SiLU(),
            Linear(self.hidden_nf, 1) # 输出坐标更新
        )
            
    def forward(self, h, x, e, edge_index):
        """
        Args:
            h: (n_nodes, d) 节点特征
            x: (n_nodes, 3) 节点坐标
            e: (n_edges, d) 边特征
            edge_index: (2, n_edges) 边索引
        
        Returns:
            h: (n_nodes, d) 更新后的节点特征
            x: (n_nodes, 3) 更新后的节点坐标
        """
        row, col = edge_index
        
        # 计算距离并与边特征拼接
        distances, coord_diff = coord2diff(x, edge_index) # (n_edges, 1), (n_edges, 3)
        e = torch.cat([distances, e], dim=1) # (n_edges, d+1)

        h = self.node_embedding(h) # (n_nodes, d)
        e = self.edge_embedding(e) # (n_edges, d)
        
        # 通过所有 E3AttentionBlock
        for block in self.blocks:
            h, e = block(h, e, edge_index)
        
        # 在所有 blocks 之后统一更新坐标
        dx_scalar = self.coord_mlp(e) # (n_edges, 1)
        dx = coord_diff * dx_scalar # (n_edges, 3)
        dx = segment_sum(dx, row, x.size(0), 'sum') # (n_nodes, 3)
        
        return h, dx

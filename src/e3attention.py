import torch
import torch.nn as nn
from torch.nn import Linear, SiLU, Sequential as Seq

def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_diff = coord_diff / (radial + 1e-8) ** 0.5
    return radial, coord_diff

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
    elif aggregation == 'mean':
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids_expanded, data)

        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids_expanded, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
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
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation}")
    
    return result


class E3AttentionBlock(nn.Module):
    """
    集成了关系注意力、坐标更新和消息传递的 GNN 层。
    """
    def __init__(self, hidden_nf):
        """
        Args:
            hidden_nf (int): 隐藏特征维度 (d)。假设节点特征和边特征维度相等。
        """
        super().__init__()
        self.hidden_nf = hidden_nf

        # 1. QKV 投影层 (用于节点特征)
        self.node_query_embedding = Linear(hidden_nf, hidden_nf)
        self.node_key_embedding = Linear(hidden_nf, hidden_nf)
        self.node_value_embedding = Linear(hidden_nf, hidden_nf)
        
        # 2. Attention Logits 偏置 MLP
        # 输入维度: distances(1) + e(d) + h_q(d) + h_k(d) = 1 + 3*hidden_nf
        self.edge_attention = Seq(
            Linear(1 + 3 * hidden_nf, hidden_nf),
            SiLU(),
            Linear(hidden_nf, 1) # 输出标量 Logit 偏置
        )

        # 3. 边值（消息）生成 MLP
        # 输入维度: e(d) + h_v(d) = 2*hidden_nf
        self.edge_value_embedding = Seq(
            Linear(2 * hidden_nf, hidden_nf),
            SiLU(),
            Linear(hidden_nf, hidden_nf) # 输出边消息向量 e_v (d 维)
        )

        # 4. 坐标更新 MLP (Equivariant part)
        # 输入维度: 加权消息 e_v (d)
        self.coord_mlp = Seq(
            Linear(hidden_nf, hidden_nf),
            SiLU(),
            Linear(hidden_nf, 1) # 输出标量权重，用于缩放 coord_diff
        )

        self.norm_h = nn.LayerNorm(hidden_nf) # 添加 LayerNorm
        
        # 自动初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化。
        - 使用 Xavier/Glorot Uniform 初始化 QKV 投影层。
        - 使用 Kaiming Uniform (He) 初始化激活函数为 SiLU 的 MLP 层。
        - 初始化 LayerNorm 的权重和偏置。
        """
        
        def _reset(m):
            if isinstance(m, Linear):
                # 默认使用 Xavier Uniform
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            # LayerNorm 初始化
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        self.apply(_reset)

        # 针对使用 SiLU 激活函数的 MLP，应用 Kaiming Uniform 初始化
        # (SiLU 类似于 ReLU，He/Kaiming 更合适)
        
        # edge_attention 是 Seq(LayerNorm, Linear, SiLU, Linear) 
        # 只对第一个 Linear 层使用 Kaiming，因为它是 SiLU 的输入
        nn.init.kaiming_uniform_(self.edge_attention[0].weight, nonlinearity='relu')
        # edge_value_embedding 是 Seq(LayerNorm, Linear, SiLU, Linear)
        nn.init.kaiming_uniform_(self.edge_value_embedding[0].weight, nonlinearity='relu')
        # coord_mlp 是 Seq(LayerNorm, Linear, SiLU, Linear)
        nn.init.kaiming_uniform_(self.coord_mlp[0].weight, nonlinearity='relu')
        
        # 注意：MLP 中第二个 Linear 层（输出层）通常保留默认的 Xavier 或标准初始化。
        # 如果输出层没有激活函数，使用 Xavier 是安全的选择。

    def forward(self, h, x, e, edge_index):
        # h: (n_nodes, d)
        # x: (n_nodes, d)
        # e: (n_edges, d)
        # edge_index: (2, n_edges)
        row, col = edge_index
        distances, coord_diff = coord2diff(x, edge_index) # (n_edges, 1), (n_edges, d)

        h = self.norm_h(h) # (n_nodes, d)

        # 节点特征投影
        h_q = self.node_query_embedding(h) # (n_nodes, d)
        h_k = self.node_key_embedding(h) # (n_nodes, d)
        h_v = self.node_value_embedding(h) # (n_nodes, d)

        h_q = h_q[row] # (n_edges, d)
        h_k = h_k[col] # (n_edges, d)
        h_v = h_v[row] # (n_edges, d)

        # --- 1. Attention Logits 计算 ---
        # 拼接所有用于计算注意力偏置的特征
        e_att_features = torch.cat([distances, e, h_q, h_k], dim=1) # (n_edges, 1 + d*3)
        e_att = self.edge_attention(e_att_features) # (n_edges, 1)

        # 点积相似性 (高效相似性度量)
        att_logits = torch.einsum('ij,ij->i', h_q, h_k).unsqueeze(-1) # (n_edges, 1)
        att_logits = att_logits / (self.hidden_nf ** 0.5) # 缩放
        
        # 加上 MLP 偏置
        att_logits = att_logits + e_att # (n_edges, 1)

        # --- 2. Softmax 归一化 ---
        max_logits = segment_sum(att_logits, row, h.size(0), 'max')  # LogSumExp max
        exp_logits = torch.exp(att_logits - max_logits[row]) # (n_edges, 1)
        sum_exp_logits = segment_sum(exp_logits, row, h.size(0), 'sum') # (n_edges, 1)
        att_weights = exp_logits / (sum_exp_logits[row] + 1e-8)  # (n_edges, 1)

        # --- 3. 消息生成与加权 ---
        # 边值 (消息) 生成
        e_v_features = torch.cat([e, h_v], dim=1) # (n_edges, d*2)
        e_v = self.edge_value_embedding(e_v_features) # (n_edges, d)

        # 应用注意力权重
        e_v = e_v * att_weights # (n_edges, d)

        # --- 4. 状态更新 ---

        # 更新节点特征 h (残差连接)
        h = h + segment_sum(e_v, row, h.size(0), 'sum') # (n_nodes, d)

        # 更新坐标 x (Equivariant 更新)
        dx_scalar = self.coord_mlp(e_v) # (n_edges, 1)
        dx = coord_diff * dx_scalar # (n_edges, d)
        x = x + segment_sum(dx, row, x.size(0), 'sum') # (n_nodes, d)

        # 返回核心状态 (h, x)，边特征 e 仅作为输入，不作为状态返回
        return h, x

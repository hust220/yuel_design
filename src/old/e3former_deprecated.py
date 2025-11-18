import torch
import torch.nn as nn
from torch.nn import Linear, SiLU, Sequential as Seq

def coord2diff(x):
    """
    Compute distance matrix and normalized coordinate differences for all node pairs.
    
    Args:
        x: Node coordinates with shape (n, 3)
    
    Returns:
        tuple: (distance_matrix, coord_diff)
            - distance_matrix: Distance matrix with shape (n, n)
            - coord_diff: Normalized coordinate differences with shape (n, n, 3)
    """
    n = x.shape[0]
    # Expand x to compute all pairwise differences
    # x.unsqueeze(1): (n, 1, 3), x.unsqueeze(0): (1, n, 3)
    # x.unsqueeze(1) - x.unsqueeze(0): (n, n, 3)
    coord_diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n, n, 3)
    
    # Compute squared distances
    squared_distances = torch.sum(coord_diff ** 2, dim=2)  # (n, n)
    
    # Compute distance matrix (square root of squared distances)
    distance_matrix = torch.sqrt(squared_distances + 1e-8)  # (n, n)
    
    # Normalize coordinate differences
    coord_diff = coord_diff / (distance_matrix.unsqueeze(-1) + 1e-8)  # (n, n, 3)
    
    return distance_matrix, coord_diff


class E3AttentionBlock(nn.Module):
    """
    E3AttentionBlock for fully connected graphs using matrix operations.
    """
    def __init__(self, hidden_nf):
        """
        Args:
            hidden_nf (int): Hidden feature dimension (d). Node and pair features have the same dimension.
        """
        super().__init__()
        self.hidden_nf = hidden_nf

        # 1. QKV projection layers (for node features)
        self.node_query_embedding = Linear(hidden_nf, hidden_nf)
        self.node_key_embedding = Linear(hidden_nf, hidden_nf)
        self.node_value_embedding = Linear(hidden_nf, hidden_nf)
        
        # 2. Attention Logits bias MLP
        # Input dimension: z(d) + h_q(d) + h_k(d) = 3*hidden_nf
        self.edge_attention = Seq(
            Linear(3 * hidden_nf, hidden_nf),
            SiLU(),
            Linear(hidden_nf, 1)  # Output scalar logit bias
        )

        # 3. Edge value (message) generation MLP
        # Input dimension: z(d) + h_v_concat(2*d) = 3*hidden_nf
        self.edge_value_embedding = Seq(
            Linear(3 * hidden_nf, hidden_nf),
            SiLU(),
            Linear(hidden_nf, hidden_nf)  # Output edge message vector z_v (d-dim)
        )

        self.norm_h = nn.LayerNorm(hidden_nf)
        
    def forward(self, h, z):
        """
        Args:
            h: (n_nodes, d) Node features
            z: (n_nodes, n_nodes, d) Pair features for all node pairs
        
        Returns:
            h: (n_nodes, d) Updated node features
            z: (n_nodes, n_nodes, d) Updated pair features (messages for coordinate update)
        """
        n_nodes = h.shape[0]
        
        h = self.norm_h(h)  # (n_nodes, d)

        # Node feature projections
        h_q = self.node_query_embedding(h)  # (n_nodes, d)
        h_k = self.node_key_embedding(h)  # (n_nodes, d)
        h_v = self.node_value_embedding(h)  # (n_nodes, d)

        # Expand to all node pairs
        # h_q[i] corresponds to query from node i to all nodes
        # h_k[j] corresponds to key from node j to all nodes
        # h_q.unsqueeze(1): (n_nodes, 1, d) -> (n_nodes, n_nodes, d) when broadcasting
        # h_k.unsqueeze(0): (1, n_nodes, d) -> (n_nodes, n_nodes, d) when broadcasting
        h_q_expanded = h_q.unsqueeze(1).expand(-1, n_nodes, -1)  # (n_nodes, n_nodes, d)
        h_k_expanded = h_k.unsqueeze(0).expand(n_nodes, -1, -1)  # (n_nodes, n_nodes, d)
        
        # Concatenate h_v for both nodes in each pair
        # h_v[i] and h_v[j] for pair (i, j)
        h_v_i = h_v.unsqueeze(1).expand(-1, n_nodes, -1)  # (n_nodes, n_nodes, d)
        h_v_j = h_v.unsqueeze(0).expand(n_nodes, -1, -1)  # (n_nodes, n_nodes, d)
        h_v_concat = torch.cat([h_v_i, h_v_j], dim=2)  # (n_nodes, n_nodes, 2*d)

        # --- 1. Attention Logits computation ---
        z_att_features = torch.cat([z, h_q_expanded, h_k_expanded], dim=2)  # (n_nodes, n_nodes, 3*d)
        z_att = self.edge_attention(z_att_features)  # (n_nodes, n_nodes, 1)

        # Dot product similarity (efficient similarity measure)
        # Compute dot product between h_q[i] and h_k[j] for all pairs (i, j)
        att_logits = torch.einsum('ik,jk->ij', h_q, h_k)  # (n_nodes, n_nodes)
        att_logits = att_logits.unsqueeze(-1)  # (n_nodes, n_nodes, 1)
        att_logits = att_logits / (self.hidden_nf ** 0.5)  # Scale
        
        # Add MLP bias
        att_logits = att_logits + z_att  # (n_nodes, n_nodes, 1)

        # --- 2. Softmax normalization ---
        # Apply softmax along the second dimension (keys) for each query
        # att_logits: (n_nodes, n_nodes, 1) -> squeeze to (n_nodes, n_nodes)
        att_logits = att_logits.squeeze(-1)  # (n_nodes, n_nodes)
        max_logits = torch.max(att_logits, dim=1, keepdim=True)[0]  # (n_nodes, 1)
        exp_logits = torch.exp(att_logits - max_logits)  # (n_nodes, n_nodes)
        sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)  # (n_nodes, 1)
        att_weights = exp_logits / (sum_exp_logits + 1e-8)  # (n_nodes, n_nodes)
        att_weights = att_weights.unsqueeze(-1)  # (n_nodes, n_nodes, 1)

        # --- 3. Message generation and weighting ---
        # Edge value (message) generation
        z_v_features = torch.cat([z, h_v_concat], dim=2)  # (n_nodes, n_nodes, d + 2*d) = (n_nodes, n_nodes, 3*d)
        z_v = self.edge_value_embedding(z_v_features)  # (n_nodes, n_nodes, d)

        # Apply attention weights
        z_v = z_v * att_weights  # (n_nodes, n_nodes, d)

        # --- 4. State update ---
        # Update node features h (residual connection)
        # Aggregate messages: sum over all incoming messages for each node
        # z_v[i, j] is message from node j to node i
        # Sum over j (dim=1) to get aggregated messages for each node i
        h_update = torch.sum(z_v, dim=1)  # (n_nodes, d)
        h = h + h_update  # (n_nodes, d)

        # Return updated node features h and pair messages z_v (for coordinate update)
        return h, z_v


class E3former(nn.Module):
    """
    E3former module with multiple E3AttentionBlock and unified coordinate updates.
    Uses fully connected graph structure with matrix operations.
    """
    def __init__(self, kwargs):
        super().__init__()
        self.hidden_nf = kwargs['hidden_nf']  # Hidden feature dimension
        self.n_blocks = kwargs['n_blocks']  # Number of E3AttentionBlock
        self.in_node_features = kwargs['in_node_features']  # Input node feature dimension
        self.in_pair_features = kwargs['in_edge_features']  # Input pair feature dimension (renamed from in_edge_features)
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            Linear(self.in_node_features, self.hidden_nf),
            nn.SiLU(),
            Linear(self.hidden_nf, self.hidden_nf)
        )
        # Pair feature embedding
        self.pair_embedding = nn.Sequential(
            Linear(self.in_pair_features + 1, self.hidden_nf),  # +1 for distance
            nn.SiLU(),
            Linear(self.hidden_nf, self.hidden_nf)
        )
        
        # Create n_blocks E3AttentionBlock
        self.blocks = nn.ModuleList([
            E3AttentionBlock(hidden_nf=self.hidden_nf)
            for _ in range(self.n_blocks)
        ])
        
        # Coordinate update MLP (Equivariant part)
        # Input dimension: weighted message z_v (d)
        self.coord_mlp = Seq(
            Linear(self.hidden_nf, self.hidden_nf),
            SiLU(),
            Linear(self.hidden_nf, 1)  # Output scalar weight for scaling coord_diff
        )
            
    def forward(self, h, x, z):
        """
        Args:
            h: (n_nodes, d) Node features
            x: (n_nodes, 3) Node coordinates
            z: (n_nodes, n_nodes, d_pair) Pair features for all node pairs
        
        Returns:
            h: (n_nodes, d) Updated node features
            x: (n_nodes, 3) Updated node coordinates
        """
        n_nodes = h.shape[0]
        
        # Compute distances and coordinate differences for all node pairs
        distances, coord_diff = coord2diff(x)  # (n_nodes, n_nodes), (n_nodes, n_nodes, 3)
        
        # Concatenate distance with pair features
        z = torch.cat([z, distances.unsqueeze(-1)], dim=2)  # (n_nodes, n_nodes, d_pair + 1)

        h = self.node_embedding(h)  # (n_nodes, d)
        z = self.pair_embedding(z)  # (n_nodes, n_nodes, d)
        
        # Pass through all E3AttentionBlock
        for block in self.blocks:
            h, z = block(h, z)
        
        # Unified coordinate update after all blocks
        # z now contains the weighted messages (n_nodes, n_nodes, d)
        dx_scalar = self.coord_mlp(z)  # (n_nodes, n_nodes, 1)
        dx = coord_diff * dx_scalar  # (n_nodes, n_nodes, 3)
        # Aggregate coordinate updates: sum over all incoming messages for each node
        # dx[i, j] is coordinate update from node j to node i
        # Sum over j (dim=1) to get total coordinate update for each node i
        x = x + torch.sum(dx, dim=1)  # (n_nodes, 3)
        
        return h, x


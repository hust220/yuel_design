import torch
import torch.nn as nn
from torch.nn import Linear as TorchLinear, SiLU, Sequential as Seq
from src.evoformer import EvoformerStack, Linear

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

class E3former(nn.Module):
    """
    E3former module that integrates Evoformer with coordinate updates.
    """
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Configuration parameters including:
                - seq_input_dim: Input dimension for sequence features
                - z_input_dim: Input dimension for pair features
                - c_m: Sequence channel dimension
                - c_z: Pair channel dimension
                - c_hidden_seq_att: Hidden dimension in sequence attention
                - c_hidden_opm: Hidden dimension in outer product module
                - c_hidden_mul: Hidden dimension in multiplicative updates
                - c_hidden_pair_att: Hidden dimension in triangular attention
                - no_heads_seq: Number of heads for sequence attention
                - no_heads_pair: Number of heads for pair attention
                - no_blocks: Number of Evoformer blocks
                - transition_n: Factor for transition layers
                - blocks_per_ckpt: Blocks per checkpoint
                - inf: Infinity value for attention masks
                - eps: Epsilon for numerical stability
        """
        super().__init__()
        
        # Create Evoformer stack
        self.evoformer = EvoformerStack(
            c_m=kwargs.get('c_m', 64),
            c_z=kwargs.get('c_z', 64),
            c_hidden_seq_att=kwargs.get('c_hidden_seq_att', 32),
            c_hidden_opm=kwargs.get('c_hidden_opm', 32),
            c_hidden_mul=kwargs.get('c_hidden_mul', 32),
            c_hidden_pair_att=kwargs.get('c_hidden_pair_att', 32),
            c_s=kwargs.get('c_s', 1),  # Not used in this module
            no_heads_seq=kwargs.get('no_heads_seq', 8),
            no_heads_pair=kwargs.get('no_heads_pair', 4),
            no_blocks=kwargs.get('no_blocks', 4),
            transition_n=kwargs.get('transition_n', 4),
            blocks_per_ckpt=kwargs.get('blocks_per_ckpt', 4),
            inf=kwargs.get('inf', 1e9),
            eps=kwargs.get('eps', 1e-10),
        )
        
        # Embed sequence features
        seq_input_dim = kwargs.get('seq_input_dim', 32)
        c_m = kwargs.get('c_m', 64)
        self.embed_seq = TorchLinear(seq_input_dim, c_m)
        
        # Embed pair features
        z_input_dim = kwargs.get('z_input_dim', 1)
        c_z = kwargs.get('c_z', 64)
        self.embed_z = TorchLinear(z_input_dim + 1, c_z)
        
        # Coordinate update MLP: takes pair representation and outputs scalar weights
        # Input: z_out (n, n, c_z), Output: (n, n, 1) for scaling coord_diff
        self.coord_mlp = Seq(
            Linear(c_z, c_z, init="relu"),
            SiLU(),
            Linear(c_z, c_z, init="relu"),
            SiLU(),
            Linear(c_z, 1, init="final")
        )

    def forward(self, seq, x, z, seq_mask, pair_mask, chunk_size):
        # seq: (b, n, seq_input_dim)
        # x: (b, n, 3)
        # z: (b, n, n, z_input_dim)
        # seq_mask: (b, n)
        # pair_mask: (b, n, n)
        # chunk_size: int

        # Compute distances for each batch separately
        b, n = seq.shape[0], seq.shape[1]
        distances_list = []
        coord_diff_list = []
        for i in range(b):
            dist_i, coord_diff_i = coord2diff(x[i])  # (n, n), (n, n, 3)
            distances_list.append(dist_i)
            coord_diff_list.append(coord_diff_i)
        distances = torch.stack(distances_list, dim=0)  # (b, n, n)
        coord_diff = torch.stack(coord_diff_list, dim=0)  # (b, n, n, 3)

        seq = self.embed_seq(seq)  # (b, n, c_m)
        z = torch.cat([z, distances.unsqueeze(-1)], dim=-1)  # (b, n, n, z_input_dim + 1)
        z = self.embed_z(z)  # (b, n, n, c_z)

        _, z_out, _ = self.evoformer(seq, z, seq_mask, pair_mask, chunk_size)  # (b, n, c_m), (b, n, n, c_z), (b, n, c_s)

        weighted_diff = self.coord_mlp(z_out) * coord_diff  # (b, n, n, 1) * (b, n, n, 3) = (b, n, n, 3)
        x = x + torch.sum(weighted_diff, dim=2)  # (b, n, 3)

        return x



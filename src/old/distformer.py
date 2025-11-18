import torch
import torch.nn as nn

from src.DistAttention import DistAttention

def x_diff(x):
    coord_diff = x.unsqueeze(2) - x.unsqueeze(1)  # [batch_size, n, n, 3]
    squared_distances = torch.sum(coord_diff ** 2, dim=-1)  # [batch_size, n, n]
    distance_matrix = torch.sqrt(squared_distances + 1e-8)  # [batch_size, n, n]
    coord_diff = coord_diff / (distance_matrix.unsqueeze(-1) + 1e-8)  # [batch_size, n, n, 3]
    
    return distance_matrix, coord_diff

class Distformer(nn.Module):
    def __init__(self, n_blocks, d_model, C_h, C_z, 
                 dim_feedforward=2048, eps=1e-8):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        
        # Embedding layers
        self.embed_h = nn.Linear(C_h, d_model)
        self.embed_z = nn.Linear(C_z + 1, d_model)
        
        # DistAttention layers
        self.blocks = nn.ModuleList([
            DistAttention(d_model, dim_feedforward, eps)
            for i in range(n_blocks)
        ])
        
        # Output layers
        self.output_z = nn.Linear(d_model, 1)
        
    def forward(self, h, x, z, mask):
        # Embed node features
        h = self.embed_h(h)
        # h = torch.einsum('bid,bjd->bijd', h, h)
        
        # Compute distance and direction
        d, dx = x_diff(x)  # (b, n, n), (b, n, n, 3)
        
        # Concatenate edge features with distance and embed
        z = torch.cat([z, d.unsqueeze(-1)], dim=-1)
        z = self.embed_z(z)

        # Apply DistAttention layers
        for i, block in enumerate(self.blocks):
            h, z = block(h, z)

        z = self.output_z(z) * mask.unsqueeze(-1)

        # Update coordinates
        x_update = z * dx  # (b, n, n, 1) * (b, n, n, 3) = (b, n, n, 3)
        x = x + torch.sum(x_update, dim=2)  # (b, n, 3)

        return x


import torch

def add_time_step(t, x, node_mask):
    # t: (B)
    # x: (B, N, D)
    b, n, d = x.shape

    # Reshaping node features & adding time feature
    # x = x.view(b * n, -1).clone()  # (B*N, D)
    # expand t to (B, N, 1)
    # if t is a integer, create a tensor of shape (B, N, 1)
    if isinstance(t, int):
        t = torch.full((b, n, 1), fill_value=t, device=x.device)
    else:
        # Handle different t shapes
        if t.dim() == 0:  # scalar tensor
            t = torch.full((b, n, 1), fill_value=t.item(), device=x.device)
        elif t.shape == torch.Size([b]):  # (B,) shape
            t = t.unsqueeze(1).unsqueeze(2).expand(b, n, 1)  # (B, 1, 1) -> (B, N, 1)
        elif t.shape == torch.Size([b, n]):  # (B, N) shape
            t = t.unsqueeze(2)  # (B, N, 1)
        elif t.shape == torch.Size([b, n, 1]):  # already correct shape
            pass
        elif t.shape == torch.Size([b, 1]):  # (B, 1) shape - common case
            t = t.unsqueeze(1).expand(b, n, 1)  # (B, 1, 1) -> (B, N, 1)
        else:
            # Try to reshape if possible
            print(f"Unknown t shape: {t.shape}")
            t = t.view(b, n, 1)
    return torch.cat([x, t], dim=2) * node_mask  # (B, N, D+1)

def flatten_batch(h, edge_index, edge_attr, node_mask, edge_mask, node_attr=None, linker_mask=None):
    b, n, d = h.shape
    _, n_edges, d_edges = edge_attr.shape

    h = h.view(b * n, d)  # (B*N, D)
    node_mask = node_mask.view(b * n, 1)  # (B*N, 1)
    edge_mask = edge_mask.view(b * n_edges, 1)  # (B*N_EDGES, 1)
    edge_attr = edge_attr.view(b * n_edges, d_edges)  # (B*N_EDGES, edge_feat_nf)

    # Handle optional node_attr
    if node_attr is not None:
        node_attr = node_attr.view(b * n, node_attr.size(-1))  # (B*N, node_attr_dim)
    
    # Handle optional linker_mask
    if linker_mask is not None:
        linker_mask = linker_mask.view(b * n, 1)  # (B*N, 1)

    edge_index = edge_index.reshape(b, n_edges, 2)  # Ensure shape is (batch_size, n_edges, 2)
    # Calculate fixed node offsets for each graph in the batch
    node_offsets = torch.arange(0, b * n, n, device=edge_index.device, dtype=torch.long)
    # Add offsets to edge indices directly using broadcasting
    edge_index = edge_index + node_offsets.view(-1, 1, 1)
    # Reshape and transpose to (2, batch_size*n_edges)
    edge_index = edge_index.reshape(-1, 2).t()
    
    return h, edge_index, edge_attr, node_mask, edge_mask, node_attr, linker_mask

def filter_valid_edges(edge_index, edge_attr, edge_mask):
    """
    Filter out invalid edges based on edge_mask.
    
    Args:
        edge_index: (2, num_edges) tensor of edge indices
        edge_attr: (num_edges, edge_feat_dim) tensor of edge attributes
        edge_mask: (num_edges, 1) tensor indicating valid edges (1) vs invalid (0)
    
    Returns:
        filtered_edge_index, filtered_edge_attr, filtered_edge_mask
    """
    # Flatten edge_mask to 1D
    valid_mask = edge_mask.squeeze(-1).bool()  # (num_edges,)
    
    # Filter edge_index, edge_attr, and edge_mask
    filtered_edge_index = edge_index[:, valid_mask]  # (2, num_valid_edges)
    filtered_edge_attr = edge_attr[valid_mask]  # (num_valid_edges, edge_feat_dim)
    filtered_edge_mask = edge_mask[valid_mask]  # (num_valid_edges, 1)
    
    return filtered_edge_index, filtered_edge_attr, filtered_edge_mask

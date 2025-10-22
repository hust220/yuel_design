import torch
import numpy as np

# Distance binning configurations
DISTANCE_BIN_CONFIGS = {
    "b12": {
        "edges": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, float("inf")],
        "midpoints": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0, 12.5, 15.0, 20.0, 35.0]
    },
}

# Functions to get configuration values (for backward compatibility)
def get_bin_edges(config_name: str):
    """Get bin edges for the specified configuration"""
    return DISTANCE_BIN_CONFIGS[config_name]["edges"]

def get_bin_midpoints(config_name: str):
    """Get bin midpoints for the specified configuration"""
    return DISTANCE_BIN_CONFIGS[config_name]["midpoints"]

def get_num_distance_bins(config_name: str):
    """Get number of distance bins for the specified configuration"""
    return len(DISTANCE_BIN_CONFIGS[config_name]["edges"])

def bin_distances(distances: torch.Tensor, config_name: str) -> torch.Tensor:
    """Bin continuous distances into discrete classes using the specified configuration"""
    config = DISTANCE_BIN_CONFIGS[config_name]
    edges = torch.tensor(config["edges"], device=distances.device, dtype=distances.dtype)
    distances = distances.clamp_min(0)
    x = distances.unsqueeze(-1) <= edges
    return x.float().argmax(dim=-1).long()

def classes_to_distances(classes: torch.Tensor, config_name: str) -> torch.Tensor:
    """Convert discrete classes back to distance values using bin midpoints"""
    config = DISTANCE_BIN_CONFIGS[config_name]
    mids = torch.tensor(config["midpoints"], device=classes.device)
    return mids[classes]

def discretize_distance_numpy(distance: float, config_name: str) -> int:
    """Discretize a single distance value using numpy (for use in data processing)"""
    config = DISTANCE_BIN_CONFIGS[config_name]
    edges = config["edges"]
    distance = max(0, distance)
    for i, edge in enumerate(edges):
        if distance <= edge:
            return i
    return len(edges) - 1

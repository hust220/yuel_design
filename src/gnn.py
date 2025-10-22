import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class Graph:
    """
    Custom graph class to replace DGL functionality.
    
    This class represents a graph with nodes and edges, supporting both single graphs
    and batched graphs (multiple subgraphs combined into one graph).
    
    Attributes:
        num_nodes (int): Number of nodes in the graph
        num_edges (int): Number of edges in the graph
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
        ndata (dict): Node data storage
        edata (dict): Edge data storage
        batch_size (int): Number of subgraphs (1 for single graph, >1 for batched)
        batch_num_nodes (list): Number of nodes in each subgraph
        batch_num_edges (list): Number of edges in each subgraph
        node_offset (int): Node offset within batch (0 for single graph)
        edge_offset (int): Edge offset within batch (0 for single graph)
    """
    
    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Initialize graph with edge index and number of nodes.
        
        Args:
            edge_index: Tensor of shape [2, num_edges] containing source and target node indices
            num_nodes: Number of nodes in the graph
        """
        # Validate edge_index shape
        if edge_index.size(0) != 2:
            raise ValueError(f"edge_index should have shape [2, num_edges], got {edge_index.shape}")
        
        # Validate edge indices are in valid range
        if edge_index.numel() > 0:
            if edge_index.min() < 0 or edge_index.max() >= num_nodes:
                raise ValueError(f"Edge indices must be in range [0, {num_nodes})")
        
        self.num_nodes = num_nodes
        self.num_edges = edge_index.size(1)
        
        # Store edge information
        self.edge_index = edge_index.clone()
        
        # Node and edge data storage
        self.ndata = {}
        self.edata = {}
        
        # Batch information (for batched graphs)
        self.batch_size = 1
        self.batch_num_nodes = [num_nodes]
        self.batch_num_edges = [self.num_edges]
        self.node_offset = 0
        self.edge_offset = 0
    
    @property
    def src_nodes(self):
        """Source nodes of edges."""
        return self.edge_index[0]
    
    @property
    def dst_nodes(self):
        """Destination nodes of edges."""
        return self.edge_index[1]
    
    def add_node_data(self, key: str, data: torch.Tensor):
        """Add node data to the graph."""
        if data.size(0) != self.num_nodes:
            raise ValueError(f"Node data size {data.size(0)} doesn't match number of nodes {self.num_nodes}")
        self.ndata[key] = data
    
    def add_edge_data(self, key: str, data: torch.Tensor):
        """Add edge data to the graph."""
        if data.size(0) != self.num_edges:
            raise ValueError(f"Edge data size {data.size(0)} doesn't match number of edges {self.num_edges}")
        self.edata[key] = data
    
    def get_node_data(self, key: str) -> torch.Tensor:
        """Get node data by key. Raises KeyError if key not found."""
        return self.ndata[key]
    
    def get_edge_data(self, key: str) -> torch.Tensor:
        """Get edge data by key. Raises KeyError if key not found."""
        return self.edata[key]
    
    def has_node_data(self, key: str) -> bool:
        """Check if node data exists for the given key."""
        return key in self.ndata
    
    def has_edge_data(self, key: str) -> bool:
        """Check if edge data exists for the given key."""
        return key in self.edata
    
    def is_batched(self) -> bool:
        """Check if this graph is a batched graph (contains multiple subgraphs)."""
        return self.batch_size > 1
    
    def get_num_subgraphs(self) -> int:
        """Get the number of subgraphs in this batched graph."""
        return self.batch_size
    
    def to(self, device):
        """Move graph to device."""
        self.edge_index = self.edge_index.to(device)
        
        for key in self.ndata:
            self.ndata[key] = self.ndata[key].to(device)
        for key in self.edata:
            self.edata[key] = self.edata[key].to(device)
        
        return self
    
    def cuda(self):
        """Move graph to CUDA."""
        return self.to('cuda')
    
    def cpu(self):
        """Move graph to CPU."""
        return self.to('cpu')
    
    def clone(self):
        """Create a deep copy of the graph."""
        new_graph = Graph(self.edge_index.clone(), self.num_nodes)
        new_graph.ndata = {k: v.clone() for k, v in self.ndata.items()}
        new_graph.edata = {k: v.clone() for k, v in self.edata.items()}
        
        # Copy batch information
        new_graph.batch_size = self.batch_size
        new_graph.batch_num_nodes = self.batch_num_nodes.copy()
        new_graph.batch_num_edges = self.batch_num_edges.copy()
        new_graph.node_offset = self.node_offset
        new_graph.edge_offset = self.edge_offset
        
        return new_graph
    
    def __repr__(self):
        return f"Graph(num_nodes={self.num_nodes}, num_edges={self.num_edges})"


def batch(graphs: List[Graph]) -> Graph:
    """
    Batch multiple graphs into a single graph.
    
    Args:
        graphs: List of Graph objects to batch
        
    Returns:
        Batched Graph object
    """
    if not graphs:
        raise ValueError("Cannot batch empty list of graphs")
    
    if len(graphs) == 1:
        return graphs[0].clone()
    
    # Clone all graphs to avoid side effects
    graphs = [g.clone() for g in graphs]
    
    # Determine target device (use first non-empty graph's device)
    target_device = None
    for g in graphs:
        if g.num_nodes > 0:
            target_device = g.edge_index.device
            break
    
    # Move all graphs to target device
    if target_device is not None:
        for g in graphs:
            if g.edge_index.device != target_device:
                g.to(target_device)
    
    # Calculate offsets for nodes and edges
    node_offsets = [0]
    edge_offsets = [0]
    
    for i, graph in enumerate(graphs):
        node_offsets.append(node_offsets[-1] + graph.num_nodes)
        edge_offsets.append(edge_offsets[-1] + graph.num_edges)
    
    # Combine edge indices with offsets
    batched_edge_index = []
    for i, graph in enumerate(graphs):
        if graph.num_edges > 0:  # Only add non-empty edge indices
            offset = node_offsets[i]
            edge_index = graph.edge_index + offset
            batched_edge_index.append(edge_index)
    
    # Handle empty edge list - preserve dtype from original graphs
    if not batched_edge_index:
        # Get dtype from first graph's edge_index
        dtype = graphs[0].edge_index.dtype if graphs else torch.long
        batched_edge_index = torch.empty((2, 0), dtype=dtype)
    else:
        batched_edge_index = torch.cat(batched_edge_index, dim=1)
    
    total_nodes = sum(graph.num_nodes for graph in graphs)
    
    # Create batched graph
    batched_graph = Graph(batched_edge_index, total_nodes)
    batched_graph.batch_size = len(graphs)
    batched_graph.batch_num_nodes = [g.num_nodes for g in graphs]
    batched_graph.batch_num_edges = [g.num_edges for g in graphs]
    # For batched graphs, node_offset and edge_offset are not meaningful
    # as they represent offsets within the batch, not the batch itself
    batched_graph.node_offset = 0
    batched_graph.edge_offset = 0
    
    # Combine node data
    node_data_keys = set()
    for graph in graphs:
        node_data_keys.update(graph.ndata.keys())
    
    for key in node_data_keys:
        data_list = []
        # Find the first graph that has this key to get shape and dtype info
        reference_graph = None
        for graph in graphs:
            if key in graph.ndata:
                reference_graph = graph
                break
        
        if reference_graph is None:
            # No graph has this key, skip it
            continue
            
        for graph in graphs:
            if key in graph.ndata:
                data_list.append(graph.ndata[key])
            else:
                # Create zero tensor for missing data
                shape = list(reference_graph.ndata[key].shape)
                shape[0] = graph.num_nodes
                zero_data = torch.zeros(shape, dtype=reference_graph.ndata[key].dtype, device=reference_graph.ndata[key].device)
                data_list.append(zero_data)
        
        if data_list:
            batched_graph.ndata[key] = torch.cat(data_list, dim=0)
    
    # Combine edge data
    edge_data_keys = set()
    for graph in graphs:
        edge_data_keys.update(graph.edata.keys())
    
    for key in edge_data_keys:
        data_list = []
        # Find the first graph that has this key to get shape and dtype info
        reference_graph = None
        for graph in graphs:
            if key in graph.edata:
                reference_graph = graph
                break
        
        if reference_graph is None:
            # No graph has this key, skip it
            continue
            
        for graph in graphs:
            if key in graph.edata:
                data_list.append(graph.edata[key])
            else:
                # Create zero tensor for missing data
                shape = list(reference_graph.edata[key].shape)
                shape[0] = graph.num_edges
                zero_data = torch.zeros(shape, dtype=reference_graph.edata[key].dtype, device=reference_graph.edata[key].device)
                data_list.append(zero_data)
        
        if data_list:
            batched_graph.edata[key] = torch.cat(data_list, dim=0)
    
    return batched_graph


def graph(edge_index: torch.Tensor, num_nodes: int) -> Graph:
    """
    Create a graph from edge index and number of nodes.
    This is a convenience function to match DGL's API.
    
    Args:
        edge_index: Tensor of shape [2, num_edges] containing source and target node indices
        num_nodes: Number of nodes in the graph
        
    Returns:
        Graph object
    """
    return Graph(edge_index, num_nodes)



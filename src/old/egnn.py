import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src import utils

def get_activation(activation_name):
    """Convert activation name string to PyTorch activation function"""
    if activation_name == 'silu':
        return nn.SiLU()
    elif activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")

def attention_aggregation(att_logits, row, col, num_nodes, bidirectional=False, edge_feat=None):
    max_logits = unsorted_segment_sum(att_logits, row, num_nodes, 1.0, 'max') # You need a max aggregation
    if bidirectional:
        max_logits_col = unsorted_segment_sum(att_logits, col, num_nodes, 1.0, 'max')
        max_logits = torch.max(max_logits, max_logits_col, dim=0)

    exp_logits = torch.exp(att_logits - max_logits[row]) # Shift for stability
    sum_exp_logits = unsorted_segment_sum(exp_logits, row, num_nodes, 1.0, 'sum') # You need a sum aggregation
    if bidirectional:
        exp_logits_col = torch.exp(att_logits - max_logits_col[col])
        sum_exp_logits_col = unsorted_segment_sum(exp_logits_col, col, num_nodes, 1.0, 'sum')
        sum_exp_logits = sum_exp_logits + sum_exp_logits_col

    normalized_att_weights = exp_logits / (sum_exp_logits[row] + 1e-8)
    edge_feat_norm = edge_feat * normalized_att_weights

    edge_feat_norm_col = None
    if bidirectional:
        normalized_att_weights_col = exp_logits_col / (sum_exp_logits_col[col] + 1e-8)
        edge_feat_norm_col = edge_feat * normalized_att_weights_col

    return edge_feat_norm, edge_feat_norm_col


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, activation,
                 edges_in_d=0, nodes_att_dim=0, attention=False, normalization=None, bidirectional=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.bidirectional = bidirectional

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            nn.LayerNorm(hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.LayerNorm(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )
        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Linear(hidden_nf, 1)
            # self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_logits = self.att_mlp(mij)
        else:
            att_logits = None

        if edge_mask is not None:
            mij = mij * edge_mask[:, None]
            
        return mij, att_logits

    # edge_index: (2, E)
    def node_model(self, x, edge_index, edge_feat, att_logits, node_attr):
        row, col = edge_index
        num_nodes = x.size(0)

        if self.attention and att_logits is not None:
            edge_feat_norm, edge_feat_norm_col = attention_aggregation(att_logits, row, col, num_nodes, self.bidirectional, edge_feat)
            agg = unsorted_segment_sum(edge_feat_norm, row, num_segments=num_nodes, # num_nodes replaces x.size(0) for clarity
                                     normalization_factor=self.normalization_factor, # This factor might be redundant now
                                     aggregation_method=self.aggregation_method)
            if self.bidirectional:
                agg_col = unsorted_segment_sum(edge_feat_norm_col, col, num_segments=num_nodes,
                                            normalization_factor=self.normalization_factor,
                                            aggregation_method=self.aggregation_method)
                agg = agg + agg_col
        else:
            agg = unsorted_segment_sum(edge_feat, row, num_segments=num_nodes,
                                     normalization_factor=self.normalization_factor,
                                     aggregation_method=self.aggregation_method)
            if self.bidirectional:
                # Aggregate messages from incoming edges
                agg_col = unsorted_segment_sum(edge_feat, col, num_segments=x.size(0),
                                            normalization_factor=self.normalization_factor,
                                            aggregation_method=self.aggregation_method)
                # Combine incoming and outgoing messages
                agg = agg + agg_col

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)

        out = x + self.node_mlp(agg)
        return out

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, att_logits = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h = self.node_model(h, edge_index, edge_feat, att_logits, node_attr)
        if node_mask is not None:
            h = h * node_mask[:, None]
        return h, att_logits

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation=nn.SiLU(), tanh=False, coords_range=10.0, bidirectional=False):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.bidirectional = bidirectional
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
                    
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            nn.LayerNorm(hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            activation,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, att_logits, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.bidirectional:
            input_tensor_col = torch.cat([h[col], h[row], edge_attr], dim=1)

        if att_logits is not None:
            input_tensor, input_tensor_col = attention_aggregation(att_logits, row, col, coord.size(0), self.bidirectional, input_tensor)

        trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask[:, None]
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                     normalization_factor=self.normalization_factor,
                                     aggregation_method=self.aggregation_method)

        if self.bidirectional:
            trans_col = coord_diff * self.coord_mlp(input_tensor_col)
            if edge_mask is not None:
                trans_col = trans_col * edge_mask[:, None]
            agg_col = unsorted_segment_sum(trans_col, col, num_segments=coord.size(0),
                                        normalization_factor=self.normalization_factor,
                                        aggregation_method=self.aggregation_method)
            agg = agg + agg_col

        # coord = coord + agg
        return agg

    def forward(
            self, h, coord, edge_index, coord_diff, edge_attr=None, att_logits=None, node_mask=None, edge_mask=None
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, att_logits, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask[:, None]
        return coord

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', activation=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, 
                 normalization_factor=100, aggregation_method='sum', bidirectional=False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf+2,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method,
                                              bidirectional=bidirectional))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf+2, activation=activation, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,
                                                       bidirectional=bidirectional))
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, x, edge_index, edge_attr, node_mask=None, edge_mask=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        
        for i in range(0, self.n_layers):
            h, att_logits = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            att_logits=att_logits,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        if node_mask is not None:
            h = h * node_mask[:, None]
            
        return h, x

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, device='cpu', activation=nn.SiLU(), 
                 n_layers=3, attention=False, node_attr_nf=0, edge_feat_nf=2,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=1,
                 normalization_factor=100, aggregation_method='sum', bidirectional=False, low_memory=False, use_checkpoint=False):
        super(EGNN, self).__init__()
        
        if isinstance(activation, str):
            activation = get_activation(activation)
            
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.node_attr_nf = node_attr_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.use_checkpoint = use_checkpoint
        self.low_memory = low_memory
        if self.low_memory:
            self.use_checkpoint = False
        # Add node type embedding
        self.node_type_embedding = nn.Embedding(2, hidden_nf)  # 2 types: movable and fixed
        
        self.embedding = nn.Linear(in_node_nf + node_attr_nf, self.hidden_nf)  # +hidden_nf for type embedding
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               bidirectional=bidirectional))
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, x, edge_index, edge_attr, node_attr=None, node_mask=None, edge_mask=None):
        distances, _ = coord2diff(x, edge_index)
        edge_attr = torch.cat([distances, edge_attr], dim=1)

        if self.node_attr_nf > 0:
            h = torch.cat([h, node_attr], dim=1)
            
        h = self.embedding(h)
        
        # Define checkpoint wrapper function outside the loop
        def checkpoint_wrapper(layer, h, x, edge_index, edge_attr, node_mask, edge_mask):
            return layer(h, x, edge_index, edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        
        def blocks_wrapper(h, x, edge_index, edge_attr, node_mask, edge_mask):
            for i in range(0, self.n_layers):
                layer = self._modules["e_block_%d" % i]
                if self.use_checkpoint and i % 2 == 0 and i < self.n_layers - 1:
                    h, x = checkpoint(checkpoint_wrapper, layer, h, x, edge_index, edge_attr, node_mask, edge_mask, use_reentrant=False)
                else:
                    h, x = layer(h, x, edge_index, edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            return h, x

        if self.low_memory:
            h, x = checkpoint(blocks_wrapper, h, x, edge_index, edge_attr, node_mask, edge_mask, use_reentrant=False)
        else:
            h, x = blocks_wrapper(h, x, edge_index, edge_attr, node_mask, edge_mask)

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask[:, None]
        return h, x

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor=1.0, aggregation_method: str = 'sum'):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
        Added 'max' aggregation.
    """
    if aggregation_method == 'sum':
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids_expanded, data)
        # Apply normalization_factor for 'sum' method as per original code
        result = result / normalization_factor
    elif aggregation_method == 'mean':
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids_expanded, data)

        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids_expanded, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    elif aggregation_method == 'max':
        # For 'max' aggregation, initialize with negative infinity for robust max finding
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, float('-inf')) # Initialize with -inf
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
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")
    
    return result

class GCL2D(nn.Module):
    def __init__(self,
        hidden_nf,
        in_node_nf, out_node_nf,
        in_edge_nf, out_edge_nf,
        aggregation_method='sum',
        normalization_factor=1,
        activation=nn.SiLU(),
        attention=True,
        bidirectional=False,
    ):

        super(GCL2D, self).__init__()

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.bidirectional = bidirectional

        # Convert activation string to PyTorch module if needed
        if isinstance(activation, str):
            activation = get_activation(activation)

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_nf + 2*in_node_nf + 1, hidden_nf),
            nn.LayerNorm(hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        self.node_mlp = nn.Sequential(
            nn.Linear(out_edge_nf + in_node_nf, hidden_nf),
            nn.BatchNorm1d(hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
            nn.BatchNorm1d(out_node_nf),
        )

        if self.attention:
            self.att_mlp = nn.Linear(hidden_nf+1, 1)

    def edge_model(self, source, target, edge_feat, edge_mask):
        # source: (n_edges, d), target: (n_edges, d), edge_feat: (n_edges, d_edge)
        att = torch.sum(source * target, dim=1, keepdim=True) # (n_edges, 1)
        edge_feat_input = torch.cat([source, target, edge_feat, att], dim=1)
        # edge_mlp: (d_edge + 2*d + 1, hidden_nf) -> (n_edges, hidden_nf)
        mij = self.edge_mlp(edge_feat_input) # (n_edges, hidden_nf)
        
        # ResNet: add residual connection
        # mij = mij + edge_feat

        if self.attention:
            # att_mlp: hidden_nf+1 -> 1
            att = self.att_mlp(torch.cat([att, mij], dim=1)) 
        else:
            att = None

        return mij*edge_mask, att*edge_mask

    # edge_index: (2, E)
    def node_model(self, x, edge_index, edge_feat, attention, node_mask):
        # edge_feat: (n_edges, out_edge_nf)
        row, col = edge_index
        num_nodes = x.size(0)

        if attention is not None:
            edge_feat_norm, edge_feat_norm_col = attention_aggregation(attention, row, col, num_nodes, self.bidirectional, edge_feat)
            agg = unsorted_segment_sum(edge_feat_norm, row, num_segments=num_nodes, # num_nodes replaces x.size(0) for clarity
                                     normalization_factor=self.normalization_factor, # This factor might be redundant now
                                     aggregation_method=self.aggregation_method)
            if self.bidirectional:
                agg_col = unsorted_segment_sum(edge_feat_norm_col, col, num_segments=num_nodes,
                                            normalization_factor=self.normalization_factor,
                                            aggregation_method=self.aggregation_method)
                agg = agg + agg_col
        else:
            agg = unsorted_segment_sum(edge_feat, row, num_segments=num_nodes,
                                     normalization_factor=self.normalization_factor,
                                     aggregation_method=self.aggregation_method)
            if self.bidirectional:
                # Aggregate messages from incoming edges
                agg_col = unsorted_segment_sum(edge_feat, col, num_segments=x.size(0),
                                            normalization_factor=self.normalization_factor,
                                            aggregation_method=self.aggregation_method)
                # Combine incoming and outgoing messages
                agg = agg + agg_col

        # agg: (n_nodes, out_edge_nf+in_node_nf)
        agg = torch.cat([x, agg], dim=1)

        # node_mlp: out_edge_nf + in_node_nf -> out_node_nf
        out = x + self.node_mlp(agg)
        return out*node_mask

    def forward(self, h, edge_index, edge_feat, node_mask, edge_mask):
        # edge_index: (2, n_edges), h: (b*n, d), edge_feat: (b*n_edges, d_edge)
        # row: (n_edges,), col: (n_edges,)
        # h[row]: (n_edges, d), h[col]: (n_edges, d)
        row, col = edge_index
        edge_feat, attention = self.edge_model(h[row], h[col], edge_feat, edge_mask)
        h = self.node_model(h, edge_index, edge_feat, attention, node_mask)
        return h*node_mask, edge_feat*edge_mask

class GNN(nn.Module):
    def __init__(self,
        hidden_nf,
        in_node_nf, out_node_nf,
        in_edge_nf, out_edge_nf,
        activation=nn.SiLU(),
        n_layers=16,
        attention=True,
        normalization_factor=1,
        aggregation_method='sum', 
        bidirectional=False,
    ):

        super(GNN, self).__init__()

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.n_layers = n_layers

        self.embedding_node = nn.Linear(in_node_nf, hidden_nf)  # +hidden_nf for type embedding
        self.embedding_node_out = nn.Linear(hidden_nf, out_node_nf)
        self.embedding_edge = nn.Linear(in_edge_nf, hidden_nf)
        self.embedding_edge_out = nn.Linear(hidden_nf, out_edge_nf)
        
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL2D(
                hidden_nf=hidden_nf,
                in_node_nf=hidden_nf,
                out_node_nf=hidden_nf,
                in_edge_nf=hidden_nf,
                out_edge_nf=hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                activation=activation,
                attention=attention,
                bidirectional=bidirectional,
            ))
        
    def forward(self, h, edge_index, edge_feat, node_mask, edge_mask):            
        h = self.embedding_node(h)
        e = self.embedding_edge(edge_feat)
        
        for i in range(0, self.n_layers):
            h, e = self._modules["gcl_%d" % i](
                h=h,
                edge_index=edge_index,
                edge_feat=e,
                node_mask=node_mask,
                edge_mask=edge_mask
            )

        h = self.embedding_node_out(h)
        e = self.embedding_edge_out(e)

        return h*node_mask, e*edge_mask


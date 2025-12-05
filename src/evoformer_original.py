
import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partialmethod, partial
from typing import Any, Tuple, List, Callable, Optional, Union, Dict
import math
import numpy as np
from scipy.stats import truncnorm


# ============================================================================
# UTILITY FUNCTIONS (from primitives.py)
# ============================================================================

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


# ============================================================================
# TENSOR UTILITIES (from tensor_utils.py)
# ============================================================================

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.
    """
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    # Get the batch dimension from the first tensor
    def _get_batch_dim(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape[0]
        elif isinstance(obj, dict):
            for v in obj.values():
                batch_dim = _get_batch_dim(v)
                if batch_dim is not None:
                    return batch_dim
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                batch_dim = _get_batch_dim(item)
                if batch_dim is not None:
                    return batch_dim
        return None

    batch_dim = _get_batch_dim(inputs)
    if batch_dim is None or batch_dim == 0:
        return layer(**inputs)

    # Simple chunking implementation
    if batch_dim <= chunk_size:
        return layer(**inputs)

    # Chunk the inputs
    def _chunk_tensor(tensor, start, end):
        if isinstance(tensor, torch.Tensor):
            return tensor[start:end]
        elif isinstance(tensor, dict):
            return {k: _chunk_tensor(v, start, end) for k, v in tensor.items()}
        elif isinstance(tensor, (list, tuple)):
            return type(tensor)(_chunk_tensor(item, start, end) for item in tensor)
        else:
            return tensor

    outputs = []
    for i in range(0, batch_dim, chunk_size):
        end_i = min(i + chunk_size, batch_dim)
        chunked_inputs = _chunk_tensor(inputs, i, end_i)
        output = layer(**chunked_inputs)
        outputs.append(output)

    # Concatenate outputs
    def _concat_outputs(outputs):
        if not outputs:
            return None
        
        first_output = outputs[0]
        if isinstance(first_output, torch.Tensor):
            return torch.cat(outputs, dim=0)
        elif isinstance(first_output, dict):
            result = {}
            for key in first_output.keys():
                result[key] = torch.cat([out[key] for out in outputs], dim=0)
            return result
        elif isinstance(first_output, (list, tuple)):
            return type(first_output)(
                torch.cat([out[i] for out in outputs], dim=0) 
                for i in range(len(first_output))
            )
        else:
            return outputs

    return _concat_outputs(outputs)


# ============================================================================
# ATTENTION UTILITIES (from primitives.py)
# ============================================================================

@torch.jit.ignore
def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    if(d is torch.bfloat16):
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


#@torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, Q, C_hidden]
    query = permute_final_dims(query, (1, 0, 2))

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 2, 0))

    # [*, H, V, C_hidden]
    value = permute_final_dims(value, (1, 0, 2))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a

# ============================================================================
# LINEAR AND LAYERNORM CLASSES (from primitives.py)
# ============================================================================

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        if d is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out



# ============================================================================
# ATTENTION CLASSES (from primitives.py)
# ============================================================================

class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor,
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if(self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_lma: bool = False,
        q_chunk_size: Optional[int] = None,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_lma:
                Whether to use low-memory attention
            q_chunk_size:
                Query chunk size (for LMA)
            kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if(biases is None):
            biases = []
        if(use_lma and (q_chunk_size is None or kv_chunk_size is None)):
            raise ValueError(
                "If use_lma is specified, q_chunk_size and kv_chunk_size must "
                "be provided"
            )

        q, k, v = self._prep_qkv(q_x, kv_x)

        if(use_lma):
            biases = [
                b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],))
                for b in biases
            ]

            o = _lma(q, k, v, biases, q_chunk_size, kv_chunk_size)
        else:
            o = _attention(q, k, v, biases)

        o = self._wrap_up(o, q_x)

        return o




# ============================================================================
# CHECKPOINTING MODULE
# ============================================================================

BLOCK_ARG = Any
BLOCK_ARGS = List[BLOCK_ARG]


def get_checkpoint_fn():
    # Use PyTorch's built-in checkpointing
    checkpoint = torch.utils.checkpoint.checkpoint
    return checkpoint


@torch.jit.ignore
def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.
    Implements Subsection 1.11.8
    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    Returns:
        The output of the final block
    """
    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)
        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn()

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args)
        args = wrap(args)

    return args

# ============================================================================
# PAIR TRANSITION MODULE
# ============================================================================

class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) * mask

        return z

    @torch.jit.ignore
    def _chunk(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)

        return z



# ============================================================================
# TRIANGULAR ATTENTION MODULE
# ============================================================================

class TriangleAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, starting, inf=1e9
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = float(inf)

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }
        return chunk_layer(
            partial(self.mha),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
        )

    def forward(self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        # Shape annotations assume self.starting. Else, I and J are flipped
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        inf_value = float(self.inf)
        inf_tensor = torch.full_like(mask, inf_value, dtype=mask.dtype)
        mask_bias = (inf_tensor * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size)
        else:
            x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


class TriangleAttentionStartingNode(TriangleAttention):
    """
    Implements Algorithm 13.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)


# ============================================================================
# TRIANGULAR MULTIPLICATIVE UPDATE MODULE
# ============================================================================

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        b = b * mask
        x = self._combine_projections(a, b)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        z = x * g

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    def _combine_projections(self,
        a: torch.Tensor,  # [*, N_i, N_k, C]
        b: torch.Tensor,  # [*, N_j, N_k, C]
    ):
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 0, 1)),
            permute_final_dims(b, (2, 1, 0)),
        )

        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    def _combine_projections(self,
        a: torch.Tensor,  # [*, N_k, N_i, C]
        b: torch.Tensor,  # [*, N_k, N_j, C]
    ):
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 1, 0)),
            permute_final_dims(b, (2, 0, 1)),
        )

        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))

class SequenceAttention(nn.Module):
    """
    Single sequence attention with pair bias.
    Replaces MSA attention for single sequence processing.
    """
    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        super(SequenceAttention, self).__init__()
        
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = float(inf)
        
        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        
        self.linear_z = Linear(c_z, no_heads, bias=False, init="normal")
        
        self.linear_q = Linear(c_m, c_hidden * no_heads, bias=False, init="glorot")
        self.linear_k = Linear(c_m, c_hidden * no_heads, bias=False, init="glorot")
        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False, init="glorot")
        self.linear_o = Linear(c_hidden * no_heads, c_m, init="final")
        
        self.linear_g = Linear(c_m, c_hidden * no_heads, init="gating")
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, seq, z, mask=None, chunk_size=None):
        """
        Args:
            seq: [*, N_res, C_m] sequence embedding
            z: [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res] sequence mask
        Returns:
            seq: [*, N_res, C_m] updated sequence embedding
        """
        # Layer normalization
        seq = self.layer_norm_m(seq)
        z = self.layer_norm_z(z)
        
        # Prepare query, key, value
        q = self.linear_q(seq)
        k = self.linear_k(seq)
        v = self.linear_v(seq)
        
        # Reshape for multi-head attention: [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # Scale query
        q = q / (self.c_hidden ** 0.5)

        # Compute attention scores per head
        # q_h: [*, H, N_res, C_hidden], k_h: [*, H, N_res, C_hidden]
        q_h = q.transpose(-3, -2)
        k_h = k.transpose(-3, -2)
        # scores_h: [*, H, N_res, N_res]
        scores_h = torch.matmul(q_h, k_h.transpose(-1, -2))
        # scores: [*, N_res, N_res, H]
        scores = scores_h.permute(0, 2, 3, 1)

        # Add pair bias: [*, N_res, N_res, H]
        z_bias = self.linear_z(z)  # [*, N_res, N_res, H]
        scores = scores + z_bias

        # Apply key mask: broadcast over query and heads
        if mask is not None:
            # key_mask: [*, 1, N_res, 1]
            inf_value = float(self.inf)
            inf_tensor = torch.full_like(mask, inf_value, dtype=mask.dtype)
            key_mask = (inf_tensor * (mask - 1)).unsqueeze(-2).unsqueeze(-1)
            scores = scores + key_mask

        # Softmax over keys dimension
        attn = torch.softmax(scores, dim=2)  # [*, N_res, N_res, H]

        # Prepare for value aggregation
        # attn_h: [*, H, N_res, N_res]
        attn_h = attn.permute(0, 3, 1, 2)
        # v_h: [*, H, N_res, C_hidden]
        v_h = v.transpose(-3, -2)
        # out_h: [*, H, N_res, C_hidden]
        out_h = torch.matmul(attn_h, v_h)
        # out: [*, N_res, H, C_hidden]
        out = out_h.transpose(-3, -2)

        # Gating before output projection
        g = self.sigmoid(self.linear_g(seq))  # [*, N_res, H*C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))  # [*, N_res, H, C_hidden]
        out = out * g

        # Reshape and project to C_m
        out = out.reshape(out.shape[:-2] + (-1,))  # [*, N_res, H*C_hidden]
        out = self.linear_o(out)  # [*, N_res, C_m]
        
        return out


class SequenceTransition(nn.Module):
    """
    Feed-forward network for sequence activations.
    Replaces MSATransition for single sequence processing.
    """
    def __init__(self, c_m, n):
        super(SequenceTransition, self).__init__()
        
        self.c_m = c_m
        self.n = n
        
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, n * c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(n * c_m, c_m, init="final")
    
    def forward(self, seq, mask=None, chunk_size=None):
        """
        Args:
            seq: [*, N_res, C_m] sequence embedding
            mask: [*, N_res] sequence mask
        Returns:
            seq: [*, N_res, C_m] updated sequence embedding
        """
        if mask is None:
            mask = seq.new_ones(seq.shape[:-1])
        
        mask = mask.unsqueeze(-1)
        
        seq = self.layer_norm(seq)
        
        seq = self.linear_1(seq)
        seq = self.relu(seq)
        seq = self.linear_2(seq) * mask
        
        return seq


class SequenceOuterProduct(nn.Module):
    """
    Outer product for sequence embeddings.
    Replaces OuterProductMean for single sequence processing.
    """
    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        super(SequenceOuterProduct, self).__init__()
        
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")
        # self.linear_out = Linear(c_hidden, c_z, init="final")
    
    def forward(self, seq, mask=None, chunk_size=None):
        """
        Args:
            seq: [*, N_res, C_m] sequence embedding
            mask: [*, N_res] sequence mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = seq.new_ones(seq.shape[:-1])
        
        seq = self.layer_norm(seq)
        
        mask = mask.unsqueeze(-1)
        a = self.linear_1(seq) * mask
        b = self.linear_2(seq) * mask
        
        # Outer product: [*, N_res, N_res, C_hidden, C_hidden]
        outer = torch.einsum("...ia,...jb->...ijab", a, b)
        
        # Alternative strategy
        # outer = a[:, :, None, :] * b[:, None, :, :]
        
        # Flatten and project
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        
        # Normalize
        norm = torch.einsum("...ia,...ja->...ij", mask, mask)
        outer = outer / (self.eps + norm.unsqueeze(-1))
        
        return outer


class EvoformerBlockCore(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_seq: int,
        no_heads_pair: int,
        transition_n: int,
        inf: float,
        eps: float,
        _is_extra_seq_stack: bool = False,
    ):
        super(EvoformerBlockCore, self).__init__()

        self.seq_transition = SequenceTransition(
            c_m=c_m,
            n=transition_n,
        )

        self.seq_outer_product = SequenceOuterProduct(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

    def forward(
        self,
        seq: torch.Tensor,
        z: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        seq_trans_mask = seq_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None

        seq = seq + self.seq_transition(
            seq, mask=seq_trans_mask, chunk_size=chunk_size
        )
        z = z + self.seq_outer_product(
            seq, mask=seq_mask, chunk_size=chunk_size
        )
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.pair_transition(
            z, mask=pair_trans_mask, chunk_size=chunk_size
        )

        return seq, z


class EvoformerBlock(nn.Module):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_seq_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_seq: int,
        no_heads_pair: int,
        transition_n: int,
        inf: float,
        eps: float,
    ):
        super(EvoformerBlock, self).__init__()

        self.seq_att = SequenceAttention(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_seq_att,
            no_heads=no_heads_seq,
            inf=inf,
        )

        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_seq=no_heads_seq,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            inf=inf,
            eps=eps,
        )

    def forward(self,
        seq: torch.Tensor,
        z: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = seq + self.seq_att(seq, z=z, mask=seq_mask, chunk_size=chunk_size)
        seq, z = self.core(
            seq,
            z,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            _mask_trans=_mask_trans,
        )

        return seq, z




class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_seq_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_seq: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                Sequence channel dimension
            c_z:
                Pair channel dimension
            c_hidden_seq_att:
                Hidden dimension in sequence attention
            c_hidden_opm:
                Hidden dimension in outer product module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_seq:
                Number of heads used for sequence attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the SequenceTransition
                hidden dimension
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_seq_att=c_hidden_seq_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_seq=no_heads_seq,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    def forward(self,
        seq: torch.Tensor,
        z: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            seq:
                [*, N_res, C_m] sequence embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            seq_mask:
                [*, N_res] sequence mask
            pair_mask:
                [*, N_res, N_res] pair mask
        Returns:
            seq:
                [*, N_res, C_m] sequence embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding
        """
        blocks = [
            partial(
                b,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if(self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args):
                torch.cuda.empty_cache()
                return block(*args)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]
        blocks_per_ckpt = self.blocks_per_ckpt
        if(not torch.is_grad_enabled()):
            blocks_per_ckpt = None

        seq, z = checkpoint_blocks(
            blocks,
            args=(seq, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        s = self.linear(seq)

        return seq, z, s



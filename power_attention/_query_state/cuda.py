## QUERY STATE CUSTOM OP
# Computes the query state forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from typing import Optional, Tuple
from power_attention._utils import compute_expanded_dim
from power_attention_cuda import compute_query_states as compute_query_states_cuda, query_states_bwd as query_states_bwd_cuda
from power_attention._config import eps

## FWD ##

@torch.library.custom_op('power_attention::query_state_forward', mutates_args=(), device_types='cuda')
def query_state_fwd(Q : torch.Tensor, S : torch.Tensor,
                    Y : Optional[torch.Tensor],
                    rowmax : Optional[torch.Tensor], deg : int,
                    scale : float, zero_initial_state : bool) -> torch.Tensor:
    """Compute query state forward pass.
    
    Input shapes:
        Q: [batch, num_chunks, chunk_size, head, dim] - Query tensor
        S: [batch, num_chunks, head, state_dim, dim] - State tensor
        Y: Optional[batch, num_chunks, chunk_size, head, dim] - Optional output tensor
        rowmax: Optional[batch, num_chunks, chunk_size, head] - Optional rowmax tensor
        deg: int - Power attention degree
        scale: float - Scaling factor
        zero_initial_state: bool - Whether the initial state is zero
        
    Output shapes:
        O: [batch, num_chunks, chunk_size, head, dim] - Output tensor

    Input restrictions:
        - Q contiguous along the last dimension
        - Q feature dimension must be 32 or 64
        - fp16 or bf16 only
        - chunk_size must be a multiple of 16, and at least 128
    """
    # TODO(jbuckman): Figure out how to make this traceable despite buffer reuse, and remove clone()
    Y = Y.clone() if Y is not None else None
    O, _ = compute_query_states_cuda(Q, S, Y, rowmax,
                                        None, deg, scale, zero_initial_state, eps,
                                        False, True)
    return O

# Fake implementation for tracing and testing
@query_state_fwd.register_fake
def query_state_fwd_fake(Q, S, Y, rowmax, deg, scale, zero_initial_state):
    b, n, c, h, d = Q.shape
    return torch.empty(b, n, c, h, d, device=Q.device, dtype=Q.dtype)

## BWD ##

@torch.library.custom_op("power_attention::query_state_bwd", mutates_args=(), device_types='cuda')
def query_state_bwd(Q : torch.Tensor, S : torch.Tensor,
                               dO : torch.Tensor, rowmax : Optional[torch.Tensor],
                               deg : int, scale : float, zero_initial_state : bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute query state backward.

    Computes gradients with respect to inputs Q, S, s for the query state operation.
    
    Input shapes:
        Q: [batch, seq, chunk, head, dim] - Query tensor
        S: [batch, seq, head, expanded_dim, dim] - State tensor
        s: [batch, seq, head, expanded_dim] - State scaling tensor
        dO: [batch, seq, chunk, head, dim] - Gradient of loss with respect to output O
        do: [batch, seq, chunk, head] - Gradient of loss with respect to output scaling o
        rowmax: [batch, seq, head, expanded_dim] - Rowmax tensor
        deg: int - Power attention degree
        scale: float - Scaling factor
        zero_initial_state: bool - Whether the initial state is zero

    Output shapes:
        dQ: [batch, seq, chunk, head, dim] - Gradient with respect to Q
        dS: [batch, seq, head, expanded_dim, dim] - Gradient with respect to S
        dY_attn: [batch, seq, chunk, head, dim] - Gradient with respect to Y_attn

    Input restrictions:
        - Q, S contiguous along the last dimension
        - Q, S have same feature dimension, and it must be 32 or 64
        - Q, S have same dtype
        - fp16 or bf16 only
        - do, s must be float32
    """
    dQ, dS, _, dY_attn = query_states_bwd_cuda(
        Q, S, dO, rowmax, None, deg, scale, zero_initial_state, False, False)
    return dQ, dS, dY_attn

@query_state_bwd.register_fake
def query_state_bwd_fake(Q, S, dO, rowmax, deg, scale, zero_initial_state):
    b, n, c, h, d = Q.shape
    _, _, _, D, _ = S.shape
    return (torch.empty_like(Q), 
            torch.empty_like(S), 
            torch.empty_like(dO) if rowmax is not None else torch.empty([0], device=dO.device, dtype=dO.dtype))


## IMPL ##

# Define the primary query_state entrypoint
@torch.library.custom_op("power_attention::query_state", mutates_args=())
def query_state(Q : torch.Tensor, S : torch.Tensor, Y : Optional[torch.Tensor],
                rowmax : Optional[torch.Tensor],
                deg : int, scale : Optional[float], zero_initial_state : bool) -> torch.Tensor:
    r"""Compute query interaction with expanded state vectors.

    This function implements the query-state interaction from [1]. It computes how queries
    interact with the expanded state vectors using symmetric power embeddings, which provide
    an efficient way to compute higher-order attention patterns.

    For each query $Q_t$, we compute:

    $$Y_t = \phi(Q_t) \cdot S_{<t}$$

    where $\phi$ maps $Q_t$ to its symmetric power embedding of degree $deg$. This is equivalent to
    computing attention scores with keys and values from previous chunks, but using the
    compressed state representation for efficiency.

    The computation can be numerically unstable in fp16, so we provide two stabilization
    mechanisms:

    1. A scaling factor that scales the symmetric power embedding
    2. Optional output scaling using Y and rowmax from the attention computation

    Args:
        Q: Query tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
        S: State tensor of shape `(batch_size, num_chunks, num_heads, expanded_dim, head_dim)`.
           Contains expanded state vectors from update_state.
        Y: Optional output tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
           Used for output scaling when provided.
        rowmax: Optional scaling tensor of shape `(batch_size, num_chunks, chunk_size, num_heads)`.
           Used for matching the scale of the attention output and that of the query state output.
        deg: Power attention degree. Must be even for symmetric power formulation.
        scale: Optional stabilization factor. Defaults to state_dim for fp16, 1.0 otherwise.
            Helps prevent overflow in symmetric power computation.
        zero_initial_state: Whether the initial state should be treated as zero.

    Returns:
        O: Output tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`. 
           Note that the output is normalized by the minimum of the scale and the rowmax.

    Note:
        - Q must be contiguous along the last dimension
        - Feature dimension must be 32 or 64
        - Inputs must be fp16 or bf16
        - chunk_size must be at least 128 and a multiple of 16
        - Scaling is particularly important for deg > 2

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    b, n, c, h, d = Q.shape
    if scale is None:
        scale = 1.
    O = query_state_fwd(Q, S, Y, rowmax, deg, scale, zero_initial_state)
    return O
@query_state.register_fake
def query_state_fake(Q, S, Y, rowmax, deg, scale, zero_initial_state):
    b, n, c, h, d = Q.shape
    return torch.empty(b, n, c, h, d, device=Q.device, dtype=Q.dtype)
# Autograd setup
def query_state_setup(ctx, inputs, output):
    Q, S, Y, rowmax, deg, scale, zero_initial_state = inputs
    b, n, c, h, d = Q.shape
    if scale is None:
        scale = 1.
    ctx.save_for_backward(Q, S, rowmax)
    ctx.deg = deg
    ctx.scale = scale
    ctx.fused = Y is not None
    ctx.zero_initial_state = zero_initial_state
def query_state_backward(ctx, dO):
    Q, S, rowmax = ctx.saved_tensors
    dQ, dS, dY_attn = query_state_bwd(Q, S, dO, rowmax, ctx.deg, ctx.scale, ctx.zero_initial_state)
    if ctx.fused:
        dY = dY_attn
    else:
        dY = None
    return dQ, dS, dY, None, None, None, None
# Register autograd
torch.library.register_autograd(
    "power_attention::query_state", query_state_backward, setup_context=query_state_setup
)

## TUTORIAL ##
if __name__ == '__main__':
    from perf._inspect import print_runtime
    from power_attention._query_state.create_inputs import create_inputs

    # Hyperparameters
    b, n, c, h, d = (8, 8, 128, 16, 64)
    dtype = torch.float16
    # Create inputs
    inputs = create_inputs(b, n, c, h, d, dtype, 'cuda', requires_grad=True)

    # Benchmark
    print(f"Benchmarking query state \n {b=} {n=} {c=} {h=} {d=} {dtype}")
    print_runtime(query_state, *inputs)


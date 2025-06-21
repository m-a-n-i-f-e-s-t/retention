## UPDATE STATE CUSTOM OP
# Computes the update state forward pass and backward pass using CUDA.

## FWD ##
import torch
from power_attention_cuda import compute_update_states as update_states_fwd_cuda, update_states_bwd as update_states_bwd_cuda
from power_attention._utils import compute_expanded_dim
from typing import Tuple


@torch.library.custom_op("power_attention::update_state_fwd", mutates_args=(), device_types='cuda')
def update_state_fwd(K : torch.Tensor, V : torch.Tensor, deg : int) -> torch.Tensor:
    """Compute state update for symmetric power attention kernel.
    
    This operation computes the states needed for the symmetric power attention kernel.
    It processes chunks of the input sequence to compute intermediate states efficiently.

    Args:
        K: [batch, seq_len, chunks, heads, head_dim] - Key tensor
        V: [batch, seq_len, chunks, heads, head_dim] - Value tensor  
        deg: int - Degree of the power series expansion (right now must be 2)

    Returns:
        S: [batch, seq_len, heads, expanded_dim, head_dim] - Main state

    Shape Requirements:
        - head_dim must be 32 or 64
        - chunk_size must be a multiple of 16 and of Block_DT
    """
    S, s = update_states_fwd_cuda(K, V, deg, False, True)
    return S, s

@update_state_fwd.register_fake
def update_state_fwd_fake(K, V, deg):
    b, n, c, h, d = K.shape
    D = compute_expanded_dim(d, deg)
    S = torch.empty(b, n, h, D, d, device=K.device, dtype=K.dtype)
    s = torch.empty(b, n, h, D, device=K.device, dtype=K.dtype)
    return (S, s)

## BWD ##

@torch.library.custom_op("power_attention::update_state_bwd", mutates_args=(), device_types='cuda')
def update_state_bwd(K : torch.Tensor, V : torch.Tensor, dS : torch.Tensor, ds : torch.Tensor, deg : int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for chunk state operation.

    Computes gradients with respect to inputs K and V for the chunk state operation.
    
    Input shapes:
        K: [batch, chunk_n, chunk_size, heads, head_size] - Key tensor from forward pass
        V: [batch, chunk_n, chunk_size, heads, head_size] - Value tensor from forward pass
        dS: [batch, chunk_n, heads, expanded_dim, head_size] - Gradient of loss w.r.t. forward S output
        deg: int - Degree parameter used in forward pass

    Output shapes:
        dK: [batch, chunk_n, chunk_size, heads, head_size] - Gradient with respect to input K
        dV: [batch, chunk_n, chunk_size, heads, head_size] - Gradient with respect to input V

    Restrictions:
        - head_size must be a multiple of InnerBlock_DT
        - expanded_dim is computed based on head_size and deg
        - fp16 or bf16
    """
    dK, dV = update_states_bwd_cuda(K, V, dS, ds, deg)
    return dK, dV

@update_state_bwd.register_fake
def update_state_bwd_fake(K, V, dS, ds, deg):
    return torch.empty_like(K), torch.empty_like(V)

## IMPLEMENTATION ##

# Define the primary update_state entrypoint
@torch.library.custom_op("power_attention::update_state", mutates_args=())
def update_state(K : torch.Tensor, V : torch.Tensor, deg : int) -> torch.Tensor:
    r"""Compute expanded state vectors for symmetric power attention.
    
    This function implements the state expansion mechanism from [1]. It uses symmetric tensors
    to efficiently capture higher-order interactions between keys and values, achieving massive
    dimensionality reduction compared to full tensor products (e.g., 96% reduction for deg=4).

    For each chunk i, the state tensor accumulates symmetric outer products:

    $$
    S_{ij} = \sum_{t \in \text{chunk}_i} \phi(K_t)_i \phi(V_t)_j \quad \text{for } i+j < deg
    $$

    where ϕ maps vectors to their symmetric power embedding, i,j are power indices, and t
    iterates over positions in the chunk. By using symmetric tensors instead of full tensor
    products, we achieve both better performance and much smaller state sizes.

    The expanded dimension is computed as:

    $$
    \text{expanded_dim} = \binom{\text{head_dim} + deg - 1}{deg}
    $$

    Args:
        K: Key tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
        V: Value tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
        deg: Power attention degree. Controls the size of expanded state vectors.
            Must be even for symmetric power formulation.

    Returns:
        S: Expanded state tensor of shape `(batch_size, num_chunks, num_heads, expanded_dim, head_dim)`,
           where expanded_dim is computed using the binomial formula above.

    Note:
        - K and V must have matching shapes and dtypes (fp16 or bf16)
        - K and V must be contiguous along the last dimension
        - chunk_size must be at least 128
        - The symmetric formulation provides massive memory savings:
          * deg=2: 49% reduction
          * deg=4: 96% reduction
          * deg=6: 99.8% reduction

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    S, s = update_state_fwd(K, V, deg)
    return S, s
# Make it traceable
@torch.library.register_fake("power_attention::update_state")
def update_state_fake(K, V, deg):
    b, n, c, h, d = K.shape
    D = compute_expanded_dim(d, deg)
    S = torch.empty(b, n, h, D, d, device=K.device, dtype=K.dtype)
    s = torch.empty(b, n, h, D, device=K.device, dtype=K.dtype)
    return (S, s)
# Autograd setup
def update_state_setup(ctx, inputs, output):
    K, V, deg = inputs
    ctx.save_for_backward(K, V)
    ctx.deg = deg
def update_state_backward(ctx, dS, ds):
    K, V = ctx.saved_tensors
    dS = dS.contiguous()
    ds = ds.contiguous()
    dK, dV = update_state_bwd(K, V, dS, ds, ctx.deg)
    return dK, dV, None
# Register autograd
torch.library.register_autograd(
    "power_attention::update_state", update_state_backward, setup_context=update_state_setup
)


if __name__ == '__main__':
    from perf._timing import benchmark_speed
    from power_attention._update_state.create_inputs import create_inputs_fwd as create_inputs
    # Hyperparameters
    kw = dict(b=8, n=8, c=256, h=16, d=64, dtype=torch.bfloat16, device='cuda')
    
    print(f"Benchmarking chunk state \n {kw=}")

    # benchmark
    fwd_time = benchmark_speed('fwd', update_state, create_inputs, kw)
    print(f"Fwd time: {fwd_time:.2f} ms")

    bwd_time = benchmark_speed('bwd', update_state, create_inputs, kw)
    print(f"Bwd time: {bwd_time:.2f} ms")

    fwd_bwd_time = benchmark_speed('fwd+bwd', update_state, create_inputs, kw)
    print(f"Fwd+bwd time: {fwd_bwd_time:.2f} ms")

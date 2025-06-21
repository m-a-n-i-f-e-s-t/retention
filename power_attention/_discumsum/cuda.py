## DISCOUNTED CUMSUM CUSTOM OP
# Computes the discounted cumsum forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.utils._pytree import tree_map
from types import NoneType
from typing import Optional, Tuple
from power_attention_cuda import discumsum as discumsum_cuda_fwd, discumsum_bwd as discumsum_bwd_cuda
## FWD ## 
@torch.library.custom_op('power_attention::discumsum_fwd', mutates_args=(), device_types='cuda')
def discumsum_fwd(X : torch.Tensor, log_G : torch.Tensor) -> torch.Tensor:
    """Compute discounted cumulative sum along time axis.

    Computes the discounted cumulative sum of X using gating factors exp(log_G).
    The time axis is the second dimension (axis=1).
    
    Input shapes:
        X: [batch, time, heads, features] - Input tensor to accumulate
        log_G: [batch, time, heads] - Log discount factors, broadcasted along features

    Output shape: 
        [batch, time+1, heads, features] - Accumulated sums
        NOTE: Output has one more timestep than input, with zeros at t=0

    Shape Restrictions:
        - Time dimension must be a multiple of 4
        - Feature dimension must be a multiple of 8
    
    The 0th (batch) and 2nd (heads) dimensions are treated as batch dimensions.
    The discount factors are broadcasted along the final feature dimension.

    TODO(jbuckman): Accept an initial state
    """
    b, n, h, d = X.shape
    cum_X = torch.empty(b, n+1, h, d, device=X.device, dtype=X.dtype)
    cum_X[:,0] = 0.
    discumsum_cuda_fwd(X, log_G, cum_X)
    return cum_X

# Fake implementation for tracing and testing
@discumsum_fwd.register_fake
def discumsum_fwd_fake(X, log_G):
    b, n, h, d = X.shape
    return torch.empty(b, n+1, h, d, device=X.device, dtype=X.dtype)

## BWD ##

@torch.library.custom_op('power_attention::discumsum_bwd', mutates_args=(), device_types='cuda')
def discumsum_bwd(dout : torch.Tensor, out : torch.Tensor, log_G : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for discounted cumulative sum operation.

    Computes gradients with respect to inputs for the discounted cumulative sum operation.
    The time axis is the second dimension (axis=1).
    
    Input shapes:
        log_G: [batch, time, heads] - Log discount factors used in forward pass
        dout: [batch, time+1, heads, features] - Gradient of loss with respect to forward output
        out: [batch, time+1, heads, features] - Forward pass output (needed for gradient computation)

    Output shapes:
        dX: [batch, time, heads, features] - Gradient with respect to input X
        dlog_G: [batch, time, heads] - Gradient with respect to log discount factors

    Shape Restrictions:
        - Time dimension must be a multiple of 4
        - Feature dimension must be a multiple of 8
    
    The 0th (batch) and 2nd (heads) dimensions are treated as batch dimensions.
    The discount factors are broadcasted along the final feature dimension.
    """
    dX, dlog_G = discumsum_bwd_cuda(log_G, dout, out)
    return dX, dlog_G

# Fake implementation for tracing and testing
@discumsum_bwd.register_fake
def discumsum_bwd_fake(dout, out, log_G):
    b, n1, h, d = dout.shape
    n = n1 - 1
    return (torch.empty(b, n, h, d, device=dout.device, dtype=dout.dtype),
            torch.empty(b, n, h, device=dout.device, dtype=torch.float32))

## IMPL ## 

@torch.library.custom_op("power_attention::discumsum", mutates_args=())
def discumsum(X : torch.Tensor, log_G : torch.Tensor) -> torch.Tensor:
    r"""Compute discounted cumulative sum for recurrent processing.

    This function implements the discounted cumulative sum operation from [1]. It is a key
    component in transforming symmetric power attention into a linear-cost RNN, allowing
    efficient processing of long sequences through chunked computation.

    The operation implements the state update equation of the RNN:

    $$Y_t = X_t + \exp(\log G_t) \cdot Y_{t-1}$$

    with initial condition $Y_0 = 0$

    In the context of symmetric power attention:

    - $X_t$ contains expanded state vectors from the current chunk
    - $Y_t$ accumulates state information across chunks
    - $exp(log\ G_t)$ controls how much past information influences the current computation
    - The +1 in the output time dimension allows for proper causality in the RNN

    This formulation enables O(n) complexity instead of O(n²) for long sequences, while
    maintaining the expressivity of power attention through the expanded state representation.

    Args:
        X: Input tensor of shape `(batch_size, time, num_heads, *feature_dims)`.
           The tensor to be accumulated along the time dimension.
        log_G: Log discount factors of shape `(batch_size, time, num_heads)`.
           Natural logarithm of the discount/gating factors.
           These are broadcasted along the feature dimensions.

    Returns:
        Y: Accumulated tensor of shape `(batch_size, time+1, num_heads, *feature_dims)`.
           Note that the output has one more timestep than the input, with zeros at t=0.

    Note:
        - Time dimension must be a multiple of 4
        - Product of feature dimensions must be a multiple of 8
        - The batch and heads dimensions are treated as independent batch dimensions
        - Initial state support is planned but not yet implemented
        - The RNN formulation maintains O(1) memory per layer regardless of sequence length

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    b, n, h, *ds = X.shape
    if len(X.shape) > 4:
        X = X.view(*X.shape[:3], -1)
    cum_X = discumsum_fwd(X, log_G)
    return cum_X.view(b, n+1, h, *ds)

# Make it traceable
@discumsum.register_fake
def discumsum_fake(X, log_G):
    b, n, h, *ds = X.shape
    return torch.empty(b, n+1, h, *ds, device=X.device, dtype=X.dtype)

# Autograd setup
def discumsum_setup(ctx, inputs, output):
    X, log_G = inputs
    ctx.save_for_backward(log_G, output)
    ctx.original_shape = X.shape

def discumsum_backward(ctx, dcum_X):
    log_G, cum_X = ctx.saved_tensors
    b, n, h, *ds = ctx.original_shape
    dcum_X = dcum_X.view(b, n+1, h, -1)
    cum_X = cum_X.view(b, n+1, h, -1)
    dX, dlog_G = discumsum_bwd(dcum_X, cum_X, log_G)
    return dX.view(b, n, h, *ds), dlog_G

# Register autograd
torch.library.register_autograd(
    "power_attention::discumsum", discumsum_backward, setup_context=discumsum_setup
)


## TUTORIAL ##
if __name__ == '__main__':
    from power_attention._discumsum.create_inputs import create_inputs
    # Hyperparameters
    b, n, h, D, d = (2, 4, 8, 64, 16)
    dtype = torch.float16
    # Create inputs
    X, log_G = create_inputs(b, n, h, D, d, dtype, 'cuda', requires_grad=True)
    # Run forward
    cum_X = discumsum(X, log_G)
    # Run backward
    torch.autograd.backward(cum_X, cum_X)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_discumsum = torch.compile(discumsum, fullgraph=True)
    for _ in range(3):
        inputs = create_inputs(b, n, h, D, d, dtype, 'cuda', requires_grad=True)
        outputs = compiled_discumsum(*inputs)
        torch.autograd.backward(outputs, outputs)


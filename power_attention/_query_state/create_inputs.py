
import torch
import torch.nn.functional as F
from power_attention._utils import compute_expanded_dim
from torch.utils._pytree import tree_map
from vidrial.mosaic.utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim


def create_inputs(b=2, n=3, c=128, h=1, d=32, dtype=torch.bfloat16, device='cuda', deg=2, zero_initial_state=False, seed=42, d_tile=None, use_vidrial_layout=False, requires_grad=False, scale=1.):
    torch.manual_seed(seed)
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    Q = torch.randn(b, n, c, h, d, dtype=dtype, device=device, requires_grad=requires_grad)
    if not use_vidrial_layout:
        D = compute_expanded_dim(d, deg=deg)
    else:
        D = sympow_dim(d, deg, d_tile=d_tile)
    S = torch.randn(b, n, h, D, d, dtype=dtype, device=device, requires_grad=requires_grad)
    s = torch.abs(torch.randn(b, n, h, D, dtype=torch.float32, device=device, requires_grad=requires_grad) + 1e2).detach().requires_grad_(requires_grad)
    Y_attn = torch.randn(b, n, c, h, d, dtype=dtype, device=device, requires_grad=requires_grad)
    l_attn = (torch.abs(torch.randn(b, n, c, h, dtype=torch.float32, device=device)) + 1e2).detach().requires_grad_(requires_grad)
    rowmax = (torch.abs(torch.randn(b, n, c, h, dtype=torch.float32, device=device)) * 10.)

    return dict(Q=Q, S=S, s=s, Y_attn=Y_attn, l_attn=l_attn, rowmax=rowmax, deg=deg, scale=scale, zero_initial_state=zero_initial_state)



def input_properties(b=2, n=3, c=128, h=1, d=32, dtype=torch.bfloat16, device='cuda', deg=2, zero_initial_state=False, seed=42, d_tile=None, use_vidrial_layout=False, requires_grad=False, scale=1.):
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    if not use_vidrial_layout:
        D = compute_expanded_dim(d, deg=deg)
    else:
        D = sympow_dim(d, deg, d_tile=d_tile)
    return dict(
        Q=((b, n, c, h, d), dtype, device),
        S=((b, n, h, D, d), dtype, device),
        s=((b, n, h, D), torch.float32, device),
        Y_attn=((b, n, c, h, d), dtype, device),
        l_attn=((b, n, c, h), torch.float32, device),
        rowmax=((b, n, c, h), torch.float32, device),
    )

def output_properties(b=2, n=3, c=128, h=1, d=32, dtype=torch.bfloat16, device='cuda', deg=2, zero_initial_state=False, seed=42, d_tile=None, use_vidrial_layout=False, requires_grad=False, scale=1.):
    return dict(
        O=((b, n, c, h, d), dtype, device),
    )
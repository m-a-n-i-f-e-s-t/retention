import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', scale=1.0, deg=2, r=1, w=1, causal=True, gating=False, head_first=True, requires_grad=False, seed=42, std=1.0, norm=True, use_log2=False):
    torch.manual_seed(seed)
    shape_base = (b, h, t) if head_first else (b, t, h)
    Q = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    log_G = F.logsigmoid(torch.rand(size=shape_base, dtype=torch.float32, device=device)) if gating else None
    if requires_grad:
        Q, K, V, log_G = tree_map(lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, r=r, w=w, causal=causal, head_first=head_first, scale=scale, norm=norm, use_log2=use_log2)

def create_inputs_cuda(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', scale=1.0, deg=2, r=1, w=1, causal=True, gating=False, requires_grad=False, seed=42, std=1.0, norm=True, use_log2=False):
    torch.manual_seed(seed)
    shape_base = (b, t, h)
    Q = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    log_G = F.logsigmoid(torch.rand(size=shape_base, dtype=torch.float32, device=device)) if gating else None
    if requires_grad:
        Q, K, V, log_G = tree_map(lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, r=r, w=w, causal=causal, scale=scale, head_first=False, norm=norm, use_log2=use_log2)

def input_properties(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', gating=False, head_first=True, **kwargs):
    shape_base = (b, h, t) if head_first else (b, t, h)
    return dict(
        Q=((*shape_base, d), dtype, device), 
        K=((*shape_base, d), dtype, device), 
        V=((*shape_base, d), dtype, device), 
    ) | ({'log_G': (shape_base, torch.float32, device)} if gating else {})

def output_properties(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', head_first=True, norm=True, **kwargs):
    shape_base = (b, h, t) if head_first else (b, t, h)
    if norm:
        return dict(
            Y=((*shape_base, d), dtype, device),
        )
    else:
        return dict(
            Y=((*shape_base, d), dtype, device),
            l=(shape_base, dtype, device),
            rowmax=(shape_base, torch.float32, device)
        )
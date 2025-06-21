import torch
from torch.utils._pytree import tree_map

# Useful function to create sample inputs   
def create_inputs(b=2, n=4, h=8, D=64, d=16, dtype=torch.float16, device='cuda', requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(42)
    X = torch.randn(size=(b, n, h, D, d), dtype=dtype, device=device, generator=generator)
    log_G = torch.zeros(size=(b, n, h), dtype=torch.float32, device=device) - 0.01
    if requires_grad:
        X, log_G = tree_map(
            lambda x: x.requires_grad_(True), (X, log_G)
        )
    return dict(
        X=X,
        log_G=log_G
    )

def input_properties(b=2, n=4, h=8, D=64, d=16, dtype=torch.float16, device='cuda', requires_grad=False):
    return dict(
        X=((b, n, h, D, d), dtype, device),
        log_G=((b, n, h), torch.float32, device),
    )

def output_properties(b=2, n=4, h=8, D=64, d=16, dtype=torch.float16, device='cuda', requires_grad=False):
    return dict(
        cum_X=((b, n+1, h, D, d), dtype, device),
    )
import torch

def create_inputs(b, t, h, d, dtype, device, requires_grad=False, scale=1.0, is_causal=True, qhead_ratio=1, dropout_p=0.0, enable_gqa=False):
    q = torch.randn((b, h * qhead_ratio, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)

    return dict(
        query=q,
        key=k,
        value=v,
        scale=scale,
        is_causal=is_causal,
        dropout_p=dropout_p,
        enable_gqa=enable_gqa,
    )

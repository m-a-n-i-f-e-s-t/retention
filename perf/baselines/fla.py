import torch

def create_inputs(b, t, h, d, dtype, device, requires_grad=False, scale=1.0, initial_state=None, output_final_state=False, normalize=False, head_first=False):
    if head_first:
        q = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    else:
        q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)

    return dict(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        normalize=normalize,
        head_first=head_first,
    )

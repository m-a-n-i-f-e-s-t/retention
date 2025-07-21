import torch
import torch.nn.functional as F
from power_attention._utils import dummify
import math
flash_equivalent = False
normal_space = False
normalize_output = False
ε = 1e-6

def attention(Q, K, V, log_G, deg, r=1, w=1, causal=True, head_first=False, scale=1.0, norm=False, use_log2=False):
    if head_first:
        b, hq, ctx, d, hk, e = *Q.shape, K.shape[1], V.shape[-1]
    else:
        b, ctx, hq, d, hk, e = *Q.shape, K.shape[2], V.shape[-1]
    assert hq % r == 0, "hq must be divisible by r"
    assert hk % w == 0, "hk must be divisible by w"
    assert hq // r == hk // w, "hq // r must be equal to hk // w"
    assert isinstance(deg, int) and deg % 2 == 0, "deg must be a positive even integer"
    h = hq // r
    if log_G is not None:
        if head_first:
            assert log_G.shape == (b, h, ctx)
        else:
            assert log_G.shape == (b, ctx, h)
            log_G = log_G.transpose(1, 2) # (b, h, ctx)
    if head_first:
        Q = Q.view(b, h, ctx * r, d)
        K = K.view(b, h, ctx * w, d)
        V = V.view(b, h, ctx * w, e)
    else:
        Q = Q.view(b, ctx * r, h, d).transpose(1, 2)
        K = K.view(b, ctx * w, h, d).transpose(1, 2)
        V = V.view(b, ctx * w, h, e).transpose(1, 2)
    
    exp = torch.exp if not use_log2 else torch.exp2
    log = torch.log if not use_log2 else torch.log2

    _qidx = torch.arange(ctx*r, device=Q.device).unsqueeze(1)
    _kidx = torch.arange(ctx*w, device=K.device).unsqueeze(0)
    s = torch.matmul(Q, K.transpose(2,3)) * scale
    m = (_qidx // r) >= (_kidx // w) if causal else torch.ones_like(s, dtype=torch.bool)
    signs = torch.sign(s)
    s = float(deg) * torch.where(m, log(s.abs() + 1e-7), -float("inf"))
    if log_G is not None:
        s = s + (log_G.repeat_interleave(r, dim=2)[..., :, None] - log_G.repeat_interleave(w, dim=2)[..., None, :])
    rowmax = torch.max(s, dim=-1, keepdim=True).values.detach()
    if deg % 2 == 0:
        p = exp(s - rowmax).to(V.dtype)
    else:
        p = exp(s - rowmax).to(V.dtype) * signs
    l = torch.sum(p, dim=-1)
    o = torch.matmul(p, V)
    if norm:
        o = o / l[..., None]
    if not head_first:
        o = o.transpose(1, 2)
        rowmax = rowmax.transpose(1, 2)
        l = l.transpose(1, 2)
    if norm:
        return o
    else:
        return o, l, rowmax.squeeze(-1)


attention_fwd = attention





## This is useful sometimes but not used at the moment

class FlashAttentionReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale=0.0):
        """Reference implementation of the attention forward pass
        args:
            Q, K, V: [b, t, h, d]
            ε: float
        returns:
            Y: [b, t, h, d]
        """
        device = Q.device
        b, t, h, d = Q.shape

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # [b, h, t, d]
        S = torch.matmul(Q, K.transpose(-1, -2)) * torch.tensor(softmax_scale, dtype=torch.float32, device=device)
        mask = torch.tril(torch.full(S.shape[-2:], True, dtype=torch.bool, device=device))
        S = torch.where(mask, S, -torch.inf)
        R = S.max(dim=-1, keepdim=True)[0]
        P = F.softmax(S - R, dim=-1)
        Y = P.to(Q.dtype) @ V

        ctx.save_for_backward(Q, K, V)
        ctx.softmax_scale = softmax_scale
        return Y.transpose(1, 2).contiguous() # [b, t, h, d]

    @staticmethod
    def backward(ctx, dY):
        """Reference implementation of the attention backward pass
        args:
            Q, K, V: [b, t, h, d]
            dY: [b, t, h, d]
        returns:
            dQ: [b, t, h, d]
            dK: [b, t, h, d]
            dV: [b, t, h, d]
            dS: [b, h, t, t]
        """
        Q, K, V = ctx.saved_tensors
        dY = dY.transpose(1, 2) # [b, h, t, d]

        # recompute forward
        S = torch.matmul(Q, K.transpose(-1, -2)) * torch.tensor(ctx.softmax_scale, dtype=Q.dtype, device=Q.device)
        mask = torch.tril(torch.full(S.shape[-2:], True, dtype=torch.bool, device=Q.device))
        S = torch.where(mask, S, -torch.inf)
        R = S.max(dim=-1, keepdim=True)[0]
        P = torch.exp(S - R)
        y = P.sum(-1, dtype=torch.float32) # [b, h, t]

        # compute backward
        dP_adj = dY.to(Q.dtype) @ V.transpose(-1, -2)
        dy = - (dP_adj * P).sum(-1) / y**2 
        dV = (P / y.unsqueeze(-1)).to(Q.dtype).transpose(-1, -2) @ dY # [b, h, t, d]
        dP = dP_adj / y.unsqueeze(-1) + dy.unsqueeze(-1)
        dS = dP * P * ctx.softmax_scale

        dQ = dS.to(Q.dtype) @ K  # [b, h, t, d]
        dK = dS.transpose(-1, -2).to(Q.dtype) @ Q  # [b, h, t, d]
        dQ, dK, dV = dQ.transpose(1, 2), dK.transpose(1, 2), dV.transpose(1, 2) # [b, t, h, d]
        return dQ, dK, dV, None
        


def flash_attention_reference(*args, **kwargs):
    if args and kwargs:
        raise ValueError("Cannot pass both args and kwargs")
    if kwargs:
        args = (kwargs['Q'], kwargs['K'], kwargs['V'], kwargs['softmax_scale'])
    return FlashAttentionReference.apply(*args)
flash_attention_reference_fwd = dummify(FlashAttentionReference.forward)


def create_inputs_flash(b, t, h, d, dtype, device, softmax_scale=0.0, requires_grad=False, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    K = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    V = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    return Q, K, V, softmax_scale
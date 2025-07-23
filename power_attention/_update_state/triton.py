import torch
import triton
import triton.language as tl
import os
from math import comb
from power_attention.kernelgen import kernelgen
from power_attention._utils import dummify

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BLOCK_E"] <= nargs["e"] and config.kwargs["BLOCK_D"] <= nargs["D"] and config.kwargs["BLOCK_T"] <= nargs["T"]:
            pruned_configs.append(config)
            if os.environ.get("TRITON_NO_AUTOTUNE", "0") == "1":
                return pruned_configs
    return pruned_configs

fwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_E': BE, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for BD in [128, 256]
    for BE in [32, 64]
    for BT in [16, 32]
    for block1 in [16]
    for nw in [4, 8]
    for ns in [1, 3]
]

def keep(config):
    block1 = config.kwargs["block1"]
    block2 = config.kwargs["BLOCK_D"] // block1
    if block1 >= block2 and block1 % block2 == 0:
        return True
    return False

@triton.jit
def get_offsets_p2(off_D, d, block1, block_D):
    """ Return off_d1, off_d2, and the multiplier for the starting offset on dimension 1 and 2, given block offset of the expanded dimension D. 

    Define block1, block2 to be the block size along the first, the second dimension in the hypercube. Define m, n to be the offset in unit of blocks along the first, the second dimension in the hypercube.

    We use the following invariant to find the offset
       
       block2 <= block1
       m*(1+m)*block1/2 <= off_D*block2 <= (m+1)*(m+2)*block1/2
       
       or, let z = = off_D*block2/block1*2
       m*(1+m) <= z <= (m+1)*(m+2)
    """
    tl.static_assert(d % block1 == 0)
    block2: tl.constexpr = block_D // block1
    tl.static_assert(block1 >= block2 and block1 % block2 == 0)
    z = off_D.to(tl.float32)/(block1//block2)*2
    m = (tl.math.floor((tl.math.sqrt(1 + 4*z) - 1) / 2)).to(tl.int32)
    n = off_D - (m*(1+m)*(block1//block2)/2).to(tl.int32)
    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
    return m*block1, n*block2, multiplier

@triton.autotune(list(filter(keep, fwd_configs)), key=["deg", "d", "e", "D"])
@triton.jit
@kernelgen(list(filter(keep, fwd_configs)))
def _update_state_fwd(K, V, S, N, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_vb, stride_vt, stride_vh, stride_ve,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     stride_nb, stride_nh, stride_nD,
                     T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr):
    """ 
    This is a templated kernel, which, when called, will render the embedded
    template into a triton kernel in ./_rendered/_update_state_fwd_dispatcher.py, 
    and call the rendered kernel.

    Note that the rendered kernel is only for inspection purpose, modifying it will
    have no effect on runtime.

    <kernelgen>
block2: tl.constexpr = BLOCK_D // block1
BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_D = tl.program_id(1)
off_e = tl.program_id(2)
off_d1, off_d2, multiplier = get_offsets_p2(off_D, d, block1, BLOCK_D)
off_d1 = tl.multiple_of(off_d1, block1)
off_d2 = tl.multiple_of(off_d2, block2)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
S += off_b.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh + off_D.to(tl.int64) * BLOCK_D * stride_sD
N += off_b.to(tl.int64) * stride_nb + off_h.to(tl.int64) * stride_nh + off_D.to(tl.int64) * BLOCK_D * stride_nD

range_t = tl.arange(0, BLOCK_T).to(tl.int64)
range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
range_e = tl.arange(0, BLOCK_E_VALID).to(tl.int64) + off_e * BLOCK_E_VALID
p_k_d1 = K + range_d1[:, None] * stride_kd + range_t[None, :] * stride_kt # [block1 x BLOCK_T]
p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve # [BLOCK_T x BLOCK_E_VALID]

{% set block2 = BLOCK_D // block1 -%}
{% for i in range(block2) -%}
p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd
s_{{i}} = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
n_{{i}} = tl.zeros((block1,), dtype=tl.float32)
{% endfor -%}

for tid in range(0, tl.cdiv(T, BLOCK_T)):
    mask_t = range_t + tid * BLOCK_T < T
    k_d1 = tl.load(p_k_d1, mask=mask_t[None, :], other=0.0) # block1 x BLOCK_T
    v = tl.load(p_v, mask=mask_t[:, None], other=0.0)
    {% for i in range(block2) -%}
    k_d2_{{i}} = tl.load(p_k_d2_{{i}}, mask=mask_t, other=0.0) * multiplier # BLOCK_T
    phik_{{i}} = k_d1 * k_d2_{{i}}
    n_{{i}} += tl.sum(phik_{{i}}, 1) # block1
    {% endfor -%}
    {% for i in range(block2) -%}
    s_{{i}} = tl.dot(phik_{{i}}.to(K.dtype.element_ty), v, s_{{i}})
    {% endfor -%}
    p_v += BLOCK_T * stride_vt
    p_k_d1 += BLOCK_T * stride_kt
    {% for i in range(block2) -%}
    p_k_d2_{{i}} += BLOCK_T * stride_kt
    {% endfor %}

{% for i in range(block2) -%}
range_d2_{{i}} = tl.arange(0, block1).to(tl.int64) + {{i}} * block1
p_n_{{i}} = N + (range_d2_{{i}} * stride_nD)
tl.store(p_n_{{i}}, n_{{i}})
p_s_{{i}} = S + range_d2_{{i}}[:, None] * stride_sD + range_e[None, :] * stride_se
tl.store(p_s_{{i}}, s_{{i}})
{% endfor -%}
    </kernelgen>
    """
    pass

bwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_T': BT, 'V_IN_REGS': V_IN_REGS}, num_warps=nw, num_stages=ns)
    for BD in [16, 32]
    for BT in [128]
    for block1 in [16]
    for nw in [4]
    for ns in [1, 3]
    for V_IN_REGS in [True, False]
]

@triton.autotune(list(filter(keep, bwd_configs)), key=["deg", "d", "e", "D"])
@triton.jit
@kernelgen(list(filter(keep, bwd_configs)))
def _update_state_bwd(K, V, dS, dN, dK, dV, deg: tl.constexpr,
                      stride_kb, stride_kt, stride_kh, stride_kd,
                      stride_vb, stride_vt, stride_vh, stride_ve,
                      stride_dsb, stride_dsh, stride_dsD, stride_dse,
                      stride_dnb, stride_dnh, stride_dnD,
                      stride_dkb, stride_dkt, stride_dkh, stride_dkd,
                      stride_dvb, stride_dvt, stride_dvh, stride_dve,
                      T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                      block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, V_IN_REGS: tl.constexpr):
    """
    In this case, the kernel template is given a list of possible values that some
    constexpr variables might take. This information is needed to be provided 
    manually because the autotune config doesn't have it.

    <kernelgen d=(32, 64, 128)>
block2: tl.constexpr = BLOCK_D // block1
{% set block1 = block1 -%}
{% set block2 = BLOCK_D // block1 -%}
tl.static_assert(block1 >= block2 and block1 % block2 == 0)
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_t = tl.program_id(1)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
dN += off_b.to(tl.int64) * stride_dnb + off_h.to(tl.int64) * stride_dnh
dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
# dPhiK += off_b.to(tl.int64) * stride_dpb + off_h.to(tl.int64) * stride_dph
range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, e).to(tl.int64)
range_d1 = tl.arange(0, block1)
p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
dv = tl.zeros((BLOCK_T, e), dtype=tl.float32)
{% for j in range(d//block1) -%}
dk_{{j}} = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
{% endfor -%}

mask_T = range_t < T
if V_IN_REGS:
    v = tl.load(p_v, mask=mask_T[:, None], other=0.)

for m in range(0, d//block1):
    p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
    k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)

    for n in range(0, (m+1)*block1//block2):
        off_d2 = n*block2
        multiplier = 1 if (n + 1) * block2 > m * block1 else 2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
        {% for i in range(block2) -%}
        p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd # BLOCK_T
        p_ds_{{i}} = dS + (range_d1[:, None] + off_D + {{i}} * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x e
        p_dN_{{i}} = dN + (range_d1 + off_D + {{i}} * block1) * stride_dnD
        # p_dP_{{i}} = dPhiK + range_t[:, None] * stride_dpt + (range_d1[None, :] + off_D + {{i}} * block1) * stride_dpd
        {% endfor -%}

        {% for i in range(block2) -%}
        k_d2_{{i}} = tl.load(p_k_d2_{{i}}, mask=mask_T, other=0.) # BLOCK_T
        ds_{{i}} = (tl.load(p_ds_{{i}}) * multiplier).to(K.dtype.element_ty) # block1 x e
        {% endfor -%}
        {% for i in range(block2) -%}
        phik_{{i}} = k_d1 * (k_d2_{{i}}[:, None]) # BLOCK_T x block1
        dv = tl.dot(phik_{{i}}.to(K.dtype.element_ty), ds_{{i}}, dv) # BLOCK_T x e
        {% endfor %}
        if not V_IN_REGS:
            v = tl.load(p_v, mask=mask_T[:, None], other=0.)

        {% for i in range(block2) %}
        dN_{{i}} = tl.load(p_dN_{{i}}) * multiplier # block1
        dphik_{{i}} = tl.dot(v, tl.trans(ds_{{i}})).to(tl.float32) + dN_{{i}}[None, :] # BLOCK_T x block1
        # tl.store(p_dP_{{i}}, dphik_{{i}})
        if m == 0:
            dk_0 += dphik_{{i}} * k_d2_{{i}}[:, None] # BLOCK_T x block1
        {% for j in range(1, d//block1 - 1) -%}
        elif m == {{j}}:
            dk_{{j}} += dphik_{{i}} * k_d2_{{i}}[:, None] # BLOCK_T x block1
        {% endfor -%}
        else:
            dk_{{d//block1 - 1}} += dphik_{{i}} * k_d2_{{i}}[:, None] # BLOCK_T x block1
        {% endfor -%}
        
        {% for i in range(block2) -%}
        dk_d2_{{i}} = tl.sum(dphik_{{i}} * k_d1, 1) # BLOCK_T
        if off_d2//block1 == 0:
            mask = ((range_d1 + {{0}} * block1) == (off_d2 + {{i}}))
            dk_{{0}} += tl.where(mask[None, :].broadcast_to(dk_{{0}}.shape), dk_d2_{{i}}[:, None].broadcast_to(dk_{{0}}.shape), 0.)
        {% for j in range(1, d//block1 - 1) -%}
        elif off_d2//block1 == {{j}}:
            mask = ((range_d1 + {{j}} * block1) == (off_d2 + {{i}}))
            dk_{{j}} += tl.where(mask[None, :].broadcast_to(dk_{{j}}.shape), dk_d2_{{i}}[:, None].broadcast_to(dk_{{j}}.shape), 0.)
        {% endfor -%}
        else:
            mask = ((range_d1 + {{d//block1 - 1}} * block1) == (off_d2 + {{i}}))
            dk_{{d//block1 - 1}} += tl.where(mask[None, :].broadcast_to(dk_{{d//block1 - 1}}.shape), dk_d2_{{i}}[:, None].broadcast_to(dk_{{d//block1 - 1}}.shape), 0.)
        {% endfor %}


# save dk, dv
mask_T = range_t < T
{% for j in range(d//block1) -%}
p_dk_{{j}} = dK + range_t[:, None].to(tl.int64) * stride_dkt + ({{j}} * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
tl.store(p_dk_{{j}}, dk_{{j}}, mask=mask_T[:, None])
{% endfor -%}
p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
tl.store(p_dv, dv, mask=mask_T[:, None])
    
    </kernelgen>
    """
    pass
    

def compute_expanded_dim(d, deg, d_block=16):
    """ Compute the expanded state dimension for symmetric power for any given degree.

        Args:
            d: int, feature dimension of input tensor
            deg: int, degree of symmetric power attention
            d_block: int, block size for a single dimension. The smaller d_block is, the less waste there is.

        Returns:
            int, expanded state dimension D
    """
    hyper_cube_dim = d_block ** deg
    num_blocks_per_dim = d // d_block
    D = hyper_cube_dim * comb(num_blocks_per_dim + deg - 1, deg)
    return D


class _update_state(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K, V, deg):
        """ Args: 
            K: (b, n, h, t, d)
            V: (b, n, h, t, e)
            deg: int

            Returns:
            S: (b, n, h, D, e) where D > comb(d + deg - 1, deg) with padding
            N: (b, n, h, D) sum of keys
        """
        assert K.shape == V.shape
        assert K.shape[1] == V.shape[1]
        assert K.shape[2] == V.shape[2]
        assert K.shape[3] == V.shape[3]
        assert K.shape[4] == V.shape[4]

        b, n, t, h, d, e = *K.shape, V.shape[-1]
        K = K.view(b*n, t, h, d)
        V = V.view(b*n, t, h, e)

        D = compute_expanded_dim(d, deg, 16)

        S = torch.empty((b*n, h, D, e), device=K.device, dtype=K.dtype)
        N = torch.empty((b*n, h, D), device=K.device, dtype=torch.float32)

        grid = lambda args: (b*n*h, triton.cdiv(D, args["BLOCK_D"]), triton.cdiv(e, args["BLOCK_E"]))

        if deg != 2:
            raise NotImplementedError("Only deg = 2 is supported for now")

        _update_state_fwd[grid](
            K, V, S, N, deg,
            *K.stride(),
            *V.stride(),
            *S.stride(),
            *N.stride(),
            t, h, d, e, D)

        ctx.save_for_backward(K, V)
        ctx.deg = deg
        ctx.d = d
        ctx.e = e
        ctx.D = D

        return S.view(b, n, h, D, e), N.view(b, n, h, D)

    @staticmethod
    def backward(ctx, dS, ds):
        k, v = ctx.saved_tensors
        deg, d = ctx.deg, ctx.d

        b, n, h, D, e, t = *dS.shape, k.shape[1]
        assert D == ctx.D

        dS = dS.view(b*n, h, D, e)
        ds = ds.view(b*n, h, D)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid = lambda args: (b*n*h, triton.cdiv(t, args["BLOCK_T"]))
        _update_state_bwd[grid](
            k, v, dS, ds, dk, dv, deg,
            *k.stride(),
            *v.stride(),
            *dS.stride(),
            *ds.stride(),
            *dk.stride(),
            *dv.stride(),
            t, h, d, e, D)
        
        dk = dk.view(b, n, t, h, d)
        dv = dv.view(b, n, t, h, e)
        return dk, dv, None


def _update_state_fn(K, V, deg):
    return _update_state.apply(K, V, deg)

update_state = torch.compiler.disable(_update_state_fn)


if __name__ == '__main__':
    from perf._timing import benchmark_speed
    from power_attention._update_state.create_inputs import create_inputs
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





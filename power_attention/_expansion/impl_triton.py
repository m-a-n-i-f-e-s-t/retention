import torch
import triton
import triton.language as tl
from math import comb
from power_attention.kernelgen import kernelgen
from einops import rearrange

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BLOCK_D"] <= nargs["D"] and config.kwargs["BLOCK_T"] <= nargs["T"]:
            pruned_configs.append(config)
    return pruned_configs

fwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for BD in [128, 256]
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

@triton.autotune(list(filter(keep, fwd_configs)), key=["deg", "d", "D"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(list(filter(keep, fwd_configs)))
def _expand_kernel_split_D(K, phi_K, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_pb, stride_pt, stride_ph, stride_pD,
                     T, H, d: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    """ 
    This is a templated kernel, which, when called, will render the embedded
    template into a triton kernel in ./_rendered/_update_state_fwd_dispatcher.py, 
    and call the rendered kernel.

    Note that the rendered kernel is only for inspection purpose, modifying it will
    have no effect on runtime.

    <kernelgen>
block2: tl.constexpr = BLOCK_D // block1
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_D = tl.program_id(1)
off_d1, off_d2, multiplier = get_offsets_p2(off_D, d, block1, BLOCK_D)
off_d1 = tl.multiple_of(off_d1, block1)
off_d2 = tl.multiple_of(off_d2, block2)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
phi_K += off_b.to(tl.int64) * stride_pb + off_h.to(tl.int64) * stride_ph

range_t = tl.arange(0, BLOCK_T).to(tl.int64)
range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
p_k_d1 = K + range_t[None, :] * stride_kt + range_d1[:, None] * stride_kd # [block1 x BLOCK_T]
{% set block2 = BLOCK_D // block1 -%}
{% for i in range(block2) -%}
p_phi_K_{{i}} = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + {{i}} * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd # [BLOCK_T]
{% endfor -%}

for tid in range(0, tl.cdiv(T, BLOCK_T)):
    k_d1 = tl.load(p_k_d1) # [BLOCK_T x block1]
    {% for i in range(block2) -%}
    k_d2_{{i}} = tl.load(p_k_d2_{{i}}) * multiplier # [BLOCK_T]
    phik_{{i}} = k_d1 * k_d2_{{i}}[None, :] # [block1 x BLOCK_T]
    {% endfor -%}
    {% for i in range(block2) -%}
    tl.store(p_phi_K_{{i}}, phik_{{i}})
    {% endfor -%}
    p_k_d1 += BLOCK_T * stride_kt
    {% for i in range(block2) -%}
    p_k_d2_{{i}} += BLOCK_T * stride_kt
    p_phi_K_{{i}} += BLOCK_T * stride_pt
    {% endfor %}

</kernelgen>
    """
    pass


@triton.autotune(list(filter(keep, fwd_configs)), key=["deg", "d", "D"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(list(filter(keep, fwd_configs)))
def _expand_kernel_split_T(K, phi_K, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_pb, stride_pt, stride_ph, stride_pD,
                     T, H, d: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    """ 
    This is a templated kernel, which, when called, will render the embedded
    template into a triton kernel in ./_rendered/_update_state_fwd_dispatcher.py, 
    and call the rendered kernel.

    Note that the rendered kernel is only for inspection purpose, modifying it will
    have no effect on runtime.

    <kernelgen>
block2: tl.constexpr = BLOCK_D // block1
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_t = tl.program_id(1)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
phi_K += off_b.to(tl.int64) * stride_pb + off_h.to(tl.int64) * stride_ph

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_d1 = tl.arange(0, block1).to(tl.int64)

{% set block2 = BLOCK_D // block1 -%}

for m in range(0, d//block1):
    p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
    k_d1 = tl.load(p_k_d1) # BLOCK_T x block1
    
    for n in range(0, (m+1)*block1//block2):
        off_d2 = n*block2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
        multiplier = 1 if (n + 1) * block2 > m * block1 else 2
        {% for i in range(block2) -%}
        p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd # [BLOCK_T]
        p_phi_K_{{i}} = phi_K + range_t[:, None] * stride_pt + (off_D + {{i}} * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_pD # [BLOCK_T x block1]
        {% endfor -%}

        {% for i in range(block2) -%}
        k_d2_{{i}} = tl.load(p_k_d2_{{i}}) * multiplier # [BLOCK_T]
        {% endfor -%}
        {% for i in range(block2) -%}
        phik_{{i}} = k_d1 * k_d2_{{i}}[:, None] # [BLOCK_T x block1]
        {% endfor -%}
        {% for i in range(block2) -%}
        tl.store(p_phi_K_{{i}}, phik_{{i}})
        {% endfor -%}

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


class _expand(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k, deg, split):
        """ Args: 
            K: (B, N, T, H, d)
            deg: int
            split: str, 'D' or 'T'

            Returns:
            phi_K: (B, N, T, H, D) where D > comb(d + deg - 1, deg) with padding
        """
        b, n, t, h, d = k.shape
        k = k.view(b*n, t, h, d)
        stride_kb, stride_kt, stride_kh, stride_kd = k.stride()

        D = compute_expanded_dim(d, deg, 16)

        phi_K = torch.empty((b*n, t, h, D), device=k.device, dtype=k.dtype)
        stride_pb, stride_pt, stride_ph, stride_pD = phi_K.stride()

        if deg != 2:
            raise NotImplementedError("Only deg = 2 is supported for now")

        if split == 'D':
            grid = lambda args: (b*n*h, triton.cdiv(D, args["BLOCK_D"]), 1)
            _expand_kernel_split_D[grid](
                k, phi_K, deg,
                stride_kb, stride_kt, stride_kh, stride_kd,
                stride_pb, stride_pt, stride_ph, stride_pD,
                t, h, d, D)
        elif split == 'T':
            grid = lambda args: (b*n*h, triton.cdiv(t, args["BLOCK_T"]), 1)
            _expand_kernel_split_T[grid](
                k, phi_K, deg,
                stride_kb, stride_kt, stride_kh, stride_kd,
                stride_pb, stride_pt, stride_ph, stride_pD,
                t, h, d, D)

        ctx.save_for_backward(k, phi_K)
        ctx.deg = deg
        ctx.d = d
        ctx.D = D

        return phi_K.view(b, n, t, h, D)


def expand(K, deg, split):
    return _expand.apply(K, deg, split)


def expand_reference(K, deg, split):
    """ Reference implementation of key expansion
    args:
        K: [b, n, c, h, d]
        deg: int
    returns:
        phi_K: [b, n, c, h, D]
    """
    OuterBlock_DT = 1
    InnerBlock_DT = 16
    K = K.permute(0, 1, 3, 2, 4)
    b, n, h, c, d = K.shape
    K_outer = rearrange(K, 'b n h c (x o) -> b n h c x o', o=OuterBlock_DT)
    K_inner = rearrange(K, 'b n h c (y i) -> b n h c y i', i=InnerBlock_DT)
    phi_K_unmasked = torch.einsum('bnhcxo,bnhcyi->bnhcxyoi', K_outer, K_inner).to(K.dtype)
    _, _, _, _, x, y, o, i = phi_K_unmasked.shape
    phi_K_shape = (b, n, h, c, int((InnerBlock_DT // OuterBlock_DT + x) * y // 2), o, i)
    phi_K = torch.empty(phi_K_shape, device=K.device, dtype=K.dtype)
    idx = 0
    for y_idx in range(y):
        for x_idx in range(x):
            if (x_idx * OuterBlock_DT) < (y_idx + 1) * InnerBlock_DT:
                multiplier = 1 if (x_idx + 1) * OuterBlock_DT > y_idx * InnerBlock_DT else 2
                phi_K[:, :, :, :, idx, :, :] = multiplier * phi_K_unmasked[:, :, :, :, x_idx, y_idx, :, :]
                idx += 1
    phi_K = rearrange(phi_K, 'b n h c k o i -> b n h c (k o i)') # [b, n, h, c, D]
    phi_K = phi_K.permute(0, 1, 3, 2, 4)
    return phi_K.to(K.dtype)


def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42, requires_grad=False, split='D'):
    torch.manual_seed(seed)
    K = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    if requires_grad:
        K = K.requires_grad_(True)
    return dict(K=K, deg=2, split=split)


if __name__ == "__main__":
    from perf._timing import benchmark_speed

    # Hyperparameters
    kw = dict(b=1, n=1, c=128, h=12, d=64, dtype=torch.bfloat16, device='cuda', seed=42, split='T')

    # Check correctness
    inputs_triton = create_inputs(**(kw | dict(requires_grad=True)))
    inputs_ref = create_inputs(**(kw | dict(requires_grad=True)))
    phi_k_triton = expand(inputs_triton['K'], inputs_triton['deg'], inputs_triton['split'])
    phi_k_ref = expand_reference(inputs_ref['K'], inputs_ref['deg'], inputs_ref['split'])
    torch.testing.assert_close(phi_k_triton, phi_k_ref, atol=1e-4, rtol=1e-2)
    print("Fwd correctness check passed")

    # Thorough benchmarking
    def print_rowstr(rowstr):
        print(" | ".join([f"{r.upper():<10}" for r in rowstr.split(",")]))

    ctx = 16384
    for mode in ['fwd']:
        print(f"triton-vs-cutlass-batch{kw['b']}-ctx{ctx}-head{kw['h']}-dim{kw['d']}-{mode}")
        print_rowstr("chunk_size,triton,ref,triton speedup")
        for chunk_size in [2**i for i in range(6, 14)]:
            kw['c'] = chunk_size
            kw['n'] = ctx // chunk_size
            triton_time = benchmark_speed(mode, expand, create_inputs, kw, compile=False)
            ref_time = benchmark_speed(mode, expand_reference, create_inputs, kw, compile=False)
            speedup = ref_time / triton_time
            print_rowstr(f"{chunk_size}, {triton_time:.2f}, {ref_time:.2f}, {speedup:.2f}")





import torch
import triton
import triton.language as tl
from math import comb
from power_attention.kernelgen import kernelgen
from _rendered._update_state_fwd_dispatcher import _update_state_fwd as _update_state_fwd_dispatcher

fwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_E': BE, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for BD in [128, 256]
    for BE in [16, 32, 64]
    for BT in [16, 32]
    for block1 in [16]
    for nw in [4, 8, 16]
    for ns in [1, 3, 5, 7]
]

def get_offsets_p2(off_D, d, block_d, block_D):
    """ Return off_d1, off_d2, and the multiplier for the starting offset on dimension 1 and 2, given block offset of the expanded dimension D. 

    Define block1, block2 to be the block size along the first, the second dimension in the hypercube. Define m, n to be the offset in unit of blocks along the first, the second dimension in the hypercube.

    We use the following invariant to find the offset
       
       block2 <= block1
       m*(1+m)*block1/2 <= off_D*block2 <= (m+1)*(m+2)*block1/2
       
       or, let z = = off_D*block2/block1*2
       m*(1+m) <= z <= (m+1)*(m+2)
    """
    tl.static_assert(d % block_d == 0)
    block1: tl.constexpr = block_d
    block2: tl.constexpr = block_D // block1
    tl.static_assert(block1 >= block2 and block1 % block2 == 0)
    z = off_D.to(tl.float32)/(block1//block2)*2
    m = (tl.math.floor((tl.math.sqrt(1 + 4*z) - 1) / 2)).to(tl.int32)
    n = off_D - (m*(1+m)*(block1//block2)/2).to(tl.int32)
    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
    return m*block1, n*block2, multiplier


@triton.autotune(fwd_configs, key=["deg", "d"])
@triton.jit
@kernelgen(fwd_configs)
def _update_state_fwd(K, V, S, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_vb, stride_vt, stride_vh, stride_ve,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr):
    """ 
    This is a templated kernel, which, when called with env var KERNELGEN=1, will render the embedded
    template into a triton kernel in ./_rendered/_update_state_fwd_dispatcher.py. When KERNELGEN is set
    to any other value, the kernelgen decorator is a no-op.

    Due to how triton.jit works, user who wants to update the kernel needs to run the kernelgen 
    decorator with KERNELGEN=1 (which will produce error like "OSError: could not get source code"), 
    before running the kernel.

    For example, to render this specific kernel, run:
    KERNELGEN=1 python impl_triton.py

    Subsequently, run the kernel directly:
    python impl_triton.py

    <kernelgen>
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_D = tl.program_id(1)
off_e = tl.program_id(2)
off_d1, off_d2, multiplier = get_offsets_p2(off_D, d, block1, BLOCK_D)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
S += off_b.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh + off_D.to(tl.int64) * BLOCK_D * stride_sD

range_t = tl.arange(0, BLOCK_T).to(tl.int64)
range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
p_k_d1 = K + range_d1[:, None] * stride_kd + range_t[None, :] * stride_kt
p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
block2: tl.constexpr = BLOCK_D // block1
{% set block2 = BLOCK_D // block1 %}
{% for i in range(block2) %}
p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd
s_{{i}} = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
{% endfor %}

for tid in range(0, tl.cdiv(T, BLOCK_T)):
    k_d1 = tl.load(p_k_d1) # block1 x BLOCK_T
    v = tl.load(p_v)
    {% for i in range(block2) %}
    k_d2_{{i}} = tl.load(p_k_d2_{{i}}) * multiplier # BLOCK_T
    k_{{i}} = k_d1 * k_d2_{{i}}
    {% endfor %}
    {% for i in range(block2) %}
    s_{{i}} = tl.dot(k_{{i}}.to(K.dtype.element_ty), v, s_{{i}})
    {% endfor %}
    p_v += BLOCK_T * stride_vt
    p_k_d1 += BLOCK_T * stride_kt
    {% for i in range(block2) %}
    p_k_d2_{{i}} += BLOCK_T * stride_kt
    {% endfor %}

{% for i in range(block2) %}
range_d2_{{i}} = tl.arange(0, block1).to(tl.int64) + {{i}} * block1
p_s_{{i}} = S + range_d2_{{i}}[:, None] * stride_sD + range_e[None, :] * stride_se
tl.store(p_s_{{i}}, s_{{i}})
{% endfor %}
    </kernelgen>
    """
    return _update_state_fwd_dispatcher(K, V, S, deg, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_vb, stride_vt, stride_vh, stride_ve,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     T, H, d, e, D,
                     block1, BLOCK_D, BLOCK_E, BLOCK_T)
    
    

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
    def forward(ctx, k, v, deg):
        """ Args: 
            K: (B, N, T, H, d)
            V: (B, N, T, H, e)
            deg: int

            Returns:
            S: (B, H, D, e) where D > comb(d + deg - 1, deg) with padding
        """
        assert k.shape == v.shape
        assert k.shape[1] == v.shape[1]
        assert k.shape[2] == v.shape[2]
        assert k.shape[3] == v.shape[3]
        assert k.shape[4] == v.shape[4]

        b, n, t, h, d, e = *k.shape, v.shape[-1]
        k = k.view(b*n, t, h, d)
        v = v.view(b*n, t, h, e)
        stride_kb, stride_kt, stride_kh, stride_kd = k.stride()
        stride_vb, stride_vt, stride_vh, stride_ve = v.stride()

        D = compute_expanded_dim(d, deg, 16)

        s = torch.empty((b*n, h, D, e), device=k.device, dtype=k.dtype)
        stride_sb, stride_sh, stride_sD, stride_se = s.stride()
        grid = lambda args: (b*n*h, triton.cdiv(D, args["BLOCK_D"]), triton.cdiv(e, args["BLOCK_E"]))

        if deg != 2:
            raise NotImplementedError("Only deg = 2 is supported for now")

        _update_state_fwd[grid](
            k, v, s, deg,
            stride_kb, stride_kt, stride_kh, stride_kd,
            stride_vb, stride_vt, stride_vh, stride_ve,
            stride_sb, stride_sh, stride_sD, stride_se,
            t, h, d=d, e=e, D=D)

        return s.view(b, n, h, D, e)


def update_state(K, V, deg):
    return _update_state.apply(K, V, deg)


if __name__ == "__main__":
    from power_attention._update_state.impl import create_inputs, update_state as update_state_impl
    from perf._timing import benchmark_speed


    # Hyperparameters
    kw = dict(b=8, n=8, c=128, h=16, d=64, dtype=torch.bfloat16, device='cuda')

    # Check correctness
    inputs = create_inputs(**kw)
    s_triton = update_state(inputs['K'], inputs['V'], inputs['deg'])
    s_impl = update_state_impl(inputs['K'], inputs['V'], inputs['deg'])
    torch.testing.assert_close(s_triton, s_impl)
    print("Correctness check passed")

    print(f"Benchmarking triton implemented chunk state \n {kw=}")
    fwd_time = benchmark_speed('fwd', update_state, create_inputs, kw, compile=False)
    print(f"Fwd time: {fwd_time:.2f} ms")

    
import torch
import triton
import triton.language as tl
import os
from math import comb, log, sqrt
from power_attention.kernelgen import kernelgen
from power_attention._utils import compute_expanded_dim, diff

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs.get("BLOCK_E", 0) <= nargs["e"] and config.kwargs["BLOCK_D"] <= nargs["D"] and config.kwargs["BLOCK_T"] <= nargs["c"]:
            pruned_configs.append(config)
            if os.environ.get("TRITON_NO_AUTOTUNE", "0") == "1":
                return pruned_configs
    return pruned_configs

fwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_E': BE, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for block1 in [16]
    for BD in [16, 32]
    for BE in [32, 64]
    for BT in [128, 256]
    for nw in [4, 8]
    for ns in [1, 3]
]

@triton.jit
def get_offsets_p2(off_D, d, block1, block_D):
    """ Return off_d1, off_d2 for the starting offset on dimension 1 and 2, given block offset of the expanded dimension D. 

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
    return m*block1, n*block2


@triton.autotune(fwd_configs, key=["deg", "d", "e"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(fwd_configs)
def _query_state_fwd(Q, S, SK, Y_attn, L_attn, M, O, L,
                     stride_qb, stride_qt, stride_qh, stride_qd,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     stride_skb, stride_skh, stride_skD,
                     stride_yb, stride_yt, stride_yh, stride_ye,
                     stride_lab, stride_lat, stride_lah,
                     stride_mb, stride_mt, stride_mh,
                     stride_ob, stride_ot, stride_oh, stride_oe,
                     stride_lb, stride_lt, stride_lh,
                     n, h, c, d: tl.constexpr, D, e: tl.constexpr,
                     zero_initial_state: tl.constexpr,
                     deg: tl.constexpr,
                     scale_p,
                     block1: tl.constexpr,
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    """
Compute query state output. It computes the following equation:

        ( (Q @ (S/γ)) * (scale^p) * γ/α + Y_attn * exp(rowmax)/α )
    O = -------------------------------------------------------------
        ( (Q @ (s/γ)) * (scale^p) * γ/α + l_attn * exp(rowmax)/α )
    where α = max(γ, exp(rowmax))
            γ = 1.0 if S == 0 else sqrt(D)


    <kernelgen>
block2: tl.constexpr = BLOCK_D // block1
BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
{% set block2 = BLOCK_D // block1 -%}
tl.static_assert(block1 >= block2 and block1 % block2 == 0)
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_n = off_bn % n
off_t = tl.program_id(1)
off_e = tl.program_id(2)

Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
SK += off_bn.to(tl.int64) * stride_skb + off_h.to(tl.int64) * stride_skh
O += off_bn.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
Y_attn += off_bn.to(tl.int64) * stride_yb + off_h.to(tl.int64) * stride_yh
L_attn += off_bn.to(tl.int64) * stride_lab + off_h.to(tl.int64) * stride_lah
M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, BLOCK_E_VALID).to(tl.int64) + off_e * BLOCK_E_VALID
range_d1 = tl.arange(0, block1).to(tl.int64)

y = tl.zeros((BLOCK_T, BLOCK_E_VALID), dtype=tl.float32)
l = tl.zeros((BLOCK_T,), dtype=tl.float32)
if off_n == 0 and zero_initial_state:
    gamma = 1.0
    gamma = gamma.to(tl.float32)
else:
    gamma = tl.sqrt(D).to(tl.float32)

mask_T = range_t < c

for m_loop in range(0, d//block1):
    p_q_d1 = Q + range_t[:, None] * stride_qt + (m_loop*block1 + range_d1[None, :]) * stride_qd # BLOCK_T x block1
    q_d1 = tl.load(p_q_d1, mask=mask_T[:, None], other=0.) # BLOCK_T x block1

    for n_loop in range(0, (m_loop+1)*block1//block2):
        off_d2 = n_loop*block2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m_loop*(1+m_loop)//2)*block1*block1 + off_d2*block1
        {% for i in range(block2) -%}
        p_q_d2_{{i}} = Q + range_t[:] * stride_qt + (off_d2 + {{i}}) * stride_qd # BLOCK_T
        p_s_{{i}} = S + (range_d1[:, None] + off_D + {{i}} * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E_VALID
        p_sk_{{i}} = SK + (range_d1 + off_D + {{i}} * block1) * stride_skD
        {% endfor -%}


        {% for i in range(block2) -%}
        q_d2_{{i}} = tl.load(p_q_d2_{{i}}, mask=mask_T, other=0.) # BLOCK_T
        {% endfor -%}
        {% for i in range(block2) -%}
        phik_{{i}} = (q_d1 * (q_d2_{{i}}[:, None])).to(Q.dtype.element_ty) # BLOCK_T x block1
        s_{{i}} = tl.load(p_s_{{i}}) # block1 x BLOCK_E_VALID
        sk_{{i}} = tl.load(p_sk_{{i}}) # BLOCK_D
        s_{{i}} = (s_{{i}} / gamma).to(Q.dtype.element_ty) # block1 x BLOCK_E_VALID
        sk_{{i}} = (sk_{{i}} / gamma).to(Q.dtype.element_ty) # BLOCK_D
        y = tl.dot(phik_{{i}}, s_{{i}}, y) # BLOCK_T x BLOCK_E_VALID
        l += tl.sum(phik_{{i}} * sk_{{i}}[None, :], 1) # BLOCK_T
        {% endfor %}

p_y_attn = Y_attn + range_t[:, None] * stride_yt + range_e[None, :] * stride_ye
p_m = M + range_t * stride_mt
p_l_attn = L_attn + range_t * stride_lat
rowmax = tl.load(p_m, mask=mask_T, other=float(0.0)) # BLOCK_T
y_attn = tl.load(p_y_attn, mask=mask_T[:, None], other=0.).to(tl.float32) # BLOCK_T x BLOCK_E_VALID
l_attn = tl.load(p_l_attn, mask=mask_T, other=0.).to(tl.float32) # BLOCK_T
m = tl.exp(rowmax)
alpha = tl.maximum(gamma, m) # BLOCK_T
qs_factor = gamma / alpha # BLOCK_T
attn_factor = m / alpha # BLOCK_T

o = y_attn * attn_factor[:, None] + y * scale_p * qs_factor[:, None] # BLOCK_T x BLOCK_E_VALID
l = l_attn * attn_factor + l * scale_p * qs_factor # BLOCK_T
o = o / l[:, None] # BLOCK_T x BLOCK_E_VALID

# store y back to O
p_o = O + range_t[:, None] * stride_ot + range_e[None, :] * stride_oe
p_l = L + range_t * stride_lt
tl.store(p_o, o.to(O.dtype.element_ty), mask=mask_T[:, None])
tl.store(p_l, l, mask=mask_T)
    </kernelgen>
    """
    pass


dQ_bwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_T': BT, 'BLOCK_D': BD}, num_warps=nw, num_stages=ns)
    for block1 in [16]
    for BT in [128, 256]
    for BD in [16, 32]
    for nw in [4, 8]
    for ns in [1, 3]
]

@triton.autotune(dQ_bwd_configs, key=["deg", "d", "e"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(dQ_bwd_configs)
def _query_state_bwd_dQ(Q, S, SK, dO, Delta, M, L, dQ, dY_attn, dL_attn,
                        stride_qb, stride_qt, stride_qh, stride_qd,
                        stride_sb, stride_sh, stride_sD, stride_se,
                        stride_skb, stride_skh, stride_skD,
                        stride_dob, stride_dot, stride_doh, stride_doe,
                        stride_db, stride_dt, stride_dh,
                        stride_mb, stride_mt, stride_mh,
                        stride_lb, stride_lt, stride_lh,
                        stride_dqb, stride_dqt, stride_dqh, stride_dqd,
                        stride_dyb, stride_dyt, stride_dyh, stride_dye,
                        stride_dlb, stride_dlt, stride_dlh,
                        n, h, c, d: tl.constexpr, D, e: tl.constexpr,
                        zero_initial_state: tl.constexpr,
                        deg: tl.constexpr,
                        scale_p,
                        block1: tl.constexpr, BLOCK_T: tl.constexpr, 
                        BLOCK_D: tl.constexpr):
    """
    This kernel will compute dQ and dY_attn and dl_attn.
        dl_attn = -exp(rowmax) /alpha / l * delta
        dY_attn = exp(rowmax) / alpha / l * dO
        dphi_Q = (scale^p) / alpha / l * (dO @ S^T - delta @ s^T)
        where alpha = max(gamma, exp(rowmax))
              gamma = 1.0 if S == 0 else sqrt(D)

    <kernelgen d=(32, 64, 128)>
block2: tl.constexpr = BLOCK_D // block1
{% set block2 = BLOCK_D // block1 -%}
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_t = tl.program_id(1)

Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
SK += off_bn.to(tl.int64) * stride_skb + off_h.to(tl.int64) * stride_skh
dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
Delta += off_bn.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
dL_attn += off_bn.to(tl.int64) * stride_dlb + off_h.to(tl.int64) * stride_dlh

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, e).to(tl.int64)
range_d1 = tl.arange(0, block1)
p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x e]
p_m = M + range_t * stride_mt
p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x e]
p_l = L + range_t * stride_lt
p_dl_attn = dL_attn + range_t * stride_dlt

# --- compute dl_attn ---
rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
l = tl.load(p_l, mask=range_t < c, other=float('inf'))
p_delta = Delta + range_t * stride_dt
delta = tl.load(p_delta, mask=range_t < c, other=0.)

chunk_id = off_bn % n
if (chunk_id == 0 and zero_initial_state):
    gamma = 1.0
    gamma = gamma.to(tl.float32)
else:
    gamma = tl.sqrt(D).to(tl.float32)
m = tl.exp(rowmax)
alpha = tl.maximum(gamma, m) # BLOCK_T
qs_factor = scale_p / alpha / l # BLOCK_T
attn_factor = m / alpha / l # BLOCK_T
dl_attn = -attn_factor * delta # BLOCK_T
tl.store(p_dl_attn, dl_attn, mask=range_t < c)

# --- compute dY_attn ---
do = tl.load(p_do, mask=(range_t < c)[:, None], other=0.0) # [BLOCK_T x e]
dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty) # BLOCK_T x e
tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])

# --- compute dQ ---
do = (do * qs_factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
delta = delta * qs_factor # BLOCK_T

{% for j in range(d//block1) -%}
dq_{{j}} = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
{% endfor -%}

for m_loop in range(0, d//block1):
    p_q_d1 = Q + range_t[:, None] * stride_qt + (m_loop*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
    q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
    dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)

    for n_loop in range(0, (m_loop+1)*block1//block2):
        off_d2 = n_loop*block2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m_loop*(1+m_loop)//2)*block1*block1 + off_d2*block1
        {% for i in range(block2) %}
        p_q_d2_{{i}} = Q + range_t[:] * stride_qt + (off_d2 + {{i}}) * stride_qd # [BLOCK_T]
        p_sT_{{i}} = S + (range_d1[None, :] + off_D + {{i}} * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E_VALID x block1]
        p_sk_{{i}} = SK + (range_d1 + off_D + {{i}} * block1) * stride_skD # [block1]
        {% endfor -%}

        {% for i in range(block2) -%}
        q_d2_{{i}} = tl.load(p_q_d2_{{i}}, mask=(range_t < c), other=0.) # [BLOCK_T]
        sT_{{i}} = tl.load(p_sT_{{i}}) # [BLOCK_E_VALID x block1]
        sk_{{i}} = tl.load(p_sk_{{i}}) # [block1]
        {% endfor -%}

        {% for i in range(block2) %}
        dpq_{{i}} = tl.dot(do, sT_{{i}}) - delta[:, None] * sk_{{i}}[None, :] # [BLOCK_T x block1]
        if m_loop == 0:
            dq_0 += dpq_{{i}} * q_d2_{{i}}[:, None]
        {% for j in range(1, d//block1 - 1) -%}
        elif m_loop == {{j}}:
            dq_{{j}} += dpq_{{i}} * q_d2_{{i}}[:, None]
        {% endfor -%}
        else:
            dq_{{d//block1 - 1}} += dpq_{{i}} * q_d2_{{i}}[:, None]
        {% endfor -%}

        {% for i in range(block2) %}
        dq_d2_{{i}} = tl.sum(dpq_{{i}} * q_d1, 1) # [BLOCK_T]
        if off_d2//block1 == 0:
            mask = (tl.arange(0, block1) + {{0}} * block1) == (off_d2 + {{i}})
            dq_{{0}} += tl.where(mask[None, :].broadcast_to(dq_{{0}}.shape), dq_d2_{{i}}[:, None].broadcast_to(dq_{{0}}.shape), 0.)
        {% for j in range(1, d//block1 - 1) -%}
        elif off_d2//block1 == {{j}}:
            mask = (tl.arange(0, block1) + {{j}} * block1) == (off_d2 + {{i}})
            dq_{{j}} += tl.where(mask[None, :].broadcast_to(dq_{{j}}.shape), dq_d2_{{i}}[:, None].broadcast_to(dq_{{j}}.shape), 0.)
        {% endfor -%}
        else:
            mask = (tl.arange(0, block1) + {{d//block1 - 1}} * block1) == (off_d2 + {{i}})
            dq_{{d//block1 - 1}} += tl.where(mask[None, :].broadcast_to(dq_{{d//block1 - 1}}.shape), dq_d2_{{i}}[:, None].broadcast_to(dq_{{d//block1 - 1}}.shape), 0.)
        {% endfor -%}

        
# save dq
{% for j in range(d//block1) -%}
p_dq_{{j}} = dQ + range_t[:, None] * stride_dqt + ({{j}} * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
tl.store(p_dq_{{j}}, dq_{{j}}, mask=(range_t < c)[:, None])
{% endfor -%}
    </kernelgen>
    """
    pass

dS_bwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_T': BT, 'BLOCK_D': BD, 'BLOCK_E': BE}, num_warps=nw, num_stages=ns)
    for block1 in [16]
    for BT in [16, 32]
    for BD in [128, 256]
    for BE in [32, 64]
    for nw in [4, 8]
    for ns in [1, 3]
]

@triton.autotune(dS_bwd_configs, key=["deg", "d", "e"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(dS_bwd_configs)
def _query_state_bwd_dS(Q, L, M, dO, Delta, dS, dSK,
                        stride_qb, stride_qt, stride_qh, stride_qd,
                        stride_lb, stride_lt, stride_lh,
                        stride_mb, stride_mt, stride_mh,
                        stride_dob, stride_dot, stride_doh, stride_doe,
                        stride_db, stride_dt, stride_dh,
                        stride_dsb, stride_dsh, stride_dsD, stride_dse,
                        stride_dskb, stride_dskh, stride_dskD,
                        n, h, c, d: tl.constexpr, D, e: tl.constexpr,
                        zero_initial_state: tl.constexpr,
                        deg: tl.constexpr,
                        scale_p,
                        block1: tl.constexpr, BLOCK_T: tl.constexpr, 
                        BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr):
    """
Computes the following equation for the backward pass:

    delta = (dO * O) @ 1
    dS = phi_Q^T @ (dO / alpha / l)
    ds = -phi_Q^T @ (delta / alpha / l)
    where alpha = max(gamma, exp(rowmax))
          gamma = 1.0 if S == 0 else sqrt(D)

    <kernelgen d=(32, 64, 128)>
block2: tl.constexpr = BLOCK_D // block1
BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
{% set block2 = BLOCK_D // block1 -%}
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_D = tl.program_id(1)
off_e = tl.program_id(2)
off_d1, off_d2 = get_offsets_p2(off_D, d, block1, BLOCK_D)
off_d1 = tl.multiple_of(off_d1, block1)
off_d2 = tl.multiple_of(off_d2, block2)

Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
Delta += off_bn.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
dS += off_bn.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh + off_D.to(tl.int64) * BLOCK_D * stride_dsD
dSK += off_bn.to(tl.int64) * stride_dskb + off_h.to(tl.int64) * stride_dskh + off_D.to(tl.int64) * BLOCK_D * stride_dskD

range_t = tl.arange(0, BLOCK_T).to(tl.int64)
range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
range_e = tl.arange(0, BLOCK_E_VALID).to(tl.int64) + off_e * BLOCK_E_VALID
p_qT_d1 = Q + range_d1[:, None] * stride_qd + range_t[None, :] * stride_qt # [block1 x BLOCK_T]
p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E_VALID]

{% set block2 = BLOCK_D // block1 -%}
{% for i in range(block2) -%}
p_q_d2_{{i}} = Q + range_t[:] * stride_qt + (off_d2 + {{i}}) * stride_qd # [BLOCK_T]
{% endfor -%}

{% for i in range(block2) -%}
ds_{{i}} = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
dsk_{{i}} = tl.zeros((block1,), dtype=tl.float32)
{% endfor -%}

off_n = off_bn % n
if zero_initial_state and off_n == 0:
    gamma = 1.0
    gamma = gamma.to(tl.float32)
else:
    gamma = tl.sqrt(D).to(tl.float32)

for tid in range(0, tl.cdiv(c, BLOCK_T)):
    real_range_t = range_t + tid * BLOCK_T
    p_m = M + real_range_t * stride_mt # [BLOCK_T]
    p_delta = Delta + real_range_t * stride_dt # [BLOCK_T]
    p_l = L + real_range_t * stride_lt # [BLOCK_T]
    mask_t = real_range_t < c
    rowmax = tl.load(p_m, mask=mask_t, other=float('inf'))
    delta = tl.load(p_delta, mask=mask_t, other=0.)
    l = tl.load(p_l, mask=mask_t, other=float('inf'))
    qT_d1 = tl.load(p_qT_d1, mask=mask_t[None, :], other=0.0) # block1 x BLOCK_T
    do = tl.load(p_do, mask=mask_t[:, None], other=0.0) # [BLOCK_T x BLOCK_E_VALID]

    {% for i in range(block2) -%}
    q_d2_{{i}} = tl.load(p_q_d2_{{i}}, mask=mask_t, other=0.0) # BLOCK_T
    phiqT_{{i}} = qT_d1 * q_d2_{{i}}[None, :] # [block1 x BLOCK_T]
    {% endfor -%}

    alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
    factor = 1 / alpha / l # [BLOCK_T]
    delta_factor = delta * factor # [BLOCK_T]
    do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]

    {% for i in range(block2) -%}
    ds_{{i}} = tl.dot(phiqT_{{i}}, do, ds_{{i}}) # [block1 x BLOCK_E_VALID]
    dsk_{{i}} += -tl.sum(phiqT_{{i}} * delta_factor[None, :], 1) # [block1]
    {% endfor -%}
    p_do += BLOCK_T * stride_dot
    p_qT_d1 += BLOCK_T * stride_qt
    {% for i in range(block2) -%}
    p_q_d2_{{i}} += BLOCK_T * stride_qt
    {% endfor %}

{% for i in range(block2) -%}
range_d2_{{i}} = tl.arange(0, block1).to(tl.int64) + {{i}} * block1
p_dsk_{{i}} = dSK + range_d2_{{i}} * stride_dskD # [block1]
tl.store(p_dsk_{{i}}, dsk_{{i}} * scale_p)
p_ds_{{i}} = dS + range_d2_{{i}}[:, None] * stride_dsD + range_e[None, :] * stride_dse
tl.store(p_ds_{{i}}, ds_{{i}} * scale_p)
{% endfor -%}
    </kernelgen>
    """
    pass


preprocess_configs = [
    triton.Config({'BM': BM})
    for BM in [64, 128, 256]
]

@triton.autotune(preprocess_configs, key=["HEAD_DIM"])
@triton.jit
def _query_state_bwd_preprocess(O, DO, Delta,  #
                         stride_ob, stride_om, stride_oh, stride_oe, #
                         stride_dob, stride_dom, stride_doh, stride_doe, #
                         stride_db, stride_dm, stride_dh, #
                         BM: tl.constexpr, HEAD_DIM: tl.constexpr, M_CTX: tl.constexpr  #
                         ):
    range_m = tl.program_id(0) * BM + tl.arange(0, BM)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    mask = range_m < M_CTX
    o = tl.load(O + off_b * stride_ob + off_h * stride_oh + range_m[:, None] * stride_om + off_n[None, :] * stride_oe, cache_modifier=".cg", mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + range_m[:, None] * stride_dom + off_n[None, :] * stride_doe, cache_modifier=".cg", mask=mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_b * stride_db + off_h * stride_dh + range_m * stride_dm, delta, mask=mask)



class _query_state(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state):
        """Compute query state output. It computes the following equation:

                ( (Q @ (S/γ)) * (scale^p) * γ/α + Y_attn * exp(rowmax)/α )
            O = -------------------------------------------------------------
                ( (Q @ (s/γ)) * (scale^p) * γ/α + l_attn * exp(rowmax)/α )
            where α = max(γ, exp(rowmax))
                  γ = 1.0 if S == 0 else sqrt(D)

        args:
            Q: [b, n, c, h, d] - query
            S: [b, n, h, D, d] - state
            s: [b, n, h, D] - sum of keys
            Y_attn: [b, n, c, h, e] - attention output
            l_attn: [b, n, c, h] - sum of powered attention scores, in normal space
            rowmax: [b, n, c, h] - max of powered attention scores, in log space
            deg: int - degree of power
            scale: float - sm_scale used in attention
            zero_initial_state: bool - whether the initial state is zero
        returns:
            O: [b, n, c, h, e] - temporal-normalized output
        """
        
        b, n, c, h, d, D, e = *Q.shape, S.shape[-2], S.shape[-1]

        O = torch.empty((b*n, c, h, e), device=Q.device, dtype=Q.dtype) # [b*n, c, h, e]
        l = torch.empty((b*n, h, c), device=Q.device, dtype=torch.float32).transpose(-1, -2) # [b*n, c, h]
        # put batch dims together
        Q, S, s, Y_attn, l_attn, rowmax = map(lambda x: x.view(b*n, *x.shape[2:]), (Q, S, s, Y_attn, l_attn, rowmax))

        grid = lambda args: (b*n*h, triton.cdiv(c, args["BLOCK_T"]), triton.cdiv(e, args["BLOCK_E"]))
        _query_state_fwd[grid](
            Q, S, s, Y_attn, l_attn, rowmax, O, l, 
            *Q.stride(),
            *S.stride(),
            *s.stride(),
            *Y_attn.stride(),
            *l_attn.stride(),
            *rowmax.stride(),
            *O.stride(),
            *l.stride(),
            n, h, c, d, float(D), e, zero_initial_state, deg, scale**deg
        )

        O = O.view(b, n, c, h, e)
        ctx.save_for_backward(Q, S, O, s, l, rowmax)
        ctx.deg = deg
        ctx.zero_initial_state = zero_initial_state
        ctx.b = b
        ctx.n = n
        ctx.c = c
        ctx.h = h
        ctx.d = d
        ctx.D = D
        ctx.e = e
        ctx.scale_p = scale**deg
        return O
        
    @staticmethod
    def backward(ctx, dO):
        """ Computes the following equation for the backward pass:

            delta = (dO * O) @ 1
            dl_attn = -exp(rowmax) / alpha / l * delta
            dY_attn = exp(rowmax) / alpha / l * dO
            dS = 1 / alpha * phi_Q^T @ (dO / l)
            ds = -1 / alpha * phi_Q^T @ (delta / l)
            dphi_Q = 1 / alpha / l * (dO @ S^T - delta @ s^T)

        args:
            dO: [b, n, c, h, e]
        returns:
            dQ: [b, n, c, h, d]
            dS: [b, n, h, D, e]
        """
        Q, S, O, s, l, rowmax = ctx.saved_tensors
        b, n, c, h, d, D, e = ctx.b, ctx.n, ctx.c, ctx.h, ctx.d, ctx.D, ctx.e
        deg = ctx.deg
        zero_initial_state = ctx.zero_initial_state
        scale_p = ctx.scale_p

        dQ = torch.empty_like(Q) # [b*n, c, h, d]
        dS = torch.empty((b*n, h, D, e), device=Q.device, dtype=Q.dtype)
        ds = torch.empty((b*n, h, D), device=Q.device, dtype=torch.float32)
        delta = torch.empty((b*n, h, c), device=Q.device, dtype=torch.float32).transpose(-1, -2)
        dY_attn = torch.empty((b*n, c, h, e), device=Q.device, dtype=Q.dtype)
        dl_attn = torch.empty((b*n, h, c), device=Q.device, dtype=torch.float32).transpose(-1, -2)
        O, dO = O.view(b*n, c, h, e), dO.view(b*n, c, h, e)
        
        # --- compute delta ---
        _query_state_bwd_preprocess[lambda args: (triton.cdiv(c, args["BM"]), b*n, h)](
            O, dO, delta,
            *O.stride(),
            *dO.stride(),
            *delta.stride(),
            HEAD_DIM=e,
            M_CTX=c
        )

        # --- compute dQ and dY_attn and dl_attn ---
        _query_state_bwd_dQ[lambda args: (b*n*h, triton.cdiv(c, args["BLOCK_T"]))](
            Q, S, s, dO, delta, rowmax, l, dQ, dY_attn, dl_attn,
            *Q.stride(),
            *S.stride(),
            *s.stride(),
            *dO.stride(),
            *delta.stride(),
            *rowmax.stride(),
            *l.stride(),
            *dQ.stride(),
            *dY_attn.stride(),
            *dl_attn.stride(),
            n, h, c, d, float(D), e,
            zero_initial_state=zero_initial_state,
            deg=deg,
            scale_p=scale_p
        )

        # --- compute dS and ds ---
        _query_state_bwd_dS[lambda args: (b*n*h, triton.cdiv(D, args["BLOCK_D"]), triton.cdiv(e, args["BLOCK_E"]))](
            Q, l, rowmax, dO, delta, dS, ds,
            *Q.stride(),
            *l.stride(),
            *rowmax.stride(),
            *dO.stride(),
            *delta.stride(),
            *dS.stride(),
            *ds.stride(),
            n, h, c, d, float(D), e,
            zero_initial_state=zero_initial_state,
            deg=deg,
            scale_p=scale_p
        )

        dQ, dS, ds, dY_attn, dl_attn = map(lambda x: x.view(b, n, *x.shape[1:]), (dQ, dS, ds, dY_attn, dl_attn))
        return dQ, dS, ds, dY_attn, dl_attn, None, None, None, None


def _query_state_fn(Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state=True):
    return _query_state.apply(Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state)

query_state = torch.compiler.disable(_query_state_fn)

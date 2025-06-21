import torch
import triton
import triton.language as tl
import math
import os
from torch.utils._pytree import tree_map
import torch.nn.functional as F

from power_attention._utils import diff

def attention_reference(Q, K, V, log_G, deg, scale, r=1, w=1, causal=True, head_first=False, norm=False, use_log2=False):
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
    m = (_qidx // r) >= (_kidx // w)
    s = torch.matmul(Q, K.transpose(2,3)) * scale
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


fwd_configs = [
    triton.Config({'BM': BM, 'BN': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128, 256]\
    for BN in [32, 64]\
    for s in [1, 2, 3]\
    for w in [4, 8]
]

def keep(conf):
    BM = conf.kwargs["BM"]
    BN = conf.kwargs["BN"]
    if BM * BN < 128 * 128 and conf.num_warps == 8:
        return False
    return True

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        M_CTX = nargs["M_CTX"] if "M_CTX" in nargs else kwargs["M_CTX"]
        N_CTX = nargs["N_CTX"] if "N_CTX" in nargs else kwargs["N_CTX"]
        if config.kwargs["BM"] <= M_CTX and config.kwargs["BN"] <= N_CTX:
            pruned_configs.append(config)
            if os.environ.get("TRITON_NO_AUTOTUNE", "0") == "1":
                return pruned_configs
    return pruned_configs

def prune_configs_bwd(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        M_CTX = nargs["M_CTX"] if "M_CTX" in nargs else kwargs["M_CTX"]
        N_CTX = nargs["N_CTX"] if "N_CTX" in nargs else kwargs["N_CTX"]
        if config.kwargs["BM1"] <= M_CTX and config.kwargs["BN1"] <= N_CTX and config.kwargs["BM2"] <= M_CTX and config.kwargs["BN2"] <= N_CTX:
            pruned_configs.append(config)
            if os.environ.get("TRITON_NO_AUTOTUNE", "0") == "1":
                return pruned_configs
    return pruned_configs

@triton.jit
def _power(a, deg: tl.constexpr):
    if deg == 1:
        return a
    elif deg == 2:
        return a + a
    elif deg == 4:
        a = a + a
        return a + a
    elif deg == 8:
        a = a + a
        a = a + a
        return a + a
    else:
        return deg * a

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, #
                     start_m, range_m, range_n, r: tl.constexpr, w: tl.constexpr, #
                     deg: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, #
                     BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                     M_CTX: tl.constexpr, N_CTX: tl.constexpr, STAGE: tl.constexpr, use_log2: tl.constexpr):
    if STAGE == 1: # causal, non-masking part
        lo, hi = 0, start_m * BM
    elif STAGE == 2: # causal, masking part
        lo, hi = start_m * BM, (start_m + 1) * BM
        lo = tl.multiple_of(lo, BM)
        hi = tl.multiple_of(hi, BM)
    else: # non-causal
        lo, hi = 0, N_CTX

    p_k = tl.advance(p_k, (0, lo))
    p_v = tl.advance(p_v, (lo, 0))
    if gating:
        p_gk = tl.advance(p_gk, (lo,))

    for start_n in range(lo, hi, BN):
        start_n = tl.multiple_of(start_n, BN)
        # -- compute qk ----
        k = tl.load(p_k)
        s = tl.dot(q, k) * scale
        signs = tl.where(s > 0, 1, -1)
        if gating:
            gk = tl.load(p_gk)
        else:
            gk = None
        if use_log2:
            s = _power(tl.log2(s.abs() + 1e-7), deg)
        else:
            s = _power(tl.log(s.abs() + 1e-7), deg)
        if gating:
            s = s + gq[:, None] - gk[None, :]
        if STAGE == 2:
            mask = (range_m[:, None] // r) >= ((start_n + range_n[None, :]) // w)
            s = s + tl.where(mask, 0., -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        if deg % 2 == 0:
            if use_log2:
                p = tl.exp2(s - m_ij[:, None])
            else:
                p = tl.exp(s - m_ij[:, None])
        else:
            if use_log2:
                p = tl.exp2(s - m_ij[:, None]) * signs
            else:
                p = tl.exp(s - m_ij[:, None]) * signs
        l_ij = tl.sum(p, 1)
        # -- scale acc --
        if use_log2:
            alpha = tl.exp2(m_i - m_ij)
        else:
            alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(p_v)
        acc = tl.dot(p.to(v.dtype), v, acc)
        # -- update m_i
        m_i = m_ij
        p_k = tl.advance(p_k, (0, BN))
        p_v = tl.advance(p_v, (BN, 0))
        if gating:
            p_gk = tl.advance(p_gk, (BN,))

    return acc, l_i, m_i


@triton.autotune(list(filter(keep, fwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "deg", "norm"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
def _attn_fwd(Q, K, V, LOG_GQ, LOG_GK, L, M, Out,  #
              stride_qb, stride_qh, stride_qm, stride_qd,  #
              stride_kb, stride_kh, stride_kn, stride_kd,  #
              stride_vb, stride_vh, stride_vn, stride_ve,  #
              stride_mb, stride_mh, stride_mm, #
              stride_gqb, stride_gqh, stride_gqd,  #
              stride_gkb, stride_gkh, stride_gkd,  #
              stride_ob, stride_oh, stride_om, stride_oe,  #
              stride_lb, stride_lh, stride_lm, #
              H, M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
              deg: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr,  #
              DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, STAGE: tl.constexpr,  #
              BM: tl.constexpr, BN: tl.constexpr, norm: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    q_offset = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    k_offset = off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    v_offset = off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    gq_offset = off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
    gk_offset = off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    p_q = tl.make_block_ptr(Q+q_offset, (M_CTX, DIM_QK), (stride_qm, stride_qd), (start_m*BM, 0), (BM, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V+v_offset, (N_CTX, DIM_VO), (stride_vn, stride_ve), (0, 0), (BN, DIM_VO), (1, 0))
    p_k = tl.make_block_ptr(K+k_offset, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, 0), (DIM_QK, BN), (0, 1))
    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ+gq_offset, (M_CTX,), (stride_gqd,), (start_m*BM,), (BM,), (0,))
    else:
        p_gq = None
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK+gk_offset, (N_CTX,), (stride_gkd,), (0,), (BN,), (0,))
    else:
        p_gk = None

    range_m = start_m * BM + tl.arange(0, BM)
    range_n = tl.arange(0, BN)

    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32) + 1.0
    acc = tl.zeros([BM, DIM_VO], dtype=tl.float32)

    q = tl.load(p_q, cache_modifier=".cg")
    if gating:
        gq = tl.load(p_gq, cache_modifier=".cg")
    else:
        gq = None

    if STAGE & 1: # non-masking part
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, #
                                   start_m, range_m, range_n, r, w, #
                                   deg, scale, gating, BM, BN, DIM_QK, DIM_VO, #
                                   M_CTX, N_CTX, 4 - STAGE, use_log2)

    if STAGE & 2: # masking part
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, #
                                   start_m, range_m, range_n, r, w, #
                                   deg, scale, gating, BM, BN, DIM_QK, DIM_VO, #
                                   M_CTX, N_CTX, 2, use_log2)
        
    # epilogue
    if norm: # normalize by temporal norm and fuse norm into rowmax
        if use_log2:
            m_i += tl.log2(l_i)
        else:
            m_i += tl.log(l_i)
        acc = acc / l_i[:, None]
    o_offset = off_b.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
    m_offset = off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    l_offset = off_b.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
    p_o = tl.make_block_ptr(Out+o_offset, (M_CTX, DIM_VO), (stride_om, stride_oe), (start_m*BM, 0), (BM, DIM_VO), (1, 0))
    p_l = L + l_offset + range_m * stride_lm
    p_m = M + m_offset + range_m * stride_mm
    tl.store(p_m, m_i)
    if not norm: # store temporal norm
        tl.store(p_l, l_i)
    tl.store(p_o, acc.to(Out.type.element_ty))


bwd_configs = [
    triton.Config({'BN1': BN1, 'BM1': BM1, 'BN2': BN2, 'BM2': BM2, 'BLK_SLICE_FACTOR': BLK_SLICE_FACTOR}, num_stages=s, num_warps=w) \
    for BN1 in [64, 128]\
    for BM1 in [16, 32]\
    for BM2 in [64, 128]\
    for BN2 in [16, 32]\
    for s in [1, 3]\
    for w in [4, 8]\
    for BLK_SLICE_FACTOR in [1, 2]\
]

def keep_bwd(conf):
    BN1 = conf.kwargs["BN1"]
    BM2 = conf.kwargs["BM2"]
    BN2 = conf.kwargs["BN2"]
    BM1 = conf.kwargs["BM1"]
    FACTOR = conf.kwargs["BLK_SLICE_FACTOR"]
    if BN1 != BM2 or BN2 // FACTOR < 16 or BM1 // FACTOR < 16:
        return False
    return True


preprocess_configs = [
    triton.Config({'BM': BM})
    for BM in [64, 128, 256]
]

@triton.autotune(preprocess_configs, key=["M_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,  #
                         stride_ob, stride_oh, stride_om, stride_oe, #
                         stride_dob, stride_doh, stride_dom, stride_doe, #
                         stride_db, stride_dh, stride_dm, #
                         BM: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    range_m = tl.program_id(0) * BM + tl.arange(0, BM)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_b * stride_ob + off_h * stride_oh + range_m[:, None] * stride_om + off_n[None, :] * stride_oe, cache_modifier=".cg")
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + range_m[:, None] * stride_dom + off_n[None, :] * stride_doe, cache_modifier=".cg").to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_b * stride_db + off_h * stride_dh + range_m * stride_dm, delta)


@triton.jit
def _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                    Q, LOG_GQ, DO, M, Delta, DL, #
                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                    M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
                    deg: tl.constexpr, log_scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                    start_n, start_m, num_steps: tl.constexpr, #
                    MASK: tl.constexpr, norm: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_n = start_n + tl.arange(0, BN)

    p_qT = tl.make_block_ptr(Q, (DIM_QK, M_CTX), (stride_qd, stride_qm), (0, start_m), (DIM_QK, BM), (0, 1))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_doe), (start_m, 0), (BM, DIM_VO), (1, 0))
    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM,), (0,))
    else:
        p_gq = None

    curr_m = start_m
    for _ in range(num_steps):
        range_m = curr_m + tl.arange(0, BM)
        # --- re-compute p ---
        qT = tl.load(p_qT)
        if gating:
            gq = tl.load(p_gq)
        else:
            gq = None
        sT = tl.dot(k, qT) # (N, M)
        s_signs = tl.where(sT > 0, 1, -1)
        if use_log2:
            zT = _power(tl.log2(sT.abs() + 1e-7) + log_scale, deg)
        else:
            zT = _power(tl.log(sT.abs() + 1e-7) + log_scale, deg)
        if gating:
            zT = zT + gq[None, :] - gk[:, None]
        p_m = M + range_m * stride_mm
        m = tl.load(p_m)
        if MASK:
            mask = (range_m[None, :] // r) >= (range_n[:, None] // w)
            zT = tl.where(mask, zT, -float("inf"))
        if use_log2:
            pT = tl.exp2(zT - m[None, :])
        else:
            pT = tl.exp(zT - m[None, :])
        if deg % 2 == 1:
            pT = pT * s_signs

        # --- compute dv ---
        do = tl.load(p_do)
        dv = tl.dot(pT.to(Q.type.element_ty), do, dv)

        # --- compute dp ---
        if norm:
            dl_or_delta = - tl.load(Delta + range_m * stride_dm)
        else:
            dl_or_delta = tl.load(DL + range_m * stride_dlm)
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32) # (BN, BM)
        dsT = pT * (dpT + dl_or_delta[None, :])
        if gating:
            dgk += -tl.sum(dsT, 1, keep_dims=False)
        signs = tl.where(sT > 0, 1, -1) # recreate signs to allow compiler to free s_signs, hopefully
        if use_log2:
            dsT = _power(dsT * signs * 1.44269504 / (tl.abs(sT) + 1e-7), deg)
        else:
            dsT = _power(dsT * signs / (tl.abs(sT) + 1e-7), deg)
        dk = tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT), dk)
        # increment pointers
        curr_m += BM
        p_qT = tl.advance(p_qT, (0, BM))
        p_do = tl.advance(p_do, (BM, 0))
        if gating:
            p_gq = tl.advance(p_gq, (BM,))

    return dk, dv, dgk


@triton.jit
def _attn_bwd_dq(dq, dgq, q, gq, do, m, dl_or_delta, #
                  K, V, LOG_GK, #
                  stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                  M_CTX, N_CTX, r, w, #
                  deg: tl.constexpr, log_scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                  start_m, start_n, num_steps: tl.constexpr, #
                  MASK: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_m = start_m + tl.arange(0, BM)

    p_kT = tl.make_block_ptr(K, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, start_n), (DIM_QK, BN), (0, 1))
    p_vT = tl.make_block_ptr(V, (DIM_VO, N_CTX), (stride_ve, stride_vn), (0, start_n), (DIM_VO, BN), (0, 1))
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN,), (0,))
    else:
        p_gk = None

    curr_n = start_n
    for _ in range(num_steps):
        range_n = curr_n + tl.arange(0, BN)
        # --- re-compute p ---
        kT = tl.load(p_kT)
        vT = tl.load(p_vT)
        if gating:
            gk = tl.load(p_gk)
        else:
            gk = None
        s = tl.dot(q, kT) # (M, N)
        s_signs = tl.where(s > 0, 1, -1)
        if use_log2:
            z = _power(tl.log2(s.abs() + 1e-7) + log_scale, deg)
        else:
            z = _power(tl.log(s.abs() + 1e-7) + log_scale, deg)
        if gating:
            z = z + gq[:, None] - gk[None, :]
        if MASK:
            mask = (range_m[:, None] // r) >= (range_n[None, :] // w)
            z = tl.where(mask, z, -float("inf"))
        
        if use_log2:
            p = tl.exp2(z - m[:, None])
        else:
            p = tl.exp(z - m[:, None])
        if deg % 2 == 1:
            p = p * s_signs

        # --- compute dQ ---
        dp = tl.dot(do, vT, out_dtype=tl.float32)
        ds = p * (dp + dl_or_delta[:, None])
        if gating:
            dgq += tl.sum(ds, 1, keep_dims=False)
        signs = tl.where(s > 0, 1, -1) # recreate signs to allow compiler to free s_signs, hopefully
        if use_log2:
            ds = _power(ds * signs * 1.44269504 / (tl.abs(s) + 1e-7), deg)
        else:
            ds = _power(ds * signs / (tl.abs(s) + 1e-7), deg)
        dq = tl.dot(ds.to(kT.type.element_ty), tl.trans(kT), dq)
        # increment pointers
        curr_n += BN
        p_kT = tl.advance(p_kT, (0, BN))
        p_vT = tl.advance(p_vT, (0, BN))
        if gating:
            p_gk = tl.advance(p_gk, (BN,))

    return dq, dgq


@triton.autotune(list(filter(keep_bwd, bwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "deg"], prune_configs_by={'early_config_prune': prune_configs_bwd})
@triton.jit
def _attn_bwd(Q, K, V, LOG_GQ, LOG_GK, M, Delta, DO, DL, DQ, DK, DV, DLOG_GQ, DLOG_GK, #
              stride_qb, stride_qh, stride_qm, stride_qd, #
              stride_kb, stride_kh, stride_kn, stride_kd, #
              stride_vb, stride_vh, stride_vn, stride_ve, #
              stride_mb, stride_mh, stride_mm, #
              stride_db, stride_dh, stride_dm, #
              stride_dob, stride_doh, stride_dom, stride_doe, #
              stride_dlb, stride_dlh, stride_dlm, #
              stride_dqb, stride_dqh, stride_dqm, stride_dqd, #
              stride_dkb, stride_dkh, stride_dkn, stride_dkd, #
              stride_dvb, stride_dvh, stride_dvn, stride_dve, #
              stride_gqb, stride_gqh, stride_gqm, #
              stride_gkb, stride_gkh, stride_gkn, #
              H, M_CTX, N_CTX, r, w, #
              deg: tl.constexpr,  #
              scale: tl.constexpr,  #
              gating: tl.constexpr,  #
              DIM_QK: tl.constexpr,  #
              DIM_VO: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              BM1: tl.constexpr,  #
              BN1: tl.constexpr,  #
              BM2: tl.constexpr,  #
              BN2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              norm: tl.constexpr,  #
              use_log2: tl.constexpr,  #
              ):
    if STAGE == 3:
        tl.static_assert((BM1 // BLK_SLICE_FACTOR) % r == 0, "Sliced BM1 must be divisible by w")
        tl.static_assert((BN2 // BLK_SLICE_FACTOR) % w == 0, "Sliced BN2 must be divisible by w")
    else:
        tl.static_assert(BM1 % r == 0, "BM1 must be divisible by r")
        tl.static_assert(BN2 % w == 0, "BN2 must be divisible by w")
    tl.static_assert(BN1 % w == 0, "BN1 must be divisible by w")
    tl.static_assert(BM2 % r == 0, "BM2 must be divisible by r")

    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    start_n = tl.program_id(0)*BN1
    log_scale = tl.log2(scale) if use_log2 else tl.log(scale)


    Q += off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    M += off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    Delta += off_b.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
    DO += off_b.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
    DL += off_b.to(tl.int64) * stride_dlb + off_h.to(tl.int64) * stride_dlh
    DQ += off_b.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
    DK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
    DV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
    if gating:
        LOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        LOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh
        DLOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        DLOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    # -- First part: compute dk, dv
    MASK_BLOCK_M1: tl.constexpr = BM1 // BLK_SLICE_FACTOR
    range_n = start_n + tl.arange(0, BN1)

    dv = tl.zeros([BN1, DIM_VO], dtype=tl.float32)
    dk = tl.zeros([BN1, DIM_QK], dtype=tl.float32)
    dgk = tl.zeros([BN1,], dtype=tl.float32)

    # load k, v, gk
    p_k = tl.make_block_ptr(K, (N_CTX, DIM_QK), (stride_kn, stride_kd), (start_n, 0), (BN1, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V, (N_CTX, DIM_VO), (stride_vn, stride_ve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    k = tl.load(p_k, cache_modifier=".cg")
    v = tl.load(p_v, cache_modifier=".cg")
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN1,), (0,))
        gk = tl.load(p_gk, cache_modifier=".cg")
    else:
        gk = None

    start_m = start_n if STAGE == 3 else 0
    if STAGE & 2: # masked blocks
        num_steps = BN1 // MASK_BLOCK_M1
        dk, dv, dgk = _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                                    Q, LOG_GQ, DO, M, Delta, DL, #
                                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                                    M_CTX, N_CTX, r, w, #
                                    deg, log_scale, gating, MASK_BLOCK_M1, BN1, DIM_QK, DIM_VO, #
                                    start_n, start_m, num_steps, #
                                    MASK=True, norm=norm, use_log2=use_log2)
        start_m += num_steps * MASK_BLOCK_M1
        
    # unmasked blocks
    num_steps = (M_CTX - start_m) // BM1
    dk, dv, dgk = _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                                Q, LOG_GQ, DO, M, Delta, DL, #
                                stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                                M_CTX, N_CTX, r, w, #
                                deg, log_scale, gating, BM1, BN1, DIM_QK, DIM_VO, #
                                start_n, start_m, num_steps, #
                                MASK=False, norm=norm, use_log2=use_log2)

    p_dv = tl.make_block_ptr(DV, (N_CTX, DIM_VO), (stride_dvn, stride_dve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    p_dk = tl.make_block_ptr(DK, (N_CTX, DIM_QK), (stride_dkn, stride_dkd), (start_n, 0), (BN1, DIM_QK), (1, 0))

    tl.store(p_dv, dv.to(DV.type.element_ty))
    tl.store(p_dk, dk.to(DK.type.element_ty))
    if gating:
        p_dgk = DLOG_GK + range_n * stride_gkn
        tl.store(p_dgk, dgk)

    # -- Second part: compute dq
    start_m = tl.program_id(0) * BM2

    MASK_BLOCK_N2: tl.constexpr = BN2 // BLK_SLICE_FACTOR

    # load q, gq
    p_q = tl.make_block_ptr(Q, (M_CTX, DIM_QK), (stride_qm, stride_qd), (start_m, 0), (BM2, DIM_QK), (1, 0))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_doe), (start_m, 0), (BM2, DIM_VO), (1, 0))

    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM2,), (0,))
    else:
        p_gq = None
    q = tl.load(p_q, cache_modifier=".cg")
    if gating:
        gq = tl.load(p_gq, cache_modifier=".cg")
    else:
        gq = None
    m = tl.load(M + (start_m + tl.arange(0, BM2)) * stride_mm)
    if norm:
        dl_or_delta = - tl.load(Delta + (start_m + tl.arange(0, BM2)) * stride_dm, cache_modifier=".cg")
    else:
        dl_or_delta = tl.load(DL + (start_m + tl.arange(0, BM2)) * stride_dlm, cache_modifier=".cg")
    do = tl.load(p_do, cache_modifier=".cg")

    dq = tl.zeros([BM2, DIM_QK], dtype=tl.float32)
    dgq = tl.zeros([BM2,], dtype=tl.float32)

    end_n = (start_m + BM2) if STAGE & 2 else M_CTX
    if STAGE & 2: # masked blocks
        num_steps = BM2 // MASK_BLOCK_N2
        dq, dgq = _attn_bwd_dq(dq, dgq, q, gq, do, m, dl_or_delta, #
                               K, V, LOG_GK, #
                               stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                               M_CTX, N_CTX, r, w, #
                               deg, log_scale, gating, BM2, MASK_BLOCK_N2, DIM_QK, DIM_VO, #
                               start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, #
                               MASK=True, use_log2=use_log2)
        end_n -= num_steps * MASK_BLOCK_N2
    
    # unmasked blocks
    num_steps = end_n // BN2
    dq, dgq = _attn_bwd_dq(dq, dgq, q, gq, do, m, dl_or_delta, #
                           K, V, LOG_GK, #
                           stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                           M_CTX, N_CTX, r, w, #
                           deg, log_scale, gating, BM2, BN2, DIM_QK, DIM_VO, #
                           start_m, end_n - num_steps * BN2, num_steps, #
                           MASK=False, use_log2=use_log2)

    # store dq, dgq
    p_dq = tl.make_block_ptr(DQ, (M_CTX, DIM_QK), (stride_dqm, stride_dqd), (start_m, 0), (BM2, DIM_QK), (1, 0))
    tl.store(p_dq, dq.to(DQ.type.element_ty))
    if gating:
        p_dgq = tl.make_block_ptr(DLOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM2,), (0,))
        tl.store(p_dgq, dgq)


class _power_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, log_G, deg, r, w, causal, head_first, scale, norm, use_log2):
        """ 
            Attention formulation of power attention.
            
            When norm is True, the output is normalized by temporal norm (l), and the returned
            temporal norm is uninitialized.
            When norm is False, the output is not normalized, and l is the temporal norm (sum of exponentials of the powered attention scores)


            Args:
            Q: (B, H_Q, CTX, D)
            K: (B, H_K, CTX, D)
            V: (B, H_K, CTX, E)
            deg: int
            log_G: (B, H_Q // R, CTX) or (B, CTX, H_Q // R)
            r: int, number of heads in q to form a group
            w: int, number of heads in k to form a group
            causal: bool
            head_first: bool
            norm: bool
            use_log2: bool

            Returns:
                o: (B, H_Q // R, CTX, E) if head_first else (B, CTX, H_Q // R, E)
                l: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
                rowmax: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
        """
        if head_first:
            b, hq, t, d, hk, e = *Q.shape, K.shape[1], V.shape[-1]
        else:
            b, t, hq, d, hk, e = *Q.shape, K.shape[2], V.shape[-1]
        assert r in {1, 2, 4, 8, 16}, "r must be 1, 2, 4, 8, or 16"
        assert w in {1, 2, 4, 8, 16}, "w must be 1, 2, 4, 8, or 16"
        assert hq % r == 0, "hq must be divisible by r"
        assert hk % w == 0, "hk must be divisible by w"
        assert hq // r == hk // w, "hq // r must be equal to hk // w"
        assert isinstance(deg, int) and deg > 0, "deg must be a positive integer"
        assert d in {16, 32, 64, 128, 256}, "d must be 16, 32, 64, 128, or 256"
        assert e in {16, 32, 64, 128, 256}, "e must be 16, 32, 64, 128, or 256"

        h = hq // r
        gating = log_G is not None
        if use_log2:
            log_G = log_G * math.log2(math.e)
        if head_first:
            o = torch.empty((b, h, t, e), device=Q.device, dtype=Q.dtype)
            if gating:
                assert log_G.shape == (b, h, t)
                log_GQ = log_G.repeat_interleave(r, dim=2)
                log_GK = log_G.repeat_interleave(w, dim=2)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(1), log_GQ.stride(2))
                gk_strides = (log_GK.stride(0), log_GK.stride(1), log_GK.stride(2))
            else:
                log_GQ = None
                log_GK = None
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, h, t * r, d)
            K = K.view(b, h, t * w, d)
            V = V.view(b, h, t * w, e)
            rowmax = torch.empty((b, h, t), device=Q.device, dtype=torch.float32)
            l = torch.empty((b, h, t), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3))
            k_strides = (K.stride(0), K.stride(1), K.stride(2), K.stride(3))
            v_strides = (V.stride(0), V.stride(1), V.stride(2), V.stride(3))
            l_strides = (l.stride(0), l.stride(1), l.stride(2))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(1), rowmax.stride(2))
            o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
        else:
            o = torch.empty((b, t, h, e), device=Q.device, dtype=Q.dtype)
            if gating:
                assert log_G.shape == (b, t, h)
                log_GQ = log_G.repeat_interleave(r, dim=1)
                log_GK = log_G.repeat_interleave(w, dim=1)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(2), log_GQ.stride(1))
                gk_strides = (log_GK.stride(0), log_GK.stride(2), log_GK.stride(1))
            else:
                log_GQ = None
                log_GK = None
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, t * r, h, d)
            K = K.view(b, t * w, h, d)
            V = V.view(b, t * w, h, e)
            rowmax = torch.empty((b, t, h), device=Q.device, dtype=torch.float32)
            l = torch.empty((b, t, h), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(2), Q.stride(1), Q.stride(3))
            k_strides = (K.stride(0), K.stride(2), K.stride(1), K.stride(3))
            v_strides = (V.stride(0), V.stride(2), V.stride(1), V.stride(3))
            l_strides = (l.stride(0), l.stride(2), l.stride(1))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(2), rowmax.stride(1))
            o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

        stage = 3 if causal else 1

        grid = lambda args: (triton.cdiv(r*t, args["BM"]), b * h)
        _attn_fwd[grid](
            Q, K, V, log_GQ, log_GK, l, rowmax, o, *q_strides, *k_strides, *v_strides, *rowmax_strides, *gq_strides, *gk_strides, *o_strides, *l_strides,
            H=h, M_CTX=t*r, N_CTX=t*w, r=r, w=w, deg=deg, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        
        ctx.save_for_backward(Q, K, V, l, rowmax, o, log_GQ, log_GK)
        ctx.b = b
        ctx.h = h
        ctx.t = t
        ctx.r = r
        ctx.w = w
        ctx.grid = grid
        ctx.d = d
        ctx.e = e
        ctx.deg = deg
        ctx.gating = gating
        ctx.scale = scale
        ctx.norm = norm
        ctx.q_strides = q_strides
        ctx.k_strides = k_strides
        ctx.v_strides = v_strides
        ctx.rowmax_strides = rowmax_strides
        ctx.gq_strides = gq_strides
        ctx.gk_strides = gk_strides
        ctx.o_strides = o_strides
        ctx.head_first = head_first
        ctx.stage = stage
        ctx.use_log2 = use_log2
        return o, l, rowmax

    @staticmethod
    def backward(ctx, do, dl, drowmax):
        Q, K, V, l, rowmax, o, log_GQ, log_GK = ctx.saved_tensors
        if log_GQ is not None:
            assert log_GQ.is_contiguous() # needed for reuse log_GQ's strides for dlog_GQ
            assert log_GK.is_contiguous() # needed for reuse log_GK's strides for dlog_GK
        assert do.is_contiguous()
        b, h, t, stage, norm, gating, use_log2 = ctx.b, ctx.h, ctx.t, ctx.stage, ctx.norm, ctx.gating, ctx.use_log2
        q_strides, k_strides, v_strides, rowmax_strides, gq_strides, gk_strides = ctx.q_strides, ctx.k_strides, ctx.v_strides, ctx.rowmax_strides, ctx.gq_strides, ctx.gk_strides
        r, w, d, e, deg, scale = ctx.r, ctx.w, ctx.d, ctx.e, ctx.deg, ctx.scale

        dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)
        dlog_GQ = torch.empty_like(log_GQ) if gating else None
        dlog_GK = torch.empty_like(log_GK) if gating else None
        delta = torch.empty_like(rowmax) if norm else torch.empty((0, 0, 0))

        if ctx.head_first:
            do_strides, dQ_strides, dK_strides, dV_strides, o_strides = map(lambda x: (x.stride(0), x.stride(1), x.stride(2), x.stride(3)), (do, dQ, dK, dV, o))
            dl_strides, delta_strides = map(lambda x: (x.stride(0), x.stride(1), x.stride(2)), (dl, delta))
        else:
            do_strides, dQ_strides, dK_strides, dV_strides, o_strides = map(lambda x: (x.stride(0), x.stride(2), x.stride(1), x.stride(3)), (do, dQ, dK, dV, o))
            dl_strides, delta_strides = map(lambda x: (x.stride(0), x.stride(2), x.stride(1)), (dl, delta))

        if norm:
            _attn_bwd_preprocess[lambda args: (triton.cdiv(t, args["BM"]), b, h)](
                o, do, delta,
                *o_strides, *do_strides, *delta_strides,
                HEAD_DIM=e
            )

        _attn_bwd[lambda args: (triton.cdiv(w*t, args["BN1"]), b * h)](
            Q, K, V, log_GQ, log_GK, rowmax, delta, do, dl, dQ, dK, dV, dlog_GQ, dlog_GK,
            *q_strides, *k_strides, *v_strides,
            *rowmax_strides, *delta_strides, *do_strides, *dl_strides, 
            *dQ_strides, *dK_strides, *dV_strides,
            *gq_strides, *gk_strides,
            H=h, M_CTX=t*r, N_CTX=t*w, r=r, w=w, deg=deg, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        if gating:
            if ctx.head_first:
                dlog_G = dlog_GQ.view(b, h, t, r).sum(dim=-1) + dlog_GK.view(b, h, t, w).sum(dim=-1)
            else:
                dlog_G = dlog_GQ.view(b, t, r, h).sum(dim=-2) + dlog_GK.view(b, t, w, h).sum(dim=-2)
        else:
            dlog_G = None
        return dQ, dK, dV, dlog_G, None, None, None, None, None, None, None, None

def _attention_fn(Q, K, V, log_G, deg, r=1, w=1, causal=True, head_first=False, scale=1.0, norm=False, use_log2=False):
    Y, l, rowmax = _power_attention.apply(Q, K, V, log_G, deg, r, w, causal, head_first, scale, norm, use_log2)
    rowmax = rowmax.detach()
    if norm:
        return Y
    return Y, l, rowmax

attention = torch.compiler.disable(_attention_fn)

if __name__ == "__main__":
    from power_attention._attention.impl import attention as attention_cutlass, create_inputs as create_inputs_cutlass
    from power_attention._attention.reference import attention_reference as attention_reference_old
    from power_attention._attention.create_inputs import create_inputs
    from perf._timing import benchmark_speed

    VERBOSE = True

    # Thorough benchmarking
    kw = dict(b=1, h=6, d=64, dtype=torch.bfloat16, device='cuda', scale=1.0, deg=2, seed=42, std=1/8.0, gating=True, norm=False)
    def print_rowstr(rowstr):
        print(" | ".join([f"{r.upper():<10}" for r in rowstr.split(",")]))

    token_count = 2**16
    for deg in [2, 4]:
        for mode in ['fwd', 'bwd']:
            print(f"triton-vs-cutlass-token{token_count}-head{kw['h']}-dim{kw['d']}-deg{deg}-{mode}")
            print_rowstr("chunk_size,triton,cutlass,triton speedup")
            for ctx in [2**i for i in range(7, 16)]:
                kw['t'] = ctx
                kw['b'] = token_count // ctx
                kw['deg'] = deg
                triton_time = benchmark_speed(mode, attention, create_inputs, kw, compile=False)
                cutlass_time = benchmark_speed(mode, attention_cutlass, create_inputs_cutlass, {key: kw[key] for key in kw if key != 'norm'}, compile=False)
                speedup = cutlass_time / triton_time
                print_rowstr(f"{ctx}, {triton_time:.2f}, {cutlass_time:.2f}, {speedup:.2f}")

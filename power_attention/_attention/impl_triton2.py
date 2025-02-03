"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch

import triton
import triton.language as tl

DEVICE = torch.cuda._get_device(0)
NORM : tl.constexpr = False

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, acc_log_g_q,  #
                    K_block_ptr, ALG_K_block_ptr, V_block_ptr,  #
                    start_m, sm_scale, r, w, deg: tl.constexpr, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    ALG_K_block_ptr = tl.advance(ALG_K_block_ptr, (lo,))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        acc_log_g_k = tl.load(ALG_K_block_ptr)
        qk = tl.dot(q, k) * sm_scale
        log_decay = acc_log_g_q[:, None] - acc_log_g_k[None, :]
        if True: # STAGE == 2:
            # todo
            mask = (offs_m[:, None] // r) >= ((start_n + offs_n[None, :]) // w)
            log_decay = log_decay + tl.where(mask, 0., -float('inf'))
            qk = qk * tl.exp(log_decay)
            m_ij = tl.maximum(m_i, tl.max(tl.abs(qk), 1))
            p = qk / m_ij[:, None]
            if deg == 1:
                p = p
            elif deg == 2:
                p = p * p
        else:
            qk = qk * tl.exp(log_decay)
            m_ij = tl.maximum(m_i, tl.max(tl.abs(qk), 1))
            p = qk / m_ij[:, None]
        # -- update m_i and l_i
        alpha = m_i / m_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        ALG_K_block_ptr = tl.advance(ALG_K_block_ptr, (BLOCK_N,))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, ALGR, ALGW, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_algrz, stride_algrh, stride_algrk,  #
              stride_algwz, stride_algwh, stride_algwk,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX, r, w, deg: tl.constexpr, #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    algr_offset = off_z.to(tl.int64) * stride_algrz + off_h.to(tl.int64) * stride_algrh
    algw_offset = off_z.to(tl.int64) * stride_algwz + off_h.to(tl.int64) * stride_algwh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    ALG_Q_block_ptr = tl.make_block_ptr(
        base=ALGR + algr_offset,
        shape=(N_CTX,),
        strides=(stride_qk,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    ALG_K_block_ptr = tl.make_block_ptr(
        base=ALGW + algw_offset,
        shape=(N_CTX,),
        strides=(stride_qk,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-7
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    alg_q = tl.load(ALG_Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, _, m_i = _attn_fwd_inner(acc, l_i, m_i, q, alg_q, K_block_ptr, ALG_K_block_ptr, V_block_ptr,  #
                                        start_m, sm_scale, r, w, deg, #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, _, m_i = _attn_fwd_inner(acc, l_i, m_i, q, alg_q, K_block_ptr, ALG_K_block_ptr, V_block_ptr,  #
                                        start_m, sm_scale, r, w, deg, #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    # m_i += tl.math.log2(l_i)
    if NORM:
        acc = acc - (tl.sum(acc, axis=-1, keep_dims=True) / HEAD_DIM)
        acc = acc / tl.sqrt(tl.sum(acc*acc, axis=-1, keep_dims=True) / HEAD_DIM)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv, dalg_k,  #
                   Q, ALG, k, v, alg_k, sm_scale,  #
                   DO,  #
                   M, D,  #
                   stride_qtok, stride_qd,  #
                   stride_ktok, stride_kd,  #
                   H, N_CTX, r, w, deg: tl.constexpr, #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_qtok + offs_k[:, None] * stride_qd
    do_ptrs = DO + offs_m[:, None] * stride_qtok + offs_k[None, :] * stride_qd
    alg_q_ptrs = ALG + offs_m * 1
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        alg_q = tl.load(alg_q_ptrs)
        log_decay_T = alg_q[None, :] - alg_k[:, None]
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT) * sm_scale
        pT_pre = qkT / m[None, :]
        # Autoregressive masking.
        if True: # MASK:
            # todo
            mask = (offs_m[None, :] // r) >= (offs_n[:, None] // w)
            log_decay_T = tl.where(mask, log_decay_T, -float('inf'))
        decay_T = tl.exp(log_decay_T)
        pT = pT_pre * decay_T
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        if True: # MASK:
            # todo
            dpT = tl.where(mask, dpT, 0.0)
        dqkT = decay_T * dpT * sm_scale / m[None, :]
        dqkT = dqkT.to(tl.float16)
        dk += tl.dot(dqkT, tl.trans(qT))
        # Compute dacc_log_g.
        ddecayT = pT_pre * dpT
        dlog_decayT = ddecayT * decay_T
        dalg_k += -tl.sum(dlog_decayT, 1)
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qtok
        do_ptrs += step_m * stride_qtok
        alg_q_ptrs += step_m * 1
    return dk, dv, dalg_k


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, dalg_q, q, alg_q, K, V, ALG, sm_scale, #
                 do, m, D,
                 stride_qtok, stride_qd,  #
                 stride_ktok, stride_kd,  #
                 H, N_CTX, r, w, deg: tl.constexpr, #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_ktok + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n[None, :] * stride_ktok + offs_k[:, None] * stride_kd
    alg_k_ptrs = ALG + offs_n * 1
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        alg_k = tl.load(alg_k_ptrs)
        qk = tl.dot(q, kT) * sm_scale
        log_decay = alg_q[:, None] - alg_k[None, :]
        p_pre = qk / m
        if deg == 1:
            p = p_pre
        elif deg == 2:
            p = p_pre * p_pre
        # Autoregressive masking.
        if True: # MASK:
            # todo
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] // r) >= (offs_n[None, :] // w)
            log_decay = tl.where(mask, log_decay, -float('inf'))
        decay = tl.exp(log_decay)
        p = p_pre * decay
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        if True: # MASK:
            # todo
            dp = tl.where(mask, dp, 0.0)
        dqk = decay * dp / m * sm_scale
        if deg == 1:
            dqk = dqk
        elif deg == 2:
            dqk = 2 * dqk
        dqk = dqk.to(tl.float16)
        # Compute dQ.
        dq += tl.dot(dqk, tl.trans(kT))
        # Compute dacc_log_g.
        ddecay = p_pre * dp
        dlog_decay = ddecay * decay
        dalg_q += tl.sum(dlog_decay, 1)
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_ktok
        vT_ptrs += step_n * stride_ktok
        alg_k_ptrs += step_n * 1
    return dq, dalg_q


@triton.jit
def _attn_bwd(Q, K, V, ALGR, ALGW, sm_scale,  #
              DO,  #
              DQ, DK, DV, DALG_Q, DALG_K,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_qz, stride_qh, stride_qtok, stride_qd,  #
              stride_kz, stride_kh, stride_ktok, stride_kd,  #
              stride_algrz, stride_algrh, stride_algrk,  #
              stride_algwz, stride_algwh, stride_algwk,  #
              H, N_CTX, r, w, deg: tl.constexpr, #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adjq = (stride_qh * (bhid % H) + stride_qz * (bhid // H)).to(tl.int64)
    adjk = (stride_kh * (bhid % H) + stride_kz * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adjq
    K += adjk
    V += adjk
    DO += adjq
    DQ += adjq
    DK += adjk
    DV += adjk
    M += off_chz
    D += off_chz
    ALGR += (stride_algrh * (bhid % H) + stride_algrz * (bhid // H)).to(tl.int64)
    ALGW += (stride_algwh * (bhid % H) + stride_algwz * (bhid // H)).to(tl.int64)
    DALG_Q += (stride_algrh * (bhid % H) + stride_algrz * (bhid // H)).to(tl.int64)
    DALG_K += (stride_algwh * (bhid % H) + stride_algwz * (bhid // H)).to(tl.int64)

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dalg_k = tl.zeros([BLOCK_N1,], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd)
    v = tl.load(V + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd)
    alg_k = tl.load(ALGW + offs_n * stride_algwk)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv, dalg_k = _attn_bwd_dkdv(dk, dv, dalg_k, #
                            Q, ALGR, k, v, alg_k, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_qtok, stride_qd,  #
                            stride_ktok, stride_kd,  #
                            H, N_CTX, r, w, deg, #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv, dalg_k = _attn_bwd_dkdv(  #
        dk, dv, dalg_k,  #
        Q, ALGR, k, v, alg_k, sm_scale,  #
        DO,  #
        M, D,  #
        stride_qtok, stride_qd,  #
        stride_ktok, stride_kd,  #
        H, N_CTX, r, w, deg, #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk_ptrs = DK + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd
    tl.store(dk_ptrs, dk)

    dalgk_ptrs = DALG_K + offs_n * 1
    tl.store(dalgk_ptrs, dalg_k)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_qtok + offs_k[None, :] * stride_qd)
    alg_q = tl.load(ALGR + offs_m * 1)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    dalg_q = tl.zeros([BLOCK_M2,], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_qtok + offs_k[None, :] * stride_qd)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq, dalg_q = _attn_bwd_dq(dq, dalg_q, q, alg_q, K, V, ALGW, sm_scale, #
                      do, m, D,  #
                      stride_qtok, stride_qd,  #
                      stride_ktok, stride_kd,  #
                      H, N_CTX, r, w, deg, #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq, dalg_q = _attn_bwd_dq(dq, dalg_q, q, alg_q, K, V, ALGW, sm_scale, #
                      do, m, D,  #
                      stride_qtok, stride_qd,  #
                      stride_ktok, stride_kd,  #
                      H, N_CTX, r, w, deg, #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_qtok + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq)
    dalgq_ptrs = DALG_Q + offs_m * 1
    tl.store(dalgq_ptrs, dalg_q)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, acc_log_g, r, w, deg, sm_scale):
        causal = True
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert r == w, f'unequal reads and writes not supported: {r = } {w = }'
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        # repeat the gating
        acc_log_g_r = acc_log_g.repeat_interleave(r, dim=2)
        acc_log_g_w = acc_log_g.repeat_interleave(w, dim=2)

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, acc_log_g_r, acc_log_g_w, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            acc_log_g_r.stride(0), acc_log_g_r.stride(1), acc_log_g_r.stride(2),  #
            acc_log_g_w.stride(0), acc_log_g_w.stride(1), acc_log_g_w.stride(2),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            r=r, w=w,
            deg=deg,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, acc_log_g_r, acc_log_g_w, o, M)
        ctx.r = r
        ctx.w = w
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.deg = deg
        return o, M

    @staticmethod
    def backward(ctx, do, dM=None):
        q, k, v, acc_log_g_r, acc_log_g_w, o, M = ctx.saved_tensors
        do = do.contiguous()
        assert do.is_contiguous()
        assert q.stride() == o.stride() == do.stride(), f'{q.stride() = } {o.stride() = } {do.stride() = }'
        assert k.stride() == v.stride(), f'{k.stride() = } {v.stride() = }'
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dalg_q = torch.empty_like(acc_log_g_r)
        dalg_k = torch.empty_like(acc_log_g_w)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        delta = torch.empty_like(M)
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, k, v, acc_log_g_r, acc_log_g_w, ctx.sm_scale, do, dq, dk, dv, dalg_q, dalg_k,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            acc_log_g_r.stride(0), acc_log_g_r.stride(1), acc_log_g_r.stride(2),  #
            acc_log_g_w.stride(0), acc_log_g_w.stride(1), acc_log_g_w.stride(2),  #
            N_HEAD, N_CTX, ctx.r, ctx.w, ctx.deg, #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )
        dalg = dalg_q.view(BATCH, N_HEAD, -1, ctx.r).sum(dim=-1) + dalg_k.view(BATCH, N_HEAD, -1, ctx.w).sum(dim=-1)
        return dq, dk, dv, dalg, None, None, None, None

def attention(q, k, v, acc_log_g, r, w, deg, sm_scale):
    assert deg == 1, "only deg=1 is supported right now"
    return _attention.apply(q, k, v, acc_log_g, r, w, deg, sm_scale)

def reference_power_attention(q, k, v, acc_log_g, deg, sm_scale):
    # reference implementation
    M = torch.tril(torch.ones((t, t), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if acc_log_g is not None:
        log_decay = acc_log_g[..., :, None] - acc_log_g[..., None, :]
        log_decay[:, :, M == 0] = -float("inf")
        decay = torch.exp(log_decay)
        p = (p * decay).to(p.dtype)
    p[:, :, M == 0] = 0.
    rowmax = torch.max(p.abs(), dim=-1, keepdim=True).values.detach()
    p = p / (rowmax + 1e-7)
    p = p ** deg
    out_raw = torch.matmul(p, v)
    if NORM:
        out_centered = out_raw - out_raw.mean(dim=-1, keepdim=True)
        out = out_centered / ((out_centered**2).mean(dim=-1, keepdim=True)**.5)
    else:
        out = out_raw
    return out

def reference_power_attention_multirw(q, k, v, acc_log_g, r, w, deg, sm_scale):
    _qidx = torch.arange(t*r, device=DEVICE).unsqueeze(1)
    _kidx = torch.arange(t*w, device=DEVICE).unsqueeze(0)
    M = (_qidx // r) >= (_kidx // w)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if acc_log_g is not None:
        log_decay = acc_log_g.repeat_interleave(r, dim=2)[..., :, None] - acc_log_g.repeat_interleave(w, dim=2)[..., None, :]
        log_decay[:, :, M == 0] = -float("inf")
        decay = torch.exp(log_decay)
        p = (p * decay).to(p.dtype)
    rowmax = torch.max(p.abs(), dim=-1, keepdim=True).values.detach()
    p = p / (rowmax + 1e-7)
    p = p ** deg
    out_raw = torch.matmul(p, v)
    if NORM:
        out_centered = out_raw - out_raw.mean(dim=-1, keepdim=True)
        out = out_centered / ((out_centered**2).mean(dim=-1, keepdim=True)**.5)
    else:
        out = out_raw
    return out

def test_op(b, h, t, d, r, w, deg, dtype=torch.float16):
    torch.manual_seed(20)
    sm_scale = .5
    q_gold = (torch.empty((b, h, t*r, d), dtype=torch.float32, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k_gold = (torch.empty((b, h, t*w, d), dtype=torch.float32, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v_gold = (torch.empty((b, h, t*w, d), dtype=torch.float32, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    acc_log_g_gold = torch.cumsum(torch.nn.functional.logsigmoid(torch.empty((b, h, t), dtype=torch.float32, device=DEVICE).normal_(mean=0.0, std=0.5)), dim=-1).requires_grad_()
    dout_gold = torch.randn_like(q_gold)
    q, k, v, acc_log_g, dout = q_gold.to(dtype), k_gold.to(dtype), v_gold.to(dtype), acc_log_g_gold.clone(), dout_gold.to(dtype)
    _q, _k, _v, _acc_log_g, _dout = q_gold.to(dtype), k_gold.to(dtype), v_gold.to(dtype), acc_log_g_gold.clone(), dout_gold.to(dtype)
    q.requires_grad_(True).retain_grad()
    k.requires_grad_(True).retain_grad()
    v.requires_grad_(True).retain_grad()
    acc_log_g.requires_grad_(True).retain_grad()
    _q.requires_grad_(True).retain_grad()
    _k.requires_grad_(True).retain_grad()
    _v.requires_grad_(True).retain_grad()
    _acc_log_g.requires_grad_(True).retain_grad()
    
    # reference implementation at high precision
    gold_out = reference_power_attention_multirw(q_gold, k_gold, v_gold, acc_log_g_gold, r, w, deg, sm_scale)
    gold_out.backward(dout_gold)
    gold_dv = v_gold.grad.clone()
    gold_dk = k_gold.grad.clone()
    gold_dq = q_gold.grad.clone()
    gold_dacc_log_g = acc_log_g_gold.grad.clone()
    # reference implementation at low precision
    ref_out = reference_power_attention_multirw(q, k, v, acc_log_g, r, w, deg, sm_scale)
    ref_out.backward(dout)
    ref_dv = v.grad.clone()
    ref_dk = k.grad.clone()
    ref_dq = q.grad.clone()
    ref_dacc_log_g = acc_log_g.grad.clone()
    # # triton implementation
    tri_out, _ = attention(_q, _k, _v, _acc_log_g, r, w, deg, sm_scale)
    tri_out.backward(_dout)
    tri_dv = _v.grad.clone()
    tri_dk = _k.grad.clone()
    tri_dq = _q.grad.clone()
    tri_dacc_log_g = _acc_log_g.grad.clone()
    # compare
    # print(f'gold_dk: {gold_dk[0, 0, :3, :3]}')
    # print(f'tri_dk: {tri_dk[0, 0, :3, :3]}')
    print('reference errors')
    print(f'\tout: {(gold_out - ref_out.float()).abs().max():.8f}')
    print(f'\tdv: {(gold_dv - ref_dv.float()).abs().max():.8f}')
    print(f'\tdk: {(gold_dk - ref_dk.float()).abs().max():.8f}')
    print(f'\tdq: {(gold_dq - ref_dq.float()).abs().max():.8f}')
    print(f'\tdacc_log_g: {(gold_dacc_log_g - ref_dacc_log_g.float()).abs().max():.8f}')
    print('triton errors')
    print(f'\tout: {(gold_out - tri_out.float()).abs().max():.8f}')
    print(f'\tdv: {(gold_dv - tri_dv.float()).abs().max():.8f}')
    print(f'\tdk: {(gold_dk - tri_dk.float()).abs().max():.8f}')
    print(f'\tdq: {(gold_dq - tri_dq.float()).abs().max():.8f}')
    print(f'\tdacc_log_g: {(gold_dacc_log_g - tri_dacc_log_g.float()).abs().max():.8f}')
    # import code; code.interact(local=locals()|globals())
    assert torch.allclose(gold_out, tri_out.float(), atol=1e-1, rtol=0)
    rtol = 0.0
    assert torch.allclose(gold_dv, tri_dv.float(), atol=1e-1, rtol=rtol)
    assert torch.allclose(gold_dk, tri_dk.float(), atol=1, rtol=rtol)
    assert torch.allclose(gold_dq, tri_dq.float(), atol=1, rtol=rtol)
    assert torch.allclose(gold_dacc_log_g, tri_dacc_log_g.float(), atol=1e-1, rtol=rtol)
    print("passed.")


def create_inputs(b=4, t=128, h=4, d=32, deg=1, dtype=torch.float32, device='cuda', requires_grad=False, qhead_ratio=1):
    torch.manual_seed(0)
    assert qhead_ratio == 1, "qhead_ratio must be 1"
    dtype = torch.float16
    
    q = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    log_g = torch.randn((b, h, t), dtype=torch.float32, device=device, requires_grad=requires_grad)
    acc_log_g = torch.cumsum(torch.nn.functional.logsigmoid(log_g), dim=2)

    return dict(q=q, k=k, v=v, acc_log_g=acc_log_g, r=1, w=1, deg=deg, sm_scale=1/d**.5)

if __name__ == "__main__":
    b, h, t, d = 2, 2, 1024, 64
    deg = 1
    r, w = 2, 2
    test_op(b, h, t, d, r, w, deg)
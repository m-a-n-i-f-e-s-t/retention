import triton
import triton.language as tl

def _expand_kernel_split_D(K, phi_K, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_pb, stride_pt, stride_ph, stride_pD,
                     T, H, d: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    block2: tl.constexpr = BLOCK_D // block1
    if ((BLOCK_D == 128) and ((BLOCK_T == 16) and ((block1 == 16)))) or ((BLOCK_D == 128) and ((BLOCK_T == 32) and ((block1 == 16)))):
        
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
        p_phi_K_0 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 0 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # [BLOCK_T]
        p_phi_K_1 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 1 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd # [BLOCK_T]
        p_phi_K_2 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 2 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_2 = K + range_t[:] * stride_kt + (off_d2 + 2) * stride_kd # [BLOCK_T]
        p_phi_K_3 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 3 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_3 = K + range_t[:] * stride_kt + (off_d2 + 3) * stride_kd # [BLOCK_T]
        p_phi_K_4 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 4 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_4 = K + range_t[:] * stride_kt + (off_d2 + 4) * stride_kd # [BLOCK_T]
        p_phi_K_5 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 5 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_5 = K + range_t[:] * stride_kt + (off_d2 + 5) * stride_kd # [BLOCK_T]
        p_phi_K_6 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 6 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_6 = K + range_t[:] * stride_kt + (off_d2 + 6) * stride_kd # [BLOCK_T]
        p_phi_K_7 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 7 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_7 = K + range_t[:] * stride_kt + (off_d2 + 7) * stride_kd # [BLOCK_T]
        for tid in range(0, tl.cdiv(T, BLOCK_T)):
            k_d1 = tl.load(p_k_d1) # [BLOCK_T x block1]
            k_d2_0 = tl.load(p_k_d2_0) * multiplier # [BLOCK_T]
            phik_0 = k_d1 * k_d2_0[None, :] # [block1 x BLOCK_T]
            k_d2_1 = tl.load(p_k_d2_1) * multiplier # [BLOCK_T]
            phik_1 = k_d1 * k_d2_1[None, :] # [block1 x BLOCK_T]
            k_d2_2 = tl.load(p_k_d2_2) * multiplier # [BLOCK_T]
            phik_2 = k_d1 * k_d2_2[None, :] # [block1 x BLOCK_T]
            k_d2_3 = tl.load(p_k_d2_3) * multiplier # [BLOCK_T]
            phik_3 = k_d1 * k_d2_3[None, :] # [block1 x BLOCK_T]
            k_d2_4 = tl.load(p_k_d2_4) * multiplier # [BLOCK_T]
            phik_4 = k_d1 * k_d2_4[None, :] # [block1 x BLOCK_T]
            k_d2_5 = tl.load(p_k_d2_5) * multiplier # [BLOCK_T]
            phik_5 = k_d1 * k_d2_5[None, :] # [block1 x BLOCK_T]
            k_d2_6 = tl.load(p_k_d2_6) * multiplier # [BLOCK_T]
            phik_6 = k_d1 * k_d2_6[None, :] # [block1 x BLOCK_T]
            k_d2_7 = tl.load(p_k_d2_7) * multiplier # [BLOCK_T]
            phik_7 = k_d1 * k_d2_7[None, :] # [block1 x BLOCK_T]
            tl.store(p_phi_K_0, phik_0)
            tl.store(p_phi_K_1, phik_1)
            tl.store(p_phi_K_2, phik_2)
            tl.store(p_phi_K_3, phik_3)
            tl.store(p_phi_K_4, phik_4)
            tl.store(p_phi_K_5, phik_5)
            tl.store(p_phi_K_6, phik_6)
            tl.store(p_phi_K_7, phik_7)
            p_k_d1 += BLOCK_T * stride_kt
            p_k_d2_0 += BLOCK_T * stride_kt
            p_phi_K_0 += BLOCK_T * stride_pt
            p_k_d2_1 += BLOCK_T * stride_kt
            p_phi_K_1 += BLOCK_T * stride_pt
            p_k_d2_2 += BLOCK_T * stride_kt
            p_phi_K_2 += BLOCK_T * stride_pt
            p_k_d2_3 += BLOCK_T * stride_kt
            p_phi_K_3 += BLOCK_T * stride_pt
            p_k_d2_4 += BLOCK_T * stride_kt
            p_phi_K_4 += BLOCK_T * stride_pt
            p_k_d2_5 += BLOCK_T * stride_kt
            p_phi_K_5 += BLOCK_T * stride_pt
            p_k_d2_6 += BLOCK_T * stride_kt
            p_phi_K_6 += BLOCK_T * stride_pt
            p_k_d2_7 += BLOCK_T * stride_kt
            p_phi_K_7 += BLOCK_T * stride_pt
            
        
    elif ((BLOCK_D == 256) and ((BLOCK_T == 16) and ((block1 == 16)))) or ((BLOCK_D == 256) and ((BLOCK_T == 32) and ((block1 == 16)))):
        
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
        p_phi_K_0 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 0 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # [BLOCK_T]
        p_phi_K_1 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 1 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd # [BLOCK_T]
        p_phi_K_2 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 2 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_2 = K + range_t[:] * stride_kt + (off_d2 + 2) * stride_kd # [BLOCK_T]
        p_phi_K_3 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 3 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_3 = K + range_t[:] * stride_kt + (off_d2 + 3) * stride_kd # [BLOCK_T]
        p_phi_K_4 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 4 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_4 = K + range_t[:] * stride_kt + (off_d2 + 4) * stride_kd # [BLOCK_T]
        p_phi_K_5 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 5 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_5 = K + range_t[:] * stride_kt + (off_d2 + 5) * stride_kd # [BLOCK_T]
        p_phi_K_6 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 6 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_6 = K + range_t[:] * stride_kt + (off_d2 + 6) * stride_kd # [BLOCK_T]
        p_phi_K_7 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 7 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_7 = K + range_t[:] * stride_kt + (off_d2 + 7) * stride_kd # [BLOCK_T]
        p_phi_K_8 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 8 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_8 = K + range_t[:] * stride_kt + (off_d2 + 8) * stride_kd # [BLOCK_T]
        p_phi_K_9 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 9 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_9 = K + range_t[:] * stride_kt + (off_d2 + 9) * stride_kd # [BLOCK_T]
        p_phi_K_10 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 10 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_10 = K + range_t[:] * stride_kt + (off_d2 + 10) * stride_kd # [BLOCK_T]
        p_phi_K_11 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 11 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_11 = K + range_t[:] * stride_kt + (off_d2 + 11) * stride_kd # [BLOCK_T]
        p_phi_K_12 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 12 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_12 = K + range_t[:] * stride_kt + (off_d2 + 12) * stride_kd # [BLOCK_T]
        p_phi_K_13 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 13 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_13 = K + range_t[:] * stride_kt + (off_d2 + 13) * stride_kd # [BLOCK_T]
        p_phi_K_14 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 14 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_14 = K + range_t[:] * stride_kt + (off_d2 + 14) * stride_kd # [BLOCK_T]
        p_phi_K_15 = phi_K + range_t[None, :] * stride_pt + (off_D*BLOCK_D + 15 * block1 + tl.arange(0, block1)[:, None]).to(tl.int64) * stride_pD # [block1 x BLOCK_T]
        p_k_d2_15 = K + range_t[:] * stride_kt + (off_d2 + 15) * stride_kd # [BLOCK_T]
        for tid in range(0, tl.cdiv(T, BLOCK_T)):
            k_d1 = tl.load(p_k_d1) # [BLOCK_T x block1]
            k_d2_0 = tl.load(p_k_d2_0) * multiplier # [BLOCK_T]
            phik_0 = k_d1 * k_d2_0[None, :] # [block1 x BLOCK_T]
            k_d2_1 = tl.load(p_k_d2_1) * multiplier # [BLOCK_T]
            phik_1 = k_d1 * k_d2_1[None, :] # [block1 x BLOCK_T]
            k_d2_2 = tl.load(p_k_d2_2) * multiplier # [BLOCK_T]
            phik_2 = k_d1 * k_d2_2[None, :] # [block1 x BLOCK_T]
            k_d2_3 = tl.load(p_k_d2_3) * multiplier # [BLOCK_T]
            phik_3 = k_d1 * k_d2_3[None, :] # [block1 x BLOCK_T]
            k_d2_4 = tl.load(p_k_d2_4) * multiplier # [BLOCK_T]
            phik_4 = k_d1 * k_d2_4[None, :] # [block1 x BLOCK_T]
            k_d2_5 = tl.load(p_k_d2_5) * multiplier # [BLOCK_T]
            phik_5 = k_d1 * k_d2_5[None, :] # [block1 x BLOCK_T]
            k_d2_6 = tl.load(p_k_d2_6) * multiplier # [BLOCK_T]
            phik_6 = k_d1 * k_d2_6[None, :] # [block1 x BLOCK_T]
            k_d2_7 = tl.load(p_k_d2_7) * multiplier # [BLOCK_T]
            phik_7 = k_d1 * k_d2_7[None, :] # [block1 x BLOCK_T]
            k_d2_8 = tl.load(p_k_d2_8) * multiplier # [BLOCK_T]
            phik_8 = k_d1 * k_d2_8[None, :] # [block1 x BLOCK_T]
            k_d2_9 = tl.load(p_k_d2_9) * multiplier # [BLOCK_T]
            phik_9 = k_d1 * k_d2_9[None, :] # [block1 x BLOCK_T]
            k_d2_10 = tl.load(p_k_d2_10) * multiplier # [BLOCK_T]
            phik_10 = k_d1 * k_d2_10[None, :] # [block1 x BLOCK_T]
            k_d2_11 = tl.load(p_k_d2_11) * multiplier # [BLOCK_T]
            phik_11 = k_d1 * k_d2_11[None, :] # [block1 x BLOCK_T]
            k_d2_12 = tl.load(p_k_d2_12) * multiplier # [BLOCK_T]
            phik_12 = k_d1 * k_d2_12[None, :] # [block1 x BLOCK_T]
            k_d2_13 = tl.load(p_k_d2_13) * multiplier # [BLOCK_T]
            phik_13 = k_d1 * k_d2_13[None, :] # [block1 x BLOCK_T]
            k_d2_14 = tl.load(p_k_d2_14) * multiplier # [BLOCK_T]
            phik_14 = k_d1 * k_d2_14[None, :] # [block1 x BLOCK_T]
            k_d2_15 = tl.load(p_k_d2_15) * multiplier # [BLOCK_T]
            phik_15 = k_d1 * k_d2_15[None, :] # [block1 x BLOCK_T]
            tl.store(p_phi_K_0, phik_0)
            tl.store(p_phi_K_1, phik_1)
            tl.store(p_phi_K_2, phik_2)
            tl.store(p_phi_K_3, phik_3)
            tl.store(p_phi_K_4, phik_4)
            tl.store(p_phi_K_5, phik_5)
            tl.store(p_phi_K_6, phik_6)
            tl.store(p_phi_K_7, phik_7)
            tl.store(p_phi_K_8, phik_8)
            tl.store(p_phi_K_9, phik_9)
            tl.store(p_phi_K_10, phik_10)
            tl.store(p_phi_K_11, phik_11)
            tl.store(p_phi_K_12, phik_12)
            tl.store(p_phi_K_13, phik_13)
            tl.store(p_phi_K_14, phik_14)
            tl.store(p_phi_K_15, phik_15)
            p_k_d1 += BLOCK_T * stride_kt
            p_k_d2_0 += BLOCK_T * stride_kt
            p_phi_K_0 += BLOCK_T * stride_pt
            p_k_d2_1 += BLOCK_T * stride_kt
            p_phi_K_1 += BLOCK_T * stride_pt
            p_k_d2_2 += BLOCK_T * stride_kt
            p_phi_K_2 += BLOCK_T * stride_pt
            p_k_d2_3 += BLOCK_T * stride_kt
            p_phi_K_3 += BLOCK_T * stride_pt
            p_k_d2_4 += BLOCK_T * stride_kt
            p_phi_K_4 += BLOCK_T * stride_pt
            p_k_d2_5 += BLOCK_T * stride_kt
            p_phi_K_5 += BLOCK_T * stride_pt
            p_k_d2_6 += BLOCK_T * stride_kt
            p_phi_K_6 += BLOCK_T * stride_pt
            p_k_d2_7 += BLOCK_T * stride_kt
            p_phi_K_7 += BLOCK_T * stride_pt
            p_k_d2_8 += BLOCK_T * stride_kt
            p_phi_K_8 += BLOCK_T * stride_pt
            p_k_d2_9 += BLOCK_T * stride_kt
            p_phi_K_9 += BLOCK_T * stride_pt
            p_k_d2_10 += BLOCK_T * stride_kt
            p_phi_K_10 += BLOCK_T * stride_pt
            p_k_d2_11 += BLOCK_T * stride_kt
            p_phi_K_11 += BLOCK_T * stride_pt
            p_k_d2_12 += BLOCK_T * stride_kt
            p_phi_K_12 += BLOCK_T * stride_pt
            p_k_d2_13 += BLOCK_T * stride_kt
            p_phi_K_13 += BLOCK_T * stride_pt
            p_k_d2_14 += BLOCK_T * stride_kt
            p_phi_K_14 += BLOCK_T * stride_pt
            p_k_d2_15 += BLOCK_T * stride_kt
            p_phi_K_15 += BLOCK_T * stride_pt
            
        
    else:
        tl.static_assert(False, "No matching config found")
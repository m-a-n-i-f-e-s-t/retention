import triton
import triton.language as tl

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
                     zero_initial_state,
                     deg: tl.constexpr,
                     scale_p,
                     block1: tl.constexpr,
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    block2: tl.constexpr = BLOCK_D // block1
    BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
    if ((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((block1 == 16))))) or (((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 256) and ((block1 == 16))))) or (((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((block1 == 16))))) or ((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 256) and ((block1 == 16))))))):
        
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
        
        for m in range(0, d//block1):
            p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # BLOCK_T x block1
            q_d1 = tl.load(p_q_d1, mask=mask_T[:, None], other=0.) # BLOCK_T x block1
        
            for n in range(0, (m+1)*block1//block2):
                off_d2 = n*block2
                off_d2 = tl.multiple_of(off_d2, block2)
                off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # BLOCK_T
                p_s_0 = S + (range_d1[:, None] + off_D + 0 * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E_VALID
                p_sk_0 = SK + (range_d1 + off_D + 0 * block1) * stride_skD
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_T, other=0.) # BLOCK_T
                phik_0 = (q_d1 * (q_d2_0[:, None])).to(Q.dtype.element_ty) # BLOCK_T x block1
                s_0 = tl.load(p_s_0) # block1 x BLOCK_E_VALID
                sk_0 = tl.load(p_sk_0) # BLOCK_D
                s_0 = (s_0 / gamma).to(Q.dtype.element_ty) # block1 x BLOCK_E_VALID
                sk_0 = (sk_0 / gamma).to(Q.dtype.element_ty) # BLOCK_D
                y = tl.dot(phik_0, s_0, y) # BLOCK_T x BLOCK_E_VALID
                l += tl.sum(phik_0 * sk_0[None, :], 1) # BLOCK_T
                
        
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
            
    elif ((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((block1 == 16))))) or (((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 256) and ((block1 == 16))))) or (((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((block1 == 16))))) or ((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 256) and ((block1 == 16))))))):
        
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
        
        for m in range(0, d//block1):
            p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # BLOCK_T x block1
            q_d1 = tl.load(p_q_d1, mask=mask_T[:, None], other=0.) # BLOCK_T x block1
        
            for n in range(0, (m+1)*block1//block2):
                off_d2 = n*block2
                off_d2 = tl.multiple_of(off_d2, block2)
                off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # BLOCK_T
                p_s_0 = S + (range_d1[:, None] + off_D + 0 * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E_VALID
                p_sk_0 = SK + (range_d1 + off_D + 0 * block1) * stride_skD
                p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # BLOCK_T
                p_s_1 = S + (range_d1[:, None] + off_D + 1 * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E_VALID
                p_sk_1 = SK + (range_d1 + off_D + 1 * block1) * stride_skD
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_T, other=0.) # BLOCK_T
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_T, other=0.) # BLOCK_T
                phik_0 = (q_d1 * (q_d2_0[:, None])).to(Q.dtype.element_ty) # BLOCK_T x block1
                s_0 = tl.load(p_s_0) # block1 x BLOCK_E_VALID
                sk_0 = tl.load(p_sk_0) # BLOCK_D
                s_0 = (s_0 / gamma).to(Q.dtype.element_ty) # block1 x BLOCK_E_VALID
                sk_0 = (sk_0 / gamma).to(Q.dtype.element_ty) # BLOCK_D
                y = tl.dot(phik_0, s_0, y) # BLOCK_T x BLOCK_E_VALID
                l += tl.sum(phik_0 * sk_0[None, :], 1) # BLOCK_T
                phik_1 = (q_d1 * (q_d2_1[:, None])).to(Q.dtype.element_ty) # BLOCK_T x block1
                s_1 = tl.load(p_s_1) # block1 x BLOCK_E_VALID
                sk_1 = tl.load(p_sk_1) # BLOCK_D
                s_1 = (s_1 / gamma).to(Q.dtype.element_ty) # block1 x BLOCK_E_VALID
                sk_1 = (sk_1 / gamma).to(Q.dtype.element_ty) # BLOCK_D
                y = tl.dot(phik_1, s_1, y) # BLOCK_T x BLOCK_E_VALID
                l += tl.sum(phik_1 * sk_1[None, :], 1) # BLOCK_T
                
        
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
            
    else:
        tl.static_assert(False, "No matching config found")
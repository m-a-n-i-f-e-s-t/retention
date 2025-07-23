import triton
import triton.language as tl

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
    block2: tl.constexpr = BLOCK_D // block1
    BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
    if ((BLOCK_D == 128) and ((BLOCK_E == 32) and ((BLOCK_T == 16) and ((block1 == 16))))) or (((BLOCK_D == 128) and ((BLOCK_E == 32) and ((BLOCK_T == 32) and ((block1 == 16))))) or (((BLOCK_D == 128) and ((BLOCK_E == 64) and ((BLOCK_T == 16) and ((block1 == 16))))) or ((BLOCK_D == 128) and ((BLOCK_E == 64) and ((BLOCK_T == 32) and ((block1 == 16))))))):
        
        if (d == 32):     
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
            
            p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
            p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
            p_q_d2_2 = Q + range_t[:] * stride_qt + (off_d2 + 2) * stride_qd # [BLOCK_T]
            p_q_d2_3 = Q + range_t[:] * stride_qt + (off_d2 + 3) * stride_qd # [BLOCK_T]
            p_q_d2_4 = Q + range_t[:] * stride_qt + (off_d2 + 4) * stride_qd # [BLOCK_T]
            p_q_d2_5 = Q + range_t[:] * stride_qt + (off_d2 + 5) * stride_qd # [BLOCK_T]
            p_q_d2_6 = Q + range_t[:] * stride_qt + (off_d2 + 6) * stride_qd # [BLOCK_T]
            p_q_d2_7 = Q + range_t[:] * stride_qt + (off_d2 + 7) * stride_qd # [BLOCK_T]
            ds_0 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_0 = tl.zeros((block1,), dtype=tl.float32)
            ds_1 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_1 = tl.zeros((block1,), dtype=tl.float32)
            ds_2 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_2 = tl.zeros((block1,), dtype=tl.float32)
            ds_3 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_3 = tl.zeros((block1,), dtype=tl.float32)
            ds_4 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_4 = tl.zeros((block1,), dtype=tl.float32)
            ds_5 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_5 = tl.zeros((block1,), dtype=tl.float32)
            ds_6 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_6 = tl.zeros((block1,), dtype=tl.float32)
            ds_7 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_7 = tl.zeros((block1,), dtype=tl.float32)
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
            
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_0 = qT_d1 * q_d2_0[None, :] # [block1 x BLOCK_T]
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_1 = qT_d1 * q_d2_1[None, :] # [block1 x BLOCK_T]
                q_d2_2 = tl.load(p_q_d2_2, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_2 = qT_d1 * q_d2_2[None, :] # [block1 x BLOCK_T]
                q_d2_3 = tl.load(p_q_d2_3, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_3 = qT_d1 * q_d2_3[None, :] # [block1 x BLOCK_T]
                q_d2_4 = tl.load(p_q_d2_4, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_4 = qT_d1 * q_d2_4[None, :] # [block1 x BLOCK_T]
                q_d2_5 = tl.load(p_q_d2_5, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_5 = qT_d1 * q_d2_5[None, :] # [block1 x BLOCK_T]
                q_d2_6 = tl.load(p_q_d2_6, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_6 = qT_d1 * q_d2_6[None, :] # [block1 x BLOCK_T]
                q_d2_7 = tl.load(p_q_d2_7, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_7 = qT_d1 * q_d2_7[None, :] # [block1 x BLOCK_T]
                alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
                factor = 1 / alpha / l # [BLOCK_T]
                delta_factor = delta * factor # [BLOCK_T]
                do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
            
                ds_0 = tl.dot(phiqT_0, do, ds_0) # [block1 x BLOCK_E_VALID]
                dsk_0 += -tl.sum(phiqT_0 * delta_factor[None, :], 1) # [block1]
                ds_1 = tl.dot(phiqT_1, do, ds_1) # [block1 x BLOCK_E_VALID]
                dsk_1 += -tl.sum(phiqT_1 * delta_factor[None, :], 1) # [block1]
                ds_2 = tl.dot(phiqT_2, do, ds_2) # [block1 x BLOCK_E_VALID]
                dsk_2 += -tl.sum(phiqT_2 * delta_factor[None, :], 1) # [block1]
                ds_3 = tl.dot(phiqT_3, do, ds_3) # [block1 x BLOCK_E_VALID]
                dsk_3 += -tl.sum(phiqT_3 * delta_factor[None, :], 1) # [block1]
                ds_4 = tl.dot(phiqT_4, do, ds_4) # [block1 x BLOCK_E_VALID]
                dsk_4 += -tl.sum(phiqT_4 * delta_factor[None, :], 1) # [block1]
                ds_5 = tl.dot(phiqT_5, do, ds_5) # [block1 x BLOCK_E_VALID]
                dsk_5 += -tl.sum(phiqT_5 * delta_factor[None, :], 1) # [block1]
                ds_6 = tl.dot(phiqT_6, do, ds_6) # [block1 x BLOCK_E_VALID]
                dsk_6 += -tl.sum(phiqT_6 * delta_factor[None, :], 1) # [block1]
                ds_7 = tl.dot(phiqT_7, do, ds_7) # [block1 x BLOCK_E_VALID]
                dsk_7 += -tl.sum(phiqT_7 * delta_factor[None, :], 1) # [block1]
                p_do += BLOCK_T * stride_dot
                p_qT_d1 += BLOCK_T * stride_qt
                p_q_d2_0 += BLOCK_T * stride_qt
                p_q_d2_1 += BLOCK_T * stride_qt
                p_q_d2_2 += BLOCK_T * stride_qt
                p_q_d2_3 += BLOCK_T * stride_qt
                p_q_d2_4 += BLOCK_T * stride_qt
                p_q_d2_5 += BLOCK_T * stride_qt
                p_q_d2_6 += BLOCK_T * stride_qt
                p_q_d2_7 += BLOCK_T * stride_qt
                
            
            range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
            p_dsk_0 = dSK + range_d2_0 * stride_dskD # [block1]
            tl.store(p_dsk_0, dsk_0 * scale_p)
            p_ds_0 = dS + range_d2_0[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_0, ds_0 * scale_p)
            range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
            p_dsk_1 = dSK + range_d2_1 * stride_dskD # [block1]
            tl.store(p_dsk_1, dsk_1 * scale_p)
            p_ds_1 = dS + range_d2_1[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_1, ds_1 * scale_p)
            range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
            p_dsk_2 = dSK + range_d2_2 * stride_dskD # [block1]
            tl.store(p_dsk_2, dsk_2 * scale_p)
            p_ds_2 = dS + range_d2_2[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_2, ds_2 * scale_p)
            range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
            p_dsk_3 = dSK + range_d2_3 * stride_dskD # [block1]
            tl.store(p_dsk_3, dsk_3 * scale_p)
            p_ds_3 = dS + range_d2_3[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_3, ds_3 * scale_p)
            range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
            p_dsk_4 = dSK + range_d2_4 * stride_dskD # [block1]
            tl.store(p_dsk_4, dsk_4 * scale_p)
            p_ds_4 = dS + range_d2_4[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_4, ds_4 * scale_p)
            range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
            p_dsk_5 = dSK + range_d2_5 * stride_dskD # [block1]
            tl.store(p_dsk_5, dsk_5 * scale_p)
            p_ds_5 = dS + range_d2_5[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_5, ds_5 * scale_p)
            range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
            p_dsk_6 = dSK + range_d2_6 * stride_dskD # [block1]
            tl.store(p_dsk_6, dsk_6 * scale_p)
            p_ds_6 = dS + range_d2_6[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_6, ds_6 * scale_p)
            range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
            p_dsk_7 = dSK + range_d2_7 * stride_dskD # [block1]
            tl.store(p_dsk_7, dsk_7 * scale_p)
            p_ds_7 = dS + range_d2_7[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_7, ds_7 * scale_p)
            
        
        elif (d == 64):     
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
            
            p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
            p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
            p_q_d2_2 = Q + range_t[:] * stride_qt + (off_d2 + 2) * stride_qd # [BLOCK_T]
            p_q_d2_3 = Q + range_t[:] * stride_qt + (off_d2 + 3) * stride_qd # [BLOCK_T]
            p_q_d2_4 = Q + range_t[:] * stride_qt + (off_d2 + 4) * stride_qd # [BLOCK_T]
            p_q_d2_5 = Q + range_t[:] * stride_qt + (off_d2 + 5) * stride_qd # [BLOCK_T]
            p_q_d2_6 = Q + range_t[:] * stride_qt + (off_d2 + 6) * stride_qd # [BLOCK_T]
            p_q_d2_7 = Q + range_t[:] * stride_qt + (off_d2 + 7) * stride_qd # [BLOCK_T]
            ds_0 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_0 = tl.zeros((block1,), dtype=tl.float32)
            ds_1 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_1 = tl.zeros((block1,), dtype=tl.float32)
            ds_2 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_2 = tl.zeros((block1,), dtype=tl.float32)
            ds_3 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_3 = tl.zeros((block1,), dtype=tl.float32)
            ds_4 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_4 = tl.zeros((block1,), dtype=tl.float32)
            ds_5 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_5 = tl.zeros((block1,), dtype=tl.float32)
            ds_6 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_6 = tl.zeros((block1,), dtype=tl.float32)
            ds_7 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_7 = tl.zeros((block1,), dtype=tl.float32)
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
            
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_0 = qT_d1 * q_d2_0[None, :] # [block1 x BLOCK_T]
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_1 = qT_d1 * q_d2_1[None, :] # [block1 x BLOCK_T]
                q_d2_2 = tl.load(p_q_d2_2, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_2 = qT_d1 * q_d2_2[None, :] # [block1 x BLOCK_T]
                q_d2_3 = tl.load(p_q_d2_3, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_3 = qT_d1 * q_d2_3[None, :] # [block1 x BLOCK_T]
                q_d2_4 = tl.load(p_q_d2_4, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_4 = qT_d1 * q_d2_4[None, :] # [block1 x BLOCK_T]
                q_d2_5 = tl.load(p_q_d2_5, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_5 = qT_d1 * q_d2_5[None, :] # [block1 x BLOCK_T]
                q_d2_6 = tl.load(p_q_d2_6, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_6 = qT_d1 * q_d2_6[None, :] # [block1 x BLOCK_T]
                q_d2_7 = tl.load(p_q_d2_7, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_7 = qT_d1 * q_d2_7[None, :] # [block1 x BLOCK_T]
                alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
                factor = 1 / alpha / l # [BLOCK_T]
                delta_factor = delta * factor # [BLOCK_T]
                do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
            
                ds_0 = tl.dot(phiqT_0, do, ds_0) # [block1 x BLOCK_E_VALID]
                dsk_0 += -tl.sum(phiqT_0 * delta_factor[None, :], 1) # [block1]
                ds_1 = tl.dot(phiqT_1, do, ds_1) # [block1 x BLOCK_E_VALID]
                dsk_1 += -tl.sum(phiqT_1 * delta_factor[None, :], 1) # [block1]
                ds_2 = tl.dot(phiqT_2, do, ds_2) # [block1 x BLOCK_E_VALID]
                dsk_2 += -tl.sum(phiqT_2 * delta_factor[None, :], 1) # [block1]
                ds_3 = tl.dot(phiqT_3, do, ds_3) # [block1 x BLOCK_E_VALID]
                dsk_3 += -tl.sum(phiqT_3 * delta_factor[None, :], 1) # [block1]
                ds_4 = tl.dot(phiqT_4, do, ds_4) # [block1 x BLOCK_E_VALID]
                dsk_4 += -tl.sum(phiqT_4 * delta_factor[None, :], 1) # [block1]
                ds_5 = tl.dot(phiqT_5, do, ds_5) # [block1 x BLOCK_E_VALID]
                dsk_5 += -tl.sum(phiqT_5 * delta_factor[None, :], 1) # [block1]
                ds_6 = tl.dot(phiqT_6, do, ds_6) # [block1 x BLOCK_E_VALID]
                dsk_6 += -tl.sum(phiqT_6 * delta_factor[None, :], 1) # [block1]
                ds_7 = tl.dot(phiqT_7, do, ds_7) # [block1 x BLOCK_E_VALID]
                dsk_7 += -tl.sum(phiqT_7 * delta_factor[None, :], 1) # [block1]
                p_do += BLOCK_T * stride_dot
                p_qT_d1 += BLOCK_T * stride_qt
                p_q_d2_0 += BLOCK_T * stride_qt
                p_q_d2_1 += BLOCK_T * stride_qt
                p_q_d2_2 += BLOCK_T * stride_qt
                p_q_d2_3 += BLOCK_T * stride_qt
                p_q_d2_4 += BLOCK_T * stride_qt
                p_q_d2_5 += BLOCK_T * stride_qt
                p_q_d2_6 += BLOCK_T * stride_qt
                p_q_d2_7 += BLOCK_T * stride_qt
                
            
            range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
            p_dsk_0 = dSK + range_d2_0 * stride_dskD # [block1]
            tl.store(p_dsk_0, dsk_0 * scale_p)
            p_ds_0 = dS + range_d2_0[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_0, ds_0 * scale_p)
            range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
            p_dsk_1 = dSK + range_d2_1 * stride_dskD # [block1]
            tl.store(p_dsk_1, dsk_1 * scale_p)
            p_ds_1 = dS + range_d2_1[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_1, ds_1 * scale_p)
            range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
            p_dsk_2 = dSK + range_d2_2 * stride_dskD # [block1]
            tl.store(p_dsk_2, dsk_2 * scale_p)
            p_ds_2 = dS + range_d2_2[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_2, ds_2 * scale_p)
            range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
            p_dsk_3 = dSK + range_d2_3 * stride_dskD # [block1]
            tl.store(p_dsk_3, dsk_3 * scale_p)
            p_ds_3 = dS + range_d2_3[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_3, ds_3 * scale_p)
            range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
            p_dsk_4 = dSK + range_d2_4 * stride_dskD # [block1]
            tl.store(p_dsk_4, dsk_4 * scale_p)
            p_ds_4 = dS + range_d2_4[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_4, ds_4 * scale_p)
            range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
            p_dsk_5 = dSK + range_d2_5 * stride_dskD # [block1]
            tl.store(p_dsk_5, dsk_5 * scale_p)
            p_ds_5 = dS + range_d2_5[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_5, ds_5 * scale_p)
            range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
            p_dsk_6 = dSK + range_d2_6 * stride_dskD # [block1]
            tl.store(p_dsk_6, dsk_6 * scale_p)
            p_ds_6 = dS + range_d2_6[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_6, ds_6 * scale_p)
            range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
            p_dsk_7 = dSK + range_d2_7 * stride_dskD # [block1]
            tl.store(p_dsk_7, dsk_7 * scale_p)
            p_ds_7 = dS + range_d2_7[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_7, ds_7 * scale_p)
            
        
        elif (d == 128):     
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
            
            p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
            p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
            p_q_d2_2 = Q + range_t[:] * stride_qt + (off_d2 + 2) * stride_qd # [BLOCK_T]
            p_q_d2_3 = Q + range_t[:] * stride_qt + (off_d2 + 3) * stride_qd # [BLOCK_T]
            p_q_d2_4 = Q + range_t[:] * stride_qt + (off_d2 + 4) * stride_qd # [BLOCK_T]
            p_q_d2_5 = Q + range_t[:] * stride_qt + (off_d2 + 5) * stride_qd # [BLOCK_T]
            p_q_d2_6 = Q + range_t[:] * stride_qt + (off_d2 + 6) * stride_qd # [BLOCK_T]
            p_q_d2_7 = Q + range_t[:] * stride_qt + (off_d2 + 7) * stride_qd # [BLOCK_T]
            ds_0 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_0 = tl.zeros((block1,), dtype=tl.float32)
            ds_1 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_1 = tl.zeros((block1,), dtype=tl.float32)
            ds_2 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_2 = tl.zeros((block1,), dtype=tl.float32)
            ds_3 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_3 = tl.zeros((block1,), dtype=tl.float32)
            ds_4 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_4 = tl.zeros((block1,), dtype=tl.float32)
            ds_5 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_5 = tl.zeros((block1,), dtype=tl.float32)
            ds_6 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_6 = tl.zeros((block1,), dtype=tl.float32)
            ds_7 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_7 = tl.zeros((block1,), dtype=tl.float32)
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
            
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_0 = qT_d1 * q_d2_0[None, :] # [block1 x BLOCK_T]
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_1 = qT_d1 * q_d2_1[None, :] # [block1 x BLOCK_T]
                q_d2_2 = tl.load(p_q_d2_2, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_2 = qT_d1 * q_d2_2[None, :] # [block1 x BLOCK_T]
                q_d2_3 = tl.load(p_q_d2_3, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_3 = qT_d1 * q_d2_3[None, :] # [block1 x BLOCK_T]
                q_d2_4 = tl.load(p_q_d2_4, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_4 = qT_d1 * q_d2_4[None, :] # [block1 x BLOCK_T]
                q_d2_5 = tl.load(p_q_d2_5, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_5 = qT_d1 * q_d2_5[None, :] # [block1 x BLOCK_T]
                q_d2_6 = tl.load(p_q_d2_6, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_6 = qT_d1 * q_d2_6[None, :] # [block1 x BLOCK_T]
                q_d2_7 = tl.load(p_q_d2_7, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_7 = qT_d1 * q_d2_7[None, :] # [block1 x BLOCK_T]
                alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
                factor = 1 / alpha / l # [BLOCK_T]
                delta_factor = delta * factor # [BLOCK_T]
                do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
            
                ds_0 = tl.dot(phiqT_0, do, ds_0) # [block1 x BLOCK_E_VALID]
                dsk_0 += -tl.sum(phiqT_0 * delta_factor[None, :], 1) # [block1]
                ds_1 = tl.dot(phiqT_1, do, ds_1) # [block1 x BLOCK_E_VALID]
                dsk_1 += -tl.sum(phiqT_1 * delta_factor[None, :], 1) # [block1]
                ds_2 = tl.dot(phiqT_2, do, ds_2) # [block1 x BLOCK_E_VALID]
                dsk_2 += -tl.sum(phiqT_2 * delta_factor[None, :], 1) # [block1]
                ds_3 = tl.dot(phiqT_3, do, ds_3) # [block1 x BLOCK_E_VALID]
                dsk_3 += -tl.sum(phiqT_3 * delta_factor[None, :], 1) # [block1]
                ds_4 = tl.dot(phiqT_4, do, ds_4) # [block1 x BLOCK_E_VALID]
                dsk_4 += -tl.sum(phiqT_4 * delta_factor[None, :], 1) # [block1]
                ds_5 = tl.dot(phiqT_5, do, ds_5) # [block1 x BLOCK_E_VALID]
                dsk_5 += -tl.sum(phiqT_5 * delta_factor[None, :], 1) # [block1]
                ds_6 = tl.dot(phiqT_6, do, ds_6) # [block1 x BLOCK_E_VALID]
                dsk_6 += -tl.sum(phiqT_6 * delta_factor[None, :], 1) # [block1]
                ds_7 = tl.dot(phiqT_7, do, ds_7) # [block1 x BLOCK_E_VALID]
                dsk_7 += -tl.sum(phiqT_7 * delta_factor[None, :], 1) # [block1]
                p_do += BLOCK_T * stride_dot
                p_qT_d1 += BLOCK_T * stride_qt
                p_q_d2_0 += BLOCK_T * stride_qt
                p_q_d2_1 += BLOCK_T * stride_qt
                p_q_d2_2 += BLOCK_T * stride_qt
                p_q_d2_3 += BLOCK_T * stride_qt
                p_q_d2_4 += BLOCK_T * stride_qt
                p_q_d2_5 += BLOCK_T * stride_qt
                p_q_d2_6 += BLOCK_T * stride_qt
                p_q_d2_7 += BLOCK_T * stride_qt
                
            
            range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
            p_dsk_0 = dSK + range_d2_0 * stride_dskD # [block1]
            tl.store(p_dsk_0, dsk_0 * scale_p)
            p_ds_0 = dS + range_d2_0[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_0, ds_0 * scale_p)
            range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
            p_dsk_1 = dSK + range_d2_1 * stride_dskD # [block1]
            tl.store(p_dsk_1, dsk_1 * scale_p)
            p_ds_1 = dS + range_d2_1[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_1, ds_1 * scale_p)
            range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
            p_dsk_2 = dSK + range_d2_2 * stride_dskD # [block1]
            tl.store(p_dsk_2, dsk_2 * scale_p)
            p_ds_2 = dS + range_d2_2[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_2, ds_2 * scale_p)
            range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
            p_dsk_3 = dSK + range_d2_3 * stride_dskD # [block1]
            tl.store(p_dsk_3, dsk_3 * scale_p)
            p_ds_3 = dS + range_d2_3[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_3, ds_3 * scale_p)
            range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
            p_dsk_4 = dSK + range_d2_4 * stride_dskD # [block1]
            tl.store(p_dsk_4, dsk_4 * scale_p)
            p_ds_4 = dS + range_d2_4[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_4, ds_4 * scale_p)
            range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
            p_dsk_5 = dSK + range_d2_5 * stride_dskD # [block1]
            tl.store(p_dsk_5, dsk_5 * scale_p)
            p_ds_5 = dS + range_d2_5[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_5, ds_5 * scale_p)
            range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
            p_dsk_6 = dSK + range_d2_6 * stride_dskD # [block1]
            tl.store(p_dsk_6, dsk_6 * scale_p)
            p_ds_6 = dS + range_d2_6[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_6, ds_6 * scale_p)
            range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
            p_dsk_7 = dSK + range_d2_7 * stride_dskD # [block1]
            tl.store(p_dsk_7, dsk_7 * scale_p)
            p_ds_7 = dS + range_d2_7[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_7, ds_7 * scale_p)
            
    elif ((BLOCK_D == 256) and ((BLOCK_E == 32) and ((BLOCK_T == 16) and ((block1 == 16))))) or (((BLOCK_D == 256) and ((BLOCK_E == 32) and ((BLOCK_T == 32) and ((block1 == 16))))) or (((BLOCK_D == 256) and ((BLOCK_E == 64) and ((BLOCK_T == 16) and ((block1 == 16))))) or ((BLOCK_D == 256) and ((BLOCK_E == 64) and ((BLOCK_T == 32) and ((block1 == 16))))))):
        
        if (d == 32):     
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
            
            p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
            p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
            p_q_d2_2 = Q + range_t[:] * stride_qt + (off_d2 + 2) * stride_qd # [BLOCK_T]
            p_q_d2_3 = Q + range_t[:] * stride_qt + (off_d2 + 3) * stride_qd # [BLOCK_T]
            p_q_d2_4 = Q + range_t[:] * stride_qt + (off_d2 + 4) * stride_qd # [BLOCK_T]
            p_q_d2_5 = Q + range_t[:] * stride_qt + (off_d2 + 5) * stride_qd # [BLOCK_T]
            p_q_d2_6 = Q + range_t[:] * stride_qt + (off_d2 + 6) * stride_qd # [BLOCK_T]
            p_q_d2_7 = Q + range_t[:] * stride_qt + (off_d2 + 7) * stride_qd # [BLOCK_T]
            p_q_d2_8 = Q + range_t[:] * stride_qt + (off_d2 + 8) * stride_qd # [BLOCK_T]
            p_q_d2_9 = Q + range_t[:] * stride_qt + (off_d2 + 9) * stride_qd # [BLOCK_T]
            p_q_d2_10 = Q + range_t[:] * stride_qt + (off_d2 + 10) * stride_qd # [BLOCK_T]
            p_q_d2_11 = Q + range_t[:] * stride_qt + (off_d2 + 11) * stride_qd # [BLOCK_T]
            p_q_d2_12 = Q + range_t[:] * stride_qt + (off_d2 + 12) * stride_qd # [BLOCK_T]
            p_q_d2_13 = Q + range_t[:] * stride_qt + (off_d2 + 13) * stride_qd # [BLOCK_T]
            p_q_d2_14 = Q + range_t[:] * stride_qt + (off_d2 + 14) * stride_qd # [BLOCK_T]
            p_q_d2_15 = Q + range_t[:] * stride_qt + (off_d2 + 15) * stride_qd # [BLOCK_T]
            ds_0 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_0 = tl.zeros((block1,), dtype=tl.float32)
            ds_1 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_1 = tl.zeros((block1,), dtype=tl.float32)
            ds_2 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_2 = tl.zeros((block1,), dtype=tl.float32)
            ds_3 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_3 = tl.zeros((block1,), dtype=tl.float32)
            ds_4 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_4 = tl.zeros((block1,), dtype=tl.float32)
            ds_5 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_5 = tl.zeros((block1,), dtype=tl.float32)
            ds_6 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_6 = tl.zeros((block1,), dtype=tl.float32)
            ds_7 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_7 = tl.zeros((block1,), dtype=tl.float32)
            ds_8 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_8 = tl.zeros((block1,), dtype=tl.float32)
            ds_9 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_9 = tl.zeros((block1,), dtype=tl.float32)
            ds_10 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_10 = tl.zeros((block1,), dtype=tl.float32)
            ds_11 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_11 = tl.zeros((block1,), dtype=tl.float32)
            ds_12 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_12 = tl.zeros((block1,), dtype=tl.float32)
            ds_13 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_13 = tl.zeros((block1,), dtype=tl.float32)
            ds_14 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_14 = tl.zeros((block1,), dtype=tl.float32)
            ds_15 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_15 = tl.zeros((block1,), dtype=tl.float32)
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
            
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_0 = qT_d1 * q_d2_0[None, :] # [block1 x BLOCK_T]
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_1 = qT_d1 * q_d2_1[None, :] # [block1 x BLOCK_T]
                q_d2_2 = tl.load(p_q_d2_2, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_2 = qT_d1 * q_d2_2[None, :] # [block1 x BLOCK_T]
                q_d2_3 = tl.load(p_q_d2_3, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_3 = qT_d1 * q_d2_3[None, :] # [block1 x BLOCK_T]
                q_d2_4 = tl.load(p_q_d2_4, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_4 = qT_d1 * q_d2_4[None, :] # [block1 x BLOCK_T]
                q_d2_5 = tl.load(p_q_d2_5, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_5 = qT_d1 * q_d2_5[None, :] # [block1 x BLOCK_T]
                q_d2_6 = tl.load(p_q_d2_6, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_6 = qT_d1 * q_d2_6[None, :] # [block1 x BLOCK_T]
                q_d2_7 = tl.load(p_q_d2_7, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_7 = qT_d1 * q_d2_7[None, :] # [block1 x BLOCK_T]
                q_d2_8 = tl.load(p_q_d2_8, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_8 = qT_d1 * q_d2_8[None, :] # [block1 x BLOCK_T]
                q_d2_9 = tl.load(p_q_d2_9, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_9 = qT_d1 * q_d2_9[None, :] # [block1 x BLOCK_T]
                q_d2_10 = tl.load(p_q_d2_10, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_10 = qT_d1 * q_d2_10[None, :] # [block1 x BLOCK_T]
                q_d2_11 = tl.load(p_q_d2_11, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_11 = qT_d1 * q_d2_11[None, :] # [block1 x BLOCK_T]
                q_d2_12 = tl.load(p_q_d2_12, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_12 = qT_d1 * q_d2_12[None, :] # [block1 x BLOCK_T]
                q_d2_13 = tl.load(p_q_d2_13, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_13 = qT_d1 * q_d2_13[None, :] # [block1 x BLOCK_T]
                q_d2_14 = tl.load(p_q_d2_14, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_14 = qT_d1 * q_d2_14[None, :] # [block1 x BLOCK_T]
                q_d2_15 = tl.load(p_q_d2_15, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_15 = qT_d1 * q_d2_15[None, :] # [block1 x BLOCK_T]
                alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
                factor = 1 / alpha / l # [BLOCK_T]
                delta_factor = delta * factor # [BLOCK_T]
                do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
            
                ds_0 = tl.dot(phiqT_0, do, ds_0) # [block1 x BLOCK_E_VALID]
                dsk_0 += -tl.sum(phiqT_0 * delta_factor[None, :], 1) # [block1]
                ds_1 = tl.dot(phiqT_1, do, ds_1) # [block1 x BLOCK_E_VALID]
                dsk_1 += -tl.sum(phiqT_1 * delta_factor[None, :], 1) # [block1]
                ds_2 = tl.dot(phiqT_2, do, ds_2) # [block1 x BLOCK_E_VALID]
                dsk_2 += -tl.sum(phiqT_2 * delta_factor[None, :], 1) # [block1]
                ds_3 = tl.dot(phiqT_3, do, ds_3) # [block1 x BLOCK_E_VALID]
                dsk_3 += -tl.sum(phiqT_3 * delta_factor[None, :], 1) # [block1]
                ds_4 = tl.dot(phiqT_4, do, ds_4) # [block1 x BLOCK_E_VALID]
                dsk_4 += -tl.sum(phiqT_4 * delta_factor[None, :], 1) # [block1]
                ds_5 = tl.dot(phiqT_5, do, ds_5) # [block1 x BLOCK_E_VALID]
                dsk_5 += -tl.sum(phiqT_5 * delta_factor[None, :], 1) # [block1]
                ds_6 = tl.dot(phiqT_6, do, ds_6) # [block1 x BLOCK_E_VALID]
                dsk_6 += -tl.sum(phiqT_6 * delta_factor[None, :], 1) # [block1]
                ds_7 = tl.dot(phiqT_7, do, ds_7) # [block1 x BLOCK_E_VALID]
                dsk_7 += -tl.sum(phiqT_7 * delta_factor[None, :], 1) # [block1]
                ds_8 = tl.dot(phiqT_8, do, ds_8) # [block1 x BLOCK_E_VALID]
                dsk_8 += -tl.sum(phiqT_8 * delta_factor[None, :], 1) # [block1]
                ds_9 = tl.dot(phiqT_9, do, ds_9) # [block1 x BLOCK_E_VALID]
                dsk_9 += -tl.sum(phiqT_9 * delta_factor[None, :], 1) # [block1]
                ds_10 = tl.dot(phiqT_10, do, ds_10) # [block1 x BLOCK_E_VALID]
                dsk_10 += -tl.sum(phiqT_10 * delta_factor[None, :], 1) # [block1]
                ds_11 = tl.dot(phiqT_11, do, ds_11) # [block1 x BLOCK_E_VALID]
                dsk_11 += -tl.sum(phiqT_11 * delta_factor[None, :], 1) # [block1]
                ds_12 = tl.dot(phiqT_12, do, ds_12) # [block1 x BLOCK_E_VALID]
                dsk_12 += -tl.sum(phiqT_12 * delta_factor[None, :], 1) # [block1]
                ds_13 = tl.dot(phiqT_13, do, ds_13) # [block1 x BLOCK_E_VALID]
                dsk_13 += -tl.sum(phiqT_13 * delta_factor[None, :], 1) # [block1]
                ds_14 = tl.dot(phiqT_14, do, ds_14) # [block1 x BLOCK_E_VALID]
                dsk_14 += -tl.sum(phiqT_14 * delta_factor[None, :], 1) # [block1]
                ds_15 = tl.dot(phiqT_15, do, ds_15) # [block1 x BLOCK_E_VALID]
                dsk_15 += -tl.sum(phiqT_15 * delta_factor[None, :], 1) # [block1]
                p_do += BLOCK_T * stride_dot
                p_qT_d1 += BLOCK_T * stride_qt
                p_q_d2_0 += BLOCK_T * stride_qt
                p_q_d2_1 += BLOCK_T * stride_qt
                p_q_d2_2 += BLOCK_T * stride_qt
                p_q_d2_3 += BLOCK_T * stride_qt
                p_q_d2_4 += BLOCK_T * stride_qt
                p_q_d2_5 += BLOCK_T * stride_qt
                p_q_d2_6 += BLOCK_T * stride_qt
                p_q_d2_7 += BLOCK_T * stride_qt
                p_q_d2_8 += BLOCK_T * stride_qt
                p_q_d2_9 += BLOCK_T * stride_qt
                p_q_d2_10 += BLOCK_T * stride_qt
                p_q_d2_11 += BLOCK_T * stride_qt
                p_q_d2_12 += BLOCK_T * stride_qt
                p_q_d2_13 += BLOCK_T * stride_qt
                p_q_d2_14 += BLOCK_T * stride_qt
                p_q_d2_15 += BLOCK_T * stride_qt
                
            
            range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
            p_dsk_0 = dSK + range_d2_0 * stride_dskD # [block1]
            tl.store(p_dsk_0, dsk_0 * scale_p)
            p_ds_0 = dS + range_d2_0[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_0, ds_0 * scale_p)
            range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
            p_dsk_1 = dSK + range_d2_1 * stride_dskD # [block1]
            tl.store(p_dsk_1, dsk_1 * scale_p)
            p_ds_1 = dS + range_d2_1[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_1, ds_1 * scale_p)
            range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
            p_dsk_2 = dSK + range_d2_2 * stride_dskD # [block1]
            tl.store(p_dsk_2, dsk_2 * scale_p)
            p_ds_2 = dS + range_d2_2[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_2, ds_2 * scale_p)
            range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
            p_dsk_3 = dSK + range_d2_3 * stride_dskD # [block1]
            tl.store(p_dsk_3, dsk_3 * scale_p)
            p_ds_3 = dS + range_d2_3[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_3, ds_3 * scale_p)
            range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
            p_dsk_4 = dSK + range_d2_4 * stride_dskD # [block1]
            tl.store(p_dsk_4, dsk_4 * scale_p)
            p_ds_4 = dS + range_d2_4[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_4, ds_4 * scale_p)
            range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
            p_dsk_5 = dSK + range_d2_5 * stride_dskD # [block1]
            tl.store(p_dsk_5, dsk_5 * scale_p)
            p_ds_5 = dS + range_d2_5[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_5, ds_5 * scale_p)
            range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
            p_dsk_6 = dSK + range_d2_6 * stride_dskD # [block1]
            tl.store(p_dsk_6, dsk_6 * scale_p)
            p_ds_6 = dS + range_d2_6[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_6, ds_6 * scale_p)
            range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
            p_dsk_7 = dSK + range_d2_7 * stride_dskD # [block1]
            tl.store(p_dsk_7, dsk_7 * scale_p)
            p_ds_7 = dS + range_d2_7[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_7, ds_7 * scale_p)
            range_d2_8 = tl.arange(0, block1).to(tl.int64) + 8 * block1
            p_dsk_8 = dSK + range_d2_8 * stride_dskD # [block1]
            tl.store(p_dsk_8, dsk_8 * scale_p)
            p_ds_8 = dS + range_d2_8[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_8, ds_8 * scale_p)
            range_d2_9 = tl.arange(0, block1).to(tl.int64) + 9 * block1
            p_dsk_9 = dSK + range_d2_9 * stride_dskD # [block1]
            tl.store(p_dsk_9, dsk_9 * scale_p)
            p_ds_9 = dS + range_d2_9[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_9, ds_9 * scale_p)
            range_d2_10 = tl.arange(0, block1).to(tl.int64) + 10 * block1
            p_dsk_10 = dSK + range_d2_10 * stride_dskD # [block1]
            tl.store(p_dsk_10, dsk_10 * scale_p)
            p_ds_10 = dS + range_d2_10[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_10, ds_10 * scale_p)
            range_d2_11 = tl.arange(0, block1).to(tl.int64) + 11 * block1
            p_dsk_11 = dSK + range_d2_11 * stride_dskD # [block1]
            tl.store(p_dsk_11, dsk_11 * scale_p)
            p_ds_11 = dS + range_d2_11[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_11, ds_11 * scale_p)
            range_d2_12 = tl.arange(0, block1).to(tl.int64) + 12 * block1
            p_dsk_12 = dSK + range_d2_12 * stride_dskD # [block1]
            tl.store(p_dsk_12, dsk_12 * scale_p)
            p_ds_12 = dS + range_d2_12[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_12, ds_12 * scale_p)
            range_d2_13 = tl.arange(0, block1).to(tl.int64) + 13 * block1
            p_dsk_13 = dSK + range_d2_13 * stride_dskD # [block1]
            tl.store(p_dsk_13, dsk_13 * scale_p)
            p_ds_13 = dS + range_d2_13[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_13, ds_13 * scale_p)
            range_d2_14 = tl.arange(0, block1).to(tl.int64) + 14 * block1
            p_dsk_14 = dSK + range_d2_14 * stride_dskD # [block1]
            tl.store(p_dsk_14, dsk_14 * scale_p)
            p_ds_14 = dS + range_d2_14[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_14, ds_14 * scale_p)
            range_d2_15 = tl.arange(0, block1).to(tl.int64) + 15 * block1
            p_dsk_15 = dSK + range_d2_15 * stride_dskD # [block1]
            tl.store(p_dsk_15, dsk_15 * scale_p)
            p_ds_15 = dS + range_d2_15[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_15, ds_15 * scale_p)
            
        
        elif (d == 64):     
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
            
            p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
            p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
            p_q_d2_2 = Q + range_t[:] * stride_qt + (off_d2 + 2) * stride_qd # [BLOCK_T]
            p_q_d2_3 = Q + range_t[:] * stride_qt + (off_d2 + 3) * stride_qd # [BLOCK_T]
            p_q_d2_4 = Q + range_t[:] * stride_qt + (off_d2 + 4) * stride_qd # [BLOCK_T]
            p_q_d2_5 = Q + range_t[:] * stride_qt + (off_d2 + 5) * stride_qd # [BLOCK_T]
            p_q_d2_6 = Q + range_t[:] * stride_qt + (off_d2 + 6) * stride_qd # [BLOCK_T]
            p_q_d2_7 = Q + range_t[:] * stride_qt + (off_d2 + 7) * stride_qd # [BLOCK_T]
            p_q_d2_8 = Q + range_t[:] * stride_qt + (off_d2 + 8) * stride_qd # [BLOCK_T]
            p_q_d2_9 = Q + range_t[:] * stride_qt + (off_d2 + 9) * stride_qd # [BLOCK_T]
            p_q_d2_10 = Q + range_t[:] * stride_qt + (off_d2 + 10) * stride_qd # [BLOCK_T]
            p_q_d2_11 = Q + range_t[:] * stride_qt + (off_d2 + 11) * stride_qd # [BLOCK_T]
            p_q_d2_12 = Q + range_t[:] * stride_qt + (off_d2 + 12) * stride_qd # [BLOCK_T]
            p_q_d2_13 = Q + range_t[:] * stride_qt + (off_d2 + 13) * stride_qd # [BLOCK_T]
            p_q_d2_14 = Q + range_t[:] * stride_qt + (off_d2 + 14) * stride_qd # [BLOCK_T]
            p_q_d2_15 = Q + range_t[:] * stride_qt + (off_d2 + 15) * stride_qd # [BLOCK_T]
            ds_0 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_0 = tl.zeros((block1,), dtype=tl.float32)
            ds_1 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_1 = tl.zeros((block1,), dtype=tl.float32)
            ds_2 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_2 = tl.zeros((block1,), dtype=tl.float32)
            ds_3 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_3 = tl.zeros((block1,), dtype=tl.float32)
            ds_4 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_4 = tl.zeros((block1,), dtype=tl.float32)
            ds_5 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_5 = tl.zeros((block1,), dtype=tl.float32)
            ds_6 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_6 = tl.zeros((block1,), dtype=tl.float32)
            ds_7 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_7 = tl.zeros((block1,), dtype=tl.float32)
            ds_8 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_8 = tl.zeros((block1,), dtype=tl.float32)
            ds_9 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_9 = tl.zeros((block1,), dtype=tl.float32)
            ds_10 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_10 = tl.zeros((block1,), dtype=tl.float32)
            ds_11 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_11 = tl.zeros((block1,), dtype=tl.float32)
            ds_12 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_12 = tl.zeros((block1,), dtype=tl.float32)
            ds_13 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_13 = tl.zeros((block1,), dtype=tl.float32)
            ds_14 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_14 = tl.zeros((block1,), dtype=tl.float32)
            ds_15 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_15 = tl.zeros((block1,), dtype=tl.float32)
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
            
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_0 = qT_d1 * q_d2_0[None, :] # [block1 x BLOCK_T]
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_1 = qT_d1 * q_d2_1[None, :] # [block1 x BLOCK_T]
                q_d2_2 = tl.load(p_q_d2_2, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_2 = qT_d1 * q_d2_2[None, :] # [block1 x BLOCK_T]
                q_d2_3 = tl.load(p_q_d2_3, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_3 = qT_d1 * q_d2_3[None, :] # [block1 x BLOCK_T]
                q_d2_4 = tl.load(p_q_d2_4, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_4 = qT_d1 * q_d2_4[None, :] # [block1 x BLOCK_T]
                q_d2_5 = tl.load(p_q_d2_5, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_5 = qT_d1 * q_d2_5[None, :] # [block1 x BLOCK_T]
                q_d2_6 = tl.load(p_q_d2_6, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_6 = qT_d1 * q_d2_6[None, :] # [block1 x BLOCK_T]
                q_d2_7 = tl.load(p_q_d2_7, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_7 = qT_d1 * q_d2_7[None, :] # [block1 x BLOCK_T]
                q_d2_8 = tl.load(p_q_d2_8, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_8 = qT_d1 * q_d2_8[None, :] # [block1 x BLOCK_T]
                q_d2_9 = tl.load(p_q_d2_9, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_9 = qT_d1 * q_d2_9[None, :] # [block1 x BLOCK_T]
                q_d2_10 = tl.load(p_q_d2_10, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_10 = qT_d1 * q_d2_10[None, :] # [block1 x BLOCK_T]
                q_d2_11 = tl.load(p_q_d2_11, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_11 = qT_d1 * q_d2_11[None, :] # [block1 x BLOCK_T]
                q_d2_12 = tl.load(p_q_d2_12, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_12 = qT_d1 * q_d2_12[None, :] # [block1 x BLOCK_T]
                q_d2_13 = tl.load(p_q_d2_13, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_13 = qT_d1 * q_d2_13[None, :] # [block1 x BLOCK_T]
                q_d2_14 = tl.load(p_q_d2_14, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_14 = qT_d1 * q_d2_14[None, :] # [block1 x BLOCK_T]
                q_d2_15 = tl.load(p_q_d2_15, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_15 = qT_d1 * q_d2_15[None, :] # [block1 x BLOCK_T]
                alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
                factor = 1 / alpha / l # [BLOCK_T]
                delta_factor = delta * factor # [BLOCK_T]
                do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
            
                ds_0 = tl.dot(phiqT_0, do, ds_0) # [block1 x BLOCK_E_VALID]
                dsk_0 += -tl.sum(phiqT_0 * delta_factor[None, :], 1) # [block1]
                ds_1 = tl.dot(phiqT_1, do, ds_1) # [block1 x BLOCK_E_VALID]
                dsk_1 += -tl.sum(phiqT_1 * delta_factor[None, :], 1) # [block1]
                ds_2 = tl.dot(phiqT_2, do, ds_2) # [block1 x BLOCK_E_VALID]
                dsk_2 += -tl.sum(phiqT_2 * delta_factor[None, :], 1) # [block1]
                ds_3 = tl.dot(phiqT_3, do, ds_3) # [block1 x BLOCK_E_VALID]
                dsk_3 += -tl.sum(phiqT_3 * delta_factor[None, :], 1) # [block1]
                ds_4 = tl.dot(phiqT_4, do, ds_4) # [block1 x BLOCK_E_VALID]
                dsk_4 += -tl.sum(phiqT_4 * delta_factor[None, :], 1) # [block1]
                ds_5 = tl.dot(phiqT_5, do, ds_5) # [block1 x BLOCK_E_VALID]
                dsk_5 += -tl.sum(phiqT_5 * delta_factor[None, :], 1) # [block1]
                ds_6 = tl.dot(phiqT_6, do, ds_6) # [block1 x BLOCK_E_VALID]
                dsk_6 += -tl.sum(phiqT_6 * delta_factor[None, :], 1) # [block1]
                ds_7 = tl.dot(phiqT_7, do, ds_7) # [block1 x BLOCK_E_VALID]
                dsk_7 += -tl.sum(phiqT_7 * delta_factor[None, :], 1) # [block1]
                ds_8 = tl.dot(phiqT_8, do, ds_8) # [block1 x BLOCK_E_VALID]
                dsk_8 += -tl.sum(phiqT_8 * delta_factor[None, :], 1) # [block1]
                ds_9 = tl.dot(phiqT_9, do, ds_9) # [block1 x BLOCK_E_VALID]
                dsk_9 += -tl.sum(phiqT_9 * delta_factor[None, :], 1) # [block1]
                ds_10 = tl.dot(phiqT_10, do, ds_10) # [block1 x BLOCK_E_VALID]
                dsk_10 += -tl.sum(phiqT_10 * delta_factor[None, :], 1) # [block1]
                ds_11 = tl.dot(phiqT_11, do, ds_11) # [block1 x BLOCK_E_VALID]
                dsk_11 += -tl.sum(phiqT_11 * delta_factor[None, :], 1) # [block1]
                ds_12 = tl.dot(phiqT_12, do, ds_12) # [block1 x BLOCK_E_VALID]
                dsk_12 += -tl.sum(phiqT_12 * delta_factor[None, :], 1) # [block1]
                ds_13 = tl.dot(phiqT_13, do, ds_13) # [block1 x BLOCK_E_VALID]
                dsk_13 += -tl.sum(phiqT_13 * delta_factor[None, :], 1) # [block1]
                ds_14 = tl.dot(phiqT_14, do, ds_14) # [block1 x BLOCK_E_VALID]
                dsk_14 += -tl.sum(phiqT_14 * delta_factor[None, :], 1) # [block1]
                ds_15 = tl.dot(phiqT_15, do, ds_15) # [block1 x BLOCK_E_VALID]
                dsk_15 += -tl.sum(phiqT_15 * delta_factor[None, :], 1) # [block1]
                p_do += BLOCK_T * stride_dot
                p_qT_d1 += BLOCK_T * stride_qt
                p_q_d2_0 += BLOCK_T * stride_qt
                p_q_d2_1 += BLOCK_T * stride_qt
                p_q_d2_2 += BLOCK_T * stride_qt
                p_q_d2_3 += BLOCK_T * stride_qt
                p_q_d2_4 += BLOCK_T * stride_qt
                p_q_d2_5 += BLOCK_T * stride_qt
                p_q_d2_6 += BLOCK_T * stride_qt
                p_q_d2_7 += BLOCK_T * stride_qt
                p_q_d2_8 += BLOCK_T * stride_qt
                p_q_d2_9 += BLOCK_T * stride_qt
                p_q_d2_10 += BLOCK_T * stride_qt
                p_q_d2_11 += BLOCK_T * stride_qt
                p_q_d2_12 += BLOCK_T * stride_qt
                p_q_d2_13 += BLOCK_T * stride_qt
                p_q_d2_14 += BLOCK_T * stride_qt
                p_q_d2_15 += BLOCK_T * stride_qt
                
            
            range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
            p_dsk_0 = dSK + range_d2_0 * stride_dskD # [block1]
            tl.store(p_dsk_0, dsk_0 * scale_p)
            p_ds_0 = dS + range_d2_0[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_0, ds_0 * scale_p)
            range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
            p_dsk_1 = dSK + range_d2_1 * stride_dskD # [block1]
            tl.store(p_dsk_1, dsk_1 * scale_p)
            p_ds_1 = dS + range_d2_1[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_1, ds_1 * scale_p)
            range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
            p_dsk_2 = dSK + range_d2_2 * stride_dskD # [block1]
            tl.store(p_dsk_2, dsk_2 * scale_p)
            p_ds_2 = dS + range_d2_2[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_2, ds_2 * scale_p)
            range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
            p_dsk_3 = dSK + range_d2_3 * stride_dskD # [block1]
            tl.store(p_dsk_3, dsk_3 * scale_p)
            p_ds_3 = dS + range_d2_3[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_3, ds_3 * scale_p)
            range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
            p_dsk_4 = dSK + range_d2_4 * stride_dskD # [block1]
            tl.store(p_dsk_4, dsk_4 * scale_p)
            p_ds_4 = dS + range_d2_4[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_4, ds_4 * scale_p)
            range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
            p_dsk_5 = dSK + range_d2_5 * stride_dskD # [block1]
            tl.store(p_dsk_5, dsk_5 * scale_p)
            p_ds_5 = dS + range_d2_5[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_5, ds_5 * scale_p)
            range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
            p_dsk_6 = dSK + range_d2_6 * stride_dskD # [block1]
            tl.store(p_dsk_6, dsk_6 * scale_p)
            p_ds_6 = dS + range_d2_6[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_6, ds_6 * scale_p)
            range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
            p_dsk_7 = dSK + range_d2_7 * stride_dskD # [block1]
            tl.store(p_dsk_7, dsk_7 * scale_p)
            p_ds_7 = dS + range_d2_7[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_7, ds_7 * scale_p)
            range_d2_8 = tl.arange(0, block1).to(tl.int64) + 8 * block1
            p_dsk_8 = dSK + range_d2_8 * stride_dskD # [block1]
            tl.store(p_dsk_8, dsk_8 * scale_p)
            p_ds_8 = dS + range_d2_8[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_8, ds_8 * scale_p)
            range_d2_9 = tl.arange(0, block1).to(tl.int64) + 9 * block1
            p_dsk_9 = dSK + range_d2_9 * stride_dskD # [block1]
            tl.store(p_dsk_9, dsk_9 * scale_p)
            p_ds_9 = dS + range_d2_9[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_9, ds_9 * scale_p)
            range_d2_10 = tl.arange(0, block1).to(tl.int64) + 10 * block1
            p_dsk_10 = dSK + range_d2_10 * stride_dskD # [block1]
            tl.store(p_dsk_10, dsk_10 * scale_p)
            p_ds_10 = dS + range_d2_10[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_10, ds_10 * scale_p)
            range_d2_11 = tl.arange(0, block1).to(tl.int64) + 11 * block1
            p_dsk_11 = dSK + range_d2_11 * stride_dskD # [block1]
            tl.store(p_dsk_11, dsk_11 * scale_p)
            p_ds_11 = dS + range_d2_11[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_11, ds_11 * scale_p)
            range_d2_12 = tl.arange(0, block1).to(tl.int64) + 12 * block1
            p_dsk_12 = dSK + range_d2_12 * stride_dskD # [block1]
            tl.store(p_dsk_12, dsk_12 * scale_p)
            p_ds_12 = dS + range_d2_12[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_12, ds_12 * scale_p)
            range_d2_13 = tl.arange(0, block1).to(tl.int64) + 13 * block1
            p_dsk_13 = dSK + range_d2_13 * stride_dskD # [block1]
            tl.store(p_dsk_13, dsk_13 * scale_p)
            p_ds_13 = dS + range_d2_13[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_13, ds_13 * scale_p)
            range_d2_14 = tl.arange(0, block1).to(tl.int64) + 14 * block1
            p_dsk_14 = dSK + range_d2_14 * stride_dskD # [block1]
            tl.store(p_dsk_14, dsk_14 * scale_p)
            p_ds_14 = dS + range_d2_14[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_14, ds_14 * scale_p)
            range_d2_15 = tl.arange(0, block1).to(tl.int64) + 15 * block1
            p_dsk_15 = dSK + range_d2_15 * stride_dskD # [block1]
            tl.store(p_dsk_15, dsk_15 * scale_p)
            p_ds_15 = dS + range_d2_15[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_15, ds_15 * scale_p)
            
        
        elif (d == 128):     
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
            
            p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
            p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
            p_q_d2_2 = Q + range_t[:] * stride_qt + (off_d2 + 2) * stride_qd # [BLOCK_T]
            p_q_d2_3 = Q + range_t[:] * stride_qt + (off_d2 + 3) * stride_qd # [BLOCK_T]
            p_q_d2_4 = Q + range_t[:] * stride_qt + (off_d2 + 4) * stride_qd # [BLOCK_T]
            p_q_d2_5 = Q + range_t[:] * stride_qt + (off_d2 + 5) * stride_qd # [BLOCK_T]
            p_q_d2_6 = Q + range_t[:] * stride_qt + (off_d2 + 6) * stride_qd # [BLOCK_T]
            p_q_d2_7 = Q + range_t[:] * stride_qt + (off_d2 + 7) * stride_qd # [BLOCK_T]
            p_q_d2_8 = Q + range_t[:] * stride_qt + (off_d2 + 8) * stride_qd # [BLOCK_T]
            p_q_d2_9 = Q + range_t[:] * stride_qt + (off_d2 + 9) * stride_qd # [BLOCK_T]
            p_q_d2_10 = Q + range_t[:] * stride_qt + (off_d2 + 10) * stride_qd # [BLOCK_T]
            p_q_d2_11 = Q + range_t[:] * stride_qt + (off_d2 + 11) * stride_qd # [BLOCK_T]
            p_q_d2_12 = Q + range_t[:] * stride_qt + (off_d2 + 12) * stride_qd # [BLOCK_T]
            p_q_d2_13 = Q + range_t[:] * stride_qt + (off_d2 + 13) * stride_qd # [BLOCK_T]
            p_q_d2_14 = Q + range_t[:] * stride_qt + (off_d2 + 14) * stride_qd # [BLOCK_T]
            p_q_d2_15 = Q + range_t[:] * stride_qt + (off_d2 + 15) * stride_qd # [BLOCK_T]
            ds_0 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_0 = tl.zeros((block1,), dtype=tl.float32)
            ds_1 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_1 = tl.zeros((block1,), dtype=tl.float32)
            ds_2 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_2 = tl.zeros((block1,), dtype=tl.float32)
            ds_3 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_3 = tl.zeros((block1,), dtype=tl.float32)
            ds_4 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_4 = tl.zeros((block1,), dtype=tl.float32)
            ds_5 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_5 = tl.zeros((block1,), dtype=tl.float32)
            ds_6 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_6 = tl.zeros((block1,), dtype=tl.float32)
            ds_7 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_7 = tl.zeros((block1,), dtype=tl.float32)
            ds_8 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_8 = tl.zeros((block1,), dtype=tl.float32)
            ds_9 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_9 = tl.zeros((block1,), dtype=tl.float32)
            ds_10 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_10 = tl.zeros((block1,), dtype=tl.float32)
            ds_11 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_11 = tl.zeros((block1,), dtype=tl.float32)
            ds_12 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_12 = tl.zeros((block1,), dtype=tl.float32)
            ds_13 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_13 = tl.zeros((block1,), dtype=tl.float32)
            ds_14 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_14 = tl.zeros((block1,), dtype=tl.float32)
            ds_15 = tl.zeros((block1, BLOCK_E_VALID), dtype=tl.float32)
            dsk_15 = tl.zeros((block1,), dtype=tl.float32)
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
            
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_0 = qT_d1 * q_d2_0[None, :] # [block1 x BLOCK_T]
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_1 = qT_d1 * q_d2_1[None, :] # [block1 x BLOCK_T]
                q_d2_2 = tl.load(p_q_d2_2, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_2 = qT_d1 * q_d2_2[None, :] # [block1 x BLOCK_T]
                q_d2_3 = tl.load(p_q_d2_3, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_3 = qT_d1 * q_d2_3[None, :] # [block1 x BLOCK_T]
                q_d2_4 = tl.load(p_q_d2_4, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_4 = qT_d1 * q_d2_4[None, :] # [block1 x BLOCK_T]
                q_d2_5 = tl.load(p_q_d2_5, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_5 = qT_d1 * q_d2_5[None, :] # [block1 x BLOCK_T]
                q_d2_6 = tl.load(p_q_d2_6, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_6 = qT_d1 * q_d2_6[None, :] # [block1 x BLOCK_T]
                q_d2_7 = tl.load(p_q_d2_7, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_7 = qT_d1 * q_d2_7[None, :] # [block1 x BLOCK_T]
                q_d2_8 = tl.load(p_q_d2_8, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_8 = qT_d1 * q_d2_8[None, :] # [block1 x BLOCK_T]
                q_d2_9 = tl.load(p_q_d2_9, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_9 = qT_d1 * q_d2_9[None, :] # [block1 x BLOCK_T]
                q_d2_10 = tl.load(p_q_d2_10, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_10 = qT_d1 * q_d2_10[None, :] # [block1 x BLOCK_T]
                q_d2_11 = tl.load(p_q_d2_11, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_11 = qT_d1 * q_d2_11[None, :] # [block1 x BLOCK_T]
                q_d2_12 = tl.load(p_q_d2_12, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_12 = qT_d1 * q_d2_12[None, :] # [block1 x BLOCK_T]
                q_d2_13 = tl.load(p_q_d2_13, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_13 = qT_d1 * q_d2_13[None, :] # [block1 x BLOCK_T]
                q_d2_14 = tl.load(p_q_d2_14, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_14 = qT_d1 * q_d2_14[None, :] # [block1 x BLOCK_T]
                q_d2_15 = tl.load(p_q_d2_15, mask=mask_t, other=0.0) # BLOCK_T
                phiqT_15 = qT_d1 * q_d2_15[None, :] # [block1 x BLOCK_T]
                alpha = tl.maximum(gamma, tl.exp(rowmax)) # [BLOCK_T]
                factor = 1 / alpha / l # [BLOCK_T]
                delta_factor = delta * factor # [BLOCK_T]
                do = (do * factor[:, None]).to(Q.dtype.element_ty) # [BLOCK_T x BLOCK_E_VALID]
            
                ds_0 = tl.dot(phiqT_0, do, ds_0) # [block1 x BLOCK_E_VALID]
                dsk_0 += -tl.sum(phiqT_0 * delta_factor[None, :], 1) # [block1]
                ds_1 = tl.dot(phiqT_1, do, ds_1) # [block1 x BLOCK_E_VALID]
                dsk_1 += -tl.sum(phiqT_1 * delta_factor[None, :], 1) # [block1]
                ds_2 = tl.dot(phiqT_2, do, ds_2) # [block1 x BLOCK_E_VALID]
                dsk_2 += -tl.sum(phiqT_2 * delta_factor[None, :], 1) # [block1]
                ds_3 = tl.dot(phiqT_3, do, ds_3) # [block1 x BLOCK_E_VALID]
                dsk_3 += -tl.sum(phiqT_3 * delta_factor[None, :], 1) # [block1]
                ds_4 = tl.dot(phiqT_4, do, ds_4) # [block1 x BLOCK_E_VALID]
                dsk_4 += -tl.sum(phiqT_4 * delta_factor[None, :], 1) # [block1]
                ds_5 = tl.dot(phiqT_5, do, ds_5) # [block1 x BLOCK_E_VALID]
                dsk_5 += -tl.sum(phiqT_5 * delta_factor[None, :], 1) # [block1]
                ds_6 = tl.dot(phiqT_6, do, ds_6) # [block1 x BLOCK_E_VALID]
                dsk_6 += -tl.sum(phiqT_6 * delta_factor[None, :], 1) # [block1]
                ds_7 = tl.dot(phiqT_7, do, ds_7) # [block1 x BLOCK_E_VALID]
                dsk_7 += -tl.sum(phiqT_7 * delta_factor[None, :], 1) # [block1]
                ds_8 = tl.dot(phiqT_8, do, ds_8) # [block1 x BLOCK_E_VALID]
                dsk_8 += -tl.sum(phiqT_8 * delta_factor[None, :], 1) # [block1]
                ds_9 = tl.dot(phiqT_9, do, ds_9) # [block1 x BLOCK_E_VALID]
                dsk_9 += -tl.sum(phiqT_9 * delta_factor[None, :], 1) # [block1]
                ds_10 = tl.dot(phiqT_10, do, ds_10) # [block1 x BLOCK_E_VALID]
                dsk_10 += -tl.sum(phiqT_10 * delta_factor[None, :], 1) # [block1]
                ds_11 = tl.dot(phiqT_11, do, ds_11) # [block1 x BLOCK_E_VALID]
                dsk_11 += -tl.sum(phiqT_11 * delta_factor[None, :], 1) # [block1]
                ds_12 = tl.dot(phiqT_12, do, ds_12) # [block1 x BLOCK_E_VALID]
                dsk_12 += -tl.sum(phiqT_12 * delta_factor[None, :], 1) # [block1]
                ds_13 = tl.dot(phiqT_13, do, ds_13) # [block1 x BLOCK_E_VALID]
                dsk_13 += -tl.sum(phiqT_13 * delta_factor[None, :], 1) # [block1]
                ds_14 = tl.dot(phiqT_14, do, ds_14) # [block1 x BLOCK_E_VALID]
                dsk_14 += -tl.sum(phiqT_14 * delta_factor[None, :], 1) # [block1]
                ds_15 = tl.dot(phiqT_15, do, ds_15) # [block1 x BLOCK_E_VALID]
                dsk_15 += -tl.sum(phiqT_15 * delta_factor[None, :], 1) # [block1]
                p_do += BLOCK_T * stride_dot
                p_qT_d1 += BLOCK_T * stride_qt
                p_q_d2_0 += BLOCK_T * stride_qt
                p_q_d2_1 += BLOCK_T * stride_qt
                p_q_d2_2 += BLOCK_T * stride_qt
                p_q_d2_3 += BLOCK_T * stride_qt
                p_q_d2_4 += BLOCK_T * stride_qt
                p_q_d2_5 += BLOCK_T * stride_qt
                p_q_d2_6 += BLOCK_T * stride_qt
                p_q_d2_7 += BLOCK_T * stride_qt
                p_q_d2_8 += BLOCK_T * stride_qt
                p_q_d2_9 += BLOCK_T * stride_qt
                p_q_d2_10 += BLOCK_T * stride_qt
                p_q_d2_11 += BLOCK_T * stride_qt
                p_q_d2_12 += BLOCK_T * stride_qt
                p_q_d2_13 += BLOCK_T * stride_qt
                p_q_d2_14 += BLOCK_T * stride_qt
                p_q_d2_15 += BLOCK_T * stride_qt
                
            
            range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
            p_dsk_0 = dSK + range_d2_0 * stride_dskD # [block1]
            tl.store(p_dsk_0, dsk_0 * scale_p)
            p_ds_0 = dS + range_d2_0[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_0, ds_0 * scale_p)
            range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
            p_dsk_1 = dSK + range_d2_1 * stride_dskD # [block1]
            tl.store(p_dsk_1, dsk_1 * scale_p)
            p_ds_1 = dS + range_d2_1[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_1, ds_1 * scale_p)
            range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
            p_dsk_2 = dSK + range_d2_2 * stride_dskD # [block1]
            tl.store(p_dsk_2, dsk_2 * scale_p)
            p_ds_2 = dS + range_d2_2[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_2, ds_2 * scale_p)
            range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
            p_dsk_3 = dSK + range_d2_3 * stride_dskD # [block1]
            tl.store(p_dsk_3, dsk_3 * scale_p)
            p_ds_3 = dS + range_d2_3[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_3, ds_3 * scale_p)
            range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
            p_dsk_4 = dSK + range_d2_4 * stride_dskD # [block1]
            tl.store(p_dsk_4, dsk_4 * scale_p)
            p_ds_4 = dS + range_d2_4[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_4, ds_4 * scale_p)
            range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
            p_dsk_5 = dSK + range_d2_5 * stride_dskD # [block1]
            tl.store(p_dsk_5, dsk_5 * scale_p)
            p_ds_5 = dS + range_d2_5[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_5, ds_5 * scale_p)
            range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
            p_dsk_6 = dSK + range_d2_6 * stride_dskD # [block1]
            tl.store(p_dsk_6, dsk_6 * scale_p)
            p_ds_6 = dS + range_d2_6[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_6, ds_6 * scale_p)
            range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
            p_dsk_7 = dSK + range_d2_7 * stride_dskD # [block1]
            tl.store(p_dsk_7, dsk_7 * scale_p)
            p_ds_7 = dS + range_d2_7[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_7, ds_7 * scale_p)
            range_d2_8 = tl.arange(0, block1).to(tl.int64) + 8 * block1
            p_dsk_8 = dSK + range_d2_8 * stride_dskD # [block1]
            tl.store(p_dsk_8, dsk_8 * scale_p)
            p_ds_8 = dS + range_d2_8[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_8, ds_8 * scale_p)
            range_d2_9 = tl.arange(0, block1).to(tl.int64) + 9 * block1
            p_dsk_9 = dSK + range_d2_9 * stride_dskD # [block1]
            tl.store(p_dsk_9, dsk_9 * scale_p)
            p_ds_9 = dS + range_d2_9[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_9, ds_9 * scale_p)
            range_d2_10 = tl.arange(0, block1).to(tl.int64) + 10 * block1
            p_dsk_10 = dSK + range_d2_10 * stride_dskD # [block1]
            tl.store(p_dsk_10, dsk_10 * scale_p)
            p_ds_10 = dS + range_d2_10[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_10, ds_10 * scale_p)
            range_d2_11 = tl.arange(0, block1).to(tl.int64) + 11 * block1
            p_dsk_11 = dSK + range_d2_11 * stride_dskD # [block1]
            tl.store(p_dsk_11, dsk_11 * scale_p)
            p_ds_11 = dS + range_d2_11[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_11, ds_11 * scale_p)
            range_d2_12 = tl.arange(0, block1).to(tl.int64) + 12 * block1
            p_dsk_12 = dSK + range_d2_12 * stride_dskD # [block1]
            tl.store(p_dsk_12, dsk_12 * scale_p)
            p_ds_12 = dS + range_d2_12[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_12, ds_12 * scale_p)
            range_d2_13 = tl.arange(0, block1).to(tl.int64) + 13 * block1
            p_dsk_13 = dSK + range_d2_13 * stride_dskD # [block1]
            tl.store(p_dsk_13, dsk_13 * scale_p)
            p_ds_13 = dS + range_d2_13[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_13, ds_13 * scale_p)
            range_d2_14 = tl.arange(0, block1).to(tl.int64) + 14 * block1
            p_dsk_14 = dSK + range_d2_14 * stride_dskD # [block1]
            tl.store(p_dsk_14, dsk_14 * scale_p)
            p_ds_14 = dS + range_d2_14[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_14, ds_14 * scale_p)
            range_d2_15 = tl.arange(0, block1).to(tl.int64) + 15 * block1
            p_dsk_15 = dSK + range_d2_15 * stride_dskD # [block1]
            tl.store(p_dsk_15, dsk_15 * scale_p)
            p_ds_15 = dS + range_d2_15[:, None] * stride_dsD + range_e[None, :] * stride_dse
            tl.store(p_ds_15, ds_15 * scale_p)
            
    else:
        tl.static_assert(False, "No matching config found")
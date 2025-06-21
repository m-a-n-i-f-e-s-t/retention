import torch
import math
from einops import rearrange
from power_attention._utils import dummify, compute_expanded_dim, diff

InnerBlock = 16
OuterBlock = 1

class QueryStateReference(torch.autograd.Function):
    @staticmethod
    def expand(Q, deg):
        """ Reference implementation of key expansion
        args:
            Q: [b, n, h, c d]
            deg: int
        returns:
            phi_Q: [b, n, h, c, D]
        """
        b, n, h, c, d = Q.shape
        Q_outer = rearrange(Q, 'b n h c (x o) -> b n h c x o', o=OuterBlock)
        Q_inner = rearrange(Q, 'b n h c (y i) -> b n h c y i', i=InnerBlock)
        phi_Q_unmasked = torch.einsum('bnhcxo,bnhcyi->bnhcxyoi', Q_outer, Q_inner).to(Q.dtype)
        _, _, _, _, x, y, o, i = phi_Q_unmasked.shape
        phi_Q_shape = (b, n, h, c, int((InnerBlock // OuterBlock + x) * y // 2), o, i)
        phi_Q = torch.empty(phi_Q_shape, device=Q.device, dtype=Q.dtype)
        idx = 0
        for y_idx in range(y):
            for x_idx in range(x):
                if (x_idx * OuterBlock) < (y_idx + 1) * InnerBlock:
                    phi_Q[:, :, :, :, idx, :, :] = phi_Q_unmasked[:, :, :, :, x_idx, y_idx, :, :]
                    idx += 1
        phi_Q = rearrange(phi_Q, 'b n h c k o i -> b n h c (k o i)') # [b, n, h, c, D]
        return phi_Q.to(Q.dtype)

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
            Y_attn: [b, n, c, h, d] - attention output
            l_attn: [b, n, c, h] - sum of powered attention scores, in normal space
            rowmax: [b, n, c, h] - max of powered attention scores, in log space
            deg: int - degree of power
            scale: float - sm_scale used in attention
            zero_initial_state: bool - whether the initial state is zero
        returns:
            O: [b, n, c, h, d] - temporal-normalized output
        """

        b, n, c, h, d = Q.shape
        _, _, _, D, _ = S.shape
        scale_p = scale**deg

        γ = torch.ones(n, device=Q.device, dtype=torch.float32) * math.sqrt(float(D))
        # special case for zero initial state, no need to scale by γ
        if zero_initial_state:
            γ[0] = 1.0
        γ = γ.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, n, 1, 1]

        Q = rearrange(Q, 'b n c h d -> b n h c d') # [b, n, h, c, d]
        phi_Q = QueryStateReference.expand(Q, deg) # [b, n, h, c, D]
        S = (S / γ.unsqueeze(-1)).to(S.dtype) # [b, n, h, D, d]
        s = s / γ # [b, n, h, D]

        Y_qs = torch.matmul(phi_Q.to(Q.dtype), S.to(Q.dtype)).to(Q.dtype)  # [b n h c d]
        l_qs = torch.matmul(phi_Q.to(torch.float32), s.unsqueeze(-1).to(torch.float32)).squeeze(-1) # [b n h c]
        Y_qs = rearrange(Y_qs, 'b n h c d -> b n c h d')
        l_qs = rearrange(l_qs, 'b n h c -> b n c h')

        alpha = torch.maximum(γ, torch.exp(rowmax)) # [b, n, c, h]
        attn_factor = torch.exp(rowmax) / alpha # [b, n, c, h]
        qs_factor = γ / alpha # [b, n, c, h]
        O = Y_attn * attn_factor.unsqueeze(-1) + Y_qs * scale_p * qs_factor.unsqueeze(-1) # [b, n, c, h, d]
        l = l_attn * attn_factor + l_qs * scale_p * qs_factor # [b, n, c, h]
        O = (O.to(torch.float32) / l.unsqueeze(-1)).to(Y_qs.dtype)

        S = (S * γ.unsqueeze(-1)).to(S.dtype)
        s = (s * γ).to(s.dtype)
        ctx.save_for_backward(Q, S, O, s, l, alpha, rowmax)
        ctx.zero_initial_state = zero_initial_state
        ctx.deg = deg
        ctx.d = d
        ctx.scale_p = scale_p
        return O.contiguous()

    # TODO: fix

    # @staticmethod
    # def backward(ctx, dO):
    #     """ Computes the following equation for the backward pass:

    #         delta = (dO * O) @ 1
    #         dl_attn = -exp(rowmax) / alpha / l * delta
    #         dY_attn = exp(rowmax) / alpha / l * dO
    #         dS = phi_Q^T @ (dO * (scale**p) / alpha / l)
    #         ds = -phi_Q^T @ (delta * (scale**p) / alpha / l)
    #         dphi_Q = (scale**p) / alpha / l * (dO @ S^T - delta @ s^T)

    #     args:
    #         dO: [b, n, c, h, e]
    #     returns:
    #         dQ: [b, n, c, h, d]
    #         dS: [b, n, h, D, e]
    #     """
    #     Q, S, O, s, l, alpha, rowmax = ctx.saved_tensors # S: [b, n, h, D, d], l: [b, n, c, h], alpha: [b, n, 1, h], rowmax: [b, n, c, h], s: [b, n, h, D]
    #     scale_p = ctx.scale_p

    #     # put batch dimensions together
    #     O, dO, l, rowmax, alpha = map(lambda x: x.transpose(2, 3), (O, dO, l, rowmax, alpha)) # [b, n, h, ...]

    #     delta = torch.sum(O * dO, dim=-1) # [b, n, h, c]
    #     factor = 1 / alpha / l # [b, n, h, c]

    #     # --- compute dS, ds
    #     phi_Q = QueryStateReference.expand(Q, ctx.deg)# [b, n, h, c, D]
    #     dS = torch.matmul(phi_Q.transpose(-1, -2) , (dO * scale_p * factor.unsqueeze(-1)).to(dO.dtype)).to(S.dtype)  # [b, n, h, D, d]
    #     ds = torch.matmul(phi_Q.transpose(-1, -2).to(s.dtype), -(delta * scale_p * factor).unsqueeze(-1)).squeeze(-1).to(s.dtype) # [b, n, h, D]
      
    #     # --- compute dPhiQ
    #     S_t = S.transpose(-1, -2)  # [b, n, h, d, D]
    #     dphi_Q = (torch.matmul(dO, S_t) - torch.matmul(delta.unsqueeze(-1), s.unsqueeze(-2).to(delta.dtype))) * scale_p * factor.unsqueeze(-1) # [b, n, h, c, D]

    #     # --- compute dQ
    #     dQ = torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device) # [b, n, h, c, d]
    #     dphi_Q = rearrange(dphi_Q, 'b n h c (z i o) -> b n h c z i o', i=InnerBlock, o=OuterBlock).to(Q.dtype)
    #     Q_inner = rearrange(Q, 'b n h c (y i) -> b n h c y i', i=InnerBlock)
    #     Q_outer = rearrange(Q, 'b n h c (x o) -> b n h c x o', o=OuterBlock)
    #     y, x = ctx.d // InnerBlock, ctx.d // OuterBlock

    #     for j in range(ctx.d // OuterBlock):
    #         for i in range((j * OuterBlock) // InnerBlock, y):
    #             z = (i * (i + 1) // 2 * InnerBlock // OuterBlock) + j
    #             dPQ_block = dphi_Q[..., z, :, :] # [b, n, h, c, InnerBlock, OuterBlock]
    #             dPQ_block = rearrange(dPQ_block, 'b n h c i o -> b n h c (i o)')
    #             Q_i = Q_inner[..., i, :] # [b, n, h, c, InnerBlock]
    #             Q_o = Q_outer[..., j, :] # [b, n, h, c, OuterBlock]
    #             dQ_i = (dPQ_block * Q_o)
    #             dQ_o = (dPQ_block * Q_i).sum(dim=-1, keepdim=True)
    #             dQ[..., i * InnerBlock:(i + 1) * InnerBlock] += dQ_i
    #             dQ[..., j * OuterBlock:(j + 1) * OuterBlock] += dQ_o

    #     dQ = dQ.transpose(2, 3)

    #     # --- compute dY_attn and dl_attn
    #     attn_factor = torch.exp(rowmax) * factor # [b, n, h, c]
    #     dY_attn = attn_factor.unsqueeze(-1) * dO # [b, n, h, c, d]
    #     dl_attn = -attn_factor * delta # [b, n, h, c]
    #     dY_attn, dl_attn = map(lambda x: x.transpose(2, 3), (dY_attn, dl_attn)) # [b, n, c, ...]

    #     return dQ, dS, ds, dY_attn, dl_attn, None, None, None, None

def query_state_reference(*args, **kwargs):
    if args and kwargs:
        raise ValueError("Cannot pass both args and kwargs")
    if kwargs:
        args = (kwargs['Q'], kwargs['S'], kwargs['s'], kwargs['Y_attn'], kwargs['l_attn'], kwargs['rowmax'], kwargs['deg'], kwargs['scale'], kwargs['zero_initial_state'])
    return QueryStateReference.apply(*args)
query_state_reference_fwd = dummify(QueryStateReference.forward)
query_state_reference = query_state_reference_fwd
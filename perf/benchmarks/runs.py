
import torch
import torch.nn.functional as F
import math
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from power_attention_cuda import mosaic_sympow
from power_attention.power_full import power_full, power_full_triton
from power_attention._expansion.impl_triton import expand as triton_expand
from power_attention._attention import attention_triton, attention_reference, attention
from power_attention._update_state import update_state_triton, update_state
from power_attention._query_state import query_state_triton, query_state
from power_attention._discumsum import discumsum

class SDPA():
    @staticmethod
    def make_run(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        scale = kw.get('scale', 1.0)
        is_causal = kw.get('is_causal', True)
        qhead_ratio = kw.get('qhead_ratio', 1)
        dropout_p = kw.get('dropout_p', 0.0)
        enable_gqa = kw.get('enable_gqa', False)
        requires_grad = kw.get('requires_grad', False)

        q = torch.randn((b, h * qhead_ratio, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)

        inputs = dict(
            query=q,
            key=k,
            value=v,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

        return lambda: torch.nn.functional.scaled_dot_product_attention(**inputs)


class FLA():
    @staticmethod
    def make_run(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        scale = kw.get('scale', 1.0)
        initial_state = kw.get('initial_state', None)
        output_final_state = kw.get('output_final_state', False)
        normalize = kw.get('normalize', False)
        head_first = kw.get('head_first', False)
        requires_grad = kw.get('requires_grad', False)

        if head_first:
            q = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
            k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
            v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        else:
            q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
            k = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
            v = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)

        if kw.get('fused', False):
            fn = fused_chunk_linear_attn
        else:
            fn = chunk_linear_attn
        
        inputs = dict(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            normalize=normalize,
            head_first=head_first,
        )

        def only_return_o():
            o, final_state = fn(**inputs)
            return o

        return only_return_o


class Power():
    @staticmethod
    def make_run(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        chunk_size = kw.get('chunk_size', None)
        gating = kw.get('gating', False)
        requires_grad = kw.get('requires_grad', False)
        scale = kw.get('scale', 1.0)

        # 128 is not supported by power_full yet
        if d == 128:
            return lambda: None

        Q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        K = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        V = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)

        if gating:
            log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device, requires_grad=requires_grad))
        else:
            log_G = None

        inputs = dict(
            Q=Q,
            K=K,
            V=V,
            log_G=log_G,
            deg=deg,
            scale=scale,
            chunk_size=chunk_size,
        )

        return lambda: power_full(**inputs)
    

class PowerTriton():
    @staticmethod
    def make_run(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        chunk_size = kw.get('chunk_size', None)
        gating = kw.get('gating', False)
        requires_grad = kw.get('requires_grad', False)
        scale = kw.get('scale', 1.0)

        Q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        K = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        V = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)

        if gating:
            log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device, requires_grad=requires_grad))
        else:
            log_G = None

        inputs = dict(
            Q=Q,
            K=K,
            V=V,
            log_G=log_G,
            deg=deg,
            scale=scale,
            chunk_size=chunk_size,
        )

        return lambda: power_full_triton(**inputs)


class TritonExpansion():
    @staticmethod
    def make_run(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        split = kw.get('split', 'T')

        K = torch.randn((b, 1, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)

        return lambda: triton_expand(K, deg, split)


class MosaicExpansion():
    @staticmethod
    def make_run(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        d_block = 8

        K = torch.randn((t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        K = K.transpose(0, 1).contiguous().transpose(0, 1)

        return lambda: mosaic_sympow(K, deg, d_block)[0]
    

class QueryState():
    @staticmethod
    def create_inputs(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        d_block = kw.get('d_block', 16)
        fused = kw.get('fused', False)
        scale = kw.get('scale', 1.0)
        zero_initial_state = kw.get('zero_initial_state', False)
        D = math.comb(d//d_block + deg - 1, deg) * (d_block**deg)
        Q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        S = torch.randn((b, t, h, D), dtype=dtype, device=device, requires_grad=requires_grad)
        if fused:
            Y = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
            rowmax = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device, requires_grad=requires_grad))
        else:
            Y = None
            rowmax = None
        if zero_initial_state:
            S[:, 0] = 0
        return dict(Q=Q, S=S, Y=Y, rowmax=rowmax, deg=deg, scale=scale, zero_initial_state=zero_initial_state)
        
    @staticmethod
    def make_run(**kw):
        inputs = QueryState.create_inputs(**kw)
        return lambda: query_state(**inputs)
    

class QueryStateTriton(QueryState):
    @staticmethod
    def make_run(**kw):
        inputs = QueryState.create_inputs(**kw)
        return lambda: query_state_triton(**inputs)
    

class UpdateState():
    @staticmethod
    def create_inputs(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        K = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        V = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        return dict(K=K, V=V, deg=deg)
        
    @staticmethod
    def make_run(**kw):
        inputs = UpdateState.create_inputs(**kw)
        return lambda: update_state(**inputs)
    

class UpdateStateTriton(UpdateState):
    @staticmethod
    def make_run(**kw):
        inputs = UpdateState.create_inputs(**kw)
        return lambda: update_state_triton(**inputs)
    

class PowerAttention():
    @staticmethod
    def create_inputs(**kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        scale = kw.get('scale', 1.0)
        gating = kw.get('gating', False)
        Q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        K = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        V = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        if gating:
            log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device, requires_grad=requires_grad))
        else:
            log_G = None
        return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, scale=scale)
    
    @staticmethod
    def make_run(**kw):
        inputs = PowerAttention.create_inputs(**kw)
        return lambda: attention(**inputs)
    

class PowerAttentionTriton(PowerAttention):
    @staticmethod
    def make_run(**kw):
        inputs = PowerAttention.create_inputs(**kw)
        return lambda: attention_triton(**inputs)

    
    
    



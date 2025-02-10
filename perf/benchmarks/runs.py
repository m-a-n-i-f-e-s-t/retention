
import torch
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from power_attention_cuda import mosaic_sympow
from power_attention.power_full import power_full, power_full_triton, create_inputs as create_inputs_power
from power_attention._expansion import expand as triton_expand, create_inputs as create_inputs_expansion
from power_attention._attention import attention_triton, attention, create_inputs as create_inputs_attention
from power_attention._update_state import update_state_triton, update_state, create_inputs as create_inputs_update_state
from power_attention._query_state import query_state_triton, query_state, create_inputs as create_inputs_query_state
from perf.baselines.fla import create_inputs as create_inputs_fla
from perf.baselines.sdpa import create_inputs as create_inputs_sdpa


def sanitize_kwargs(fn):
    from functools import wraps
    import inspect
    
    @wraps(fn)
    def wrapper(**kwargs):
        sig = inspect.signature(fn)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**valid_kwargs)
    return wrapper


class SDPA():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_sdpa)(**kw)

        return lambda: torch.nn.functional.scaled_dot_product_attention(**inputs)


class FLA():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_fla)(**kw)

        if kw.get('fused', False):
            fn = fused_chunk_linear_attn
        else:
            fn = chunk_linear_attn

        def only_return_o():
            o, final_state = fn(**inputs)
            return o

        return only_return_o


class Power():
    @staticmethod
    def make_run(**kw):
        # 128 is not supported by power_full yet
        if kw['d'] == 128:
            def raise_not_implemented():
                raise NotImplementedError
            return raise_not_implemented

        inputs = sanitize_kwargs(create_inputs_power)(**kw)

        return lambda: power_full(**inputs)
    

class PowerTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_power)(**kw)

        return lambda: power_full_triton(**inputs)


class TritonExpansion():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_expansion)(**kw)
        return lambda: triton_expand(**inputs)


class MosaicExpansion():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_expansion)(**kw)

        return lambda: mosaic_sympow(inputs['K'], 2, 8)[0]
    

class QueryState():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_query_state)(**kw)
        return lambda: query_state(**inputs)
    

class QueryStateTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_query_state)(**kw)
        return lambda: query_state_triton(**inputs)
    

class UpdateState():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_update_state)(**kw)
        return lambda: update_state(**inputs)
    

class UpdateStateTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_update_state)(**kw)
        return lambda: update_state_triton(**inputs)
    

class PowerAttention():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_attention)(**kw)
        return lambda: attention(**inputs)
    

class PowerAttentionTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_attention)(**kw)
        return lambda: attention_triton(**inputs)

    
    
    



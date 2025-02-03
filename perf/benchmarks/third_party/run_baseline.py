""" A simple script that runs a baseline, intended to be used for profiling with nsys
"""

import torch
import math
import json
import sys
import triton
from power_attention.power_full import power_full, create_inputs as create_inputs_power
from power_attention._utils import compute_expanded_dim
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn

def str_to_dtype(dtype_str):
    """Convert string dtype to torch.dtype"""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat16': torch.bfloat16,
    }
    return dtype_map[dtype_str]

def create_inputs_sdpa(b, t, h, d, dtype, device, qhead_ratio=1, requires_grad=False):
    if isinstance(dtype, str):
        dtype = str_to_dtype(dtype)
    q = torch.randn((b, h * qhead_ratio, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    return {
        'query': q,
        'key': k,
        'value': v,
        'dropout_p': 0.0,
        'scale': 1.0 / math.sqrt(d),
        'is_causal': True,
        'enable_gqa': qhead_ratio > 1,
    }

def create_inputs_fla(b, t, h, d, dtype, device, qhead_ratio=1, requires_grad=False, scale=1.0, initial_state=None, output_final_state=False, head_first=False, normalize=True):
    if isinstance(dtype, str):
        dtype = str_to_dtype(dtype)
    if head_first:
        q = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn((b, h, t, d), dtype=dtype, device=device, requires_grad=requires_grad)
    else:
        q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)
    return {
        'q': q,
        'k': k,
        'v': v,
        'scale': scale,
        'initial_state': initial_state,
        'output_final_state': output_final_state,
        'normalize': normalize,
        'head_first': head_first,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', type=str, default='sdpa')
    parser.add_argument('--mode', type=str, default='fwd')
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--t', type=int, default=1024)
    parser.add_argument('--h', type=int, default=12)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--deg', type=int, default=2)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--dtype', type=lambda x: torch.bfloat16 if x == 'bfloat16' else torch.float16, default=torch.bfloat16)
    parser.add_argument('--fused', action='store_true', default=False)
    args = parser.parse_args()
    
    print(f"Running with args: {args}")

    # Setup based on provider
    if args.provider == "sdpa":
        create_inputs = create_inputs_sdpa
        fn = torch.nn.functional.scaled_dot_product_attention
    elif args.provider == "fla":
        create_inputs = create_inputs_fla
        fused = args.fused
        def run_fla(**kw):
            if fused:
                o, s = fused_chunk_linear_attn(kw['q'], kw['k'], kw['v'], scale=kw['scale'], 
                                             initial_state=kw['initial_state'], 
                                             output_final_state=kw['output_final_state'], 
                                             head_first=kw['head_first'], normalize=kw['normalize'])
            else:
                o, s = chunk_linear_attn(kw['q'], kw['k'], kw['v'], scale=kw['scale'], 
                                       initial_state=kw['initial_state'], 
                                       output_final_state=kw['output_final_state'], 
                                       head_first=kw['head_first'], normalize=kw['normalize'])
            return o
        fn = run_fla
    elif args.provider == "power":
        create_inputs = lambda *ags, **kw: create_inputs_power(*ags, **kw, qhead_ratio=1, deg=args.deg, gating=True, chunk_size=args.chunk_size)
        fn = power_full
    
    # Create inputs
    inputs = create_inputs(args.b, args.t, args.h, args.d, dtype=args.dtype, device='cuda', requires_grad=(args.mode != "fwd"))
    
    torch.cuda.synchronize()
    
    # Run the actual computation
    fn = torch.compile(fn, dynamic=False)
    def run_fn():
        out = fn(**inputs)
        if args.mode != "fwd":
            if isinstance(out, tuple):
                loss = out[0].sum()
            else:
                loss = out.sum()
            loss.backward()

    # run_fn()
    
    torch.cuda.synchronize()

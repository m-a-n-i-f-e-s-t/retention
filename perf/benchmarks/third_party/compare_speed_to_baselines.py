import torch
import triton
import math
import os
from perf._timing import benchmark_speed

# TODO(sean): enable this when ready
# from power_attention._attention.impl_triton2 import attention as triton_power_attention, create_inputs as create_inputs_triton_power
from power_attention.power_full import power_full, create_inputs as create_inputs_power
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn

torch._dynamo.config.cache_size_limit = 128 # Increased from a default of 8 to prevent warnings

FWD, BWD, FWD_BWD = "fwd", "bwd", "fwd+bwd"
SDPA, POWER, TRITON_POWER, FLA, FLASH = "sdpa", "power", "triton_power", "fla", "flash"
USE_TRITON_BENCH = os.environ.get("USE_TRITON_BENCH", "0") == "1"

def compute_expanded_dim(head_dim, deg):
    if deg == 1:
        return head_dim
    elif deg == 2:
        if head_dim == 64:
            return math.comb(deg + head_dim - 1, deg)
        else:
            raise ValueError(f"Unsupported head dimension: {head_dim}")
    else:
        raise ValueError(f"Unsupported degree: {deg}")

def estimate_flops(ctx, batch, head_q, head_k, head_dim, direction, dtype, device, causal, algo, deg=None, chunk_size=None, gating=False, **kw):
    """ calculate theoretical flops

    Returns:
        fwd_flops: FLOPs for forward pass
        bwd_flops: FLOPs for backward pass
    """
    if direction == FWD_BWD:
        fwd_flops = estimate_flops(ctx, batch, head_q, head_k, head_dim, FWD, dtype, device, causal, algo, deg, chunk_size, gating)
        bwd_flops = estimate_flops(ctx, batch, head_q, head_k, head_dim, BWD, dtype, device, causal, algo, deg, chunk_size, gating)
        return fwd_flops + bwd_flops
    if direction == BWD:
        return estimate_flops(ctx, batch, head_q, head_k, head_dim, FWD, dtype, device, causal, algo, deg, chunk_size, gating)

    power = algo == POWER
    def _attention_flops(batch, ctx, head_q, head_k, head_dim, direction, gating, causal):
        if direction == FWD:
            return batch * head_q * (2 * ctx * ctx * head_dim * 2 + (ctx * ctx if gating else 0) + (ctx * ctx * 3 if power else 0)) * (0.5 if causal else 1.0)
        else:
            return batch * head_q * (ctx * ctx * head_dim * 2 # QK^T
                    + (ctx * ctx if gating else 0) # gating
                    + (ctx * ctx * 3 if power else 0) # power 
                    + ctx * head_dim * ctx * 2 # dV
                    + ctx * ctx * head_dim * 2 # dP
                    + ctx * ctx # dS
                    + ctx * head_dim * ctx * 2 # dQ
                    + ctx * head_dim * ctx * 2 # dK
                    ) * (0.5 if causal else 1.0)
    
    def _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, direction, D, gating, causal):
        if direction == FWD:
            return batch * head_q * (ctx/chunk_size) * (
                    + chunk_size * D * 2 # state expansion
                    + D * head_dim * chunk_size * 2 # update state
                    + chunk_size * head_dim * D * 2) # query state
        else:
            return batch * head_q * (ctx/chunk_size) * (
                + D * head_dim * chunk_size * 2 # dS
                + chunk_size * head_dim * D * 2 # dQ
                + chunk_size * head_dim * D * 2 # dK
                + chunk_size * head_dim * D * 2 # dV
                )

    if algo == POWER:
        D = compute_expanded_dim(head_dim, deg)
        if chunk_size is None:
            return _attention_flops(batch, ctx, head_q, head_k, head_dim, direction, gating, causal)
        else:
            attn_flops = _attention_flops(batch * ctx // chunk_size, chunk_size, head_q, head_k, head_dim, direction, gating, causal)
            chunk_flops = _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, direction, D, gating, causal)
            return attn_flops + chunk_flops
    elif algo == SDPA or algo == FLASH:
        return _attention_flops(batch, ctx, head_q, head_k, head_dim, direction, gating, causal)
    elif algo == FLA:
        chunk_size = min(64, max(16, triton.next_power_of_2(ctx)))
        attn_flops = _attention_flops(batch, ctx, head_q, head_k, head_dim, direction, gating, causal)
        chunk_flops = _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, direction, head_dim, gating, causal)
        return attn_flops + chunk_flops
    else:
        raise ValueError(f"Unknown algo: {algo}")


def create_inputs_sdpa(b, t, h, d, dtype, device, qhead_ratio=1, requires_grad=False):
    """Create inputs for attention benchmarking
    
    Args:
        b: batch size
        t: sequence length
        h: number of heads
        d: head dimension
        dtype: data type
        device: device to create tensors on
        qhead_ratio: ratio of query heads to key/value heads (for GQA)
    """
    # Create query, key, value tensors
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
    """Create inputs for FLA"""
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
    parser.add_argument("--measure", type=str, default="time", choices=["time", "flops", "throughput"])
    parser.add_argument("--csv", action="store_true", help="Print results in CSV format")
    parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "bwd", "fwd+bwd"])
    args = parser.parse_args()

    def print_rowstr(rowstr):
        if not args.csv:
            print(" | ".join([f"{r.upper():<10}" for r in rowstr.split(",")]))
        else:
            print(rowstr)

    print_rowstr("ctx,sdpa,p1_att,p2_att,p1_chunk,p2_chunk")
    for ctx in [2**i for i in range(10, 17)]:
        rowstr = f"{ctx},"
        for provider in [SDPA, (POWER, 1, None), (POWER, 2, None), (POWER, 1, 128), (POWER, 2, 1024)]:
            if provider == SDPA:
                create_inputs = create_inputs_sdpa
                algo = SDPA
                fn = torch.nn.functional.scaled_dot_product_attention
                algo_kw = {}
            elif FLA in provider:
                create_inputs = create_inputs_fla
                fused = provider[1]
                algo = FLA
                def run_fla(**kw):
                    if fused:
                        o, s = fused_chunk_linear_attn(kw['q'], kw['k'], kw['v'], scale=kw['scale'], initial_state=kw['initial_state'], output_final_state=kw['output_final_state'], head_first=kw['head_first'], normalize=kw['normalize'])
                    else:
                        o, s = chunk_linear_attn(kw['q'], kw['k'], kw['v'], scale=kw['scale'], initial_state=kw['initial_state'], output_final_state=kw['output_final_state'], head_first=kw['head_first'], normalize=kw['normalize'])
                    return o
                fn = run_fla
                algo_kw = {
                    'head_first': False,
                    'normalize': True,
                }
            else:
                create_inputs = create_inputs_power
                algo, deg, chunk_size = provider
                fn = power_full
                algo_kw = {
                    'deg': deg,
                    'chunk_size': chunk_size,
                    'gating': True,
                }

            def measure_time():
                if USE_TRITON_BENCH:
                    compiled_fn = torch.compile(fn, dynamic=False) if (FLA not in provider) else fn
                    inputs = create_inputs(**kw)
                    time = triton.testing.do_bench(lambda: compiled_fn(**inputs), warmup=2, rep=10)
                else:
                    time = benchmark_speed(args.mode, fn, create_inputs, kw, compile=(FLA not in provider), num1=2, num2=8, warmup=2)
                return time
            
            def measure_flops():
                flops = estimate_flops(ctx=kw["t"], batch=kw["b"], head_q=kw["h"] * kw["qhead_ratio"], head_k=kw["h"], head_dim=kw["d"], direction=args.mode, dtype=kw["dtype"], device=kw["device"], causal=True, algo=algo, **algo_kw)
                return flops

            if algo == POWER and chunk_size is not None and ctx < chunk_size * 4:
                rowstr += ","
            else:
                kw = {
                    'b': 2**16//ctx,
                    't': ctx,
                    'h': 12,
                    'd': 64,
                    'dtype': torch.bfloat16,
                    'device': 'cuda',
                    'qhead_ratio': 1,
                } | algo_kw
                if args.measure == "time":  
                    time = measure_time()
                    rowstr += f"{time:.6f},"
                elif args.measure == "flops":
                    flops = measure_flops()
                    rowstr += f"{flops:.3e},"
                elif args.measure == "throughput":
                    time = measure_time()
                    flops = measure_flops()
                    throughput = flops * 1e-12 / (time * 1e-3)
                    rowstr += f"{throughput:.3e},"
        print_rowstr(rowstr[:-1])

import torch
import triton
from itertools import product

from power_attention.power_full import power_full
# from power_attention._attention.impl_triton2 import attention as jacob_attention
from power_attention._utils import compute_expanded_dim

torch._dynamo.config.cache_size_limit = 128 # Increased from a default of 8 to prevent warnings

providers = ["sdpa"]
params = {
    "dtype": [torch.bfloat16],
    "device": ["cuda"],
    "gating": [True],
    "deg": [1, 2],
    "direction": ["fwd+bwd"], # can also do `fwd+bwd`
    "b": [2],
    "hq": [12],
    "hk": [12],
    "d": [64],
    "causal": [True]
}
colors = {
    "sdpa": "black",
    "power-p1-chunk128": "#E6B3FF",  # light purple
    "power-p2-chunk1024": "#9933FF",  # darker purple
    "power-p1": "#FFB366",  # light orange
    "power-p2": "#FF8000",  # dark orange
    "tritonpower-p1": "#B3FFB3",  # light green
    "tritonpower-p2": "#33CC33",  # darker green
    "flash": "#FFB3B3",     # light red
    "fla-chunk": "#B3FFB3", # light green
    "fla-fused": "#33CC33", # darker green
    "rwkv-chunk": "#B3E6FF", # light blue
    "rwkv-fused": "#0099FF",  # darker blue
    "rebased": "#00FF00"  # green
}

try:
    from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
    # TODO: add rwkv for comparison
    # from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
    providers += ["fla-chunk", "fla-fused"]
    from fla.ops.rebased.parallel import parallel_rebased
    # providers += ["rebased"]
except BaseException:
    pass


providers += ["power-p1", "power-p2", "power-p1-chunk128", "power-p2-chunk1024"]

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    providers.append("flash")
except BaseException:
    pass



def calculate_flops(ctx, batch, head_q, head_k, head_dim, mode, dtype, device, gating, causal, provider):
    """ calculate theoretical flops

    Returns:
        fwd_flops: FLOPs for forward pass
        bwd_flops: FLOPs for backward pass
    """
    def _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=False):
        if mode == "fwd":
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
    
    def _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, mode, D, gating, causal, power=False):
        if mode == "fwd":
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
        

    if "power" in provider:
        deg = int(provider.split("-")[1][1:])
        chunk_size = int(provider.split("-")[2][5:]) if len(provider.split("-")) > 2 else None
        D = compute_expanded_dim(head_dim, deg)
        if chunk_size is None:
            return _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=True)
        else:
            attn_flops = _attention_flops(batch * ctx // chunk_size, chunk_size, head_q, head_k, head_dim, mode, gating, causal, power=True)
            chunk_flops = _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, mode, D, gating, causal, power=True)
            return attn_flops + chunk_flops
    elif "flash" in provider or "sdpa" in provider or "rebased" in provider:
        return _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=False)
    elif "fla" in provider:
        chunk_size = min(64, max(16, triton.next_power_of_2(ctx)))
        attn_flops = _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=False)
        chunk_flops = _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, mode, head_dim, gating, causal, power=False)
        return attn_flops + chunk_flops
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _bench_compare(ctx, head_q, head_k, head_dim, mode, dtype, device, gating, causal, provider, measure):
    batch = 2**15//ctx
    assert head_q % head_k == 0, "head_q must be divisible by head_k"
    q = torch.randn((batch, ctx, head_q, head_dim), device=device, dtype=dtype, requires_grad=("bwd" in mode))
    k = torch.randn((batch, ctx, head_k, head_dim), device=device, dtype=dtype, requires_grad=("bwd" in mode))
    v = torch.randn((batch, ctx, head_k, head_dim), device=device, dtype=dtype, requires_grad=("bwd" in mode))

    if "power" in provider:
        deg = int(provider.split("-")[1][1:])
        chunk_size = int(provider.split("-")[2][5:]) if len(provider.split("-")) > 2 else None

        if chunk_size is None or ctx % (4*chunk_size) == 0:
            if gating:
                log_g = torch.randn((batch, ctx, head_q), device=device, dtype=torch.float32)
            else:
                log_g = None
            compiled_fn = torch.compile(power_full, dynamic=False)
            
            def run_power():
                return compiled_fn(q, k, v, log_G=log_g, deg=deg, scale=1.0 / head_dim**0.5, chunk_size=chunk_size)
            run_power()
            fn = run_power
        else:
            fn = lambda: None
    elif "flash" in provider:
        def run_flash():
            return torch.compile(flash_attn_func)(q, k, v, causal=causal, softmax_scale=1.0 / head_dim**0.5)
        fn = run_flash
    elif "sdpa" in provider:
        q_t = q.clone().permute(0, 2, 1, 3)
        k_t = k.clone().permute(0, 2, 1, 3)
        v_t = v.clone().permute(0, 2, 1, 3)
        def run_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t,
                                                                 attn_mask=None,
                                                                 dropout_p=0,
                                                                 is_causal=True,
                                                                 scale=1.0 / head_dim**0.5,
                                                                 enable_gqa=head_q > head_k)
        fn = run_sdpa
    elif "fla" in provider:
        def run_fla():
            is_chunked = "chunk" in provider
            if is_chunked:
                o, s = chunk_linear_attn(q, k, v, scale=1.0 / head_dim**0.5, initial_state=None, output_final_state=False, head_first=False, normalize=True)
                return o
            else:
                o, s = fused_chunk_linear_attn(q, k, v, scale=1.0 / head_dim**0.5, initial_state=None, output_final_state=False, head_first=False, normalize=True)
                return o
        fn = run_fla
    elif "rebased" in provider:
        def run_rebased():
            o = parallel_rebased(q, k, v, head_first=False)
            return o
        fn = run_rebased
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if "power" not in provider or chunk_size is None or ctx % (4*chunk_size) == 0:
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        elif mode == "fwd+bwd":
            fwd_fn = fn
            def fwd_bwd():
                o = fwd_fn()
                do = torch.randn_like(o)
                return o.backward(do, retain_graph=True)
            fn = fwd_bwd
        else:
            fn = fn
        ms = triton.testing.do_bench(fn, warmup=2, rep=10) if measure != "flops" else None
    else:
        ms = 0

    flops = calculate_flops(ctx, batch, head_q, head_k, head_dim, mode, dtype, device, gating, causal, provider) if measure != "time" else None
    if measure == "throughput":
        return flops * 1e-12 / (ms * 1e-3) if ms > 0 else 0
    elif measure == "time":
        return ms
    elif measure == "flops":
        return flops
    else:
        raise ValueError(f"Unknown measure: {measure}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", type=str, default="throughput", choices=["throughput", "time", "flops"])
    args = parser.parse_args()

    configs = [
        triton.testing.Benchmark(
            x_names=["ctx"],
            x_vals=[2**i for i in range(10, 16)],
            line_arg="provider",
            line_vals=providers,
            line_names=[provider.upper() for provider in providers],
            styles=[(colors[provider], "-") for provider in providers],
            ylabel="TFLOPS" if args.measure == "throughput" else "ms" if args.measure == "time" else "TFLOPs",
            plot_name=f"power-attention-compare-{args.measure}-headQ{head_q}-headK{head_k}-d{head_dim}{'-gating' if gating else ''}{'-causal' if causal else ''}-{mode}",
            args={
                "head_q": head_q,
                "head_k": head_k,
                "head_dim": head_dim,
                "mode": mode,
                "dtype": dtype,
                "device": device,
                "gating": gating,
                "causal": causal
            }
        )
        for head_q, head_k, head_dim, mode, dtype, device, gating, causal in product(params['hq'], params['hk'], params['d'], params['direction'], params['dtype'], params['device'], params['gating'], params['causal'])
    ]
    bench_compare = triton.testing.perf_report(configs)(_bench_compare)
    bench_compare.run(save_path=".", print_data=True, measure=args.measure)

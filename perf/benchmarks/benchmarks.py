""" This script enables running things we care about
"""
import torch
import click
import time
from perf._timing import benchmark_speed
from collections import defaultdict
from typing import Dict, List, Any, Callable, Iterator
from tabulate import tabulate

from rungroups import *
from runs import *

# Increase PyTorch compilation cache size to avoid recompilation
torch._dynamo.config.cache_size_limit = 512

def str_to_dtype(s: str):
    if s == 'float16':
        return torch.float16
    elif s == 'float32':
        return torch.float32
    elif s == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {s}")

def _speed_benchmark(run_groups: Dict[str, Iterator[Callable[[], Any]]], param_name: str, param_range: List[Any], mode='fwd', compile=True):
    """ Measure the speed of given run groups.
    """
    start_time = time.time()
    results = defaultdict(list)
    for group_name, runs in run_groups.items():
        for run in runs:
            ms = benchmark_speed(
                direction=mode,
                fn=run,
                create_inputs=lambda **kw: {},
                create_inputs_kwargs={},
                compile=compile,
                num1=3,
                num2=10,
                warmup=1,
            )
            results[group_name].append(ms)

    # Create table with results
    headers = [param_name] + list(results.keys())
    table_data = []
    for i, param in enumerate(param_range):
        row = [param]
        for group in results.keys():
            row.append(f"{results[group][i]:.2f}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".3f"))
    print("Unit in (ms)")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return results
    
def _profile_benchmark(run_groups: Dict[str, Iterator[Callable[[], Any]]], param_range: List[Any]):
    """ Run the given run groups.
    """
    for group_name, group in run_groups.items():
        for run in group.make_runs(param_range):
            run()

@click.group()
def bench():
    pass

# Available benchmarks
@bench.command()
@click.option('--b', type=int, default=1)
@click.option('--h', type=int, default=1)
@click.option('--d', type=int, default=64)
@click.option('--dtype', type=str, default='float16')
@click.option('--device', type=str, default='cuda')
@click.option('--deg', type=int, default=2)
def expansion_speed(b: int, h: int, d: int, dtype: str, device: str, deg: int):
    """Comparing different expansion implementations"""
    fixed_kwargs = {
        'b': b,
        'h': h,
        'd': d,
        'dtype': str_to_dtype(dtype),
        'device': device,
        'deg': deg
    }
    t_range = [2**i for i in range(10, 17)]
    run_groups = {
        'triton': triton_expansion_context_scaling(t_range, **fixed_kwargs),
        'mosaic': mosaic_expansion_context_scaling(t_range, **fixed_kwargs),
    }
    print("Speed benchmark for expansion implementations")
    print(f"{b = }, {h = }, {d = }, {dtype = }, {device = }, {deg = }")
    print("------------------------------------------")
    return _speed_benchmark(run_groups, 'ctx', t_range)

@bench.command()
@click.option('--tokens', type=int, default=65536)
@click.option('--h', type=int, default=1)
@click.option('--d', type=int, default=64)
@click.option('--dtype', type=str, default='float16')
@click.option('--device', type=str, default='cuda')
@click.option('--deg', type=int, default=2)
@click.option('--chunk_size', type=int, default=1024)
@click.option('--gating', type=bool, default=False)
@click.option('--mode', type=str, default='fwd', help='fwd or bwd or fwd+bwd')
@click.option('--compile', type=bool, default=True)
def context_scaling(tokens: int, h: int, d: int, dtype: str, device: str, deg: int, chunk_size: int, gating: bool, mode: str, compile: bool):
    """Comparing the speed of all implementations"""
    fixed_kwargs = {
        'h': h,
        'd': d,
        'dtype': str_to_dtype(dtype),
        'device': device,
        'deg': deg,
        'chunk_size': chunk_size,
        'gating': gating,
        'requires_grad': 'bwd' in mode
    }
    ctx_range = [2**i for i in range(10, 17)]
    ctx_range = [ctx for ctx in ctx_range if ctx >= chunk_size*4 and ctx <= tokens]
    if not ctx_range:
        raise ValueError(f"No valid context size found. Please try a different chunk size or number of tokens.")
    batch_sizes = [tokens // ctx for ctx in ctx_range]
    params_range = list(zip(batch_sizes, ctx_range))
    run_groups = {
        'sdpa': sdpa_context_scaling(params_range, **fixed_kwargs),
        'fla': fla_context_scaling(params_range, **fixed_kwargs),
        'power': power_context_scaling(params_range, **fixed_kwargs),
        'power_triton': power_triton_context_scaling(params_range, **fixed_kwargs),
    }
    print("Speed benchmark for context scaling")
    print(f"{tokens = }, {h = }, {d = }, {dtype = }, {device = }, {deg = }, {chunk_size = }, {gating = }, {mode = }, {compile = }")
    print("------------------------------------------")
    return _speed_benchmark(run_groups, 'ctx', ctx_range)


@bench.command()
@click.option('--b', type=int, default=4)
@click.option('--t', type=int, default=8192)
@click.option('--h', type=int, default=8)
@click.option('--dtype', type=str, default='float16')
@click.option('--device', type=str, default='cuda')
@click.option('--chunk_size', type=int, default=1024)
@click.option('--gating', type=bool, default=False)
@click.option('--mode', type=str, default='fwd', help='fwd or bwd or fwd+bwd')
@click.option('--compile', type=bool, default=True)
def dim_scaling(b: int, t: int, h: int, dtype: str, device: str, chunk_size: int, gating: bool, mode: str, compile: bool):
    """Comparing the speed of all implementations"""
    fixed_kwargs = {
        'h': h,
        'b': b,
        't': t,
        'dtype': str_to_dtype(dtype),
        'device': device,
        'chunk_size': chunk_size,
        'gating': gating,
        'requires_grad': 'bwd' in mode
    }
    d_range = [32, 64, 128]
    run_groups = {
        'sdpa': sdpa_headdim_scaling(d_range, **fixed_kwargs),
        'fla': fla_headdim_scaling(d_range, **fixed_kwargs),
        'power_p1': power_triton_headdim_scaling(d_range, **{**fixed_kwargs, 'deg': 1}),
        'power_p2': power_triton_headdim_scaling(d_range, **{**fixed_kwargs, 'deg': 2}),
        'power_triton_p2': power_triton_headdim_scaling(d_range, **{**fixed_kwargs, 'deg': 2}),
    }
    print("Speed benchmark for head dimension scaling")
    print(f"{b = }, {t = }, {h = }, {dtype = }, {device = }, {chunk_size = }, {gating = }, {mode = }, {compile = }")
    print("------------------------------------------")
    return _speed_benchmark(run_groups, 'd', d_range)


PROFILABLE_KERNELS = ['sdpa', 'fla', 'power', 'power_triton', 'query_state', 'query_state_triton',
                     'update_state', 'update_state_triton', 'expand_triton', 'expand_mosaic',
                     'discumsum', 'power_attention', 'power_attention_triton']
@bench.command()
@click.argument('kernel', type=str, default='sdpa')
@click.option('--b', type=int, default=4)
@click.option('--t', type=int, default=8192)
@click.option('--h', type=int, default=1)
@click.option('--d', type=int, default=64)
@click.option('--dtype', type=str, default='float16')
@click.option('--device', type=str, default='cuda')
@click.option('--deg', type=int, default=2)
@click.option('--chunk_size', type=int, default=1024)
@click.option('--gating', type=bool, default=False)
@click.option('--mode', type=str, default='fwd', help='fwd or bwd or fwd+bwd')
@click.option('--compile', type=bool, default=True)
@click.option('--measure', is_flag=True, default=False, help='Run the kernel multiple times to measure the speed')
def run(kernel: str, b: int, t: int, h: int, d: int, dtype: str, device: str, deg: int, chunk_size: int, gating: bool, mode: str, compile: bool, measure: bool):
    """Run the given kernel, optionally measuring the speed."""
    if kernel not in PROFILABLE_KERNELS:
        raise ValueError(f"Invalid kernel: {kernel}. Available kernels: {', '.join(PROFILABLE_KERNELS)}")
    
    fixed_kwargs = {
        'b': b,
        't': t,
        'h': h,
        'd': d,
        'dtype': str_to_dtype(dtype),
        'device': device,
        'deg': deg,
        'chunk_size': chunk_size,
        'gating': gating
    }
    run = {
        'sdpa': SDPA.make_run(**fixed_kwargs),
        'fla': FLA.make_run(**fixed_kwargs),
        'power': Power.make_run(**fixed_kwargs),
        'power_triton': PowerTriton.make_run(**fixed_kwargs),
        'query_state': QueryState.make_run(**fixed_kwargs),
        'query_state_triton': QueryStateTriton.make_run(**fixed_kwargs),
        'update_state': UpdateState.make_run(**fixed_kwargs),
        'update_state_triton': UpdateStateTriton.make_run(**fixed_kwargs),
        'power_attention': PowerAttention.make_run(**fixed_kwargs),
        'power_attention_triton': PowerAttentionTriton.make_run(**fixed_kwargs),
        'expand_triton': TritonExpansion.make_run(**fixed_kwargs),
        'expand_mosaic': MosaicExpansion.make_run(**fixed_kwargs),
    }[kernel]
    print(f"Running {kernel} with {b = }, {t = }, {h = }, {d = }, {dtype = }, {device = }, {deg = }, {chunk_size = }, {gating = }, {mode = }, {compile = }")
    if measure:
        ms = benchmark_speed(
            direction=mode,
            fn=run,
            create_inputs=lambda **kw: {},
            create_inputs_kwargs={},
            compile=compile,
            num1=3,
            num2=10,
            warmup=1,
        )
        print(f"Run time: {ms:.2f} ms")
    else:
        run()


if __name__ == '__main__':
    bench()
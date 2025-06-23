""" This script enables running things we care about
"""
import torch
import copy
import time
import argparse
from pathlib import Path
from perf._timing import benchmark_speed
from collections import defaultdict
from typing import Dict, List, Any, Callable, Iterator
from tabulate import tabulate
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

from .db import KVDB
from .benchmarks.runs import *  # noqa
from power_attention import default_D
from vidrial.mosaic.utils.gpu import get_cuda_device_basic_props

benchmark_db = KVDB(os.path.expanduser('~/.power-attention-benchmark.db'))
logger = logging.getLogger(__name__)
plots_dir = Path(__file__).parent.parent / 'plots/benchmark_results'

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


PROFILABLE_RUNS = {
    'sdpa': SDPA.make_run,
    'power_full_triton': PowerFullTriton.make_run,
    'power_full_vidrial': PowerFullVidrial.make_run,
    'query_state_triton': QueryStateTriton.make_run,
    'query_state_vidrial': QueryStateVidrial.make_run,
    'update_state_triton': UpdateStateTriton.make_run,
    'update_state_vidrial': UpdateStateVidrial.make_run,
    'power_attention_cuda': PowerAttentionCuda.make_run,
    'power_attention_triton': PowerAttentionTriton.make_run,
    'flash_attn': FlashAttn.make_run,
    'discumsum': Discumsum.make_run,
}


def run(kernel: str, b: int = 2, t: int = 4096, n: int = 8, h: int = 1, d: int = 64, dtype: str = 'float16', device: str = 'cuda', deg: int = 2, chunk_size: int = 1024, gating: bool = True, mode: str = 'fwd', norm: bool = True, compile: bool = True, measure: bool = True):
    """Run the given kernel, optionally measuring the speed."""
    fixed_kwargs = {
        'b': b,
        't': t,
        'c': t,
        'h': h,
        'd': d,
        'n': n,
        'dtype': dtype,
        'device': device,
        'deg': deg,
        'norm': norm,
        'D': default_D(d, deg),
        'chunk_size': chunk_size,
        'gating': gating,
        'requires_grad': 'bwd' in mode,
    }
    key = copy.deepcopy(fixed_kwargs) | {'mode': mode, 'compile': compile, 'kernel': kernel}
    if os.environ.get('UPDATE_DB', '1') == '0' and key in benchmark_db:
        logger.info(f"Using cached result for {key}")
        return benchmark_db.get(lambda k: k == key)[0][1]
    fixed_kwargs['dtype'] = str_to_dtype(fixed_kwargs['dtype'])
    run = PROFILABLE_RUNS[kernel](**fixed_kwargs)
    logger.info(f"Running {kernel} with {b = }, {t = }, {h = }, {d = }, {n = }, {dtype = }, {device = }, {deg = }, {chunk_size = }, {gating = }, {mode = }, {compile = }")
    if measure:
        start = time.time()
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
        logger.info(f"Run time: {ms:.2f} ms, time taken: {time.time() - start:.2f} seconds")
        benchmark_db.put(key, ms)
        return ms
    else:
        run()


def plot_problem(b: int, t: int, n: int, h: int, d: int, dtype: str, device: str, deg: int, chunk_size: int, gating: bool, mode: str, compile: bool, measure: bool, impl: str):
    kwargs = {
        'b': b,
        't': t,
        'n': n,
        'h': h,
        'd': d,
        'dtype': dtype,
        'device': device,
        'deg': deg,
        'chunk_size': chunk_size,
        'gating': gating,
        'mode': mode,
        'compile': compile,
        'measure': measure,
    }
    problem_str = f"b_{b}_t_{t}_n_{n}_h_{h}_d_{d}_dtype_{dtype}_device_{device}_deg_{deg}_chunk_size_{chunk_size}_gating_{gating}_mode_{mode}_compile_{compile}"
    power_full_triton = run('power_full_triton', **kwargs)
    query_state_triton = run('query_state_triton', **{**kwargs, 't': chunk_size})
    query_state_vidrial = run('query_state_vidrial', **{**kwargs, 't': chunk_size})
    power_full_vidrial = run('power_full_vidrial', **kwargs)
    update_state_triton = run('update_state_triton', **{**kwargs, 't': chunk_size})
    update_state_vidrial = run('update_state_vidrial', **{**kwargs, 't': chunk_size})
    discumsum = run('discumsum', **kwargs)
    chunked_attention_triton = run('power_attention_triton', **{**kwargs, 'b': b*n, 't': chunk_size, 'norm': False})
    power_attn_triton = run('power_attention_triton', **kwargs)
    power_attn_cuda = run('power_attention_cuda', **kwargs)
    sdpa = run('sdpa', **kwargs)
    flash_attn = run('flash_attn', **kwargs)


    # Plot the results using a categorical bar chart
    data = {
        'Implementation': [
            'Power Full (Triton)', 'Power Full (Vidrial)',
            'Power Full Breakdown (Triton)', 'Power Full Breakdown (Vidrial)',
            'Power Attention (CUDA)', 'Power Attention (Triton)',
            'SDPA', 'Flash Attention',
        ],
        'Query State (ms)': [0, 0, query_state_triton, query_state_vidrial, 0, 0, 0, 0],
        'Update State (ms)': [0, 0, update_state_triton, update_state_vidrial, 0, 0, 0, 0],
        'Chunked Attention (ms)': [0, 0, chunked_attention_triton, chunked_attention_triton, 0, 0, 0, 0],
        'Discumsum (ms)': [0, 0, discumsum, discumsum, 0, 0, 0, 0],
        'Total (ms)': [
            power_full_triton, power_full_vidrial,
            0, 0,
            power_attn_cuda, power_attn_triton,
            sdpa, flash_attn,
        ]
    }
    df = pd.DataFrame(data)
    df = df.set_index('Implementation')
    if impl == 'default':
        df = df.drop(index=['Power Full Breakdown (Vidrial)', 'Power Full (Vidrial)', 'Power Attention (CUDA)', 'Power Attention (Triton)'])
    ax = df.plot(kind='bar', stacked=True, figsize=(12, 7))
    plt.xticks(rotation=15, ha='right')
    gpu_name = get_cuda_device_basic_props()[0]['name'].replace(' ', '_')
    plt.ylabel('Time (milliseconds)')
    plt.title('Performance Comparison of Different Implementations')

    # Add value labels on top of each bar
    for i in range(len(df.index)):
        total = df.iloc[i].sum()
        if total > 0:  # Only add label if bar has height
            ax.text(i, total, f'{total:.1f}ms',
                   ha='center', va='bottom')
    
    # Add problem details as text below the graph
    problem_details = f'Problem: batch={b}, seq_len={t}, heads={h}, head_dim={d}, n_chunks={n}, dtype={dtype}, deg={deg}, chunk_size={chunk_size}, gating={gating}, mode={mode}, compile={compile}'
    hardware_details = f'Hardware: {gpu_name.replace("_", " ")}'
    plt.figtext(0.5, 0.00, f'{problem_details}\n{hardware_details}', 
                ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Make room for the text below
    fname = f'implementation_comparison/{problem_str}_{gpu_name}.png'
    plt.savefig(plots_dir / fname, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {plots_dir / fname}")


def plot_throughput_by_ctx(b: int, ts: list[int], n: int, h: int, d: int, dtype: str, device: str, deg: int, chunk_size: int, gating: bool, mode: str, compile: bool, impl: str):
    data = defaultdict(list)
    kwargs = {
        'b': b,
        'n': n,
        'h': h,
        'd': d,
        'dtype': dtype,
        'device': device,
        'deg': deg,
        'chunk_size': chunk_size,
        'gating': gating,
        'mode': mode,
        'compile': compile,
    }
    for t in ts:
        kwargs['t'] = t
        data['ctx'].append(t)
        data['tokens'].append(b*t)
        data['power_attention_triton'].append(run('power_attention_triton', **kwargs, measure=True))
        data['sdpa'].append(run('sdpa', **kwargs, measure=True))
        data['flash_attn'].append(run('flash_attn', **kwargs, measure=True))
        data['power_attention_cuda'].append(run('power_attention_cuda', **kwargs, measure=True))
        data['power_full'].append(run('power_full_triton', **kwargs, measure=True))
        data['power_full_vidrial'].append(run('power_full_vidrial', **kwargs, measure=True))


    df = pd.DataFrame(data)
    for kernel in ['power_attention_triton', 'sdpa', 'flash_attn', 'power_attention_cuda', 'power_full', 'power_full_vidrial']:
        df[kernel] = df['tokens'] / df[kernel]
    del df['tokens']
    
    df = df.set_index(['ctx'])
    if impl == 'default':
        df = df.drop(columns=['power_full_vidrial', 'power_attention_cuda', 'power_attention_triton'])
    ax = df.plot(kind='line', figsize=(12, 7), linewidth=3, marker='o')
    plt.ylabel('Throughput (tokens/s)')
    gpu_name = get_cuda_device_basic_props()[0]['name'].replace(' ', '_')
    plt.title('Throughput vs Context Length Comparison')

    # Add problem details as text below the graph
    problem_details = f'Problem: batch={b}, heads={h}, head_dim={d}, dtype={dtype}, deg={deg}, chunk_size={chunk_size}, gating={gating}, mode={mode}, compile={compile}'
    hardware_details = f'Hardware: {gpu_name.replace("_", " ")}'
    plt.figtext(0.5, 0.02, f'{problem_details}\n{hardware_details}', 
                ha='center', va='bottom', fontsize=9, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text below
    fig_name = f'throughput_by_ctx/b_{b}_n_{n}_h_{h}_d_{d}_dtype_{dtype}_device_{device}_deg_{deg}_chunk_size_{chunk_size}_gating_{gating}_mode_{mode}_compile_{compile}'
    plt.savefig(plots_dir / f'{fig_name}_{gpu_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {plots_dir / f'{fig_name}_{gpu_name}.png'}")


def main():
    """Main CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Power Attention Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run forward pass benchmark
  python benchmark.py fwd

  # Run backward pass benchmark  
  python benchmark.py bwd

  # Run forward + backward benchmark
  python benchmark.py fwdbwd

  # Run with custom parameters
  python benchmark.py fwd --b 4 --t 8192
        """
    )
    
    parser.add_argument('mode', choices=['fwd', 'bwd', 'fwd+bwd'], 
                       help='Execution mode')
    parser.add_argument('--b', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--t', type=int, default=65536, help='Sequence length (default: 65536)')
    parser.add_argument('--h', type=int, default=8, help='Number of heads (default: 8)')
    parser.add_argument('--d', type=int, default=64, help='Head dimension (default: 64)')
    parser.add_argument('--dtype', type=str, default='bfloat16', 
                       choices=['float16', 'float32', 'bfloat16'],
                       help='Data type (default: bfloat16)')
    parser.add_argument('--deg', type=int, default=2, help='Degree (default: 2)')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Chunk size (default: 1024)')
    parser.add_argument('--impl', type=str, default='default', help='Implementations to compare', choices=['default', 'all'])
    
    args = parser.parse_args()
    
    # Calculate n based on t and chunk_size
    n = args.t // args.chunk_size
    
    # Run the benchmark for a single problem
    plot_problem(
        b=args.b,
        t=args.t,
        n=n,
        h=args.h,
        d=args.d,
        dtype=args.dtype,
        device='cuda',
        deg=args.deg,
        chunk_size=args.chunk_size,
        gating=True,
        mode=args.mode,
        compile=True,
        measure=True,
        impl=args.impl,
    )

    # Run ctx benchmark
    plot_throughput_by_ctx(
        b=args.b,
        ts=[4096, 8192, 16384, 32768, 65536],
        n=n,
        h=args.h,
        d=args.d,
        dtype=args.dtype,
        device='cuda',
        deg=args.deg,
        chunk_size=args.chunk_size,
        gating=True,
        mode=args.mode,
        compile=True,
        impl=args.impl,
    )


if __name__ == '__main__':
    main()

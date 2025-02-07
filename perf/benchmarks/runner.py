""" This script enables running things we care about
"""
import torch
import torch.nn.functional as F
import argparse
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from power_attention_cuda import mosaic_sympow
from power_attention.power_full import power_full, power_full_triton
from power_attention._expansion.impl_triton import expand as triton_expand
from abc import ABC, abstractmethod
from perf._timing import benchmark_speed
from dataclasses import dataclass
from typing import Dict, List, Type, Any
from tabulate import tabulate

class Run(ABC):
    """ A class that can be used to run a given algorithm.
    """
    @abstractmethod
    def create_inputs(self, **kw):
        pass

    @abstractmethod
    def __call__(self, **kw):
        pass


class SDPA(Run):
    def create_inputs(self, **kw):
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

        return dict(
            query=q,
            key=k,
            value=v,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    def __call__(self, **inputs):
        return torch.nn.functional.scaled_dot_product_attention(**inputs)


class FLA(Run):
    def create_inputs(self, **kw):
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
            self.fn = fused_chunk_linear_attn
        else:
            self.fn = chunk_linear_attn
        
        return dict(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            normalize=normalize,
            head_first=head_first,
        )

    def __call__(self, **inputs):
        return self.fn(**inputs)

class Power(Run):
    def create_inputs(self, **kw):
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

        return dict(
            Q=Q,
            K=K,
            V=V,
            log_G=log_G,
            deg=deg,
            scale=scale,
            chunk_size=chunk_size,
        )

    def __call__(self, **inputs):
        return power_full(**inputs)
    

class PowerTriton(Run):
    def create_inputs(self, **kw):
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

        return dict(
            Q=Q,
            K=K,
            V=V,
            log_G=log_G,
            deg=deg,
            scale=scale,
            chunk_size=chunk_size,
        )
    
    def __call__(self, **inputs):
        return power_full_triton(**inputs)


class TritonExpansion(Run):
    def create_inputs(self, **kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        split = kw.get('split', 'T')

        K = torch.randn((b, 1, t, h, d), dtype=dtype, device=device, requires_grad=requires_grad)

        return dict(K=K, deg=deg, split=split)
    
    def __call__(self, **inputs):
        phi_K = triton_expand(**inputs)
        return phi_K


class MosaicExpansion(Run):
    def create_inputs(self, **kw):
        b, t, h, d, dtype, device = kw['b'], kw['t'], kw['h'], kw['d'], kw['dtype'], kw['device']
        deg = kw.get('deg', 2)
        requires_grad = kw.get('requires_grad', False)
        d_block = 8

        K = torch.randn((t, d), dtype=dtype, device=device, requires_grad=requires_grad)
        K = K.transpose(0, 1).contiguous().transpose(0, 1)

        return dict(K=K, deg=deg, d_block=d_block)
    
    def __call__(self, **inputs):
        phi_K = mosaic_sympow(inputs['K'], inputs['deg'], inputs['d_block'])[0]
        return phi_K
        

# Dictionary mapping run names to their classes
AVAILABLE_RUNS = {
    'sdpa': SDPA,
    'fla': FLA,
    'power': Power,
    'power_triton': PowerTriton,
    'triton_expansion': TritonExpansion,
    'mosaic_expansion': MosaicExpansion,
}


def list_runs():
    """Print available runs and their descriptions"""
    print("\nAvailable runs:")
    for name, run_class in AVAILABLE_RUNS.items():
        print(f"  {name}: {run_class.__doc__ or 'No description available'}")


def start_run(run_class, mode, measure=False, compile=False, **kwargs):
    """Start a single run with optional timing measurement and compilation
    
    Args:
        run_class: The Run class to instantiate
        mode: One of 'fwd', 'bwd', 'fwd+bwd'
        measure: Whether to measure execution time
        compile: Whether to use torch.compile
        **kwargs: Arguments to pass to create_inputs
        
    Returns:
        float or None: Time in milliseconds if measure=True, None otherwise
    """
    run_instance = run_class()
    
    # Create inputs with requires_grad based on mode
    inputs = run_instance.create_inputs(**kwargs, requires_grad=(mode != "fwd"))
    
    # Compile the run if requested
    if compile:
        original_call = run_instance.__call__
        run_instance.__call__ = torch.compile(original_call, dynamic=False)
    
    if measure:
        # Use benchmark_speed for timing measurement
        time_ms = benchmark_speed(
            direction=mode,
            fn=run_instance.__call__,
            create_inputs=run_instance.create_inputs,
            create_inputs_kwargs=kwargs,
            compile=compile
        )
        return time_ms
    else:
        # Just run once
        out = run_instance(**inputs)
        if mode != "fwd":
            if isinstance(out, tuple):
                loss = out[0].sum()
            else:
                loss = out.sum()
            loss.backward()
        return None


@dataclass
class BenchmarkGroup:
    """A group in a benchmark, representing one implementation to test"""
    name: str
    run_class: Type[Run]
    fixed_kwargs: Dict[str, Any]

@dataclass
class Benchmark:
    """A benchmark that compares multiple implementations across a parameter range"""
    name: str
    groups: List[BenchmarkGroup]
    param_name: str
    param_range: List[Any]
    fixed_kwargs: Dict[str, Any]
    
    def run(self, mode='fwd', compile=False):
        """Run the benchmark and return results"""
        results = []
        
        for param_value in self.param_range:
            row = [param_value]
            
            for group in self.groups:
                # Merge kwargs with param value
                kwargs = {
                    **self.fixed_kwargs,
                    **group.fixed_kwargs,
                    self.param_name: param_value
                }
                # Use start_run to handle both measurement and execution
                time_ms = start_run(group.run_class, mode, measure=True, compile=compile, **kwargs)
                row.append(time_ms)
            
            results.append(row)
        
        return results

# Available benchmarks
def create_mosaic_expand_benchmark():
    """Create benchmark comparing Triton and Mosaic expansion implementations"""
    return Benchmark(
        name="mosaic_expand",
        groups=[
            BenchmarkGroup(
                name="triton",
                run_class=TritonExpansion,
                fixed_kwargs={}
            ),
            BenchmarkGroup(
                name="mosaic",
                run_class=MosaicExpansion,
                fixed_kwargs={}
            )
        ],
        param_name="t",
        param_range=[2**i for i in range(10, 17)],  # 1024 to 65536
        fixed_kwargs={
            'b': 1,
            'h': 1,
            'd': 64,
            'dtype': torch.float16,
            'device': 'cuda',
            'deg': 2
        }
    )

AVAILABLE_BENCHMARKS = {
    'mosaic_expand': create_mosaic_expand_benchmark
}

def run_benchmark(benchmark_name: str, mode='fwd', compile=False):
    """Run a predefined benchmark and display results"""
    if benchmark_name not in AVAILABLE_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    benchmark = AVAILABLE_BENCHMARKS[benchmark_name]()
    results = benchmark.run(mode=mode, compile=compile)
    
    # Prepare headers for the table
    headers = [benchmark.param_name] + [group.name for group in benchmark.groups]
    
    # Print results using tabulate
    print(f"\nResults for {benchmark.name} benchmark:")
    print(tabulate(results, headers=headers, floatfmt=".3f"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run and benchmark attention implementations")
    parser.add_argument('--list', action='store_true', help='List available runs')
    parser.add_argument('--run', type=str, choices=list(AVAILABLE_RUNS.keys()), help='Run to execute')
    parser.add_argument('--benchmark', type=str, choices=list(AVAILABLE_BENCHMARKS.keys()), help='Run a predefined benchmark')
    parser.add_argument('--mode', type=str, default='fwd', choices=['fwd', 'bwd', 'fwd+bwd'], help='Run mode (default: fwd)')
    parser.add_argument('--measure', action='store_true', help='Measure execution time (default: False)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile on the run (default: False)')
    
    # Input parameters
    parser.add_argument('--b', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--t', type=int, default=1024, help='Sequence length (default: 1024)')
    parser.add_argument('--h', type=int, default=16, help='Number of heads (default: 16)')
    parser.add_argument('--d', type=int, default=64, help='Head dimension (default: 64)')
    parser.add_argument('--dtype', type=lambda x: getattr(torch, x), default=torch.float16, help='Data type (e.g., bfloat16, float16, float32) (default: float16)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (default: cuda)')
    
    # Additional parameters
    parser.add_argument('--deg', type=int, default=2, help='Degree for power attention (default: 2)')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for chunked attention (default: None)')
    parser.add_argument('--gating', action='store_true', help='Enable gating (default: False)')
    parser.add_argument('--scale', type=float, default=1.0, help='Attention scale factor (default: 1.0)')
    parser.add_argument('--fused', action='store_true', help='Use fused implementation where available (default: False)')
    
    args = parser.parse_args()
    
    if args.list:
        list_runs()
    elif args.benchmark:
        run_benchmark(args.benchmark, mode=args.mode, compile=args.compile)
    elif args.run:
        run_class = AVAILABLE_RUNS[args.run]
        kwargs = {
            'b': args.b,
            't': args.t,
            'h': args.h,
            'd': args.d,
            'dtype': args.dtype,
            'device': args.device,
            'deg': args.deg,
            'chunk_size': args.chunk_size,
            'gating': args.gating,
            'scale': args.scale,
            'fused': args.fused,
        }
        start_run(run_class, args.mode, args.measure, args.compile, **kwargs)
    else:
        parser.print_help()
    
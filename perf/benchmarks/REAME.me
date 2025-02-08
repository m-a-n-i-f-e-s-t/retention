# How do we benchmark?

```bash
python benchmarks.py <benchmark_name>
```

# Motivation

We have a lot of things we care about during development. For example, the speed of the kernel under different contexts (input, machine, etc.), the speed of various components, the accuracy of kernels, comparison with other alternative attention mechanisms, etc. Therefore some abstractions are in order.

# Run

A run is **a callable that does a thing**. Specifically, it is a class that implements the `make_run` method, which takes in a set of keyword arguments and return a argument-less callable. 

```python
class Run():
    @staticmethod
    def make_run(self, **kw) -> Callable[[], Any]:
        pass
```


```python
class SDPA():
    @staticmethod
    def make_run(self, **kw):
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
```

# RunGroup

**A `RunGroup` is a set of runs grouped together for a reason.** Specifically, it is a function that returns a group of runs. For example, a `RunGroup` can return a group of SDPA runs with different context.

For example, to create a `RunGroup` of SDPA runs with different context, we do:

```python
def sdpa_context_scaling(ctx_range: List[int], **kw):
    for ctx in ctx_range:
        yield SDPA().make_run(t=ctx, **kw)
```

# Benchmark

**A `Benchmark` is a study on the behavior of a set of `RunGroup`s.** Specifically, it is a function that does something with a collection of `RunGroup`s. Common benchmarks include measuring the speed of a set of runs, so it has been abstracted out into utility functions.

There're currently two types of benchmarks: speed benchmark and profile benchmark. Speed benchmark will measure the speed of all the runs in the `RunGroup`s, while profile benchmark will simply run the runs.


# Usage

In additon to installing `power-attention`, you might want to install [fla](https://github.com/fla-org/flash-linear-attention).

```bash
pip install -U git+https://github.com/fla-org/flash-linear-attention
```

To see what benchmarks are available, run

```bash
python benchmarks.py --help
```

To run benchmarks, simply run

```bash
python benchmarks.py <benchmark_name>
```


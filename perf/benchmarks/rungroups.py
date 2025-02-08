""" This module contains the definition of run groups, which are groups of runs that share the same input variation. """
from typing import List, Tuple
from runs import SDPA, FLA, Power, PowerTriton, TritonExpansion, MosaicExpansion


def sdpa_context_scaling(params_range: List[Tuple[int, int]], **kw):
    for b, ctx in params_range:
        yield SDPA.make_run(b=b, t=ctx, **kw)

def sdpa_headdim_scaling(params_range: List[Tuple[int, int]], **kw):
    for d in params_range:
        yield SDPA.make_run(d=d, **kw)
    
def fla_context_scaling(params_range: List[Tuple[int, int]], **kw):
    for b, ctx in params_range:
        yield FLA.make_run(b=b, t=ctx, **kw)

def fla_headdim_scaling(params_range: List[Tuple[int, int]], **kw):
    for d in params_range:
        yield FLA.make_run(d=d, **kw)

def power_context_scaling(params_range: List[Tuple[int, int]], **kw):
    for b, ctx in params_range:
        yield Power.make_run(b=b, t=ctx, **kw)

def power_headdim_scaling(params_range: List[Tuple[int, int]], **kw):
    for d in params_range:
        yield Power.make_run(d=d, **kw)

def power_triton_context_scaling(params_range: List[Tuple[int, int]], **kw):
    for b, ctx in params_range:
        yield PowerTriton.make_run(b=b, t=ctx, **kw)

def power_triton_headdim_scaling(params_range: List[Tuple[int, int]], **kw):
    for d in params_range:
        yield PowerTriton.make_run(d=d, **kw)

def mosaic_expansion_context_scaling(t_range: List[int], **kw):
    for t in t_range:
        yield MosaicExpansion.make_run(t=t, **kw)

def triton_expansion_context_scaling(t_range: List[int], **kw):
    for t in t_range:
        yield TritonExpansion.make_run(t=t, **kw)
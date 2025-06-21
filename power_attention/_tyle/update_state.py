from kernels import sympow_mma
import torch
import sys
from pathlib import Path

from mosaic.utils.common import sympow_dim
from power_attention_cuda import (
    InnerBlock_DT,
    OuterBlock_DT,
    InnerBlock_TD,
    OuterBlock_TD,
)

def update_state(K, V, deg):

    b, n, c, h, d = K.shape
    B = b * n * h
    K = K.transpose(2, 3).contiguous().view(B, c, d)
    V = V.transpose(2, 3).contiguous().view(B, c, d)
    K = K.transpose(1, 2)
    D = sympow_dim(d, deg, d_tile=InnerBlock_DT)

    #S = torch.empty((B, D, d), device=K.device, dtype=K.dtype)
    S = sympow_mma(K, V, power=deg, expand_dim=-2, d_tile=InnerBlock_DT)
    #s = torch.ones((B, D), device=K.device, dtype=K.dtype)
    s = sympow_mma(K, torch.ones((B, 1, 1), device=K.device, dtype=K.dtype), power=deg, expand_dim=-2, d_tile=InnerBlock_DT)

    return S.view(b, n, h, D, d), s.view(b, n, h, D)

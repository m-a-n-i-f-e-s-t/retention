# Installation

Power Attention can be installed either via pre-built wheels or built from source.

## From PyPI (Recommended)

```bash
pip install power-attention
```

## From Source

### Prerequisites
- Python 3.11 or 3.12 (3.13 depends on the upcoming [Triton 3.2 release](https://github.com/triton-lang/triton/issues/5215))
- CUDA Toolkit 12.4
- GCC/G++ with C++17 support
- Linux (Windows/MacOS not supported)

1. Clone the repository:
```bash
git clone https://github.com/manifest-ai/power-attention.git
cd power-attention
```

2. Install with development dependencies:
```bash
pip install -e .[dev]
```

All other dependencies (PyTorch, Ninja build system, etc.) will be automatically installed through pip.

## Build Configuration

When building from source, the build process can be customized using environment variables:

### Compilation Settings
- `NVCC_THREADS`: Number of threads for NVCC compilation (default: 4)
- `MAX_JOBS`: Maximum number of parallel compilation jobs (auto-configured based on CPU cores and memory)
- `NVCC_VERBOSE`: Enable verbose NVCC output and keep temporary files

### Fast Build Mode
For development, you can enable fast build mode which compiles only a subset of kernels:

```bash
export FAST_BUILD=1
export FAST_HEAD_DIM=64    # Head dimension [32, 64]
export FAST_DEG=4          # Power parameter p [1-4]
export FAST_STATE_DEG=2    # State power parameter [1-4]
export FAST_IS_FP16=false  # Use FP16 vs BF16 [true/false]
export FAST_IS_CAUSAL=true # Enable causal attention [true/false]
export FAST_GATING=true    # Enable gating mechanism [true/false]

pip install -e .[dev]
```

## Development Setup

The package uses pip's editable install mode for development. First, activate your Python virtual environment, then:

```bash
# Install base package in editable mode
pip install -e .

# Install development dependencies
pip install psutil
pip install flash_attn==2.7.3 --no-build-isolation
pip install -e .[dev]
```

## Verifying Installation

```python
import torch
from power_attention.power_full import create_inputs, power_full

t = 1024
chunk_size=128
b = 8
h = 16
d = 64
deg = 2
gating = True
dtype = torch.float16
inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)

output = power_full(**inputs)
torch.autograd.backward((output,), grad_tensors=(output,))

print("Ran power attention forwards & backwards, output shape:", output.shape)
```

## Training Example

To immediately see the kernel in action, `cd train` and use:

```bash
# Create the dataset first
python prepare_owt.py

# Single GPU training
python train.py \
  --batch_size=32 \
  --attention_kernel=power \
  --degree=2 \
  --chunk_size=128 \
  --disable_gating=False

# Multi-GPU training with DDP (example with 4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py \
  --batch_size=32 \
  --attention_kernel=power \
  --degree=2 \
  --chunk_size=128 \
  --disable_gating=False
```

For distributed training across multiple nodes:
```bash
# On the first (master) node with IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

# On the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

Note: If your cluster does not have Infiniband interconnect, prepend `NCCL_IB_DISABLE=1` to the commands.

## Troubleshooting

### CUDA/GPU Issues
1. Check CUDA toolkit version:
```bash
nvcc --version
```

2. Check GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

3. Verify PyTorch CUDA:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Memory Issues
- Reduce batch size or sequence length
- Use chunking with smaller chunk sizes
- Try fast build with lower degree parameters

For more help, check [GitHub issues](https://github.com/manifest-ai/power-attention/issues).

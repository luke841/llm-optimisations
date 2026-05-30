# CUDA GPU Programming Demos

Progressive examples that build intuition for how GPUs execute compute workloads.

## Prerequisites

- NVIDIA GPU (any recent one works)
- CUDA Toolkit installed (`nvcc` on your PATH)
- Linux or WSL recommended (macOS does not support CUDA)

```bash
# Check your setup
nvcc --version
nvidia-smi
```

## Files

| File | Concept | Key Takeaway |
|------|---------|--------------|
| `01_vector_add.cu` | Kernel launch, thread indexing, host/device memory | GPU = thousands of threads doing simple work in parallel |
| `02_matmul_naive.cu` | 2D grids, timing, GFLOPS measurement | Naive approach hammers global memory — performance is memory-bound |
| `03_matmul_tiled.cu` | Shared memory, tiling, `__syncthreads()` | Shared memory reduces global reads by TILE_SIZE×, massive speedup |

## Build & Run

```bash
make all
./vector_add
./matmul_naive
./matmul_tiled
```

## GPU Memory Hierarchy (what these demos illustrate)

```
┌─────────────────────────────────────────────────┐
│  Global Memory (HBM)                            │
│  - Large (16-80 GB)                             │
│  - Slow (~400 cycles latency)                   │
│  - All threads can access                       │
├─────────────────────────────────────────────────┤
│  Shared Memory (on-chip SRAM, per block)        │
│  - Small (48-164 KB per SM)                     │
│  - Fast (~5 cycles latency)                     │
│  - Only threads in same block can access        │
├─────────────────────────────────────────────────┤
│  Registers (per thread)                         │
│  - Tiny (255 max per thread)                    │
│  - Instant (0 cycles)                           │
│  - Private to each thread                       │
└─────────────────────────────────────────────────┘
```

## Concepts Progression

### 01 — Vector Addition
- **Thread model**: 1 thread = 1 output element
- **Memory**: simple host→device→host copy
- **Launch config**: 1D grid of 1D blocks

### 02 — Naive Matrix Multiply
- **Thread model**: 1 thread = 1 output element (dot product)
- **Problem**: each thread reads 2K floats from slow global memory
- **Observation**: neighboring threads re-read the same data

### 03 — Tiled Matrix Multiply
- **Solution**: load tiles into shared memory collaboratively
- **Pattern**: load → barrier → compute → barrier → repeat
- **Result**: TILE_SIZE× fewer global memory accesses

This tiling pattern is the foundation of every high-performance GPU kernel in LLM inference (FlashAttention, cuBLAS GEMM, etc).

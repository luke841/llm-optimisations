/*
 * 05_profiling_guide.cu — Designed to show interesting things in Nsight profilers
 *
 * Contains three kernels with deliberately different performance characteristics:
 *   1. Compute-bound (lots of math, little memory)
 *   2. Memory-bound (little math, lots of memory access)
 *   3. Latency-bound (divergent branches, poor occupancy)
 *
 * Profile with:
 *   ncu --set full ./profiling_guide        # kernel-level metrics
 *   nsys profile --stats=true ./profiling_guide   # system timeline
 *
 * Compile: nvcc -o profiling_guide 05_profiling_guide.cu -lineinfo
 *   (-lineinfo lets the profiler map metrics back to source lines)
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

#define N (1 << 22)  // 4M elements

// ---------------------------------------------------------------------------
// Kernel 1: COMPUTE-BOUND — many FLOPs per byte loaded
// ---------------------------------------------------------------------------
// Nsight Compute will show:
//   - High "Compute (SM) Throughput" (>70%)
//   - Low "Memory Throughput" (<30%)
//   - Bottleneck: "Math Pipe" or "SM"
__global__ void compute_bound(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // 100 FLOPs per element loaded — very high arithmetic intensity
        for (int i = 0; i < 50; i++) {
            val = val * 1.00001f + 0.00001f;
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: MEMORY-BOUND — few FLOPs per byte loaded
// ---------------------------------------------------------------------------
// Nsight Compute will show:
//   - Low "Compute (SM) Throughput" (<30%)
//   - High "Memory Throughput" (>70%)
//   - Bottleneck: "Memory" or "L2"
__global__ void memory_bound(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 1 FLOP per 8 bytes (load + store) — very low arithmetic intensity
        out[idx] = in[idx] + 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: LATENCY-BOUND — warp divergence and poor utilization
// ---------------------------------------------------------------------------
// Nsight Compute will show:
//   - Low both compute AND memory throughput
//   - High "Stall: Not Selected" or "Stall: Wait"
//   - Warp divergence in "Branch Efficiency"
__global__ void latency_bound(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Divergent branch: threads in the same warp take different paths.
        // The warp must execute BOTH paths serially (threads not on active path are masked).
        if (idx % 2 == 0) {
            for (int i = 0; i < 20; i++) val = val * 1.001f + 0.001f;
        } else {
            for (int i = 0; i < 5; i++) val = val * 0.999f - 0.001f;
        }
        out[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Main — runs all three with markers for Nsight Systems timeline
// ---------------------------------------------------------------------------
int main() {
    size_t bytes = N * sizeof(float);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer_create(&timer);

    printf("Running 3 kernels with different bottlenecks...\n");
    printf("Profile with: ncu --set full ./profiling_guide\n\n");

    // --- Kernel 1: Compute-bound ---
    timer_start(&timer);
    compute_bound<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK_KERNEL();
    float ms1 = timer_stop(&timer);

    long long flops1 = (long long)N * 100;  // 50 iterations x 2 ops
    report_gflops("compute_bound", flops1, ms1);
    report_roofline("compute_bound", flops1, 2 * bytes, ms1);

    // --- Kernel 2: Memory-bound ---
    timer_start(&timer);
    memory_bound<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK_KERNEL();
    float ms2 = timer_stop(&timer);

    long long flops2 = (long long)N * 1;
    report_gflops("memory_bound", flops2, ms2);
    report_roofline("memory_bound", flops2, 2 * bytes, ms2);

    // --- Kernel 3: Latency-bound ---
    timer_start(&timer);
    latency_bound<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK_KERNEL();
    float ms3 = timer_stop(&timer);

    long long flops3 = (long long)N * 25;  // average ~25 ops per thread
    report_gflops("latency_bound", flops3, ms3);
    report_roofline("latency_bound", flops3, 2 * bytes, ms3);

    // --- What to look for ---
    printf("\n");
    printf("┌────────────────────────────────────────────────────────────────┐\n");
    printf("│  NSIGHT COMPUTE: WHAT TO LOOK FOR                              │\n");
    printf("├────────────────────────────────────────────────────────────────┤\n");
    printf("│                                                                │\n");
    printf("│  Run: ncu --set full ./profiling_guide                         │\n");
    printf("│                                                                │\n");
    printf("│  For each kernel, check the \"GPU Speed of Light\" section:      │\n");
    printf("│                                                                │\n");
    printf("│  compute_bound:                                                │\n");
    printf("│    - SM Throughput: HIGH (>60%%)                                │\n");
    printf("│    - Memory Throughput: LOW (<30%%)                             │\n");
    printf("│    → GPU is busy doing math. To speed up: reduce FLOPs.        │\n");
    printf("│                                                                │\n");
    printf("│  memory_bound:                                                 │\n");
    printf("│    - SM Throughput: LOW (<30%%)                                 │\n");
    printf("│    - Memory Throughput: HIGH (>60%%)                            │\n");
    printf("│    → GPU is waiting for data. To speed up: tiling, caching.    │\n");
    printf("│                                                                │\n");
    printf("│  latency_bound:                                                │\n");
    printf("│    - BOTH throughputs LOW (<30%%)                               │\n");
    printf("│    - Warp State: \"Stall\" dominates                             │\n");
    printf("│    → GPU is idle. Fix: remove divergence, increase occupancy.  │\n");
    printf("│                                                                │\n");
    printf("├────────────────────────────────────────────────────────────────┤\n");
    printf("│  NSIGHT SYSTEMS: WHAT TO LOOK FOR                              │\n");
    printf("├────────────────────────────────────────────────────────────────┤\n");
    printf("│                                                                │\n");
    printf("│  Run: nsys profile --stats=true ./profiling_guide              │\n");
    printf("│                                                                │\n");
    printf("│  The timeline shows:                                           │\n");
    printf("│    - Blue bars: GPU kernel execution                           │\n");
    printf("│    - Green bars: memory copies (Host<->Device)                 │\n");
    printf("│    - Gaps: CPU/GPU idle time (pipeline bubbles)                │\n");
    printf("│                                                                │\n");
    printf("│  Key metrics in --stats output:                                │\n");
    printf("│    - \"CUDA Kernel Statistics\": time per kernel                 │\n");
    printf("│    - \"CUDA Memory Operation Statistics\": transfer times        │\n");
    printf("│    - \"OS Runtime\" calls: driver overhead                       │\n");
    printf("│                                                                │\n");
    printf("│  Open the .nsys-rep file in Nsight Systems GUI for the         │\n");
    printf("│  visual timeline (drag and zoom to explore).                   │\n");
    printf("│                                                                │\n");
    printf("└────────────────────────────────────────────────────────────────┘\n");

    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);

    return 0;
}

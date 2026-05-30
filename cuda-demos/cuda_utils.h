/*
 * cuda_utils.h — Helpers that make GPU behavior visible
 *
 * Include this in any demo to get error checking, timing, and bandwidth reporting.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking — wraps every CUDA call to catch errors immediately
// ---------------------------------------------------------------------------
// Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
// Without this, CUDA errors are silent — the program continues with garbage data.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                        \
        }                                                                  \
    } while (0)

// Check for errors after kernel launches (which don't return an error code)
#define CUDA_CHECK_KERNEL()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "Kernel launch error at %s:%d — %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                        \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Timer — wraps CUDA events for clean kernel timing
// ---------------------------------------------------------------------------
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

static inline void timer_create(GpuTimer *t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

static inline void timer_start(GpuTimer *t) {
    cudaEventRecord(t->start);
}

static inline float timer_stop(GpuTimer *t) {
    cudaEventRecord(t->stop);
    cudaEventSynchronize(t->stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

static inline void timer_destroy(GpuTimer *t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

// ---------------------------------------------------------------------------
// Bandwidth & throughput reporting
// ---------------------------------------------------------------------------
// Effective bandwidth: how fast data actually moves (vs theoretical peak)
static inline void report_bandwidth(const char *label, size_t bytes_moved, float ms) {
    float gb = (float)bytes_moved / 1e9f;
    float seconds = ms / 1000.0f;
    float gbps = gb / seconds;
    printf("  %-20s  %7.3f ms | %6.1f GB/s | %zu bytes moved\n",
           label, ms, gbps, bytes_moved);
}

// Arithmetic throughput for matmul (2*M*N*K FLOPs)
static inline void report_gflops(const char *label, long long flops, float ms) {
    float gflops = (float)flops / (ms * 1e6f);
    printf("  %-20s  %7.3f ms | %6.1f GFLOPS\n", label, ms, gflops);
}

// ---------------------------------------------------------------------------
// Launch configuration helper — prints what's happening
// ---------------------------------------------------------------------------
static inline void print_launch_config(const char *kernel_name,
                                       dim3 grid, dim3 block) {
    int total_threads = grid.x * grid.y * grid.z * block.x * block.y * block.z;
    int threads_per_block = block.x * block.y * block.z;

    printf("\n┌── Launching: %s\n", kernel_name);
    printf("│   Grid:    (%d, %d, %d) = %d blocks\n",
           grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);
    printf("│   Block:   (%d, %d, %d) = %d threads/block\n",
           block.x, block.y, block.z, threads_per_block);
    printf("│   Total:   %d threads\n", total_threads);
    printf("│   Warps:   %d per block (%d threads execute together)\n",
           threads_per_block / 32, 32);
    printf("└──\n\n");
}

// ---------------------------------------------------------------------------
// Roofline model helper — shows whether kernel is memory or compute bound
// ---------------------------------------------------------------------------
// Arithmetic intensity = FLOPs / bytes moved
// If intensity < machine's ridge point, you're memory-bound.
static inline void report_roofline(const char *label,
                                   long long flops, size_t bytes_moved, float ms) {
    float ai = (float)flops / (float)bytes_moved;  // FLOPs per byte
    float gflops = (float)flops / (ms * 1e6f);
    float gbps = (float)bytes_moved / (ms * 1e6f);

    printf("\n  ── Roofline Analysis: %s ──\n", label);
    printf("  Arithmetic Intensity: %.2f FLOPs/byte\n", ai);
    printf("  Achieved Compute:     %.1f GFLOPS\n", gflops);
    printf("  Achieved Bandwidth:   %.1f GB/s\n", gbps);

    if (ai < 10.0f) {
        printf("  Diagnosis: MEMORY-BOUND (AI < ~10)\n");
        printf("  → Optimization: reduce memory accesses (tiling, caching)\n");
    } else {
        printf("  Diagnosis: COMPUTE-BOUND (AI > ~10)\n");
        printf("  → Optimization: reduce arithmetic (approximation, lower precision)\n");
    }
}

#endif // CUDA_UTILS_H

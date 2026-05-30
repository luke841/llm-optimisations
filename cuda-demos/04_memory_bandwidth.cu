/*
 * 04_memory_bandwidth.cu — Measure actual memory bandwidth with different access patterns
 *
 * Shows WHY coalesced access matters and HOW the memory hierarchy performs in practice.
 * This is the root cause of most GPU performance problems in LLM inference.
 *
 * Compile: nvcc -o memory_bandwidth 04_memory_bandwidth.cu
 * Run:     ./memory_bandwidth
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

// ---------------------------------------------------------------------------
// Kernel 1: Coalesced access — threads read consecutive addresses
// ---------------------------------------------------------------------------
// Thread 0 reads [0], thread 1 reads [1], thread 2 reads [2], ...
// Hardware combines these into wide 128-byte transactions. Optimal.
__global__ void copy_coalesced(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Strided access — threads read with a stride between addresses
// ---------------------------------------------------------------------------
// Thread 0 reads [0], thread 1 reads [STRIDE], thread 2 reads [2*STRIDE], ...
// Hardware can't coalesce — each thread triggers a separate memory transaction.
__global__ void copy_strided(const float *in, float *out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * stride;
    if (strided_idx < n) {
        out[idx] = in[strided_idx];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Random access — threads read from random positions
// ---------------------------------------------------------------------------
// Worst case: completely unpredictable, no coalescing possible.
__global__ void copy_random(const float *in, float *out, const int *indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[indices[idx]];
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Shared memory demo — load coalesced, then access freely
// ---------------------------------------------------------------------------
// Pattern used everywhere in practice: load a tile from global (coalesced),
// then do arbitrary access patterns within shared memory (fast).
__global__ void shared_mem_demo(const float *in, float *out, int n) {
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: coalesced load from global -> shared (one wide transaction)
    if (idx < n) {
        tile[threadIdx.x] = in[idx];
    }
    __syncthreads();

    // Step 2: reversed read from shared memory (would be terrible in global, free in shared)
    int reversed = blockDim.x - 1 - threadIdx.x;
    if (idx < n) {
        out[idx] = tile[reversed];
    }
}

// ---------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------
int main() {
    const int N = 1 << 24;  // 16M elements = 64 MB
    const size_t bytes = N * sizeof(float);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    printf("Memory Bandwidth Benchmark\n");
    printf("Data size: %d elements = %zu MB\n\n", N, bytes / (1024 * 1024));

    // Allocate
    float *h_in = (float *)malloc(bytes);
    int *h_indices = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
        h_indices[i] = rand() % N;
    }

    float *d_in, *d_out;
    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer_create(&timer);

    // Warmup
    copy_coalesced<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // --- Test 1: Coalesced ---
    printf("┌────────────────────────────────────────────────────────────┐\n");
    printf("│  ACCESS PATTERN COMPARISON                                 │\n");
    printf("├────────────────────────────────────────────────────────────┤\n");

    timer_start(&timer);
    copy_coalesced<<<blocks, threads>>>(d_in, d_out, N);
    float coalesced_ms = timer_stop(&timer);
    report_bandwidth("Coalesced (stride=1)", 2 * bytes, coalesced_ms);

    // --- Test 2: Strided (various strides) ---
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        int stride = strides[s];
        int strided_blocks = (N / stride + threads - 1) / threads;

        timer_start(&timer);
        copy_strided<<<strided_blocks, threads>>>(d_in, d_out, N, stride);
        float ms = timer_stop(&timer);

        char label[64];
        snprintf(label, sizeof(label), "Strided (stride=%d)", stride);
        size_t effective_bytes = 2 * (N / stride) * sizeof(float);
        report_bandwidth(label, effective_bytes, ms);
    }

    // --- Test 3: Random ---
    timer_start(&timer);
    copy_random<<<blocks, threads>>>(d_in, d_out, d_indices, N);
    float random_ms = timer_stop(&timer);
    report_bandwidth("Random", 2 * bytes, random_ms);

    // --- Test 4: Shared memory ---
    timer_start(&timer);
    shared_mem_demo<<<blocks, threads>>>(d_in, d_out, N);
    float shared_ms = timer_stop(&timer);
    report_bandwidth("Shared mem (reversed)", 2 * bytes, shared_ms);

    printf("└────────────────────────────────────────────────────────────┘\n");

    // --- Summary ---
    printf("\n");
    printf("┌────────────────────────────────────────────────────────────┐\n");
    printf("│  KEY TAKEAWAYS                                             │\n");
    printf("├────────────────────────────────────────────────────────────┤\n");
    printf("│                                                            │\n");
    printf("│  1. Coalesced access is FAST — hardware merges reads       │\n");
    printf("│     into wide transactions (128 bytes at once).            │\n");
    printf("│                                                            │\n");
    printf("│  2. Strided/random access is SLOW — each thread causes     │\n");
    printf("│     a separate memory transaction. Bandwidth drops         │\n");
    printf("│     proportional to the stride.                            │\n");
    printf("│                                                            │\n");
    printf("│  3. Shared memory rescues non-coalesced patterns —         │\n");
    printf("│     load coalesced into shared, then access freely.        │\n");
    printf("│                                                            │\n");
    printf("│  This is why EVERY high-perf kernel (FlashAttention,       │\n");
    printf("│  cuBLAS) uses the pattern: coalesced load → shared mem     │\n");
    printf("│  → arbitrary compute → coalesced store.                    │\n");
    printf("│                                                            │\n");
    printf("└────────────────────────────────────────────────────────────┘\n");

    printf("\n  Slowdown vs coalesced:\n");
    printf("    Random access:  %.1fx slower\n", random_ms / coalesced_ms);
    printf("    Shared memory:  %.2fx (nearly as fast as coalesced!)\n",
           shared_ms / coalesced_ms);

    // Cleanup
    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_indices));
    free(h_in);
    free(h_indices);

    return 0;
}

/*
 * 01_vector_add.cu — Simplest possible CUDA program
 *
 * Adds two vectors element-wise on the GPU.
 * Demonstrates: kernel launch, thread indexing, device memory, host<->device transfer.
 *
 * Compile: nvcc -o vector_add 01_vector_add.cu
 * Run:     ./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// GPU Kernel — this function runs on EVERY thread in parallel
// ---------------------------------------------------------------------------
// Each thread computes ONE element of the output vector.
// The GPU launches thousands of these simultaneously.
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // blockIdx.x  = which block this thread belongs to (0, 1, 2, ...)
    // blockDim.x  = how many threads per block (e.g. 256)
    // threadIdx.x = this thread's index within its block (0..255)
    //
    // Together they give a unique global index:
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: we may launch more threads than elements (due to rounding up)
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------------------------------------------------------------------
// Host (CPU) code
// ---------------------------------------------------------------------------
int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    // 1. Allocate host (CPU) memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // 2. Initialize input data on the CPU
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // 3. Allocate device (GPU) memory
    //    The GPU has its OWN memory — data must be explicitly copied there.
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 4. Copy input data from host -> device
    //    This crosses the PCIe bus (or NVLink). It's slow relative to compute.
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 5. Launch the kernel
    //    We choose 256 threads per block (a common default).
    //    We need enough blocks to cover all N elements.
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;  // ceiling division

    printf("Launching %d blocks x %d threads = %d total threads (for %d elements)\n",
           blocks, threads_per_block, blocks * threads_per_block, N);

    // <<<blocks, threads_per_block>>> is CUDA's kernel launch syntax.
    // This is asynchronous — the CPU doesn't wait for the GPU to finish.
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    // 6. Copy result back from device -> host
    //    cudaMemcpy implicitly synchronizes (waits for the kernel to finish).
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 7. Verify correctness
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_c[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("MISMATCH at %d: got %f, expected %f\n", i, h_c[i], expected);
            }
        }
    }

    if (errors == 0) {
        printf("SUCCESS: All %d elements match.\n", N);
    } else {
        printf("FAILED: %d mismatches.\n", errors);
    }

    // 8. Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

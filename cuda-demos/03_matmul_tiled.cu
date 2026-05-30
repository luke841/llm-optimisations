/*
 * 03_matmul_tiled.cu — Tiled matrix multiplication using shared memory
 *
 * Key insight: threads in the same block read overlapping data from global memory.
 * By loading a TILE into fast shared memory (on-chip SRAM, ~100x faster than global),
 * we reduce global memory reads from O(K) per thread to O(K/TILE_SIZE) per thread.
 *
 * This is the single most important GPU optimization pattern — it appears everywhere
 * in LLM inference (attention, linear layers, etc).
 *
 * Compile: nvcc -o matmul_tiled 03_matmul_tiled.cu
 * Run:     ./matmul_tiled
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ---------------------------------------------------------------------------
// GPU Kernel — tiled matmul with shared memory
// ---------------------------------------------------------------------------
__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    // Shared memory tiles — one for A, one for B.
    // These are shared by ALL threads in this block (fast on-chip SRAM).
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Slide the tile window across the K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Step 1: Collaboratively load one tile of A and one tile of B
        //         Each thread loads exactly ONE element of each tile.
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // Bounds check (matrices may not be tile-aligned)
        tile_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        // Step 2: Wait for ALL threads in the block to finish loading.
        //         Without this barrier, some threads would read stale/unloaded values.
        __syncthreads();

        // Step 3: Compute partial dot product using shared memory (fast!)
        //         16 multiply-adds, all from shared memory — no global memory access.
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // Step 4: Wait before loading the next tile (don't overwrite while others read)
        __syncthreads();
    }

    // Write final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Naive kernel (for comparison)
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------
void fill_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 10) / 10.0f;
    }
}

int main() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C_naive = (float *)malloc(bytes_C);
    float *h_C_tiled = (float *)malloc(bytes_C);

    srand(42);
    fill_matrix(h_A, M, K);
    fill_matrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix sizes: (%d x %d) * (%d x %d) = (%d x %d)\n", M, K, K, N, M, N);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Grid: (%d, %d) blocks\n\n", grid_dim.x, grid_dim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Benchmark naive ---
    cudaEventRecord(start);
    matmul_naive<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0;
    cudaEventElapsedTime(&naive_ms, start, stop);
    cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // --- Benchmark tiled ---
    cudaEventRecord(start);
    matmul_tiled<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_ms = 0;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    cudaMemcpy(h_C_tiled, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Verify tiled matches naive
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = h_C_tiled[i] - h_C_naive[i];
        if (diff > 1e-3f || diff < -1e-3f) {
            errors++;
        }
    }

    // Results
    float gflops_naive = (2.0f * M * N * K) / (naive_ms * 1e6f);
    float gflops_tiled = (2.0f * M * N * K) / (tiled_ms * 1e6f);

    printf("Naive:  %7.3f ms  (%6.1f GFLOPS)\n", naive_ms, gflops_naive);
    printf("Tiled:  %7.3f ms  (%6.1f GFLOPS)\n", tiled_ms, gflops_tiled);
    printf("Speedup: %.2fx\n", naive_ms / tiled_ms);
    printf("Correctness: %s\n\n", errors == 0 ? "PASS" : "FAIL");

    // Explanation of WHY tiling helps
    printf("=== Why tiling works ===\n");
    printf("Naive:  Each thread reads %d floats from GLOBAL memory (slow, ~400 cycles)\n", 2 * K);
    printf("Tiled:  Each thread reads %d floats from global, %d from SHARED memory (fast, ~5 cycles)\n",
           2 * K / TILE_SIZE * TILE_SIZE == 2 * K ? 2 * (K / TILE_SIZE) : 2 * ((K + TILE_SIZE - 1) / TILE_SIZE),
           2 * K);
    printf("\nGlobal memory reads per thread:\n");
    printf("  Naive:  %d (one per K iteration, twice)\n", 2 * K);
    printf("  Tiled:  %d (one per tile, twice)\n", 2 * (K / TILE_SIZE));
    printf("  Reduction: %dx fewer global memory accesses!\n", TILE_SIZE);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);

    return 0;
}

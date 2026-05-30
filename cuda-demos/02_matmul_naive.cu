/*
 * 02_matmul_naive.cu — Naive matrix multiplication on the GPU
 *
 * Computes C = A * B where A is (M x K), B is (K x N), C is (M x N).
 * Each thread computes ONE element of C by doing a dot product.
 *
 * This is intentionally naive — it hammers global memory. The tiled version
 * (03_matmul_tiled.cu) shows how shared memory fixes this.
 *
 * Compile: nvcc -o matmul_naive 02_matmul_naive.cu
 * Run:     ./matmul_naive
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// GPU Kernel — naive matmul
// ---------------------------------------------------------------------------
// Each thread computes C[row][col] = dot(A[row,:], B[:,col])
// Problem: each thread reads K floats from A and K floats from B from GLOBAL memory.
// For an MxN output, that's M*N*2K global memory reads total — extremely wasteful
// because neighboring threads re-read the same rows/columns.
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

void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 1024;  // rows of A and C
    const int K = 512;   // cols of A, rows of B
    const int N = 1024;  // cols of B and C

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate and initialize host matrices
    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);
    float *h_C_ref = (float *)malloc(bytes_C);

    srand(42);
    fill_matrix(h_A, M, K);
    fill_matrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // Launch kernel
    // 2D thread blocks: 16x16 = 256 threads per block
    // Each thread computes one element of the output matrix.
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (M + block_dim.y - 1) / block_dim.y);

    printf("Matrix sizes: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);
    printf("Grid: (%d, %d) blocks of (%d, %d) threads\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    // Time the GPU kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_naive<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // CPU reference for correctness check
    cpu_matmul(h_A, h_B, h_C_ref, M, K, N);

    // Verify
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = h_C[i] - h_C_ref[i];
        if (diff > 1e-3f || diff < -1e-3f) {
            errors++;
            if (errors <= 3) {
                printf("MISMATCH at %d: GPU=%f, CPU=%f\n", i, h_C[i], h_C_ref[i]);
            }
        }
    }

    if (errors == 0) {
        printf("SUCCESS: GPU matches CPU reference.\n");
    } else {
        printf("FAILED: %d mismatches.\n", errors);
    }

    // Performance
    float gflops = (2.0f * M * N * K) / (gpu_ms * 1e6f);
    printf("GPU time: %.3f ms (%.1f GFLOPS)\n", gpu_ms, gflops);
    printf("\nNote: This naive version is memory-bound.\n");
    printf("Each output element loads %d floats from global memory (slow!).\n", 2 * K);
    printf("See 03_matmul_tiled.cu for the shared-memory optimization.\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}

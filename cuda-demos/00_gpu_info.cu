/*
 * 00_gpu_info.cu — Query and display GPU hardware properties
 *
 * Run this first to understand what you're working with.
 * Maps hardware specs to the programming model concepts in the other demos.
 *
 * Compile: nvcc -o gpu_info 00_gpu_info.cu
 * Run:     ./gpu_info
 */

#include <stdio.h>
#include <cuda_runtime.h>

void print_separator(const char *title) {
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  %s\n", title);
    printf("══════════════════════════════════════════════════════════════\n\n");
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("No CUDA-capable GPU found.\n");
        return 1;
    }

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        print_separator("DEVICE IDENTITY");
        printf("  Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Driver/Runtime:  CUDA %d.%d\n",
               prop.major, prop.minor);

        print_separator("MEMORY HIERARCHY");
        printf("  Global Memory:      %zu MB (HBM/GDDR — large, slow ~400 cycles)\n",
               prop.totalGlobalMem / (1024 * 1024));
        printf("  L2 Cache:           %d KB\n",
               prop.l2CacheSize / 1024);
        printf("  Shared Mem / Block: %zu KB (on-chip SRAM — small, fast ~5 cycles)\n",
               prop.sharedMemPerBlock / 1024);
        printf("  Shared Mem / SM:    %zu KB\n",
               prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers / Block:  %d (32-bit)\n",
               prop.regsPerBlock);
        printf("  Memory Bus Width:   %d bits\n", prop.memoryBusWidth);
        printf("  Memory Bandwidth:   %.0f GB/s (theoretical peak)\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

        print_separator("EXECUTION MODEL");
        printf("  Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
        printf("  Max Threads / SM:       %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads / Block:    %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size:              %d threads (execute in lockstep)\n", prop.warpSize);
        printf("  Max Blocks / SM:        %d\n", prop.maxBlocksPerMultiProcessor);
        printf("\n  Max Grid Dimensions:    (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Dimensions:   (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

        print_separator("WHAT THIS MEANS FOR YOU");
        int total_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        printf("  Max concurrent threads: %d SMs x %d threads/SM = %d threads\n",
               prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, total_threads);
        printf("  That's %dx more parallelism than a 16-core CPU!\n\n", total_threads / 16);

        printf("  Block size trade-offs:\n");
        printf("    - Too small (32):  underutilizes SM resources, low occupancy\n");
        printf("    - Sweet spot (128-256): good occupancy, enough warps to hide latency\n");
        printf("    - Too large (1024): may run out of registers/shared mem per block\n\n");

        printf("  Shared memory budget per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("    -> For tiled matmul with TILE=16: 2 tiles x 16x16 x 4 bytes = 2 KB (plenty)\n");
        printf("    -> For tiled matmul with TILE=32: 2 tiles x 32x32 x 4 bytes = 8 KB (still fine)\n");

        print_separator("MEMORY ACCESS PATTERNS");
        printf("  Memory Clock:       %d MHz\n", prop.memoryClockRate / 1000);
        printf("  Bus Width:          %d bits = %d bytes per transaction\n",
               prop.memoryBusWidth, prop.memoryBusWidth / 8);
        printf("\n  Coalescing Rule:\n");
        printf("    When threads in a warp access CONSECUTIVE addresses,\n");
        printf("    the hardware combines them into one wide transaction.\n");
        printf("    -> 32 threads x 4 bytes = 128 bytes = one transaction (good!)\n");
        printf("    -> 32 threads accessing random addresses = 32 transactions (32x slower!)\n");

        print_separator("OCCUPANCY QUICK REFERENCE");
        printf("  Registers per SM:   %d\n", prop.regsPerMultiprocessor);
        printf("  Shared mem per SM:  %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("\n  If your kernel uses 32 registers/thread and 256 threads/block:\n");
        printf("    Register usage:  256 * 32 = 8192 registers per block\n");
        printf("    Blocks per SM:   %d / 8192 = %d (register-limited)\n",
               prop.regsPerMultiprocessor, prop.regsPerMultiprocessor / 8192);
        printf("    Active threads:  %d * 256 = %d / %d max = %.0f%% occupancy\n",
               prop.regsPerMultiprocessor / 8192,
               (prop.regsPerMultiprocessor / 8192) * 256,
               prop.maxThreadsPerMultiProcessor,
               100.0f * (prop.regsPerMultiprocessor / 8192) * 256 / prop.maxThreadsPerMultiProcessor);
    }

    printf("\n");
    return 0;
}

/*
 * 01_device_query.cu
 *
 * Concept: Whole GPU chip - SM, SP, warp, shared memory, DRAM hierarchy
 * Diagram blocks: Host Interface, TPC, SM, SP, SFU, Shared Memory, ROP, L2, VRAM
 *
 * What to observe:
 *   - Number of SMs (Streaming Multiprocessors) in the GPU
 *   - Warp size (always 32 threads for NVIDIA)
 *   - Max threads per block / per SM
 *   - Shared memory per block (on-chip, ultra-low latency)
 *   - L2 cache size and VRAM capacity
 *   - Estimated CUDA core count (SMs * cores_per_SM)
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("========== CUDA Device Query ==========\n");
    printf("Number of GPUs: %d\n\n", device_count);

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("GPU %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("\n[HOST INTERFACE & WORK DISTRIBUTORS]\n");
        printf("  PCIe Bus ID: %d\n", prop.pciBusID);

        printf("\n[STREAMING MULTIPROCESSORS (SMs)]\n");
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);

        printf("\n[STREAMING PROCESSOR (SP) / CUDA Cores]\n");
        // For Turing (7.5): 64 CUDA cores per SM
        // For Ampere (8.x): 128 CUDA cores per SM
        // For Volta (7.0): 64 CUDA cores per SM
        int cores_per_sm = 64;  // Default for Turing (GTX 1650 Ti)
        if (prop.major == 8) cores_per_sm = 128;  // Ampere
        else if (prop.major == 9) cores_per_sm = 128;  // Ada

        int total_cores = prop.multiProcessorCount * cores_per_sm;
        printf("  Cores per SM: %d\n", cores_per_sm);
        printf("  Total CUDA Cores: %d\n", total_cores);

        printf("\n[WARP & THREAD SCHEDULING]\n");
        printf("  Warp Size: %d threads\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

        printf("\n[SHARED MEMORY]\n");
        printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);

        printf("\n[CONSTANT MEMORY (C-Cache)]\n");
        printf("  Constant Memory: %zu bytes\n", prop.totalConstMem);

        printf("\n[L2 CACHE]\n");
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);

        printf("\n[MEMORY HIERARCHY - DRAM]\n");
        printf("  Total Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Memory Clock Rate: %.1f MHz\n", prop.memoryClockRate / 1000.0);

        printf("\n========== GPU HIERARCHY TREE ==========\n");
        printf("GPU: %s\n", prop.name);
        printf("├─ %d Texture Processing Clusters (TPC)\n", (prop.multiProcessorCount + 1) / 2);
        printf("│  └─ SMs (Streaming Multiprocessors): %d\n", prop.multiProcessorCount);
        for (int i = 0; i < (prop.multiProcessorCount < 4 ? prop.multiProcessorCount : 4); i++) {
            printf("│     ├─ SM[%d]: %d CUDA Cores (SPs)\n", i, cores_per_sm);
            printf("│     │  ├─ %d Warps (32 threads each)\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
            printf("│     │  └─ Shared Memory: %zu bytes\n", prop.sharedMemPerBlock);
        }
        if (prop.multiProcessorCount > 4) {
            printf("│     └─ SM[%d..%d]: ... (%d more)\n", 4, prop.multiProcessorCount - 1,
                   prop.multiProcessorCount - 4);
        }
        printf("└─ Global Memory: %.1f GB (VRAM via L2 cache and ROP units)\n",
               prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    }

    printf("\n");
    return 0;
}

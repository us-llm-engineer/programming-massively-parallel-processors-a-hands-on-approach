/*
 * 02_hello_threads.cu
 *
 * Concept: Work distributor dispatching blocks and threads across SMs
 * Diagram blocks: Host Interface (work distributor) → TPC → SM scheduling
 *
 * What to observe:
 *   - blockIdx shows which block in the grid
 *   - threadIdx shows which thread in the block
 *   - warpId = threadIdx.x / 32 (32 threads per warp)
 *   - lane = threadIdx.x % 32 (position within warp)
 *   - smId: which SM is executing this warp (read via inline assembly)
 *   - Multiple blocks running on different SMs in parallel
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void hello_threads_kernel() {
    // Get thread and block indices
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    // Warp ID: every 32 threads form a warp
    int warp_id = thread_id / 32;

    // Lane ID: position within the warp (0-31)
    int lane_id = thread_id % 32;

    // SM ID: read via inline assembly (for compute capability 7.0+)
    unsigned int sm_id = 0;
    asm("mov.u32 %0, %%smid;" : "=r"(sm_id));

    // Print only once per warp to reduce output
    if (lane_id == 0) {
        printf("Block[%d] Warp[%d] (threads %d-%d) executing on SM[%u]\n",
               block_id, warp_id,
               thread_id, thread_id + 31,
               sm_id);
    }

    __syncthreads();  // Wait for all threads in block

    // Thread 0 of each block prints a summary
    if (thread_id == 0) {
        printf("  └─ Block[%d] complete: %d threads across %d warps\n",
               block_id, blockDim.x, (blockDim.x + 31) / 32);
    }
}

int main() {
    printf("========== Thread Hierarchy & SM Distribution ==========\n\n");

    // Query GPU properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d threads\n\n", prop.warpSize);

    // Launch configuration:
    // - 4 blocks (will be distributed across SMs)
    // - 64 threads per block = 2 warps per block
    int num_blocks = 4;
    int threads_per_block = 64;

    printf("Launch configuration: <<<blocks=%d, threads=%d>>>\n", num_blocks, threads_per_block);
    printf("Expected: %d warps per block, %d threads per warp\n\n",
           threads_per_block / 32, 32);

    printf("========== Kernel Execution ==========\n\n");

    hello_threads_kernel<<<num_blocks, threads_per_block>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n========== Interpretation ==========\n");
    printf("Each line shows one warp (32 threads) and which SM it ran on.\n");
    printf("Notice:\n");
    printf("  1. Multiple blocks may run on the same SM (SM[0], SM[1], etc.)\n");
    printf("  2. Each warp is the scheduling unit (always 32 threads)\n");
    printf("  3. Threads within a warp can execute the same instruction (SIMT)\n");
    printf("  4. The work distributor (from diagram) assigns blocks to SMs\n\n");

    return 0;
}

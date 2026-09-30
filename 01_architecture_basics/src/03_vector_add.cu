/*
 * 03_vector_add.cu
 *
 * Concept: SPs (CUDA cores) executing the same instruction on many threads (SIMT)
 * Diagram blocks: SP execution units, host interface (PCIe), L2 cache, DRAM
 *
 * What to observe:
 *   - 1M elements: c[i] = a[i] + b[i]
 *   - All SPs execute c[i]=a[i]+b[i] on different data (SIMT: Single Instruction, Multiple Threads)
 *   - Timing breakdown:
 *       * H2D (Host to Device): PCIe data transfer
 *       * Kernel: execution on SPs
 *       * D2H (Device to Host): PCIe data transfer back
 *   - Verification against CPU to confirm correctness
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define N 1000000  // 1 million elements

__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
    // Global thread index: which element this thread is responsible for
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: only process valid elements
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("========== Vector Add: SIMT Execution Example ==========\n\n");

    // Host memory
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c_gpu = (float *)malloc(N * sizeof(float));
    float *h_c_cpu = (float *)malloc(N * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i / N;
        h_b[i] = 1.0f - h_a[i];
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    printf("Input: %d elements\n", N);
    printf("Memory per array: %.1f MB\n\n", (N * sizeof(float)) / (1024.0 * 1024.0));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ========== H2D: Host to Device Transfer ==========
    printf("========== H2D Transfer (Host → Device via PCIe) ==========\n");
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_time;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time, start, stop));
    printf("H2D Time: %.3f ms\n", h2d_time);
    printf("H2D Bandwidth: %.1f GB/s\n\n",
           (2.0 * N * sizeof(float) / (1024*1024*1024)) / (h2d_time / 1000.0));

    // ========== Kernel Execution ==========
    printf("========== Kernel: Vector Addition on SPs ==========\n");

    // Grid/block configuration
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    printf("Launch: <<<blocks=%d, threads=%d>>>\n", blocks, threads_per_block);
    printf("Total threads: %d (covers %d elements)\n\n",
           blocks * threads_per_block, N);

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));
    printf("Kernel Time: %.3f ms\n\n", kernel_time);

    // ========== D2H: Device to Host Transfer ==========
    printf("========== D2H Transfer (Device → Host via PCIe) ==========\n");
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_time;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time, start, stop));
    printf("D2H Time: %.3f ms\n", d2h_time);
    printf("D2H Bandwidth: %.1f GB/s\n\n",
           (N * sizeof(float) / (1024*1024*1024)) / (d2h_time / 1000.0));

    // ========== Verification ==========
    printf("========== Verification (CPU Baseline) ==========\n");
    vector_add_cpu(h_a, h_b, h_c_cpu, N);

    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabs(h_c_gpu[i] - h_c_cpu[i]);
        if (error > 1e-6) {
            correct = false;
            if (error > max_error) max_error = error;
        }
    }

    if (correct) {
        printf("✓ PASS: Results match CPU baseline\n");
        printf("  Max error: %.2e (within tolerance)\n\n", max_error);
    } else {
        printf("✗ FAIL: Results do not match\n");
        printf("  Max error: %.2e\n\n", max_error);
    }

    // ========== Summary ==========
    printf("========== Timing Summary ==========\n");
    printf("H2D (transfer in):  %8.3f ms\n", h2d_time);
    printf("Kernel (compute):   %8.3f ms  ← SPs executing c[i]=a[i]+b[i] in parallel\n", kernel_time);
    printf("D2H (transfer out): %8.3f ms\n", d2h_time);
    printf("Total:              %8.3f ms\n\n", h2d_time + kernel_time + d2h_time);

    printf("Kernel throughput:  %.1f GFLOP/s (1M adds at %.1f GHz effective)\n",
           (N / (kernel_time / 1000.0)) / 1e9,
           (N / (kernel_time / 1000.0)) / 1e9 / 1024);  // Rough estimate

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n");
    return 0;
}

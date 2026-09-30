/*
 * 02_sum_reduction.cu   (PMPP Chapter 6: reduction)
 *
 * Kernels, straight from the book:
 *   v1  Figure 6.2  interleaved threads     if (t % (2*stride) == 0)   stride 1,2,4,...
 *   v2  Figure 6.4  contiguous threads      if (t < stride)            stride blockDim/2,...,1
 *   v3  Fig 6.4 + warp unrolling of the last 5 steps (book: removes the remaining
 *       divergence when stride < 32)
 *
 * Multi-block reduction (book: hierarchical, multi-pass): each pass reduces
 * every 512-element section to one partial sum written by thread 0; the host
 * re-launches the same kernel on the partial sums until one value is left.
 *
 * Measurement rules (the first version of this file broke them):
 *   - one untimed warm-up run of every variant before any timing
 *   - each variant timed as the average of REPS complete multi-pass reductions
 *   - variants timed in round-robin order so none pays first-launch cost
 *   - N is large (16M floats) so the kernels, not launch overhead, dominate
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BLOCK_SIZE 512          // book's section size: 512 elements per block
#define N (1u << 24)            // 16,777,216 floats = 64 MB
#define REPS 50

// ---- v1: Figure 6.2 -------------------------------------------------------
__global__ void reduce_v1_interleaved(const float *in, float *out, unsigned n) {
    __shared__ float partialSum[BLOCK_SIZE];
    unsigned t = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + t;
    partialSum[t] = (i < n) ? in[i] : 0.0f;

    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0)
            partialSum[t] += partialSum[t + stride];
    }
    if (t == 0) out[blockIdx.x] = partialSum[0];
}

// ---- v2: Figure 6.4 -------------------------------------------------------
__global__ void reduce_v2_contiguous(const float *in, float *out, unsigned n) {
    __shared__ float partialSum[BLOCK_SIZE];
    unsigned t = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + t;
    partialSum[t] = (i < n) ? in[i] : 0.0f;

    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    if (t == 0) out[blockIdx.x] = partialSum[0];
}

// ---- v3: Figure 6.4 + unrolled last warp ----------------------------------
// volatile + __syncwarp: on Volta/Turing and newer, threads of a warp are not
// guaranteed lock-step, so the book's plain unrolled code needs explicit warp sync.
__global__ void reduce_v3_unrolled(const float *in, float *out, unsigned n) {
    __shared__ float partialSum[BLOCK_SIZE];
    unsigned t = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + t;
    partialSum[t] = (i < n) ? in[i] : 0.0f;

    for (unsigned stride = blockDim.x >> 1; stride >= 32; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();
    if (t < 32) {
        volatile float *v = partialSum;
        v[t] += v[t + 16]; __syncwarp();
        v[t] += v[t + 8];  __syncwarp();
        v[t] += v[t + 4];  __syncwarp();
        v[t] += v[t + 2];  __syncwarp();
        v[t] += v[t + 1];
    }
    if (t == 0) out[blockIdx.x] = partialSum[0];
}

typedef void (*reduce_kernel)(const float *, float *, unsigned);

// Multi-pass driver: reduce 'n' values in d_a down to one, ping-ponging d_a/d_b.
// Returns the pointer that holds the single result at index 0.
static float *reduce_all(reduce_kernel k, float *d_a, float *d_b, unsigned n) {
    float *src = d_a, *dst = d_b;
    while (n > 1) {
        unsigned blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k<<<blocks, BLOCK_SIZE>>>(src, dst, n);
        n = blocks;
        float *tmp = src; src = dst; dst = tmp;
    }
    return src;
}

int main() {
    const char *names[3] = {
        "v1 Fig 6.2  interleaved  (t % (2*stride) == 0)",
        "v2 Fig 6.4  contiguous   (t < stride)",
        "v3 Fig 6.4 + last-warp unroll" };
    reduce_kernel kernels[3] = { reduce_v1_interleaved, reduce_v2_contiguous, reduce_v3_unrolled };

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("========== PARALLEL SUM REDUCTION (PMPP Chapter 6) ==========\n");
    printf("GPU: %s | N = %u floats (%.0f MB) | block = %d threads | reps = %d\n\n",
           prop.name, N, N * 4.0 / (1 << 20), BLOCK_SIZE, REPS);

    float *h_in = (float *)malloc(N * sizeof(float));
    double cpu = 0.0;
    srand(1);
    for (unsigned i = 0; i < N; i++) { h_in[i] = (float)rand() / RAND_MAX; cpu += h_in[i]; }

    float *d_in, *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,  N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- correctness: one full run per variant, compared with a double-precision CPU sum
    printf("---- Correctness (CPU double-precision sum = %.4f) ----\n", cpu);
    bool all_ok = true;
    for (int v = 0; v < 3; v++) {
        CUDA_CHECK(cudaMemcpy(d_a, d_in, N * sizeof(float), cudaMemcpyDeviceToDevice));
        float *res = reduce_all(kernels[v], d_a, d_b, N);
        CUDA_CHECK(cudaGetLastError());
        float g;
        CUDA_CHECK(cudaMemcpy(&g, res, sizeof(float), cudaMemcpyDeviceToHost));
        double rel = fabs((double)g - cpu) / cpu;
        bool ok = rel < 1e-4;
        all_ok &= ok;
        printf("%-50s GPU = %.4f  rel.err = %.2e  %s\n", names[v], g, rel, ok ? "PASS" : "FAIL");
    }

    // ---- timing: warm-up pass (untimed), then round-robin timed passes
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    float total_ms[3] = {0, 0, 0}, min_ms[3] = {1e9f, 1e9f, 1e9f};

    for (int rep = -1; rep < REPS; rep++) {           // rep = -1 is the warm-up
        for (int v = 0; v < 3; v++) {
            CUDA_CHECK(cudaMemcpy(d_a, d_in, N * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaEventRecord(s));
            reduce_all(kernels[v], d_a, d_b, N);
            CUDA_CHECK(cudaEventRecord(e));
            CUDA_CHECK(cudaEventSynchronize(e));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
            if (rep >= 0) { total_ms[v] += ms; if (ms < min_ms[v]) min_ms[v] = ms; }
        }
    }

    printf("\n---- Timing (warm, %d reps, full multi-pass reduction of %u floats) ----\n", REPS, N);
    printf("%-50s %10s %10s %12s %9s\n", "variant", "avg ms", "min ms", "GB/s (avg)", "vs v1");
    float avg1 = total_ms[0] / REPS;
    for (int v = 0; v < 3; v++) {
        float avg = total_ms[v] / REPS;
        double gbs = (N * 4.0 / 1e9) / (avg / 1e3);   // bytes read in pass 1 dominate
        printf("%-50s %10.4f %10.4f %12.1f %8.2fx\n", names[v], avg, min_ms[v], gbs, avg1 / avg);
    }
    double peak = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("\nReference: theoretical DRAM peak from device properties = %.0f GB/s "
           "(2 x %d kHz x %d-bit bus).\n", peak, prop.memoryClockRate, prop.memoryBusWidth);
    printf("A GB/s figure near that peak means the kernel is memory-bound, "
           "so warp divergence in shared memory cannot make it much faster.\n");

    free(h_in);
    cudaFree(d_in); cudaFree(d_a); cudaFree(d_b);
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

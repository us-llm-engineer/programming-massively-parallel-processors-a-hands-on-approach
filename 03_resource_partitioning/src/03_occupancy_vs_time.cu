/*
 * 03_occupancy_vs_time.cu
 *
 * Topic: does lower occupancy (from resource partitioning) really cost time?
 * Book (Ch.6): with fewer resident warps the SM has fewer warps to switch to while others wait
 * for memory, so latency is exposed. This program MEASURES that, and shows when it does NOT hold.
 *
 * Trick: the same kernel is launched with growing DYNAMIC shared memory per block. The kernel
 * never uses it; it only makes the SM refuse more blocks. Compute and memory work stay
 * identical, so any time change is caused by residency alone.
 *
 * Four kernels, block = 128 threads, 1M threads total:
 *   RAND : 8 dependent loads at RANDOM addresses in a 64 MB table (uncoalesced, DRAM-throughput-bound)
 *   SEQ  : 16 dependent loads, consecutive threads -> consecutive addresses (coalesced, latency-bound)
 *   FMA1 : one dependent chain of 2000 fused multiply-adds
 *   FMA4 : four independent chains x 500 (same FLOPs; instruction-level parallelism per thread)
 * Timing: 1 warm-up round, then REPS timed launches averaged.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BS 128
#define THREADS (1 << 20)
#define TABLE (1 << 24)            // 16M ints = 64 MB, far bigger than L2
#define REPS 15
#define NK 4

__global__ void mem_rand(const unsigned *next, float *out) {
    extern __shared__ char dyn[];
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = (i * 2654435761u) & (TABLE - 1);
    for (int r = 0; r < 8; r++) idx = next[idx];
    out[i] = (float)idx;
    if (idx == 0xFFFFFFFFu) dyn[0] = 1;
}
__global__ void mem_seq(const unsigned *next, float *out) {     // next[x] = (x + 1M) & mask
    extern __shared__ char dyn[];
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = i;
    for (int r = 0; r < 16; r++) idx = next[idx];               // dependent, but coalesced
    out[i] = (float)idx;
    if (idx == 0xFFFFFFFFu) dyn[0] = 1;
}
__global__ void fma1(float *out) {
    extern __shared__ char dyn[];
    float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 1e-7f;
    for (int r = 0; r < 2000; r++) a = fmaf(a, b, c);
    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
    if (a == 12345.f) dyn[0] = 1;
}
__global__ void fma4(float *out) {
    extern __shared__ char dyn[];
    float a = threadIdx.x * 1e-3f, b = a + 1.f, d = a + 2.f, e = a + 3.f, m = 1.000001f, c = 1e-7f;
    for (int r = 0; r < 500; r++) { a = fmaf(a, m, c); b = fmaf(b, m, c); d = fmaf(d, m, c); e = fmaf(e, m, c); }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a + b + d + e;
    if (a == 12345.f) dyn[0] = 1;
}

static int occ(const void *k, size_t dyn) {
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, BS, dyn); return nb;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("========== OCCUPANCY vs MEASURED TIME on %s ==========\n", p.name);
    printf("block = %d threads, %d threads total, %d reps after warm-up\n\n", BS, THREADS, REPS);

    unsigned *h = (unsigned *)malloc((size_t)TABLE * 4);
    unsigned *hs = (unsigned *)malloc((size_t)TABLE * 4);
    srand(7);
    for (unsigned i = 0; i < TABLE; i++) {
        h[i]  = ((unsigned)rand() * 65536u + (unsigned)rand()) & (TABLE - 1);   // random pointers
        hs[i] = (i + (1u << 20)) & (TABLE - 1);                                  // predictable stride, coalesced
    }
    unsigned *d_rand, *d_seq; float *d_out;
    cudaMalloc(&d_rand, (size_t)TABLE * 4); cudaMalloc(&d_seq, (size_t)TABLE * 4); cudaMalloc(&d_out, THREADS * 4);
    cudaMemcpy(d_rand, h, (size_t)TABLE * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq, hs, (size_t)TABLE * 4, cudaMemcpyHostToDevice);
    free(h); free(hs);

    // dynamic-smem sizes giving each distinct blocks/SM value (largest smem that still gives it)
    size_t chosen[32]; int nb_of[32], n = 0;
    for (size_t s = 0; s <= 49152; s += 256) {
        int nb = occ((const void *)fma1, s);
        if (nb == 0) break;
        if (n == 0 || nb != nb_of[n - 1]) { chosen[n] = s; nb_of[n] = nb; n++; }
        else chosen[n - 1] = s;
    }

    const char *kn[NK] = { "RAND", "SEQ", "FMA1", "FMA4" };
    float tot[NK][32] = {};
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    int grid = THREADS / BS;

    for (int rep = -1; rep < REPS; rep++)
        for (int q = 0; q < n; q++)
            for (int kk = 0; kk < NK; kk++) {
                cudaEventRecord(s0);
                if (kk == 0) mem_rand<<<grid, BS, chosen[q]>>>(d_rand, d_out);
                if (kk == 1) mem_seq <<<grid, BS, chosen[q]>>>(d_seq, d_out);
                if (kk == 2) fma1    <<<grid, BS, chosen[q]>>>(d_out);
                if (kk == 3) fma4    <<<grid, BS, chosen[q]>>>(d_out);
                cudaEventRecord(s1); cudaEventSynchronize(s1);
                float ms; cudaEventElapsedTime(&ms, s0, s1);
                if (cudaGetLastError() != cudaSuccess) { printf("launch failed\n"); return 1; }
                if (rep >= 0) tot[kk][q] += ms;
            }

    printf("%-8s %-9s %-9s %-8s |", "smem/blk", "blocks/SM", "warps/SM", "occup.");
    for (int kk = 0; kk < NK; kk++) printf(" %-17s|", kn[kk]);
    printf("\n%-8s %-9s %-9s %-8s |", "", "", "", "");
    for (int kk = 0; kk < NK; kk++) printf(" %-17s|", "ms (x vs full)");
    printf("\n");
    for (int q = 0; q < n; q++) {
        printf("%-8zu %-9d %-9d %6.1f%%  |", chosen[q], nb_of[q], nb_of[q] * BS / 32,
               100.0 * nb_of[q] * BS / p.maxThreadsPerMultiProcessor);
        for (int kk = 0; kk < NK; kk++) {
            float t = tot[kk][q] / REPS, t0 = tot[kk][0] / REPS;
            printf(" %8.3f (%5.2fx) |", t, t / t0);
        }
        printf("\n");
    }
    printf("\n'x vs full' = time relative to the first row (max blocks/SM). 2.00x means twice as slow.\n");
    printf("Small steps (a few %%) are within run-to-run noise; an earlier run showed one 13%% outlier.\n");
    printf("\nHow to read:\n");
    printf("  RAND : random 64 MB gathers are limited by DRAM random-access throughput; even a few warps\n");
    printf("         keep the memory system busy, so residency barely matters (no exposed-latency effect).\n");
    printf("  SEQ  : coalesced dependent loads are cheap for DRAM but each step waits a full memory latency,\n");
    printf("         so time tracks how many warps are resident to cover that wait (the book's claim).\n");
    printf("  FMA1 : a dependent FMA chain needs a few warps per scheduler; it only degrades at the lowest\n");
    printf("         occupancy rows.\n");
    printf("  FMA4 : four independent chains give each thread its own parallelism (ILP), so it tolerates\n");
    printf("         low occupancy better. Occupancy is a means to hide latency, not the goal itself.\n");

    cudaFree(d_rand); cudaFree(d_seq); cudaFree(d_out);
    return 0;
}

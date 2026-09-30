/*
 * 02_resource_cliffs.cu
 *
 * Topic: the "performance cliff" (PMPP Ch.6 s6.3): one more register per thread, or a bit more
 * shared memory per block, can remove a WHOLE block from the SM and cut resident threads sharply.
 *
 * A. REGISTER CLIFF: kernels that keep K accumulators live in registers. The compiler decides the
 *    real register count (printed, not assumed). Block size fixed at 256 threads.
 * B. SHARED-MEMORY CLIFF: same kernel, dynamic shared memory per block swept upward.
 * C. BOTH AT ONCE: which limiter wins as registers and shared memory both grow.
 * Resident-block counts come from the CUDA runtime (cudaOccupancyMaxActiveBlocksPerMultiprocessor),
 * not from a formula, so they include the hardware's allocation granularity.
 * Steps where resident threads DROP by 25% or more are marked with '<-- CLIFF'.
 */
#include <stdio.h>
#include <cuda_runtime.h>

// Keeps K floats live across the loop -> ~K+ registers per thread.
template <int K>
__global__ void reg_kernel(const float *in, float *out, int n) {
    extern __shared__ float dyn[];                     // dynamic smem (size chosen at launch)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[K];
#pragma unroll
    for (int j = 0; j < K; j++) acc[j] = in[(i + j * 977) % n];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int j = 0; j < K; j++) acc[j] = acc[j] * 1.0001f + acc[(j + 1) % K];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < K; j++) s += acc[j];
    if (dyn) dyn[threadIdx.x % 32] = s;                // touch dyn smem so it is not optimised away
    out[i] = s;
}

static int occ(const void *k, int bs, size_t dyn) {
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, bs, dyn); return nb;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    const int BS = 256, maxThr = p.maxThreadsPerMultiProcessor;
    printf("========== RESOURCE CLIFFS on %s (%d thread slots/SM, %d regs/SM, %zu B smem/SM) ==========\n\n",
           p.name, maxThr, p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor);

    struct V { int K; const void *f; } v[] = {
        {4, (const void *)reg_kernel<4>},   {8, (const void *)reg_kernel<8>},
        {16, (const void *)reg_kernel<16>}, {32, (const void *)reg_kernel<32>},
        {48, (const void *)reg_kernel<48>}, {64, (const void *)reg_kernel<64>},
        {80, (const void *)reg_kernel<80>}, {96, (const void *)reg_kernel<96>},
        {128, (const void *)reg_kernel<128>}, {160, (const void *)reg_kernel<160>} };

    printf("---- A. Register cliff (block = %d threads, no smem) ----\n", BS);
    printf("  %-6s %-10s %-14s %-10s %-10s %-10s %s\n",
           "K", "regs/thr", "regs/block", "blocks/SM", "threads/SM", "occupancy", "");
    int prev = -1;
    for (auto &x : v) {
        cudaFuncAttributes f; cudaFuncGetAttributes(&f, x.f);
        int nb = occ(x.f, BS, 0), thr = nb * BS;
        bool cliff = prev > 0 && thr * 4 <= prev * 3;
        printf("  %-6d %-10d %-14d %-10d %-10d %8.1f%%  %s%s\n", x.K, f.numRegs, f.numRegs * BS, nb, thr,
               100.0 * thr / maxThr, cliff ? "<-- CLIFF" : "", f.numRegs >= 255 ? " (spilling?)" : "");
        prev = thr;
    }
    printf("  Read: occupancy falls in whole-block steps (%d threads each), never smoothly.\n", BS);
    printf("  Registers per thread is decided by the compiler, so a 'harmless' code change can\n");
    printf("  move a kernel across a step. The book's 10->11 example is exactly this.\n\n");

    printf("---- B. Shared-memory cliff (block = %d threads, kernel with few registers) ----\n", BS);
    const void *k = (const void *)reg_kernel<8>;
    cudaFuncAttributes f8; cudaFuncGetAttributes(&f8, k);
    printf("  kernel uses %d regs/thread\n", f8.numRegs);
    printf("  %-14s %-10s %-10s %-10s %s\n", "smem/block", "blocks/SM", "threads/SM", "occupancy", "");
    prev = -1;
    size_t sizes[] = { 0, 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 49152 };
    for (size_t s : sizes) {
        int nb = occ(k, BS, s), thr = nb * BS;
        bool cliff = prev > 0 && thr * 4 <= prev * 3;
        printf("  %-14zu %-10d %-10d %8.1f%%  %s\n", s, nb, thr, 100.0 * thr / maxThr, cliff ? "<-- CLIFF" : "");
        prev = thr;
    }
    printf("  Read: with %zu B of smem per SM the block count is floor(smem_SM / smem_block);\n",
           p.sharedMemPerMultiprocessor);
    printf("  the runtime also reserves a small amount per block, so exact boundaries sit slightly\n");
    printf("  below the round numbers (book's G80 example: 5 KB/block -> 3 blocks, not 4).\n\n");

    printf("---- C. Two resources at once: which limiter wins? (block = %d) ----\n", BS);
    printf("  %-8s %-12s %-10s %-10s %s\n", "regs", "smem/block", "blocks/SM", "threads/SM", "note");
    struct C { int Kidx; size_t smem; } cs[] = { {1, 0}, {5, 0}, {1, 24576}, {5, 24576}, {7, 0}, {7, 12288} };
    for (auto &c : cs) {
        cudaFuncAttributes f; cudaFuncGetAttributes(&f, v[c.Kidx].f);
        int nb = occ(v[c.Kidx].f, BS, c.smem);
        int nr = occ(v[c.Kidx].f, BS, 0), ns = occ(k, BS, c.smem);
        const char *why = (nb == nr && nb == ns) ? "both allow the same" :
                          (nb == nr) ? "REGISTERS bind" : (nb == ns) ? "SHARED MEMORY binds" : "mixed";
        printf("  %-8d %-12zu %-10d %-10d %s\n", f.numRegs, c.smem, nb, nb * BS, why);
    }
    printf("  Read: the SM obeys the tightest of all partitions at once; fixing the one you\n");
    printf("  expected may change nothing if another resource is the real limiter.\n");
    return 0;
}

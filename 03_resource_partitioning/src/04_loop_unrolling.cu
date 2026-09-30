/*
 * 04_loop_unrolling.cu
 *
 * Topic: LOOP UNROLLING (PMPP Ch.6 s6.5 "Instruction Mix", s6.7 Fig 6.16).
 *
 * What the book says:
 *   - A normal loop iteration spends instructions on things that are not the real work:
 *     counter update, conditional branch, address arithmetic from the counter. For the tiled
 *     matmul dot-product loop the book counts 6 instructions/iteration, only 2 floating point
 *     (= 1/3), which "limits performance to no more than 1/3 of peak".
 *   - Unrolling removes counter+branch, and constant indices let the compiler fold addresses
 *     into load offsets.
 *   - Costs: more registers (can lose a whole block -> occupancy cliff, Ch.6 s6.3), bigger code.
 *   - It helps only if instruction issue is the bottleneck; if memory bandwidth is (8x8 tiles in
 *     Fig 6.16) all unroll bars are the same height.
 *
 * Three experiments, all measured warm (1 warm-up round, REPS averaged, variants interleaved):
 *   1. The book's own experiment: tiled matmul, tile 8/16/32, unroll 1 / 2 / 4 / complete.
 *      Shows registers, resident blocks per SM, time. (#pragma unroll U controls the unroll.)
 *   2. Pure instruction-issue test: 8 independent FMA chains, unroll 1..16. Loop overhead is a
 *      large share of instructions here, so unrolling should matter most.
 *   3. Memory-bound streaming sum, unroll 1..16: does unrolling help when DRAM is the limit?
 *
 * Note '#pragma unroll 1' forbids unrolling; '#pragma unroll U' asks for factor U.
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

#define N 1024
#define REPS 20

// ============================ experiment 1: tiled matmul ============================
template <int T, int U>
__global__ void mm(const float *A, const float *B, float *C, int W) {
    __shared__ float Ms[T][T], Ns[T][T];
    int tx = threadIdx.x, ty = threadIdx.y, r = blockIdx.y * T + ty, c = blockIdx.x * T + tx;
    float v = 0.f;
    for (int m = 0; m < W / T; ++m) {
        Ms[ty][tx] = A[r * W + m * T + tx];
        Ns[ty][tx] = B[(m * T + ty) * W + c];
        __syncthreads();
#pragma unroll U
        for (int k = 0; k < T; ++k) v += Ms[ty][k] * Ns[k][tx];    // <- the loop the book unrolls
        __syncthreads();
    }
    C[r * W + c] = v;
}
typedef void (*MMFn)(const float *, const float *, float *, int);
struct MV { int T, U; MMFn f; };
#define VARS(T) { T, 1, mm<T, 1> }, { T, 2, mm<T, 2> }, { T, 4, mm<T, 4> }, { T, T, mm<T, T> }
static MV mv[] = { VARS(8), VARS(16), VARS(32) };

// ============================ experiment 2: pure instruction issue ==================
template <int U>
__global__ void alu(float *out, int iters) {
    float a0 = threadIdx.x * 1e-3f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3,
          a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float m = 1.0000001f, c = 1e-7f;
#pragma unroll U
    for (int i = 0; i < iters; i++) {                 // runtime trip count: compiler cannot fully unroll
        a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
        a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// ============================ experiment 3: memory-bound stream =====================
template <int U>
__global__ void stream(const float *in, float *out, int total) {   // 16 coalesced loads per thread
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
#pragma unroll U
    for (int r = 0; r < 16; r++) s += in[i + (size_t)r * total];
    out[i] = s;
}

static int occ(const void *k, int bs) {
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, bs, 0); return nb;
}
static int regs(const void *k) { cudaFuncAttributes f; cudaFuncGetAttributes(&f, k); return f.numRegs; }

// Time nv variants: one untimed warm-up round, REPS timed rounds, variants interleaved.
template <class F>
static void bench(int nv, F launch, float *avg) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int v = 0; v < nv; v++) avg[v] = 0;
    for (int rep = -1; rep < REPS; rep++)
        for (int v = 0; v < nv; v++) {
            cudaEventRecord(s); launch(v); cudaEventRecord(e); cudaEventSynchronize(e);
            CUDA_CHECK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, s, e);
            if (rep >= 0) avg[v] += ms;
        }
    for (int v = 0; v < nv; v++) avg[v] /= REPS;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("========== LOOP UNROLLING on %s ==========\n", p.name);
    printf("warm timings: 1 warm-up round + %d averaged rounds, variants interleaved\n\n", REPS);

    // ------------------------------------------------------------------ experiment 1
    printf("---- 1. Tiled matmul dot-product loop (book Fig 6.14 / 6.16), N = %d ----\n", N);
    size_t bytes = (size_t)N * N * 4;
    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes), *hC = (float *)malloc(bytes);
    srand(1);
    for (int i = 0; i < N * N; i++) { hA[i] = (float)rand() / RAND_MAX; hB[i] = (float)rand() / RAND_MAX; }
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes)); CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    int nmv = sizeof(mv) / sizeof(mv[0]);

    bool ok = true;
    for (int v = 0; v < nmv; v++) {                    // correctness: 16 sampled rows vs double CPU
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        dim3 blk(mv[v].T, mv[v].T), grd(N / mv[v].T, N / mv[v].T);
        mv[v].f<<<grd, blk>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int s = 0; s < 16; s++) {
            int i = (s * 61 + 3) % N;
            for (int j = 0; j < N; j++) {
                double ref = 0; for (int k = 0; k < N; k++) ref += (double)hA[i * N + k] * hB[k * N + j];
                double err = fabs(ref - hC[i * N + j]) / ref; if (err > worst) worst = err;
            }
        }
        if (worst > 1e-4) { ok = false; printf("  FAIL tile %d unroll %d err %.2e\n", mv[v].T, mv[v].U, worst); }
    }
    printf("  correctness of all %d variants vs CPU (16 sampled rows): %s\n", nmv, ok ? "PASS" : "FAIL");

    float t1[16];
    bench(nmv, [&](int v) {
        dim3 blk(mv[v].T, mv[v].T), grd(N / mv[v].T, N / mv[v].T);
        mv[v].f<<<grd, blk>>>(dA, dB, dC, N);
    }, t1);
    double flops = 2.0 * N * N * N;
    printf("  %-6s %-10s %-6s %-10s %-9s %-9s %s\n", "tile", "unroll", "regs", "blocks/SM", "ms", "GFLOP/s", "vs unroll 1");
    for (int v = 0; v < nmv; v++) {
        int base = v - (v % 4);                       // first variant of this tile = unroll 1
        char u[16]; snprintf(u, sizeof u, mv[v].U == 1 ? "1 (none)" : mv[v].U == mv[v].T ? "%d (full)" : "%d", mv[v].U);
        printf("  %-6d %-10s %-6d %-10d %-9.3f %-9.1f %.2fx\n", mv[v].T, u, regs((const void *)mv[v].f),
               occ((const void *)mv[v].f, mv[v].T * mv[v].T), t1[v], flops / (t1[v] * 1e-3) / 1e9, t1[base] / t1[v]);
    }
    printf("  Book reference (G80): full unroll gave >20%% at 16x16 tiles, ~0%% at 8x8 (memory-bound).\n\n");

    // ------------------------------------------------------------------ experiment 2
    printf("---- 2. Pure instruction issue: 8 independent FMA chains ----\n");
    const int ALU_THREADS = 1 << 18, ITERS = 2048;
    float *dOut; CUDA_CHECK(cudaMalloc(&dOut, (size_t)(1 << 20) * 4));
    typedef void (*AF)(float *, int);
    AF af[] = { alu<1>, alu<2>, alu<4>, alu<8>, alu<16> };
    int au[] = { 1, 2, 4, 8, 16 };
    float t2[8];
    bench(5, [&](int v) { af[v]<<<ALU_THREADS / 256, 256>>>(dOut, ITERS); }, t2);
    double aflops = 2.0 * 8 * ITERS * ALU_THREADS;
    printf("  %-8s %-6s %-10s %-9s %-9s %s\n", "unroll", "regs", "blocks/SM", "ms", "GFLOP/s", "vs unroll 1");
    for (int v = 0; v < 5; v++)
        printf("  %-8d %-6d %-10d %-9.3f %-9.1f %.2fx\n", au[v], regs((const void *)af[v]),
               occ((const void *)af[v], 256), t2[v], aflops / (t2[v] * 1e-3) / 1e9, t2[0] / t2[v]);
    printf("  Loop body = 8 FMAs; overhead per iteration (counter, compare, branch) is a visible share\n");
    printf("  of issued instructions, which is the situation the book's 1/3 argument describes.\n\n");

    // ------------------------------------------------------------------ experiment 3
    printf("---- 3. Memory-bound streaming sum (64 MB read, 16 coalesced loads/thread) ----\n");
    const int S_THREADS = 1 << 20;
    float *dIn; CUDA_CHECK(cudaMalloc(&dIn, (size_t)16 * S_THREADS * 4));
    CUDA_CHECK(cudaMemset(dIn, 0, (size_t)16 * S_THREADS * 4));
    typedef void (*SF)(const float *, float *, int);
    SF sf[] = { stream<1>, stream<2>, stream<4>, stream<16> };
    int su[] = { 1, 2, 4, 16 };
    float t3[8];
    bench(4, [&](int v) { sf[v]<<<S_THREADS / 256, 256>>>(dIn, dOut, S_THREADS); }, t3);
    double peak = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
    printf("  %-8s %-6s %-10s %-9s %-9s %s\n", "unroll", "regs", "blocks/SM", "ms", "GB/s", "vs unroll 1");
    for (int v = 0; v < 4; v++)
        printf("  %-8d %-6d %-10d %-9.3f %-9.1f %.2fx\n", su[v], regs((const void *)sf[v]),
               occ((const void *)sf[v], 256), t3[v], (16.0 * S_THREADS * 4 / 1e9) / (t3[v] * 1e-3), t3[0] / t3[v]);
    printf("  Theoretical DRAM peak from device properties: %.0f GB/s.\n", peak);
    printf("  If every row sits at the same GB/s, DRAM (not instructions) is the limit and unrolling\n");
    printf("  cannot help, which is the book's 8x8-tile observation.\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dOut); cudaFree(dIn);
    free(hA); free(hB); free(hC);
    return ok ? 0 : 1;
}

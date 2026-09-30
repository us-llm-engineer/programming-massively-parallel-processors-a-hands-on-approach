/*
 * 01_matmul_tiling.cu   (PMPP Ch.5/6: tiling)
 *
 * Book claim: with TILE_WIDTH x TILE_WIDTH tiles staged in shared memory, global
 * memory traffic drops by a factor of TILE_WIDTH (each loaded element is reused
 * TILE_WIDTH times).  This program checks that claim two ways:
 *   (a) THEORY  - a traffic/arithmetic-intensity model (formula, not a measurement)
 *   (b) MEASURED - warm, averaged kernel times for naive vs tiled at TILE_WIDTH 8/16/32,
 *                  with the naive kernel launched with the SAME block shape as the tiled one.
 *
 * Measurement rules: one untimed warm-up launch per variant, REPS timed launches
 * averaged, variants interleaved round-robin, correctness checked on sampled rows
 * against a double-precision CPU reference.
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

#define N 1024        // matrices are N x N (N divisible by 8, 16 and 32)
#define REPS 20

// ---- naive: every thread streams a full row of A and column of B from DRAM ----
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

// ---- tiled: book's Mds/Nds kernel (Figure 5.7); TILE_WIDTH is a template arg ----
template <int TILE_WIDTH>
__global__ void matmul_tiled(const float *Md, const float *Nd, float *Pd, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];   // on-chip SRAM
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    for (int m = 0; m < Width / TILE_WIDTH; ++m) {
        Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];   // DRAM -> SRAM
        Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)                     // SRAM reads only
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    Pd[Row * Width + Col] = Pvalue;
}

// Rows-sampled verification against a double-precision CPU reference.
static bool verify(const float *h_A, const float *h_B, const float *h_C, const char *label) {
    double worst = 0.0;
    for (int s = 0; s < 32; s++) {
        int i = (s * 37 + 5) % N;                     // 32 spread-out rows
        for (int j = 0; j < N; j++) {
            double ref = 0.0;
            for (int k = 0; k < N; k++) ref += (double)h_A[i * N + k] * h_B[k * N + j];
            double err = fabs(ref - h_C[i * N + j]) / fabs(ref);
            if (err > worst) worst = err;
        }
    }
    bool ok = worst < 1e-4;
    printf("  %-22s max rel.err over 32 sampled rows = %.2e  %s\n", label, worst, ok ? "PASS" : "FAIL");
    return ok;
}

template <int T>
static void launch_tiled(const float *A, const float *B, float *C) {
    dim3 blk(T, T), grd(N / T, N / T);
    matmul_tiled<T><<<grd, blk>>>(A, B, C, N);
}
static void launch_naive(int T, const float *A, const float *B, float *C) {
    dim3 blk(T, T), grd(N / T, N / T);
    matmul_naive<<<grd, blk>>>(A, B, C, N);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("========== TILED MATRIX MULTIPLY (PMPP Ch.5) ==========\n");
    printf("GPU: %s | N = %d | reps = %d (after 1 warm-up)\n\n", prop.name, N, REPS);

    // ---------- (a) theory ----------
    printf("---- (a) THEORY: traffic model (formula, NOT measured) ----\n");
    double flops = 2.0 * N * N * N;
    printf("  FLOPs = 2N^3 = %.3e\n", flops);
    printf("  %-12s %-22s %-22s\n", "TILE_WIDTH", "DRAM bytes (model)", "FLOPs/Byte (model)");
    printf("  %-12s %-22.3e %-22.3f\n", "naive", 2.0 * N * N * N * 4, flops / (2.0 * N * N * N * 4));
    int Ts[3] = {8, 16, 32};
    for (int i = 0; i < 3; i++) {
        double bytes = 2.0 * N * N * N * 4 / Ts[i];
        printf("  %-12d %-22.3e %-22.3f\n", Ts[i], bytes, flops / bytes);
    }
    printf("  (model ignores L1/L2 caching, which is why measured gains differ)\n\n");

    // ---------- data ----------
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes), *h_B = (float *)malloc(bytes), *h_C = (float *)malloc(bytes);
    srand(1);
    for (int i = 0; i < N * N; i++) { h_A[i] = (float)rand() / RAND_MAX; h_B[i] = (float)rand() / RAND_MAX; }
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // variants: 0..2 naive T=8,16,32 ; 3..5 tiled T=8,16,32
    const char *names[6] = {"naive  block 8x8", "naive  block 16x16", "naive  block 32x32",
                            "tiled  TILE_WIDTH 8", "tiled  TILE_WIDTH 16", "tiled  TILE_WIDTH 32"};
    auto run = [&](int v) {
        switch (v) {
            case 0: launch_naive(8,  d_A, d_B, d_C); break;
            case 1: launch_naive(16, d_A, d_B, d_C); break;
            case 2: launch_naive(32, d_A, d_B, d_C); break;
            case 3: launch_tiled<8> (d_A, d_B, d_C); break;
            case 4: launch_tiled<16>(d_A, d_B, d_C); break;
            case 5: launch_tiled<32>(d_A, d_B, d_C); break;
        }
    };

    // ---------- correctness ----------
    printf("---- Correctness ----\n");
    bool all_ok = true;
    for (int v = 0; v < 6; v++) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        run(v);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        all_ok &= verify(h_A, h_B, h_C, names[v]);
    }

    // ---------- (b) measured ----------
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    float total[6] = {0}, best[6] = {1e9f, 1e9f, 1e9f, 1e9f, 1e9f, 1e9f};
    for (int rep = -1; rep < REPS; rep++) {           // rep = -1: warm-up, not recorded
        for (int v = 0; v < 6; v++) {
            CUDA_CHECK(cudaEventRecord(s));
            run(v);
            CUDA_CHECK(cudaEventRecord(e));
            CUDA_CHECK(cudaEventSynchronize(e));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
            if (rep >= 0) { total[v] += ms; if (ms < best[v]) best[v] = ms; }
        }
    }
    printf("\n---- (b) MEASURED (warm, %d-run average) ----\n", REPS);
    printf("  %-24s %10s %10s %12s %14s\n", "variant", "avg ms", "min ms", "GFLOP/s", "vs naive 32x32");
    float ref = total[2] / REPS;
    for (int v = 0; v < 6; v++) {
        float avg = total[v] / REPS;
        printf("  %-24s %10.3f %10.3f %12.1f %13.2fx\n", names[v], avg, best[v],
               flops / (avg * 1e-3) / 1e9, ref / avg);
    }
    printf("\nHow to read: 'vs naive 32x32' > 1 means faster than the naive kernel launched\n");
    printf("with the same block shape as the tiled TILE_WIDTH 32 kernel.\n");

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

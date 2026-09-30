/*
 * 01_partitioning_model.cu
 *
 * Topic: DYNAMIC PARTITIONING of SM resources  (PMPP Ch.4 s4.5, Ch.5 s5.4, Ch.6 s6.3)
 *
 * Book's statement (Ch.6 s6.3): registers, thread-block slots and thread slots of an SM are
 * "dynamically partitioned and assigned to threads", at BLOCK granularity. A block gets its
 * whole quota or is not resident. The resident-block count is therefore
 *
 *     blocks/SM = min( threadSlots/blockSize, blockSlots,
 *                      registers/(regsPerThread*blockSize), sharedMem/smemPerBlock )
 *
 * and whichever term is smallest is the LIMITER. Two hardware caps interact:
 * that interaction is the "conflict".
 *
 * This program
 *   A. lists the book's G80 limits next to this GPU's real limits
 *   B. reproduces every numeric example in the book with the formula above (checked)
 *   C. applies the formula to this GPU for block sizes 32..1024 and compares it with the
 *      CUDA runtime's own answer (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
 *   D. applies it to the batch-3 tiled matmul kernels (TILE_WIDTH 8/16/32)
 * The formula ignores allocation granularity, so small disagreements with the runtime are
 * expected and flagged.
 */
#include <stdio.h>
#include <cuda_runtime.h>

struct Lim { const char *name; int threads, blocks, regs, smem, maxTPB; };

static int blocks_model(const Lim &L, int bs, int rpt, int smemPB, const char **limiter) {
    if (bs > L.maxTPB) { *limiter = "block too big"; return 0; }
    int c[4] = { L.threads / bs, L.blocks,
                 rpt    ? L.regs / (rpt * bs)   : 1 << 30,
                 smemPB ? L.smem / smemPB       : 1 << 30 };
    const char *nm[4] = { "thread slots", "block slots", "registers", "shared mem" };
    int best = 0;
    for (int i = 1; i < 4; i++) if (c[i] < c[best]) best = i;
    *limiter = (best == 0 && c[0] == c[1]) ? "thread=block" : nm[best];   // exact tie
    return c[best];
}

__global__ void trivial_kernel(float *p) { p[blockIdx.x * blockDim.x + threadIdx.x] += 1.0f; }

template <int T>
__global__ void tiled(const float *M, const float *N, float *P, int W) {
    __shared__ float Ms[T][T], Ns[T][T];
    int tx = threadIdx.x, ty = threadIdx.y, r = blockIdx.y * T + ty, c = blockIdx.x * T + tx;
    float v = 0.f;
    for (int m = 0; m < W / T; ++m) {
        Ms[ty][tx] = M[r * W + m * T + tx]; Ns[ty][tx] = N[(m * T + ty) * W + c];
        __syncthreads();
        for (int k = 0; k < T; ++k) v += Ms[ty][k] * Ns[k][tx];
        __syncthreads();
    }
    P[r * W + c] = v;
}

static int api_blocks(const void *k, int bs, size_t dyn) {
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, bs, dyn);
    return nb;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    Lim G80 = { "G80 (book)", 768, 8, 8192, 16384, 512 };
    Lim me  = { p.name, p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor,
                p.regsPerMultiprocessor, (int)p.sharedMemPerMultiprocessor, p.maxThreadsPerBlock };

    printf("========== DYNAMIC PARTITIONING OF SM RESOURCES ==========\n\n");
    printf("---- A. Per-SM limits: the book's G80 vs this GPU ----\n");
    printf("  %-32s %12s %12s\n", "resource (per SM)", "G80 (book)", "this GPU");
    printf("  %-32s %12d %12d\n", "thread slots", G80.threads, me.threads);
    printf("  %-32s %12d %12d\n", "thread-block slots", G80.blocks, me.blocks);
    printf("  %-32s %12d %12d\n", "32-bit registers", G80.regs, me.regs);
    printf("  %-32s %12d %12d\n", "shared memory (bytes)", G80.smem, me.smem);
    printf("  %-32s %12d %12d\n", "max threads per block", G80.maxTPB, me.maxTPB);
    printf("  this GPU: %s, %d SMs, warp = %d -> max %d resident warps/SM\n\n",
           p.name, p.multiProcessorCount, p.warpSize, me.threads / p.warpSize);

    printf("---- B. The book's own examples, reproduced with the formula (G80 limits) ----\n");
    struct Ex { const char *what; int bs, rpt, smem, book_blocks; } ex[] = {
        { "256 thr, 10 regs/thr           (Ch.6: 3 blocks, 768 thr)", 256, 10, 0,    3 },
        { "256 thr, 11 regs/thr  <- cliff (Ch.6: 2 blocks, 512 thr)", 256, 11, 0,    2 },
        { "64 thr/block                   (Ch.6: 8 blocks, 512 thr)", 64,  0,  0,    8 },
        { "256 thr, 5 KB smem/block       (Ch.5: 3 blocks)", 256, 0,  5 * 1024, 3 },
        { "256 thr, 2 KB smem (16x16 tile)(Ch.5: 3 blocks, 6 KB used)", 256, 0, 2048, 3 },
    };
    printf("  %-64s %6s %6s  %-13s %s\n", "case", "model", "book", "limiter", "threads/SM");
    for (auto &e : ex) {
        const char *lim; int b = blocks_model(G80, e.bs, e.rpt, e.smem, &lim);
        printf("  %-64s %6d %6d  %-13s %d  %s\n", e.what, b, e.book_blocks, lim, b * e.bs,
               b == e.book_blocks ? "OK" : "MISMATCH");
    }
    printf("  Read: 10->11 registers per thread costs a whole block: 768 -> 512 threads (-33%%).\n");
    printf("  The book calls this a 'performance cliff'.\n\n");

    printf("---- C. This GPU: blocks/SM vs block size (trivial kernel, no smem) ----\n");
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, trivial_kernel);
    printf("  kernel uses %d registers/thread, %zu B static smem\n", fa.numRegs, fa.sharedSizeBytes);
    printf("  %-8s %-10s %-10s %-13s %-10s %-12s %s\n",
           "block", "model", "runtime", "limiter", "threads/SM", "warps/SM", "occupancy");
    for (int bs = 32; bs <= 1024; bs *= 2) {
        const char *lim; int m = blocks_model(me, bs, fa.numRegs, 0, &lim);
        int a = api_blocks((const void *)trivial_kernel, bs, 0);
        printf("  %-8d %-10d %-10d %-13s %-10d %-12d %5.1f%%  %s\n", bs, m, a, lim, a * bs,
               a * bs / p.warpSize, 100.0 * a * bs / me.threads, m == a ? "" : "(granularity)");
    }
    printf("  Read: block size 32 hits the BLOCK-slot cap (%d blocks x 1 warp = %d warps) -> 50%% occupancy\n",
           me.blocks, me.blocks);
    printf("  although thread slots and registers are free: the conflict between two hardware caps.\n");
    printf("  Sizes >= 64 fill the %d thread slots. (Book, G80: 64-thread blocks stop at 8 blocks = 512 thr.)\n\n", me.threads);

    printf("---- D. Batch-3 tiled matmul kernels on this GPU ----\n");
    struct T { int t; const void *k; } tk[] = { {8, (const void *)tiled<8>}, {16, (const void *)tiled<16>},
                                                 {32, (const void *)tiled<32>} };
    printf("  %-11s %-8s %-6s %-9s %-10s %-10s %-10s %s\n",
           "TILE_WIDTH", "threads", "regs", "smem B", "blocks/SM", "threads/SM", "occupancy", "limiter(model)");
    for (auto &x : tk) {
        cudaFuncAttributes f; cudaFuncGetAttributes(&f, x.k);
        int bs = x.t * x.t, a = api_blocks(x.k, bs, 0);
        const char *lim; blocks_model(me, bs, f.numRegs, (int)f.sharedSizeBytes, &lim);
        printf("  %-11d %-8d %-6d %-9zu %-10d %-10d %8.1f%%  %s\n", x.t, bs, f.numRegs, f.sharedSizeBytes,
               a, a * bs, 100.0 * a * bs / me.threads, lim);
    }
    printf("  Read: all three fill the thread slots (100%% occupancy; TILE_WIDTH 8 ties with the block-slot cap).\n");
    printf("  So residency does NOT explain batch 3's timings (tile 8 slowest, tile 32 fastest);\n");
    printf("  that ordering is consistent with data reuse (larger tile = fewer DRAM bytes), not occupancy.\n");
    return 0;
}

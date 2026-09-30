/*
 * 06_constant_cache_thrash.cu   -- CACHE MISSES around constant memory (book Ch.10, with the layout point from Ch.8)
 *
 * What the book says (from the chapter notes):
 *   1. Direct summation: every thread block reads the SAME atoms in the SAME order, so the small constant cache is hit
 *      more than 96 % of the time and each read is broadcast to the warp.
 *   2. Cutoff summation: different blocks work on different lattice tiles, need DIFFERENT atom neighbourhoods, and the
 *      blocks resident on one SM together want more atoms than the constant cache holds -> misses / thrashing. That is why
 *      SmallBin moves the atoms to global memory and tiles each block's neighbourhood into shared memory.
 *   3. Layout (Ch.8): separate x/y/z arrays need several cache entries per iteration; a struct keeps the fields together.
 *
 * This program measures the effects with kernels whose arithmetic is IDENTICAL, so only the memory behaviour differs:
 *   E1  every block reads the same window of W atoms   vs   every block reads a different window   x   constant vs global+shared
 *   E2  the 32 lanes of a warp read 1, 2, 4, 8, 16 or 32 distinct addresses (constant memory serialises different addresses)
 *   E3  separate arrays (SoA) vs 12-byte struct vs 16-byte struct in constant memory
 * Timing is measured here. The cache hit rates themselves come from Nsight Compute counters, collected separately with
 *   PROFILE_ONLY=1 scripts/profile_kernels.sh bin/06_constant_cache_thrash 'regex:window_kernel|lane_kernel|layout_kernel' stats/06_ncu.csv
 * and summarised in output/06_ncu_summary.txt (needs GPU performance counters enabled in the NVIDIA control panel).
 * Statistics: stats/06_constant_cache.csv
 */
#include "common.cuh"
#include "monitor.cuh"
#include <functional>
#include <sys/stat.h>

#define NA 4000                                   // 4000 atoms x 16 B = 64,000 B: the whole constant memory of the module
__constant__ float4 c_atoms[NA];
#define BLK 128
#define GRID 4096

__device__ __forceinline__ float energy_of(float4 A, float x, float y, float z) {
    float dx = x - A.x, dy = y - A.y, dz = z - A.z;
    return A.w * rsqrtf(dx * dx + dy * dy + dz * dz + 1.0f);      // +1 avoids singularities; no cutoff test: identical work everywhere
}

// ---- E1: block-dependent window of W atoms, from constant memory (MODE 0) or global memory tiled through shared memory (MODE 1) ----
template <int MODE>
__global__ void window_kernel(int W, int different, const float4 *g_atoms, float *out) {
    int gid = blockIdx.x * BLK + threadIdx.x;
    float x = (gid & 1023) * 0.01f, y = ((gid >> 10) & 255) * 0.01f, z = (gid >> 18) * 0.01f, e = 0.f;
    int start = different ? (int)((blockIdx.x * 2654435761u) % (unsigned)(NA - W + 1)) : 0;
    if (MODE == 0) {
        for (int i = 0; i < W; i++) e += energy_of(c_atoms[start + i], x, y, z);
    } else {
        __shared__ float4 sh[BLK];
        for (int t = 0; t < W; t += BLK) {
            int j = t + threadIdx.x;
            sh[threadIdx.x] = j < W ? g_atoms[start + j] : make_float4(1e9f, 1e9f, 1e9f, 0.f);
            __syncthreads();
            int m = min(BLK, W - t);
            for (int k = 0; k < m; k++) e += energy_of(sh[k], x, y, z);
            __syncthreads();
        }
    }
    out[gid] = e;
}

// ---- E2: lanes of a warp read `distinct` different addresses per iteration ----
template <int MODE>
__global__ void lane_kernel(int W, int mask, const float4 *g_atoms, float *out) {
    int gid = blockIdx.x * BLK + threadIdx.x, lane = threadIdx.x & 31;
    float x = (gid & 1023) * 0.01f, y = ((gid >> 10) & 255) * 0.01f, z = (gid >> 18) * 0.01f, e = 0.f;
    for (int i = 0; i < W; i++) {
        int idx = (i + (lane & mask)) % NA;
        float4 A = MODE == 0 ? c_atoms[idx] : g_atoms[idx];
        e += energy_of(A, x, y, z);
    }
    out[gid] = e;
}

// ---- E3: layout of the same 3 numbers per sample in constant memory: 3 arrays vs 12-byte struct vs 16-byte struct ----
struct K3 { float x, y, z; };
#define NK 1000
template <int LAYOUT>
__global__ void layout_kernel(float *out) {
    int gid = blockIdx.x * BLK + threadIdx.x;
    float x = (gid & 1023) * 0.01f, y = ((gid >> 10) & 255) * 0.01f, z = (gid >> 18) * 0.01f, acc = 0.f;
    for (int m = 0; m < NK; m++) {
        float kx, ky, kz;
        if (LAYOUT == 0) { const float *f = (const float *)c_atoms; kx = f[m]; ky = f[NK + m]; kz = f[2 * NK + m]; }      // three arrays
        else if (LAYOUT == 1) { const K3 *s = (const K3 *)c_atoms; kx = s[m].x; ky = s[m].y; kz = s[m].z; }                // 12-byte struct
        else { const float4 *s = (const float4 *)c_atoms; float4 v = s[m]; kx = v.x; ky = v.y; kz = v.z; }                 // 16-byte struct
        acc = fmaf(kx, x, fmaf(ky, y, fmaf(kz, z, acc))); acc *= 1.0000001f;
    }
    out[gid] = acc;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void csv(const char *ex, double param, const char *var, double ms, double ratio) {
    FILE *f = fopen("stats/06_constant_cache.csv", "a"); if (!f) return;
    if (ftell(f) == 0) fprintf(f, "experiment,param,variant,ms,ratio\n");
    fprintf(f, "%s,%.0f,%s,%.4f,%.3f\n", ex, param, var, ms, ratio); fclose(f);
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &p = dev_props();
    mkdir("stats", 0755); remove("stats/06_constant_cache.csv");
    printf("==================== CONSTANT-CACHE BEHAVIOUR (%s) ====================\n", p.name);
    printf("Kernels: %d blocks x %d threads; identical arithmetic per atom (no cutoff test). 'ms' = minimum of 9 launches after priming.\n", GRID, BLK);
    printf("Times below; cache hit rates for the same kernels are in output/06_ncu_summary.txt (Nsight Compute, see scripts/profile_kernels.sh).\n\n");

    std::vector<float4> h(NA); srand(3);
    for (auto &a : h) { a.x = rand() / (float)RAND_MAX * 10; a.y = rand() / (float)RAND_MAX * 10; a.z = rand() / (float)RAND_MAX * 10; a.w = 2.f * rand() / (float)RAND_MAX - 1.f; }
    float4 *g; float *o1, *o2; size_t on = (size_t)GRID * BLK;
    CUDA_CHECK(cudaMemcpyToSymbol(c_atoms, h.data(), NA * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g, NA * sizeof(float4))); CUDA_CHECK(cudaMemcpy(g, h.data(), NA * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&o1, on * 4)); CUDA_CHECK(cudaMalloc(&o2, on * 4));

    if (getenv("PROFILE_ONLY")) {                                // used by scripts/profile_kernels.sh: one launch per case, in a fixed order
        window_kernel<0><<<GRID, BLK>>>(512, 0, g, o1);  window_kernel<0><<<GRID, BLK>>>(512, 1, g, o1);   // 1,2  constant: same / different, W=512
        window_kernel<1><<<GRID, BLK>>>(512, 0, g, o1);  window_kernel<1><<<GRID, BLK>>>(512, 1, g, o1);   // 3,4  global+shared: same / different
        window_kernel<0><<<GRID, BLK>>>(64, 0, g, o1);   window_kernel<0><<<GRID, BLK>>>(64, 1, g, o1);    // 5,6  constant, small window (fits the cache)
        lane_kernel<0><<<GRID, BLK>>>(512, 0, g, o1);    lane_kernel<0><<<GRID, BLK>>>(512, 31, g, o1);     // 7,8  constant: 1 vs 32 distinct addresses per warp
        layout_kernel<0><<<GRID, BLK>>>(o1); layout_kernel<1><<<GRID, BLK>>>(o1); layout_kernel<2><<<GRID, BLK>>>(o1);   // 9,10,11 layouts
        cudaDeviceSynchronize(); printf("profile-only run done\n"); return 0;
    }
    auto timeit = [&](std::function<void()> f) {                // minimum of 9 launches (min = least disturbed by the host machine)
        double best = 1e30; cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 2; i++) { f(); } cudaDeviceSynchronize();
        for (int i = 0; i < 9; i++) { cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, (double)ms); }
        cudaEventDestroy(a); cudaEventDestroy(b); CUDA_CHECK(cudaGetLastError()); return best; };
    auto prime = [&]() { double t0 = now_ms(); while (now_ms() - t0 < 200) { window_kernel<0><<<GRID, BLK>>>(1000, 0, g, o1); cudaDeviceSynchronize(); } };

    // ------------------------------------------------------------------------------------------------------------ E1
    int rb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&rb, (const void *)window_kernel<0>, BLK, 0);
    printf("---- E1. Same window for every block vs a different window per block (%d blocks resident per SM) ----\n", rb);
    printf("[book] constant memory is efficient when all blocks read the same atoms; blocks reading different atoms overflow the cache.\n");
    printf("%-6s %-13s | %-24s | %-24s | %-22s\n", "W", "working set/SM", "constant memory (ms)", "global + shared (ms)", "different / same");
    printf("%-6s %-13s | %-11s %-12s | %-11s %-12s | %-10s %-10s\n", "atoms", "(different)", "same", "different", "same", "different", "constant", "global");
    int Ws[9] = { 16, 32, 64, 128, 256, 512, 1000, 2000, 4000 };
    bool ok = true;
    for (int W : Ws) {
        cooldown(); prime();
        double cs = timeit([&] { window_kernel<0><<<GRID, BLK>>>(W, 0, g, o1); }), cd = timeit([&] { window_kernel<0><<<GRID, BLK>>>(W, 1, g, o2); });
        double gs = timeit([&] { window_kernel<1><<<GRID, BLK>>>(W, 0, g, o1); }), gd = timeit([&] { window_kernel<1><<<GRID, BLK>>>(W, 1, g, o2); });
        // correctness: the "different" results of both storage schemes must agree
        std::vector<float> a(4096), b(4096); window_kernel<0><<<GRID, BLK>>>(W, 1, g, o1); window_kernel<1><<<GRID, BLK>>>(W, 1, g, o2); cudaDeviceSynchronize();
        cudaMemcpy(a.data(), o1, 4096 * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), o2, 4096 * 4, cudaMemcpyDeviceToHost);
        double worst = 0; for (int i = 0; i < 4096; i++) worst = std::max(worst, fabs((double)a[i] - b[i]) / (fabs((double)a[i]) + 1e-6)); ok &= worst < 1e-3;
        double ws = std::min((double)NA, (double)rb * W) * 16 / 1024.0;
        printf("%-6d %-9.1f KB  | %-11.3f %-12.3f | %-11.3f %-12.3f | %-10.2f %-10.2f\n", W, ws, cs, cd, gs, gd, cd / cs, gd / gs);
        csv("E1", W, "constant_same", cs, 1); csv("E1", W, "constant_different", cd, cd / cs); csv("E1", W, "global_shared_same", gs, 1); csv("E1", W, "global_shared_different", gd, gd / gs);
    }
    printf("  Read: 'different / same' well above 1 for constant memory and near 1 for global+shared is the book's picture (thrashing vs\n");
    printf("  immune). Working set = resident blocks x W x 16 B (capped at the 64 KB array). Results of both storage schemes agree: %s.\n\n", ok ? "PASS" : "FAIL");

    // ------------------------------------------------------------------------------------------------------------ E2
    printf("---- E2. Distinct addresses within one warp (W = 512 iterations) ----\n");
    printf("[book/hardware] constant memory broadcasts one address to the warp; different addresses in a warp are served one after the other.\n");
    printf("%-18s | %-14s %-9s | %-14s %-9s\n", "distinct / warp", "constant (ms)", "vs 1", "global (ms)", "vs 1");
    int masks[6] = { 0, 1, 3, 7, 15, 31 }; double c1 = 0, g1 = 0;
    cooldown(); prime();
    for (int m : masks) {
        double c = timeit([&] { lane_kernel<0><<<GRID, BLK>>>(512, m, g, o1); }), gg = timeit([&] { lane_kernel<1><<<GRID, BLK>>>(512, m, g, o2); });
        if (m == 0) { c1 = c; g1 = gg; }
        printf("%-18d | %-14.3f %-9.2f | %-14.3f %-9.2f\n", m + 1, c, c / c1, gg, gg / g1);
        csv("E2", m + 1, "constant", c, c / c1); csv("E2", m + 1, "global", gg, gg / g1);
    }
    printf("  Read: the constant column grows with the number of distinct addresses (serialisation); global memory serves the warp\n");
    printf("  through the normal cache path and grows much less.\n\n");

    // ------------------------------------------------------------------------------------------------------------ E3
    printf("---- E3. Layout in constant memory: %d samples of (kx, ky, kz) ----\n", NK);
    printf("[book, Ch.8] separate arrays need several cache entries per iteration; a struct keeps a sample's fields together.\n");
    printf("%-30s %-10s %-10s\n", "layout", "ms", "vs 3 arrays");
    cooldown(); prime();
    double l0 = timeit([&] { layout_kernel<0><<<GRID, BLK>>>(o1); }), l1 = timeit([&] { layout_kernel<1><<<GRID, BLK>>>(o1); }), l2 = timeit([&] { layout_kernel<2><<<GRID, BLK>>>(o1); });
    printf("%-30s %-10.3f %-10.2f\n%-30s %-10.3f %-10.2f\n%-30s %-10.3f %-10.2f\n", "3 separate arrays (SoA)", l0, 1.0, "12-byte struct {x,y,z}", l1, l1 / l0, "16-byte struct (float4)", l2, l2 / l0);
    csv("E3", 0, "soa_3_arrays", l0, 1); csv("E3", 1, "struct12", l1, l1 / l0); csv("E3", 2, "struct16", l2, l2 / l0);
    printf("  Read: a ratio below 1 means the struct layout is faster than three arrays on this GPU. The book's claim is about the\n");
    printf("  G80 constant cache; this GPU's cache behaves differently, so the size of the difference here is what it is.\n");
    printf("\n%s\n", ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return ok ? 0 : 1;
}

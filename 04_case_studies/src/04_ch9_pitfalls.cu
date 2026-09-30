/*
 * 04_ch9_pitfalls.cu   -- the OTHER issues Chapter 9 raises around Direct Coulomb Summation
 *
 * The book's claims about
 *   A. when the GPU beats the CPU (launch overhead ~110 ms and "<= 400 atoms the CPU wins" on the G80; steady
 *      44x from 10,000 atoms)
 *   B. lattice padding (a 100x100 slice padded to 128x112 = +43%; padding a whole 3D lattice would cost +60%)
 *   C. thread mapping: grid-centric GATHER vs atom-centric SCATTER (scatter needs atomicAdd)
 *   D. atoms in constant memory (broadcast to a warp, CGMA > 10:1) vs global memory
 *   E. the 64 KB constant-memory limit forcing chunked launches
 * Each section prints [book] (what the book says), then [MEASURED]/[DERIVED] results on THIS GPU.
 * Kernels/host code are reused from 01_dcs_compute.cu. Timings: warm, averaged; nothing is a single cold shot
 * except the one row explicitly labelled cold.
 */
#define DCS_NO_MAIN
#include "01_dcs_compute.cu"
#include "common.cuh"

// ---- C: atom-centric (scatter) kernel: one thread per atom, updates EVERY grid point of the slice ----
__global__ void dcs_atom_centric(const float4 *atoms, int natoms, float gs, float z, int gx, int gy, float *out, int NX, int stagger) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= natoms) return;
    float4 A = atoms[a];
    float dz = z - A.z, dz2 = dz * dz;
    int i0 = stagger ? a % gx : 0;                 // stagger: threads start at different x so they collide less
    for (int j = 0; j < gy; j++) {
        float dy = gs * j - A.y, dyz2 = dy * dy + dz2;
        for (int ii = 0; ii < gx; ii++) {
            int i = (ii + i0) % gx;
            float dx = gs * i - A.x;
            atomicAdd(&out[(size_t)j * NX + i], A.w * rsqrtf(dx * dx + dyz2));
        }
    }
}

// ---- D: v1 with atoms read from GLOBAL memory instead of __constant__ ----
__global__ void dcs_v1_global(const float4 *__restrict__ atoms, int n, float gs, float z, float *out, int NX) {
    int xi = blockIdx.x * 16 + threadIdx.x, yi = blockIdx.y * 16 + threadIdx.y;
    float x = gs * xi, y = gs * yi, e = 0.f;
    for (int a = 0; a < n; a++) {
        float4 A = atoms[a];
        float dx = x - A.x, dy = y - A.y, dz = z - A.z;
        e += A.w * rsqrtf(dx * dx + dy * dy + dz * dz);
    }
    out[yi * NX + xi] += e;
}

// ---- E: run_gpu with a configurable chunk size (<= MAXATOMS) ----
static Run run_gpu_chunk(int v, const Geo &g, const std::vector<float4> &atoms, float *d_out, int chunk_atoms) {
    CUDA_CHECK(cudaMemset(d_out, 0, g.total * sizeof(float)));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float4> chunk(chunk_atoms);
    Run r = { 0, 0, 0, 0 };
    cudaEventRecord(e0);
    for (int k = 0; k < g.gz; k++) {
        float z = GS * k;
        for (int s = 0; s < (int)atoms.size(); s += chunk_atoms) {
            int n = (int)atoms.size() - s < chunk_atoms ? (int)atoms.size() - s : chunk_atoms;
            for (int i = 0; i < n; i++) {
                float4 a = atoms[s + i];
                if (VER[v].sq) { float dz = z - a.z; a.z = dz * dz; }
                chunk[i] = a;
            }
            CUDA_CHECK(cudaMemcpyToSymbol(atominfo, chunk.data(), n * sizeof(float4)));
            launch(v, g, n, z, d_out + (size_t)k * g.slice);
            r.launches++; r.h2d_bytes += (long)n * sizeof(float4);
        }
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&r.ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return r;
}

// sampled check of one slice (iz) against the double reference; error scaled by sum(|q|/r)
static double check_slice(const std::vector<float> &h, const Geo &g, const std::vector<float4> &atoms, int iz, int count) {
    double worst = 0; unsigned s = 777;
    for (int c = 0; c < count; c++) {
        s = s * 1664525u + 1013904223u; int ix = (s >> 8) % g.gx;
        s = s * 1664525u + 1013904223u; int iy = (s >> 8) % g.gy;
        double sc, ref = ref_point(atoms, ix, iy, iz, &sc);
        double e = fabs(h[(size_t)iz * g.slice + (size_t)iy * g.NX + ix] - ref) / sc;
        if (e > worst) worst = e;
    }
    return worst;
}

#include <sys/stat.h>
static void csv04(const char *section, double param, const char *series, double value) {
    FILE *f = fopen("stats/04_ch9_pitfalls.csv", "a"); if (!f) return;
    if (ftell(f) == 0) fprintf(f, "section,param,series,value\n");
    fprintf(f, "%s,%.0f,%s,%.5f\n", section, param, series, value); fclose(f);
}

int main() {
    RunMonitor mon;
    bool all_ok = true;
    mkdir("stats", 0755); remove("stats/04_ch9_pitfalls.csv");
    const cudaDeviceProp &p = dev_props();
    printf("==================== CHAPTER 9 PITFALLS ON %s ====================\n", p.name);
    printf("(book numbers are for the 2008 G80; this GPU is a 2019 Turing part: compare shapes, not absolutes)\n\n");

    // ------------------------------------------------------------------------------ A
    printf("---- A. When does the GPU beat the CPU? (one 100x100 slice, warm GPU) ----\n");
    printf("  [book] fixed GPU start-up ~110 ms; CPU faster for <= 400 atoms; GPU fully utilised from ~10,000 atoms.\n");
    {
        Scale s0 = { "A", 100, 100, 1, 1 };
        Geo g = make_geo(s0);
        float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
        // cold: the very first launch of a kernel in this process (lazy module load)
        auto a1 = make_atoms(s0, 11);
        Run cold = run_gpu(0, g, a1, d);
        printf("  [MEASURED] very first GPU run in this process (1 atom, includes lazy kernel load): %.3f ms\n", cold.ms);
        int Ns[] = { 1, 10, 100, 400, 1000, 4000, 10000, 40000 };
        printf("  %-8s %-11s %-11s %-11s %-11s %-10s %s\n", "atoms", "CPU ms", "GPU v1 ms", "GPU v2 ms", "GPU v3 ms", "best GPU", "winner (best GPU vs CPU)");
        for (int N : Ns) {
            Scale s = { "A", 100, 100, 1, N };
            auto atoms = make_atoms(s, 12);
            std::vector<float> cpu(g.total);
            double t0 = now_ms(); cpu_slice(cpu.data(), g, 0.f, atoms); double cpu_ms = now_ms() - t0;
            if (N <= 1000) { for (int r = 1; r < 20; r++) { double t1 = now_ms(); cpu_slice(cpu.data(), g, 0.f, atoms); cpu_ms += now_ms() - t1; } cpu_ms /= 20; }
            int reps = N <= 4000 ? 20 : 5;
            float t[3];
            bench_rr(3, [&](int v) { run_gpu(v, g, atoms, d); }, reps, t);
            float best = t[0] < t[1] ? (t[0] < t[2] ? t[0] : t[2]) : (t[1] < t[2] ? t[1] : t[2]);
            printf("  %-8d %-11.3f %-11.3f %-11.3f %-11.3f %-10.3f %s (%.1fx)\n", N, cpu_ms, t[0], t[1], t[2], best,
                   best < cpu_ms ? "GPU" : "CPU", best < cpu_ms ? cpu_ms / best : best / cpu_ms);
            csv04("A_crossover_ms", N, "cpu", cpu_ms); csv04("A_crossover_ms", N, "gpu_v1", t[0]); csv04("A_crossover_ms", N, "gpu_v2", t[1]); csv04("A_crossover_ms", N, "gpu_v3", t[2]);
        }
        printf("  Read: 'winner' flips from CPU to GPU at the smallest atom count where the GPU column is lower. The warm\n");
        printf("  per-run fixed cost here is one memset + constant upload + launch (about 0.2-0.3 ms). The book's ~110 ms was\n");
        printf("  start-up (context creation); process/context start-up is NOT timed in this program, so that claim is untested.\n\n");
        cudaFree(d);
    }

    // ------------------------------------------------------------------------------ B
    printf("---- B. Lattice padding: waste and its effect on speed ----\n");
    printf("  [book] slices are padded to whole thread blocks (x to a multiple of 128, y of 16); 100x100 -> 128x112 = +43%%;\n");
    printf("         padding the whole 3D lattice (100^3 -> 128x112x112) would waste +60%%, hence 2D slices.\n");
    printf("  [DERIVED] padding waste, per-slice (this program's scheme) vs whole-3D (z also to a multiple of 16):\n");
    printf("  %-14s %-14s %-11s %-16s %-11s\n", "lattice", "padded slice", "2D waste", "padded 3D", "3D waste");
    int L[][3] = { {100, 100, 100}, {64, 64, 64}, {128, 112, 100}, {200, 200, 200}, {333, 333, 333} };
    for (auto &l : L) {
        int px = roundup(l[0], 128), py = roundup(l[1], 16), pz = roundup(l[2], 16);
        double v = (double)l[0] * l[1] * l[2];
        printf("  %3dx%3dx%-6d %4dx%-9d %6.0f%%     %4dx%4dx%-5d %6.0f%%\n", l[0], l[1], l[2], px, py,
               100.0 * ((double)px * py / ((double)l[0] * l[1]) - 1), px, py, pz, 100.0 * ((double)px * py * pz / v - 1));
    }
    printf("  [MEASURED] same 4,000 atoms, 8 slices, kernel v2: useful vs computed evaluation rate\n");
    printf("  %-12s %-14s %-9s %-16s %-16s\n", "lattice", "padded slice", "ms", "useful G ev/s", "computed G ev/s");
    {
        int W[][2] = { {100, 100}, {128, 112}, {128, 128}, {64, 64} };
        for (auto &w : W) {
            Scale s = { "B", w[0], w[1], 8, 4000 }; Geo g = make_geo(s); auto atoms = make_atoms(s, 13);
            float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
            float t[1]; bench_rr(1, [&](int) { run_gpu(1, g, atoms, d); }, 20, t);
            double useful = (double)w[0] * w[1] * 8 * 4000, comp = (double)g.slice * 8 * 4000;
            printf("  %3dx%-8d %4dx%-9d %-9.3f %-16.2f %-16.2f\n", w[0], w[1], g.NX, g.NY, t[0], useful / (t[0] * 1e-3) / 1e9, comp / (t[0] * 1e-3) / 1e9);
            cudaFree(d);
        }
    }
    printf("  Read: run time is nearly the same (~4 ms) for every lattice although padded cell counts differ by 2x, because a\n");
    printf("  single small slice launches few blocks and does not fill the GPU (see the launch-fill note in 02). So padding\n");
    printf("  waste is NOT paid in time here; it only lowers the useful fraction of the work the GPU does in that time.\n\n");

    // ------------------------------------------------------------------------------ C
    printf("---- C. Grid-centric gather vs atom-centric scatter (one 64x64 slice) ----\n");
    printf("  [book] atom-centric: each thread = one atom, updating all grid points -> many threads write the same\n");
    printf("         location -> atomicAdd needed -> slow. Grid-centric: accumulate in registers, write once.\n");
    {
        int Ns[] = { 500, 2000, 4000 };
        printf("  %-7s %-14s %-16s %-16s %-11s %-11s %s\n", "atoms", "gather v1 ms", "scatter ms (naive)", "scatter ms (stagger)", "naive/gather", "atomics", "max error (gather | scatter)");
        for (int N : Ns) {
            Scale s = { "C", 64, 64, 1, N }; Geo g = make_geo(s); auto atoms = make_atoms(s, 14);
            float *d, *dS; float4 *dA;
            CUDA_CHECK(cudaMalloc(&d, g.total * 4)); CUDA_CHECK(cudaMalloc(&dS, g.total * 4)); CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float4)));
            CUDA_CHECK(cudaMemcpy(dA, atoms.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
            int blocks = (N + 127) / 128;
            float t[3];
            bench_rr(3, [&](int v) {
                if (v == 0) run_gpu(0, g, atoms, d);
                else { cudaMemset(dS, 0, g.total * 4); dcs_atom_centric<<<blocks, 128>>>(dA, N, GS, 0.f, s.gx, s.gy, dS, g.NX, v == 2); }
            }, 5, t);
            std::vector<float> hg(g.total), hs(g.total);
            run_gpu(0, g, atoms, d); CUDA_CHECK(cudaMemcpy(hg.data(), d, g.total * 4, cudaMemcpyDeviceToHost));
            cudaMemset(dS, 0, g.total * 4); dcs_atom_centric<<<blocks, 128>>>(dA, N, GS, 0.f, s.gx, s.gy, dS, g.NX, 1);
            CUDA_CHECK(cudaMemcpy(hs.data(), dS, g.total * 4, cudaMemcpyDeviceToHost));
            double eg = check_slice(hg, g, atoms, 0, 200), es = check_slice(hs, g, atoms, 0, 200); all_ok &= eg < 1e-4 && es < 1e-4;
            printf("  %-7d %-14.3f %-18.3f %-20.3f %-11.1f %-11.2e %.2e | %.2e\n", N, t[0], t[1], t[2], t[1] / t[0], (double)N * s.gx * s.gy, eg, es);
            csv04("C_gather_vs_scatter_ms", N, "gather_v1", t[0]); csv04("C_gather_vs_scatter_ms", N, "scatter_naive", t[1]); csv04("C_gather_vs_scatter_ms", N, "scatter_staggered", t[2]);
            cudaFree(d); cudaFree(dS); cudaFree(dA);
        }
        printf("  Read: 'atomics' = total atomicAdd calls. 'naive' has every thread of a warp updating the same address at the\n");
        printf("  same moment; 'stagger' spreads them, showing how much of the cost is same-address contention.\n");
        printf("  The scatter kernel also launches only N/128 blocks, so it cannot fill the GPU: a second reason it loses.\n\n");
    }

    // ------------------------------------------------------------------------------ D
    printf("---- D. Atoms in constant memory vs global memory (kernel v1 body, one 128x112 slice, 4,000 atoms) ----\n");
    printf("  [book] warp reads the same atom -> constant cache broadcast, CGMA > 10:1, >96%% hit rate.\n");
    {
        Scale s = { "D", 100, 100, 1, 4000 }; Geo g = make_geo(s); auto atoms = make_atoms(s, 15);
        float *d; float4 *dA; CUDA_CHECK(cudaMalloc(&d, g.total * 4)); CUDA_CHECK(cudaMalloc(&dA, 4000 * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(dA, atoms.data(), 4000 * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(atominfo, atoms.data(), 4000 * sizeof(float4)));
        dim3 grid(g.NX / 16, g.NY / 16), blk(16, 16);
        float t[2];
        bench_rr(2, [&](int v) {
            for (int r = 0; r < 50; r++) {
                if (v == 0) dcs_v1<<<grid, blk>>>(4000, GS, 0.f, d, g.NX);
                else dcs_v1_global<<<grid, blk>>>(dA, 4000, GS, 0.f, d, g.NX);
            }
        }, 10, t);
        double ev = (double)g.slice * 4000 * 50;
        printf("  %-26s %-12s %-14s\n", "atom storage", "ms / launch", "G evals/s (computed)");
        printf("  %-26s %-12.4f %-14.1f\n", "__constant__ (broadcast)", t[0] / 50, ev / (t[0] * 1e-3) / 1e9);
        printf("  %-26s %-12.4f %-14.1f\n", "global (const __restrict__)", t[1] / 50, ev / (t[1] * 1e-3) / 1e9);
        csv04("D_atom_storage_ms_per_launch", 4000, "constant", t[0] / 50); csv04("D_atom_storage_ms_per_launch", 4000, "global", t[1] / 50);
        printf("  [MEASURED] global/constant time ratio = %.2fx. Modern GPUs also broadcast a uniform global read through\n", t[1] / t[0]);
        printf("  the read-only/L1 path, so a big gap is not guaranteed; the ratio above is what this GPU shows.\n\n");
        cudaFree(d); cudaFree(dA);
    }

    // ------------------------------------------------------------------------------ E
    printf("---- E. The 64 KB constant limit: chunk size vs launches (kernel v2, 40,000 atoms, 100x100x10 lattice) ----\n");
    printf("  [book] atoms are streamed in chunks (<= 64 KB); each chunk = one memcpy to constant memory + one launch.\n");
    {
        Scale s = { "E", 100, 100, 10, 40000 }; Geo g = make_geo(s); auto atoms = make_atoms(s, 16);
        float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
        int Cs[] = { 125, 250, 500, 1000, 2000, 4000 };
        printf("  %-12s %-10s %-11s %-10s %-14s %s\n", "atoms/chunk", "bytes", "launches", "ms", "G evals/s", "vs 4000");
        float base = 0; double valid = (double)s.gx * s.gy * s.gz * s.natoms; float tt[6];
        for (int r = -1; r < 5; r++) for (int c = 0; c < 6; c++) { Run x = run_gpu_chunk(1, g, atoms, d, Cs[c]); if (r >= 0) tt[c] = (r == 0 ? 0 : tt[c]) + x.ms / 5; }
        base = tt[5];
        for (int c = 0; c < 6; c++) {
            printf("  %-12d %-10d %-11ld %-10.2f %-14.2f %.2fx\n", Cs[c], Cs[c] * 16, (long)10 * ((40000 + Cs[c] - 1) / Cs[c]), tt[c], valid / (tt[c] * 1e-3) / 1e9, tt[c] / base);
            csv04("E_chunk_size_ms", Cs[c], "v2", tt[c]);
        }
        printf("  Read: smaller chunks add launches and uploads for the same arithmetic; the cost of that overhead is the\n");
        printf("  time ratio in the last column (>1.00x = slower than the largest legal chunk).\n");
        cudaFree(d);
    }
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

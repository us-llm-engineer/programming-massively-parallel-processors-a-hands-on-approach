/*
 * 01_dcs_compute.cu   -- THE COMPUTATION: Direct Coulomb Summation (DCS)
 *
 * Source: PMPP (Kirk & Hwu) Chapter 9 "Application Case Study: Molecular Visualization and Analysis".
 * (Chapter 8 in this book is MRI reconstruction; the electric-charge problem is Chapter 9.)
 *
 * PROBLEM
 *   A molecule is a list of atoms, each with a position (x,y,z) and an electric charge q.
 *   We want the electrostatic potential at every point of a regular 3D lattice:
 *
 *        potential(j) = SUM over all atoms i of   q_i / r_ij ,   r_ij = |lattice_j - atom_i|
 *
 *   Work = (#lattice points) x (#atoms). Every grid point is independent of every other one, so it
 *   is a natural GPU problem ("one of the best matches for GPU computing", book).
 *
 * THREE GPU KERNELS, in the order the book builds them
 *   v1 CUDA-Simple   1 grid point per thread; atom data in __constant__ memory (broadcast to a warp)
 *   v2 CUDA-Unroll4x 4 adjacent grid points per thread; the (dy^2+dz^2) part and the atom's x,q are
 *                    loaded/computed once and reused for the 4 points ("thread granularity")
 *   v3 CUDA-Unroll8clx 8 points per thread, spaced 16 apart so a half-warp writes consecutive
 *                    addresses (coalesced output)
 *   Common to all: rsqrtf() (one SFU instruction), 2D z-slices, lattice padded to whole thread
 *   blocks, and atoms processed in CHUNKS because constant memory is only 64 KB.
 *
 * SCALES (reduced to what this GTX 1650 Ti can do, then the full scale):
 *   TINY    32x32x4  lattice,   1,000 atoms  - every point checked against a double-precision CPU
 *   REDUCED 100x100x10 lattice,  4,000 atoms  - also timed against the plain sequential CPU code
 *   FULL    100x100x100 lattice, 100,000 atoms - the book's lattice; GPU only, verified by sampling
 *
 * Output convention: all numbers below are measured by this program; nothing is quoted from the book
 * except where labelled "book".
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "monitor.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define MAXATOMS 4000            // 4000 x 16 B = 64,000 B <= 64 KB constant memory (book uses 4000)
#define GS 1.0f                  // lattice spacing (Angstrom)
#define FLOPS_PER_EVAL 10.0      // convention implied by the book's own 186 GFLOPS / 18.6 G evals/s

__constant__ float4 atominfo[MAXATOMS];

// ------------------------------------------------------------------ kernels
// v1: one point per thread. atominfo = (x, y, z, q); dz computed here.
__global__ void dcs_v1(int n, float gs, float z, float *out, int NX) {
    int xi = blockIdx.x * 16 + threadIdx.x, yi = blockIdx.y * 16 + threadIdx.y;
    float x = gs * xi, y = gs * yi, e = 0.f;
    for (int a = 0; a < n; a++) {
        float4 A = atominfo[a];
        float dx = x - A.x, dy = y - A.y, dz = z - A.z;
        e += A.w * rsqrtf(dx * dx + dy * dy + dz * dz);
    }
    out[yi * NX + xi] += e;
}

// v2: four ADJACENT points per thread. atominfo = (x, y, dz^2, q): host pre-squares dz per slice.
__global__ void dcs_v2(int n, float gs, float z, float *out, int NX) {
    int xi = (blockIdx.x * 16 + threadIdx.x) * 4, yi = blockIdx.y * 16 + threadIdx.y;
    float x = gs * xi, y = gs * yi;
    float e1 = 0.f, e2 = 0.f, e3 = 0.f, e4 = 0.f;
    for (int a = 0; a < n; a++) {
        float4 A = atominfo[a];
        float dy = y - A.y, dyz2 = dy * dy + A.z;             // shared by all 4 points
        float dx1 = x - A.x, dx2 = dx1 + gs, dx3 = dx2 + gs, dx4 = dx3 + gs;
        e1 += A.w * rsqrtf(dx1 * dx1 + dyz2);
        e2 += A.w * rsqrtf(dx2 * dx2 + dyz2);
        e3 += A.w * rsqrtf(dx3 * dx3 + dyz2);
        e4 += A.w * rsqrtf(dx4 * dx4 + dyz2);
    }
    float *o = out + yi * NX + xi;                            // 4 consecutive floats per thread:
    o[0] += e1; o[1] += e2; o[2] += e3; o[3] += e4;           // neighbours write 16 B apart (uncoalesced)
}

// v3: eight points per thread, 16 apart -> consecutive threads write consecutive floats.
__global__ void dcs_v3(int n, float gs, float z, float *out, int NX) {
    int xi = blockIdx.x * 128 + threadIdx.x, yi = blockIdx.y * 16 + threadIdx.y;
    float x = gs * xi, y = gs * yi, g16 = gs * 16.f;
    float e1 = 0.f, e2 = 0.f, e3 = 0.f, e4 = 0.f, e5 = 0.f, e6 = 0.f, e7 = 0.f, e8 = 0.f;
    for (int a = 0; a < n; a++) {
        float4 A = atominfo[a];
        float dy = y - A.y, dyz2 = dy * dy + A.z;
        float dx1 = x - A.x, dx2 = dx1 + g16, dx3 = dx2 + g16, dx4 = dx3 + g16,
              dx5 = dx4 + g16, dx6 = dx5 + g16, dx7 = dx6 + g16, dx8 = dx7 + g16;
        e1 += A.w * rsqrtf(dx1 * dx1 + dyz2); e2 += A.w * rsqrtf(dx2 * dx2 + dyz2);
        e3 += A.w * rsqrtf(dx3 * dx3 + dyz2); e4 += A.w * rsqrtf(dx4 * dx4 + dyz2);
        e5 += A.w * rsqrtf(dx5 * dx5 + dyz2); e6 += A.w * rsqrtf(dx6 * dx6 + dyz2);
        e7 += A.w * rsqrtf(dx7 * dx7 + dyz2); e8 += A.w * rsqrtf(dx8 * dx8 + dyz2);
    }
    float *o = out + yi * NX + xi;
    o[0] += e1; o[16] += e2; o[32] += e3; o[48] += e4; o[64] += e5; o[80] += e6; o[96] += e7; o[112] += e8;
}

// ------------------------------------------------------------------ host side
struct Scale { const char *name; int gx, gy, gz, natoms; };
struct Geo   { int NX, NY, gx, gy, gz; size_t slice, total; };   // padded dimensions
struct Run   { float ms; long launches, h2d_bytes; double evals_computed; };
struct Ver   { const char *name; int ppt; bool sq; dim3 block; };  // points per thread, dz^2 mode

static const Ver VER[3] = {
    { "v1 CUDA-Simple      (1 pt/thread) ", 1, false, dim3(16, 16) },
    { "v2 CUDA-Unroll4x    (4 pt/thread) ", 4, true,  dim3(16, 16) },
    { "v3 CUDA-Unroll8clx  (8 pt/thread) ", 8, true,  dim3(16, 16) } };

static int roundup(int v, int m) { return (v + m - 1) / m * m; }
static Geo make_geo(const Scale &s) {
    Geo g; g.gx = s.gx; g.gy = s.gy; g.gz = s.gz;
    g.NX = roundup(s.gx, 128);          // multiple of 8 half-warps (v3 needs 128)
    g.NY = roundup(s.gy, 16);
    g.slice = (size_t)g.NX * g.NY; g.total = g.slice * s.gz;
    return g;
}

// atoms: x,y,z,q per atom, uniformly random in the lattice box (same density family as a solvated molecule)
static std::vector<float4> make_atoms(const Scale &s, unsigned seed) {
    std::vector<float4> a(s.natoms);
    srand(seed);
    auto u = []() { return (float)rand() / RAND_MAX; };
    for (auto &p : a) { p.x = u() * s.gx * GS; p.y = u() * s.gy * GS; p.z = u() * s.gz * GS; p.w = 2.f * u() - 1.f; }
    return a;
}

static void launch(int v, const Geo &g, int n, float z, float *d_slice) {
    dim3 grid(g.NX / (16 * VER[v].ppt), g.NY / 16), blk = VER[v].block;
    if (v == 0) dcs_v1<<<grid, blk>>>(n, GS, z, d_slice, g.NX);
    if (v == 1) dcs_v2<<<grid, blk>>>(n, GS, z, d_slice, g.NX);
    if (v == 2) dcs_v3<<<grid, blk>>>(n, GS, z, d_slice, g.NX);
}

// Runs one version over every z-slice and every 4000-atom chunk. d_out (g.total floats) is zeroed first.
static Run run_gpu(int v, const Geo &g, const std::vector<float4> &atoms, float *d_out) {
    CUDA_CHECK(cudaMemset(d_out, 0, g.total * sizeof(float)));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float4> chunk(MAXATOMS);
    Run r = { 0, 0, 0, 0 };
    cudaEventRecord(e0);
    for (int k = 0; k < g.gz; k++) {
        float z = GS * k;
        for (int s = 0; s < (int)atoms.size(); s += MAXATOMS) {
            int n = (int)atoms.size() - s < MAXATOMS ? (int)atoms.size() - s : MAXATOMS;
            for (int i = 0; i < n; i++) {
                float4 a = atoms[s + i];
                if (VER[v].sq) { float dz = z - a.z; a.z = dz * dz; }      // v2/v3: host pre-squares dz
                chunk[i] = a;
            }
            CUDA_CHECK(cudaMemcpyToSymbol(atominfo, chunk.data(), n * sizeof(float4)));
            launch(v, g, n, z, d_out + (size_t)k * g.slice);
            r.launches++; r.h2d_bytes += (long)n * sizeof(float4);
            r.evals_computed += (double)g.slice * n;
        }
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&r.ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return r;
}

// Fig 9.3 style sequential CPU code (float, sqrtf + divide), one z-slice.
static void cpu_slice(float *out, const Geo &g, float z, const std::vector<float4> &atoms) {
    for (int j = 0; j < g.gy; j++) {
        float y = GS * j;
        for (int i = 0; i < g.gx; i++) {
            float x = GS * i, e = 0.f;
            for (auto &a : atoms) {
                float dx = x - a.x, dy = y - a.y, dz = z - a.z;
                e += a.w / sqrtf(dx * dx + dy * dy + dz * dz);
            }
            out[(size_t)j * g.NX + i] = e;
        }
    }
}

// Double-precision reference at one lattice point; also returns sum(|q|/r) as the natural error scale
// (positive and negative charges cancel, so error relative to the potential itself would be misleading).
static double ref_point(const std::vector<float4> &atoms, int ix, int iy, int iz, double *scale) {
    double e = 0, sc = 0, x = (double)GS * ix, y = (double)GS * iy, z = (double)GS * iz;
    for (auto &a : atoms) {
        double dx = x - a.x, dy = y - a.y, dz = z - a.z, r = sqrt(dx * dx + dy * dy + dz * dz);
        e += a.w / r; sc += fabs((double)a.w) / r;
    }
    *scale = sc; return e;
}

struct Err { double max, mean; int n; };
// Compare GPU output with the double reference at `count` pseudo-random valid lattice points.
static Err verify_samples(const std::vector<float> &h, const Geo &g, const std::vector<float4> &atoms, int count) {
    Err e = { 0, 0, count };
    unsigned s = 12345;
    for (int c = 0; c < count; c++) {
        s = s * 1664525u + 1013904223u; int ix = (s >> 8) % g.gx;
        s = s * 1664525u + 1013904223u; int iy = (s >> 8) % g.gy;
        s = s * 1664525u + 1013904223u; int iz = (s >> 8) % g.gz;
        double sc, ref = ref_point(atoms, ix, iy, iz, &sc);
        double err = fabs(h[(size_t)iz * g.slice + (size_t)iy * g.NX + ix] - ref) / sc;
        if (err > e.max) e.max = err;
        e.mean += err / count;
    }
    return e;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void print_scale_row(const Scale &s) {
    Geo g = make_geo(s);
    double valid = (double)s.gx * s.gy * s.gz, comp = (double)g.total;
    printf("  %-8s %3dx%3dx%3d   padded slice %3dx%3d (+%4.0f%%)   %7d atoms   %d chunk(s)/slice   %.3e valid evals\n",
           s.name, s.gx, s.gy, s.gz, g.NX, g.NY, 100.0 * (comp / valid - 1), s.natoms,
           (s.natoms + MAXATOMS - 1) / MAXATOMS, valid * s.natoms);
}

#include <sys/stat.h>
static void csv01(const char *scale, const char *version, double ms, double geval, double gflops, long launches, double err, bool ok) {
    FILE *f = fopen("stats/01_dcs.csv", "a"); if (!f) return;
    if (ftell(f) == 0) fprintf(f, "scale,version,ms,g_evals_per_s,gflops_10_per_eval,launches,max_err,check\n");
    fprintf(f, "%s,\"%s\",%.4f,%.4f,%.2f,%ld,%.3e,%s\n", scale, version, ms, geval, gflops, launches, err, ok ? "PASS" : "FAIL");
    fclose(f);
}

#ifndef DCS_NO_MAIN
int main() {
    RunMonitor mon;                                                  // prints the GPU duty cycle of this run at the end
    bool all_ok = true;
    mkdir("stats", 0755); remove("stats/01_dcs.csv");
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("==================== DIRECT COULOMB SUMMATION: RESULTS ====================\n");
    printf("GPU: %s | book source: PMPP Chapter 9 (in this edition Chapter 8 is MRI)\n\n", p.name);

    printf("---- The problem ----\n");
    printf("  potential(j) = sum_i q_i / sqrt((xj-xi)^2 + (yj-yi)^2 + (zj-zi)^2)\n");
    printf("  lattice spacing %.1f A, atoms uniform in the lattice box, charges uniform in [-1,+1].\n", GS);
    printf("  Padded-lattice cells are computed but not counted as useful work.\n");
    printf("  Error metric: |GPU - double CPU| / sum(|q|/r)  (charges cancel, so plain relative error misleads).\n\n");

    Scale tiny = { "TINY", 32, 32, 4, 1000 }, red = { "REDUCED", 100, 100, 10, 4000 }, full = { "FULL", 100, 100, 100, 100000 };
    printf("---- Scale ladder ----\n");
    print_scale_row(tiny); print_scale_row(red); print_scale_row(full);
    printf("\n");

    // ------------------------------------------------------------ TINY: every point vs double CPU
    {
        Geo g = make_geo(tiny); auto atoms = make_atoms(tiny, 1);
        float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
        printf("---- TINY: every lattice point vs double-precision CPU ----\n");
        std::vector<float> h(g.total);
        // full reference (double) once
        std::vector<double> ref((size_t)tiny.gx * tiny.gy * tiny.gz), sc(ref.size());
        for (int k = 0, i = 0; k < tiny.gz; k++) for (int j = 0; j < tiny.gy; j++) for (int x = 0; x < tiny.gx; x++, i++)
            ref[i] = ref_point(atoms, x, j, k, &sc[i]);
        for (int v = 0; v < 3; v++) {
            run_gpu(v, g, atoms, d);
            CUDA_CHECK(cudaMemcpy(h.data(), d, g.total * 4, cudaMemcpyDeviceToHost));
            double mx = 0; int i = 0;
            for (int k = 0; k < tiny.gz; k++) for (int j = 0; j < tiny.gy; j++) for (int x = 0; x < tiny.gx; x++, i++) {
                double e = fabs(h[(size_t)k * g.slice + (size_t)j * g.NX + x] - ref[i]) / sc[i];
                if (e > mx) mx = e;
            }
            printf("  %s all %d points: max error %.2e  %s\n", VER[v].name, (int)ref.size(), mx, mx < 1e-4 ? "PASS" : "FAIL");
            all_ok &= mx < 1e-4;
        }
        cudaFree(d); printf("\n");
    }

    // ------------------------------------------------------------ REDUCED: timed vs sequential CPU
    double cpu_rate = 0;
    {
        Geo g = make_geo(red); auto atoms = make_atoms(red, 2);
        float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
        printf("---- REDUCED: %dx%dx%d lattice, %d atoms ----\n", red.gx, red.gy, red.gz, red.natoms);
        std::vector<float> cpu(g.total, 0.f), h(g.total);
        double t0 = now_ms();
        for (int k = 0; k < red.gz; k++) cpu_slice(cpu.data() + (size_t)k * g.slice, g, GS * k, atoms);
        double cpu_ms = now_ms() - t0, valid = (double)red.gx * red.gy * red.gz * red.natoms;
        cpu_rate = valid / (cpu_ms * 1e-3);
        printf("  sequential CPU (Fig 9.3 style, 1 core): %9.1f ms   %7.3f G evals/s\n", cpu_ms, cpu_rate / 1e9);
        for (int v = 0; v < 3; v++) run_gpu(v, g, atoms, d);          // warm-up
        printf("  %-38s %9s %12s %10s %12s %12s\n", "version", "ms", "G evals/s", "GFLOPS", "vs CPU", "max err (200 pts)");
        for (int v = 0; v < 3; v++) {
            Run r = run_gpu(v, g, atoms, d);
            CUDA_CHECK(cudaMemcpy(h.data(), d, g.total * 4, cudaMemcpyDeviceToHost));
            Err e = verify_samples(h, g, atoms, 200);
            double rate = valid / (r.ms * 1e-3);
            // also compare with the float CPU output on the valid region
            double cm = 0; for (int k = 0; k < red.gz; k += 3) for (int j = 0; j < red.gy; j++) for (int x = 0; x < red.gx; x++) {
                size_t idx = (size_t)k * g.slice + (size_t)j * g.NX + x;
                double dd = fabs(h[idx] - cpu[idx]); if (dd > cm) cm = dd; }
            printf("  %-38s %9.2f %12.3f %10.1f %11.1fx %12.2e\n", VER[v].name, r.ms, rate / 1e9,
                   rate * FLOPS_PER_EVAL / 1e9, rate / cpu_rate, e.max);
            all_ok &= e.max < 1e-4; csv01("reduced", VER[v].name, r.ms, rate / 1e9, rate * FLOPS_PER_EVAL / 1e9, r.launches, e.max, e.max < 1e-4);
            if (v == 2) printf("  (v3 vs float-CPU grid: max abs diff %.3e over every third slice; both are float, so small differences are expected)\n", cm);
        }
        cudaFree(d); printf("\n");
    }

    // ------------------------------------------------------------ FULL scale
    {
        Geo g = make_geo(full); auto atoms = make_atoms(full, 3);
        float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
        double valid = (double)full.gx * full.gy * full.gz * full.natoms;
        printf("---- FULL: %dx%dx%d lattice, %d atoms (%.2e valid atom-point evaluations) ----\n",
               full.gx, full.gy, full.gz, full.natoms, valid);
        printf("  %-38s %9s %12s %10s %12s %10s %10s\n", "version", "seconds", "G evals/s", "GFLOPS", "launches", "H2D MB", "max err");
        std::vector<float> h(g.total), best;
        double best_rate = 0;
        for (int v = 0; v < 3; v++) {
            Run r = run_gpu(v, g, atoms, d);
            CUDA_CHECK(cudaMemcpy(h.data(), d, g.total * 4, cudaMemcpyDeviceToHost));
            Err e = verify_samples(h, g, atoms, 200);
            double rate = valid / (r.ms * 1e-3);
            printf("  %-38s %9.3f %12.3f %10.1f %12ld %10.1f %10.2e %s\n", VER[v].name, r.ms / 1e3, rate / 1e9,
                   rate * FLOPS_PER_EVAL / 1e9, r.launches, r.h2d_bytes / 1e6, e.max, e.max < 1e-4 ? "PASS" : "FAIL");
            all_ok &= e.max < 1e-4; csv01("full", VER[v].name, r.ms, rate / 1e9, rate * FLOPS_PER_EVAL / 1e9, r.launches, e.max, e.max < 1e-4);
            if (v == 2) { best = h; best_rate = rate; }
        }
        printf("  book (G80, 8800 GTX, for scale): v1 18.6, v2 33.4, v3 39.5 G evals/s\n");
        printf("  Sequential CPU at the measured %.3f G evals/s would need about %.0f s (%.1f min) for this size: not run.\n\n",
               cpu_rate / 1e9, valid / cpu_rate, valid / cpu_rate / 60);
        printf("  v3 vs CPU rate: %.0fx (book: 44x on G80 vs its CPU)\n\n", best_rate / cpu_rate);

        // ---- the physics result: the potential map
        double mn = 1e30, mx = -1e30, sum = 0; long cnt = 0; int pos = 0;
        for (int k = 0; k < full.gz; k++) for (int j = 0; j < full.gy; j++) for (int x = 0; x < full.gx; x++) {
            float v = best[(size_t)k * g.slice + (size_t)j * g.NX + x];
            if (v < mn) mn = v; if (v > mx) mx = v; sum += v; cnt++; pos += v > 0;
        }
        printf("---- Potential map (FULL, from v3) ----\n");
        printf("  points: %ld  min %.3f  max %.3f  mean %.4f  positive %.1f%%  (units: charge/Angstrom)\n",
               cnt, mn, mx, sum / cnt, 100.0 * pos / cnt);
        printf("  sample points (ix,iy,iz -> GPU value | double CPU reference):\n");
        int pts[5][3] = { {0, 0, 0}, {50, 50, 50}, {99, 99, 99}, {25, 75, 10}, {80, 20, 90} };
        for (auto &q : pts) {
            double sc, ref = ref_point(atoms, q[0], q[1], q[2], &sc);
            printf("    (%3d,%3d,%3d)  %10.4f | %10.4f\n", q[0], q[1], q[2],
                   best[(size_t)q[2] * g.slice + (size_t)q[1] * g.NX + q[0]], ref);
        }
        // ASCII map of slice z = 50: 64 columns x 25 rows, 9-level ramp from slice min to slice max
        int zs = 50; float smn = 1e30f, smx = -1e30f;
        for (int j = 0; j < full.gy; j++) for (int x = 0; x < full.gx; x++) {
            float v = best[(size_t)zs * g.slice + (size_t)j * g.NX + x];
            if (v < smn) smn = v; if (v > smx) smx = v; }
        const char *ramp = " .:-=+*#%@";
        printf("\n  ASCII map of slice z=%d (each cell averages 1.6 x 4 lattice points; ' ' = most negative, '@' = most positive;\n", zs);
        printf("  slice range %.2f .. %.2f; x to the right, y downward):\n", smn, smx);
        for (int r = 0; r < 25; r++) {
            printf("  |");
            for (int c = 0; c < 64; c++) {
                double a = 0; int m = 0;
                for (int dy = 0; dy < 4; dy++) for (int dx = 0; dx < 2; dx++) {
                    int yy = r * 4 + dy, xx = (int)(c * 1.5625) + dx;
                    if (yy < full.gy && xx < full.gx) { a += best[(size_t)zs * g.slice + (size_t)yy * g.NX + xx]; m++; }
                }
                double t = (a / m - smn) / (smx - smn + 1e-9);
                printf("%c", ramp[(int)(t * 9.999)]);
            }
            printf("|\n");
        }
        cudaFree(d);
    }
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}
#endif

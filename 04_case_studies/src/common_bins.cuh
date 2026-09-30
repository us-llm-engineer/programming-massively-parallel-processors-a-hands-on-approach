// common_bins.cuh - shared pieces of the cutoff-summation programs (05a..05d, 06, 07)
//
// Problem (book Ch.10): electrostatic potential on a lattice, but each atom only contributes to lattice points within
// a cutoff radius rc:  V(p) = sum over atoms with |p - a| < rc of q / |p - a|.
// Atoms are sorted into uniform box BINS so a lattice point only looks at a neighbourhood of bins instead of all atoms.
#pragma once
#include "common.cuh"
#include "monitor.cuh"
#include <algorithm>
#include <functional>
#include <stdarg.h>

#define BIN_CAP 8            // fixed atoms per bin (SmallBin); extra atoms go to an overflow list
#define SB_TX 8              // SmallBin thread block = 8 x 8 x 4 lattice points
#define SB_TY 8
#define SB_TZ 4

struct Cut { float L, h, rc, bs; int gx, gy, gz, nb, natoms; double density; };
// cubic domain of edge L (Angstrom), lattice spacing h, cutoff rc, bin edge bs, atom density (atoms per A^3)
static Cut make_cut(float L, float h, float rc, float bs, double density) {
    Cut c; c.L = L; c.h = h; c.rc = rc; c.bs = bs; c.density = density;
    c.gx = c.gy = c.gz = (int)ceil(L / h); c.nb = (int)ceil(L / bs); c.natoms = (int)(density * L * L * L);
    return c;
}
static double volume(const Cut &c) { return (double)c.L * c.L * c.L; }
static size_t npoints(const Cut &c) { return (size_t)c.gx * c.gy * c.gz; }

static std::vector<float4> make_atoms_box(const Cut &c, unsigned seed) {
    std::vector<float4> a(c.natoms);
    srand(seed);
    auto u = []() { return (float)rand() / ((float)RAND_MAX + 1.f); };
    for (auto &p : a) { p.x = u() * c.L; p.y = u() * c.L; p.z = u() * c.L; p.w = 2.f * u() - 1.f; }
    return a;
}

// ---- binning ---------------------------------------------------------------------------------------------------------
static int bin_of(float v, const Cut &c) { int b = (int)(v / c.bs); return b < 0 ? 0 : (b >= c.nb ? c.nb - 1 : b); }
struct Bins {                      // fixed-capacity bins + overflow list (SmallBin data layout)
    std::vector<float4> a; std::vector<int> cnt; std::vector<float4> ovf;
};
static Bins make_bins(const std::vector<float4> &atoms, const Cut &c) {
    Bins b; size_t nbins = (size_t)c.nb * c.nb * c.nb;
    b.a.assign(nbins * BIN_CAP, make_float4(1e9f, 1e9f, 1e9f, 0.f)); b.cnt.assign(nbins, 0);
    for (auto &p : atoms) {
        size_t idx = ((size_t)bin_of(p.z, c) * c.nb + bin_of(p.y, c)) * c.nb + bin_of(p.x, c);
        if (b.cnt[idx] < BIN_CAP) b.a[idx * BIN_CAP + b.cnt[idx]++] = p; else b.ovf.push_back(p);
    }
    return b;
}
struct Cells {                     // unlimited bins in compressed form (LargeBin / neighbourhood queries)
    std::vector<int> start; std::vector<float4> a;
};
static Cells make_cells(const std::vector<float4> &atoms, const Cut &c) {
    size_t nbins = (size_t)c.nb * c.nb * c.nb; Cells s; s.start.assign(nbins + 1, 0);
    std::vector<size_t> id(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) { id[i] = ((size_t)bin_of(atoms[i].z, c) * c.nb + bin_of(atoms[i].y, c)) * c.nb + bin_of(atoms[i].x, c); s.start[id[i] + 1]++; }
    for (size_t b = 0; b < nbins; b++) s.start[b + 1] += s.start[b];
    s.a.resize(atoms.size()); std::vector<int> fill(nbins, 0);
    for (size_t i = 0; i < atoms.size(); i++) s.a[s.start[id[i]] + fill[id[i]]++] = atoms[i];
    return s;
}

// ---- references and CPU baselines -------------------------------------------------------------------------------------
// double-precision value at one lattice point; also sum(|q|/r) over atoms inside the cutoff as the error scale
static double ref_cutoff_point(const std::vector<float4> &atoms, const Cut &c, int ix, int iy, int iz, double *scale, int *inside = nullptr) {
    double e = 0, sc = 0, x = (double)c.h * ix, y = (double)c.h * iy, z = (double)c.h * iz, rc2 = (double)c.rc * c.rc; int n = 0;
    for (auto &a : atoms) {
        double dx = x - a.x, dy = y - a.y, dz = z - a.z, r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) { double r = sqrt(r2); e += a.w / r; sc += fabs((double)a.w) / r; n++; }
    }
    *scale = sc; if (inside) *inside = n; return e;
}
// sampled check against the double reference; returns the worst error scaled by sum(|q|/r)
static double check_cutoff(const std::vector<float> &v, const Cut &c, const std::vector<float4> &atoms, int count, double *mean_inside = nullptr) {
    double worst = 0, sin = 0; unsigned s = 4242;
    for (int k = 0; k < count; k++) {
        s = s * 1664525u + 1013904223u; int ix = (s >> 8) % c.gx;
        s = s * 1664525u + 1013904223u; int iy = (s >> 8) % c.gy;
        s = s * 1664525u + 1013904223u; int iz = (s >> 8) % c.gz;
        double sc; int in; double ref = ref_cutoff_point(atoms, c, ix, iy, iz, &sc, &in);
        double e = sc > 0 ? fabs(v[((size_t)iz * c.gy + iy) * c.gx + ix] - ref) / sc : fabs(v[((size_t)iz * c.gy + iy) * c.gx + ix] - ref);
        if (e > worst) worst = e; sin += in;
    }
    if (mean_inside) *mean_inside = sin / count;
    return worst;
}
// atom-centric CPU cutoff summation (the book's sequential algorithm); adds into out for lattice z in [zlo, zhi)
static void cpu_atom_centric(const std::vector<float4> &atoms, const Cut &c, std::vector<float> &out, int zlo, int zhi) {
    float rc2 = c.rc * c.rc;
    for (auto &a : atoms) {
        int i0 = std::max(0, (int)ceilf((a.x - c.rc) / c.h)), i1 = std::min(c.gx - 1, (int)floorf((a.x + c.rc) / c.h));
        int j0 = std::max(0, (int)ceilf((a.y - c.rc) / c.h)), j1 = std::min(c.gy - 1, (int)floorf((a.y + c.rc) / c.h));
        int k0 = std::max(zlo, (int)ceilf((a.z - c.rc) / c.h)), k1 = std::min(zhi - 1, (int)floorf((a.z + c.rc) / c.h));
        for (int k = k0; k <= k1; k++) { float dz = c.h * k - a.z, dz2 = dz * dz;
            for (int j = j0; j <= j1; j++) { float dy = c.h * j - a.y, dyz2 = dy * dy + dz2;
                for (int i = i0; i <= i1; i++) { float dx = c.h * i - a.x, r2 = dx * dx + dyz2;
                    if (r2 < rc2) out[((size_t)k * c.gy + j) * c.gx + i] += a.w / sqrtf(r2); } } }
    }
}

// ---- GPU kernels shared by several programs ----------------------------------------------------------------------------
// DirectSum with the cutoff test: grid-centric, ALL atoms visited through shared-memory tiles. O(points x atoms).
// Visits atoms [a0, a1) and ADDS to out, so a long run can be split into short launches (keeps each launch well under
// the Windows GPU watchdog limit); zero `out` first.
__global__ void direct_cutoff_kernel(const float4 *atoms, int a0, int a1, float h, float rc2, int gx, int gy, int gz, float *out) {
    __shared__ float4 sh[256];
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x, tot = (size_t)gx * gy * gz;
    int ix = (int)(idx % gx), iy = (int)((idx / gx) % gy), iz = (int)(idx / ((size_t)gx * gy));
    float x = h * ix, y = h * iy, z = h * iz, e = 0.f;
    for (int t = a0; t < a1; t += 256) {
        int j = t + threadIdx.x;
        sh[threadIdx.x] = j < a1 ? atoms[j] : make_float4(1e9f, 1e9f, 1e9f, 0.f);
        __syncthreads();
        if (idx < tot)
            for (int k = 0; k < 256; k++) {
                float4 A = sh[k]; float dx = x - A.x, dy = y - A.y, dz = z - A.z, r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < rc2) e += A.w * rsqrtf(r2);
            }
        __syncthreads();
    }
    if (idx < tot) out[idx] += e;
}
// host helper: run the direct kernel over all atoms in launches of at most `chunk` atoms (out is zeroed first)
static void direct_cutoff_run(const float4 *dAtoms, int n, const Cut &c, float *dOut, int chunk = 16384) {
    CUDA_CHECK(cudaMemset(dOut, 0, npoints(c) * 4));
    int blocks = (int)((npoints(c) + 255) / 256);
    for (int a0 = 0; a0 < n; a0 += chunk)
        direct_cutoff_kernel<<<blocks, 256>>>(dAtoms, a0, std::min(n, a0 + chunk), c.h, c.rc * c.rc, c.gx, c.gy, c.gz, dOut);
}

// SmallBin: a block owns a tile of lattice points; it walks the bins overlapping (tile +- rc), 4 bins at a time,
// copying them from global memory into shared memory, then every thread tests the atoms against the cutoff.
__global__ void smallbin_kernel(const float4 *binAtoms, const int *binCount, int nb, float bs, float rc, float h,
                                int gx, int gy, int gz, int zoff_blocks, float *out) {
    __shared__ float4 sh[4 * BIN_CAP];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, tid = (tz * SB_TY + ty) * SB_TX + tx;
    int zb = blockIdx.z + zoff_blocks;
    int ix = blockIdx.x * SB_TX + tx, iy = blockIdx.y * SB_TY + ty, iz = zb * SB_TZ + tz;
    float x = h * ix, y = h * iy, z = h * iz;
    float x0 = h * (blockIdx.x * SB_TX), x1 = x0 + h * (SB_TX - 1);
    float y0 = h * (blockIdx.y * SB_TY), y1 = y0 + h * (SB_TY - 1);
    float z0 = h * (zb * SB_TZ),        z1 = z0 + h * (SB_TZ - 1);
    int bx0 = max(0, (int)floorf((x0 - rc) / bs)), bx1 = min(nb - 1, (int)floorf((x1 + rc) / bs));
    int by0 = max(0, (int)floorf((y0 - rc) / bs)), by1 = min(nb - 1, (int)floorf((y1 + rc) / bs));
    int bz0 = max(0, (int)floorf((z0 - rc) / bs)), bz1 = min(nb - 1, (int)floorf((z1 + rc) / bs));
    int dxn = bx1 - bx0 + 1, dyn = by1 - by0 + 1, dzn = bz1 - bz0 + 1, nbn = dxn * dyn * dzn;
    float rc2 = rc * rc, e = 0.f;
    for (int b0 = 0; b0 < nbn; b0 += 4) {
        if (tid < 4 * BIN_CAP) {
            int lb = b0 + tid / BIN_CAP, slot = tid % BIN_CAP;
            float4 a = make_float4(1e9f, 1e9f, 1e9f, 0.f);
            if (lb < nbn) {
                int lx = lb % dxn, ly = (lb / dxn) % dyn, lz = lb / (dxn * dyn);
                int bin = ((bz0 + lz) * nb + (by0 + ly)) * nb + (bx0 + lx);
                if (slot < binCount[bin]) a = binAtoms[(size_t)bin * BIN_CAP + slot];
            }
            sh[tid] = a;
        }
        __syncthreads();
        for (int k = 0; k < 4 * BIN_CAP; k++) {
            float4 A = sh[k]; float dx = x - A.x, dy = y - A.y, dz = z - A.z, r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rc2) e += A.w * rsqrtf(r2);
        }
        __syncthreads();
    }
    if (ix < gx && iy < gy && iz < gz) out[((size_t)iz * gy + iy) * gx + ix] = e;
}

// average number of stored atoms a SmallBin block examines per lattice point (block neighbourhood size), from the data
static double smallbin_mean_candidates(const Bins &b, const Cut &c) {
    double sum = 0; long blocks = 0;
    int nbx = (c.gx + SB_TX - 1) / SB_TX, nby = (c.gy + SB_TY - 1) / SB_TY, nbz = (c.gz + SB_TZ - 1) / SB_TZ;
    for (int bz = 0; bz < nbz; bz++) for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
        auto rng = [&](int blk, int T, int &lo, int &hi) { float a = c.h * blk * T, z1 = a + c.h * (T - 1);
            lo = std::max(0, (int)floorf((a - c.rc) / c.bs)); hi = std::min(c.nb - 1, (int)floorf((z1 + c.rc) / c.bs)); };
        int x0, x1, y0, y1, z0, z1; rng(bx, SB_TX, x0, x1); rng(by, SB_TY, y0, y1); rng(bz, SB_TZ, z0, z1);
        long s = 0;
        for (int k = z0; k <= z1; k++) for (int j = y0; j <= y1; j++) for (int i = x0; i <= x1; i++) s += b.cnt[((size_t)k * c.nb + j) * c.nb + i];
        sum += s; blocks++;
    }
    return sum / blocks;
}

// The slow sequential CPU baseline is measured ONCE and cached in stats/cpu_baseline.csv (delete the file to re-measure),
// so the GPU programs spend their time on the GPU. -1 if there is no cached value for this edge L.
static double cpu_baseline_ms(float L) {
    FILE *f = fopen("stats/cpu_baseline.csv", "r"); if (!f) return -1;
    char line[512]; double r = -1;
    while (fgets(line, sizeof line, f)) {
        char alg[64]; float l; double vol, ms; long pts; int atoms;
        if (sscanf(line, "%63[^,],%f,%lf,%ld,%d,%lf", alg, &l, &vol, &pts, &atoms, &ms) == 6 && !strcmp(alg, "cpu_atom_centric") && fabsf(l - L) < 0.01f) r = ms;
    }
    fclose(f); return r;
}

// ms of the row (algorithm, L) in a stats CSV written by another program; -1 if missing
static double stat_ms(const char *path, const char *algo, float L) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[600]; double r = -1;
    while (fgets(line, sizeof line, f)) {
        char alg[64]; float l; double vol, ms; long pts; int atoms;
        if (sscanf(line, "%63[^,],%f,%lf,%ld,%d,%lf", alg, &l, &vol, &pts, &atoms, &ms) == 6 && !strcmp(alg, algo) && fabsf(l - L) < 0.01f) r = ms;
    }
    fclose(f); return r;
}


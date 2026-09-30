/*
 * 07_cutoff_divergence.cu   -- "The grid approach serialises because each thread in a warp has a different job"
 *
 * Book (Ch.10): in the grid-centric (gather) cutoff kernels every thread owns one lattice point and tests each candidate atom
 * against the cutoff sphere. An atom can be inside the sphere for one thread and outside for a neighbour in the same warp,
 * which is CONTROL DIVERGENCE: the warp executes both paths one after the other. The book calls this "a small price to pay"
 * compared with the atom-centric alternative (atomicAdd on the lattice).
 *
 * This program measures it instead of assuming it, on the SmallBin kernel with real bins:
 *   1. WARP SHAPE. The same 256 lattice points per block are assigned to threads with different tile shapes, so a warp of 32
 *      threads covers a line, a plane patch or a compact box of lattice points. For every distance test the warp votes
 *      (__ballot_sync): all lanes inside, all lanes outside, or MIXED (divergent). We report the share of mixed tests and the
 *      average fraction of lanes that are active when the "inside" branch runs.
 *   2. COST. Branchy kernel (skip the rsqrt when outside) vs branchless kernel (always compute, multiply by a 0/1 mask) vs a
 *      no-test kernel (the arithmetic lower bound; NOT the physics). Do divergent warps cost more than computing everything?
 *   3. CONTEXT. The atom-centric scatter time from 05a and the SmallBin total from 05c at comparable sizes, if available.
 * Domain: L = 48 A (96^3 lattice points, ~11,000 atoms), rc = 8 A, bins 3 A, capacity 8. Statistics: stats/07_divergence.csv
 */
#include "common_bins.cuh"
#include <sys/stat.h>

// SmallBin kernel with a configurable tile shape (TX,TY,TZ), test style (MODE) and optional warp-vote counters (COUNT).
template <int TX, int TY, int TZ, int MODE, bool COUNT>
__global__ void sb_probe(const float4 *binAtoms, const int *binCount, int nb, float bs, float rc, float h, float *out, unsigned long long *ctr) {
    __shared__ float4 sh[4 * BIN_CAP];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, tid = (tz * TY + ty) * TX + tx;
    int ix = blockIdx.x * TX + tx, iy = blockIdx.y * TY + ty, iz = blockIdx.z * TZ + tz;
    int gx = gridDim.x * TX, gy = gridDim.y * TY;
    float x = h * ix, y = h * iy, z = h * iz;
    float x0 = h * (blockIdx.x * TX), x1 = x0 + h * (TX - 1), y0 = h * (blockIdx.y * TY), y1 = y0 + h * (TY - 1), z0 = h * (blockIdx.z * TZ), z1 = z0 + h * (TZ - 1);
    int bx0 = max(0, (int)floorf((x0 - rc) / bs)), bx1 = min(nb - 1, (int)floorf((x1 + rc) / bs));
    int by0 = max(0, (int)floorf((y0 - rc) / bs)), by1 = min(nb - 1, (int)floorf((y1 + rc) / bs));
    int bz0 = max(0, (int)floorf((z0 - rc) / bs)), bz1 = min(nb - 1, (int)floorf((z1 + rc) / bs));
    int dxn = bx1 - bx0 + 1, dyn = by1 - by0 + 1, dzn = bz1 - bz0 + 1, nbn = dxn * dyn * dzn;
    float rc2 = rc * rc, e = 0.f;
    unsigned c_out = 0, c_in = 0, c_mix = 0, c_any = 0, c_lanes = 0;
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
            if (COUNT) {
                unsigned m = __ballot_sync(0xffffffffu, r2 < rc2);
                if ((tid & 31) == 0) { if (m == 0) c_out++; else if (m == 0xffffffffu) c_in++; else c_mix++; if (m) { c_any++; c_lanes += __popc(m); } }
            }
            if (MODE == 0) { if (r2 < rc2) e += A.w * rsqrtf(r2); }
            else if (MODE == 1) { float mk = r2 < rc2 ? 1.f : 0.f; e += mk * A.w * rsqrtf(fmaxf(r2, 1e-6f)); }
            else e += A.w * rsqrtf(r2 + 1.f);
        }
        __syncthreads();
    }
    if (COUNT && (tid & 31) == 0) { atomicAdd(&ctr[0], c_out); atomicAdd(&ctr[1], c_in); atomicAdd(&ctr[2], c_mix); atomicAdd(&ctr[3], c_any); atomicAdd(&ctr[4], c_lanes); }
    out[((size_t)iz * gy + iy) * gx + ix] = e;
}

struct Shape { int TX, TY, TZ; double mixed, allin, allout, lane_eff, t0, t1, t2, err0, err1; };

template <int TX, int TY, int TZ>
static Shape run_shape(const Cut &c, const std::vector<float4> &stored, const float4 *dBin, const int *dCnt, float *dOut, unsigned long long *dCtr) {
    Shape s = { TX, TY, TZ, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    dim3 blk(TX, TY, TZ), grd(c.gx / TX, c.gy / TY, c.gz / TZ);
    cudaMemset(dCtr, 0, 8 * sizeof(unsigned long long));
    sb_probe<TX, TY, TZ, 0, true><<<grd, blk>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, dOut, dCtr);
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
    unsigned long long h[8]; cudaMemcpy(h, dCtr, sizeof h, cudaMemcpyDeviceToHost);
    double tot = (double)(h[0] + h[1] + h[2]);
    s.allout = 100.0 * h[0] / tot; s.allin = 100.0 * h[1] / tot; s.mixed = 100.0 * h[2] / tot; s.lane_eff = h[3] ? 100.0 * h[4] / (32.0 * h[3]) : 0;
    std::vector<float> r0(npoints(c)), r1(npoints(c));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto tmin = [&](int mode, std::vector<float> *res) {
        double best = 1e30;
        for (int i = 0; i < 8; i++) {
            cudaEventRecord(a);
            if (mode == 0) sb_probe<TX, TY, TZ, 0, false><<<grd, blk>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, dOut, dCtr);
            else if (mode == 1) sb_probe<TX, TY, TZ, 1, false><<<grd, blk>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, dOut, dCtr);
            else sb_probe<TX, TY, TZ, 2, false><<<grd, blk>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, dOut, dCtr);
            cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (i > 0) best = std::min(best, (double)ms);
        }
        if (res) cudaMemcpy(res->data(), dOut, npoints(c) * 4, cudaMemcpyDeviceToHost);
        return best; };
    s.t0 = tmin(0, &r0); s.t1 = tmin(1, &r1); s.t2 = tmin(2, nullptr);
    CUDA_CHECK(cudaGetLastError());
    s.err0 = check_cutoff(r0, c, stored, 200); s.err1 = check_cutoff(r1, c, stored, 200);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return s;
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &p = dev_props();
    mkdir("stats", 0755); remove("stats/07_divergence.csv");
    Cut c = make_cut(48, 0.5f, 8.0f, 3.0f, 0.1);
    auto atoms = make_atoms_box(c, 148);
    Bins bins = make_bins(atoms, c);
    std::vector<float4> stored;                                     // atoms that fit in a bin (the overflow list is not used here)
    for (size_t b = 0; b < bins.cnt.size(); b++) for (int k = 0; k < bins.cnt[b]; k++) stored.push_back(bins.a[b * BIN_CAP + k]);
    printf("==================== CUTOFF DIVERGENCE (%s) ====================\n", p.name);
    printf("L = 48 A: %zu lattice points, %zu atoms (%zu stored in bins, %zu overflow ignored), cutoff 8 A, bin edge 3 A, 256 threads/block.\n\n", npoints(c), atoms.size(), stored.size(), bins.ovf.size());

    float4 *dBin; int *dCnt; float *dOut; unsigned long long *dCtr;
    CUDA_CHECK(cudaMalloc(&dBin, bins.a.size() * sizeof(float4))); CUDA_CHECK(cudaMalloc(&dCnt, bins.cnt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOut, npoints(c) * 4)); CUDA_CHECK(cudaMalloc(&dCtr, 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(dBin, bins.a.data(), bins.a.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCnt, bins.cnt.data(), bins.cnt.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<Shape> sh;
    cooldown();
    // prime the clocks with a full run
    for (int i = 0; i < 6; i++) sb_probe<8, 8, 4, 0, false><<<dim3(c.gx / 8, c.gy / 8, c.gz / 4), dim3(8, 8, 4)>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, dOut, dCtr);
    cudaDeviceSynchronize();
    sh.push_back(run_shape<32, 8, 1>(c, stored, dBin, dCnt, dOut, dCtr)); cooldown();
    sh.push_back(run_shape<16, 16, 1>(c, stored, dBin, dCnt, dOut, dCtr)); cooldown();
    sh.push_back(run_shape<8, 8, 4>(c, stored, dBin, dCnt, dOut, dCtr)); cooldown();
    sh.push_back(run_shape<4, 4, 16>(c, stored, dBin, dCnt, dOut, dCtr)); cooldown();
    sh.push_back(run_shape<2, 4, 32>(c, stored, dBin, dCnt, dOut, dCtr));

    printf("---- 1 + 2. Warp shape: how often do the 32 lanes of a warp disagree, and what does it cost? ----\n");
    printf("A warp = 32 consecutive threads (x fastest). Its footprint in lattice points (and Angstrom, spacing 0.5) depends on the tile shape.\n");
    printf("%-11s %-16s %-19s | %-7s %-7s %-7s | %-11s | %-9s %-11s %-9s | %-8s\n", "tile", "warp footprint", "warp extent (A)", "all in", "all out", "MIXED", "lane use", "branchy", "branchless", "no test", "branchy/");
    printf("%-11s %-16s %-19s | %-7s %-7s %-7s | %-11s | %-9s %-11s %-9s | %-8s\n", "x*y*z", "(lattice pts)", "x  y  z", "%", "%", "%", "in branch %", "ms", "ms", "ms", "branchless");
    bool ok = true;
    for (auto &s : sh) {
        int wx = std::min(32, s.TX), rem = 32 / wx, wy = std::min(rem, s.TY), wz = rem / wy;
        char tile[16], foot[24], ext[32];
        snprintf(tile, sizeof tile, "%dx%dx%d", s.TX, s.TY, s.TZ); snprintf(foot, sizeof foot, "%dx%dx%d", wx, wy, wz);
        snprintf(ext, sizeof ext, "%.1f %.1f %.1f", (wx - 1) * 0.5, (wy - 1) * 0.5, (wz - 1) * 0.5);
        bool good = s.err0 < 1e-4 && s.err1 < 1e-4; ok &= good;
        printf("%-11s %-16s %-19s | %-7.1f %-7.1f %-7.1f | %-11.1f | %-9.2f %-11.2f %-9.2f | %-8.2f %s\n", tile, foot, ext, s.allin, s.allout, s.mixed, s.lane_eff, s.t0, s.t1, s.t2, s.t0 / s.t1, good ? "PASS" : "FAIL");
        FILE *f = fopen("stats/07_divergence.csv", "a"); if (f) { if (ftell(f) == 0) fprintf(f, "tile,warp_footprint,all_in_pct,all_out_pct,mixed_pct,lane_use_pct,branchy_ms,branchless_ms,notest_ms,check\n");
            fprintf(f, "%s,%s,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%s\n", tile, foot, s.allin, s.allout, s.mixed, s.lane_eff, s.t0, s.t1, s.t2, good ? "PASS" : "FAIL"); fclose(f); }
    }
    printf("  MIXED = share of warp-level distance tests where some lanes are inside the sphere and others outside (divergent).\n");
    printf("  lane use = of the lanes in the warp, the average fraction that are inside when the inside branch executes (1 lane of 32 = 3 %%).\n");
    printf("  'branchy' skips the rsqrt when outside; 'branchless' always computes it and multiplies by 0 or 1; 'no test' is the arithmetic\n");
    printf("  lower bound (different physics, timing only). branchy/branchless < 1 means the divergent version is still faster.\n");
    printf("  Both physical kernels are checked against a double-precision reference at 200 lattice points.\n\n");

    printf("---- 3. Context: divergence (gather) vs atomics (scatter) [MEASURED, other programs] ----\n");
    double sc40 = stat_ms("stats/05a_cutoff.csv", "gpu_atom_centric", 40), sc80 = stat_ms("stats/05a_cutoff.csv", "gpu_atom_centric", 80);
    double sb40 = stat_ms("stats/05c_cutoff.csv", "gpu_smallbin", 40), sb80 = stat_ms("stats/05c_cutoff.csv", "gpu_smallbin", 80);
    if (sc40 > 0 && sb40 > 0) printf("  L = 40 A: atom-centric scatter (atomicAdd) %.1f ms   vs   SmallBin gather with the divergent test %.1f ms  -> %.1fx\n", sc40, sb40, sc40 / sb40);
    if (sc80 > 0 && sb80 > 0) printf("  L = 80 A: atom-centric scatter (atomicAdd) %.1f ms   vs   SmallBin gather with the divergent test %.1f ms  -> %.1fx\n", sc80, sb80, sc80 / sb80);
    if (sc40 < 0 && sc80 < 0) printf("  (run 05a and 05c first to fill this section)\n");
    printf("  [book] the divergence of the gather kernel is the smaller evil compared with atomics. The two lines above are the local test\n");
    printf("  of that statement (05c includes host binning and upload, 05a is the kernel only).\n");
    printf("\n%s\n", ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return ok ? 0 : 1;
}

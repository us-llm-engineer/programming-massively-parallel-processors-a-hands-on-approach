/*
 * 05c_cutoff_smallbin.cu   -- CUTOFF SUMMATION, version 3: SmallBin
 *
 * Book (Ch.10, Fig 10.3): store ALL atom bins in GLOBAL memory in a fixed-size layout (equal capacity, aligned so loads
 * coalesce). Different thread blocks own different lattice tiles, so different blocks read DIFFERENT neighbourhoods; the
 * threads of a block cooperatively copy the bins around their tile into SHARED memory (the constant cache would thrash,
 * see 06) and every thread tests the atoms against the cutoff. Bin capacity is fixed: atoms that do not fit go to an
 * OVERFLOW LIST that the CPU processes afterwards (here strictly after the GPU: the non-overlapped version; the overlapped
 * one is 05d).
 *
 * Measured: (1) volume ladder with bin edge 3 A (capacity 8 atoms per bin), (2) bin edge sweep at L = 40 A showing the
 * capacity/overflow trade-off. Statistics: stats/05c_cutoff.csv. Correctness: 200 sampled points vs a double reference.
 */
#include "common_bins.cuh"
#include <string>
#include <sys/stat.h>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string ratio_str(double a, double b) {          // "12.3x" or "n/a"
    if (a <= 0 || b <= 0) return "n/a";
    char t[32]; snprintf(t, sizeof t, "%.1fx", a / b); return t;
}

struct SBRun { double host_bin, h2d, kernel, ovf_cpu, total; size_t ovf_atoms; double cand, inside, err; long blocks; bool ok; };

static SBRun run_smallbin(const Cut &c, const std::vector<float4> &atoms, float *dOut, std::vector<float> &result) {
    SBRun r = {};
    double t0 = now_ms(); Bins bins = make_bins(atoms, c); r.host_bin = now_ms() - t0;
    r.ovf_atoms = bins.ovf.size(); r.cand = smallbin_mean_candidates(bins, c);
    float4 *dBin; int *dCnt;
    CUDA_CHECK(cudaMalloc(&dBin, bins.a.size() * sizeof(float4))); CUDA_CHECK(cudaMalloc(&dCnt, bins.cnt.size() * sizeof(int)));
    cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(dBin, bins.a.data(), bins.a.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCnt, bins.cnt.data(), bins.cnt.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    dim3 blk(SB_TX, SB_TY, SB_TZ), grd((c.gx + SB_TX - 1) / SB_TX, (c.gy + SB_TY - 1) / SB_TY, (c.gz + SB_TZ - 1) / SB_TZ);
    r.blocks = (long)grd.x * grd.y * grd.z;
    auto launch = [&]() { smallbin_kernel<<<grd, blk>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, c.gx, c.gy, c.gz, 0, dOut); };
    launch(); cudaDeviceSynchronize();                                 // warm-up (not part of the upload time)
    cudaEventRecord(e3);
    for (int i = 0; i < 3; i++) launch();
    cudaEventRecord(e2); cudaEventSynchronize(e2); CUDA_CHECK(cudaGetLastError());
    float ms_h2d, ms_k; cudaEventElapsedTime(&ms_h2d, e0, e1); cudaEventElapsedTime(&ms_k, e3, e2);
    r.h2d = ms_h2d; r.kernel = ms_k / 3;
    result.assign(npoints(c), 0.f);
    CUDA_CHECK(cudaMemcpy(result.data(), dOut, npoints(c) * 4, cudaMemcpyDeviceToHost));
    // serial CPU pass over the overflow atoms (added into the GPU result)
    std::vector<float> ovf(npoints(c), 0.f); double t1 = now_ms();
    cpu_atom_centric(bins.ovf, c, ovf, 0, c.gz); r.ovf_cpu = now_ms() - t1;
    for (size_t i = 0; i < result.size(); i++) result[i] += ovf[i];
    r.total = r.host_bin + r.h2d + r.kernel + r.ovf_cpu;
    r.err = check_cutoff(result, c, atoms, 200, &r.inside); r.ok = r.err < 1e-4;
    cudaFree(dBin); cudaFree(dCnt); cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    return r;
}

int main(int argc, char **argv) {
    RunMonitor mon;
    const cudaDeviceProp &p = dev_props();
    float only_L = argc > 1 ? (float)atof(argv[1]) : 0;              // optional: run just one lattice edge, e.g. ./05c_cutoff_smallbin 120
    mkdir("stats", 0755); if (!only_L) remove("stats/05c_cutoff.csv");
    const char *HDR = "algorithm,L_angstrom,volume_A3,points,atoms,ms,max_err,check,bin_edge_A,host_bin_ms,h2d_ms,kernel_ms,overflow_cpu_ms,overflow_atoms,mean_candidates,mean_inside";
    printf("==================== CUTOFF SUMMATION v3: SMALLBIN (%s) ====================\n", p.name);
    printf("[book] bins in global memory (fixed capacity %d), per-block neighbourhood tiled into shared memory, overflow atoms on the CPU.\n", BIN_CAP);
    printf("lattice 0.5 A, cutoff 8 A, density 0.1 / A^3. Blocks of %dx%dx%d lattice points. 'total' = host binning + bin upload + kernel + serial CPU overflow pass.\n\n", SB_TX, SB_TY, SB_TZ);

    bool all_ok = true;
    printf("---- 1. Volume ladder, bin edge 3 A (mean 2.7 atoms per bin, capacity %d: essentially no overflow) ----\n", BIN_CAP);
    printf("(edge 4 A would overflow about 7%% of the atoms; that overflow-heavy case is the subject of section 2 and of 05d)\n");
    printf("%-6s %-11s %-9s %-8s | %-9s %-8s %-9s %-10s %-9s | %-9s %-9s | %-10s %-9s | %s\n", "L (A)", "volume A^3", "points", "atoms", "bin host", "upload", "kernel", "overflow", "total ms",
           "vs CPU", "vs Large", "candidates", "useful", "max err");
    float Ls[6] = { 10, 20, 40, 80, 120, 160 };          // 120 and 160 A are GPU-only scales (no sequential CPU baseline)
    for (float L : Ls) {
        if (only_L && L != only_L) continue;
        cooldown();
        Cut c = make_cut(L, 0.5f, 8.0f, 3.0f, 0.1);
        auto atoms = make_atoms_box(c, 100 + (unsigned)L);
        float *d; CUDA_CHECK(cudaMalloc(&d, npoints(c) * 4)); std::vector<float> res;
        run_smallbin(c, atoms, d, res);                                  // warm-up of the whole pipeline (first call pays lazy start-up)
        SBRun r = run_smallbin(c, atoms, d, res); all_ok &= r.ok;
        double cpu = cpu_baseline_ms(L), lb = stat_ms("stats/05b_cutoff.csv", "gpu_largebin", L);
        printf("%-6.0f %-11.0f %-9zu %-8d | %-9.2f %-8.2f %-9.2f %-10.2f %-9.2f | %-9s %-9s | %-10.0f %-8.1f%% | %.1e %s\n", L, volume(c), npoints(c), c.natoms, r.host_bin, r.h2d, r.kernel, r.ovf_cpu, r.total,
               ratio_str(cpu, r.total).c_str(), ratio_str(lb, r.total).c_str(), r.cand, 100.0 * r.inside / r.cand, r.err, r.ok ? "PASS" : "FAIL");
        csv_row("stats/05c_cutoff.csv", HDR, "gpu_smallbin,%.0f,%.0f,%zu,%d,%.3f,%.2e,%s,3,%.3f,%.3f,%.3f,%.3f,%zu,%.1f,%.1f", L, volume(c), npoints(c), c.natoms, r.total, r.err, r.ok ? "PASS" : "FAIL",
                r.host_bin, r.h2d, r.kernel, r.ovf_cpu, r.ovf_atoms, r.cand, r.inside);
        if (L == 80) {
            printf("  overflow at L = 80: %zu of %d atoms (%.1f %%) did not fit in a bin of capacity %d\n", r.ovf_atoms, c.natoms, 100.0 * r.ovf_atoms / c.natoms, BIN_CAP);
            printf("  launch fill (L = 80): "); ledger_header();
            ledger_row("  SmallBin kernel", (const void *)smallbin_kernel, SB_TX * SB_TY * SB_TZ, r.blocks, 0);
        }
        hot_note();
        cudaFree(d);
    }
    printf("  'vs Large' = LargeBin total (05b, subvolume 16 A) divided by SmallBin total; > 1 means SmallBin is faster.\n\n");

    printf("---- 2. Bin edge sweep at L = 40 A (fixed capacity %d atoms per bin) ----\n", BIN_CAP);
    printf("[book] fixed capacity wastes slots in sparse bins and overflows in dense ones; the overflow list is the CPU's job.\n");
    printf("%-9s %-8s %-11s %-10s %-9s %-10s %-11s %-9s | %s\n", "bin edge", "bins", "atoms/bin", "overflow", "kernel", "overflow", "total ms", "useful", "max err");
    printf("%-9s %-8s %-11s %-10s %-9s %-10s %-11s %-9s |\n", "(A)", "", "(mean)", "atoms %", "ms", "CPU ms", "", "");
    {
        float bss[4] = { 2, 3, 4, 5 };
        for (float bs : bss) {
            cooldown();
            Cut c = make_cut(40, 0.5f, 8.0f, bs, 0.1);
            auto atoms = make_atoms_box(c, 140);
            float *d; CUDA_CHECK(cudaMalloc(&d, npoints(c) * 4)); std::vector<float> res;
            run_smallbin(c, atoms, d, res);
            SBRun r = run_smallbin(c, atoms, d, res); all_ok &= r.ok;
            printf("%-9.0f %-8d %-11.1f %-9.1f%% %-9.2f %-10.2f %-11.2f %-8.1f%% | %.1e %s\n", bs, c.nb * c.nb * c.nb, (double)c.natoms / (c.nb * c.nb * c.nb), 100.0 * r.ovf_atoms / c.natoms, r.kernel, r.ovf_cpu, r.total,
                   100.0 * r.inside / r.cand, r.err, r.ok ? "PASS" : "FAIL");
            csv_row("stats/05c_cutoff.csv", HDR, "gpu_smallbin_binsweep,40,%.0f,%zu,%d,%.3f,%.2e,%s,%.0f,%.3f,%.3f,%.3f,%.3f,%zu,%.1f,%.1f", volume(c), npoints(c), c.natoms, r.total, r.err, r.ok ? "PASS" : "FAIL", bs,
                    r.host_bin, r.h2d, r.kernel, r.ovf_cpu, r.ovf_atoms, r.cand, r.inside);
            cudaFree(d);
        }
    }
    printf("\nHow to read: a small bin edge gives tight neighbourhoods (high 'useful') but many nearly empty bins and more loop\n");
    printf("iterations; a large edge overflows most atoms to the CPU. The right edge balances GPU time against the CPU overflow pass.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

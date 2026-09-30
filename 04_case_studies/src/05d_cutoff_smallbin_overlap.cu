/*
 * 05d_cutoff_smallbin_overlap.cu   -- CUTOFF SUMMATION, version 4: SmallBin-Overlap
 *
 * Book (Ch.10, Fig 10.3): SmallBin keeps a fixed number of atoms per bin; atoms that do not fit go to an OVERFLOW LIST that
 * the CPU handles with the sequential cutoff algorithm. In plain SmallBin (05c) the GPU finishes and then the GPU sits idle
 * while the CPU works through the overflow list. SmallBin-Overlap cuts the lattice into subvolumes: while the GPU runs
 * subvolume i, the CPU already computes the overflow contributions of subvolume i; the CPU work is hidden behind the GPU work.
 *
 * Here the lattice is cut into z-slabs. "serial" = one GPU launch, then the CPU overflow pass. "overlap" = per slab, launch
 * the kernel asynchronously and immediately run the CPU overflow pass for the same slab while the kernel executes.
 * Timings cover the compute phase only (binning and the one-off bin upload are identical in both modes and excluded).
 * The CPU pass is single-threaded scalar code (the book used SSE), so it is slower than the book's; see the reading notes.
 *
 * Measured: (1) volume ladder at bin edge 4 A, (2) bin-edge sweep (how much overflow) at L = 40 A, (3) slab count at L = 40 A.
 * Statistics: stats/05d_overlap.csv and stats/05d_overlap_gantt.csv (a timeline of GPU slabs and CPU overflow slabs).
 * Correctness: every result checked at 200 sampled points against a double-precision reference.
 */
#include "common_bins.cuh"
#include <tuple>
#include <sys/stat.h>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define REPS 5

struct OvRun { double serial, overlap, gpu_only, cpu_only, ideal, hidden_pct, ovf_pct, err; bool ok; };
static std::vector<std::tuple<int, int, double, double>> g_spans;        // (engine 0=GPU 1=CPU, slab, t0, t1) of the last overlapped run

static OvRun run_case(const Cut &c, const std::vector<float4> &atoms, int slabs, bool record) {
    OvRun r = {};
    cooldown();
    Bins bins = make_bins(atoms, c);
    r.ovf_pct = 100.0 * bins.ovf.size() / atoms.size();
    float4 *dBin; int *dCnt; float *dOut; size_t np = npoints(c);
    CUDA_CHECK(cudaMalloc(&dBin, bins.a.size() * sizeof(float4))); CUDA_CHECK(cudaMalloc(&dCnt, bins.cnt.size() * sizeof(int))); CUDA_CHECK(cudaMalloc(&dOut, np * 4));
    CUDA_CHECK(cudaMemcpy(dBin, bins.a.data(), bins.a.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCnt, bins.cnt.data(), bins.cnt.size() * sizeof(int), cudaMemcpyHostToDevice));
    dim3 blk(SB_TX, SB_TY, SB_TZ); int bxn = (c.gx + SB_TX - 1) / SB_TX, byn = (c.gy + SB_TY - 1) / SB_TY, bzn = (c.gz + SB_TZ - 1) / SB_TZ;
    if (slabs > bzn) slabs = bzn;
    auto launch_range = [&](int zb0, int zb1) { smallbin_kernel<<<dim3(bxn, byn, zb1 - zb0), blk>>>(dBin, dCnt, c.nb, c.bs, c.rc, c.h, c.gx, c.gy, c.gz, zb0, dOut); };
    std::vector<float> ovf(np), gpu(np), total(np);
    auto zb_of = [&](int s) { return (int)((long)bzn * s / slabs); };

    // An idle GPU falls back to a low clock, so a short kernel that follows a CPU-only phase starts slow. Before each timing
    // group keep the GPU busy for ~150 ms so every group starts from the same (high) clock state.
    auto prime = [&]() { double t0 = now_ms(); while (now_ms() - t0 < 150) { launch_range(0, bzn); cudaDeviceSynchronize(); } };

    // reference times of the two pieces run alone
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    prime();
    double gmin = 1e30, cmin = 1e30;                                // minimum of repeated runs: robust to interference on a shared machine
    for (int i = 0; i < REPS; i++) {
        cudaEventRecord(a); launch_range(0, bzn); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); gmin = std::min(gmin, (double)ms);
        std::fill(ovf.begin(), ovf.end(), 0.f); double t0 = now_ms(); cpu_atom_centric(bins.ovf, c, ovf, 0, c.gz); cmin = std::min(cmin, now_ms() - t0);
    }
    r.gpu_only = gmin; r.cpu_only = cmin; r.ideal = std::max(r.gpu_only, r.cpu_only);

    // serial: whole GPU launch, then the CPU overflow pass, then copy back and add
    double ss = 1e30, os = 1e30;
    prime();
    for (int i = 0; i <= REPS; i++) {                                              // i = 0 is a warm-up
        std::fill(ovf.begin(), ovf.end(), 0.f);
        double t0 = now_ms();
        launch_range(0, bzn); cudaDeviceSynchronize();
        cpu_atom_centric(bins.ovf, c, ovf, 0, c.gz);
        CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, np * 4, cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < np; k++) total[k] = gpu[k] + ovf[k];
        if (i > 0) ss = std::min(ss, now_ms() - t0);
    }
    r.serial = ss;
    // overlap: per slab launch (async) then the CPU overflow pass of the same slab while the kernel executes
    cudaEvent_t base, es[64][2]; cudaEventCreate(&base);
    for (int s = 0; s < 64; s++) { cudaEventCreate(&es[s][0]); cudaEventCreate(&es[s][1]); }
    prime();
    for (int i = 0; i <= REPS; i++) {
        std::fill(ovf.begin(), ovf.end(), 0.f);
        cudaDeviceSynchronize(); cudaEventRecord(base); cudaEventSynchronize(base);
        double t0 = now_ms(); std::vector<std::pair<double, double>> cspan(slabs);
        for (int s = 0; s < slabs; s++) {
            int zb0 = zb_of(s), zb1 = zb_of(s + 1); if (zb1 <= zb0) continue;
            cudaEventRecord(es[s][0]); launch_range(zb0, zb1); cudaEventRecord(es[s][1]);
            double c0 = now_ms() - t0;
            cpu_atom_centric(bins.ovf, c, ovf, zb0 * SB_TZ, std::min(c.gz, zb1 * SB_TZ));
            cspan[s] = { c0, now_ms() - t0 };
        }
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, np * 4, cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < np; k++) total[k] = gpu[k] + ovf[k];
        double wall = now_ms() - t0;
        if (i > 0) os = std::min(os, wall);
        if (record && i == REPS) {
            g_spans.clear();
            for (int s = 0; s < slabs; s++) { float g0, g1; cudaEventElapsedTime(&g0, base, es[s][0]); cudaEventElapsedTime(&g1, base, es[s][1]);
                                              g_spans.push_back({ 0, s, g0, g1 }); g_spans.push_back({ 1, s, cspan[s].first, cspan[s].second }); }
        }
    }
    r.overlap = os;
    double lo = std::min(r.gpu_only, r.cpu_only);
    r.hidden_pct = lo >= 1.0 ? 100.0 * (r.serial - r.overlap) / lo : -999;        // -999 = shorter piece under 1 ms: not measurable
    r.err = check_cutoff(total, c, atoms, 200); r.ok = r.err < 1e-4;
    for (int s = 0; s < 64; s++) { cudaEventDestroy(es[s][0]); cudaEventDestroy(es[s][1]); }
    cudaEventDestroy(base); cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(dBin); cudaFree(dCnt); cudaFree(dOut);
    return r;
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &p = dev_props();
    mkdir("stats", 0755); remove("stats/05d_overlap.csv"); remove("stats/05d_overlap_gantt.csv");
    const char *HDR = "section,L_angstrom,bin_edge_A,slabs,overflow_pct,gpu_only_ms,cpu_overflow_only_ms,serial_ms,overlap_ms,ideal_ms,hidden_pct_of_shorter,max_err,check";
    printf("==================== CUTOFF SUMMATION v4: SMALLBIN-OVERLAP (%s) ====================\n", p.name);
    printf("[book] the CPU processes the overflow atoms of subvolume i while the GPU executes the next kernel launch.\n");
    printf("Compute phase only (no binning/upload). Minimum of %d runs after a warm-up (robust to interference). 'ideal' = max(GPU alone, CPU alone).\n", REPS);
    printf("'hidden' = (serial - overlap) / min(GPU alone, CPU alone): 100 %% = the shorter piece is completely hidden; n/a when the\n");
    printf("shorter piece is under 1 ms. Values well outside 0-100 %% are measurement noise (see the note on GPU clocks below).\n\n");

    bool all_ok = true;
    auto row = [&](const char *sec, float L, float be, int slabs, const OvRun &r, bool first) {
        (void)first;
        char hid[16]; if (r.hidden_pct < -900) snprintf(hid, sizeof hid, "n/a"); else snprintf(hid, sizeof hid, "%.0f%%", r.hidden_pct);
        printf("%-6.0f %-8.2f %-6d %-9.1f%% | %-10.2f %-12.2f | %-10.2f %-10.2f %-9.2f | %-9s | %.1e %s\n", L, be, slabs, r.ovf_pct, r.gpu_only, r.cpu_only, r.serial, r.overlap, r.ideal, hid, r.err, r.ok ? "PASS" : "FAIL");
        csv_row("stats/05d_overlap.csv", HDR, "%s,%.0f,%.2f,%d,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.2e,%s", sec, L, be, slabs, r.ovf_pct, r.gpu_only, r.cpu_only, r.serial, r.overlap, r.ideal, r.hidden_pct, r.err, r.ok ? "PASS" : "FAIL");
        all_ok &= r.ok;
        hot_note();
    };
    const char *TH = "%-6s %-8s %-6s %-10s | %-10s %-12s | %-10s %-10s %-9s | %-9s | %s\n";

    printf("---- 1. Volume ladder, bin edge 4 A (about 7 %% of the atoms overflow), 8 slabs ----\n");
    printf(TH, "L (A)", "edge A", "slabs", "overflow", "GPU alone", "CPU alone", "serial", "overlap", "ideal", "hidden", "max err");
    float Ls[3] = { 20, 40, 80 };
    for (float L : Ls) {
        Cut c = make_cut(L, 0.5f, 8.0f, 4.0f, 0.1); auto atoms = make_atoms_box(c, 100 + (unsigned)L);
        row("ladder", L, 4.0f, 8, run_case(c, atoms, 8, false), true);
    }
    printf("\n---- 2. How much overflow? Bin edge sweep at L = 40 A, 8 slabs ----\n");
    printf("[book] overlap pays when the CPU overflow work is comparable to the GPU work; if either dominates, the shorter one is all that can be hidden.\n");
    printf(TH, "L (A)", "edge A", "slabs", "overflow", "GPU alone", "CPU alone", "serial", "overlap", "ideal", "hidden", "max err");
    {
        float edges[6] = { 3.0f, 3.25f, 3.5f, 3.75f, 4.0f, 4.5f };
        for (float be : edges) {
            Cut c = make_cut(40, 0.5f, 8.0f, be, 0.1); auto atoms = make_atoms_box(c, 140);
            row("edge_sweep", 40, be, 8, run_case(c, atoms, 8, false), true);
        }
    }
    printf("\n---- 3. How many slabs? L = 40 A, bin edge 3.75 A ----\n");
    printf(TH, "L (A)", "edge A", "slabs", "overflow", "GPU alone", "CPU alone", "serial", "overlap", "ideal", "hidden", "max err");
    {
        Cut c = make_cut(40, 0.5f, 8.0f, 3.75f, 0.1); auto atoms = make_atoms_box(c, 140);
        int sl[5] = { 1, 2, 4, 8, 16 };
        for (int s : sl) {
            OvRun r = run_case(c, atoms, s, s == 8);
            row("slab_sweep", 40, 3.75f, s, r, true);
            if (s == 8) {
                FILE *f = fopen("stats/05d_overlap_gantt.csv", "w"); fprintf(f, "engine,slab,t_start_ms,t_end_ms\n");
                for (auto &sp : g_spans) fprintf(f, "%s,%d,%.4f,%.4f\n", std::get<0>(sp) == 0 ? "GPU_kernel" : "CPU_overflow", std::get<1>(sp), std::get<2>(sp), std::get<3>(sp));
                fclose(f);
            }
        }
    }
    printf("\nHow to read:\n");
    printf("  * Overlap can never beat 'ideal'. If CPU alone >> GPU alone, overlap hides only the GPU time; the CPU pass sets the total.\n");
    printf("  * The book's setting has a fast CPU overflow pass (SSE) and few overflow atoms, so the GPU dominates and the CPU pass is\n");
    printf("    fully hidden. This CPU code is scalar, so at 4 A the CPU dominates instead; the edge sweep finds where they balance.\n");
    printf("  * One slab = no overlap possible (the CPU pass has to wait for the whole launch); more slabs let the CPU start earlier.\n");
    printf("  * The timeline of section 3 (8 slabs) is in stats/05d_overlap_gantt.csv.\n");
    printf("  * Clock effect: while the CPU works the GPU idles and drops to a low clock (the notes above show 360-435 MHz), so the\n");
    printf("    next short kernel starts slow. Each timing group is primed with ~150 ms of GPU work and the minimum of %d runs is\n", REPS);
    printf("    kept, but the overlap pipeline itself keeps re-entering that idle state; treat single-digit-ms differences with care.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

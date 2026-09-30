/*
 * 02_gpu_monitor.cu   -- GPU SYSTEM MONITORING while the DCS workload runs
 *
 * This program re-uses the kernels and host driver of 01_dcs_compute.cu (#include with DCS_NO_MAIN)
 * and reports what the GPU is doing, not what it computed:
 *
 *   A. Device inventory and theoretical peaks
 *   B. STORAGE ledger: registers, shared, constant, local (spill), L2, global DRAM - capacity vs use
 *   C. THREAD ledger per kernel: grid/block/warps/launches, blocks per SM, occupancy, waves, limiter
 *   D. TRAFFIC ledger (derived from launch counts): constant loads, global read/write bytes,
 *      host-to-device bytes, arithmetic intensity
 *   E. LIVE NVML sampling during each kernel's full-scale run: utilisation, clocks, power, temperature,
 *      device memory in use (sampled every 50 ms from a separate thread)
 *   F. Achieved throughput vs theoretical peak
 *
 * Provenance labels: [API] read from the CUDA runtime/NVML, [COUNT] exact from launch parameters,
 * [DERIVED] computed from those, [MEASURED] timed here. NVML metrics that the driver does not
 * report on this system are printed as n/a.
 */
#define DCS_NO_MAIN
#include "01_dcs_compute.cu"
#include "monitor.cuh"

static void stats(const std::vector<Sample> &s, double Sample::*f, double *mn, double *av, double *mx) {
    *mn = 1e30; *mx = -1e30; *av = 0; int n = 0;
    for (auto &x : s) { double v = x.*f; if (v < 0) continue; if (v < *mn) *mn = v; if (v > *mx) *mx = v; *av += v; n++; }
    if (!n) { *mn = *av = *mx = -1; } else *av /= n;
}
static void pr(const char *label, double mn, double av, double mx, const char *unit) {
    if (av < 0) printf("    %-26s n/a (not reported by the driver)\n", label);
    else printf("    %-26s min %9.1f  avg %9.1f  max %9.1f  %s\n", label, mn, av, mx, unit);
}

int main() {
    CUDA_CHECK(cudaFree(0));                                   // create the context before any measurement
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int clk_khz = 0, memclk_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, 0);
    Scale full = { "FULL", 100, 100, 100, 100000 };
    Geo g = make_geo(full);
    auto atoms = make_atoms(full, 3);

    printf("==================== GPU SYSTEM MONITOR: DCS FULL-SCALE RUN ====================\n");
    printf("Workload: 100x100x100 lattice, 100,000 atoms (kernels from 01_dcs_compute.cu)\n\n");

    // ---------------------------------------------------------------- A
    double fp32_peak = (double)p.multiProcessorCount * 64 * 2 * clk_khz * 1e3 / 1e9;   // GFLOPS at the reported clock
    double dram_peak = 2.0 * memclk_khz * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
    printf("---- A. Device inventory [API] ----\n");
    printf("  %s | compute capability %d.%d | %d SMs x 64 FP32 lanes = %d CUDA cores | warp %d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.multiProcessorCount * 64, p.warpSize);
    printf("  rated SM clock %.0f MHz (attribute), memory clock %.0f MHz, bus %d bit\n", clk_khz / 1e3, memclk_khz / 1e3, p.memoryBusWidth);
    printf("  theoretical FP32 peak at the rated clock : %.0f GFLOPS [DERIVED: SMs x 64 x 2 x clock]\n", fp32_peak);
    printf("  theoretical DRAM bandwidth               : %.0f GB/s   [DERIVED: 2 x mem clock x bus/8]\n\n", dram_peak);

    // ---------------------------------------------------------------- B (part 1) memory before/after allocation
    size_t free0, total0, free1, total1;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));
    float *d; CUDA_CHECK(cudaMalloc(&d, g.total * 4));
    CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
    size_t sym = 0; CUDA_CHECK(cudaGetSymbolSize(&sym, atominfo));
    const void *kern[3] = { (const void *)dcs_v1, (const void *)dcs_v2, (const void *)dcs_v3 };
    cudaFuncAttributes fa[3];
    for (int v = 0; v < 3; v++) CUDA_CHECK(cudaFuncGetAttributes(&fa[v], kern[v]));

    printf("---- B. STORAGE ledger ----\n");
    printf("  %-24s %-22s %-24s %s\n", "level", "capacity", "used by this program", "note");
    printf("  %-24s %-22s %-24s %s\n", "registers (per SM)", "65,536 x 4 B = 256 KB", "see per-kernel table below", "[API]");
    printf("  %-24s %-22zu %-24d %s\n", "shared memory (per SM)", (size_t)p.sharedMemPerMultiprocessor, (int)fa[0].sharedSizeBytes + (int)fa[1].sharedSizeBytes + (int)fa[2].sharedSizeBytes, "bytes; kernels use none (atoms come from constant memory) [API]");
    printf("  %-24s %-22zu %-24zu %s\n", "constant memory (GPU)", (size_t)p.totalConstMem, sym, "bytes = atominfo[4000] float4; the 64 KB limit forces atom chunking [API]");
    printf("  %-24s %-22d %-24d %s\n", "L2 cache (GPU)", p.l2CacheSize, 0, "bytes; shared by all SMs, not directly allocated [API]");
    printf("  %-24s %-22.1f %-24.2f %s\n", "global DRAM (GPU) MB", total1 / 1048576.0, (free0 - free1) / 1048576.0, "MB = the energy grid (padded) [API cudaMemGetInfo]");
    printf("  %-24s %-22s %-24.2f %s\n", "  DRAM free before/after", "", free1 / 1048576.0, "MB free after allocation");
    printf("  padded grid buffer = %d x %d x %d floats = %.2f MB (valid %d x %d x %d = %.2f MB; padding wastes %.0f%%) [COUNT]\n\n",
           g.NX, g.NY, full.gz, g.total * 4 / 1048576.0, full.gx, full.gy, full.gz,
           (double)full.gx * full.gy * full.gz * 4 / 1048576.0, 100.0 * (1.0 - (double)full.gx * full.gy / g.slice));

    // ---------------------------------------------------------------- C
    printf("---- C. THREAD ledger per kernel (FULL scale) ----\n");
    long launches = (long)full.gz * ((full.natoms + MAXATOMS - 1) / MAXATOMS);
    int nb[3];
    for (int v = 0; v < 3; v++) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[v], kern[v], 256, 0);
    printf("  %-38s %-14s %-9s %-9s %-10s %-9s %s\n", "kernel", "grid x block", "blocks", "threads", "warps", "launches", "total threads launched");
    for (int v = 0; v < 3; v++) {
        int gxb = g.NX / (16 * VER[v].ppt), gyb = g.NY / 16; long blocks = (long)gxb * gyb;
        char gb[32]; snprintf(gb, sizeof gb, "%dx%d x 16x16", gxb, gyb);
        printf("  %-38s %-14s %-9ld %-9ld %-10ld %-9ld %.3e\n", VER[v].name, gb, blocks, blocks * 256, blocks * 8, launches, (double)blocks * 256 * launches);
    }
    printf("  (per launch; 8 warps per 256-thread block) [COUNT]\n\n");
    printf("  %-38s %-6s %-11s %-10s %-11s %-9s %-9s %s\n", "kernel", "regs", "regs/block", "blocks/SM", "warps/SM", "occup.", "waves", "limiter");
    for (int v = 0; v < 3; v++) {
        int gxb = g.NX / (16 * VER[v].ppt), gyb = g.NY / 16; long blocks = (long)gxb * gyb;
        int regs = fa[v].numRegs, rb = regs * 256;
        int lim_thr = p.maxThreadsPerMultiProcessor / 256, lim_blk = p.maxBlocksPerMultiProcessor, lim_reg = p.regsPerMultiprocessor / rb;
        int m = lim_thr < lim_blk ? lim_thr : lim_blk; m = m < lim_reg ? m : lim_reg;
        const char *why = (lim_reg == m && lim_reg < lim_thr) ? "registers" : "thread slots";
        printf("  %-38s %-6d %-11d %-10d %-11d %6.1f%%   %-9.2f %s (model %d, runtime %d)\n", VER[v].name, regs, rb, nb[v],
               nb[v] * 8, 100.0 * nb[v] * 256 / p.maxThreadsPerMultiProcessor,
               (double)blocks / ((double)p.multiProcessorCount * nb[v]), why, m, nb[v]);
    }
    printf("\n  LAUNCH FILL: per-SM occupancy above is what one SM CAN hold; a launch (one z-slice x one atom chunk)\n");
    printf("  may not have enough blocks to use every SM. [COUNT]\n");
    printf("  %-38s %-9s %-10s %-24s %s\n", "kernel", "blocks", "SMs busy", "avg resident warps/SM", "of 32-warp capacity");
    for (int v = 0; v < 3; v++) {
        int gxb = g.NX / (16 * VER[v].ppt), gyb = g.NY / 16; long blocks = (long)gxb * gyb;
        long resident = blocks < (long)p.multiProcessorCount * nb[v] ? blocks : (long)p.multiProcessorCount * nb[v];
        printf("  %-38s %-9ld %2ld of %-5d %-24.1f %.1f%%\n", VER[v].name, blocks,
               blocks < p.multiProcessorCount ? blocks : (long)p.multiProcessorCount, p.multiProcessorCount,
               resident * 8.0 / p.multiProcessorCount, 100.0 * resident * 8.0 / p.multiProcessorCount / 32);
    }
    printf("  local (spill) memory per thread: v1 %zu B, v2 %zu B, v3 %zu B; static smem: %zu/%zu/%zu B; const in kernel: %zu/%zu/%zu B [API]\n\n",
           fa[0].localSizeBytes, fa[1].localSizeBytes, fa[2].localSizeBytes, fa[0].sharedSizeBytes, fa[1].sharedSizeBytes,
           fa[2].sharedSizeBytes, fa[0].constSizeBytes, fa[1].constSizeBytes, fa[2].constSizeBytes);

    // ---------------------------------------------------------------- D
    double comp_evals = (double)g.slice * full.natoms * full.gz, valid = (double)full.gx * full.gy * full.gz * full.natoms;
    double gbytes = (double)launches * g.slice * 4 * 2;           // each thread-point: read + write once per chunk
    printf("---- D. TRAFFIC ledger (FULL, one complete run per kernel) [COUNT/DERIVED] ----\n");
    printf("  useful atom-point evaluations   : %.3e\n", valid);
    printf("  computed incl. padding          : %.3e  (+%.0f%%)\n", comp_evals, 100.0 * (comp_evals / valid - 1));
    printf("  host -> device (atom chunks)    : %.1f MB over %ld cudaMemcpyToSymbol calls (identical for all kernels)\n",
           (double)launches * MAXATOMS * 16 / 1e6, launches);
    printf("  global DRAM read+write of grid  : %.1f MB (each chunk launch does grid += partial sum)\n", gbytes / 1e6);
    printf("  %-38s %-22s %-24s\n", "kernel", "const loads / eval", "FP work per byte of DRAM");
    for (int v = 0; v < 3; v++)
        printf("  %-38s %-22.3f %.0f flop/byte (at %.0f flops/eval)\n", VER[v].name, 1.0 / VER[v].ppt, comp_evals * FLOPS_PER_EVAL / gbytes, FLOPS_PER_EVAL);
    printf("  Read: thousands of flops per DRAM byte means DRAM is not the bottleneck; the constant cache\n");
    printf("  broadcast serves every atom fetch (one float4 per atom shared by the whole warp).\n\n");

    // ---------------------------------------------------------------- E
    Sampler smp;
    printf("---- E. LIVE monitoring during the full-scale run [NVML, 50 ms sampling] ----\n");
    if (!smp.ok) printf("  NVML could not be initialised; live metrics unavailable.\n");
    // warm-up on a tiny problem so CUDA context and kernel loading is not counted
    { Scale t = { "warm", 32, 32, 2, 500 }; Geo tg = make_geo(t); auto ta = make_atoms(t, 9); float *td; cudaMalloc(&td, tg.total * 4);
      for (int v = 0; v < 3; v++) run_gpu(v, tg, ta, td); cudaFree(td); }
    Run runs[3]; double avg_clk[3] = { 0, 0, 0 };
    if (smp.ok) { smp.start(); std::this_thread::sleep_for(std::chrono::milliseconds(300)); smp.stop();
        double a, b, c; stats(smp.s, &Sample::gpu, &a, &b, &c);
        printf("  idle baseline (300 ms): %zu samples, GPU util avg %.0f%%, SM clock %.0f MHz, power %.1f W, temp %.0f C, %.0f MiB in use\n\n",
               smp.s.size(), b, smp.s.back().sm_mhz, smp.s.back().power_w, smp.s.back().temp, smp.s.back().used_mib); }
    for (int v = 0; v < 3; v++) {
        if (smp.ok) smp.start();
        runs[v] = run_gpu(v, g, atoms, d);
        if (smp.ok) smp.stop();
        printf("  [%s] wall %.3f s [MEASURED]\n", VER[v].name, runs[v].ms / 1e3);
        if (!smp.ok) continue;
        double mn, av, mx;
        stats(smp.s, &Sample::gpu, &mn, &av, &mx);   pr("GPU utilisation", mn, av, mx, "%");
        stats(smp.s, &Sample::mem, &mn, &av, &mx);   pr("memory-controller util", mn, av, mx, "%");
        stats(smp.s, &Sample::sm_mhz, &mn, &av, &mx); pr("SM clock", mn, av, mx, "MHz"); avg_clk[v] = av;
        stats(smp.s, &Sample::mem_mhz, &mn, &av, &mx); pr("memory clock", mn, av, mx, "MHz");
        stats(smp.s, &Sample::power_w, &mn, &av, &mx); pr("power draw", mn, av, mx, "W");
        stats(smp.s, &Sample::used_mib, &mn, &av, &mx); pr("device memory in use", mn, av, mx, "MiB (whole GPU incl. Windows)");
        printf("    temperature %.0f -> %.0f C, %zu samples\n", smp.s.front().temp, smp.s.back().temp, smp.s.size());
        printf("    timeline (t ms | util%% | memctl%% | SM MHz | power W | temp C):\n");
        size_t n = smp.s.size(), step = n > 6 ? n / 6 : 1;
        for (size_t i = 0; i < n; i += step)
            printf("      %7.0f | %4.0f | %5.0f | %6.0f | %7.1f | %5.0f\n", smp.s[i].t_ms, smp.s[i].gpu, smp.s[i].mem, smp.s[i].sm_mhz, smp.s[i].power_w, smp.s[i].temp);
        printf("\n");
        ensure_dir("stats");
        write_samples_csv("stats/02_gpu_monitor_timeline.csv", VER[v].name, smp.s);
    }

    // ---------------------------------------------------------------- F
    printf("---- F. Achieved vs theoretical peak [MEASURED / DERIVED] ----\n");
    printf("  %-38s %-10s %-12s %-12s %-14s %s\n", "kernel", "seconds", "G evals/s", "GFLOPS", "% of FP32 peak", "SM clock used");
    for (int v = 0; v < 3; v++) {
        double rate = valid / (runs[v].ms * 1e-3), gf = rate * FLOPS_PER_EVAL / 1e9;
        double pk = avg_clk[v] > 0 ? (double)p.multiProcessorCount * 64 * 2 * avg_clk[v] * 1e6 / 1e9 : fp32_peak;
        printf("  %-38s %-10.3f %-12.3f %-12.1f %-14.1f %s\n", VER[v].name, runs[v].ms / 1e3, rate / 1e9, gf, 100.0 * gf / pk,
               avg_clk[v] > 0 ? "measured avg (NVML)" : "rated clock");
    }
    printf("  Caveats: GFLOPS assumes 10 flops per evaluation for every kernel (the book's convention). v2/v3 do\n");
    printf("  fewer real flops per evaluation (dy^2+dz^2 is shared), so their %% of peak can look inflated; the\n");
    printf("  evaluations/second column is the fair comparison. NVML power/util may be coarse on WSL.\n");
    printf("\nDONE\n");
    cudaFree(d);
    return 0;
}

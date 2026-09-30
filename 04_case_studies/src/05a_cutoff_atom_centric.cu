/*
 * 05a_cutoff_atom_centric.cu   -- CUTOFF SUMMATION, version 1: the ATOM-CENTRIC strategy (and why the book rejects it)
 *
 * Book (Ch.10): with a cutoff radius rc an atom only affects lattice points inside a sphere around it, which makes the
 * cost O(Volume) instead of O(Volume^2). The natural sequential algorithm is ATOM-CENTRIC: for each atom, find the
 * lattice points within rc and add its contribution to them. On a GPU one thread per atom means MANY threads update the
 * SAME lattice point (scatter), which needs atomicAdd on global memory; the book prefers grid-centric (gather) kernels.
 *
 * This program measures, over a ladder of volumes (same atom density, so atoms grow with volume):
 *   CPU  atom-centric (sequential, float)                    the book's baseline algorithm (measured once, cached)
 *   GPU  atom-centric (one thread per atom, atomicAdd)       what the book advises against
 *   GPU  DirectSum + cutoff test (grid-centric, ALL atoms)   visits every atom: O(Volume^2), so only run up to L = 80
 * The bin-based grid-centric versions (LargeBin, SmallBin, SmallBin-Overlap) are 05b, 05c, 05d.
 * GPU kernels are launched in short chunks (atoms per launch) so no single launch approaches the Windows GPU watchdog limit.
 * Every result is checked against a double-precision CPU reference at 200 sampled lattice points.
 * Statistics: stats/05a_cutoff.csv; cached CPU times: stats/cpu_baseline.csv.
 */
#include "common_bins.cuh"
#include <sys/stat.h>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// one thread per atom (atoms [a0, a1) of the list): scatter its contribution to every lattice point inside the cutoff sphere
__global__ void cutoff_atom_centric_kernel(const float4 *atoms, int a0, int a1, float h, float rc, int gx, int gy, int gz, float *out) {
    int a = a0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= a1) return;
    float4 A = atoms[a]; float rc2 = rc * rc;
    int i0 = max(0, (int)ceilf((A.x - rc) / h)), i1 = min(gx - 1, (int)floorf((A.x + rc) / h));
    int j0 = max(0, (int)ceilf((A.y - rc) / h)), j1 = min(gy - 1, (int)floorf((A.y + rc) / h));
    int k0 = max(0, (int)ceilf((A.z - rc) / h)), k1 = min(gz - 1, (int)floorf((A.z + rc) / h));
    for (int k = k0; k <= k1; k++) { float dz = h * k - A.z, dz2 = dz * dz;
        for (int j = j0; j <= j1; j++) { float dy = h * j - A.y, dyz2 = dy * dy + dz2;
            for (int i = i0; i <= i1; i++) { float dx = h * i - A.x, r2 = dx * dx + dyz2;
                if (r2 < rc2) atomicAdd(&out[((size_t)k * gy + j) * gx + i], A.w * rsqrtf(r2)); } } }
}
static void scatter_run(const float4 *dA, int n, const Cut &c, float *dOut, int chunk = 16384) {
    CUDA_CHECK(cudaMemset(dOut, 0, npoints(c) * 4));
    for (int a0 = 0; a0 < n; a0 += chunk) {
        int a1 = std::min(n, a0 + chunk);
        cutoff_atom_centric_kernel<<<(a1 - a0 + 127) / 128, 128>>>(dA, a0, a1, c.h, c.rc, c.gx, c.gy, c.gz, dOut);
    }
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &p = dev_props();
    mkdir("stats", 0755); remove("stats/05a_cutoff.csv");
    const char *HDR = "algorithm,L_angstrom,volume_A3,points,atoms,ms,max_err,check";
    printf("==================== CUTOFF SUMMATION v1: ATOM-CENTRIC (%s) ====================\n", p.name);
    printf("lattice spacing 0.5 A, cutoff 8 A, atom density 0.1 per A^3 (a solvated-molecule-like density), charges in [-1,1].\n");
    printf("GPU variants: 1 warm-up + 5 timed runs (mean). The sequential CPU baseline is measured once (minimum of repeated runs)\n");
    printf("and cached in stats/cpu_baseline.csv so the run is dominated by GPU work; 'n/a' = not measured (too slow to be useful).\n");
    printf("Error = worst |GPU - double CPU| / sum(|q|/r) over 200 sampled lattice points; PASS if < 1e-4.\n\n");

    float Ls[5] = { 10, 20, 40, 80, 120 };
    printf("%-6s %-12s %-10s %-8s | %-11s %-14s %-13s | %-9s %-11s | %s\n", "L (A)", "volume A^3", "points", "atoms", "CPU ms", "GPU scatter ms", "GPU direct ms",
           "scat/CPU", "direct/scat", "max err (scatter | direct)");
    bool all_ok = true;
    std::vector<double> ts_all, td_all; std::vector<float> Lused;
    for (float L : Ls) {
        cooldown();                                                   // start every scale from a comparable GPU temperature
        Cut c = make_cut(L, 0.5f, 8.0f, 4.0f, 0.1);
        auto atoms = make_atoms_box(c, 100 + (unsigned)L);
        size_t np = npoints(c);
        double cpu_ms = cpu_baseline_ms(L);
        if (cpu_ms < 0 && L <= 80) {                               // measure once, cache
            std::vector<float> cpu(np); int reps = L <= 20 ? 10 : (L <= 40 ? 3 : 2); cpu_ms = 1e30;
            for (int r = 0; r < reps; r++) { std::fill(cpu.begin(), cpu.end(), 0.f); double t0 = now_ms(); cpu_atom_centric(atoms, c, cpu, 0, c.gz); cpu_ms = std::min(cpu_ms, now_ms() - t0); }
            csv_row("stats/cpu_baseline.csv", "algorithm,L_angstrom,volume_A3,points,atoms,ms", "cpu_atom_centric,%.0f,%.0f,%zu,%d,%.3f", L, volume(c), np, c.natoms, cpu_ms);
        }
        float4 *dA; float *dS, *dD;
        CUDA_CHECK(cudaMalloc(&dA, atoms.size() * sizeof(float4))); CUDA_CHECK(cudaMalloc(&dS, np * 4)); CUDA_CHECK(cudaMalloc(&dD, np * 4));
        CUDA_CHECK(cudaMemcpy(dA, atoms.data(), atoms.size() * sizeof(float4), cudaMemcpyHostToDevice));
        int n = (int)atoms.size(); bool do_direct = L <= 80;
        float t[2] = { 0, 0 };
        bench_rr(do_direct ? 2 : 1, [&](int v) { if (v == 0) scatter_run(dA, n, c, dS); else direct_cutoff_run(dA, n, c, dD); }, 5, t);
        std::vector<float> hs(np), hd(np);
        CUDA_CHECK(cudaMemcpy(hs.data(), dS, np * 4, cudaMemcpyDeviceToHost));
        double inside, es = check_cutoff(hs, c, atoms, 200, &inside), ed = 0;
        if (do_direct) { CUDA_CHECK(cudaMemcpy(hd.data(), dD, np * 4, cudaMemcpyDeviceToHost)); ed = check_cutoff(hd, c, atoms, 200); }
        bool ok = es < 1e-4 && ed < 1e-4; all_ok &= ok;
        char cpus[24], dirs[24], r1[24], r2[24];
        if (cpu_ms > 0) snprintf(cpus, sizeof cpus, "%.1f", cpu_ms); else snprintf(cpus, sizeof cpus, "n/a");
        if (do_direct) snprintf(dirs, sizeof dirs, "%.2f", t[1]); else snprintf(dirs, sizeof dirs, "skipped");
        if (cpu_ms > 0) snprintf(r1, sizeof r1, "%.1f", cpu_ms / t[0]); else snprintf(r1, sizeof r1, "n/a");
        if (do_direct) snprintf(r2, sizeof r2, "%.1f", t[1] / t[0]); else snprintf(r2, sizeof r2, "n/a");
        printf("%-6.0f %-12.0f %-10zu %-8d | %-11s %-14.2f %-13s | %-9s %-11s | %.1e | %.1e %s\n", L, volume(c), np, n, cpus, t[0], dirs, r1, r2, es, ed, ok ? "PASS" : "FAIL");
        csv_row("stats/05a_cutoff.csv", HDR, "gpu_atom_centric,%.0f,%.0f,%zu,%d,%.3f,%.2e,%s", L, volume(c), np, n, t[0], es, es < 1e-4 ? "PASS" : "FAIL");
        if (cpu_ms > 0) csv_row("stats/05a_cutoff.csv", HDR, "cpu_atom_centric,%.0f,%.0f,%zu,%d,%.3f,0,PASS", L, volume(c), np, n, cpu_ms);
        if (do_direct) csv_row("stats/05a_cutoff.csv", HDR, "gpu_directsum,%.0f,%.0f,%zu,%d,%.3f,%.2e,%s", L, volume(c), np, n, t[1], ed, ed < 1e-4 ? "PASS" : "FAIL");
        ts_all.push_back(t[0]); td_all.push_back(do_direct ? t[1] : -1); Lused.push_back(L);
        if (L == 10 || L == 120) {
            printf("  launch fill (L = %.0f, per launch of up to 16384 atoms): ", L); ledger_header();
            ledger_row("  GPU atom-centric", (const void *)cutoff_atom_centric_kernel, 128, std::min(n, 16384) / 128 + 1);
            ledger_row("  GPU DirectSum", (const void *)direct_cutoff_kernel, 256, (long)((np + 255) / 256));
            printf("  atomicAdd calls: about %.2e (mean %.0f atoms inside the cutoff per lattice point x %zu points) [DERIVED]\n", inside * np, inside, np);
        }
        hot_note();
        cudaFree(dA); cudaFree(dS); cudaFree(dD);
    }
    printf("\nMeasured growth per volume step, as an exponent e in time ~ volume^e (e = 1 is linear, e = 2 is quadratic) [MEASURED]:\n");
    printf("  %-16s %-14s %-14s\n", "step", "GPU scatter e", "GPU direct e");
    for (size_t i = 1; i < ts_all.size(); i++) {
        double vr = pow(Lused[i] / Lused[i - 1], 3.0), es = log(ts_all[i] / ts_all[i - 1]) / log(vr);
        char ed[24]; if (td_all[i] > 0 && td_all[i - 1] > 0) snprintf(ed, sizeof ed, "%.2f", log(td_all[i] / td_all[i - 1]) / log(vr)); else snprintf(ed, sizeof ed, "n/a");
        printf("  L %-4.0f -> %-5.0f %-14.2f %-14s\n", Lused[i - 1], Lused[i], es, ed);
    }
    printf("\nHow to read:\n");
    printf("  * Volume grows per row; atoms and lattice points grow with it (fixed density).\n");
    printf("  * [book] direct summation is O(Volume^2) and cutoff summation O(Volume). The exponents above are what this run shows;\n");
    printf("    small volumes deviate because spheres are clipped by the domain faces (less work per atom) and the GPU is\n");
    printf("    under-filled, so judge the trend on the LAST steps.\n");
    printf("  * 'scat/CPU' is the GPU scatter speedup over the sequential CPU atom-centric code; 'direct/scat' > 1 means DirectSum\n");
    printf("    is slower than scatter at that size.\n");
    printf("  * The atoms are placed at random, so few threads update the same lattice point at the same moment; the book's\n");
    printf("    atomicAdd contention worry is therefore mild in this data. It is judged against the bin versions in 05b-05d.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

/*
 * 08a_mri_fhd_ladder.cu   -- ADVANCED MRI RECONSTRUCTION (book Ch.8): computing F^H d, the optimisation ladder
 *
 * F^H d is the matrix-vector product that dominates iterative MRI reconstruction: for every image voxel n, sum over all M
 * non-Cartesian k-space samples of  mu_m * exp(i*2*pi*k_m.x_n). F (M x N, ~6e11 complex entries at the book's size) cannot be
 * stored, so every entry is recomputed with a sine and a cosine. The book's steps, reproduced here as versions v1..v5:
 *   0  loop fission: mu = conj(Phi) d is computed once per sample (cmp_mu_kernel)
 *   v1 naive: one thread per voxel (loop interchange removes write conflicts); voxel data and accumulators go through global memory
 *   v2 registers: x,y,z and the two accumulators stay in registers (global accesses per iteration 14 -> 7 in the book)
 *   v3 constant memory: k-space (kx,ky,kz) streamed in chunks through __constant__ memory as three separate arrays (64 KB limit)
 *   v4 array of structs {x,y,z} in constant memory (fewer cache entries per iteration)
 *   v5 hardware sine/cosine (__sinf/__cosf on the special function units), argument reduced to one cycle first
 * Scales: REDUCED = 64^3 voxels x 8,192 samples (all versions timed); FULL = the book's 128^3 = 2,097,152 voxels x 284,592
 * samples (v5 only; about 6e11 voxel-sample pairs), checked at sampled voxels against a double-precision CPU reference.
 * GFLOPS uses the book's convention of 13 floating-point operations per voxel-sample pair (trig counted inside those 13).
 * Statistics: stats/08a_mri_ladder.csv
 */
#include "common_mri.cuh"
#include <sys/stat.h>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// sequential CPU code of book Fig 8.4B (float, libm sin/cos), used only to obtain a CPU pair rate
static double cpu_pair_rate(const Mri &p) {
    std::vector<float> rF(p.N, 0.f), iF(p.N, 0.f);
    double t0 = now_ms();
    for (int m = 0; m < p.M; m++)
        for (int n = 0; n < p.N; n++) {
            float e = TWO_PI_F * (p.kx[m] * p.x[n] + p.ky[m] * p.y[n] + p.kz[m] * p.z[n]), c = cosf(e), s = sinf(e);
            rF[n] += p.rMu[m] * c - p.iMu[m] * s; iF[n] += p.iMu[m] * c + p.rMu[m] * s;
        }
    volatile float sink = rF[0] + iF[p.N - 1]; (void)sink;
    return (double)p.M * p.N / ((now_ms() - t0) * 1e-3);
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &pr = dev_props();
    mkdir("stats", 0755); remove("stats/08a_mri_ladder.csv");
    const char *HDR = "version,scale,voxels,samples,ms,gpairs_per_s,gflops_13,err_rel_rms,check";
    printf("==================== MRI F^H d: THE OPTIMISATION LADDER (%s) ====================\n", pr.name);
    printf("[book] G80 results for the full problem: naive 5.4 GFLOPS (41 min), constant memory + registers 22.8 GFLOPS (9.8 min), fully optimised 144.5 GFLOPS (1.5 min).\n");
    printf("Error = worst |GPU - double CPU| over 200 sampled voxels, relative to the RMS magnitude of the reference values (PASS if < 1e-3 for accurate\n");
    printf("trigonometry, < 5e-3 for hardware trigonometry). Reduced scale: mean of 5 timed runs after 1 warm-up.\n\n");

    // ---------------------------------------------------------------- step 0: loop fission (cmpMu)
    {
        int M = 65536; std::vector<float> rP(M), iP(M), rD(M), iD(M), rMu(M), iMu(M); srand(5);
        for (int m = 0; m < M; m++) { rP[m] = rand() / (float)RAND_MAX - 0.5f; iP[m] = rand() / (float)RAND_MAX - 0.5f; rD[m] = rand() / (float)RAND_MAX; iD[m] = rand() / (float)RAND_MAX; }
        float *a, *b, *c, *d, *e, *f; size_t B = M * 4;
        CUDA_CHECK(cudaMalloc(&a, B)); CUDA_CHECK(cudaMalloc(&b, B)); CUDA_CHECK(cudaMalloc(&c, B)); CUDA_CHECK(cudaMalloc(&d, B)); CUDA_CHECK(cudaMalloc(&e, B)); CUDA_CHECK(cudaMalloc(&f, B));
        cudaMemcpy(a, rP.data(), B, cudaMemcpyHostToDevice); cudaMemcpy(b, iP.data(), B, cudaMemcpyHostToDevice); cudaMemcpy(c, rD.data(), B, cudaMemcpyHostToDevice); cudaMemcpy(d, iD.data(), B, cudaMemcpyHostToDevice);
        cmp_mu_kernel<<<(M + 255) / 256, 256>>>(a, b, c, d, e, f, M); CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(rMu.data(), e, B, cudaMemcpyDeviceToHost); cudaMemcpy(iMu.data(), f, B, cudaMemcpyDeviceToHost);
        double worst = 0; for (int m = 0; m < M; m++) worst = std::max(worst, std::max((double)fabs(rMu[m] - (rP[m] * rD[m] + iP[m] * iD[m])), (double)fabs(iMu[m] - (rP[m] * iD[m] - iP[m] * rD[m]))));
        printf("---- 0. Loop fission: mu = conj(Phi) d on the GPU, %d samples: max |GPU - CPU| = %.1e  %s ----\n\n", M, worst, worst < 1e-6 ? "PASS" : "FAIL");
        cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(d); cudaFree(e); cudaFree(f);
    }

    // ---------------------------------------------------------------- reduced scale ladder
    Mri p = make_mri(8192, 64, 11);              // 262,144 voxels = 1024 blocks: enough to fill the GPU (32^3 voxels did not)
    MriDev d = mri_upload(p);
    if (getenv("PROFILE_ONLY")) {                // used by scripts/profile_kernels.sh: one pass of every version, nothing else
        for (int v = 1; v <= 5; v++) mri_run(v, p, d);
        cudaDeviceSynchronize(); printf("profile-only run done\n"); return 0;
    }
    double cpu_rate = cpu_pair_rate(make_mri(2048, 12, 7));
    printf("---- 1. Reduced scale: %d voxels (%d^3) x %d samples = %.2e pairs; sequential CPU (Fig 8.4B, 1 core): %.3f G pairs/s ----\n", p.N, p.S, p.M, (double)p.N * p.M, cpu_rate / 1e9);
    printf("%-34s %-9s %-11s %-10s %-10s %-11s %-9s | %-9s %s\n", "version", "ms", "G pairs/s", "GFLOPS", "vs v1", "vs CPU", "regs", "err", "");
    const char *names[5] = { "v1 naive (global memory)", "v2 registers", "v3 constant memory, 3 arrays", "v4 constant memory, struct", "v5 + hardware sin/cos" };
    bool all_ok = true; double t1 = 0; float tt[5];
    cooldown();
    { double t0 = now_ms(); while (now_ms() - t0 < 400) { mri_run(5, p, d); cudaDeviceSynchronize(); } }   // prime: an idle GPU runs at a low clock
    bench_rr(5, [&](int v) { mri_run(v + 1, p, d); }, 5, tt);
    std::vector<float> rF(p.N), iF(p.N);
    const void *kern[5] = { (const void *)fhd_v1, (const void *)fhd_v2, (const void *)fhd_v3, (const void *)fhd_v4, (const void *)fhd_v5<1> };
    for (int v = 0; v < 5; v++) {
        mri_run(v + 1, p, d); cudaDeviceSynchronize();
        cudaMemcpy(rF.data(), d.rF, (size_t)p.N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(iF.data(), d.iF, (size_t)p.N * 4, cudaMemcpyDeviceToHost);
        double err = check_fhd(p, rF, iF, 200), tol = v == 4 ? 5e-3 : 1e-3; bool ok = err < tol; all_ok &= ok;
        double pairs = (double)p.N * p.M / (tt[v] * 1e-3); if (v == 0) t1 = tt[v];
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, kern[v]);
        printf("%-34s %-9.2f %-11.2f %-10.1f %-10.2f %-11.1f %-9d | %-9.1e %s\n", names[v], tt[v], pairs / 1e9, pairs * 13 / 1e9, t1 / tt[v], pairs / cpu_rate, fa.numRegs, err, ok ? "PASS" : "FAIL");
        csv_row("stats/08a_mri_ladder.csv", HDR, "\"%s\",reduced,%d,%d,%.4f,%.4f,%.2f,%.2e,%s", names[v], p.N, p.M, tt[v], pairs / 1e9, pairs * 13 / 1e9, err, ok ? "PASS" : "FAIL");
    }
    printf("  'vs CPU' is the ratio of pair rates (GPU pairs/s over the single-core sequential rate). Book reference for the ladder shape: each step\n");
    printf("  should be faster than the one before; the size of each step on this GPU is what the table shows.\n\n");
    mri_free(d);

    // ---------------------------------------------------------------- full scale, v5
    printf("---- 2. FULL scale (the book's benchmark): 128^3 = 2,097,152 voxels x 284,592 samples, version v5 ----\n");
    Mri big = make_mri(284592, 128, 21);
    MriDev bd = mri_upload(big);
    double pairs_full = (double)big.N * big.M;
    cooldown();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    { Mri tiny = make_mri(2048, 32, 1); mri_run(5, tiny, bd); cudaDeviceSynchronize(); }     // warm-up: kernel load and clocks (uses the first 32^3 voxels)
    cudaEventRecord(e0);
    mri_run(5, big, bd); cudaEventRecord(e1); cudaEventSynchronize(e1); CUDA_CHECK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    std::vector<float> bR(big.N), bI(big.N);
    cudaMemcpy(bR.data(), bd.rF, (size_t)big.N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(bI.data(), bd.iF, (size_t)big.N * 4, cudaMemcpyDeviceToHost);
    double rms, err = check_fhd(big, bR, bI, 32, &rms); bool ok = err < 5e-3; all_ok &= ok;
    double gp = pairs_full / (ms * 1e-3) / 1e9;
    printf("  pairs: %.3e | time: %.2f s | %.2f G pairs/s | %.1f GFLOPS (13 per pair) | launches: %d chunks of %d samples\n", pairs_full, ms / 1e3, gp, gp * 13, (big.M + MRI_CHUNK - 1) / MRI_CHUNK, MRI_CHUNK);
    printf("  sequential CPU at %.3f G pairs/s would need about %.0f s (%.1f h): not run. Book: 342 min on its CPU, 1.5 min on the G80.\n", cpu_rate / 1e9, pairs_full / cpu_rate, pairs_full / cpu_rate / 3600);
    printf("  error at 32 sampled voxels (32 x 284,592 terms in double precision): %.2e relative to the RMS value %.1f  %s\n", err, rms, ok ? "PASS" : "FAIL");
    csv_row("stats/08a_mri_ladder.csv", HDR, "v5 full scale,full,%d,%d,%.1f,%.4f,%.2f,%.2e,%s", big.N, big.M, ms, gp, gp * 13, err, ok ? "PASS" : "FAIL");
    hot_note();
    mri_free(bd);
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

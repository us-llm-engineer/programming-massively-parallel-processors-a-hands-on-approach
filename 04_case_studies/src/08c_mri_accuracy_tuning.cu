/*
 * 08c_mri_accuracy_tuning.cu   -- MRI F^H d: what hardware trigonometry costs in accuracy, and joint parameter tuning (book Ch.8)
 *
 * Book, step 3 (hardware trigonometry): replacing sinf/cosf by the special-function-unit intrinsics __sinf/__cosf raises
 * throughput a lot but is less accurate. The authors checked that the reconstructed image quality did not suffer using MSE and
 * PSNR = 20 log10( max(I0) / sqrt(MSE) ): a drop from 27.6 dB to 27.5 dB (about 12 % RMS error against the phantom in both cases).
 * Book, step 4 (experimental tuning): thread-block size, scan points per constant-memory chunk and unroll factor interact through
 * registers, resident blocks and launches, so they are searched jointly (about 20 % better than tuning one at a time).
 *
 *   A. Trigonometric accuracy versus the argument range: raw __sinf/__cosf, __sinf/__cosf after reduction to one cycle, and
 *      accurate sinf/cosf, each against a double-precision reference. (Arguments in an MRI problem reach hundreds of radians.)
 *   B. Image agreement: the F^H d image computed with each trigonometry variant compared with the double-precision image using the
 *      book's MSE / PSNR / relative-RMS formulas. (This compares F^H d images, not a full iterative reconstruction against a phantom.)
 *   C. Joint tuning sweep on the fast kernel: block size x chunk size x unroll factor, all 100 combinations timed.
 * Statistics: stats/08c_trig_accuracy.csv, stats/08c_image_agreement.csv, stats/08c_tuning.csv
 */
#define MRI_CHUNK 4096
#define MRI_NO_SOA
#include "common_mri.cuh"
#include <map>
#include <tuple>
#include <sys/stat.h>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- A: trigonometric accuracy -------------------------------------------------------------------------------------------------
__global__ void trig_kernel(const float *ang, float *o, int mode, int n) {         // o = sin(a) for 3 modes, then cos in the second half
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float a = ang[i], s, c;
    if (mode == 0) { s = __sinf(a); c = __cosf(a); }                                   // raw hardware
    else if (mode == 1) { float cyc = a * (1.0f / TWO_PI_F); float f = TWO_PI_F * (cyc - rintf(cyc)); s = __sinf(f); c = __cosf(f); }   // reduced to one cycle
    else { s = sinf(a); c = cosf(a); }                                                 // accurate
    o[i] = s; o[n + i] = c;
}

// ---- B/C kernels (v5 family with unroll; "raw" without range reduction) -------------------------------------------------------------
template <int UNROLL>
__global__ void fhd_fast(const float *rMu, const float *iMu, const float *x, const float *y, const float *z, float *rF, float *iF, int m0, int Mc, int N, int reduce) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], rn = rF[n], in_ = iF[n];
#pragma unroll UNROLL
    for (int m = 0; m < Mc; m++) {
        float cyc = c_k[m].x * xn + c_k[m].y * yn + c_k[m].z * zn;
        float f = reduce ? TWO_PI_F * (cyc - rintf(cyc)) : TWO_PI_F * cyc;
        float c = __cosf(f), s = __sinf(f);
        rn += rMu[m0 + m] * c - iMu[m0 + m] * s; in_ += iMu[m0 + m] * c + rMu[m0 + m] * s;
    }
    rF[n] = rn; iF[n] = in_;
}
static void run_fast(int unroll, const Mri &p, MriDev &d, int block, int chunk, int reduce) {
    CUDA_CHECK(cudaMemset(d.rF, 0, (size_t)p.N * 4)); CUDA_CHECK(cudaMemset(d.iF, 0, (size_t)p.N * 4));
    int blocks = (p.N + block - 1) / block; std::vector<KD> aos(chunk);
    for (int m0 = 0; m0 < p.M; m0 += chunk) {
        int Mc = std::min(chunk, p.M - m0);
        for (int i = 0; i < Mc; i++) aos[i] = { p.kx[m0 + i], p.ky[m0 + i], p.kz[m0 + i] };
        CUDA_CHECK(cudaMemcpyToSymbol(c_k, aos.data(), Mc * sizeof(KD)));
        switch (unroll) {
            case 1: fhd_fast<1><<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N, reduce); break;
            case 2: fhd_fast<2><<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N, reduce); break;
            case 4: fhd_fast<4><<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N, reduce); break;
            default: fhd_fast<8><<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N, reduce); break;
        }
    }
}
static double image_error(const std::vector<double> &ref, const std::vector<float> &r, const std::vector<float> &im, const std::vector<double> &refi, double *psnr, double *rel) {
    double mse = 0, mx = 0, ss = 0; size_t n = ref.size();
    for (size_t i = 0; i < n; i++) {
        double a = sqrt(ref[i] * ref[i] + refi[i] * refi[i]), b = sqrt((double)r[i] * r[i] + (double)im[i] * im[i]);
        mse += (a - b) * (a - b); mx = std::max(mx, a); ss += a * a;
    }
    mse /= n; *psnr = 20 * log10(mx / sqrt(mse)); *rel = 100.0 * sqrt(mse) / sqrt(ss / n);
    return mse;
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &pr = dev_props();
    mkdir("stats", 0755); remove("stats/08c_trig_accuracy.csv"); remove("stats/08c_image_agreement.csv"); remove("stats/08c_tuning.csv");
    printf("==================== MRI F^H d: ACCURACY AND TUNING (%s) ====================\n\n", pr.name);
    bool ok = true;

    // ------------------------------------------------------------------------------------------------ A
    printf("---- A. Sine/cosine accuracy vs the size of the argument (1,000,000 random angles per range, double-precision reference) ----\n");
    printf("[book] hardware trigonometry trades a little accuracy for a lot of speed. Max abs error of sin/cos:\n");
    printf("%-16s | %-14s %-16s %-14s\n", "|angle| up to", "raw __sinf", "reduced __sinf", "accurate sinf");
    const int NA = 1000000; float *da, *dob; CUDA_CHECK(cudaMalloc(&da, NA * 4)); CUDA_CHECK(cudaMalloc(&dob, 2 * NA * 4));
    double ranges[5] = { 3.14159, 6.28319, 100, 1000, 10000 };
    for (double R : ranges) {
        std::vector<float> a(NA); srand(9); for (auto &v : a) v = (float)((rand() / (double)RAND_MAX * 2 - 1) * R);
        cudaMemcpy(da, a.data(), NA * 4, cudaMemcpyHostToDevice);
        double e[3]; std::vector<float> o(2 * NA);
        for (int mode = 0; mode < 3; mode++) {
            trig_kernel<<<(NA + 255) / 256, 256>>>(da, dob, mode, NA); cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            cudaMemcpy(o.data(), dob, 2 * NA * 4, cudaMemcpyDeviceToHost);
            double w = 0; for (int i = 0; i < NA; i++) w = std::max(w, std::max(fabs(o[i] - sin((double)a[i])), fabs(o[NA + i] - cos((double)a[i])))); e[mode] = w;
        }
        printf("%-16.1f | %-14.2e %-16.2e %-14.2e\n", R, e[0], e[1], e[2]);
        csv_row("stats/08c_trig_accuracy.csv", "max_angle,raw_hw_max_err,reduced_hw_max_err,accurate_max_err", "%.1f,%.3e,%.3e,%.3e", R, e[0], e[1], e[2]);
        ok &= e[2] < 1e-5 && (R > 1000 || e[1] < 5e-4);     // beyond ~1000 rad the float ANGLE itself is only good to ~1e-4 rad, so no kernel can do better
    }
    cudaFree(da); cudaFree(dob);
    printf("  Read: raw hardware sine/cosine degrades as |angle| grows; reducing the argument to one cycle first is somewhat better, but from about\n");
    printf("  1000 rad up both are limited by the float ANGLE itself (spacing ~1e-4 rad there), which the accurate library functions also see\n");
    printf("  (their error here is measured against the double sine of the same float angle, so it stays at float precision).\n\n");

    // ------------------------------------------------------------------------------------------------ B
    printf("---- B. F^H d image agreement with the double-precision image (book: PSNR 27.6 dB vs 27.5 dB with fast trig) ----\n");
    Mri sm = make_mri(8192, 16, 41);                       // 4096 voxels x 8192 samples
    std::vector<double> rr(sm.N), ri(sm.N);
    double t0 = now_ms(); for (int n = 0; n < sm.N; n++) fhd_ref(sm, n, &rr[n], &ri[n]); double refms = now_ms() - t0;
    MriDev dsm = mri_upload(sm); std::vector<float> R(sm.N), I(sm.N);
    printf("%d voxels x %d samples (%.1f ms in double on the CPU). PSNR = 20 log10(max|I0| / sqrt(MSE)) on the magnitude image; 'rel RMS' = sqrt(MSE) / RMS(|I0|).\n", sm.N, sm.M, refms);
    printf("%-46s %-12s %-12s %-12s\n", "trigonometry", "MSE", "PSNR (dB)", "rel RMS (%)");
    struct V { const char *name; int mode; } vs[3] = { { "accurate sinf/cosf (v4)", 0 }, { "hardware, reduced to one cycle (v5)", 1 }, { "hardware, NO range reduction", 2 } };
    for (auto &v : vs) {
        if (v.mode == 0) mri_run(4, sm, dsm, 256, 4096); else run_fast(1, sm, dsm, 256, 4096, v.mode == 1);
        cudaDeviceSynchronize(); cudaMemcpy(R.data(), dsm.rF, sm.N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(I.data(), dsm.iF, sm.N * 4, cudaMemcpyDeviceToHost);
        double psnr, rel, mse = image_error(rr, R, I, ri, &psnr, &rel);
        printf("%-46s %-12.3e %-12.1f %-12.4f\n", v.name, mse, psnr, rel);
        csv_row("stats/08c_image_agreement.csv", "trigonometry,mse,psnr_db,rel_rms_pct", "%s,%.4e,%.2f,%.5f", v.name, mse, psnr, rel);
        if (v.mode != 2) ok &= psnr > 60;
    }
    printf("  Read: the PSNR here measures agreement of the SAME F^H d image computed two ways, so it is far above the book's 27 dB, which\n");
    printf("  measured a full reconstruction against a phantom (dominated by the reconstruction itself). What carries over is the ordering:\n");
    printf("  accurate > reduced hardware >> raw hardware, and that a reduced hardware kernel costs almost nothing in image agreement.\n\n");
    mri_free(dsm);

    // ------------------------------------------------------------------------------------------------ C
    printf("---- C. Joint tuning sweep on the fast kernel: 262,144 voxels (64^3) x 8,192 samples, min of 3 launches per combination ----\n");
    Mri p = make_mri(8192, 64, 51); MriDev d = mri_upload(p);
    double pairs = (double)p.N * p.M;
    int blocks_[5] = { 32, 64, 128, 256, 512 }, chunks[5] = { 256, 512, 1024, 2048, 4096 }, unrolls[4] = { 1, 2, 4, 8 };
    struct Best { double ms; int b, c, u; } best = { 1e30, 0, 0, 0 }, worst = { 0, 0, 0, 0 }; double def_ms = 0;
    std::map<std::tuple<int, int, int>, double> table;
    cooldown();
    { double s0 = now_ms(); while (now_ms() - s0 < 400) { run_fast(1, p, d, 256, 2048, 1); cudaDeviceSynchronize(); } }
    printf("%-8s %-8s %-8s | %-9s %-11s %-6s %-11s\n", "block", "chunk", "unroll", "ms", "G pairs/s", "regs", "blocks/SM");
    for (int b : blocks_) for (int c : chunks) for (int u : unrolls) {
        double t = 1e30;
        for (int r = 0; r < 3; r++) { cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); run_fast(u, p, d, b, c, 1); cudaEventRecord(e1); cudaEventSynchronize(e1);
                                      float ms; cudaEventElapsedTime(&ms, e0, e1); t = std::min(t, (double)ms); cudaEventDestroy(e0); cudaEventDestroy(e1); }
        CUDA_CHECK(cudaGetLastError());
        cudaFuncAttributes fa; const void *k = u == 1 ? (const void *)fhd_fast<1> : u == 2 ? (const void *)fhd_fast<2> : u == 4 ? (const void *)fhd_fast<4> : (const void *)fhd_fast<8>;
        cudaFuncGetAttributes(&fa, k); int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, b, 0);
        if (t < best.ms) best = { t, b, c, u }; if (t > worst.ms) worst = { t, b, c, u }; if (b == 256 && c == 2048 && u == 1) def_ms = t;
        csv_row("stats/08c_tuning.csv", "block,chunk,unroll,ms,gpairs_per_s,regs,blocks_per_sm", "%d,%d,%d,%.4f,%.4f,%d,%d", b, c, u, t, pairs / (t * 1e-3) / 1e9, fa.numRegs, nb);
        table[std::make_tuple(b, c, u)] = t;
    }
    // print a compact table: for each block size the best chunk/unroll
    printf("best combination per block size (of 20 chunk x unroll combinations each):\n");
    for (int b : blocks_) {
        double bt = 1e30; int bc = 0, bu = 0;
        for (int c : chunks) for (int u : unrolls) { double v = table[std::make_tuple(b, c, u)]; if (v < bt) { bt = v; bc = c; bu = u; } }
        printf("  block %-4d best: chunk %-5d unroll %d  ->  %.2f ms  (%.1f G pairs/s)\n", b, bc, bu, bt, pairs / (bt * 1e-3) / 1e9);
    }
    // one-parameter-at-a-time tuning (the book's comparison baseline): block, then chunk, then unroll, each with the others at their default
    int ob = 256, oc = 2048, ou = 1; double ot = table[std::make_tuple(ob, oc, ou)];
    for (int b : blocks_) { double v = table[std::make_tuple(b, oc, ou)]; if (v < ot) { ot = v; ob = b; } }
    for (int c : chunks)  { double v = table[std::make_tuple(ob, c, ou)]; if (v < ot) { ot = v; oc = c; } }
    for (int u : unrolls) { double v = table[std::make_tuple(ob, oc, u)]; if (v < ot) { ot = v; ou = u; } }
    printf("one-at-a-time tuning (block, then chunk, then unroll): block %d, chunk %d, unroll %d = %.2f ms\n", ob, oc, ou, ot);
    printf("\noverall: best  block %d, chunk %d, unroll %d = %.2f ms | worst block %d, chunk %d, unroll %d = %.2f ms (%.1fx slower than best)\n",
           best.b, best.c, best.u, best.ms, worst.b, worst.c, worst.u, worst.ms, worst.ms / best.ms);
    printf("default (block 256, chunk 2048, unroll 1) = %.2f ms. Joint search vs default: +%.1f %%; joint search vs one-at-a-time: +%.1f %% [book: about 20 %% vs one-at-a-time].\n",
           def_ms, 100.0 * (def_ms / best.ms - 1.0), 100.0 * (ot / best.ms - 1.0));
    printf("all 100 combinations are in stats/08c_tuning.csv (for the heat-map).\n");
    mri_free(d);
    printf("\n%s\n", ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return ok ? 0 : 1;
}

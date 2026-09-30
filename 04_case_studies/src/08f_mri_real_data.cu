/*
 * 08f_mri_real_data.cu   -- MRI reconstruction from REAL measured k-space (M4Raw, in-vivo brain, 4 receive coils)
 *
 * The earlier reconstruction (08e) uses a synthetic phantom; the Parboil inputs (08a-08d) hold geometry only and no measured
 * signal. This program reconstructs images from a real scan with the book's F^H d kernels:
 *   data:  M4Raw multi-coil brain k-space (0.3 T scanner, Cartesian 256 x 256, 4 coils), one T1-weighted scan; see scripts/fetch_m4raw.py
 *   truth: the dataset's own reconstruction (inverse FFT + root-sum-of-squares over coils), an independent FFT-based reference
 * Per slice:
 *   1. FULL data: per coil  img_c = (1/256) F^H d_c  on the GPU (65,536 voxels x 65,536 samples per coil, accurate trigonometry and
 *      hardware sinf/cosf), coils combined by root-sum-of-squares. Compared with the dataset reconstruction: this checks F^H on real data.
 *   2. UNDERSAMPLED data (about 3.6x fewer phase-encode lines: central 24 lines + a random 20 % of the rest, fixed seed): zero-filled F^H d.
 *      Fewer samples mean proportionally less work for F^H d.
 *   3. Conjugate gradient on (F^H F) x = F^H d for the undersampled data: the book's iterative solver on real data. Without a prior it
 *      solves the same least-squares problem whose minimum-norm solution is the zero-filled image, so it is expected NOT to remove the
 *      aliasing; the program reports this rather than hiding it.
 * Images go to stats/08f_slice<NN>_<tag>.bin (float32 256x256) and metrics to stats/08f_real_metrics.csv; scripts/plot_real.py draws them.
 * Metrics against the dataset reconstruction: relative RMS error and PSNR = 20 log10(max(ref) / sqrt(MSE)); no rescaling is applied.
 * Data: data/m4raw/slice_NN.bin from scripts/prepare_m4raw.py (not shipped).
 */
#include "common_mri.cuh"
#include "recon_kernels.cuh"
#include <cmath>
#include <string>
#include <sys/stat.h>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Slice { int C, U, V; std::vector<float> ks, ref; };            // ks[c][u][v][2]
static bool load_slice(const std::string &path, Slice &s) {
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    int h[3]; if (fread(h, 4, 3, f) != 3) { fclose(f); return false; }
    s.C = h[0]; s.U = h[1]; s.V = h[2]; s.ks.resize((size_t)s.C * s.U * s.V * 2); s.ref.resize((size_t)s.U * s.V);
    bool ok = fread(s.ks.data(), 4, s.ks.size(), f) == s.ks.size() && fread(s.ref.data(), 4, s.ref.size(), f) == s.ref.size(); fclose(f); return ok;
}
struct Metric { double rel, psnr; };
static Metric compare(const std::vector<float> &ref, const std::vector<float> &img) {
    double mse = 0, ss = 0, mx = 0; size_t n = ref.size();
    for (size_t i = 0; i < n; i++) { double e = (double)img[i] - ref[i]; mse += e * e; ss += (double)ref[i] * ref[i]; mx = std::max(mx, (double)ref[i]); }
    mse /= n; return { 100.0 * sqrt(mse) / sqrt(ss / n), 20 * log10(mx / sqrt(mse)) };
}
static void save(const char *tag, int sl, const std::vector<float> &img) {
    char f[96]; snprintf(f, sizeof f, "stats/08f_slice%02d_%s.bin", sl, tag); FILE *o = fopen(f, "wb"); fwrite(img.data(), 4, img.size(), o); fclose(o);
}

// voxels on the pixel grid (row = y, col = x, centred), samples are the chosen (row, col) k-space positions in cycles per pixel
static Mri make_problem(const Slice &s, const std::vector<char> &keep_row) {
    Mri p; p.S = s.V; p.N = s.U * s.V; p.x.resize(p.N); p.y.resize(p.N); p.z.assign(p.N, 0.f);
    for (int r = 0; r < s.U; r++) for (int c = 0; c < s.V; c++) { p.x[r * s.V + c] = (float)(c - s.V / 2); p.y[r * s.V + c] = (float)(r - s.U / 2); }
    for (int u = 0; u < s.U; u++) if (keep_row[u]) for (int v = 0; v < s.V; v++) { p.ky.push_back((float)(u - s.U / 2) / s.U); p.kx.push_back((float)(v - s.V / 2) / s.V); }
    p.M = (int)p.kx.size(); p.kz.assign(p.M, 0.f); p.rMu.assign(p.M, 0.f); p.iMu.assign(p.M, 0.f); return p;
}
static void gather_data(const Slice &s, int coil, const std::vector<char> &keep_row, float scale, std::vector<float> &re, std::vector<float> &im) {
    re.clear(); im.clear();
    for (int u = 0; u < s.U; u++) if (keep_row[u]) for (int v = 0; v < s.V; v++) {
        size_t i = (((size_t)coil * s.U + u) * s.V + v) * 2; re.push_back(scale * s.ks[i]); im.push_back(scale * s.ks[i + 1]); }
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &pr = dev_props();
    mkdir("stats", 0755); remove("stats/08f_real_metrics.csv");
    const char *HDR = "slice,method,trig,lines_kept,samples,rel_rms_pct,psnr_db,ms";
    printf("==================== MRI RECONSTRUCTION FROM REAL MEASURED K-SPACE (%s) ====================\n", pr.name);
    printf("data: M4Raw in-vivo brain, T1w, 4 coils, Cartesian 256 x 256 (CC-BY 4.0); truth: the dataset's own FFT + root-sum-of-squares reconstruction.\n\n");
    bool all_ok = true, any = false;
    const int slices[3] = { 4, 8, 12 };
    for (int sl : slices) {
        char path[96]; snprintf(path, sizeof path, "data/m4raw/slice_%02d.bin", sl);
        Slice s; if (!load_slice(path, s)) { printf("slice %d: %s not found (run scripts/fetch_m4raw.py and scripts/prepare_m4raw.py)\n", sl, path); continue; }
        any = true; cooldown();
        int U = s.U, V = s.V, N = U * V; size_t nb = (size_t)N * 4;
        std::vector<char> all(U, 1), under(U, 0); srand(5);
        for (int u = 0; u < U; u++) under[u] = (abs(u - U / 2) < 12) || (rand() % 100 < 20);
        int kept = 0; for (char c : under) kept += c;
        printf("---- slice %d: %d coils, %d x %d; full = %d samples, undersampled = %d lines (%.1fx fewer) ----\n", sl, s.C, U, V, N, kept, (double)U / kept);
        printf("  %-46s %-9s %-9s %-11s %-10s %-9s\n", "method", "trig", "samples", "rel RMS %", "PSNR (dB)", "time ms");
        auto report = [&](const char *m, const char *trig, int lines, int samples, const Metric &q, double ms) {
            printf("  %-46s %-9s %-9d %-11.4f %-10.1f %-9.1f\n", m, trig, samples, q.rel, q.psnr, ms);
            csv_row("stats/08f_real_metrics.csv", HDR, "%d,\"%s\",%s,%d,%d,%.4f,%.3f,%.2f", sl, m, trig, lines, samples, q.rel, q.psnr, ms); };
        std::vector<float> rss(N), re(N), im(N), dr, di;
        save("reference", sl, s.ref);
        // 1 + 2: adjoint reconstructions
        for (int mode = 0; mode < 3; mode++) {                            // 0 full accurate, 1 full hardware, 2 undersampled accurate
            const std::vector<char> &keep = mode == 2 ? under : all; Mri p = make_problem(s, keep); MriDev d = mri_upload(p);
            std::fill(rss.begin(), rss.end(), 0.f); double t0 = now_ms(), gpu = 0;
            for (int c = 0; c < s.C; c++) {
                gather_data(s, c, keep, 1.f, dr, di);
                CUDA_CHECK(cudaMemcpy(d.rMu, dr.data(), p.M * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.iMu, di.data(), p.M * 4, cudaMemcpyHostToDevice));
                double a = now_ms(); mri_run(mode == 1 ? 5 : 4, p, d); CUDA_CHECK(cudaDeviceSynchronize()); gpu += now_ms() - a;
                CUDA_CHECK(cudaMemcpy(re.data(), d.rF, nb, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(im.data(), d.iF, nb, cudaMemcpyDeviceToHost));
                for (int i = 0; i < N; i++) { double a2 = re[i] / 256.0, b2 = im[i] / 256.0; rss[i] += (float)(a2 * a2 + b2 * b2); }
                if (mode == 0 && c == 0) {                                // double-precision CPU check of 64 voxels of coil 0
                    double worst = 0, scl = 0;
                    for (int t = 0; t < 64; t++) { int n = (t * 1021) % N; double sr = 0, si = 0, sa = 0;
                        for (int m = 0; m < p.M; m++) { double e = 6.283185307179586 * ((double)p.kx[m] * p.x[n] + (double)p.ky[m] * p.y[n]), cs = cos(e), sn = sin(e);
                            sr += dr[m] * cs - di[m] * sn; si += di[m] * cs + dr[m] * sn; sa += (fabs(dr[m]) + fabs(di[m])); }
                        worst = std::max(worst, hypot(sr - re[n], si - im[n])); scl = std::max(scl, sa); }
                    bool ok = worst / scl < 1e-5; all_ok &= ok; printf("  check: GPU F^H d vs double CPU on 64 voxels, error / sum|terms| = %.2e  %s\n", worst / scl, ok ? "PASS" : "FAIL"); }
            }
            for (int i = 0; i < N; i++) rss[i] = sqrtf(rss[i]);
            Metric q = compare(s.ref, rss);
            const char *nm = mode == 2 ? "zero-filled F^H d, RSS over coils" : "F^H d, RSS over coils";
            report(nm, mode == 1 ? "hardware" : "accurate", mode == 2 ? kept : U, p.M, q, gpu);
            (void)t0; if (mode == 0) { save("full", sl, rss); all_ok &= q.rel < 0.5; } if (mode == 1) all_ok &= q.rel < 1.0; if (mode == 2) save("zerofill", sl, rss);
            mri_free(d);
        }
        // 3: conjugate gradient on (F^H F) x = F^H (256 d) for the undersampled data; per coil, iterate to it_max, report at chosen counts
        {
            Mri p = make_problem(s, under); MriDev d = mri_upload(p); int M = p.M; int blocks = (N + 255) / 256;
            float *xr, *xi, *pr_, *pi_, *rr, *ri, *apr, *api; double *part; for (float **v : { &xr, &xi, &pr_, &pi_, &rr, &ri, &apr, &api }) CUDA_CHECK(cudaMalloc(v, nb));
            CUDA_CHECK(cudaMalloc(&part, blocks * sizeof(double)));
            auto dot = [&](const float *a, const float *b) { dot_kernel<<<blocks, 256>>>(a, b, part, N); std::vector<double> h(blocks); cudaMemcpy(h.data(), part, blocks * 8, cudaMemcpyDeviceToHost); double t = 0; for (double v : h) t += v; return t; };
            const int itmax = 8; // per-iteration coil images accumulate as |.|^2
            std::vector<std::vector<float>> acc(itmax + 1, std::vector<float>(N, 0.f)); double gpu = 0; int conv = 0;
            for (int c = 0; c < s.C; c++) {
                gather_data(s, c, under, 256.f, dr, di);
                CUDA_CHECK(cudaMemcpy(d.rMu, dr.data(), M * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.iMu, di.data(), M * 4, cudaMemcpyHostToDevice));
                double a = now_ms(); mri_run(4, p, d); CUDA_CHECK(cudaMemcpy(rr, d.rF, nb, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(ri, d.iF, nb, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(pr_, rr, nb, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(pi_, ri, nb, cudaMemcpyDeviceToDevice)); cudaMemset(xr, 0, nb); cudaMemset(xi, 0, nb);
                double rs = dot(rr, rr) + dot(ri, ri), rs0 = rs;
                for (int it = 1; it <= itmax; it++) {
                    fwd_kernel<<<(M + 127) / 128, 128>>>(d.kx, d.ky, d.kz, d.x, d.y, d.z, pr_, pi_, d.rMu, d.iMu, M, N, 0);          // A p = F^H F p
                    mri_run(4, p, d); CUDA_CHECK(cudaMemcpy(apr, d.rF, nb, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(api, d.iF, nb, cudaMemcpyDeviceToDevice));
                    double pap = dot(pr_, apr) + dot(pi_, api), al = rs / pap;
                    axpy_kernel<<<blocks, 256>>>((float)al, pr_, xr, N); axpy_kernel<<<blocks, 256>>>((float)al, pi_, xi, N);
                    axpy_kernel<<<blocks, 256>>>((float)-al, apr, rr, N); axpy_kernel<<<blocks, 256>>>((float)-al, api, ri, N);
                    double rs2 = dot(rr, rr) + dot(ri, ri), be = rs2 / rs; rs = rs2;
                    xpay_kernel<<<blocks, 256>>>((float)be, rr, pr_, N); xpay_kernel<<<blocks, 256>>>((float)be, ri, pi_, N);
                    CUDA_CHECK(cudaMemcpy(re.data(), xr, nb, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(im.data(), xi, nb, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < N; i++) acc[it][i] += re[i] * re[i] + im[i] * im[i];
                    if (rs <= 1e-9 * rs0) { for (int j = it + 1; j <= itmax; j++) for (int i = 0; i < N; i++) acc[j][i] += re[i] * re[i] + im[i] * im[i]; conv = std::max(conv, it); break; }   // converged: later iterations would divide by ~0
                }
                CUDA_CHECK(cudaDeviceSynchronize()); gpu += now_ms() - a;
                // note: CG uses the un-restored right-hand side of the ORIGINAL problem; mri_run reused d.rMu as scratch, so restore not needed (rebuilt per coil)
            }
            for (int it : { 1, 2, 4, 8 }) {
                std::vector<float> img(N); for (int i = 0; i < N; i++) img[i] = sqrtf(acc[it][i]);
                Metric q = compare(s.ref, img); char nm[64]; snprintf(nm, sizeof nm, "CG on undersampled data, %d iteration%s", it, it == 1 ? "" : "s");
                report(nm, "accurate", kept, M, q, gpu * std::min(it, conv ? conv : itmax) / (conv ? conv : itmax)); if (it == 8) { save("cg8", sl, img); all_ok &= std::isfinite(q.psnr); }
            }
            for (float *v : { xr, xi, pr_, pi_, rr, ri, apr, api }) cudaFree(v); cudaFree(part); mri_free(d);
        }
        printf("\n"); hot_note();
    }
    if (!any) { printf("No real data found: run scripts/fetch_m4raw.py and scripts/prepare_m4raw.py first.\n\nSOME VARIANTS FAILED\n"); return 1; }
    printf("How to read: with FULL Cartesian data F^H d is the inverse Fourier transform, so agreement with the dataset's FFT reconstruction to well under 1 %%\n");
    printf("validates the kernels on real measured data. The undersampled images show aliasing; CG on the same data converges to the same image (no prior),\n");
    printf("which is expected (the normal matrix has a single non-zero eigenvalue on the kept lines, so CG converges in one iteration and later iterations are skipped): removing aliasing needs coil-sensitivity maps or a sparsity prior, not implemented here. Time for the undersampled F^H d\n");
    printf("scales with the number of samples kept. Timings are only meaningful if the mean SM clock on the last line is near the GPU boost clock (this run\n");
    printf("may have been power-capped; accuracy columns are unaffected). Figure: scripts/plot_real.py -> figures/08_real_data.png.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

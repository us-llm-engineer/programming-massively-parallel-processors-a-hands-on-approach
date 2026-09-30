/*
 * 08e_mri_reconstruction.cu   -- an actual MRI RECONSTRUCTION: phantom -> simulated non-Cartesian scan -> images (book Ch.8)
 *
 * The book reconstructs an image by solving  (F^H F + lambda W^H W) r = F^H d  with the conjugate-gradient method; F^H d is the
 * expensive product studied in 08a-08c. This program closes the loop on a scale this GPU handles in seconds:
 *   1. a synthetic PHANTOM (2D Shepp-Logan-style head, 128 x 128; and a 3D nested-ellipsoid phantom, 32^3)
 *   2. a simulated SCAN: radial k-space samples, d = F rho + noise, computed on the GPU (each thread = one k-space sample)
 *   3. reconstructions, each compared with the phantom (book metrics: MSE, PSNR = 20 log10(max(I0)/sqrt(MSE)), relative RMS):
 *        a. plain adjoint F^H d                                  (blurred: no density compensation)
 *        b. density-compensated adjoint F^H (w d), w ~ |k|^(D-1) (the classic direct "conjugate phase / gridding-like" method)
 *        c. iterative CG on (F^H F + lambda I) x = F^H d          (the book's approach; each iteration = one F and one F^H)
 *   4. the book's accuracy question: the same reconstruction with accurate sinf/cosf and with hardware __sinf/__cosf (v5, range-reduced):
 *      book PSNR 27.6 dB vs 27.5 dB.
 * F^H uses the kernels of 08a (v4 accurate, v5 hardware trigonometry); F uses a new voxel-loop kernel with one thread per sample.
 * Images are written to stats/08e_*.bin (float32 magnitude images) for scripts/plot_recon.py -> figures/08_reconstruction*.png.
 * All scans are seeded and reproducible. Statistics: stats/08e_recon_metrics.csv
 */
#define MRI_CHUNK 2048
#include "common_mri.cuh"
#include <cmath>
#include <string>
#include <sys/stat.h>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// forward model F: t[m] = sum_n x[n] * exp(-i 2 pi k_m . r_n), one thread per k-space sample (no write conflicts: each thread owns t[m])
__global__ void fwd_kernel(const float *kx, const float *ky, const float *kz, const float *x, const float *y, const float *z, const float *ir, const float *ii,
                           float *tr, float *ti, int M, int N, int fast) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; if (m >= M) return;
    float k1 = kx[m], k2 = ky[m], k3 = kz[m], re = 0.f, im = 0.f;
    for (int n = 0; n < N; n++) {
        float cyc = k1 * x[n] + k2 * y[n] + k3 * z[n], c, s;
        if (fast) { float f = TWO_PI_F * (cyc - rintf(cyc)); c = __cosf(f); s = -__sinf(f); }
        else { float e = TWO_PI_F * cyc; c = cosf(e); s = -sinf(e); }
        re += ir[n] * c - ii[n] * s; im += ir[n] * s + ii[n] * c;
    }
    tr[m] = re; ti[m] = im;
}
__global__ void axpy_kernel(float a, const float *x, float *y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i]; }
__global__ void xpay_kernel(float a, const float *x, float *y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] + a * y[i]; }   // y = x + a*y
__global__ void dot_kernel(const float *a, const float *b, double *part, int n) {                 // block partial sums of a.b (double)
    __shared__ double sh[256]; int i = blockIdx.x * blockDim.x + threadIdx.x; sh[threadIdx.x] = i < n ? (double)a[i] * b[i] : 0.0; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

struct Scan { Mri p; std::vector<float> rho, wdc; int D, S; const char *name; };

// ---- phantoms -------------------------------------------------------------------------------------------------------------------------
static float phantom2d(float X, float Y) {                       // Shepp-Logan-style head on [-1,1]^2 (modified intensities), ellipses: A a b x0 y0 phi(deg)
    static const float E[10][6] = { {1, .69f, .92f, 0, 0, 0}, {-.8f, .6624f, .874f, 0, -.0184f, 0}, {-.2f, .11f, .31f, .22f, 0, -18}, {-.2f, .16f, .41f, -.22f, 0, 18},
        {.1f, .21f, .25f, 0, .35f, 0}, {.1f, .046f, .046f, 0, .1f, 0}, {.1f, .046f, .046f, 0, -.1f, 0}, {.1f, .046f, .023f, -.08f, -.605f, 0}, {.1f, .023f, .023f, 0, -.606f, 0}, {.1f, .023f, .046f, .06f, -.605f, 0} };
    float v = 0.f;
    for (auto &e : E) { float ph = e[5] * 3.14159265f / 180.f, cx = X - e[3], cy = Y - e[4], u = cx * cosf(ph) + cy * sinf(ph), w = -cx * sinf(ph) + cy * cosf(ph);
                        if (u * u / (e[1] * e[1]) + w * w / (e[2] * e[2]) <= 1.f) v += e[0]; }
    return v;
}
static float phantom3d(float X, float Y, float Z) {              // nested ellipsoids: A a b c x0 y0 z0
    static const float E[6][7] = { {1, .69f, .92f, .85f, 0, 0, 0}, {-.8f, .6624f, .874f, .8f, 0, -.0184f, 0}, {-.2f, .11f, .31f, .22f, .22f, 0, 0},
                                   {-.2f, .16f, .41f, .3f, -.22f, 0, 0}, {.3f, .21f, .25f, .25f, 0, .35f, .1f}, {.3f, .08f, .08f, .08f, 0, -.35f, -.2f} };
    float v = 0.f;
    for (auto &e : E) { float a = (X - e[4]) / e[1], b = (Y - e[5]) / e[2], c = (Z - e[6]) / e[3]; if (a * a + b * b + c * c <= 1.f) v += e[0]; }
    return v;
}

static Scan make_scan(int D, int S, int spokes, int per_spoke, unsigned seed) {
    Scan sc; sc.D = D; sc.S = S; sc.name = D == 2 ? "2D" : "3D";
    Mri &p = sc.p; p.S = S; p.N = D == 2 ? S * S : S * S * S; p.M = spokes * per_spoke;
    p.x.resize(p.N); p.y.resize(p.N); p.z.resize(p.N); sc.rho.resize(p.N);
    int nz = D == 2 ? 1 : S;
    for (int k = 0, n = 0; k < nz; k++) for (int j = 0; j < S; j++) for (int i = 0; i < S; i++, n++) {
        p.x[n] = ((float)i - S / 2) / S; p.y[n] = ((float)j - S / 2) / S; p.z[n] = D == 2 ? 0.f : ((float)k - S / 2) / S;
        sc.rho[n] = D == 2 ? phantom2d(2 * p.x[n], 2 * p.y[n]) : phantom3d(2 * p.x[n], 2 * p.y[n], 2 * p.z[n]); }
    p.kx.resize(p.M); p.ky.resize(p.M); p.kz.resize(p.M); p.rMu.assign(p.M, 0.f); p.iMu.assign(p.M, 0.f); sc.wdc.resize(p.M);
    for (int s = 0; s < spokes; s++) {
        float dx, dy, dz;
        if (D == 2) { float th = 3.14159265f * s / spokes; dx = cosf(th); dy = sinf(th); dz = 0; }
        else { float zc = 1.f - 2.f * (s + 0.5f) / spokes, r = sqrtf(1.f - zc * zc), ph = s * 2.39996323f; dx = r * cosf(ph); dy = r * sinf(ph); dz = zc; }   // golden-angle spiral on the sphere
        for (int t = 0; t < per_spoke; t++) { float tt = (float)t - per_spoke / 2, kr = tt * (S / 2.f) / (per_spoke / 2.f); int m = s * per_spoke + t;
            p.kx[m] = kr * dx; p.ky[m] = kr * dy; p.kz[m] = kr * dz; sc.wdc[m] = powf(fabsf(kr) + 0.5f, (float)(D - 1)); }
    }
    (void)seed; return sc;
}

// ---- image metrics (book): MSE, PSNR against the phantom (scaled by the best real factor, since an adjoint has an arbitrary scale) ----------
struct Metric { double mse, psnr, rel; };
static Metric compare(const std::vector<float> &rho, const std::vector<float> &re, const std::vector<float> &im, std::vector<float> *mag_out = nullptr) {
    size_t n = rho.size(); std::vector<double> m(n); double num = 0, den = 0, mx = 0, ss = 0;
    for (size_t i = 0; i < n; i++) { m[i] = sqrt((double)re[i] * re[i] + (double)im[i] * im[i]); num += m[i] * rho[i]; den += m[i] * m[i]; mx = std::max(mx, (double)rho[i]); ss += (double)rho[i] * rho[i]; }
    double s = den > 0 ? num / den : 1; double mse = 0;
    for (size_t i = 0; i < n; i++) { double e = s * m[i] - rho[i]; mse += e * e; if (mag_out) (*mag_out)[i] = (float)(s * m[i]); }
    mse /= n; Metric r = { mse, 20 * log10(mx / sqrt(mse)), 100.0 * sqrt(mse) / sqrt(ss / n) }; return r;
}

struct Solver {
    Scan &sc; MriDev d; int fast; float *xr, *xi, *pr, *pi, *rr, *ri, *apr, *api; double *part; int N, M, dblocks;
    Solver(Scan &s, int fast_) : sc(s), fast(fast_) {
        N = s.p.N; M = s.p.M; d = mri_upload(s.p); size_t nb = (size_t)N * 4;
        for (float **v : { &xr, &xi, &pr, &pi, &rr, &ri, &apr, &api }) CUDA_CHECK(cudaMalloc(v, nb));
        dblocks = (N + 255) / 256; CUDA_CHECK(cudaMalloc(&part, dblocks * sizeof(double)));
    }
    // y = F^H (w * F x)  for the complex image x; result into d.rF, d.iF. (w = 1 for the normal equations)
    void adjoint_of_mu() { mri_run(fast ? 5 : 4, sc.p, d); }
    void normal_apply(const float *ar, const float *ai, float *outr, float *outi, float lambda) {
        fwd_kernel<<<(M + 127) / 128, 128>>>(d.kx, d.ky, d.kz, d.x, d.y, d.z, ar, ai, d.rMu, d.iMu, M, N, fast);
        adjoint_of_mu();
        cudaMemcpy(outr, d.rF, (size_t)N * 4, cudaMemcpyDeviceToDevice); cudaMemcpy(outi, d.iF, (size_t)N * 4, cudaMemcpyDeviceToDevice);
        axpy_kernel<<<dblocks, 256>>>(lambda, ar, outr, N); axpy_kernel<<<dblocks, 256>>>(lambda, ai, outi, N);
    }
    double dot(const float *a, const float *b) {
        dot_kernel<<<dblocks, 256>>>(a, b, part, N); std::vector<double> h(dblocks); cudaMemcpy(h.data(), part, dblocks * 8, cudaMemcpyDeviceToHost);
        double s = 0; for (double v : h) s += v; return s;
    }
};

int main() {
    RunMonitor mon;
    const cudaDeviceProp &pr = dev_props();
    mkdir("stats", 0755); remove("stats/08e_recon_metrics.csv"); remove("stats/08e_images_meta.csv");
    const char *HDR = "case,method,trig,iteration,mse,psnr_db,rel_rms_pct,residual,ms";
    printf("==================== MRI RECONSTRUCTION FROM A SIMULATED NON-CARTESIAN SCAN (%s) ====================\n", pr.name);
    printf("phantom -> radial k-space samples -> d = F rho + 1 %% complex noise -> images. PSNR = 20 log10(max(I0)/sqrt(MSE)); images are scaled by the\n");
    printf("best real factor before comparison. [book] full reconstruction against a phantom: PSNR 27.6 dB (CPU and accurate GPU) vs 27.5 dB with hardware trig.\n\n");
    bool all_ok = true;

    struct Cfg { int D, S, spokes, per; const char *label; int iters; double lam; } cfgs[2] = { { 2, 128, 128, 128, "2D 128x128, 128 radial spokes x 128 samples", 40, 0.05 },
                                                                                            { 3, 32, 256, 128, "3D 32^3, 256 radial spokes x 128 samples", 25, 0.05 } };
    for (auto &cf : cfgs) {
        cooldown();
        Scan sc = make_scan(cf.D, cf.S, cf.spokes, cf.per, 1);
        int N = sc.p.N, M = sc.p.M; size_t nb = (size_t)N * 4, mb = (size_t)M * 4;
        printf("---- %s : %d voxels x %d k-space samples (%.2e pairs per F or F^H) ----\n", cf.label, N, M, (double)N * M);
        Solver S(sc, 0);                                                     // accurate trigonometry first
        // simulate the scan: d = F rho (accurate), plus complex Gaussian noise (1 % of the RMS of d)
        float *dre, *dim_, *zero; cudaMalloc(&dre, mb); cudaMalloc(&dim_, mb); cudaMalloc(&zero, nb); cudaMemset(zero, 0, nb);
        cudaMemcpy(S.xr, sc.rho.data(), nb, cudaMemcpyHostToDevice); cudaMemset(S.xi, 0, nb);
        fwd_kernel<<<(M + 127) / 128, 128>>>(S.d.kx, S.d.ky, S.d.kz, S.d.x, S.d.y, S.d.z, S.xr, S.xi, dre, dim_, M, N, 0); cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
        std::vector<float> hr(M), hi(M); cudaMemcpy(hr.data(), dre, mb, cudaMemcpyDeviceToHost); cudaMemcpy(hi.data(), dim_, mb, cudaMemcpyDeviceToHost);
        double rmsd = 0; for (int m = 0; m < M; m++) rmsd += (double)hr[m] * hr[m] + (double)hi[m] * hi[m]; rmsd = sqrt(rmsd / M);
        srand(77); auto gauss = []() { double u1 = (rand() + 1.0) / (RAND_MAX + 2.0), u2 = rand() / (RAND_MAX + 1.0); return sqrt(-2 * log(u1)) * cos(6.283185307 * u2); };
        for (int m = 0; m < M; m++) { hr[m] += (float)(0.01 * rmsd * gauss()); hi[m] += (float)(0.01 * rmsd * gauss()); }
        // scan sanity check: F rho at the DC sample (k = 0) equals the sum of the phantom
        double dc = 0; for (float v : sc.rho) dc += v; int mdc = (M / cf.per) / 2 * cf.per + cf.per / 2;
        printf("  scan check: DC sample |d| = %.2f vs sum of phantom = %.2f (F rho at k = 0)\n", sqrt((double)hr[mdc] * hr[mdc] + (double)hi[mdc] * hi[mdc]), dc);
        all_ok &= fabs(sqrt((double)hr[mdc] * hr[mdc] + (double)hi[mdc] * hi[mdc]) - dc) < 0.02 * dc + 1.0;
        printf("  %-46s %-8s %-11s %-10s %-11s %-10s\n", "method", "trig", "PSNR (dB)", "rel RMS %", "iteration", "time ms");
        std::vector<float> mag(N), re(N), im(N);
        auto save_img = [&](const char *tag, const std::vector<float> &m) {
            std::string f = std::string("stats/08e_") + cf.label[0] + "D_" + tag + ".bin"; FILE *o = fopen(f.c_str(), "wb");
            int sl = cf.D == 2 ? 0 : cf.S / 2; size_t off = (size_t)sl * cf.S * cf.S; fwrite(m.data() + off, 4, (size_t)cf.S * cf.S, o); fclose(o);
            csv_row("stats/08e_images_meta.csv", "case,tag,size", "%dD,%s,%d", cf.D, tag, cf.S); };
        { std::vector<float> ph = sc.rho; save_img("phantom", ph); }
        auto report = [&](const char *method, const char *trig, int it, const Metric &m, double ms, double resid, const char *tag) {
            printf("  %-46s %-8s %-11.1f %-10.1f %-11d %-10.1f\n", method, trig, m.psnr, m.rel, it, ms);
            csv_row("stats/08e_recon_metrics.csv", HDR, "%dD,%s,%s,%d,%.5e,%.3f,%.3f,%.4e,%.2f", cf.D, method, trig, it, m.mse, m.psnr, m.rel, resid, ms);
            if (tag) save_img(tag, mag); };
        // (a) plain adjoint: mu = d
        for (int v = 0; v < 2; v++) {
            Solver &s = v == 0 ? S : *new Solver(sc, 1);
            cudaMemcpy(s.d.rMu, hr.data(), mb, cudaMemcpyHostToDevice); cudaMemcpy(s.d.iMu, hi.data(), mb, cudaMemcpyHostToDevice);
            double t0 = now_ms(); s.adjoint_of_mu(); cudaDeviceSynchronize(); double ms = now_ms() - t0;
            cudaMemcpy(re.data(), s.d.rF, nb, cudaMemcpyDeviceToHost); cudaMemcpy(im.data(), s.d.iF, nb, cudaMemcpyDeviceToHost);
            Metric m = compare(sc.rho, re, im, &mag); report("plain adjoint F^H d", v ? "hardware" : "accurate", 0, m, ms, 0, v ? nullptr : "adjoint");
            // (b) density-compensated adjoint
            std::vector<float> wr(M), wi(M); for (int q = 0; q < M; q++) { wr[q] = hr[q] * sc.wdc[q]; wi[q] = hi[q] * sc.wdc[q]; }
            cudaMemcpy(s.d.rMu, wr.data(), mb, cudaMemcpyHostToDevice); cudaMemcpy(s.d.iMu, wi.data(), mb, cudaMemcpyHostToDevice);
            t0 = now_ms(); s.adjoint_of_mu(); cudaDeviceSynchronize(); ms = now_ms() - t0;
            cudaMemcpy(re.data(), s.d.rF, nb, cudaMemcpyDeviceToHost); cudaMemcpy(im.data(), s.d.iF, nb, cudaMemcpyDeviceToHost);
            m = compare(sc.rho, re, im, &mag); report("density-compensated adjoint F^H(w d)", v ? "hardware" : "accurate", 0, m, ms, 0, v ? nullptr : "dcf");
            if (v) delete &s;
        }
        // (c) conjugate gradient on (F^H F + lambda I) x = F^H d, both trig variants
        double lambda = cf.lam * M;
        for (int v = 0; v < 2; v++) {
            Solver &s = v == 0 ? S : *new Solver(sc, 1);
            cudaMemcpy(s.d.rMu, hr.data(), mb, cudaMemcpyHostToDevice); cudaMemcpy(s.d.iMu, hi.data(), mb, cudaMemcpyHostToDevice);
            s.adjoint_of_mu(); cudaMemcpy(s.rr, s.d.rF, nb, cudaMemcpyDeviceToDevice); cudaMemcpy(s.ri, s.d.iF, nb, cudaMemcpyDeviceToDevice);     // r = b = F^H d (x0 = 0)
            cudaMemcpy(s.pr, s.rr, nb, cudaMemcpyDeviceToDevice); cudaMemcpy(s.pi, s.ri, nb, cudaMemcpyDeviceToDevice); cudaMemset(s.xr, 0, nb); cudaMemset(s.xi, 0, nb);
            double rs = s.dot(s.rr, s.rr) + s.dot(s.ri, s.ri), rs0 = rs; double t0 = now_ms(); Metric last = {}; double best_psnr = 0;
            for (int it = 1; it <= cf.iters; it++) {
                s.normal_apply(s.pr, s.pi, s.apr, s.api, (float)lambda);
                double pap = s.dot(s.pr, s.apr) + s.dot(s.pi, s.api), alpha = rs / pap;
                axpy_kernel<<<s.dblocks, 256>>>((float)alpha, s.pr, s.xr, N); axpy_kernel<<<s.dblocks, 256>>>((float)alpha, s.pi, s.xi, N);
                axpy_kernel<<<s.dblocks, 256>>>((float)-alpha, s.apr, s.rr, N); axpy_kernel<<<s.dblocks, 256>>>((float)-alpha, s.api, s.ri, N);
                double rs2 = s.dot(s.rr, s.rr) + s.dot(s.ri, s.ri), beta = rs2 / rs; rs = rs2;
                xpay_kernel<<<s.dblocks, 256>>>((float)beta, s.rr, s.pr, N); xpay_kernel<<<s.dblocks, 256>>>((float)beta, s.ri, s.pi, N);
                cudaMemcpy(re.data(), s.xr, nb, cudaMemcpyDeviceToHost); cudaMemcpy(im.data(), s.xi, nb, cudaMemcpyDeviceToHost);
                Metric m = compare(sc.rho, re, im, &mag); last = m; best_psnr = std::max(best_psnr, m.psnr);
                bool show = it == 1 || it == 5 || it == 10 || it == cf.iters;
                char tag[24]; snprintf(tag, sizeof tag, "cg%02d_%s", it, v ? "hw" : "acc");
                if (show) report("CG (F^H F + lambda I) x = F^H d", v ? "hardware" : "accurate", it, m, now_ms() - t0, sqrt(rs / rs0), (it == 5 || it == 10 || it == cf.iters) && !v ? tag : nullptr);
                else csv_row("stats/08e_recon_metrics.csv", HDR, "%dD,%s,%s,%d,%.5e,%.3f,%.3f,%.4e,%.2f", cf.D, "CG (F^H F + lambda I) x = F^H d", v ? "hardware" : "accurate", it, m.mse, m.psnr, m.rel, sqrt(rs / rs0), now_ms() - t0);
            }
            all_ok &= last.psnr > 15 && std::isfinite(last.psnr);
            if (v) delete &s;
        }
        // the iterative reconstruction must beat the plain adjoint (the book's motivation for iterating)
        printf("\n");
        cudaFree(dre); cudaFree(dim_); cudaFree(zero);
        hot_note();
    }
    printf("How to read: 'plain adjoint' is blurred by the uneven sampling density of radial spokes; density compensation fixes the blur but keeps streaking\n");
    printf("and noise; the CG solution uses the whole data model and should give the highest PSNR. The last rows compare accurate against hardware trigonometry.\n");
    printf("Images (central slice for 3D) are in stats/08e_*.bin; run `make figures` to render figures/08_reconstruction*.png.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

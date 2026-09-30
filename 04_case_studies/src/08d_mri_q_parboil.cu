/*
 * 08d_mri_q_parboil.cu   -- MRI-Q on the real Parboil "mri-q" datasets (small and large)
 *
 * The Parboil mri-q benchmark comes from the same work as the book's MRI case study (Stone et al., "Accelerating advanced MRI
 * reconstructions on GPUs"). It computes the matrix Q, which depends only on the scan trajectory:
 *      phiMag[m] = phiR[m]^2 + phiI[m]^2
 *      Qr[n] = sum_m phiMag[m] * cos(2*pi*(kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]))
 *      Qi[n] = sum_m phiMag[m] * sin(  same argument  )
 * i.e. the same "one thread per voxel, sum over all k-space samples with a sine and a cosine" structure as F^H d (08a) with the sample
 * weight phiMag instead of mu. The input files hold NO measured signal d and NO field map, so F^H d itself cannot be computed from them.
 * File layout (checked against the benchmark's own reader): int numK, int numX, then float arrays kx, ky, kz (numK each),
 * x, y, z (numX each), phiR, phiI (numK each).
 * Data: data/parboil_mri_q/{small,large}/  (fetch with scripts/fetch_parboil_data.sh; sizes 454,664 and 3,186,696 bytes).
 * Variants: accurate sinf/cosf with k-space in global memory; accurate with {kx,ky,kz,phiMag} float4 in constant memory (chunks of 4000
 * samples = 64,000 B); hardware sine/cosine with the argument reduced to one cycle. Each checked against a double-precision reference
 * at 200 sampled voxels. Statistics: stats/08d_mri_q.csv
 */
#include "common.cuh"
#include "monitor.cuh"
#include <string>
#include <sys/stat.h>

#define QCH 4000
#define TWO_PI_F 6.28318530717959f
__constant__ float4 c_q[QCH];

struct QData { int K = 0, N = 0; std::vector<float> kx, ky, kz, x, y, z, phiR, phiI, mag; };

static bool load_qdata(const std::string &path, QData &d) {
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    int hdr[2]; bool ok = fread(hdr, 4, 2, f) == 2; d.K = hdr[0]; d.N = hdr[1];
    auto rd = [&](std::vector<float> &v, int n) { v.resize(n); ok = ok && fread(v.data(), 4, n, f) == (size_t)n; };
    if (ok) { rd(d.kx, d.K); rd(d.ky, d.K); rd(d.kz, d.K); rd(d.x, d.N); rd(d.y, d.N); rd(d.z, d.N); rd(d.phiR, d.K); rd(d.phiI, d.K); }
    fclose(f);
    if (!ok) return false;
    d.mag.resize(d.K); for (int m = 0; m < d.K; m++) d.mag[m] = d.phiR[m] * d.phiR[m] + d.phiI[m] * d.phiI[m];
    return true;
}

// v1: accurate trigonometry, k-space and weights from global memory, voxel data in registers
__global__ void q_global(const float *kx, const float *ky, const float *kz, const float *mag, const float *x, const float *y, const float *z, float *Qr, float *Qi, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], r = 0.f, im = 0.f;
    for (int m = 0; m < K; m++) { float e = TWO_PI_F * (kx[m] * xn + ky[m] * yn + kz[m] * zn); r += mag[m] * cosf(e); im += mag[m] * sinf(e); }
    Qr[n] = r; Qi[n] = im;
}
// v2 / v3: k-space chunk in constant memory as float4 {kx, ky, kz, phiMag}; FAST = hardware sine/cosine after reducing to one cycle
template <bool FAST>
__global__ void q_const(const float *x, const float *y, const float *z, float *Qr, float *Qi, int Mc, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], r = Qr[n], im = Qi[n];
    for (int m = 0; m < Mc; m++) {
        float4 k = c_q[m], cyc = make_float4(0, 0, 0, 0); (void)cyc;
        float c_, s_;
        if (FAST) { float cy = k.x * xn + k.y * yn + k.z * zn; float f = TWO_PI_F * (cy - rintf(cy)); c_ = __cosf(f); s_ = __sinf(f); }
        else { float e = TWO_PI_F * (k.x * xn + k.y * yn + k.z * zn); c_ = cosf(e); s_ = sinf(e); }
        r += k.w * c_; im += k.w * s_;
    }
    Qr[n] = r; Qi[n] = im;
}

struct QDev { float *kx, *ky, *kz, *mag, *x, *y, *z, *Qr, *Qi; };
static QDev upload(const QData &d) {
    QDev v; size_t kb = d.K * 4, nb = (size_t)d.N * 4;
    CUDA_CHECK(cudaMalloc(&v.kx, kb)); CUDA_CHECK(cudaMalloc(&v.ky, kb)); CUDA_CHECK(cudaMalloc(&v.kz, kb)); CUDA_CHECK(cudaMalloc(&v.mag, kb));
    CUDA_CHECK(cudaMalloc(&v.x, nb)); CUDA_CHECK(cudaMalloc(&v.y, nb)); CUDA_CHECK(cudaMalloc(&v.z, nb)); CUDA_CHECK(cudaMalloc(&v.Qr, nb)); CUDA_CHECK(cudaMalloc(&v.Qi, nb));
    cudaMemcpy(v.kx, d.kx.data(), kb, cudaMemcpyHostToDevice); cudaMemcpy(v.ky, d.ky.data(), kb, cudaMemcpyHostToDevice); cudaMemcpy(v.kz, d.kz.data(), kb, cudaMemcpyHostToDevice);
    cudaMemcpy(v.mag, d.mag.data(), kb, cudaMemcpyHostToDevice);
    cudaMemcpy(v.x, d.x.data(), nb, cudaMemcpyHostToDevice); cudaMemcpy(v.y, d.y.data(), nb, cudaMemcpyHostToDevice); cudaMemcpy(v.z, d.z.data(), nb, cudaMemcpyHostToDevice);
    return v;
}
static void run_variant(int v, const QData &d, QDev &q) {
    cudaMemset(q.Qr, 0, (size_t)d.N * 4); cudaMemset(q.Qi, 0, (size_t)d.N * 4);
    int blocks = (d.N + 255) / 256;
    if (v == 0) { q_global<<<blocks, 256>>>(q.kx, q.ky, q.kz, q.mag, q.x, q.y, q.z, q.Qr, q.Qi, d.K, d.N); return; }
    std::vector<float4> h(QCH);
    for (int m0 = 0; m0 < d.K; m0 += QCH) {
        int Mc = std::min(QCH, d.K - m0);
        for (int i = 0; i < Mc; i++) h[i] = make_float4(d.kx[m0 + i], d.ky[m0 + i], d.kz[m0 + i], d.mag[m0 + i]);
        CUDA_CHECK(cudaMemcpyToSymbol(c_q, h.data(), Mc * sizeof(float4)));
        if (v == 1) q_const<false><<<blocks, 256>>>(q.x, q.y, q.z, q.Qr, q.Qi, Mc, d.N); else q_const<true><<<blocks, 256>>>(q.x, q.y, q.z, q.Qr, q.Qi, Mc, d.N);
    }
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// worst error over sampled voxels against a double-precision reference, relative to the RMS magnitude of the reference
static double check_q(const QData &d, const std::vector<float> &Qr, const std::vector<float> &Qi, int count, double *rms_out = nullptr) {
    std::vector<double> rr(count), ri(count); std::vector<int> ids(count); unsigned s = 7; double ss = 0;
    for (int c = 0; c < count; c++) {
        s = s * 1664525u + 1013904223u; int n = (s >> 8) % d.N; ids[c] = n; double r = 0, im = 0;
        for (int m = 0; m < d.K; m++) { double e = 2.0 * M_PI * ((double)d.kx[m] * d.x[n] + (double)d.ky[m] * d.y[n] + (double)d.kz[m] * d.z[n]); r += d.mag[m] * cos(e); im += d.mag[m] * sin(e); }
        rr[c] = r; ri[c] = im; ss += r * r + im * im;
    }
    double rms = sqrt(ss / count), worst = 0;
    for (int c = 0; c < count; c++) worst = std::max(worst, std::max(fabs(Qr[ids[c]] - rr[c]), fabs(Qi[ids[c]] - ri[c])));
    if (rms_out) *rms_out = rms;
    return worst / rms;
}

int main(int argc, char **argv) {
    RunMonitor mon;
    const cudaDeviceProp &pr = dev_props();
    mkdir("stats", 0755); remove("stats/08d_mri_q.csv");
    const char *HDR = "dataset,numK,numX,pairs,variant,ms,gpairs_per_s,err_rel_rms,check";
    std::vector<std::pair<std::string, std::string>> sets = { { "small", "data/parboil_mri_q/small/32_32_32_dataset.bin" }, { "large", "data/parboil_mri_q/large/64_64_64_dataset.bin" } };
    for (int i = 1; i + 1 < argc; i += 2) sets.push_back({ argv[i], argv[i + 1] });
    if (argc > 2) sets.erase(sets.begin(), sets.begin() + 2);              // explicit "name path" pairs replace the defaults
    printf("==================== MRI-Q ON THE PARBOIL DATASETS (%s) ====================\n", pr.name);
    printf("Q(x) = sum_k |phi_k|^2 exp(i 2 pi k.x), one thread per voxel. Error vs a double reference at 200 sampled voxels, relative to the RMS of Q.\n");
    printf("Mean of 20 timed runs after a warm-up and a clock-priming pass.\n\n");
    bool all_ok = true; int loaded = 0;
    printf("%-6s %-8s %-9s %-11s | %-34s %-10s %-11s %-10s | %s\n", "set", "numK", "numX", "pairs", "variant", "ms", "G pairs/s", "vs CPU", "err");
    for (auto &s : sets) {
        QData d;
        if (!load_qdata(s.second, d)) { printf("%-6s cannot read %s (run scripts/fetch_parboil_data.sh first)\n", s.first.c_str(), s.second.c_str()); continue; }
        loaded++;
        QDev q = upload(d); double pairs = (double)d.K * d.N;
        // single-core CPU pair rate (float, libm) on a 2048-voxel slice of the same data
        int nc = std::min(d.N, 2048); std::vector<float> cr(nc, 0.f), ci(nc, 0.f); double t0 = now_ms();
        for (int m = 0; m < d.K; m++) for (int n = 0; n < nc; n++) { float e = TWO_PI_F * (d.kx[m] * d.x[n] + d.ky[m] * d.y[n] + d.kz[m] * d.z[n]); cr[n] += d.mag[m] * cosf(e); ci[n] += d.mag[m] * sinf(e); }
        volatile float sink = cr[0] + ci[nc - 1]; (void)sink; double cpu_rate = (double)d.K * nc / ((now_ms() - t0) * 1e-3);
        cooldown();
        { double p0 = now_ms(); while (now_ms() - p0 < 300) { run_variant(2, d, q); cudaDeviceSynchronize(); } }
        const char *names[3] = { "accurate, k-space in global memory", "accurate, float4 in constant memory", "hardware sin/cos, reduced, constant" };
        float tt[3]; bench_rr(3, [&](int v) { run_variant(v, d, q); }, 20, tt);
        std::vector<float> Qr(d.N), Qi(d.N);
        for (int v = 0; v < 3; v++) {
            run_variant(v, d, q); cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            cudaMemcpy(Qr.data(), q.Qr, (size_t)d.N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(Qi.data(), q.Qi, (size_t)d.N * 4, cudaMemcpyDeviceToHost);
            double rms, err = check_q(d, Qr, Qi, 200, &rms); bool ok = err < (v == 2 ? 5e-3 : 1e-3); all_ok &= ok;
            double gp = pairs / (tt[v] * 1e-3);
            printf("%-6s %-8d %-9d %-11.2e | %-34s %-10.3f %-11.2f %-10.0f | %.1e %s\n", s.first.c_str(), d.K, d.N, pairs, names[v], tt[v], gp / 1e9, gp / cpu_rate, err, ok ? "PASS" : "FAIL");
            csv_row("stats/08d_mri_q.csv", HDR, "%s,%d,%d,%.0f,\"%s\",%.4f,%.4f,%.2e,%s", s.first.c_str(), d.K, d.N, pairs, names[v], tt[v], gp / 1e9, err, ok ? "PASS" : "FAIL");
        }
        hot_note();
        cudaFree(q.kx); cudaFree(q.ky); cudaFree(q.kz); cudaFree(q.mag); cudaFree(q.x); cudaFree(q.y); cudaFree(q.z); cudaFree(q.Qr); cudaFree(q.Qi);
    }
    if (!loaded) { printf("\nno dataset could be read.\n"); return 2; }
    printf("\nHow to read: 'vs CPU' is the ratio of pair rates against the single-core float loop on the same data. The small set has few voxels (32^3) and\n");
    printf("the large set many more (64^3) but fewer samples; per-pair cost is what matters when comparing rows. Times include the constant-memory uploads.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

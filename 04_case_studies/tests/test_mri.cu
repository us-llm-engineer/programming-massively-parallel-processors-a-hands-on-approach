// Tests of the MRI F^H d kernels (common_mri.cuh): every version against the double-precision reference, chunk-size and block-size
// invariance, linearity in mu, the single-sample magnitude identity, and the mu computation.
#define MRI_CHUNK 256
#include "common_mri.cuh"
#include "test_util.cuh"

static void fetch(const Mri &p, MriDev &d, std::vector<float> &r, std::vector<float> &i) {
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError()); r.resize(p.N); i.resize(p.N);
    cudaMemcpy(r.data(), d.rF, (size_t)p.N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(i.data(), d.iF, (size_t)p.N * 4, cudaMemcpyDeviceToHost);
}

int main() {
    Mri p = make_mri(1000, 8, 5); MriDev d = mri_upload(p);            // 512 voxels x 1000 samples = 4 constant-memory chunks of 256
    printf("MRI kernels: %d voxels x %d samples\n", p.N, p.M);
    std::vector<float> r, im, r2, i2;
    double tol[5] = { 1e-4, 1e-4, 1e-4, 1e-4, 5e-3 };
    for (int v = 1; v <= 5; v++) { mri_run(v, p, d); fetch(p, d, r, im); double e = check_fhd(p, r, im, 200); CHECK(e < tol[v - 1], "version v%d error %.2e (limit %.0e)", v, e, tol[v - 1]); }

    // chunk size and block size must not change the result (voxels are independent; chunking only splits the sample loop)
    mri_run(4, p, d, 256, 256); fetch(p, d, r, im);
    mri_run(4, p, d, 256, 100); fetch(p, d, r2, i2);
    double worst = 0, mag = 0; for (int n = 0; n < p.N; n++) { worst = fmax(worst, fmax(fabs(r[n] - r2[n]), fabs(im[n] - i2[n]))); mag = fmax(mag, fabs(r[n])); }
    CHECK(worst < 1e-3 * mag, "chunk size changes the result: %.2e vs magnitude %.2e", worst, mag);
    mri_run(4, p, d, 32, 256); fetch(p, d, r2, i2); bool same = true; for (int n = 0; n < p.N; n++) same &= r[n] == r2[n] && im[n] == i2[n];
    CHECK(same, "block size changes the result bit-for-bit");

    // linearity: doubling mu doubles F^H d
    Mri q = p; for (auto &v : q.rMu) v *= 2; for (auto &v : q.iMu) v *= 2; MriDev dq = mri_upload(q);
    mri_run(4, q, dq); fetch(q, dq, r2, i2); mri_run(4, p, d); fetch(p, d, r, im);
    worst = 0; for (int n = 0; n < p.N; n++) worst = fmax(worst, fmax(fabs(r2[n] - 2 * r[n]), fabs(i2[n] - 2 * im[n])));
    CHECK(worst < 1e-3 * mag, "F^H d is not linear in mu (%.2e)", worst);

    // one sample only: |F^H d[n]| = |mu| for every voxel (a single complex exponential has unit modulus)
    Mri s1 = make_mri(1, 8, 9); MriDev d1 = mri_upload(s1); mri_run(4, s1, d1); fetch(s1, d1, r, im);
    double mu = sqrt((double)s1.rMu[0] * s1.rMu[0] + (double)s1.iMu[0] * s1.iMu[0]); worst = 0;
    for (int n = 0; n < s1.N; n++) worst = fmax(worst, fabs(sqrt((double)r[n] * r[n] + (double)im[n] * im[n]) - mu));
    CHECK(worst < 1e-4, "single-sample magnitude identity violated by %.2e", worst);

    // mu = conj(Phi) d on the GPU
    int M = 5000; std::vector<float> a(M), b(M), c(M), e(M), ro(M), io(M); srand(3);
    for (int m = 0; m < M; m++) { a[m] = rand() / (float)RAND_MAX; b[m] = rand() / (float)RAND_MAX; c[m] = rand() / (float)RAND_MAX; e[m] = rand() / (float)RAND_MAX; }
    float *da, *db, *dc, *de, *dr, *di; size_t B = M * 4; cudaMalloc(&da, B); cudaMalloc(&db, B); cudaMalloc(&dc, B); cudaMalloc(&de, B); cudaMalloc(&dr, B); cudaMalloc(&di, B);
    cudaMemcpy(da, a.data(), B, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), B, cudaMemcpyHostToDevice); cudaMemcpy(dc, c.data(), B, cudaMemcpyHostToDevice); cudaMemcpy(de, e.data(), B, cudaMemcpyHostToDevice);
    cmp_mu_kernel<<<(M + 255) / 256, 256>>>(da, db, dc, de, dr, di, M); cudaMemcpy(ro.data(), dr, B, cudaMemcpyDeviceToHost); cudaMemcpy(io.data(), di, B, cudaMemcpyDeviceToHost);
    worst = 0; for (int m = 0; m < M; m++) worst = fmax(worst, fmax(fabs(ro[m] - (a[m] * c[m] + b[m] * e[m])), fabs(io[m] - (a[m] * e[m] - b[m] * c[m]))));
    CHECK(worst < 1e-6, "cmp_mu error %.2e", worst);
    TEST_SUMMARY("test_mri");
}

// Catch2 tests for the Cartesian-grid use of F^H d (the setting of the real-data program 01_real_slices):
// on a full Cartesian grid, (1/sqrt(UV)) F^H d is the centred orthonormal inverse DFT, and F followed by F^H scales by the sample count.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "common_mri.cuh"
#include "recon_kernels.cuh"
#include <complex>
#include <cstdlib>
#include <cmath>

static Mri grid_problem(int U, int V) {
    Mri p; p.S = V; p.N = U * V; p.M = U * V; p.x.resize(p.N); p.y.resize(p.N); p.z.assign(p.N, 0.f);
    p.kx.resize(p.M); p.ky.resize(p.M); p.kz.assign(p.M, 0.f); p.rMu.resize(p.M); p.iMu.resize(p.M);
    srand(3);
    for (int r = 0; r < U; r++) for (int c = 0; c < V; c++) {
        int i = r * V + c; p.x[i] = (float)(c - V / 2); p.y[i] = (float)(r - U / 2);
        p.ky[i] = (float)(r - U / 2) / U; p.kx[i] = (float)(c - V / 2) / V; p.rMu[i] = rand() / (float)RAND_MAX - 0.5f; p.iMu[i] = rand() / (float)RAND_MAX - 0.5f; }
    return p;
}

TEST_CASE("F^H d on a full Cartesian grid equals the centred inverse DFT", "[mri][gpu]") {
    const int U = 24, V = 16; Mri p = grid_problem(U, V); MriDev d = mri_upload(p);
    for (int variant : { 4, 5 }) {
        mri_run(variant, p, d); REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> re(p.N), im(p.N); cudaMemcpy(re.data(), d.rF, p.N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(im.data(), d.iF, p.N * 4, cudaMemcpyDeviceToHost);
        double worst = 0, scale = 0;
        for (int n = 0; n < p.N; n++) {
            std::complex<double> s = 0;
            for (int m = 0; m < p.M; m++) s += std::complex<double>(p.rMu[m], p.iMu[m]) * std::polar(1.0, 2 * M_PI * ((double)p.kx[m] * p.x[n] + (double)p.ky[m] * p.y[n]));
            worst = std::max(worst, std::abs(s - std::complex<double>(re[n], im[n]))); scale = std::max(scale, std::abs(s)); }
        INFO("variant " << variant << " worst abs error " << worst << " of scale " << scale);
        REQUIRE(worst / scale < (variant == 4 ? 1e-5 : 5e-4));
    }
    mri_free(d);
}

TEST_CASE("F^H F on a full grid multiplies by the number of samples", "[mri][gpu]") {
    const int U = 16, V = 16; Mri p = grid_problem(U, V); MriDev d = mri_upload(p); int N = p.N;
    std::vector<float> xr(N), xi(N); for (int i = 0; i < N; i++) { xr[i] = p.rMu[i]; xi[i] = p.iMu[i]; }
    float *dr, *di; cudaMalloc(&dr, N * 4); cudaMalloc(&di, N * 4); cudaMemcpy(dr, xr.data(), N * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, xi.data(), N * 4, cudaMemcpyHostToDevice);
    fwd_kernel<<<(p.M + 127) / 128, 128>>>(d.kx, d.ky, d.kz, d.x, d.y, d.z, dr, di, d.rMu, d.iMu, p.M, N, 0);
    mri_run(4, p, d); REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> re(N), im(N); cudaMemcpy(re.data(), d.rF, N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(im.data(), d.iF, N * 4, cudaMemcpyDeviceToHost);
    double worst = 0; for (int i = 0; i < N; i++) worst = std::max(worst, std::abs(std::complex<double>(re[i] - (double)N * xr[i], im[i] - (double)N * xi[i])));
    INFO("worst abs error " << worst); REQUIRE(worst < 1e-2 * N);
    cudaFree(dr); cudaFree(di); mri_free(d);
}

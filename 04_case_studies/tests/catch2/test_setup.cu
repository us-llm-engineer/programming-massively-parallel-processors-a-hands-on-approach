// Catch2 (v3, installed through vcpkg) set-up check and template for FUTURE tests.
// The existing suites in tests/test_*.cu keep their own tiny CHECK framework and are intentionally left unchanged.
// Build and run:  make catch2      (needs vcpkg at $(VCPKG_ROOT), default ~/vcpkg, with `vcpkg install catch2`)
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>
#include <vector>

__global__ void add_one(float *p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] += 1.0f; }

TEST_CASE("a CUDA device is visible", "[setup][gpu]") {
    int n = 0;
    REQUIRE(cudaGetDeviceCount(&n) == cudaSuccess);
    REQUIRE(n >= 1);
}

TEST_CASE("a trivial kernel runs and matches the host result", "[setup][gpu]") {
    const int N = 1 << 16;
    std::vector<float> h(N), ref(N);
    for (int i = 0; i < N; i++) { h[i] = (float)i * 0.5f; ref[i] = h[i] + 1.0f; }
    float *d = nullptr;
    REQUIRE(cudaMalloc(&d, N * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    add_one<<<(N + 255) / 256, 256>>>(d, N);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(h.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d);
    bool same = true;
    for (int i = 0; i < N; i++) same = same && h[i] == ref[i];
    REQUIRE(same);
}

TEST_CASE("Catch2 floating-point matchers are available", "[setup]") {
    using namespace Catch::Matchers;
    REQUIRE_THAT(0.1 + 0.2, WithinAbs(0.3, 1e-12));
    REQUIRE_THAT(1.0e6 * 1.0000001, WithinRel(1.0e6, 1e-5));
}

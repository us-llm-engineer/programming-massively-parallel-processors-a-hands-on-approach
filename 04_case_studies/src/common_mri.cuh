// common_mri.cuh - shared pieces of the MRI reconstruction programs (08a..08d)
//
// Problem (book Ch.8): non-Cartesian MRI reconstruction needs F^H d, one value per image voxel n:
//     rFHd[n] = sum_m ( rMu[m]*cos(arg) - iMu[m]*sin(arg) )
//     iFHd[n] = sum_m ( iMu[m]*cos(arg) + rMu[m]*sin(arg) )        arg = 2*pi*(kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n])
// with mu = conj(Phi) * d computed per k-space sample m  (loop fission: cmpMu). M samples x N voxels; F is never stored.
// Kernel versions (the book's optimisation ladder), all "one thread per voxel" (loop interchange, no write conflicts):
//   v1 naive            voxel data and accumulators are re-read/re-written in global memory every iteration
//   v2 registers        x,y,z and the two accumulators live in registers
//   v3 constant SoA     k-space (kx,ky,kz) in constant memory as three separate arrays, streamed in chunks (64 KB limit)
//   v4 constant AoS     k-space as an array of {x,y,z} structs in constant memory
//   v5 fast trig        v4 + hardware __sinf/__cosf (special function units), with range reduction to one cycle
#pragma once
#include "common.cuh"
#include "monitor.cuh"
#include <algorithm>
#include <functional>

#ifndef MRI_CHUNK
#define MRI_CHUNK 2048                       // k-space samples per constant-memory chunk (2048 x 12 B = 24 KB per layout)
#endif
#define PI_F 3.14159265358979f
#define TWO_PI_F 6.28318530717959f

struct KD { float x, y, z; };
#ifndef MRI_NO_SOA
__constant__ float c_kx[MRI_CHUNK], c_ky[MRI_CHUNK], c_kz[MRI_CHUNK];
#endif
__constant__ KD c_k[MRI_CHUNK];

struct Mri {                                   // one problem instance
    int M, N, S;                               // samples, voxels, voxels per side (N = S^3)
    std::vector<float> kx, ky, kz, rMu, iMu, x, y, z;
};
// voxel grid over a field of view of 1 (coordinates in [-0.5, 0.5)); k-space samples uniform in [-S/2, S/2]^3 cycles per FOV
static Mri make_mri(int M, int S, unsigned seed) {
    Mri p; p.M = M; p.S = S; p.N = S * S * S;
    p.kx.resize(M); p.ky.resize(M); p.kz.resize(M); p.rMu.resize(M); p.iMu.resize(M);
    p.x.resize(p.N); p.y.resize(p.N); p.z.resize(p.N);
    srand(seed);
    auto u = []() { return (float)rand() / ((float)RAND_MAX + 1.f); };
    for (int m = 0; m < M; m++) {
        p.kx[m] = (u() - 0.5f) * S; p.ky[m] = (u() - 0.5f) * S; p.kz[m] = (u() - 0.5f) * S;
        float ph = TWO_PI_F * u(), mag = 0.5f + u();                       // mu = conj(Phi) d: random complex value of magnitude 0.5..1.5
        p.rMu[m] = mag * cosf(ph); p.iMu[m] = mag * sinf(ph);
    }
    for (int k = 0, n = 0; k < S; k++) for (int j = 0; j < S; j++) for (int i = 0; i < S; i++, n++) {
        p.x[n] = ((float)i - S / 2) / S; p.y[n] = ((float)j - S / 2) / S; p.z[n] = ((float)k - S / 2) / S; }
    return p;
}

// ---- double-precision CPU reference for one voxel ------------------------------------------------------------------------
static void fhd_ref(const Mri &p, int n, double *r, double *im, double *scale = nullptr) {
    double sr = 0, si = 0, sc = 0;
    for (int m = 0; m < p.M; m++) {
        double arg = 2.0 * M_PI * ((double)p.kx[m] * p.x[n] + (double)p.ky[m] * p.y[n] + (double)p.kz[m] * p.z[n]), c = cos(arg), s = sin(arg);
        sr += p.rMu[m] * c - p.iMu[m] * s; si += p.iMu[m] * c + p.rMu[m] * s; sc += sqrt((double)p.rMu[m] * p.rMu[m] + (double)p.iMu[m] * p.iMu[m]);
    }
    *r = sr; *im = si; if (scale) *scale = sc;
}
// worst |GPU - reference| over `count` sampled voxels, relative to the RMS magnitude of the reference values
static double check_fhd(const Mri &p, const std::vector<float> &rF, const std::vector<float> &iF, int count, double *rms_out = nullptr) {
    std::vector<double> ref_r(count), ref_i(count); std::vector<int> ids(count);
    unsigned s = 99; double ss = 0;
    for (int c = 0; c < count; c++) { s = s * 1664525u + 1013904223u; ids[c] = (s >> 8) % p.N; fhd_ref(p, ids[c], &ref_r[c], &ref_i[c]); ss += ref_r[c] * ref_r[c] + ref_i[c] * ref_i[c]; }
    double rms = sqrt(ss / count), worst = 0;
    for (int c = 0; c < count; c++) worst = std::max(worst, std::max(fabs(rF[ids[c]] - ref_r[c]), fabs(iF[ids[c]] - ref_i[c])));
    if (rms_out) *rms_out = rms;
    return worst / rms;
}

// ---- loop fission step: mu = conj(Phi) * d, one thread per k-space sample (book Fig 8.7, cmpMu) ------------------------------
__global__ void cmp_mu_kernel(const float *rPhi, const float *iPhi, const float *rD, const float *iD, float *rMu, float *iMu, int M) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M) { rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m]; iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m]; }
}

// ---- v1: naive. Pointers may alias, so voxel data and accumulators are reloaded/stored in global memory every iteration ------
__global__ void fhd_v1(const float *rMu, const float *iMu, const float *kx, const float *ky, const float *kz, float *x, float *y, float *z,
                       float *rFHd, float *iFHd, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int m = 0; m < M; m++) {
        float e = TWO_PI_F * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);
        float c = cosf(e), s = sinf(e);
        rFHd[n] += rMu[m] * c - iMu[m] * s;
        iFHd[n] += iMu[m] * c + rMu[m] * s;
    }
}
// ---- v2: registers for the voxel data and the accumulators (book Fig 8.11) ----------------------------------------------------
__global__ void fhd_v2(const float *rMu, const float *iMu, const float *kx, const float *ky, const float *kz, const float *x, const float *y, const float *z,
                       float *rFHd, float *iFHd, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], rn = rFHd[n], in_ = iFHd[n];
    for (int m = 0; m < M; m++) {
        float e = TWO_PI_F * (kx[m] * xn + ky[m] * yn + kz[m] * zn);
        float c = cosf(e), s = sinf(e);
        rn += rMu[m] * c - iMu[m] * s;
        in_ += iMu[m] * c + rMu[m] * s;
    }
    rFHd[n] = rn; iFHd[n] = in_;
}
#ifndef MRI_NO_SOA
// ---- v3: k-space in constant memory, three separate arrays; mu stays in global memory (chunk offset m0) ------------------------
__global__ void fhd_v3(const float *rMu, const float *iMu, const float *x, const float *y, const float *z, float *rFHd, float *iFHd, int m0, int Mc, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], rn = rFHd[n], in_ = iFHd[n];
    for (int m = 0; m < Mc; m++) {
        float e = TWO_PI_F * (c_kx[m] * xn + c_ky[m] * yn + c_kz[m] * zn);
        float c = cosf(e), s = sinf(e);
        rn += rMu[m0 + m] * c - iMu[m0 + m] * s;
        in_ += iMu[m0 + m] * c + rMu[m0 + m] * s;
    }
    rFHd[n] = rn; iFHd[n] = in_;
}
#endif
// ---- v4: array of structs in constant memory (book Fig 8.16) ------------------------------------------------------------------
__global__ void fhd_v4(const float *rMu, const float *iMu, const float *x, const float *y, const float *z, float *rFHd, float *iFHd, int m0, int Mc, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], rn = rFHd[n], in_ = iFHd[n];
    for (int m = 0; m < Mc; m++) {
        float e = TWO_PI_F * (c_k[m].x * xn + c_k[m].y * yn + c_k[m].z * zn);
        float c = cosf(e), s = sinf(e);
        rn += rMu[m0 + m] * c - iMu[m0 + m] * s;
        in_ += iMu[m0 + m] * c + rMu[m0 + m] * s;
    }
    rFHd[n] = rn; iFHd[n] = in_;
}
// ---- v5: v4 + hardware trigonometry. The argument is reduced to one cycle first because __sinf/__cosf are only accurate near [-pi, pi] ---
template <int UNROLL>
__global__ void fhd_v5(const float *rMu, const float *iMu, const float *x, const float *y, const float *z, float *rFHd, float *iFHd, int m0, int Mc, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn = x[n], yn = y[n], zn = z[n], rn = rFHd[n], in_ = iFHd[n];
#pragma unroll UNROLL
    for (int m = 0; m < Mc; m++) {
        float cyc = c_k[m].x * xn + c_k[m].y * yn + c_k[m].z * zn;         // argument in cycles
        float f = TWO_PI_F * (cyc - rintf(cyc));                            // reduced to [-pi, pi]
        float c = __cosf(f), s = __sinf(f);
        rn += rMu[m0 + m] * c - iMu[m0 + m] * s;
        in_ += iMu[m0 + m] * c + rMu[m0 + m] * s;
    }
    rFHd[n] = rn; iFHd[n] = in_;
}

// ---- host drivers: stream the k-space in constant-memory chunks (64 KB limit), one kernel launch per chunk ------------------------
struct MriDev { float *rMu, *iMu, *kx, *ky, *kz, *x, *y, *z, *rF, *iF; };
static MriDev mri_upload(const Mri &p) {
    MriDev d; size_t mb = p.M * 4, nb = (size_t)p.N * 4;
    CUDA_CHECK(cudaMalloc(&d.rMu, mb)); CUDA_CHECK(cudaMalloc(&d.iMu, mb)); CUDA_CHECK(cudaMalloc(&d.kx, mb)); CUDA_CHECK(cudaMalloc(&d.ky, mb)); CUDA_CHECK(cudaMalloc(&d.kz, mb));
    CUDA_CHECK(cudaMalloc(&d.x, nb)); CUDA_CHECK(cudaMalloc(&d.y, nb)); CUDA_CHECK(cudaMalloc(&d.z, nb)); CUDA_CHECK(cudaMalloc(&d.rF, nb)); CUDA_CHECK(cudaMalloc(&d.iF, nb));
    CUDA_CHECK(cudaMemcpy(d.rMu, p.rMu.data(), mb, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.iMu, p.iMu.data(), mb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.kx, p.kx.data(), mb, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.ky, p.ky.data(), mb, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.kz, p.kz.data(), mb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.x, p.x.data(), nb, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.y, p.y.data(), nb, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.z, p.z.data(), nb, cudaMemcpyHostToDevice));
    return d;
}
static void mri_free(MriDev &d) { cudaFree(d.rMu); cudaFree(d.iMu); cudaFree(d.kx); cudaFree(d.ky); cudaFree(d.kz); cudaFree(d.x); cudaFree(d.y); cudaFree(d.z); cudaFree(d.rF); cudaFree(d.iF); }

// run version v (1..5) over all k-space samples with `block` threads per block; the result accumulates in d.rF / d.iF (zeroed first)
static void mri_run(int v, const Mri &p, MriDev &d, int block = 256, int chunk = MRI_CHUNK) {
    CUDA_CHECK(cudaMemset(d.rF, 0, (size_t)p.N * 4)); CUDA_CHECK(cudaMemset(d.iF, 0, (size_t)p.N * 4));
    int blocks = (p.N + block - 1) / block;
    if (v == 1) { fhd_v1<<<blocks, block>>>(d.rMu, d.iMu, d.kx, d.ky, d.kz, d.x, d.y, d.z, d.rF, d.iF, p.M, p.N); return; }
    if (v == 2) { fhd_v2<<<blocks, block>>>(d.rMu, d.iMu, d.kx, d.ky, d.kz, d.x, d.y, d.z, d.rF, d.iF, p.M, p.N); return; }
    std::vector<KD> aos(chunk);
    for (int m0 = 0; m0 < p.M; m0 += chunk) {
        int Mc = std::min(chunk, p.M - m0);
#ifndef MRI_NO_SOA
        if (v == 3) {
            CUDA_CHECK(cudaMemcpyToSymbol(c_kx, p.kx.data() + m0, Mc * 4)); CUDA_CHECK(cudaMemcpyToSymbol(c_ky, p.ky.data() + m0, Mc * 4)); CUDA_CHECK(cudaMemcpyToSymbol(c_kz, p.kz.data() + m0, Mc * 4));
            fhd_v3<<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N); continue;
        }
#endif
        for (int i = 0; i < Mc; i++) aos[i] = { p.kx[m0 + i], p.ky[m0 + i], p.kz[m0 + i] };
        CUDA_CHECK(cudaMemcpyToSymbol(c_k, aos.data(), Mc * sizeof(KD)));
        if (v == 4) fhd_v4<<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N);
        else        fhd_v5<1><<<blocks, block>>>(d.rMu, d.iMu, d.x, d.y, d.z, d.rF, d.iF, m0, Mc, p.N);
    }
}

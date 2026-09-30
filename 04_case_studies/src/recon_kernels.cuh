// recon_kernels.cuh - forward model F and vector helpers shared by the reconstruction programs (08e, 08f). Include after common_mri.cuh.
#pragma once
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

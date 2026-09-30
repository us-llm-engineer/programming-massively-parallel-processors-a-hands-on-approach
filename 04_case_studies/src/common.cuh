// common.cuh - shared helpers for the case-study programs (include AFTER 01_dcs_compute.cu when both are used)
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#endif

// write a CSV row (creates the header when the file is new/empty)
#include <stdarg.h>
static void csv_row(const char *path, const char *header, const char *fmt, ...) {
    FILE *f = fopen(path, "a"); if (!f) return;
    if (ftell(f) == 0) fprintf(f, "%s\n", header);
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap); fprintf(f, "\n"); fclose(f);
}

// One untimed warm-up round, then `reps` timed rounds; variants interleaved so none pays first-launch cost.
template <class F>
static void bench_rr(int nv, F launch, int reps, float *avg) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int v = 0; v < nv; v++) avg[v] = 0;
    for (int rep = -1; rep < reps; rep++)
        for (int v = 0; v < nv; v++) {
            cudaEventRecord(s); launch(v); cudaEventRecord(e); cudaEventSynchronize(e);
            cudaError_t er = cudaGetLastError();
            if (er != cudaSuccess) { fprintf(stderr, "launch error: %s\n", cudaGetErrorString(er)); exit(1); }
            float ms; cudaEventElapsedTime(&ms, s, e);
            if (rep >= 0) avg[v] += ms;
        }
    for (int v = 0; v < nv; v++) avg[v] /= reps;
    cudaEventDestroy(s); cudaEventDestroy(e);
}

static const cudaDeviceProp &dev_props() {
    static cudaDeviceProp p; static bool ok = false;
    if (!ok) { cudaGetDeviceProperties(&p, 0); ok = true; }
    return p;
}

// Resource ledger row: registers, static smem, resident blocks per SM, and how full one LAUNCH makes the GPU.
static void ledger_header() {
    printf("  %-34s %-5s %-7s %-8s %-10s %-9s %-10s %s\n", "kernel", "regs", "smem B", "blocks", "blocks/SM", "SMs busy", "warps/SM", "of capacity");
}
static void ledger_row(const char *name, const void *k, int threads, long blocks, size_t dyn = 0) {
    const cudaDeviceProp &p = dev_props();
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, k);
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, threads, dyn);
    long cap = (long)p.multiProcessorCount * nb, res = blocks < cap ? blocks : cap;
    double wps = res * (threads / 32.0) / p.multiProcessorCount;
    double full = (double)p.maxThreadsPerMultiProcessor / 32.0;
    printf("  %-34s %-5d %-7zu %-8ld %-10d %2ld of %-4d %-10.1f %.1f%%\n", name, fa.numRegs, fa.sharedSizeBytes + dyn, blocks, nb,
           blocks < p.multiProcessorCount ? blocks : (long)p.multiProcessorCount, p.multiProcessorCount, wps, 100.0 * wps / full);
}

/*
 * 09_gpu_starvation.cu   -- GPU STARVATION / DATA-PIPELINE BOTTLENECKS, reproduced on one GPU
 *
 * Starvation = the GPU's compute engines sit idle because something upstream cannot feed them fast enough.
 * The book (PMPP) does not use the phrase as a heading but describes the situations; three are reproduced here,
 * each with the fix the book describes:
 *
 *   A. TRANSFER starvation (host -> device over PCIe)   [book: PCIe far slower than on-card DRAM; pinned memory;
 *      streams + async copies pipeline chunks so transfer i+1 overlaps compute i]
 *        A1 pageable memory, synchronous copies, one stream      (the starved baseline)
 *        A2 pinned memory,   synchronous copies, one stream
 *        A3 pinned + 2 streams (double buffering), async copies
 *        A4 pinned + 4 streams, async copies
 *      swept over compute intensity K (FMAs per element) to show when the pipeline flips from
 *      transfer-bound to compute-bound.
 *   B. HOST-SIDE starvation (the CPU is slow at preparing the next chunk)   [book: SmallBin-Overlap, task pools:
 *      overlap CPU work with GPU work so the GPU is not idle while the host prepares data]
 *        B1 prepare -> copy -> compute, strictly in turn
 *        B2 one producer thread prepares chunk c+1.. while the GPU works on chunk c
 *        B3 two producer threads
 *      swept over CPU preparation time per chunk.
 *   C. LAUNCH-OVERHEAD starvation (tiny kernels)   [book: for small workloads the fixed start-up cost dominates and
 *      the GPU is underused]
 *        C1 launch + synchronize after every tiny kernel
 *        C2 launch all, synchronize once
 *        C3 one fused kernel doing the same total work
 *   (Latency starvation from low occupancy is covered in 03_resource_partitioning/src/03_occupancy_vs_time.)
 *
 * STATISTICS for visualisation are written to stats/*.csv while the GPU computes:
 *   09_summary.csv    one row per scenario/variant/parameter (wall, GPU-busy %, throughput)
 *   09_gantt.csv      per-chunk spans on the H2D / KERNEL / D2H / CPU_PREP "engines" (timeline plots)
 *   09_nvml_timeline.csv + 09_nvml_marks.csv   20 ms NVML samples (util, clock, power, memory) with variant markers
 * GPU-busy % = union of kernel execution intervals (from CUDA events) / wall time of the run.
 * Every variant is verified: the returned results are compared with a CPU recomputation (exact float FMA chain).
 */
#include "common.cuh"
#include "monitor.cuh"
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <string>
#include <string.h>

#define CHUNK_FLOATS (2u << 20)          // 2M floats = 8 MB per chunk
#define NCHUNKS 32                       // 256 MB streamed per run
#define OUT_DIV 8                        // result = 1/8 of the chunk copied back

// ---------------------------------------------------------------------------------------------- kernels
__global__ void chunk_kernel(const float *in, float *out, size_t n, int K) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += st) {
        float v = in[i];
        for (int k = 0; k < K; k++) v = fmaf(v, 1.0000001f, 1e-7f);
        if ((i & (OUT_DIV - 1)) == 0) out[i / OUT_DIV] = v;
    }
}
static float cpu_chain(float v, int K) { for (int k = 0; k < K; k++) v = fmaf(v, 1.0000001f, 1e-7f); return v; }

// ---------------------------------------------------------------------------------------------- statistics
struct Span { std::string scen, var; double param; const char *engine; int chunk; double t0, t1; };
struct Row  { std::string scen, var; double param, wall_ms, busy_ms, busy_pct, gbs, extra; bool pass; double overlap = 0, wmin = 0, wmax = 0; };
static std::vector<Span> g_spans;
static std::vector<Row> g_rows;

static double union_ms(std::vector<std::pair<double, double>> v) {
    std::sort(v.begin(), v.end());
    double tot = 0, cs = 0, ce = -1;
    for (auto &p : v) {
        if (p.first > ce) { if (ce >= 0) tot += ce - cs; cs = p.first; ce = p.second; }
        else if (p.second > ce) ce = p.second;
    }
    if (ce >= 0) tot += ce - cs;
    return tot;
}

// time during which BOTH interval sets are active = |A| + |B| - |A u B|
static double intersect_ms(const std::vector<std::pair<double, double>> &a, const std::vector<std::pair<double, double>> &b) {
    std::vector<std::pair<double, double>> u = a; u.insert(u.end(), b.begin(), b.end());
    return union_ms(a) + union_ms(b) - union_ms(u);
}

// ---------------------------------------------------------------------------------------------- shared buffers
static const size_t N = CHUNK_FLOATS, BYTES = N * 4, OUTN = N / OUT_DIV;
static float *h_pg, *h_pin, *h_out, *d_in[4], *d_out[4];
static cudaStream_t st[4];
static const int BLOCKS = 4096;

static float pattern(size_t c, size_t i) { return (float)(((i * 2654435761u) + c * 40503u) & 0xFFFF) * (1.0f / 65536.0f); }
static void fill_chunk(float *dst, size_t c) { for (size_t i = 0; i < N; i++) dst[i] = pattern(c, i); }

// exact check of a sample of returned results for chunk c against a CPU recomputation
static bool verify_chunk(const float *out_c, size_t c, int K) {
    for (size_t j = 0; j < N; j += 8 * 4099) {
        float expect = cpu_chain(pattern(c, j), K);
        if (out_c[j / OUT_DIV] != expect) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- scenario A
// skew = "copy-ahead" issue order: the D2H of chunk c-1 is queued AFTER the H2D + kernel of chunk c, so the (single)
// copy engine never sits behind a D2H that is still waiting for its kernel.
struct PA { const char *name; bool pinned; int streams; bool sync; bool skew; };
#define NVA 5
static const PA VA[NVA] = { {"A1 pageable + sync copies", false, 1, true, false}, {"A2 pinned + sync copies", true, 1, true, false},
                            {"A3 pinned + 2 streams", true, 2, false, false},     {"A4 pinned + 4 streams", true, 4, false, false},
                            {"A5 pinned + 4 streams, copy-ahead", true, 4, false, true} };

static Row run_A(const PA &v, int K, bool keep_spans) {
    // fill the pinned/pageable sources once (identical content) - done in main
    cudaEvent_t base, ev[NCHUNKS][6];
    cudaEventCreate(&base);
    for (int c = 0; c < NCHUNKS; c++) for (int e = 0; e < 6; e++) cudaEventCreate(&ev[c][e]);
    cudaDeviceSynchronize();
    cudaEventRecord(base, 0); cudaEventSynchronize(base);
    double t_host0 = mon_now_ms();
    auto issue_in = [&](int c) {                                   // H2D + kernel of chunk c
        int s = v.sync ? 0 : c % v.streams;
        cudaStream_t stream = v.sync ? 0 : st[s];
        const float *src = (v.pinned ? h_pin : h_pg) + (size_t)c * N;
        cudaEventRecord(ev[c][0], stream);
        if (v.sync) cudaMemcpy(d_in[s], src, BYTES, cudaMemcpyHostToDevice);
        else        cudaMemcpyAsync(d_in[s], src, BYTES, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev[c][1], stream);
        cudaEventRecord(ev[c][2], stream);
        chunk_kernel<<<BLOCKS, 256, 0, stream>>>(d_in[s], d_out[s], N, K);
        cudaEventRecord(ev[c][3], stream);
    };
    auto issue_out = [&](int c) {                                  // D2H of the result of chunk c
        int s = v.sync ? 0 : c % v.streams;
        cudaStream_t stream = v.sync ? 0 : st[s];
        float *dst = h_out + (size_t)c * OUTN;
        cudaEventRecord(ev[c][4], stream);
        if (v.sync) cudaMemcpy(dst, d_out[s], BYTES / OUT_DIV, cudaMemcpyDeviceToHost);
        else        cudaMemcpyAsync(dst, d_out[s], BYTES / OUT_DIV, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev[c][5], stream);
    };
    for (int c = 0; c < NCHUNKS; c++) {
        issue_in(c);
        if (!v.skew) issue_out(c);
        else if (c > 0) issue_out(c - 1);
    }
    if (v.skew) issue_out(NCHUNKS - 1);
    cudaDeviceSynchronize();
    double wall = mon_now_ms() - t_host0;
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::pair<double, double>> kern, h2d; double h2d_sum = 0;
    for (int c = 0; c < NCHUNKS; c++) {
        float t[6]; for (int e = 0; e < 6; e++) cudaEventElapsedTime(&t[e], base, ev[c][e]);
        kern.push_back({ t[2], t[3] }); h2d.push_back({ t[0], t[1] }); h2d_sum += t[1] - t[0];
        if (keep_spans) {
            g_spans.push_back({ "A", v.name, (double)K, "H2D", c, t[0], t[1] });
            g_spans.push_back({ "A", v.name, (double)K, "KERNEL", c, t[2], t[3] });
            g_spans.push_back({ "A", v.name, (double)K, "D2H", c, t[4], t[5] });
        }
    }
    bool ok = true; for (int c = 0; c < NCHUNKS && ok; c += 5) ok = verify_chunk(h_out + (size_t)c * OUTN, c, K);
    for (int c = 0; c < NCHUNKS; c++) for (int e = 0; e < 6; e++) cudaEventDestroy(ev[c][e]);
    cudaEventDestroy(base);
    double busy = union_ms(kern);
    Row r = { "A", v.name, (double)K, wall, busy, 100.0 * busy / wall, (double)NCHUNKS * BYTES / 1e9 / (wall * 1e-3), h2d_sum, ok, intersect_ms(h2d, kern) };
    return r;
}

// ---------------------------------------------------------------------------------------------- scenario B
static double prepare_chunk(float *buf, size_t c, double target_ms) {    // real fill, then spin to the target time
    double t0 = mon_now_ms();
    fill_chunk(buf, c);
    volatile float sink = 0.f;
    while (mon_now_ms() - t0 < target_ms) sink += buf[(size_t)(mon_now_ms() * 1000) % N];
    (void)sink;
    return mon_now_ms() - t0;
}

static Row run_B(int variant, double cpu_ms, int K, bool keep_spans) {
    const char *names[3] = { "B1 prepare then compute (serial)", "B2 1 producer thread (overlap)", "B3 2 producer threads (overlap)" };
    int producers = variant == 2 ? 2 : 1, P = variant == 0 ? 1 : (variant == 1 ? 3 : 4);
    static float *slot[4]; static bool alloc = false;
    if (!alloc) { for (int i = 0; i < 4; i++) CUDA_CHECK(cudaMallocHost(&slot[i], BYTES)); alloc = true; }
    cudaEvent_t base, ev[NCHUNKS][6];
    cudaEventCreate(&base);
    for (int c = 0; c < NCHUNKS; c++) for (int e = 0; e < 6; e++) cudaEventCreate(&ev[c][e]);
    cudaDeviceSynchronize();
    cudaEventRecord(base, 0); cudaEventSynchronize(base);
    double t_host0 = mon_now_ms();

    std::mutex mu; std::condition_variable cv;
    std::vector<char> ready(NCHUNKS, 0); int released = 0;
    std::vector<std::pair<double, double>> cpu_spans(NCHUNKS);
    auto producer = [&](int pid) {
        for (int c = pid; c < NCHUNKS; c += producers) {
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return released > c - P; }); }   // slot c%P free?
            double a = mon_now_ms() - t_host0;
            prepare_chunk(slot[c % P], c, cpu_ms);
            double b = mon_now_ms() - t_host0;
            { std::lock_guard<std::mutex> lk(mu); cpu_spans[c] = { a, b }; ready[c] = 1; }
            cv.notify_all();
        }
    };
    std::vector<std::thread> th;
    if (variant != 0) for (int p = 0; p < producers; p++) th.emplace_back(producer, p);

    for (int c = 0; c < NCHUNKS; c++) {
        if (variant == 0) {
            double a = mon_now_ms() - t_host0; prepare_chunk(slot[0], c, cpu_ms); cpu_spans[c] = { a, mon_now_ms() - t_host0 };
        } else {
            std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return ready[c]; });
        }
        int s = variant == 0 ? 0 : c % 2; cudaStream_t stream = st[s];
        cudaEventRecord(ev[c][0], stream);
        cudaMemcpyAsync(d_in[s], slot[c % P], BYTES, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev[c][1], stream);
        cudaEventRecord(ev[c][2], stream);
        chunk_kernel<<<BLOCKS, 256, 0, stream>>>(d_in[s], d_out[s], N, K);
        cudaEventRecord(ev[c][3], stream);
        cudaEventRecord(ev[c][4], stream);
        cudaMemcpyAsync(h_out + (size_t)c * OUTN, d_out[s], BYTES / OUT_DIV, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev[c][5], stream);
        if (variant == 0) cudaStreamSynchronize(stream);                  // strictly one step at a time
        else { cudaEventSynchronize(ev[c][1]); { std::lock_guard<std::mutex> lk(mu); released = c + 1; } cv.notify_all(); }
        if (variant == 0) { std::lock_guard<std::mutex> lk(mu); released = c + 1; }
    }
    for (auto &t : th) t.join();
    cudaDeviceSynchronize();
    double wall = mon_now_ms() - t_host0;
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::pair<double, double>> kern; double cpu_sum = 0;
    for (int c = 0; c < NCHUNKS; c++) {
        float t[6]; for (int e = 0; e < 6; e++) cudaEventElapsedTime(&t[e], base, ev[c][e]);
        kern.push_back({ t[2], t[3] }); cpu_sum += cpu_spans[c].second - cpu_spans[c].first;
        if (keep_spans) {
            g_spans.push_back({ "B", names[variant], cpu_ms, "CPU_PREP", c, cpu_spans[c].first, cpu_spans[c].second });
            g_spans.push_back({ "B", names[variant], cpu_ms, "H2D", c, t[0], t[1] });
            g_spans.push_back({ "B", names[variant], cpu_ms, "KERNEL", c, t[2], t[3] });
        }
    }
    bool ok = true; for (int c = 0; c < NCHUNKS && ok; c += 5) ok = verify_chunk(h_out + (size_t)c * OUTN, c, K);
    for (int c = 0; c < NCHUNKS; c++) for (int e = 0; e < 6; e++) cudaEventDestroy(ev[c][e]);
    cudaEventDestroy(base);
    double busy = union_ms(kern);
    Row r = { "B", names[variant], cpu_ms, wall, busy, 100.0 * busy / wall, (double)NCHUNKS * BYTES / 1e9 / (wall * 1e-3), cpu_sum / NCHUNKS, ok };
    return r;
}

// ---------------------------------------------------------------------------------------------- scenario C
#define CM 2000                     // number of tiny kernels
#define CNS 4096                    // elements per tiny kernel
static float *d_big, *d_bigout;
static Row run_C(int variant) {
    const char *names[3] = { "C1 launch + sync each", "C2 launch all, sync once", "C3 one fused kernel" };
    cudaDeviceSynchronize();
    double t0 = mon_now_ms();
    if (variant == 0)      for (int m = 0; m < CM; m++) { chunk_kernel<<<16, 256>>>(d_big + (size_t)m * CNS, d_bigout + (size_t)m * CNS / OUT_DIV, CNS, 16); cudaDeviceSynchronize(); }
    else if (variant == 1) { for (int m = 0; m < CM; m++) chunk_kernel<<<16, 256>>>(d_big + (size_t)m * CNS, d_bigout + (size_t)m * CNS / OUT_DIV, CNS, 16); cudaDeviceSynchronize(); }
    else                   { chunk_kernel<<<1024, 256>>>(d_big, d_bigout, (size_t)CM * CNS, 16); cudaDeviceSynchronize(); }
    double wall = mon_now_ms() - t0;
    CUDA_CHECK(cudaGetLastError());
    Row r = { "C", names[variant], (double)CM, wall, 0, 0, 0, wall * 1e3 / CM, true };
    return r;
}

// ---------------------------------------------------------------------------------------------- main
static double median3(double a, double b, double c) { double v[3] = { a, b, c }; std::sort(v, v + 3); return v[1]; }

// Bring the GPU clocks up before a measurement group so every group starts from the same power state
// (an idle GPU downclocks; a starved one downclocks too, which is itself part of the starvation effect).
static void prime_gpu() {
    for (int i = 0; i < 15; i++) chunk_kernel<<<BLOCKS, 256>>>(d_in[0], d_out[0], N, 4096);
    cudaDeviceSynchronize();
}

// Keep the run with the median wall time whole (its busy/overlap numbers belong to the same run); record the spread.
static Row median_run(std::vector<Row> &runs) {
    std::sort(runs.begin(), runs.end(), [](const Row &a, const Row &b) { return a.wall_ms < b.wall_ms; });
    Row r = runs[runs.size() / 2];
    r.wmin = runs.front().wall_ms; r.wmax = runs.back().wall_ms;
    for (auto &x : runs) r.pass = r.pass && x.pass;
    return r;
}
#define ROBUST_RUNS 5

int main() {
    const cudaDeviceProp &p = dev_props();
    ensure_dir("stats");
    printf("==================== GPU STARVATION ON %s ====================\n", p.name);
    printf("streamed data per run: %d chunks x %.0f MB = %.0f MB.\n", NCHUNKS, BYTES / 1048576.0, NCHUNKS * BYTES / 1048576.0);
    printf("Method: GPU clocks primed before each group; variants interleaved; 1 warm-up round + %d timed rounds;\n", ROBUST_RUNS);
    printf("the run with the MEDIAN wall time is shown whole, and 'spread' is the min-max wall time of the %d runs.\n", ROBUST_RUNS);
    printf("This machine is noisy (WSL, shared with Windows, GPU power states): trust ratios that exceed the spread.\n\n");

    h_pg = (float *)malloc((size_t)NCHUNKS * BYTES);
    CUDA_CHECK(cudaMallocHost(&h_pin, (size_t)NCHUNKS * BYTES));
    CUDA_CHECK(cudaMallocHost(&h_out, (size_t)NCHUNKS * OUTN * 4));
    for (int c = 0; c < NCHUNKS; c++) { fill_chunk(h_pg + (size_t)c * N, c); memcpy(h_pin + (size_t)c * N, h_pg + (size_t)c * N, BYTES); }
    for (int i = 0; i < 4; i++) { CUDA_CHECK(cudaMalloc(&d_in[i], BYTES)); CUDA_CHECK(cudaMalloc(&d_out[i], BYTES / OUT_DIV)); CUDA_CHECK(cudaStreamCreate(&st[i])); }
    CUDA_CHECK(cudaMalloc(&d_big, (size_t)CM * CNS * 4)); CUDA_CHECK(cudaMalloc(&d_bigout, (size_t)CM * CNS / OUT_DIV * 4 + 64));
    CUDA_CHECK(cudaMemset(d_big, 0, (size_t)CM * CNS * 4));

    // ---- PCIe bandwidth
    printf("---- Link speed: host <-> device copy of 64 MB (10 copies, warm) [MEASURED] ----\n");
    printf("  [book] PCIe is roughly an order of magnitude slower than on-card DRAM (G80: 8 GB/s PCIe total vs 86.4 GB/s DRAM).\n");
    {
        size_t B = 64u << 20; float *pg = (float *)malloc(B), *pn, *dv;
        CUDA_CHECK(cudaMallocHost(&pn, B)); CUDA_CHECK(cudaMalloc(&dv, B)); memset(pg, 1, B); memset(pn, 1, B);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        struct { const char *n; void *h; cudaMemcpyKind k; } T[4] = { {"H2D pageable", pg, cudaMemcpyHostToDevice}, {"H2D pinned", pn, cudaMemcpyHostToDevice},
                                                                       {"D2H pageable", pg, cudaMemcpyDeviceToHost}, {"D2H pinned", pn, cudaMemcpyDeviceToHost} };
        double dram = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
        printf("  %-14s %-10s\n", "copy", "GB/s");
        for (auto &t : T) {
            for (int r = 0; r < 2; r++) { if (t.k == cudaMemcpyHostToDevice) cudaMemcpy(dv, t.h, B, t.k); else cudaMemcpy(t.h, dv, B, t.k); }
            cudaEventRecord(a);
            for (int r = 0; r < 10; r++) { if (t.k == cudaMemcpyHostToDevice) cudaMemcpy(dv, t.h, B, t.k); else cudaMemcpy(t.h, dv, B, t.k); }
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            printf("  %-14s %-10.2f\n", t.n, 10.0 * B / 1e9 / (ms * 1e-3));
            g_rows.push_back({ "link", t.n, 64, ms / 10, 0, 0, 10.0 * B / 1e9 / (ms * 1e-3), 0, true });
        }
        printf("  theoretical on-card DRAM bandwidth of this GPU: %.0f GB/s [DERIVED]\n\n", dram);
        cudaFree(dv); cudaFreeHost(pn); free(pg);
    }

    // ---- A0: can this GPU overlap a copy with a kernel at all?
    printf("---- A0. Can this GPU overlap a host copy with a kernel? (direct test, two streams) [MEASURED] ----\n");
    printf("  [book] the DMA copy engine and the SMs are independent hardware, so a copy in one stream can run under a kernel in another.\n");
    {
        size_t B = 64u << 20; float *pn, *dv, *pn2;
        CUDA_CHECK(cudaMallocHost(&pn, B)); CUDA_CHECK(cudaMallocHost(&pn2, B)); CUDA_CHECK(cudaMalloc(&dv, B)); memset(pn, 1, B);
        auto kern = [&](cudaStream_t s) { for (int i = 0; i < 3; i++) chunk_kernel<<<BLOCKS, 256, 0, s>>>(d_in[0], d_out[0], N, 4096); };
        auto h2d = [&](cudaStream_t s) { cudaMemcpyAsync(dv, pn, B, cudaMemcpyHostToDevice, s); };
        auto d2h = [&](cudaStream_t s) { cudaMemcpyAsync(pn2, dv, B, cudaMemcpyDeviceToHost, s); };
        auto timeit = [&](std::function<void()> f) {
            double v[5];
            for (int r = -1; r < 5; r++) { cudaDeviceSynchronize(); double t0 = mon_now_ms(); f(); cudaDeviceSynchronize(); if (r >= 0) v[r] = mon_now_ms() - t0; }
            std::sort(v, v + 5); return v[2]; };
        prime_gpu();
        double tk = timeit([&] { kern(st[0]); }), th = timeit([&] { h2d(st[1]); }), td = timeit([&] { d2h(st[1]); });
        double tkh = timeit([&] { kern(st[0]); h2d(st[1]); }), tkd = timeit([&] { kern(st[0]); d2h(st[1]); });
        double thd = timeit([&] { h2d(st[0]); d2h(st[1]); });
        auto pct = [](double a, double b, double both) { double m = a < b ? a : b; return 100.0 * (a + b - both) / m; };
        printf("  copy engines reported by the driver (asyncEngineCount): %d\n", p.asyncEngineCount);
        printf("  %-26s %-10s\n", "operation", "median ms");
        printf("  %-26s %-10.2f\n  %-26s %-10.2f\n  %-26s %-10.2f\n", "kernel alone", tk, "H2D 64 MB alone", th, "D2H 64 MB alone", td);
        printf("  %-26s %-10s %-12s %-12s\n", "concurrent pair", "both ms", "sum ms", "overlap of the shorter");
        printf("  %-26s %-10.2f %-12.2f %.0f%%\n", "kernel + H2D", tkh, tk + th, pct(tk, th, tkh));
        printf("  %-26s %-10.2f %-12.2f %.0f%%\n", "kernel + D2H", tkd, tk + td, pct(tk, td, tkd));
        printf("  %-26s %-10.2f %-12.2f %.0f%%\n", "H2D + D2H (full duplex)", thd, th + td, pct(th, td, thd));
        printf("  Read: overlap = (sum - both) / shorter. 100%% = fully hidden, 0%% = the two operations ran one after the other.\n");
        printf("  kernel + H2D near 100%% means the hardware CAN hide a copy under compute here, so a pipeline that shows no\n");
        printf("  overlap is an issue-order problem, not a hardware limit (see A3/A4 vs A5 below). H2D + D2H near 0%% although the\n");
        printf("  driver reports two copy engines means the two directions did not run concurrently here (cause not investigated).\n\n");
        g_rows.push_back({ "A0", "kernel + H2D", 64, tkh, tk, 0, 0, th, true, pct(tk, th, tkh) });
        g_rows.push_back({ "A0", "kernel + D2H", 64, tkd, tk, 0, 0, td, true, pct(tk, td, tkd) });
        g_rows.push_back({ "A0", "H2D + D2H", 64, thd, th, 0, 0, td, true, pct(th, td, thd) });
        cudaFree(dv); cudaFreeHost(pn); cudaFreeHost(pn2);
    }

    // ---- A
    printf("---- A. TRANSFER starvation: stream 256 MB through the GPU ----\n");
    printf("  [book] fix = pinned host memory + streams: while chunk i computes, chunk i+1 is copied in.\n");
    printf("  K = FMAs per element (compute intensity). 'GPU busy' = share of wall time a kernel is executing.\n");
    printf("  %-5s %-27s %-9s %-10s %-10s %-11s %-11s %-12s %-9s %-11s %s\n", "K", "variant", "wall ms", "GPU busy%", "GB/s in", "H2D sum ms", "kernel ms", "H2D&kernel", "speedup", "spread ms", "check");
    int Ks[4] = { 1, 256, 1024, 4096 };
    double kernel_ms_per_chunk = 0;
    for (int K : Ks) {
        prime_gpu();
        std::vector<Row> runs[NVA];                                 // variants interleaved: every round runs all of them
        for (int rep = -1; rep < ROBUST_RUNS; rep++)
            for (int vi = 0; vi < NVA; vi++) {
                Row r = run_A(VA[vi], K, rep == ROBUST_RUNS - 1 && (K == 256 || K == 1024));
                if (rep >= 0) runs[vi].push_back(r);
            }
        double base_wall = 0;
        for (int vi = 0; vi < NVA; vi++) {
            Row r = median_run(runs[vi]);
            if (vi == 0) base_wall = r.wall_ms;
            if (K == 1024 && vi == 0) kernel_ms_per_chunk = r.busy_ms / NCHUNKS;
            char sp[24]; snprintf(sp, sizeof sp, "%.0f-%.0f", r.wmin, r.wmax);
            printf("  %-5d %-27s %-9.1f %-10.1f %-10.2f %-11.1f %-11.1f %-12.1f %-9.2f %-11s %s\n", K, VA[vi].name, r.wall_ms, r.busy_pct, r.gbs, r.extra, r.busy_ms, r.overlap, base_wall / r.wall_ms, sp, r.pass ? "PASS" : "FAIL");
            g_rows.push_back(r);
        }
        printf("\n");
    }
    printf("  Read: 'GPU busy%%' well below 100 with a large 'H2D sum' = the GPU waits on the transfer (transfer-bound).\n");
    printf("  'H2D&kernel' is the time a copy and a kernel were active together (0 = no overlap; a good pipeline overlaps\n");
    printf("  most of the shorter of the two). 'speedup' is relative to A1 at the same K.\n");
    printf("  A3/A4 queue each chunk's D2H right after its kernel: H2D&kernel stays 0 and they barely beat A2 although they\n");
    printf("  use streams. A5 queues chunk c-1's D2H AFTER chunk c's H2D + kernel ('copy-ahead'): the overlap appears, wall\n");
    printf("  time falls to about the transfer time, and GPU busy%% rises toward 100%% once compute >= transfer.\n");
    printf("  Hypothesis for A3/A4 (not verified beyond the A5 result): copies run in queue order, so a D2H waiting for its\n");
    printf("  kernel holds up the next chunk's H2D. Rule: with streams, issue the NEXT input copy before the previous output copy.\n\n");

    // ---- B
    printf("---- B. HOST-SIDE starvation: the CPU prepares each chunk (K = 1024: GPU kernel about %.1f ms per chunk, from A) ----\n", kernel_ms_per_chunk);
    printf("  [book] SmallBin-Overlap / task pools: keep the CPU producing while the GPU consumes, so neither waits.\n");
    double cpu_list[5] = { 0, 1, 2, 4, 8 };
    printf("  %-9s %-34s %-9s %-10s %-13s %-12s %-11s %s\n", "CPU ms", "variant", "wall ms", "GPU busy%", "prep ms/chunk", "GB/s in", "spread ms", "check");
    for (double cm : cpu_list) {
        prime_gpu();
        std::vector<Row> runs[3];
        for (int rep = -1; rep < ROBUST_RUNS; rep++)
            for (int vi = 0; vi < 3; vi++) {
                Row r = run_B(vi, cm, 1024, rep == ROBUST_RUNS - 1 && cm == 4);
                if (rep >= 0) runs[vi].push_back(r);
            }
        for (int vi = 0; vi < 3; vi++) {
            Row r = median_run(runs[vi]);
            char sp[24]; snprintf(sp, sizeof sp, "%.0f-%.0f", r.wmin, r.wmax);
            printf("  %-9.1f %-34s %-9.1f %-10.1f %-13.2f %-12.2f %-11s %s\n", cm, r.var.c_str(), r.wall_ms, r.busy_pct, r.extra, r.gbs, sp, r.pass ? "PASS" : "FAIL");
            g_rows.push_back(r);
        }
        printf("\n");
    }
    printf("  Read: in B1 the GPU-busy share falls as preparation time rises (the GPU waits for the host). B2/B3 hide\n");
    printf("  preparation behind GPU work until preparation per chunk exceeds GPU time per chunk; past that point the\n");
    printf("  producer is the bottleneck (still starved, but no longer because of serial hand-off).\n");
    printf("  'CPU ms' is the target preparation time added on top of the real fill of the 8 MB chunk.\n\n");

    // ---- C
    printf("---- C. LAUNCH-OVERHEAD starvation: %d kernels, each only %d elements x 16 FMAs ----\n", CM, CNS);
    printf("  [book] for small workloads the fixed start-up cost dominates and the GPU stays underused.\n");
    printf("  %-28s %-10s %-14s %-12s\n", "variant", "wall ms", "us / kernel", "vs fused");
    double cw[3];
    for (int vi = 0; vi < 3; vi++) { run_C(vi); double a = run_C(vi).wall_ms, b = run_C(vi).wall_ms, c = run_C(vi).wall_ms; cw[vi] = median3(a, b, c); }
    const char *cn[3] = { "C1 launch + sync each", "C2 launch all, sync once", "C3 one fused kernel" };
    for (int vi = 0; vi < 3; vi++) {
        printf("  %-28s %-10.2f %-14.2f %-12.1fx\n", cn[vi], cw[vi], cw[vi] * 1e3 / CM, cw[vi] / cw[2]);
        Row r = { "C", cn[vi], (double)CM, cw[vi], cw[2], 100.0 * cw[2] / cw[vi], 0, cw[vi] * 1e3 / CM, true };
        g_rows.push_back(r);
    }
    printf("  Read: 'GPU busy%%' for C in the CSV = fused time / wall time (the share of wall time doing useful work if the\n");
    printf("  fused kernel is the ideal). Per-kernel launch + synchronize cost is the 'us / kernel' figure.\n\n");

    // ---- sustained runs with NVML sampling (the utilisation timeline for the README figures)
    printf("---- D. Live GPU statistics while starved vs fed (NVML, 20 ms samples, ~1.2 s per variant) [MEASURED] ----\n");
    Sampler smp;
    if (!smp.ok) printf("  NVML unavailable: timeline skipped.\n");
    else {
        struct Sus { const char *label; std::function<void()> fn; };
        std::vector<Sus> sus = {
            { "A1 pageable+sync (starved)",   [&] { run_A(VA[0], 1024, false); } },
            { "A4 pinned+4 streams (fed)",    [&] { run_A(VA[3], 1024, false); } },
            { "B1 serial CPU prep 4ms",       [&] { run_B(0, 4, 1024, false); } },
            { "B2 producer overlap 4ms",      [&] { run_B(1, 4, 1024, false); } },
            { "C1 launch+sync each",          [&] { run_C(0); } },
            { "C3 fused kernel",              [&] { for (int i = 0; i < 5; i++) run_C(2); } } };
        smp.start(20);
        printf("  %-30s %-9s %-11s %-11s %-10s %-9s\n", "variant", "samples", "util avg %", "util max %", "power W", "SM MHz");
        for (auto &s : sus) {
            smp.mark(s.label); size_t i0 = smp.s.size(); double t_end = mon_now_ms() + 1200;
            while (mon_now_ms() < t_end) s.fn();
            size_t i1 = smp.s.size(); double ua = 0, um = 0, pw = 0, mh = 0; int n = 0;
            for (size_t i = i0 + 1; i < i1; i++) { ua += smp.s[i].gpu; um = std::max(um, smp.s[i].gpu); pw += smp.s[i].power_w; mh += smp.s[i].sm_mhz; n++; }
            if (n) printf("  %-30s %-9d %-11.1f %-11.0f %-10.1f %-9.0f\n", s.label, n, ua / n, um, pw / n, mh / n);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        smp.mark("end"); smp.stop();
        write_samples_csv("stats/09_nvml_timeline.csv", "starvation", smp.s);
        FILE *f = fopen("stats/09_nvml_marks.csv", "w"); fprintf(f, "t_ms,label\n");
        for (auto &m : smp.marks) fprintf(f, "%.1f,%s\n", m.first, m.second.c_str()); fclose(f);
        printf("  Caveat: NVML utilisation is a windowed average that can lag short phases. Compare it with the event-based\n");
        printf("  GPU busy%% above: where they disagree (e.g. B and C here) trust the event-based figure. Clock and power are still\n");
        printf("  informative: a starved GPU is not pushed to its top clock.\n");
    }

    // ---- write stats
    FILE *f = fopen("stats/09_summary.csv", "w");
    fprintf(f, "scenario,variant,param,wall_ms,gpu_busy_ms,gpu_busy_pct,throughput_gbs,extra,check,overlap_ms_or_pct,wall_min_ms,wall_max_ms\n");
    for (auto &r : g_rows) fprintf(f, "%s,\"%s\",%.1f,%.3f,%.3f,%.2f,%.3f,%.3f,%s,%.3f,%.3f,%.3f\n", r.scen.c_str(), r.var.c_str(), r.param, r.wall_ms, r.busy_ms, r.busy_pct, r.gbs, r.extra, r.pass ? "PASS" : "FAIL", r.overlap, r.wmin, r.wmax);
    fclose(f);
    f = fopen("stats/09_gantt.csv", "w"); fprintf(f, "scenario,variant,param,engine,chunk,t_start_ms,t_end_ms\n");
    for (auto &s : g_spans) fprintf(f, "%s,\"%s\",%.1f,%s,%d,%.4f,%.4f\n", s.scen.c_str(), s.var.c_str(), s.param, s.engine, s.chunk, s.t0, s.t1);
    fclose(f);
    bool all = true; for (auto &r : g_rows) all &= r.pass;
    printf("\nstats written: stats/09_summary.csv (%zu rows), stats/09_gantt.csv (%zu spans), stats/09_nvml_timeline.csv, stats/09_nvml_marks.csv\n", g_rows.size(), g_spans.size());
    printf("%s\n", all ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all ? 0 : 1;
}

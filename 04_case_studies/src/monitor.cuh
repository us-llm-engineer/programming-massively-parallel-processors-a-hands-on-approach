// monitor.cuh - NVML background sampler with CSV export (used by programs that record GPU statistics).
// Link with -lnvidia-ml -pthread. Metrics the driver does not report are stored as -1 and shown as n/a.
#pragma once
#include <nvml.h>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include <stdio.h>

static double mon_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void ensure_dir(const char *p) { mkdir(p, 0755); }

// ---- thermal state: a laptop GPU that heats up slows down (thermal/power management), which distorts long benchmarks ----
static bool mon_handle(nvmlDevice_t *d) {
    static bool init = false, good = false; static nvmlDevice_t dev;
    if (!init) { init = true; good = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0, &dev) == NVML_SUCCESS; }
    if (good) *d = dev;
    return good;
}
static int gpu_temp_c() { nvmlDevice_t d; unsigned v; return mon_handle(&d) && nvmlDeviceGetTemperature(d, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS ? (int)v : -1; }
static int gpu_clock_mhz() { nvmlDevice_t d; unsigned v; return mon_handle(&d) && nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &v) == NVML_SUCCESS ? (int)v : -1; }
// wait (up to max_s seconds) until the GPU is at or below limit_c so each measurement group starts from a comparable state
static void cooldown(int limit_c = 74, int max_s = 45) {
    int t = gpu_temp_c(); if (t < 0) return;
    int waited = 0;
    while (t > limit_c && waited < max_s) { std::this_thread::sleep_for(std::chrono::seconds(1)); waited++; t = gpu_temp_c(); }
    if (waited) printf("  (waited %d s for the GPU to cool to %d C before measuring)\n", waited, t);
}
static void hot_note() {
    int t = gpu_temp_c(), c = gpu_clock_mhz();
    if (t >= 78 || (c > 0 && c < 1000)) printf("  note: GPU at %d C, SM clock %d MHz right after this row (thermal/power management can slow later rows)\n", t, c);
}

struct Sample { double t_ms, gpu, mem, sm_mhz, mem_mhz, temp, power_w, used_mib; };

class Sampler {
    nvmlDevice_t dev; std::atomic<bool> stop_; std::thread th; double t0 = 0;
public:
    std::vector<Sample> s;
    std::vector<std::pair<double, std::string>> marks;      // (t_ms, label) written by the main thread
    bool ok = false;
    Sampler() { ok = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0, &dev) == NVML_SUCCESS; }
    ~Sampler() { if (ok) nvmlShutdown(); }
    Sample take() {
        Sample x = { mon_now_ms() - t0, -1, -1, -1, -1, -1, -1, -1 };
        if (!ok) return x;
        nvmlUtilization_t u; if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) { x.gpu = u.gpu; x.mem = u.memory; }
        unsigned v; if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &v) == NVML_SUCCESS) x.sm_mhz = v;
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &v) == NVML_SUCCESS) x.mem_mhz = v;
        if (nvmlDeviceGetPowerUsage(dev, &v) == NVML_SUCCESS) x.power_w = v / 1000.0;
        if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS) x.temp = v;
        nvmlMemory_t m; if (nvmlDeviceGetMemoryInfo(dev, &m) == NVML_SUCCESS) x.used_mib = m.used / 1048576.0;
        return x;
    }
    void start(int period_ms = 50) {
        s.clear(); marks.clear(); stop_ = false; t0 = mon_now_ms();
        th = std::thread([this, period_ms]() {
            while (!stop_) { s.push_back(take()); std::this_thread::sleep_for(std::chrono::milliseconds(period_ms)); } });
    }
    void mark(const char *label) { marks.push_back({ mon_now_ms() - t0, label }); }
    double now() const { return mon_now_ms() - t0; }
    void stop() { stop_ = true; th.join(); s.push_back(take()); }
};

// RAII summary of how busy the GPU was during a whole program run: declare `RunMonitor mon;` first in main().
// (NVML utilisation = share of each 100 ms window in which at least one kernel was running.)
struct RunMonitor {
    Sampler s; double t0;
    explicit RunMonitor(int period_ms = 100) { t0 = mon_now_ms(); if (s.ok) s.start(period_ms); }
    ~RunMonitor() {
        if (!s.ok) return;
        s.stop();
        double n = 0, u = 0, hi = 0, lo = 0, c = 0, pw = 0, tmax = 0;
        for (auto &x : s.s) { if (x.gpu < 0) continue; n++; u += x.gpu; hi += x.gpu >= 50; lo += x.gpu <= 5; c += x.sm_mhz; pw += x.power_w; if (x.temp > tmax) tmax = x.temp; }
        if (n > 0)
            printf("\n[GPU during this run: %.1f s wall | mean utilisation %.0f %% | >= 50 %% for %.0f %% of the time, idle (<= 5 %%) for %.0f %% | "
                   "mean SM clock %.0f MHz | mean power %.1f W | max temp %.0f C | NVML, 100 ms samples]\n",
                   (mon_now_ms() - t0) / 1000, u / n, 100 * hi / n, 100 * lo / n, c / n, pw / n, tmax);
    }
};

// Append samples to a CSV (writes the header when the file is new/empty).
static void write_samples_csv(const char *path, const char *label, const std::vector<Sample> &v) {
    FILE *f = fopen(path, "a"); if (!f) return;
    if (ftell(f) == 0) fprintf(f, "label,t_ms,gpu_util_pct,mem_util_pct,sm_clock_mhz,mem_clock_mhz,temp_c,power_w,mem_used_mib\n");
    for (auto &x : v) fprintf(f, "%s,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,%.2f,%.1f\n", label, x.t_ms, x.gpu, x.mem, x.sm_mhz, x.mem_mhz, x.temp, x.power_w, x.used_mib);
    fclose(f);
}

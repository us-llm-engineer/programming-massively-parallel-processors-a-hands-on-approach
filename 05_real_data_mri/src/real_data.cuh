// real_data.cuh - reading real multi-coil Cartesian k-space scans and reconstruction helpers shared by 01_real_slices and 02_real_archive.
// File layout written by scripts/prepare_m4raw.py:  int32 C,U,V,S | float32 kspace[S][C][U][V][2] | float32 reference[S][U][V].
#pragma once
#include "common_mri.cuh"
#include <cmath>
#include <string>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Slice { int C, U, V; std::vector<float> ks, ref; };                 // ks[c][u][v][2]
// read the header (C, U, V, S) of a scan file; returns false if missing
static bool scan_header(const std::string &path, int h[4]) {
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false; bool ok = fread(h, 4, 4, f) == 4; fclose(f); return ok;
}
// read one slice of a scan file (seeks; does not load the others)
static bool load_slice(const std::string &path, int idx, Slice &s) {
    int h[4]; if (!scan_header(path, h) || idx >= h[3]) return false;
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    s.C = h[0]; s.U = h[1]; s.V = h[2]; size_t ks_n = (size_t)s.C * s.U * s.V * 2, ref_n = (size_t)s.U * s.V, S = h[3];
    s.ks.resize(ks_n); s.ref.resize(ref_n);
    bool ok = fseek(f, 16 + (long)(idx * ks_n * 4), SEEK_SET) == 0 && fread(s.ks.data(), 4, ks_n, f) == ks_n
           && fseek(f, 16 + (long)(S * ks_n * 4 + idx * ref_n * 4), SEEK_SET) == 0 && fread(s.ref.data(), 4, ref_n, f) == ref_n;
    fclose(f); return ok;
}
struct Metric { double rel, psnr; };
static Metric compare(const std::vector<float> &ref, const std::vector<float> &img) {
    double mse = 0, ss = 0, mx = 0; size_t n = ref.size();
    for (size_t i = 0; i < n; i++) { double e = (double)img[i] - ref[i]; mse += e * e; ss += (double)ref[i] * ref[i]; mx = std::max(mx, (double)ref[i]); }
    mse /= n; return { 100.0 * sqrt(mse) / sqrt(ss / n), 20 * log10(mx / std::max(sqrt(mse), 1e-30)) };
}
// voxels on the pixel grid (row = y, col = x, centred); samples are the kept (row, col) k-space positions in cycles per pixel
static Mri make_problem(const Slice &s, const std::vector<char> &keep_row) {
    Mri p; p.S = s.V; p.N = s.U * s.V; p.x.resize(p.N); p.y.resize(p.N); p.z.assign(p.N, 0.f);
    for (int r = 0; r < s.U; r++) for (int c = 0; c < s.V; c++) { p.x[r * s.V + c] = (float)(c - s.V / 2); p.y[r * s.V + c] = (float)(r - s.U / 2); }
    for (int u = 0; u < s.U; u++) if (keep_row[u]) for (int v = 0; v < s.V; v++) { p.ky.push_back((float)(u - s.U / 2) / s.U); p.kx.push_back((float)(v - s.V / 2) / s.V); }
    p.M = (int)p.kx.size(); p.kz.assign(p.M, 0.f); p.rMu.assign(p.M, 0.f); p.iMu.assign(p.M, 0.f); return p;
}
static void gather_data(const Slice &s, int coil, const std::vector<char> &keep_row, float scale, std::vector<float> &re, std::vector<float> &im) {
    re.clear(); im.clear();
    for (int u = 0; u < s.U; u++) if (keep_row[u]) for (int v = 0; v < s.V; v++) {
        size_t i = (((size_t)coil * s.U + u) * s.V + v) * 2; re.push_back(scale * s.ks[i]); im.push_back(scale * s.ks[i + 1]); }
}

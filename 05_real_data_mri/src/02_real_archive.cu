/*
 * 02_real_archive.cu   -- reconstruct EVERY slice of a real multi-coil scan with the F^H d kernels and score it
 *
 * Usage: 02_real_archive <scan.bin> <scan-name> <metrics.csv> [pause-fraction] [image-dir]      (scan.bin from scripts/prepare_m4raw.py; rows are appended to the CSV)
 * Driven over a whole dataset by scripts/run_real_archive.sh. The optional pause fraction idles the GPU after each slice for that fraction of the slice's GPU time (a duty-cycle limit that keeps utilisation and temperature down). For each slice it reconstructs, per coil, on the GPU
 *   (a) FULL data:   img = (1/sqrt(UV)) F^H d, coils combined by root-sum-of-squares (accurate trigonometry)
 *   (b) UNDERSAMPLED (central 24 lines + a random 20 % of the others, fixed seed): zero-filled F^H d
 * and compares with the dataset's own FFT-based reconstruction (relative RMS error, PSNR = 20 log10(max(ref)/sqrt(MSE)), no rescaling).
 * Stand-alone run (no arguments) does nothing; the case-study summary of the whole archive is in stats/archive_summary.csv.
 */
#include "common_mri.cuh"
#include "real_data.cuh"
#include <cstring>

#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 4) { printf("usage: %s <scan.bin> <scan-name> <metrics.csv> [pause-fraction] [image-dir]\n", argv[0]); return 2; }
    const double pause = argc > 4 ? atof(argv[4]) : 0.0;         // idle time after each slice as a fraction of its GPU time (duty-cycle limit)
    const char *imgdir = argc > 5 ? argv[5] : nullptr;              // if given: 8-bit images of every slice (reference, GPU full-data, GPU zero-filled) are written there
    FILE *fref = nullptr, *ffull = nullptr, *fzf = nullptr;
    int h[4]; if (!scan_header(argv[1], h)) { printf("cannot read %s\n", argv[1]); return 2; }
    const int C = h[0], U = h[1], V = h[2], S = h[3], N = U * V; size_t nb = (size_t)N * 4;
    const char *HDR = "scan,slice,method,lines_kept,samples,rel_rms_pct,psnr_db,gpu_ms,wall_ms,gpairs_per_s,gflops,ref_max,img_max";
    std::vector<char> all(U, 1), under(U, 0); srand(5);
    for (int u = 0; u < U; u++) under[u] = (abs(u - U / 2) < 12) || (rand() % 100 < 20);
    int kept = 0; for (char c : under) kept += c;
    if (imgdir) { char f[512]; snprintf(f, sizeof f, "%s/ref.u8", imgdir); fref = fopen(f, "wb"); snprintf(f, sizeof f, "%s/full.u8", imgdir); ffull = fopen(f, "wb"); snprintf(f, sizeof f, "%s/zf.u8", imgdir); fzf = fopen(f, "wb"); }
    bool ok = true; std::vector<float> rss(N), re(N), im(N), dr, di;
    for (int sl = 0; sl < S; sl++) {
        Slice s; if (!load_slice(argv[1], sl, s)) { printf("slice %d unreadable\n", sl); return 1; }
        double slice_gpu = 0;
        for (int mode = 0; mode < 2; mode++) {
            double w0 = now_ms(); Mri p = make_problem(s, mode ? under : all); MriDev d = mri_upload(p); std::fill(rss.begin(), rss.end(), 0.f); double gpu = 0;
            for (int c = 0; c < C; c++) {
                gather_data(s, c, mode ? under : all, 1.f, dr, di);
                CUDA_CHECK(cudaMemcpy(d.rMu, dr.data(), p.M * 4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d.iMu, di.data(), p.M * 4, cudaMemcpyHostToDevice));
                double a = now_ms(); mri_run(4, p, d); CUDA_CHECK(cudaDeviceSynchronize()); gpu += now_ms() - a;
                CUDA_CHECK(cudaMemcpy(re.data(), d.rF, nb, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(im.data(), d.iF, nb, cudaMemcpyDeviceToHost));
                for (int i = 0; i < N; i++) { double a2 = re[i] / sqrt((double)N), b2 = im[i] / sqrt((double)N); rss[i] += (float)(a2 * a2 + b2 * b2); }
            }
            for (int i = 0; i < N; i++) rss[i] = sqrtf(rss[i]);
            slice_gpu += gpu; Metric q = compare(s.ref, rss);
            double pairs = (double)p.M * N * C, rmax = 0, imax = 0; for (int i = 0; i < N; i++) { rmax = std::max(rmax, (double)s.ref[i]); imax = std::max(imax, (double)rss[i]); }
            csv_row(argv[3], HDR, "%s,%d,%s,%d,%d,%.6f,%.3f,%.1f,%.1f,%.3f,%.1f,%.2f,%.2f", argv[2], sl, mode ? "zero-filled" : "full", mode ? kept : U, p.M, q.rel, q.psnr,
                    gpu, now_ms() - w0, pairs / (gpu * 1e6), pairs * 13 / (gpu * 1e6), rmax, imax);      // 13 flops per voxel-sample pair, as in the book's count
            if (imgdir) {                                                     // 8-bit images, window = maximum of the reference image of this slice
                std::vector<unsigned char> u(N); for (int i = 0; i < N; i++) u[i] = (unsigned char)std::min(255.0, std::max(0.0, 255.0 * rss[i] / rmax));
                fwrite(u.data(), 1, N, mode ? fzf : ffull);
                if (mode == 0) { for (int i = 0; i < N; i++) u[i] = (unsigned char)std::min(255.0, std::max(0.0, 255.0 * s.ref[i] / rmax)); fwrite(u.data(), 1, N, fref); } }
            if (mode == 0 && q.rel > 0.5) ok = false;
            mri_free(d);
        }
        if (pause > 0) usleep((useconds_t)(slice_gpu * pause * 1000.0));
    }
    if (imgdir) { fclose(fref); fclose(ffull); fclose(fzf); }
    printf("%s: %d slices, %d coils, %dx%d %s\n", argv[2], S, C, U, V, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

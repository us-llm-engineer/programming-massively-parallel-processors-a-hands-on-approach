# Summarises a whole-archive run (stats/archive/{metrics,scans,gpu_timeline,wall}.csv) into
# stats/archive_summary.csv (key/value table) and figures/real_archive_dist.png, figures/real_archive_timeline.png.
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = os.path.join(os.path.dirname(__file__), ".."); R = sys.argv[1] if len(sys.argv) > 1 else f"{S}/stats/archive"
SUMMARY_ONLY = len(sys.argv) > 2                                   # plot_archive.py <run-dir> <summary-name.csv>: write only that summary, no figures
m = pd.read_csv(f"{R}/metrics.csv"); sc = pd.read_csv(f"{R}/scans.csv")
sc["wall_s"] = pd.to_numeric(sc.wall_s)         # written as e.g. 29238e-3 (seconds)
wall_run = float(open(f"{R}/wall.csv").read().split(",")[1].replace("e-3", "")) / 1000
t = pd.read_csv(f"{R}/gpu_timeline.csv", skipinitialspace=True); t.columns = ["ts", "util", "sm", "power", "temp", "mem"]
for c in ("util", "sm", "power", "temp", "mem"): t[c] = pd.to_numeric(t[c].astype(str).str.extract(r"([\d.]+)")[0])
t["ts"] = pd.to_datetime(t.ts); t["s"] = (t.ts - t.ts.iloc[0]).dt.total_seconds()
m["contrast"] = m.scan.str.extract(r"_(FLAIR|T1|T2)\d+$")[0]
full, zf = m[m.method == "full"], m[m.method == "zero-filled"]
rows = []
def add(k, v, unit="", label="MEASURED"): rows.append((k, v if isinstance(v, str) else round(float(v), 4), unit, label))
add("scans_reconstructed", sc.status.eq("PASS").sum(), "scans"); add("scans_failed", (~sc.status.eq("PASS")).sum(), "scans")
add("slices_reconstructed_full_and_undersampled", len(full), "slices"); add("coil_images_reconstructed", len(m) * 4, "images", "DERIVED")
pairs = (m.gpairs_per_s * m.gpu_ms * 1e6).sum(); add("voxel_sample_pairs_computed", pairs / 1e12, "x1e12", "DERIVED")
add("equivalent_flops", pairs * 13 / 1e12, "TFLOP (13 flops per pair)", "DERIVED")
add("run_wall_time", wall_run / 60, "min"); add("gpu_kernel_time", m.gpu_ms.sum() / 60000, "min"); add("mean_scan_wall", sc.wall_s.mean(), "s")
add("throughput_full_slices", len(full) / wall_run, "slices/s", "DERIVED"); add("median_kernel_rate_full", full.gpairs_per_s.median(), "Gpairs/s"); add("median_kernel_rate_zerofilled", zf.gpairs_per_s.median(), "Gpairs/s")
add("peak_kernel_rate", m.gpairs_per_s.max(), "Gpairs/s"); add("median_kernel_gflops", m.gflops.median(), "GFLOP/s (13 flops/pair)", "DERIVED")
for q in (50, 95, 99, 100): add(f"full_rel_error_p{q}", np.percentile(full.rel_rms_pct, q), "% vs dataset FFT reconstruction")
add("full_psnr_min", full.psnr_db.min(), "dB")
for q in (5, 50, 95): add(f"zerofilled_psnr_p{q}", np.percentile(zf.psnr_db, q), "dB (3.7x fewer lines)")
for c in ("T1", "T2", "FLAIR"):
    f, z = full[full.contrast == c], zf[zf.contrast == c]
    add(f"{c}_scans", f.scan.nunique(), "scans"); add(f"{c}_full_rel_error_max", f.rel_rms_pct.max(), "%"); add(f"{c}_zerofilled_psnr_median", z.psnr_db.median(), "dB")
add("gpu_util_mean", t.util.mean(), "%"); add("gpu_util_p95", np.percentile(t.util, 95), "%"); add("gpu_util_max", t.util.max(), "%")
add("sm_clock_mean", t.sm.mean(), "MHz"); add("sm_clock_min", t.sm.min(), "MHz"); add("power_mean", t.power.mean(), "W"); add("power_max", t.power.max(), "W")
add("temp_mean", t.temp.mean(), "C"); add("temp_max", t.temp.max(), "C");
add("energy_estimate", t.power.mean() * wall_run / 3600, "Wh (mean power x wall time)", "DERIVED")
add("energy_per_slice", t.power.mean() * wall_run / len(full), "J per slice (full + undersampled)", "DERIVED")
pd.DataFrame(rows, columns=["statistic", "value", "unit", "label"]).to_csv(f"{S}/stats/{sys.argv[2] if SUMMARY_ONLY else 'archive_summary.csv'}", index=False)
if SUMMARY_ONLY: sys.exit(0)
os.makedirs(f"{S}/figures", exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
ax[0].hist(full.rel_rms_pct, bins=60, color="#4c78a8"); ax[0].set_yscale("log"); ax[0].set_xlabel("relative error (%)"); ax[0].set_ylabel("slices")
ax[0].set_title(f"GPU F^H d vs dataset FFT, {len(full)} slices\nmax {full.rel_rms_pct.max():.4f} %", fontsize=9)
ax[1].hist(full.psnr_db, bins=60, color="#59a14f"); ax[1].set_xlabel("PSNR (dB)"); ax[1].set_title("full-data PSNR (float32 limit near 125 dB)", fontsize=9)
for c, col in (("T1", "#4c78a8"), ("T2", "#f28e2b"), ("FLAIR", "#e15759")): ax[2].hist(zf[zf.contrast == c].psnr_db, bins=40, alpha=.65, label=c, color=col)
ax[2].set_xlabel("PSNR (dB)"); ax[2].legend(); ax[2].set_title("zero-filled, 3.7x fewer lines", fontsize=9)
ax[3].hist(full.gpairs_per_s, bins=40, color="#b07aa1"); ax[3].set_xlabel("G voxel-sample pairs / s"); ax[3].set_title("kernel throughput per slice", fontsize=9)
fig.tight_layout(); fig.savefig(f"{S}/figures/real_archive_dist.png", dpi=110)
fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True); x = t.s / 60
ax[0].plot(x, t.util, lw=.6); ax[0].axhline(60, color="gray", ls=":"); ax[0].set_ylabel("GPU util (%)"); ax[0].set_title("GPU during the whole-archive reconstruction (nvidia-smi, 1 Hz; dotted = 60 % cap)", fontsize=10)
ax[1].plot(x, t.sm, lw=.6, color="#f28e2b"); ax[1].set_ylabel("SM clock (MHz)"); ax2 = ax[1].twinx(); ax2.plot(x, t.temp, lw=.6, color="#e15759"); ax2.set_ylabel("temp (C)")
ax[2].plot(x, t.power, lw=.6, color="#59a14f"); ax[2].set_ylabel("power (W)"); ax[2].set_xlabel("minutes")
fig.tight_layout(); fig.savefig(f"{S}/figures/real_archive_timeline.png", dpi=110); print("wrote summary and figures")

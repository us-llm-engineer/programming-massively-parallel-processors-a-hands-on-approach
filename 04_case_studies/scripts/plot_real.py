# Renders the real-data reconstructions written by src/08f_mri_real_data.cu (stats/08f_slice*_*.bin, float32 256x256).
import os, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = os.path.join(os.path.dirname(__file__), "..")
rows = list(csv.DictReader(open(f"{S}/stats/08f_real_metrics.csv")))
def m(sl, prefix, col):
    for r in rows:
        if int(r["slice"]) == sl and r["method"].startswith(prefix): return float(r[col])
slices = sorted({int(r["slice"]) for r in rows})
os.makedirs(f"{S}/figures", exist_ok=True)
fig, ax = plt.subplots(len(slices), 4, figsize=(13, 3.3 * len(slices)))
for i, sl in enumerate(slices):
    ld = lambda t: np.fromfile(f"{S}/stats/08f_slice{sl:02d}_{t}.bin", dtype=np.float32).reshape(256, 256)
    ref = ld("reference"); vmax = ref.max()
    panels = [("reference", "dataset reconstruction (FFT)", ""), ("full", "our GPU F^H d, full data", f"\nrel. error {m(sl,'F^H d, RSS','rel_rms_pct'):.4f} %"),
              ("zerofill", "zero-filled, 3.7x fewer lines", f"\nPSNR {m(sl,'zero-filled','psnr_db'):.1f} dB"), ("cg8", "CG on the same undersampled data", f"\nPSNR {m(sl,'CG on undersampled data, 8','psnr_db'):.1f} dB")]
    for a, (t, title, extra) in zip(ax[i], panels):
        a.imshow(ld(t), cmap="gray", vmin=0, vmax=vmax, origin="upper"); a.axis("off"); a.set_title(f"slice {sl}: {title}{extra}", fontsize=8)
fig.suptitle("Real in-vivo brain k-space (M4Raw, 4 coils): reconstruction with the book's F^H d kernels on the GPU", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{S}/figures/08_real_data.png", dpi=105); print("wrote figures/08_real_data.png")

# Renders the reconstructed images written by src/08e_mri_reconstruction.cu (stats/08e_*.bin, float32 magnitude slices).
import os, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(f"{S}/figures", exist_ok=True)
meta = {}
for r in csv.DictReader(open(f"{S}/stats/08e_images_meta.csv")): meta[(r["case"], r["tag"])] = int(r["size"])
rows = list(csv.DictReader(open(f"{S}/stats/08e_recon_metrics.csv")))
def psnr(case, method, trig, it):
    for r in rows:
        if r["case"] == case and r["method"].startswith(method) and r["trig"] == trig and int(r["iteration"]) == it: return float(r["psnr_db"])
def load(case, tag):
    n = meta[(case, tag)]; return np.fromfile(f"{S}/stats/08e_{case}_{tag}.bin", dtype=np.float32).reshape(n, n)
for case in ("2D", "3D"):
    last = 40 if case == "2D" else 25
    panels = [("phantom", "phantom (truth)", None), ("adjoint", "plain adjoint F^H d", psnr(case, "plain", "accurate", 0)),
              ("dcf", "density-compensated", psnr(case, "density", "accurate", 0)), ("cg05_acc", "CG, 5 iterations", psnr(case, "CG", "accurate", 5)),
              ("cg10_acc", "CG, 10 iterations", psnr(case, "CG", "accurate", 10)), (f"cg{last:02d}_acc", f"CG, {last} iterations", psnr(case, "CG", "accurate", last))]
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.4))
    for a, (tag, title, p) in zip(ax, panels):
        a.imshow(load(case, tag), cmap="gray", vmin=0, vmax=1.1, origin="lower"); a.axis("off")
        a.set_title(title + ("" if p is None else f"\nPSNR {p:.1f} dB"), fontsize=9)
    fig.suptitle(f"{case} reconstruction from simulated radial k-space (central slice); GTX 1650 Ti, F^H via the Ch.8 kernels", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{S}/figures/08_reconstruction_{case}.png", dpi=110); plt.close(fig)
    print("wrote", f"figures/08_reconstruction_{case}.png")
# PSNR vs CG iteration, accurate vs hardware trig
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, case in zip(ax, ("2D", "3D")):
    for trig, st in (("accurate", "-"), ("hardware", "--")):
        pts = sorted((int(r["iteration"]), float(r["psnr_db"])) for r in rows if r["case"] == case and r["method"].startswith("CG") and r["trig"] == trig)
        a.plot(*zip(*pts), st, label=f"CG, {trig} trig")
    a.axhline(psnr(case, "density", "accurate", 0), color="gray", ls=":", label="density-compensated adjoint")
    a.set_title(case); a.set_xlabel("CG iteration"); a.set_ylabel("PSNR (dB)"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{S}/figures/08_reconstruction_psnr.png", dpi=110); print("wrote figures/08_reconstruction_psnr.png")

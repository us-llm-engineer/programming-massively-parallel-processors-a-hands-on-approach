#!/usr/bin/env python3
"""Turn the 8-bit slice images written by 02_real_archive into PNGs.
  make_sheet.py <raw-dir> <scan-name> <png-dir> [metrics-csv]
Writes <png-dir>/<scan>.png : contact sheet of all slices of the scan reconstructed on the GPU from full k-space (6 x 3 grid, tiles downscaled to 192 px to keep the repository small; the showcase slices are full 256 px).
For the scans listed in SHOWCASE it also keeps slice 8 as three PNGs (dataset reference, GPU full-data, GPU zero-filled) in <png-dir>/showcase/."""
import sys, os, numpy as np
from PIL import Image, ImageDraw
raw, name, out = sys.argv[1:4]; csv = sys.argv[4] if len(sys.argv) > 4 else None
SHOWCASE = ("2022061203_T101", "2022061203_T201", "2022061203_FLAIR01")
U = V = 256
full = np.fromfile(f"{raw}/full.u8", np.uint8); S = full.size // (U * V); full = full.reshape(S, U, V)
os.makedirs(out, exist_ok=True)
cols = 6; rows = (S + cols - 1) // cols; T = 192; sheet = np.zeros((rows * T, cols * T), np.uint8)
for i in range(S): r, c = divmod(i, cols); sheet[r * T:(r + 1) * T, c * T:(c + 1) * T] = np.asarray(Image.fromarray(full[i]).resize((T, T), Image.LANCZOS))
im = Image.fromarray(sheet).convert("L"); d = ImageDraw.Draw(im)
for i in range(S): r, c = divmod(i, cols); d.text((c * T + 6, r * T + 4), f"{i}", fill=255)
im.save(f"{out}/{name}.png", optimize=True)
if name in SHOWCASE:
    os.makedirs(f"{out}/showcase", exist_ok=True)
    for tag, f in (("reference", "ref"), ("gpu_full", "full"), ("zero_filled", "zf")):
        a = np.fromfile(f"{raw}/{f}.u8", np.uint8).reshape(S, U, V)[8]; Image.fromarray(a).save(f"{out}/showcase/{name}_slice08_{tag}.png", optimize=True)

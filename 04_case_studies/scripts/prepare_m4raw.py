#!/usr/bin/env python3
"""Glue: convert slices of the downloaded M4Raw file (HDF5) into flat binaries the CUDA program reads.
Layout: int32 C,U,V | float32 kspace[C][U][V][2] (re,im) | float32 reference[U][V] (the dataset's own root-sum-of-squares image).
Usage: prepare_m4raw.py [h5-file] [out-dir] [slice ...]"""
import sys, h5py, numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else "data/m4raw/multicoil_val/2022061203_T101.h5"
out = sys.argv[2] if len(sys.argv) > 2 else "data/m4raw"
slices = [int(s) for s in sys.argv[3:]] or [4, 8, 12]
f = h5py.File(src, "r"); k = f["kspace"]; ref = f["reconstruction_rss"]
for s in slices:
    d = np.ascontiguousarray(k[s]); C, U, V = d.shape
    with open(f"{out}/slice_{s:02d}.bin", "wb") as o:
        o.write(np.array([C, U, V], "<i4").tobytes()); o.write(np.stack([d.real, d.imag], -1).astype("<f4").tobytes()); o.write(ref[s].astype("<f4").tobytes())
    print(f"slice {s}: {C} coils, {U}x{V}")

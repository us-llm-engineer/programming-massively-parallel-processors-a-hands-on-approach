#!/usr/bin/env python3
"""Glue: convert slices of the downloaded M4Raw file (HDF5) into flat binaries the CUDA program reads.
Layout: int32 C,U,V,S | float32 kspace[S][C][U][V][2] (re,im) | float32 reference[S][U][V] (the dataset's own root-sum-of-squares image).
Usage: prepare_m4raw.py [h5-file] [out-dir]"""
import sys, h5py, numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else "data/m4raw/multicoil_val/2022061203_T101.h5"
out = sys.argv[2] if len(sys.argv) > 2 else "data/m4raw"
f = h5py.File(src, "r"); k = f["kspace"][:]; ref = f["reconstruction_rss"][:]        # (S, C, U, V) complex64, (S, U, V) float32
S, C, U, V = k.shape
with open(f"{out}/scan.bin", "wb") as o:
    o.write(np.array([C, U, V, S], "<i4").tobytes()); o.write(np.stack([k.real, k.imag], -1).astype("<f4").tobytes()); o.write(ref.astype("<f4").tobytes())
print(f"{src}: {S} slices, {C} coils, {U}x{V}")

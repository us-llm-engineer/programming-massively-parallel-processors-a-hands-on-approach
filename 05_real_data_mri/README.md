# 05 - Reconstruction of real measured MRI data (book Ch. 8)

The earlier reconstructions (`04_case_studies/src/08e`) use a synthetic phantom, and the Parboil MRI-Q inputs (`08a`-`08d`) contain scan
geometry only, with no measured signal. This folder reconstructs images from **real scanner data** with the book's F^H d kernels,
first for three slices and then for every slice of 92 scans.

**Data:** [M4Raw](https://zenodo.org/records/8056074) validation set (in-vivo brain, 0.3 T scanner, 4 receive coils, Cartesian 256 x 256,
T1, T2 and FLAIR contrasts, CC-BY 4.0). Truth for every slice is the dataset's own reconstruction (inverse FFT + root-sum-of-squares over
coils), an independent FFT-based reference. No rescaling is applied before comparing.

| Program | What it does | Result files |
| --- | --- | --- |
| `src/01_real_slices.cu` | slices 4, 8, 12 of one scan: full data (accurate and hardware trigonometry, double-precision CPU check), 3.7x undersampled zero-filled, conjugate gradient | `output/01_real_slices.txt`, `stats/01_real_metrics.csv`, `figures/real_slices.png` |
| `src/02_real_archive.cu` | every slice of one scan: full data and zero-filled, with accuracy, kernel time, throughput per slice | `stats/archive/metrics.csv` |
| `scripts/run_real_archive.sh` | drives 02 over all complete scans of an archive, with a duty-cycle limit, and samples the GPU at 1 Hz | `stats/archive/{scans,gpu_timeline,wall}.csv` |
| `scripts/make_sheet.py` | turns the 8-bit slice images of one scan into a PNG contact sheet (all 18 slices) and full-size showcase PNGs | `figures/sheets/` (6 of 92 shipped), Drive folder below |
| `scripts/plot_archive.py` | summary statistics and figures | `stats/archive_summary.csv`, `stats/archive_png_run_summary.csv`, `figures/real_archive_*.png` |

Shared code (`common_mri.cuh`, `recon_kernels.cuh`) is in `../04_case_studies/src`; `src/real_data.cuh` reads the scan files.

## How the reconstruction works

For a Cartesian scan every k-space sample (u, v) is a point at ((v-128)/256, (u-128)/256) cycles per pixel, and every pixel is a voxel at
its centred pixel coordinates. Per coil, `F^H d` is computed with the book's kernels (65,536 voxels x 65,536 samples = 4.3 G
sine/cosine pairs per coil), scaled by 1/256, and the four coil images are combined by root-sum-of-squares. On a full grid this equals the
centred inverse DFT, so agreement with the dataset's FFT reconstruction validates the kernels on measured data. The undersampled variant keeps
the central 24 phase-encode lines plus a random 20 % of the others (70 of 256 lines, 3.7x fewer, fixed seed) and zero-fills the rest.

## Results

**Three slices** (`figures/real_slices.png`, `stats/01_real_metrics.csv`; per slice, slices 4, 8, 12 of `2022061203_T101`):

| Reconstruction | samples per coil | relative error | PSNR (dB) |
| --- | --- | --- | --- |
| GPU F^H d, full data, accurate trigonometry | 65,536 | 0.0002 % | 125.5-126.7 |
| GPU F^H d, full data, hardware sin/cos | 65,536 | 0.0002 % | 126.3-127.8 |
| zero-filled F^H d, 70 of 256 lines | 17,920 | 25.4-26.5 % | 24.6-24.7 |
| CG on the undersampled data | 17,920 | same as zero-filled | same |

GPU F^H d agrees with a double-precision CPU sum on 64 voxels to about 1e-7 (scaled by the sum of |terms|). CG converges in one iteration
(the normal matrix has one non-zero eigenvalue on the kept lines) and returns the zero-filled image, so the aliasing remains; removing it needs
coil-sensitivity maps or a sparsity prior, which are not implemented.

**Every slice of 92 scans** (`stats/archive_summary.csv`, one worker, all 92 scans PASS):

| Statistic | Value | Label |
| --- | --- | --- |
| scans / slices reconstructed | 92 (35 T1, 33 T2, 24 FLAIR) / 1,656, each full and undersampled | measured |
| coil images computed | 13,248 | derived |
| voxel-sample pairs computed | 36.2 x 10^12 (470 TFLOP at 13 flops per pair) | derived |
| run wall time / GPU kernel time | 46.0 min / 23.7 min | measured |
| throughput | 0.60 full slices per second (each with an undersampled counterpart) | derived |
| kernel rate, median / peak | 25.9 / 28.8 G pairs/s (336 GFLOP/s median) | measured |
| full-data error vs dataset reconstruction, p50 / p99 / max | 0.0002 % / 0.0003 % / 0.0003 % | measured |
| full-data PSNR, minimum | 121.0 dB | measured |
| zero-filled PSNR, p5 / p50 / p95 | 23.9 / 25.2 / 26.5 dB | measured |
| zero-filled PSNR median by contrast | T1 25.1, T2 25.8, FLAIR 24.5 dB | measured |
| GPU utilisation, mean / p95 / max | 48.8 % / 80 % / 93 % | measured |
| SM clock, mean / min | 1,306 / 300 MHz | measured |
| power, mean / max | 24.3 / 50.4 W | measured |
| temperature, mean / max | 80.5 / 86 C | measured |
| energy estimate | 18.6 Wh, 40.5 J per slice | derived (mean power x wall time) |

Reading: the worst full-data slice differs from the dataset's own reconstruction by 0.0003 % relative error, at the float32 noise floor, so the GPU kernels
reproduce the FFT result on all 1,656 slices. Zero-filled quality varies by contrast (FLAIR lowest), as expected from how much signal sits in the
kept central lines. The GPU ran at 49 % mean utilisation on purpose: the runner idles for 0.75 of each slice's GPU time (`PAUSE=0.75`) to hold utilisation to
about 60 % or less on a laptop GPU that reached 86 C; the lowest clock values are the idle gaps between scans while the next scan is extracted and converted.

**Second run, which also writes the images** (`stats/archive_png_run_summary.csv`, `stats/archive_png_run/`): the same 92 scans re-run with PNG output enabled. Accuracy is identical
(same error percentiles). It took 84.2 min instead of 46.0 min: mean SM clock 848 MHz instead of 1,306 MHz, mean power 16.0 W instead of 24.3 W, max temperature 89 C instead of 86 C,
mean GPU utilisation 53.3 % (p95 99 %). The cause of the lower clock was not isolated; the first run's statistics are the ones quoted above. The extra time per scan also includes writing the sheets.

Figures: `figures/real_archive_dist.png` (error, PSNR by contrast, throughput distributions) and `figures/real_archive_timeline.png` (utilisation, clock,
temperature and power over the run).

## Reconstructed images (all 92 scans)

`figures/sheets/` holds 6 of the 92 contact sheets (two scans each of T1, T2 and FLAIR). Every sheet is one PNG per scan: the 18 slices reconstructed on the GPU from full k-space, in a 6 x 3 grid with the slice number in each tile
(tiles downscaled to 192 px). All 92 sheets, plus 9 full-size 256 px images (slice 8 of one T1, one T2 and one FLAIR scan: dataset reconstruction, GPU full-data, GPU zero-filled), are in a shared Google Drive folder, `gpu-mri-png/`:

https://drive.google.com/drive/folders/17ZoYSozMR2q2s46q1vX1IcMIC56efkEh?usp=sharing

The Drive copy holds 101 files (92 sheets, about 0.4 MB each, plus the `showcase/` subfolder), 38 MB in total, matching the files the runner wrote (file counts, and the sizes of 17 files spot-checked); it is kept outside the repository to keep the repository small.

## Scope and limits

- Only the first 92 scans of the validation archive (37 % of its 2.98 GB) were downloaded and reconstructed; the remaining scans of that archive and the four
  other M4Raw archives were not. The runner reads a truncated zip by walking its local file headers, so it works on any complete prefix.
- The data are Cartesian, so this does not exercise the non-Cartesian advantage that motivates the book's kernels; a direct FFT is faster for Cartesian data.
- Hardware `__sinf/__cosf` gave no speedup at this problem size (`output/01_real_slices.txt`).
- Undersampled reconstructions are zero-filled and show aliasing; no parallel-imaging or compressed-sensing reconstruction is implemented.
- Timings depend on the GPU's power state: an early run under a 10 W power cap was ten times slower and was discarded and rerun.

## Reproduce

Requirements: the toolchain in the main README, plus Python with `h5py`, `requests`, `numpy`, `pandas`, `matplotlib`.

```bash
cd 05_real_data_mri
make                                   # build bin/01_real_slices and bin/02_real_archive
python3 scripts/fetch_m4raw.py get multicoil_val/2022061203_T101.h5      # one 12.6 MB scan via HTTP range request
python3 scripts/prepare_m4raw.py       # -> data/m4raw/scan.bin
make run                               # three-slice reconstruction, output/01_real_slices.txt
make test                              # Catch2: F^H d equals the centred inverse DFT on a Cartesian grid (needs vcpkg Catch2)
# whole archive: download the zip (2.98 GB, https://zenodo.org/api/records/8056074/files/M4RawV1.5_multicoil_val.zip/content) to data/m4raw/,
# or any prefix of it, then
make archive                           # 46 minutes for 92 scans; writes stats/archive/* and figures
```

SHA-256 of `2022061203_T101.h5`: `d4156466a977a8bef888dff3141f78fbe93b3191c55f2632f2bd6410522399e7`.

## Data licence and attribution

M4Raw is distributed under CC-BY 4.0 (Zenodo record 8056074). The raw data are not stored here; `stats/01_slice*.bin` hold reconstructed slices and reference
images derived from it, shared with attribution to the dataset authors.

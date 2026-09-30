# Programming Massively Parallel Processors: hands-on CUDA reproductions

Runnable CUDA C++ reproductions of the techniques and case studies in
*Programming Massively Parallel Processors: A Hands-on Approach* (Kirk & Hwu). Each program isolates one idea from the
book, runs it on a real GPU, verifies the result against a double-precision CPU reference, measures it with warm-up and
averaged repeats, and states plainly when a claim made for the book's 2008-era G80 does or does not reproduce on a
modern (Turing) GPU.

## Repository map

| Folder | Book topic | What is in it |
| --- | --- | --- |
| [`01_architecture_basics`](01_architecture_basics/README.md) | GPU architecture (Ch. 1-4) | device query, block/warp scheduling, vector add with transfer timing |
| [`02_tiling_and_reduction`](02_tiling_and_reduction/README.md) | Memory and reduction (Ch. 5-6) | naive vs tiled matrix multiply, three sum-reduction kernels |
| [`03_resource_partitioning`](03_resource_partitioning/README.md) | Performance tuning (Ch. 6-7) | occupancy model, register / shared-memory cliffs, loop unrolling |
| [`04_case_studies`](04_case_studies/README.md) | Application case studies (Ch. 8-10) | MRI F^H d kernels and synthetic-phantom reconstruction, Coulomb summation, cutoff binning, constant-cache behaviour, GPU starvation; tests, figures, statistics |
| [`05_real_data_mri`](05_real_data_mri/README.md) | Ch. 8 on real data | reconstruction of real measured brain k-space (M4Raw): 1,656 slices from 92 scans, with GPU utilisation, throughput, accuracy and energy statistics |

The case-study chapters are numbered as in the edition used here: Chapter 8 is MRI, Chapter 9 is electrostatic
(Coulomb) potential, Chapter 10 is cutoff binning.

## Results on real datasets

The table below covers only real data: the M4Raw in-vivo brain scans (reconstructed images, the main result) and the Parboil MRI-Q
inputs (timing only: they hold scan geometry, no measured signal). Kernel benchmarks on synthetic inputs and the synthetic-phantom reconstruction are in
[`04_case_studies`](04_case_studies/README.md) and in the figures below. All numbers are [MEASURED] on the test GPU unless marked.

| Dataset and study | Result | Result file |
| --- | --- | --- |
| M4Raw brain k-space, every slice of 92 scans (1,656 slices x 4 coils), GPU F^H d vs the dataset's own FFT reconstruction | relative error p50 0.0002 %, max 0.0003 %; PSNR at least 121 dB (float32 floor); 0 of 92 scans failed | `05_real_data_mri/stats/archive_summary.csv` |
| Same run, throughput and cost | 36.2e12 voxel-sample pairs in 46.0 min, median kernel rate 25.9 G pairs/s; mean GPU utilisation 48.8 %, mean 24.3 W, about 18.6 Wh (derived) | `05_real_data_mri/stats/archive_summary.csv` |
| Same data, 3.7x fewer phase-encode lines, zero-filled | PSNR median 25.2 dB (p5 23.9, p95 26.5); aliasing visible; CG reaches the same image | `05_real_data_mri/stats/archive_summary.csv` |
| Same 92 scans, re-run that also writes the images | identical accuracy; 84.2 min instead of 46.0 because the GPU ran at a mean 848 MHz (89 C max) instead of 1,306 MHz; the cause of the lower clock was not isolated | `05_real_data_mri/stats/archive_png_run_summary.csv` |
| Parboil MRI-Q large input (2,048 samples x 262,144 voxels) | accurate constant-memory kernel 15.27 ms; hardware sin/cos 5.39 ms (timing and agreement with a CPU reference; no image exists in this dataset) | `04_case_studies/output/08d_mri_q_parboil.txt` |

Limits worth knowing: M4Raw is Cartesian, not the non-Cartesian trajectories of the book's data; only the first 92 of its 240 validation scans were reconstructed;
undersampled reconstructions are zero-filled and alias (CG without a prior converges to the same image); hardware `__sinf/__cosf` gave no speedup at this problem size.

## Reconstructed images of all 92 scans

The repository ships 6 of the 92 contact sheets (below). All 92 sheets, one PNG per scan with the 18 GPU-reconstructed slices of that scan, plus the full-size slice-8 images
(dataset reconstruction, GPU full-data, GPU zero-filled) for one T1, one T2 and one FLAIR scan, are in a shared Google Drive folder:

**https://drive.google.com/drive/folders/17ZoYSozMR2q2s46q1vX1IcMIC56efkEh?usp=sharing**

Contents of `gpu-mri-png/`: 92 files named `<scan>.png` (about 0.4 MB each, 38 MB in total; 35 T1, 33 T2, 24 FLAIR scans; tiles downscaled to 192 px) and `showcase/` with 9 full-size 256 px images.
The sheets are produced by `05_real_data_mri/scripts/run_real_archive.sh`; they are not in this repository to keep it small.

## Visualizations

All 33 figures, two per row. Figures from real datasets come first; the rest use synthetic inputs and are labelled. Every figure is regenerated from the CSV files named in its caption (`make figures` in each folder).

### Real datasets: M4Raw brain scans and Parboil MRI-Q

<table><tr>
<td width="50%"><img src="05_real_data_mri/figures/sheets/2022061203_T101.png" alt="2022061203_T101"><br><sub>2022061203_T101 (T1): all 18 slices reconstructed on the GPU from full k-space (slice number in each tile).</sub></td>
<td width="50%"><img src="05_real_data_mri/figures/sheets/2022062303_T101.png" alt="2022062303_T101"><br><sub>2022062303_T101 (T1): all 18 slices reconstructed on the GPU from full k-space (slice number in each tile).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="05_real_data_mri/figures/sheets/2022061203_T201.png" alt="2022061203_T201"><br><sub>2022061203_T201 (T2): all 18 slices reconstructed on the GPU from full k-space (slice number in each tile).</sub></td>
<td width="50%"><img src="05_real_data_mri/figures/sheets/2022062303_T201.png" alt="2022062303_T201"><br><sub>2022062303_T201 (T2): all 18 slices reconstructed on the GPU from full k-space (slice number in each tile).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="05_real_data_mri/figures/sheets/2022061203_FLAIR01.png" alt="2022061203_FLAIR01"><br><sub>2022061203_FLAIR01 (FLAIR): all 18 slices reconstructed on the GPU from full k-space (slice number in each tile).</sub></td>
<td width="50%"><img src="05_real_data_mri/figures/sheets/2022062303_FLAIR01.png" alt="2022062303_FLAIR01"><br><sub>2022062303_FLAIR01 (FLAIR): all 18 slices reconstructed on the GPU from full k-space (slice number in each tile).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="05_real_data_mri/figures/real_slices.png" alt="Real-data reconstruction"><br><sub>Slices 4, 8, 12 of one scan: dataset reconstruction, our GPU F^H d from full data, zero-filled from 3.7x fewer lines, and CG on the same undersampled data (`05_real_data_mri/stats/01_real_metrics.csv`).</sub></td>
<td width="50%"><img src="05_real_data_mri/figures/real_archive_dist.png" alt="Archive distributions"><br><sub>All 1,656 slices: error against the dataset reconstruction, PSNR, zero-filled PSNR by contrast, kernel throughput (`05_real_data_mri/stats/archive/metrics.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="05_real_data_mri/figures/real_archive_timeline.png" alt="GPU timeline"><br><sub>GPU utilisation, SM clock, temperature and power over the 46-minute run (`05_real_data_mri/stats/archive/gpu_timeline.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/08_parboil_q.png" alt="Parboil MRI-Q"><br><sub>MRI-Q on the real Parboil inputs (`stats/08d_mri_q.csv`).</sub></td>
</tr></table>

### Coulomb summation (Ch. 9), synthetic inputs

<table><tr>
<td width="50%"><img src="04_case_studies/figures/01_dcs_versions.png" alt="DCS kernels"><br><sub>Direct Coulomb summation: the book's three kernels (`stats/01_dcs.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/04_ch9_pitfalls.png" alt="Chapter 9 pitfalls"><br><sub>GPU-vs-CPU crossover and the 64 KB constant-memory chunk limit (`stats/04_ch9_pitfalls.csv`).</sub></td>
</tr></table>

### Cutoff summation and the constant cache (Ch. 10), synthetic inputs

<table><tr>
<td width="50%"><img src="04_case_studies/figures/05_scaling.png" alt="Cutoff scaling"><br><sub>Time versus volume for each cutoff version (`stats/05*_cutoff.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/05_smallbin_binsweep.png" alt="SmallBin bin-edge sweep"><br><sub>SmallBin: bin edge decides how much overflows to the CPU (`stats/05c_cutoff.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/05_overlap.png" alt="Overlap"><br><sub>SmallBin-Overlap: hiding the CPU overflow pass behind GPU work (`stats/05d_overlap.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/05_overlap_gantt.png" alt="Overlap timeline"><br><sub>One overlapped run, 8 slabs (`stats/05d_overlap_gantt.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/07_divergence.png" alt="Divergence"><br><sub>How often a warp disagrees in the cutoff test, and what it costs (`stats/07_divergence.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/06_hit_rates.png" alt="Constant-cache hit rates"><br><sub>Constant-cache hit rate and time, measured with Nsight Compute (`stats/06_ncu.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/06_window_ratio.png" alt="Window ratio"><br><sub>Constant cache: blocks reading different atoms vs the same atoms (`stats/06_constant_cache.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/06_lanes_layout.png" alt="Lane and layout effects"><br><sub>Warp address uniformity and array-vs-struct layout (`stats/06_constant_cache.csv`).</sub></td>
</tr></table>

### MRI F^H d kernels and the synthetic-phantom reconstruction (Ch. 8)

<table><tr>
<td width="50%"><img src="04_case_studies/figures/08_ladder.png" alt="MRI ladder"><br><sub>F^H d optimisation ladder (`stats/08a_mri_ladder.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/08_thread_mapping.png" alt="Thread mapping"><br><sub>Mapping the two loops to threads (`stats/08b_mri_mapping.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/08_trig_accuracy.png" alt="Trig accuracy"><br><sub>Hardware trigonometry: accuracy against angle size (`stats/08c_trig_accuracy.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/08_tuning.png" alt="Tuning"><br><sub>Block x chunk x unroll tuning (`stats/08c_tuning.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/08_reconstruction_2D.png" alt="2D reconstruction"><br><sub>2D head phantom reconstructed from simulated radial k-space: truth, plain adjoint, density-compensated, and CG at 5, 10 and 40 iterations (`stats/08e_*.bin`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/08_reconstruction_3D.png" alt="3D reconstruction"><br><sub>3D phantom (32^3), central slice, same methods; too coarse to be convincing.</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/08_reconstruction_psnr.png" alt="PSNR vs CG iteration"><br><sub>PSNR against CG iteration, accurate vs hardware trigonometry (`stats/08e_recon_metrics.csv`).</sub></td>
</tr></table>

### GPU starvation (data pipeline), synthetic inputs

<table><tr>
<td width="50%"><img src="04_case_studies/figures/09_gpu_busy.png" alt="GPU busy"><br><sub>How much of the time the GPU is actually computing (`stats/09_summary.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/09_pipeline_gantt.png" alt="Pipeline timelines"><br><sub>Copy and compute timelines per transfer strategy (`stats/09_gantt.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/09_host_prep.png" alt="Host preparation"><br><sub>When the CPU is slow at preparing data (`stats/09_summary.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/09_launch_overhead.png" alt="Launch overhead"><br><sub>Tiny kernels: the launch costs more than the work (`stats/09_summary.csv`).</sub></td>
</tr></table>

<table><tr>
<td width="50%"><img src="04_case_studies/figures/09_copy_compute_overlap.png" alt="Copy/compute overlap"><br><sub>Does the hardware overlap a copy with compute? (`stats/09_summary.csv`).</sub></td>
<td width="50%"><img src="04_case_studies/figures/09_nvml_timeline.png" alt="NVML timeline"><br><sub>What the GPU reports while starved versus fed (`stats/09_nvml_timeline.csv`).</sub></td>
</tr></table>

## Requirements

- NVIDIA GPU with compute capability 7.5 (Turing). Other architectures: change `ARCH` in the Makefiles.
- CUDA Toolkit 12.x (`nvcc`), a C++14 host compiler (developed with CUDA 12.9 and gcc 11.5), GNU make.
- NVML (`libnvidia-ml`) for the GPU monitoring built into the case-study programs (ships with the driver).
- Python 3 with `numpy`, `pandas`, `matplotlib` only to regenerate figures; plus `h5py` and `requests` for the small real-data fetch/convert scripts.
- Optional: [Catch2 v3](https://github.com/catchorg/Catch2) (vcpkg: `vcpkg install catch2`) for `make catch2`; `ncu` with GPU
  performance counters enabled for `make profile`.

Developed on a GTX 1650 Ti (16 SMs, 4 GB) under WSL2. Windows enforces a GPU watchdog, so every kernel launch is kept to a few
milliseconds and long work is chunked.

## Build, run, test

```bash
cd 04_case_studies
make                 # build every program into bin/
make test            # 4 unit/invariant suites + check that every result file reports ALL VARIANTS PASS
make run             # regenerate output/<program>.txt (one file per program)
scripts/fetch_parboil_data.sh   # downloads the two Parboil MRI-Q inputs (size + sha256 verified); needed by 08d only
make figures         # regenerate figures/*.png from stats/*.csv
make catch2          # optional Catch2 toolchain check
```

The three earlier folders build the same way (`cd 01_architecture_basics && make run`). Real-data reconstruction: see [`05_real_data_mri`](05_real_data_mri/README.md) (`make`, fetch one 12.6 MB scan, `make run`, `make archive`).

## Method

- Every result is checked against a double-precision CPU reference (sampled on large problems); the error is scaled by the
  sum of absolute terms.
- Timings are warm (one or more warm-up launches) and averaged over repeats, with variants interleaved. The GPU's
  temperature is read through NVML and the run waits for it to cool, because thermal throttling changed some timings by
  up to 4x during development.
- Numbers are labelled `[MEASURED]`, `[DERIVED]` (computed from measurements) or `[book]` (quoted). A cause is stated only
  if a test supports it; otherwise the text says "hypothesis".
- Programs write their statistics to `stats/*.csv`; small Python scripts turn those into figures.

## Not included

- Multi-GPU execution (Section 9.6 of the book): needs more than one GPU and is not implemented.
- The raw M4Raw scan is not redistributed (reconstructed slices in `05_real_data_mri/stats/01_slice*.bin` are, with attribution); `05_real_data_mri/scripts/fetch_m4raw.py` fetches a scan from Zenodo, and the runner accepts the full archive or any prefix of it.
- The Parboil data files are not redistributed (licence not verified); the fetch script downloads them and checks size and
  SHA-256.

## References

- David B. Kirk and Wen-mei W. Hwu, *Programming Massively Parallel Processors: A Hands-on Approach*, Morgan Kaufmann
  (Elsevier), 2010.
- Parboil benchmark suite (MRI-Q input data), University of Illinois IMPACT group.
- M4Raw: a multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research (Zenodo record 8056074, CC-BY 4.0); file `multicoil_val/2022061203_T101.h5`, slices 4, 8 and 12 are shown; the first 92 scans of `M4RawV1.5_multicoil_val.zip` are reconstructed in full.

## License

MIT, see [LICENSE](LICENSE). The book's text and figures are not reproduced here.

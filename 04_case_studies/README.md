# 04 - Application case studies (book Ch. 8-10)

Every program writes its report to `output/<program>.txt` (never merged), its statistics to `stats/`, and ends with
`ALL VARIANTS PASS` or `SOME VARIANTS FAILED`. Every program also prints, at the end, the GPU's duty cycle, clocks, power and
temperature sampled through NVML, so an under-fed GPU is visible in the result itself.

## Programs

| Program | Book topic | Statistics | What it shows |
| --- | --- | --- | --- |
| `01_dcs_compute` | Ch. 9 direct Coulomb summation | `stats/01_dcs.csv` | grid-centric gather with atoms in constant memory; 1, 4 and 8 lattice points per thread |
| `02_gpu_monitor` | tooling | `stats/02_gpu_monitor_timeline.csv` | memory-hierarchy ledger, thread/register occupancy ledger, NVML timeline |
| `04_ch9_pitfalls` | Ch. 9 | `stats/04_ch9_pitfalls.csv` | GPU vs CPU crossover, lattice padding, scatter vs gather, constant vs global atoms, chunk size |
| `05a_cutoff_atom_centric` | Ch. 10 | `stats/05a_cutoff.csv` | atom-centric scatter with `atomicAdd` vs direct summation; time-vs-volume exponents |
| `05b_cutoff_largebin` | Ch. 10 | `stats/05b_cutoff.csv` | joint neighbourhood lists in constant memory; subvolume-size sweep |
| `05c_cutoff_smallbin` | Ch. 10 | `stats/05c_cutoff.csv` | fixed-capacity bins in global memory tiled into shared memory; overflow list; bin-edge sweep |
| `05d_cutoff_smallbin_overlap` | Ch. 10 | `stats/05d_overlap*.csv` | CPU overflow pass overlapped with GPU slabs; timeline |
| `06_constant_cache_thrash` | cache behaviour | `stats/06_constant_cache.csv` | constant cache with shared vs per-block windows; distinct addresses per warp; array vs struct layout |
| `07_cutoff_divergence` | Ch. 10 | `stats/07_divergence.csv` | control divergence inside the cutoff test, counted with `__ballot_sync` |
| `08a_mri_fhd_ladder` | Ch. 8 | `stats/08a_mri_ladder.csv` | F^H d: naive -> registers -> constant memory -> struct layout -> hardware sin/cos; full scale 128^3 x 284,592 |
| `08b_mri_thread_mapping_options` | Ch. 8 | `stats/08b_mri_mapping.csv` | thread per pair vs per sample (with `atomicAdd`) vs per voxel |
| `08c_mri_accuracy_tuning` | Ch. 8 | `stats/08c_*.csv` | sin/cos accuracy, image agreement (MSE, PSNR), block x chunk x unroll tuning |
| `08d_mri_q_parboil` | Ch. 8 | `stats/08d_mri_q.csv` | Parboil MRI-Q on the real small and large inputs |
| `08e_mri_reconstruction` | Ch. 8 | `stats/08e_recon_metrics.csv`, `stats/08e_*.bin` | phantom -> simulated radial scan -> reconstructions (adjoint, density-compensated, conjugate gradient) |
| `09_gpu_starvation` | Ch. 6-9 (data pipeline) | `stats/09_*.csv` | pageable vs pinned transfers, streams, copy-ahead, host producers, launch overhead |

Shared code: `src/common.cuh` (checks, CSV rows, warm/interleaved timing, occupancy ledger), `src/monitor.cuh` (NVML
sampler), `src/common_bins.cuh` (cutoff data structures and references), `src/common_mri.cuh` (MRI data, kernels, references), `src/recon_kernels.cuh` (forward model F and CG vector helpers).

## Tests

`make test` builds and runs four suites in `tests/` (33 checks: cutoff gather, scatter, LargeBin, MRI kernels) and then
`tests/check_outputs.sh`, which requires every result file in `output/` to report `ALL VARIANTS PASS`. The MRI test was
mutation-checked: a deliberately broken kernel makes it fail. `make catch2` builds `tests/catch2/test_setup.cu`, the Catch2
v3 toolchain check and template for new tests.

## Selected results

Numbers are [MEASURED] on a GTX 1650 Ti unless marked; each traces to the named file.

**Direct Coulomb summation** (`stats/01_dcs.csv`, full scale 100^3 lattice x 100,000 atoms, 2,500 launches): 1 point/thread
70.0 G evals/s, 4 points/thread 99.9, 8 points/thread 63.9. More work per thread helps until the register cost lowers
occupancy and fewer blocks are resident; the launch fill is reported by each program's ledger.

**Cutoff summation** (`output/05*.txt`): all variants agree with a double-precision cutoff reference to about 1e-7. In
`05c` at bin edge 4 A, 6.7 % of atoms overflow to the CPU and the overflow pass costs 45.3 ms against a 2.1 ms kernel; at
edge 3 A nothing overflows and the total is 8.5 ms. In `05d` the overlap fraction varies from 24 % to 232 % between rows, which
is timing noise around a GPU that drops its clock while the CPU works; `output/05d_*.txt` explains this and the minimum-of-5
protocol used.

**Constant cache** (`output/06_ncu_summary.txt`, hardware counters from Nsight Compute):

| Case | idc request hit rate | time |
| --- | --- | --- |
| 512 atoms, every block reads the same window | 87.8 % | 2.10 ms |
| 512 atoms, each block its own window | 49.6 % | 4.52 ms |
| 1 distinct address per warp | 90.6 % | 2.81 ms |
| 32 distinct addresses per warp | 84.3 % (520 M divergent-instruction replays) | 64.03 ms |

The book's warning that constant memory only pays when a warp reads one address reproduces strongly.

**MRI F^H d** (`stats/08a_mri_ladder.csv`, 262,144 voxels x 8,192 samples):

| Version | ms |
| --- | --- |
| v1 naive (global memory) | 99.3 |
| v2 registers | 79.4 |
| v3 constant memory, 3 arrays | 112.8 |
| v4 constant memory, struct | 78.0 |
| v5 + hardware sin/cos | 51.2 |

Version v3 (three separate constant arrays) is slower than v1 here, unlike the book's G80 result; struct layout (v4) fixes it.
The cause is a hypothesis (three separate constant-memory streams per iteration); `06_constant_cache_thrash` E3 compares array
and struct layouts but does not isolate it. Counters in
`output/08a_ncu_summary.txt` show v5 executes fewer warp instructions (2.15 G vs 4.83 G for v4) but issues more XU-pipe
instructions (201 M vs 134 M); hardware sine/cosine runs on that pipe, which limits the gain. Full scale (2,097,152 voxels x
284,592 samples, v5): 8.99 s, 66.4 G pairs/s [MEASURED]. The book's G80 figures (5.4 / 22.8 / 144.5 GFLOPS) are [book] and
are not directly comparable.

**Tuning** (`output/08c_mri_accuracy_tuning.txt`): best block 128, chunk 4096, unroll 8 = 21.74 ms; worst combination is 6.9x
slower; the default (256, 2048, 1) is 94.4 % slower than the best. Joint search beats one-knob-at-a-time by 8.7 % (book: about 20 %).

**Parboil MRI-Q** (`output/08d_mri_q_parboil.txt`; the Parboil files hold scan geometry only, with no measured signal and no image, so these runs check numbers and timing, not images; large input 2,048 samples x 262,144 voxels): accurate constant-memory kernel
15.27 ms, reduced hardware sin/cos 5.39 ms.

**Reconstruction** (`stats/08e_recon_metrics.csv`, figures `figures/08_reconstruction_*.png`): simulated 128-spoke radial
scan of a 128x128 head phantom, 1 % complex noise, PSNR against the phantom after best-scale fit.

| Method | PSNR (dB) |
| --- | --- |
| plain adjoint F^H d | 14.1 |
| density-compensated adjoint | 20.3 |
| conjugate gradient, 10 iterations | 23.2 |
| conjugate gradient, 40 iterations | 24.3 |

Hardware and accurate trigonometry gave identical PSNR to 0.1 dB. The 3D case (32^3, 256 spokes) reaches 17.9 dB after
25 iterations and is too coarse to be a convincing reconstruction. The book's 27.6 dB comes from a different data set.

**Real measured data:** see [`../05_real_data_mri`](../05_real_data_mri/README.md).

**GPU starvation** (`stats/09_summary.csv`): at 1024 FMAs per element streaming 256 MB, pageable synchronous copies take
232.5 ms with the GPU busy 23.4 % of the time; pinned memory with four streams and copy-ahead issue order takes 75.3 ms
with 73.5 % busy. Pinned copies reach 4.5 GB/s against 1.5 GB/s pageable. Launching 2,000 tiny kernels with a sync after
each takes 115.9 ms; one fused kernel does the same work in 0.375 ms.

## Figures

`make figures` runs `scripts/plot_*.py` on `stats/*.csv` and writes `figures/*.png`. The `08_reconstruction_*` images are
rendered from the float32 slices in `stats/08e_*.bin`.

Kernel benchmarks and the synthetic-phantom reconstruction. Real-data figures are in [`../05_real_data_mri`](../05_real_data_mri/README.md).

### Coulomb summation (Ch. 9)


<table><tr>
<td width="50%"><img src="figures/01_dcs_versions.png" alt="DCS kernels"><br><sub>Direct Coulomb summation: the book's three kernels (`stats/01_dcs.csv`).</sub></td>
<td width="50%"><img src="figures/04_ch9_pitfalls.png" alt="Chapter 9 pitfalls"><br><sub>GPU-vs-CPU crossover and the 64 KB constant-memory chunk limit (`stats/04_ch9_pitfalls.csv`).</sub></td>
</tr></table>

### Cutoff summation (Ch. 10)


<table><tr>
<td width="50%"><img src="figures/05_scaling.png" alt="Cutoff scaling"><br><sub>Time versus volume for each cutoff version (`stats/05*_cutoff.csv`).</sub></td>
<td width="50%"><img src="figures/05_smallbin_binsweep.png" alt="SmallBin bin-edge sweep"><br><sub>SmallBin: bin edge decides how much overflows to the CPU (`stats/05c_cutoff.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/05_overlap.png" alt="Overlap"><br><sub>SmallBin-Overlap: hiding the CPU overflow pass behind GPU work (`stats/05d_overlap.csv`).</sub></td>
<td width="50%"><img src="figures/05_overlap_gantt.png" alt="Overlap timeline"><br><sub>One overlapped run, 8 slabs (`stats/05d_overlap_gantt.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/07_divergence.png" alt="Divergence"><br><sub>How often a warp disagrees in the cutoff test, and what it costs (`stats/07_divergence.csv`).</sub></td>
<td width="50%"><img src="figures/06_hit_rates.png" alt="Constant-cache hit rates"><br><sub>Constant-cache hit rate and time, measured with Nsight Compute (`stats/06_ncu.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/06_window_ratio.png" alt="Window ratio"><br><sub>Constant cache: blocks reading different atoms vs the same atoms (`stats/06_constant_cache.csv`).</sub></td>
<td width="50%"><img src="figures/06_lanes_layout.png" alt="Lane and layout effects"><br><sub>Warp address uniformity and array-vs-struct layout (`stats/06_constant_cache.csv`).</sub></td>
</tr></table>

### MRI reconstruction (Ch. 8)


<table><tr>
<td width="50%"><img src="figures/08_ladder.png" alt="MRI ladder"><br><sub>F^H d optimisation ladder (`stats/08a_mri_ladder.csv`).</sub></td>
<td width="50%"><img src="figures/08_thread_mapping.png" alt="Thread mapping"><br><sub>Mapping the two loops to threads (`stats/08b_mri_mapping.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/08_trig_accuracy.png" alt="Trig accuracy"><br><sub>Hardware trigonometry: accuracy against angle size (`stats/08c_trig_accuracy.csv`).</sub></td>
<td width="50%"><img src="figures/08_tuning.png" alt="Tuning"><br><sub>Block x chunk x unroll tuning (`stats/08c_tuning.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/08_parboil_q.png" alt="Parboil MRI-Q"><br><sub>MRI-Q on the real Parboil inputs (`stats/08d_mri_q.csv`).</sub></td>
<td width="50%"><img src="figures/08_reconstruction_psnr.png" alt="PSNR vs CG iteration"><br><sub>PSNR against CG iteration, accurate vs hardware trigonometry (`stats/08e_recon_metrics.csv`).</sub></td>
</tr></table>


<p><img src="figures/08_reconstruction_2D.png" alt="2D reconstruction" width="100%"><br><sub>2D head phantom reconstructed from simulated radial k-space: truth, plain adjoint, density-compensated, and CG at 5, 10 and 40 iterations (`stats/08e_*.bin`).</sub></p>


<p><img src="figures/08_reconstruction_3D.png" alt="3D reconstruction" width="100%"><br><sub>3D phantom (32^3), central slice, same methods; too coarse to be convincing.</sub></p>


<table><tr>
<td width="50%"><img src="figures/09_gpu_busy.png" alt="GPU busy"><br><sub>How much of the time the GPU is actually computing (`stats/09_summary.csv`).</sub></td>
<td width="50%"><img src="figures/09_pipeline_gantt.png" alt="Pipeline timelines"><br><sub>Copy and compute timelines per transfer strategy (`stats/09_gantt.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/09_host_prep.png" alt="Host preparation"><br><sub>When the CPU is slow at preparing data (`stats/09_summary.csv`).</sub></td>
<td width="50%"><img src="figures/09_launch_overhead.png" alt="Launch overhead"><br><sub>Tiny kernels: the launch costs more than the work (`stats/09_summary.csv`).</sub></td>
</tr></table>


<table><tr>
<td width="50%"><img src="figures/09_copy_compute_overlap.png" alt="Copy/compute overlap"><br><sub>Does the hardware overlap a copy with compute? (`stats/09_summary.csv`).</sub></td>
<td width="50%"><img src="figures/09_nvml_timeline.png" alt="NVML timeline"><br><sub>What the GPU reports while starved versus fed (`stats/09_nvml_timeline.csv`).</sub></td>
</tr></table>


## Profiling

`make profile` runs `scripts/run_profiles.sh` (Nsight Compute counters for `06` and `08a`; needs GPU performance counters
enabled for non-admin users) and summarises them into `output/*_ncu_summary.txt`.

## Data

Real scanner data (M4Raw) is handled in [`../05_real_data_mri`](../05_real_data_mri/README.md).

`08d_mri_q_parboil` reads the Parboil MRI-Q inputs from `data/parboil_mri_q/{small,large}`. They are not stored here; run
`scripts/fetch_parboil_data.sh`, which downloads them (454,664 B and 3,186,696 B) and verifies size and SHA-256.

# 02 - Tiling and reduction (book Ch. 5-6)

| Program | Idea | Result file |
| --- | --- | --- |
| `src/01_matmul_tiling.cu` | naive vs shared-memory tiled matrix multiply at several block/tile widths | `output/01_matmul_tiling.txt` |
| `src/02_sum_reduction.cu` | three reduction kernels: interleaved (Fig. 6.2), contiguous (Fig. 6.4), contiguous + unrolled last warp | `output/02_sum_reduction.txt` |

Build and run: `make run`. Each program verifies against a CPU reference (sampled rows for matrix multiply; a double sum for
reduction) and ends with `ALL VARIANTS PASS`.

Method notes:
- Timings are warm with repeated launches; an early version reported a 25x reduction speedup that was a cold-start
  artifact, so single cold timings are never used.
- The reduction is memory-bound on this GPU: the output prints achieved GB/s against the device's peak bandwidth so the
  effect of removing warp divergence can be judged against that ceiling.
- The last-warp unroll uses `volatile` shared memory plus `__syncwarp()`, as required on Volta and later architectures.

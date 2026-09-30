# 03 - Resource partitioning and loop unrolling (book Ch. 6-7)

| Program | Idea | Result file |
| --- | --- | --- |
| `src/01_partitioning_model.cu` | how registers, shared memory, thread and block slots cap resident blocks per SM | `output/01_partitioning_model.txt` |
| `src/02_resource_cliffs.cu` | occupancy falls in whole-block steps as registers per thread or shared memory per block grow | `output/02_resource_cliffs.txt` |
| `src/03_occupancy_vs_time.cu` | whether higher occupancy actually reduces run time | `output/03_occupancy_vs_time.txt` |
| `src/04_loop_unrolling.cu` | unrolling the tile loop: instruction mix, registers, blocks/SM, time | `output/04_loop_unrolling.txt` |

Build and run: `make run`.

Example from `output/02_resource_cliffs.txt` [MEASURED via the CUDA occupancy API]: with 256-thread blocks the kernel stays at
100 % occupancy up to 64 registers per thread, drops to 50 % at 96 and to 25 % at 168; with shared memory, 16384 B per block
still allows 4 blocks/SM and 20480 B allows only 3. The book's claim that the drop happens in whole-block steps reproduces.
The book's loop-unrolling gain (over 20 % at 16x16 tiles on the G80) should be compared with the program's own table, which
also reports whether the kernel is instruction- or memory-bound on this GPU.

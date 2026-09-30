# CUDA GPU Architecture Learning - Basics

Maps the Unified GPU Architecture (Turing/G80 paradigm) to runnable code.

## Quick Start

```bash
cd 01_architecture_basics
make run          # Build and run all three programs
make clean        # Remove binaries
```

## Programs

| File | Diagram Block | Concept | What It Shows |
| --- | --- | --- | --- |
| **01_device_query** | Whole chip: SM, SP, TPC, L2, DRAM | GPU Hardware Hierarchy | SM count, warp size, CUDA cores, shared memory, L2 cache, VRAM. Prints a text tree of GPU → SMs → SPs. |
| **02_hello_threads** | Host Interface → Work Distributors → SMs | Block & Thread Scheduling | Blocks dispatched across SMs, warp formation (32 threads), warp ID, lane ID, which SM executed each warp. |
| **03_vector_add** | SP execution units, PCIe bridge, DRAM | SIMT Execution + Timing | Computes c[i]=a[i]+b[i] on 1M elements. Times H2D transfer, kernel, D2H transfer separately. Verifies against CPU. |

---

## How to Read the Output

### **01_device_query**

Shows your GTX 1650 Ti specs:

```
GPU 0: NVIDIA GeForce GTX 1650 Ti
  Compute Capability: 7.5
  Number of SMs: 16
  Cores per SM: 64
  Total CUDA Cores: 1024
  Warp Size: 32 threads
  Max Threads per Block: 1024
  Shared Memory per Block: 49152 bytes
  Total Global Memory: 4.0 GB
```

**Key takeaway:** Your GPU has 16 SMs, each with 64 CUDA cores (SPs). The work distributor in the diagram assigns blocks to these SMs.

---

### **02_hello_threads**

Launches 4 blocks × 64 threads (2 warps per block):

```
Block[0] Warp[0] (threads 0-31) executing on SM[0]
Block[0] Warp[1] (threads 32-63) executing on SM[0]
  └─ Block[0] complete: 64 threads across 2 warps
Block[1] Warp[0] (threads 0-31) executing on SM[1]
Block[1] Warp[1] (threads 32-63) executing on SM[1]
  └─ Block[1] complete: 64 threads across 2 warps
...
```

**Key takeaway:**
- Each line is one **warp** (32 threads, the scheduling unit from the diagram)
- Blocks may land on different SMs (SM[0], SM[1], etc.)
- The work distributor assigns blocks; the SM scheduler organizes them into warps

---

### **03_vector_add**

Adds 1M float pairs and times the execution:

```
Input: 1000000 elements
Memory per array: 3.8 MB

H2D Transfer (Host → Device via PCIe)
H2D Time: 2.134 ms
H2D Bandwidth: 3.6 GB/s

Kernel: Vector Addition on SPs
Launch: <<<blocks=3907, threads=256>>>
Total threads: 999168 (covers 1000000 elements)
Kernel Time: 0.045 ms

D2H Transfer (Device → Host via PCIe)
D2H Time: 1.067 ms
D2H Bandwidth: 3.6 GB/s

Verification (CPU Baseline)
✓ PASS: Results match CPU baseline
  Max error: 0.00e+00 (within tolerance)

Timing Summary
H2D (transfer in):      2.134 ms
Kernel (compute):       0.045 ms  ← SPs executing c[i]=a[i]+b[i] in parallel
D2H (transfer out):     1.067 ms
Total:                  3.246 ms
```

**Key takeaway:**
- **H2D/D2H dominate** (2.1 + 1.1 ms) because PCIe is slower than kernel compute (0.045 ms)
- The kernel uses all 1024 CUDA cores in parallel (SIMT: each SP runs the same add instruction on different data)
- The ROP units in the diagram handle memory writes; the L2 cache sits between DRAM and the SPs

---

## Concepts Mapped to Diagram

### From your architecture diagram:

| Diagram Block | Batch 1 Program | How It's Demonstrated |
| --- | --- | --- |
| **Host Interface** | 02, 03 | `cudaMalloc` allocates on GPU; `cudaMemcpy` transfers via PCIe |
| **Work Distributors** | 02 | `<<< blocks, threads >>>` dispatch: work distributor assigns blocks to SMs |
| **TPC (Texture Processing Cluster)** | 01 | Read via `cudaGetDeviceProperties` (not directly exposed in CUDA API) |
| **SM (Streaming Multiprocessor)** | 01, 02 | `multiProcessorCount`; 02 shows which SM runs each warp via `%%smid` |
| **SP (Streaming Processor)** | 01, 03 | CUDA cores: 1024 total (16 SMs × 64 cores); 03 runs 1M adds in parallel |
| **SFU (Special Function Unit)** | (Batch 4) | Not used yet; Batch 4 will show `sinf` vs `__sinf` |
| **Shared Memory** | (Batch 3) | 01 shows `sharedMemPerBlock`; Batch 3 will use `__shared__` for tiling |
| **Warp Scheduler** | 02 | Shown as "Warp[0], Warp[1]..." with lane IDs (0–31) |
| **L2 Cache** | 01, 03 | 01 shows size; 03 benefits from it (spatial locality in vector loops) |
| **ROP (Raster Operation)** | (Batch 6) | Not used yet; Batch 6 will show atomics and blending |
| **DRAM** | 01, 03 | 01 shows `totalGlobalMem`; 03 times H2D/D2H transfers over PCIe bridge |

---

## Next Batches (Roadmap)

| Batch | Topic | Key Concepts |
| --- | --- | --- |
| 1 | **Hierarchy & SIMT** | SM, SP, warp, shared mem size, DRAM ← **You are here** |
| 2 | Warp Divergence | If statements in kernels; serialization cost; `__ballot_sync` |
| 3 | Shared Memory & Sync | Tiled matrix multiply, `__syncthreads`, bank conflicts |
| 4 | SFU & Constant Memory | `sinf` vs `__sinf`, C-cache, `__constant__` |
| 5 | Memory Hierarchy | Coalesced vs strided access, L1/L2 behavior, streams, pinned memory |
| 6 | ROP & Advanced | Atomics, reductions, alpha blending, histograms |

---

## Compilation Notes

- **Arch:** `-arch=sm_75` (Turing, GTX 1650 Ti). Adjust if needed:
  - Ampere (RTX 30xx): `-arch=sm_80`
  - Ada (RTX 40xx): `-arch=sm_89`
- **Optimization:** `-O2` is default. Use `-O3` for max speed or `-g` for debugging.
- **Error checking:** All CUDA calls wrapped in `CUDA_CHECK` macro to catch issues early.

---

## Running Individual Programs

```bash
# Just build
make

# Run one program
./bin/01_device_query
./bin/02_hello_threads
./bin/03_vector_add

# Full rebuild
make clean && make run
```

---

## Expected GPU Output (GTX 1650 Ti)

- **01:** 16 SMs, 1024 CUDA cores, 4 GB VRAM, 32-thread warps
- **02:** Blocks distributed across SM[0]–SM[15], warps with lane IDs 0–31
- **03:** ~0.04 ms kernel time, ~3 GB/s PCIe bandwidth, PASS on verification

---

## Questions?

Each program includes comments explaining:
- Which diagram block it maps to
- What to observe in the output
- How CUDA C concepts relate to the GPU architecture

Read the `.cu` file comments before running.

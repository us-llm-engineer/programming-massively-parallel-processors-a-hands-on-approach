/*
 * 05b_cutoff_largebin.cu   -- CUTOFF SUMMATION, version 2: LargeBin
 *
 * Book (Ch.10, Fig 10.2C / 10.3): keep the grid-centric direct-summation kernel, but do not give it ALL atoms. Split the
 * lattice into LARGE subvolumes; for each one the host gathers the JOINT NEIGHBOURHOOD (every atom within rc of any point
 * of the subvolume, found through the bins) into constant memory (64 KB limit -> chunks of at most 4000 atoms) and launches
 * the kernel. Every block of a launch reads the SAME atom list, so the constant cache broadcasts it. Each thread still
 * applies the cutoff test, so atoms in the joint neighbourhood that are outside its own sphere are wasted work.
 *
 * Measured here: (1) a volume ladder at subvolume edge 16 A, (2) the effect of the subvolume size at L = 40 A.
 * Statistics: stats/05b_cutoff.csv. Correctness: 200 sampled lattice points against a double-precision reference.
 */
#include "common_bins.cuh"
#include <string>
#include <sys/stat.h>

#define LB_MAX 4000                 // 4000 x 16 B = 64,000 B of constant memory
__constant__ float4 lb_atoms[LB_MAX];

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void largebin_kernel(int n, float h, float rc2, int x0, int y0, int z0, int sx, int sy, int sz, int gx, int gy, float *out) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x, tot = (size_t)sx * sy * sz;
    if (idx >= tot) return;
    int i = x0 + (int)(idx % sx), j = y0 + (int)((idx / sx) % sy), k = z0 + (int)(idx / ((size_t)sx * sy));
    float x = h * i, y = h * j, z = h * k, e = 0.f;
    for (int a = 0; a < n; a++) {
        float4 A = lb_atoms[a]; float dx = x - A.x, dy = y - A.y, dz = z - A.z, r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) e += A.w * rsqrtf(r2);
    }
    out[((size_t)k * gy + j) * gx + i] += e;
}

struct LBStats { double ms, host_ms; long launches, subvols; double mean_joint; };

// run LargeBin over the whole lattice with subvolume edge sub_A (Angstrom); returns timing and list statistics
static LBStats run_largebin(const Cut &c, const Cells &cells, float sub_A, float *dOut) {
    int sp = std::max(1, (int)lroundf(sub_A / c.h));                     // subvolume edge in lattice points
    CUDA_CHECK(cudaMemset(dOut, 0, npoints(c) * 4));
    LBStats s = { 0, 0, 0, 0, 0 }; double weighted = 0, pts = 0;
    std::vector<float4> joint; joint.reserve(20000);
    cudaDeviceSynchronize();
    double t0 = now_ms();
    for (int z0 = 0; z0 < c.gz; z0 += sp) for (int y0 = 0; y0 < c.gy; y0 += sp) for (int x0 = 0; x0 < c.gx; x0 += sp) {
        int sx = std::min(sp, c.gx - x0), sy = std::min(sp, c.gy - y0), sz = std::min(sp, c.gz - z0);
        double h0 = now_ms();
        auto rng = [&](int o, int n, int &lo, int &hi) { lo = std::max(0, (int)floorf((c.h * o - c.rc) / c.bs)); hi = std::min(c.nb - 1, (int)floorf((c.h * (o + n - 1) + c.rc) / c.bs)); };
        int bx0, bx1, by0, by1, bz0, bz1; rng(x0, sx, bx0, bx1); rng(y0, sy, by0, by1); rng(z0, sz, bz0, bz1);
        joint.clear();
        for (int bz = bz0; bz <= bz1; bz++) for (int by = by0; by <= by1; by++) for (int bx = bx0; bx <= bx1; bx++) {
            size_t b = ((size_t)bz * c.nb + by) * c.nb + bx;
            joint.insert(joint.end(), cells.a.begin() + cells.start[b], cells.a.begin() + cells.start[b + 1]);
        }
        s.host_ms += now_ms() - h0;
        weighted += (double)joint.size() * sx * sy * sz; pts += (double)sx * sy * sz; s.subvols++;
        size_t tot = (size_t)sx * sy * sz; int blocks = (int)((tot + 255) / 256);
        for (size_t off = 0; off < joint.size(); off += LB_MAX) {
            int n = (int)std::min((size_t)LB_MAX, joint.size() - off);
            CUDA_CHECK(cudaMemcpyToSymbol(lb_atoms, joint.data() + off, n * sizeof(float4)));
            largebin_kernel<<<blocks, 256>>>(n, c.h, c.rc * c.rc, x0, y0, z0, sx, sy, sz, c.gx, c.gy, dOut);
            s.launches++;
        }
    }
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
    s.ms = now_ms() - t0; s.mean_joint = weighted / pts;
    return s;
}

int main() {
    RunMonitor mon;
    const cudaDeviceProp &p = dev_props();
    mkdir("stats", 0755); remove("stats/05b_cutoff.csv");
    const char *HDR = "algorithm,L_angstrom,volume_A3,points,atoms,ms,max_err,check,subvolume_A,launches,mean_candidates_per_point,mean_inside_per_point";
    printf("==================== CUTOFF SUMMATION v2: LARGEBIN (%s) ====================\n", p.name);
    printf("[book] joint neighbourhood of a large subvolume -> constant memory (<= 64 KB per launch) -> direct-summation kernel with the cutoff test.\n");
    printf("lattice 0.5 A, cutoff 8 A, atom density 0.1 / A^3, bin edge 4 A. Time = host list gathering + constant uploads + kernels\n");
    printf("(all included, as in the real application); mean of 3 runs after 1 warm-up. CPU time is the sequential atom-centric run of 05a.\n\n");

    bool all_ok = true;
    printf("---- 1. Volume ladder, subvolume edge 16 A (32 x 32 x 32 lattice points) ----\n");
    printf("%-6s %-11s %-9s %-9s | %-10s %-11s %-9s %-9s | %-12s %-13s %-9s | %s\n", "L (A)", "volume A^3", "points", "atoms", "GPU ms", "host gather", "launches", "vs CPU",
           "candidates", "inside/point", "useful", "max err");
    float Ls[6] = { 10, 20, 40, 80, 120, 160 };          // 120 and 160 A are GPU-only scales (no sequential CPU baseline)
    for (float L : Ls) {
        cooldown();
        Cut c = make_cut(L, 0.5f, 8.0f, 4.0f, 0.1);
        auto atoms = make_atoms_box(c, 100 + (unsigned)L); Cells cells = make_cells(atoms, c);
        float *d; CUDA_CHECK(cudaMalloc(&d, npoints(c) * 4));
        run_largebin(c, cells, 16, d);
        LBStats a[3]; for (int r = 0; r < 3; r++) a[r] = run_largebin(c, cells, 16, d);
        double ms = (a[0].ms + a[1].ms + a[2].ms) / 3, hm = (a[0].host_ms + a[1].host_ms + a[2].host_ms) / 3;
        std::vector<float> h(npoints(c)); CUDA_CHECK(cudaMemcpy(h.data(), d, npoints(c) * 4, cudaMemcpyDeviceToHost));
        double inside, err = check_cutoff(h, c, atoms, 200, &inside); bool ok = err < 1e-4; all_ok &= ok;
        double cpu = cpu_baseline_ms(L);
        printf("%-6.0f %-11.0f %-9zu %-9d | %-10.2f %-11.2f %-9ld %-9s | %-12.0f %-13.0f %-8.1f%% | %.1e %s\n", L, volume(c), npoints(c), c.natoms, ms, hm, a[0].launches,
               cpu > 0 ? (std::to_string((int)(cpu / ms)) + "x").c_str() : "n/a", a[0].mean_joint, inside, 100.0 * inside / a[0].mean_joint, err, ok ? "PASS" : "FAIL");
        csv_row("stats/05b_cutoff.csv", HDR, "gpu_largebin,%.0f,%.0f,%zu,%d,%.3f,%.2e,%s,16,%ld,%.1f,%.1f", L, volume(c), npoints(c), c.natoms, ms, err, ok ? "PASS" : "FAIL", a[0].launches, a[0].mean_joint, inside);
        hot_note();
        cudaFree(d);
    }
    printf("  candidates = mean atoms in the joint list a lattice point is tested against; inside/point = mean atoms actually within rc;\n");
    printf("  useful = inside/candidates. The rest is the cost of the coarse neighbourhood (every atom outside the point's own sphere is a wasted test).\n\n");

    printf("---- 2. Subvolume size at L = 40 A (6,400 atoms, 512,000 points) ----\n");
    printf("[book] larger subvolumes give bigger joint lists (more wasted tests); smaller ones give more launches and uploads.\n");
    printf("%-14s %-12s %-10s %-11s %-13s %-10s %-9s | %s\n", "subvolume A", "subvolumes", "launches", "GPU ms", "host gather", "candidates", "useful", "max err");
    {
        Cut c = make_cut(40, 0.5f, 8.0f, 4.0f, 0.1);
        auto atoms = make_atoms_box(c, 140); Cells cells = make_cells(atoms, c);
        float *d; CUDA_CHECK(cudaMalloc(&d, npoints(c) * 4)); std::vector<float> h(npoints(c));
        float subs[4] = { 8, 16, 24, 40 };
        for (float S : subs) {
            cooldown();
            run_largebin(c, cells, S, d);
            LBStats a[3]; for (int r = 0; r < 3; r++) a[r] = run_largebin(c, cells, S, d);
            double ms = (a[0].ms + a[1].ms + a[2].ms) / 3, hm = (a[0].host_ms + a[1].host_ms + a[2].host_ms) / 3;
            CUDA_CHECK(cudaMemcpy(h.data(), d, npoints(c) * 4, cudaMemcpyDeviceToHost));
            double inside, err = check_cutoff(h, c, atoms, 200, &inside); bool ok = err < 1e-4; all_ok &= ok;
            printf("%-14.0f %-12ld %-10ld %-11.2f %-13.2f %-10.0f %-8.1f%% | %.1e %s\n", S, a[0].subvols, a[0].launches, ms, hm, a[0].mean_joint, 100.0 * inside / a[0].mean_joint, err, ok ? "PASS" : "FAIL");
            csv_row("stats/05b_cutoff.csv", HDR, "gpu_largebin_sweep,%.0f,%.0f,%zu,%d,%.3f,%.2e,%s,%.0f,%ld,%.1f,%.1f", 40.0, volume(c), npoints(c), c.natoms, ms, err, ok ? "PASS" : "FAIL", S, a[0].launches, a[0].mean_joint, inside);
        }
        cudaFree(d);
    }
    printf("\nHow to read: 'GPU ms' contains everything the host does per subvolume. If 'host gather' is a large share of it, the\n");
    printf("kernel is not the bottleneck; if 'useful' is small the kernel does mostly wasted distance tests. 05c (SmallBin) shrinks\n");
    printf("the candidate list per block, which is the book's reason for preferring it at large volumes.\n");
    printf("\n%s\n", all_ok ? "ALL VARIANTS PASS" : "SOME VARIANTS FAILED");
    return all_ok ? 0 : 1;
}

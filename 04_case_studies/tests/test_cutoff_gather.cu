// Unit tests for the cutoff-summation building blocks in src/common_bins.cuh: binning, cell lists, the CPU and GPU kernels
// (DirectSum + cutoff test, SmallBin + overflow list) against the double-precision reference, plus invariants
// (chunk splitting, slab splitting, linearity in the charges).
#include "common_bins.cuh"
#include "test_util.cuh"

int main() {
    Cut c = make_cut(16, 0.5f, 8.0f, 4.0f, 0.1);                          // 32^3 lattice points, ~410 atoms
    auto atoms = make_atoms_box(c, 1); size_t np = npoints(c);
    printf("cutoff building blocks: %zu atoms, %zu lattice points\n", atoms.size(), np);

    // T1 binning conserves atoms and places each one in its own bin
    Bins b = make_bins(atoms, c); size_t stored = 0; bool placed = true;
    for (size_t k = 0; k < b.cnt.size(); k++) {
        stored += b.cnt[k];
        int bx = (int)(k % c.nb), by = (int)((k / c.nb) % c.nb), bz = (int)(k / ((size_t)c.nb * c.nb));
        for (int s = 0; s < b.cnt[k]; s++) { float4 a = b.a[k * BIN_CAP + s]; placed &= bin_of(a.x, c) == bx && bin_of(a.y, c) == by && bin_of(a.z, c) == bz; }
    }
    CHECK(stored + b.ovf.size() == atoms.size(), "atoms lost in binning: %zu stored + %zu overflow != %zu", stored, b.ovf.size(), atoms.size());
    CHECK(placed, "an atom sits in the wrong bin");
    CHECK(b.ovf.size() > 0, "bin edge 4 should overflow some atoms (capacity %d) so the overflow path is exercised", BIN_CAP);

    // T2 cell list (compressed bins) holds every atom exactly once, in its own cell
    Cells cells = make_cells(atoms, c); bool ok = cells.a.size() == atoms.size() && cells.start.back() == (int)atoms.size();
    for (size_t k = 0; k + 1 < cells.start.size(); k++) for (int i = cells.start[k]; i < cells.start[k + 1]; i++) {
        float4 a = cells.a[i]; ok &= (((size_t)bin_of(a.z, c) * c.nb + bin_of(a.y, c)) * c.nb + bin_of(a.x, c)) == k; }
    CHECK(ok, "cell list is inconsistent");

    // T3 sequential CPU atom-centric algorithm vs double reference
    std::vector<float> cpu(np, 0.f); cpu_atom_centric(atoms, c, cpu, 0, c.gz);
    double e_cpu = check_cutoff(cpu, c, atoms, 300);
    CHECK(e_cpu < 1e-5, "CPU atom-centric error %.2e", e_cpu);

    // T4 GPU DirectSum + cutoff test vs reference, and chunk-splitting invariance
    float4 *dA; float *d1, *d2; cudaMalloc(&dA, atoms.size() * 16); cudaMalloc(&d1, np * 4); cudaMalloc(&d2, np * 4);
    cudaMemcpy(dA, atoms.data(), atoms.size() * 16, cudaMemcpyHostToDevice);
    direct_cutoff_run(dA, (int)atoms.size(), c, d1, 1 << 20); direct_cutoff_run(dA, (int)atoms.size(), c, d2, 100);
    std::vector<float> g1(np), g2(np); cudaMemcpy(g1.data(), d1, np * 4, cudaMemcpyDeviceToHost); cudaMemcpy(g2.data(), d2, np * 4, cudaMemcpyDeviceToHost);
    CHECK(check_cutoff(g1, c, atoms, 300) < 1e-5, "direct kernel error");
    double worst = 0, mag = 0; for (size_t i = 0; i < np; i++) { worst = fmax(worst, fabs(g1[i] - g2[i])); mag = fmax(mag, fabs(g1[i])); }
    CHECK(worst < 1e-3 * mag, "chunk size changes the direct-sum result: %.2e vs magnitude %.2e", worst, mag);

    // T5 SmallBin + CPU overflow list vs reference; T6 splitting the launch into z-slabs gives the same result bit for bit
    float4 *dB; int *dC; cudaMalloc(&dB, b.a.size() * 16); cudaMalloc(&dC, b.cnt.size() * 4);
    cudaMemcpy(dB, b.a.data(), b.a.size() * 16, cudaMemcpyHostToDevice); cudaMemcpy(dC, b.cnt.data(), b.cnt.size() * 4, cudaMemcpyHostToDevice);
    int bzn = (c.gz + SB_TZ - 1) / SB_TZ; dim3 blk(SB_TX, SB_TY, SB_TZ);
    smallbin_kernel<<<dim3(c.gx / SB_TX, c.gy / SB_TY, bzn), blk>>>(dB, dC, c.nb, c.bs, c.rc, c.h, c.gx, c.gy, c.gz, 0, d1);
    smallbin_kernel<<<dim3(c.gx / SB_TX, c.gy / SB_TY, bzn / 2), blk>>>(dB, dC, c.nb, c.bs, c.rc, c.h, c.gx, c.gy, c.gz, 0, d2);
    smallbin_kernel<<<dim3(c.gx / SB_TX, c.gy / SB_TY, bzn - bzn / 2), blk>>>(dB, dC, c.nb, c.bs, c.rc, c.h, c.gx, c.gy, c.gz, bzn / 2, d2);
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(g1.data(), d1, np * 4, cudaMemcpyDeviceToHost); cudaMemcpy(g2.data(), d2, np * 4, cudaMemcpyDeviceToHost);
    bool same = true; for (size_t i = 0; i < np; i++) same &= g1[i] == g2[i];
    CHECK(same, "slab-split SmallBin differs from the single launch");
    std::vector<float> ovf(np, 0.f); cpu_atom_centric(b.ovf, c, ovf, 0, c.gz); for (size_t i = 0; i < np; i++) g1[i] += ovf[i];
    CHECK(check_cutoff(g1, c, atoms, 300) < 1e-5, "SmallBin + overflow error");

    // T7 linearity: doubling every charge doubles the potential
    std::vector<float4> a2 = atoms; for (auto &a : a2) a.w *= 2;
    cudaMemcpy(dA, a2.data(), a2.size() * 16, cudaMemcpyHostToDevice); direct_cutoff_run(dA, (int)a2.size(), c, d2, 1 << 20);
    cudaMemcpy(g2.data(), d2, np * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dA, atoms.data(), atoms.size() * 16, cudaMemcpyHostToDevice); direct_cutoff_run(dA, (int)atoms.size(), c, d1, 1 << 20); cudaMemcpy(g1.data(), d1, np * 4, cudaMemcpyDeviceToHost);
    worst = 0; for (size_t i = 0; i < np; i++) worst = fmax(worst, fabs(g2[i] - 2 * g1[i])); CHECK(worst < 1e-3 * mag, "potential is not linear in the charges (%.2e)", worst);

    // T8 a single atom: the value at a lattice point equals q / r inside the cutoff and 0 outside
    std::vector<float4> one = { make_float4(8.0f, 8.0f, 8.0f, 1.5f) }; Cut c1 = c; c1.natoms = 1;
    cudaMemcpy(dA, one.data(), 16, cudaMemcpyHostToDevice); direct_cutoff_run(dA, 1, c1, d1, 1 << 20); cudaMemcpy(g1.data(), d1, np * 4, cudaMemcpyDeviceToHost);
    int ix = 20, iy = 16, iz = 16; double r = sqrt(pow(0.5 * ix - 8.0, 2) + pow(0.5 * iy - 8.0, 2) + pow(0.5 * iz - 8.0, 2));
    CHECK(fabs(g1[((size_t)iz * c.gy + iy) * c.gx + ix] - 1.5 / r) < 1e-4, "single-atom value %.5f expected %.5f", g1[((size_t)iz * c.gy + iy) * c.gx + ix], 1.5 / r);
    CHECK(g1[0] == 0.f, "a point beyond the cutoff must receive nothing (corner value %g)", g1[0]);
    TEST_SUMMARY("test_cutoff_gather");
}

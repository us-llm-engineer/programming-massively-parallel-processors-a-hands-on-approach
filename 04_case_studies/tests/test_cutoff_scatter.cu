// Tests of the atom-centric (scatter, atomicAdd) GPU kernel from 05a: agreement with the reference, chunk-splitting, boundary atoms.
#define main program05a_main
#include "05a_cutoff_atom_centric.cu"
#undef main
#include "test_util.cuh"

int main() {
    Cut c = make_cut(16, 0.5f, 8.0f, 4.0f, 0.1); auto atoms = make_atoms_box(c, 2); size_t np = npoints(c);
    printf("cutoff scatter kernel: %zu atoms, %zu lattice points\n", atoms.size(), np);
    float4 *dA; float *d; cudaMalloc(&dA, atoms.size() * 16); cudaMalloc(&d, np * 4); cudaMemcpy(dA, atoms.data(), atoms.size() * 16, cudaMemcpyHostToDevice);
    std::vector<float> g(np), h(np);
    scatter_run(dA, (int)atoms.size(), c, d, 1 << 20); cudaMemcpy(g.data(), d, np * 4, cudaMemcpyDeviceToHost);
    CHECK(check_cutoff(g, c, atoms, 300) < 1e-5, "scatter kernel error");
    scatter_run(dA, (int)atoms.size(), c, d, 37); cudaMemcpy(h.data(), d, np * 4, cudaMemcpyDeviceToHost);      // many small launches
    CHECK(check_cutoff(h, c, atoms, 300) < 1e-5, "chunked scatter error");
    // an atom sitting exactly on the domain boundary and one outside the lattice must not write out of bounds
    std::vector<float4> edge = { make_float4(0.f, 0.f, 0.f, 1.f), make_float4(16.f, 16.f, 16.f, 1.f), make_float4(-3.f, 5.f, 5.f, 1.f) };
    cudaMemcpy(dA, edge.data(), 3 * 16, cudaMemcpyHostToDevice); scatter_run(dA, 3, c, d, 16384); cudaError_t e = cudaDeviceSynchronize();
    CHECK(e == cudaSuccess, "boundary atoms caused a CUDA error: %s", cudaGetErrorString(e));
    cudaMemcpy(g.data(), d, np * 4, cudaMemcpyDeviceToHost);
    double sc; Cut c3 = c; ref_cutoff_point(edge, c3, 3, 3, 3, &sc);
    CHECK(check_cutoff(g, c, edge, 200) < 1e-5, "boundary atoms give a wrong potential");
    TEST_SUMMARY("test_cutoff_scatter");
}

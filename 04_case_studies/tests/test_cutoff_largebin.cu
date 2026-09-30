// Tests of LargeBin from 05b: agreement with the reference for several subvolume sizes, and the constant-memory chunking path
// (a joint list longer than the 4000-atom constant buffer is streamed in several launches).
#define main program05b_main
#include "05b_cutoff_largebin.cu"
#undef main
#include "test_util.cuh"

int main() {
    for (int pass = 0; pass < 2; pass++) {
        Cut c = pass == 0 ? make_cut(16, 0.5f, 8.0f, 4.0f, 0.1) : make_cut(16, 0.5f, 8.0f, 4.0f, 1.3);       // pass 1: ~5300 atoms > 4000
        auto atoms = make_atoms_box(c, 3 + pass); Cells cells = make_cells(atoms, c); size_t np = npoints(c);
        printf("largebin, %s density: %zu atoms\n", pass ? "high" : "normal", atoms.size());
        float *d; cudaMalloc(&d, np * 4); std::vector<float> g(np);
        float subs[3] = { 4.0f, 8.0f, 16.0f };
        for (float S : subs) {
            LBStats st = run_largebin(c, cells, S, d); cudaMemcpy(g.data(), d, np * 4, cudaMemcpyDeviceToHost);
            CHECK(check_cutoff(g, c, atoms, 250) < 1e-5, "LargeBin subvolume %.0f A, %s density: error too large", S, pass ? "high" : "normal");
            if (pass == 1 && S == 16.0f) CHECK(st.launches > st.subvols, "the joint list exceeded 4000 atoms, so extra chunk launches were expected (%ld launches, %ld subvolumes)", st.launches, st.subvols);
        }
        cudaFree(d);
    }
    TEST_SUMMARY("test_cutoff_largebin");
}

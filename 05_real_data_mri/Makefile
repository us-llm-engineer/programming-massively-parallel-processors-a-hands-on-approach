NVCC = /usr/local/cuda/bin/nvcc
ARCH = sm_75
CFLAGS = -O2 -std=c++14 -arch=$(ARCH)
# SHARED holds common.cuh, monitor.cuh, common_mri.cuh, recon_kernels.cuh
SHARED = ../04_case_studies/src
LDLIBS = -L/usr/local/cuda/lib64/stubs -lnvidia-ml -Xcompiler -pthread

SOURCES = $(wildcard src/*.cu)
TARGETS = $(patsubst src/%.cu,bin/%,$(SOURCES))
.PHONY: all run test catch2 archive figures clean
all: $(TARGETS)

bin/%: src/%.cu $(wildcard src/*.cuh) $(wildcard $(SHARED)/*.cuh) $(SHARED)/01_dcs_compute.cu
	@mkdir -p bin
	$(NVCC) $(CFLAGS) -I$(SHARED) -Isrc -o $@ $< $(LDLIBS)

# one scan: the three slices shown in the README (needs data/m4raw/scan.bin, see README)
run: bin/01_real_slices
	@mkdir -p output && ./bin/01_real_slices 2>&1 | tee output/01_real_slices.txt

# every slice of every scan in the validation archive (46 minutes for 92 scans on a GTX 1650 Ti)
archive: bin/02_real_archive
	PAUSE=0.75 bash scripts/run_real_archive.sh data/m4raw/M4RawV1.5_multicoil_val.zip 1 stats/archive
	python3 scripts/plot_archive.py stats/archive

# Catch2 v3 (vcpkg) tests
VCPKG_ROOT ?= $(HOME)/vcpkg
VCPKG_TRIPLET ?= x64-linux
CATCH_INC = $(VCPKG_ROOT)/installed/$(VCPKG_TRIPLET)/include
CATCH_LIB = $(VCPKG_ROOT)/installed/$(VCPKG_TRIPLET)/lib
bin/catch2_%: tests/%.cu $(wildcard $(SHARED)/*.cuh)
	@mkdir -p bin
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -I$(CATCH_INC) -I$(SHARED) -Isrc -o $@ $< -L$(CATCH_LIB)/manual-link -L$(CATCH_LIB) -lCatch2Main -lCatch2
test catch2: bin/catch2_test_cartesian_recon
	./bin/catch2_test_cartesian_recon

figures:
	python3 scripts/plot_real.py
	python3 scripts/plot_archive.py stats/archive

clean:
	rm -rf bin

NVCC = /usr/local/cuda/bin/nvcc
ARCH = sm_75
CFLAGS = -O2 -std=c++14 -arch=$(ARCH)
SRCDIR = src
BINDIR = bin
OUTDIR = output

SOURCES = $(wildcard $(SRCDIR)/*.cu)
TARGETS = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%,$(SOURCES))
OUTPUTS = $(patsubst $(SRCDIR)/%.cu,$(OUTDIR)/%.txt,$(SOURCES))

.PHONY: all run clean

all: $(TARGETS)

# 02_gpu_monitor uses NVML (driver library; the stub is only for linking) and includes 01
# every program can read the GPU's temperature/clocks (NVML; the stub is only for linking, the driver library is used at run time)
LDLIBS = -L/usr/local/cuda/lib64/stubs -lnvidia-ml -Xcompiler -pthread


$(BINDIR)/%: $(SRCDIR)/%.cu $(SRCDIR)/01_dcs_compute.cu $(wildcard $(SRCDIR)/*.cuh)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $< $(LDLIBS)

# one output file per program
run: $(OUTPUTS)

$(OUTDIR)/%.txt: $(BINDIR)/%
	@mkdir -p $(OUTDIR)
	$< 2>&1 | tee $@

# unit/invariant tests (C++): build, run, then check that every program's result file reports ALL VARIANTS PASS
TESTS = $(BINDIR)/test_cutoff_gather $(BINDIR)/test_cutoff_scatter $(BINDIR)/test_cutoff_largebin $(BINDIR)/test_mri
$(BINDIR)/test_%: tests/test_%.cu tests/test_util.cuh $(wildcard $(SRCDIR)/*.cu) $(wildcard $(SRCDIR)/*.cuh)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) -I$(SRCDIR) -o $@ $< $(LDLIBS)

.PHONY: test
test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done
	@bash tests/check_outputs.sh

# Catch2 (from vcpkg) is the framework for FUTURE tests; the suites above stay as they are.
#   one-time set-up:  git clone https://github.com/microsoft/vcpkg ~/vcpkg && ~/vcpkg/bootstrap-vcpkg.sh -disableMetrics && ~/vcpkg/vcpkg install catch2
VCPKG_ROOT ?= $(HOME)/vcpkg
VCPKG_TRIPLET ?= x64-linux
CATCH_INC = $(VCPKG_ROOT)/installed/$(VCPKG_TRIPLET)/include
CATCH_LIB = $(VCPKG_ROOT)/installed/$(VCPKG_TRIPLET)/lib
$(BINDIR)/catch2_setup: tests/catch2/test_setup.cu
	@mkdir -p $(BINDIR)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -I$(CATCH_INC) -I$(SRCDIR) -o $@ $< -L$(CATCH_LIB)/manual-link -L$(CATCH_LIB) -lCatch2Main -lCatch2

.PHONY: catch2
catch2: $(BINDIR)/catch2_setup
	./$(BINDIR)/catch2_setup

# hardware-counter measurements (needs GPU performance counters enabled) and figures from stats/*.csv
.PHONY: profile figures
profile: $(BINDIR)/06_constant_cache_thrash $(BINDIR)/08a_mri_fhd_ladder
	bash scripts/run_profiles.sh
figures:
	@for s in scripts/plot_*.py; do python3 $$s; done

clean:
	rm -rf $(BINDIR) $(OUTDIR)

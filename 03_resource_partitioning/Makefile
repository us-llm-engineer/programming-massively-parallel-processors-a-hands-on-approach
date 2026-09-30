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

$(BINDIR)/%: $(SRCDIR)/%.cu
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $<

# one output file per program: output/01_matmul_tiling.txt, output/02_sum_reduction.txt
run: $(OUTPUTS)

$(OUTDIR)/%.txt: $(BINDIR)/%
	@mkdir -p $(OUTDIR)
	$< 2>&1 | tee $@

clean:
	rm -rf $(BINDIR) $(OUTDIR)

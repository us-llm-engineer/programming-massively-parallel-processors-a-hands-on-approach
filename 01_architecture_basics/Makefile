NVCC = /usr/local/cuda/bin/nvcc
ARCH = sm_75
CFLAGS = -O2 -arch=$(ARCH)
SRCDIR = src
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.cu)
TARGETS = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%,$(SOURCES))

.PHONY: all run clean

all: $(TARGETS)

$(BINDIR)/%: $(SRCDIR)/%.cu
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) -o $@ $<
	@echo "Built $@"

run: all
	@echo "\n=========================================="
	@echo "01. Device Query"
	@echo "=========================================="
	$(BINDIR)/01_device_query
	@echo ""
	@echo "=========================================="
	@echo "02. Hello Threads (Work Distribution)"
	@echo "=========================================="
	$(BINDIR)/02_hello_threads
	@echo ""
	@echo "=========================================="
	@echo "03. Vector Add (SIMT + Timing)"
	@echo "=========================================="
	$(BINDIR)/03_vector_add

clean:
	rm -rf $(BINDIR)

help:
	@echo "Targets:"
	@echo "  make all   - Build all programs"
	@echo "  make run   - Build and run all programs"
	@echo "  make clean - Remove binaries"

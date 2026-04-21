# MAKEFILE for aa
.PHONY: default clean purge test bench

OBJECTS = src/aa.o

PROFILING = 0
CFLAGS += -g -Wall -O3 -Iinclude -DPROFILING=$(PROFILING)

# BLAS/LAPACK linkage — override to swap in MKL, Accelerate, OpenBLAS, etc.
#   e.g.  make LDLIBS="-lmkl_rt -lpthread -lm -ldl"
#         make LDLIBS="-framework Accelerate"
# `-lm` is needed by run_tests (sqrt, fabs); harmless for gd.
LDLIBS ?= -lblas -llapack -lm

SRC_FILES = $(wildcard src/*.c)
INC_FILES = $(wildcard include/*.h)

OUT = out
ARCHIVE = ar -rv
RANLIB = ranlib

default: $(OUT)/libaa.a $(OUT)/gd $(OUT)/bench

%.o : src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

src/aa.o	: $(SRC_FILES) $(INC_FILES)

$(OUT)/libaa.a: $(OBJECTS)
	mkdir -p $(OUT)
	$(ARCHIVE) $@ $^
	- ranlib $@

$(OUT)/gd: tests/c/gd.c $(OUT)/libaa.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(OUT)/bench: tests/c/bench.c $(OUT)/libaa.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

clean:
	@rm -rf $(OBJECTS)
purge: clean
	@rm -rf $(OUT)

test: $(OUT)/run_tests $(OUT)/gd $(OUT)/bench
	$(OUT)/run_tests
	tests/c/check_gd_convergence.sh
	$(OUT)/bench

bench: $(OUT)/bench
	$(OUT)/bench

$(OUT)/run_tests: tests/c/run_tests.c $(OUT)/libaa.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)


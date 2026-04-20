# MAKEFILE for aa
.PHONY: default clean purge test

OBJECTS = src/aa.o

PROFILING = 0
CFLAGS += -g -Wall -O3 -Iinclude -DPROFILING=$(PROFILING)
# Override on the command line to link against MKL, OpenBLAS-only, etc.
# `-lm` is needed by run_tests (sqrt, fabs); harmless for gd.
LDLIBS ?= -lblas -llapack -lm

SRC_FILES = $(wildcard src/*.c)
INC_FILES = $(wildcard include/*.h)

OUT = out
ARCHIVE = ar -rv
RANLIB = ranlib

default: $(OUT)/libaa.a $(OUT)/gd

%.o : src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

src/aa.o	: $(SRC_FILES) $(INC_FILES)

$(OUT)/libaa.a: $(OBJECTS)
	mkdir -p $(OUT)
	$(ARCHIVE) $@ $^
	- ranlib $@

$(OUT)/gd: examples/gd.c $(OUT)/libaa.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

clean:
	@rm -rf $(OBJECTS)
purge: clean
	@rm -rf $(OUT)

# `make test` builds and runs the suite; use `make $(OUT)/run_tests` if you
# only want to build it.
test: $(OUT)/run_tests
	$(OUT)/run_tests

$(OUT)/run_tests: test/run_tests.c $(OUT)/libaa.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)


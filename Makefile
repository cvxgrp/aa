# MAKEFILE for aa
.PHONY: default clean purge

OBJECTS = src/aa.o

PROFILING = 0
CFLAGS += -g -Wall -O3 -Iinclude -DPROFILING=$(PROFILING)

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
	$(CC) $(CFLAGS) -o $@ $^ -lblas -llapack -lm

clean:
	@rm -rf $(OBJECTS)
purge: clean
	@rm -rf $(OUT)


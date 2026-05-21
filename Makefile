USE_CUDA ?= 0
USE_HIP  ?= 0
HIP_ARCH ?= gfx90a
BIN = libmpipancake.so

ifeq ($(USE_CUDA),1)
    CC      := nvcc
    MPI_CFLAGS  := $(shell mpicxx --showme:compile)
    MPI_LDFLAGS := $(addprefix -L,$(shell mpicxx --showme:libdirs)) $(addprefix -l,$(shell mpicxx --showme:libs))
    CFLAGS  := -ccbin mpicxx -DNOPROFILE -O3 -std=c++17 -Xcompiler="-fPIC -Wall -Wextra -march=native" -x cu
    LDFLAGS := $(MPI_LDFLAGS)

else ifeq ($(USE_HIP),1)
    CC      := hipcc
    MPI_CFLAGS  := $(shell mpicxx --showme:compile)
    MPI_LDFLAGS := $(addprefix -L,$(shell mpicxx --showme:libdirs)) $(addprefix -l,$(shell mpicxx --showme:libs))
    CFLAGS  := -DNOPROFILE -O3 -std=c++17 -fPIC -ffast-math -Wall -Wextra -x hip --offload-arch=$(HIP_ARCH)
    LDFLAGS := $(MPI_LDFLAGS)
else
    CC      := mpicxx
    CFLAGS  := -DNOPROFILE -O3 -std=c++17 -fPIC -Wall -Wextra -march=native
    LDFLAGS := -ldl
endif

.PHONY: all clean

all: libmpipancake.so libmpisniffer.so test

libmpipancake.so: mpi_pancake.cpp
	$(CC) $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

libmpisniffer.so: mpi_sniffer.cpp
	$(CC) $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

test: test.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f libmpipancake.so libmpisniffer.so test

USE_CUDA ?= 0
USE_HIP  ?= 0
MPI_CFLAGS := $(shell mpicxx --showme:compile)
MPI_LDFLAGS := $(addprefix -L,$(shell mpicxx --showme:libdirs)) $(addprefix -l,$(shell mpicxx --showme:libs))
BIN = libmpipancake.so

ifeq ($(USE_CUDA), 1)
    CC      := nvcc
    CFLAGS  := -ccbin mpicxx -O3 -std=c++17 -Xcompiler="-fPIC  -Wall -Wextra -march=native -O3 " -x cu
    LDFLAGS := -Xcompiler="$(MPI_LDFLAGS)" -lnvToolsExt
else ifeq ($(USE_HIP), 1)
    CC      := hipcc
    CFLAGS  := -O3 -std=c++17 -fPIC -ffast-math -Wall -Werror -Wextra -x hip
    LDFLAGS := -$(MPI_LDFLAGS)
    CFLAGS += --offload-arch=$(HIP_ARCH)
else
    $(error No backend specified: USE_CUDA=1 or USE_HIP=1)
endif


CFLAGS += -Xcompiler="$(MPI_CFLAGS)"
.PHONY: all clean

all: libmpipancake.so libmpisniffer.so test

libmpipancake.so: mpi_pancake.cpp
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -o $@ $<

libmpisniffer.so: mpi_sniffer.cpp
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -o $@ $<

test: test.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm libmpipancake.so libmpisniffer.so test

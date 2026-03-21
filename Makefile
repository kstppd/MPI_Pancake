USE_CUDA ?= 0
USE_HIP  ?= 0
MPI_CFLAGS := $(shell mpicxx --showme:compile)
MPI_LDFLAGS := $(addprefix -L,$(shell mpicxx --showme:libdirs)) $(addprefix -l,$(shell mpicxx --showme:libs))
BIN = libmpipancake.so

ifeq ($(USE_CUDA), 1)
    CC      := nvcc
    CFLAGS  := -O3 -std=c++17 -Xcompiler="-fPIC  -Wall -Werror -march=native -O3 " -x cu
    LDFLAGS := -shared -Xcompiler="$(MPI_LDFLAGS)"
else ifeq ($(USE_HIP), 1)
    CC      := hipcc
    CFLAGS  := -O3 -std=c++17 -fPIC -ffast-math -Wall -Werror -x hip
    LDFLAGS := -shared  $(MPI_LDFLAGS)
    CFLAGS += --offload-arch=$(HIP_ARCH)
else
    $(error No backend specified: USE_CUDA=1 or USE_HIP=1)
endif


CFLAGS += -Xcompiler="$(MPI_CFLAGS)"
all: libmpipancake

libmpipancake: mpi_pancake.cpp
	$(CC) ${CFLAGS} $(LDFLAGS) -o ${BIN} mpi_pancake.cpp

clean:
	rm -f $(BIN)

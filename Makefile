NVCC       = nvcc
HOSTCXX    = mpicxx
OPT        = -O3
CXXSTD     = -std=c++17
COMMON     = $(OPT) $(CXXSTD) -ccbin $(HOSTCXX) -x cu
XWARN      = -Wall -Wextra
LIB        = libmpi_pancake.so
LIB_SRC    = mpi_pancake.cpp
BENCH      = bench-gpu
BENCH_SRC  = bench-gpu.cpp


all: $(LIB) $(BENCH)

$(LIB): $(LIB_SRC)
	$(NVCC) $(COMMON) -Xcompiler="-fPIC -shared $(XWARN)" mpi_pancake.cpp -o $(LIB)

$(BENCH): $(BENCH_SRC)
	$(NVCC) $(COMMON) -Xcompiler="$(XWARN)" bench-gpu.cpp -o $(BENCH)

allclean:
	rm -f $(LIB) $(BENCH)


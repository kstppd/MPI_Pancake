#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include "catch.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuHostMalloc hipHostMalloc
#define gpuHostFree hipHostFree
#define gpuMemcpy hipMemcpy
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice
#define gpuGetErrorString hipGetErrorString
#else
#include <cuda_runtime.h>
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuHostMalloc cudaMallocHost
#define gpuHostFree cudaFreeHost
#define gpuMemcpy cudaMemcpy
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuSetDevice cudaSetDevice
#define gpuGetErrorString cudaGetErrorString
#endif

static inline void gpu_errchk(gpuError_t r, const char *f, int l) {
  if (r != gpuSuccess) {
    std::fprintf(stderr, "GPU error: %s at %s:%d\n", gpuGetErrorString(r), f,
                 l);
    std::fflush(stderr);
    std::exit(EXIT_FAILURE);
  }
}
#define GPU_ERRCHK(x) gpu_errchk((x), __FILE__, __LINE__)

static bool hindexed_pingpong_roundtrip_ok(size_t N_doubles,
                                           size_t whitespace_ratio,
                                           bool sync_mode) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double *h_init = nullptr;
  double *h_back = nullptr;
  GPU_ERRCHK(gpuHostMalloc((void **)&h_init, sizeof(double) * N_doubles));
  GPU_ERRCHK(gpuHostMalloc((void **)&h_back, sizeof(double) * N_doubles));

  for (size_t k = 0; k < N_doubles; ++k) {
    h_init[k] = std::sin(600.0 * double(k) / double(N_doubles));
  }

  double *d_buf = nullptr;
  GPU_ERRCHK(gpuMalloc((void **)&d_buf, sizeof(double) * N_doubles));
  GPU_ERRCHK(
      gpuMemcpy(d_buf, h_init, sizeof(double) * N_doubles, gpuMemcpyDefault));
  GPU_ERRCHK(gpuDeviceSynchronize());

  constexpr size_t partitions = 1u << 8; // 256
  const size_t total_bytes = sizeof(double) * N_doubles;
  const size_t part_bytes =
      (partitions == 0) ? total_bytes : (total_bytes / partitions);
  const size_t whitespace =
      (whitespace_ratio == 0) ? 0 : (part_bytes / whitespace_ratio);
  const size_t block_bytes =
      (part_bytes > whitespace) ? (part_bytes - whitespace) : 0;

  std::vector<MPI_Aint> displs(partitions);
  std::vector<int> blens(partitions);
  for (size_t i = 0; i < partitions; ++i) {
    displs[i] = static_cast<MPI_Aint>(i * part_bytes);
    size_t len = block_bytes + (i & 1u);
    if (displs[i] + static_cast<MPI_Aint>(len) >
        static_cast<MPI_Aint>(total_bytes)) {
      if (static_cast<MPI_Aint>(total_bytes) > displs[i]) {
        len =
            static_cast<size_t>(static_cast<MPI_Aint>(total_bytes) - displs[i]);
      } else {
        len = 0;
      }
    }
    blens[i] = static_cast<int>(len);
  }

  MPI_Datatype hindexed_bytes, wrapped;
  MPI_Type_create_hindexed(static_cast<int>(partitions), blens.data(),
                           displs.data(), MPI_BYTE, &hindexed_bytes);
  MPI_Type_commit(&hindexed_bytes);
  const int one = 1;
  const MPI_Aint zero_disp = 0;
  MPI_Type_create_struct(one, &one, &zero_disp, &hindexed_bytes, &wrapped);
  MPI_Type_commit(&wrapped);

  MPI_Request req;
  MPI_Status stat;
  if (rank == 0) {
    if (sync_mode) {
      MPI_Send(d_buf, 1, wrapped, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(d_buf, 1, wrapped, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Isend(d_buf, 1, wrapped, 1, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
      MPI_Irecv(d_buf, 1, wrapped, 1, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
    }
  } else if (rank == 1) {
    if (sync_mode) {
      MPI_Recv(d_buf, 1, wrapped, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(d_buf, 1, wrapped, 0, 0, MPI_COMM_WORLD);
    } else {
      MPI_Irecv(d_buf, 1, wrapped, 0, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
      MPI_Isend(d_buf, 1, wrapped, 0, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
    }
  }

  GPU_ERRCHK(
      gpuMemcpy(h_back, d_buf, sizeof(double) * N_doubles, gpuMemcpyDefault));
  GPU_ERRCHK(gpuDeviceSynchronize());

  bool ok = true;
  for (size_t i = 0; i < N_doubles; ++i) {
    if (h_back[i] != h_init[i]) {
      ok = false;
      break;
    }
  }

  MPI_Type_free(&wrapped);
  MPI_Type_free(&hindexed_bytes);
  GPU_ERRCHK(gpuFree(d_buf));
  GPU_ERRCHK(gpuHostFree(h_back));
  GPU_ERRCHK(gpuHostFree(h_init));
  return ok;
}


TEST_CASE("COMM_TYPE:: STRUCT HINDEXED MPI_BYTE ") {
  int world_size = 0, rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (world_size != 2) {
    if (rank == 0)
      WARN("Run this test with exactly 2 MPI processes (mpirun -np 2).");
    SUCCEED(); 
    return;
  }
  const size_t N = 1u << 20; 
  const size_t whitespace_ratio = 5;

  SECTION("Blocking Send/Recv") {
    bool ok =
        hindexed_pingpong_roundtrip_ok(N, whitespace_ratio, /*sync*/ true);
    REQUIRE(ok);
  }

  SECTION("Nonblocking Isend/Irecv") {
    bool ok =
        hindexed_pingpong_roundtrip_ok(N, whitespace_ratio, /*sync*/ false);
    REQUIRE(ok);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int devcount = 0;
  GPU_ERRCHK(gpuGetDeviceCount(&devcount));
  if (devcount > 0) {
    GPU_ERRCHK(gpuSetDevice(rank % devcount));
  } else {
    if (rank == 0)
      std::fprintf(stderr, "Warning: no GPU devices visible.\n");
  }

  Catch::Session session;
  int result = session.applyCommandLine(argc, argv);
  if (result != 0) {
    MPI_Finalize();
    return result;
  }

  result = session.run();

  MPI_Finalize();
  return result;
}

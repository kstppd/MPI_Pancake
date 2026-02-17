// Mainly based on Juhani Kataja's benchmark
#include <cstdint>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuStream_t hipStream_t
#define gpuSuccess hipSuccess
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMalloc(ptr, size) hipMalloc(ptr, size)
#define gpuFree(ptr) hipFree(ptr)
#define gpuHostMalloc(ptr, size) hipHostMalloc(ptr, size)
#define gpuHostFree(ptr) hipHostFree(ptr)
#define gpuMemcpy(dst, src, sz, k) hipMemcpy(dst, src, sz, k)
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice
#define gpuGetErrorString hipGetErrorString
#else
#include <cuda_runtime.h>
#define gpuError_t cudaError_t
#define gpuStream_t cudaStream_t
#define gpuSuccess cudaSuccess
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMalloc(ptr, size) cudaMalloc(ptr, size)
#define gpuFree(ptr) cudaFree(ptr)
#define gpuHostMalloc(ptr, size) cudaMallocHost(ptr, size)
#define gpuHostFree(ptr) cudaFreeHost(ptr)
#define gpuMemcpy(dst, src, sz, k) cudaMemcpy(dst, src, sz, k)
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuSetDevice cudaSetDevice
#define gpuGetErrorString cudaGetErrorString
#endif

#define NITER 20
#define gpu_ERRCHK(result) gpu_errchk(result, __FILE__, __LINE__)

static inline void gpu_errchk(gpuError_t result, const char *file,
                              int32_t line) {
  if (result != gpuSuccess) {
    printf("\n\n%s in %s at line %d\n", gpuGetErrorString(result), file, line);
    exit(EXIT_FAILURE);
  }
}

struct BenchResult {
  double avg_time;
  bool is_sane;
};

BenchResult pingpong_hindexed_bytes(int rank, int N, size_t whitespace_ratio,
                                    bool synch, int tag) {
  float *data_d;
  float *data_h;

  gpuHostMalloc((void **)&data_h, sizeof(float) * N);
  for (size_t k = 0; k < N; ++k) {
    data_h[k] = sin(6 * 100.0 * ((float)k) / ((float)N));
  }

  gpu_ERRCHK(gpuMalloc((void **)&data_d, sizeof(float) * N));
  gpu_ERRCHK(gpuMemcpy(data_d, data_h, sizeof(float) * N, gpuMemcpyDefault));

  constexpr size_t partitions = 1 << 8;
  size_t partition_size = sizeof(float) * N / partitions;
  size_t whitespace_size = partition_size / whitespace_ratio;
  size_t blocksize = partition_size - whitespace_size;

  MPI_Aint displ[partitions];
  int blens[partitions];
  for (size_t i = 0; i < partitions; ++i) {
    displ[i] = i * partition_size;
    blens[i] = (int)blocksize + (i % 2);
  }

  MPI_Datatype hindexed_bytes, s_hindexed_bytes;
  MPI_Type_create_hindexed(partitions, blens, displ, MPI_BYTE, &hindexed_bytes);
  MPI_Type_commit(&hindexed_bytes);

  MPI_Aint s_displ[1] = {0};
  int s_blen[1] = {1};
  const MPI_Datatype s_oldtypes[1] = {hindexed_bytes};
  MPI_Type_create_struct(1, s_blen, s_displ, s_oldtypes, &s_hindexed_bytes);
  MPI_Type_commit(&s_hindexed_bytes);

  float t1, t2;
  MPI_Request req;
  MPI_Status stat;

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if (rank == 0) {
      if (synch) {
        MPI_Send(data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Isend(data_d, 1, s_hindexed_bytes, 1, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &stat);
        MPI_Irecv(data_d, 1, s_hindexed_bytes, 1, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &stat);
      }
    } else if (rank == 1) {
      if (synch) {
        MPI_Recv(data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(data_d, 1, s_hindexed_bytes, 0, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &stat);
        MPI_Isend(data_d, 1, s_hindexed_bytes, 0, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &stat);
      }
    }
  }
  t2 = MPI_Wtime();

  bool sane = true;
  if (rank == 0) {
    float *verify_h = (float *)malloc(sizeof(float) * N);
    gpu_ERRCHK(
        gpuMemcpy(verify_h, data_d, sizeof(float) * N, gpuMemcpyDefault));

    for (size_t i = 0; i < N; ++i) {
      if (std::abs(verify_h[i] - data_h[i]) > 1e-4) {
        sane = false;
        break;
      }
    }
    free(verify_h);
  }

  int sane_int = sane ? 1 : 0;
  MPI_Bcast(&sane_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  sane = (sane_int == 1);

  MPI_Type_free(&hindexed_bytes);
  MPI_Type_free(&s_hindexed_bytes);
  gpu_ERRCHK(gpuFree(data_d));
  gpu_ERRCHK(gpuHostFree(data_h));
  return {(t2 - t1) / NITER, sane};
}

int main(int argc, char *argv[]) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0)
      printf("Run with exactly 2 processes.\n");
    MPI_Finalize();
    return 0;
  }

  // Print Header once before the loop
  if (rank == 0) {
    fprintf(stderr, "%-15s | %-12s | %-12s | %-12s | %-12s | %-6s\n", "Bytes",
            "Pancake FP32 (ms)", "Pancake FP16 (ms)", "Pancake FP8 (ms)",
            "Control (ms)", "VALIDATION");
    fprintf(stderr, "----------------------------------------------------------"
                    "----------------------------\n");
  }

  for (int N = 1 << 5; N < (1 << 20); N = N << 1) {
    BenchResult res_pancake_fp32 =
        pingpong_hindexed_bytes(rank, N, 5, false, 0);
    BenchResult res_pancake_fp16 =
        pingpong_hindexed_bytes(rank, N, 5, false, 42);
    BenchResult res_pancake_fp8 =
        pingpong_hindexed_bytes(rank, N, 5, false, 43);
    BenchResult res_sync = pingpong_hindexed_bytes(rank, N, 5, true, 0);

    if (rank == 0) {
      float moved_bytes = sizeof(float) * ((float)N) * (5.0 / 6.0);
      bool all_sane = res_pancake_fp32.is_sane && res_pancake_fp16.is_sane &&
                      res_pancake_fp8.is_sane && res_sync.is_sane;

      fprintf(stderr, "%-15zu | %-12.4f | %-12.4f | %-12.4f | %-12.4f | %-6s\n",
              (size_t)moved_bytes, res_pancake_fp32.avg_time * 1000.0,
              res_pancake_fp16.avg_time * 1000.0,
              res_pancake_fp8.avg_time * 1000.0, res_sync.avg_time * 1000.0,
              all_sane ? "true" : "false");
    }
  }

  MPI_Finalize();
  return 0;
}

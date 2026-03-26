#include <cstdint>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMalloc(ptr, size) hipMalloc(ptr, size)
#define gpuFree(ptr) hipFree(ptr)
#define gpuMemset(ptr, val, size) hipMemset(ptr, val, size)
#define gpuHostMalloc(ptr, size) hipHostMalloc(ptr, size)
#define gpuHostFree(ptr) hipHostFree(ptr)
#define gpuMemcpy(dst, src, sz, k) hipMemcpy(dst, src, sz, k)
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuGetErrorString hipGetErrorString
#else
#include <cuda_runtime.h>
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMalloc(ptr, size) cudaMalloc(ptr, size)
#define gpuFree(ptr) cudaFree(ptr)
#define gpuMemset(ptr, val, size) cudaMemset(ptr, val, size)
#define gpuHostMalloc(ptr, size) cudaMallocHost(ptr, size)
#define gpuHostFree(ptr) cudaFreeHost(ptr)
#define gpuMemcpy(dst, src, sz, k) cudaMemcpy(dst, src, sz, k)
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
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

template <typename T = float>
BenchResult pingpong_hindexed_bytes(int rank, int N, size_t whitespace_ratio,
                                    bool synch, int tag) {
  T *data_d;
  T *data_h;
  gpuHostMalloc((void **)&data_h, sizeof(T) * N);
  for (size_t k = 0; k < (size_t) N; ++k) {
    data_h[k] = sin(6 * 100.0 * ((T)k) / ((T)N));
  }

  gpu_ERRCHK(gpuMalloc((void **)&data_d, sizeof(T) * N));
  gpu_ERRCHK(gpuMemset(data_d, 0, sizeof(T) * N));
  if (rank == 0) {
    gpu_ERRCHK(gpuMemcpy(data_d, data_h, sizeof(T) * N, gpuMemcpyDefault));
  }
  gpu_ERRCHK(gpuDeviceSynchronize());

  constexpr size_t partitions = 1 << 8;
  size_t partition_size = sizeof(T) * N / partitions;
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

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  MPI_Request req;
  for (size_t iter = 0; iter < NITER; ++iter) {
    if (rank == 0) {
      if (synch) {
        MPI_Send(data_d, 1, s_hindexed_bytes, 1, tag, MPI_COMM_WORLD);
        MPI_Recv(data_d, 1, s_hindexed_bytes, 1, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Isend(data_d, 1, s_hindexed_bytes, 1, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Irecv(data_d, 1, s_hindexed_bytes, 1, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
      }
    } else if (rank == 1) {
      if (synch) {
        MPI_Recv(data_d, 1, s_hindexed_bytes, 0, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(data_d, 1, s_hindexed_bytes, 0, tag, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(data_d, 1, s_hindexed_bytes, 0, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Isend(data_d, 1, s_hindexed_bytes, 0, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
      }
    }
  }

  double t2 = MPI_Wtime();
  bool local_sane = true;
  T *verify_h = (T *)malloc(sizeof(T) * N);
  gpu_ERRCHK(gpuMemcpy(verify_h, data_d, sizeof(T) * N, gpuMemcpyDefault));
  char *v_ptr = (char *)verify_h;
  char *o_ptr = (char *)data_h;
  for (size_t i = 0; i < partitions; ++i) {
    for (int j = 0; j < blens[i]; ++j) {
      size_t idx = (size_t)displ[i] + j;
      if (v_ptr[idx] != o_ptr[idx]) {
        local_sane = false;
        break;
      }
    }
    if (!local_sane)
      break;
  }

  free(verify_h);
  int sane_int = local_sane ? 1 : 0;
  int global_sane;
  MPI_Allreduce(&sane_int, &global_sane, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  MPI_Type_free(&hindexed_bytes);
  MPI_Type_free(&s_hindexed_bytes);
  gpu_ERRCHK(gpuFree(data_d));
  gpu_ERRCHK(gpuHostFree(data_h));
  return {(t2 - t1) / NITER, global_sane == 1};
}

template <typename T> bool run_test() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    printf("%-12s | %-12s | %-12s | %-8s\n", "Bytes Sent", "Async(ms)",
           "Sync(ms)", "Sane");
    printf("------------------------------------------------------------\n");
  }

  for (int N = 1 << 10; N <= (1 << 20); N <<= 1) {
    BenchResult res_async = pingpong_hindexed_bytes<T>(rank, N, 5, false, 0);
    BenchResult res_sync = pingpong_hindexed_bytes<T>(rank, N, 5, true, 0);
    bool sane = res_async.is_sane && res_sync.is_sane;
    if (!sane) {
      fprintf(stderr,"ERROR in MPI_PACNAKE\n");
      return false;
    }
    if (rank == 0) {
      size_t moved_bytes = (N * sizeof(T) * 4) / 5;
      printf("%-12zu | %-12.4f | %-12.4f | %-8s\n", moved_bytes,
             res_async.avg_time * 1000.0, res_sync.avg_time * 1000.0,
             sane ? "OK" : "FAIL");
    }
  }
  printf("\n");
  return true;
}

int main(int argc, char *argv[]) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0) {
      printf("Run with 2 tasks.\n");
    }
    goto error_defer;
  }

  if (!run_test<int>())     goto error_defer;
  if (!run_test<int64_t>()) goto error_defer;
  if (!run_test<float>())   goto error_defer;
  if (!run_test<double>())  goto error_defer;
  
  MPI_Finalize();
  return 0;

error_defer:
  MPI_Finalize();
  return 1;
}

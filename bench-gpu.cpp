#include <cstdint>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

#define GPU_ERRCHK(result) gpu_errchk(result, __FILE__, __LINE__)
static inline void gpu_errchk(gpuError_t result, const char *file,
                              int32_t line) {
  if (result != gpuSuccess) {
    printf("\n\n%s in %s at line %d\n", gpuGetErrorString(result), file, line);
    exit(EXIT_FAILURE);
  }
}

#define NITER 100

double pingpong_hindexed_bytes(int rank, int N, size_t whitespace_ratio,
                               bool synch, double *out_max_abs_err) {
  double *data_d;
  double *data_after;
  GPU_ERRCHK(gpuHostMalloc((void **)&data_after, sizeof(double) * N));

  double *data_init;
  GPU_ERRCHK(gpuHostMalloc((void **)&data_init, sizeof(double) * N));
  for (size_t k = 0; k < (size_t)N; ++k) {
    data_init[k] = sin(6 * 100.0 * ((double)k) / ((double)N));
  }
  GPU_ERRCHK(gpuMalloc((void **)&data_d, sizeof(double) * N));
  GPU_ERRCHK(
      gpuMemcpy(data_d, data_init, sizeof(double) * N, gpuMemcpyDefault));

  constexpr size_t partitions = 1 << 8;
  size_t partition_size = sizeof(double) * (size_t)N / partitions;
  size_t whitespace_size = partition_size / whitespace_ratio;
  size_t blocksize = partition_size - whitespace_size;

  MPI_Aint displ[partitions];
  int blens[partitions];
  for (size_t i = 0; i < partitions; ++i) {
    displ[i] = (MPI_Aint)(i * partition_size);
    blens[i] = (int)(blocksize + (i % 2));
  }

  MPI_Datatype hindexed_bytes, s_hindexed_bytes;
  MPI_Type_create_hindexed((int)partitions, blens, displ, MPI_BYTE,
                           &hindexed_bytes);
  MPI_Type_commit(&hindexed_bytes);

  const MPI_Aint s_displ[1] = {0};
  const int s_blen[1] = {1};
  const MPI_Datatype s_oldtypes[1] = {hindexed_bytes};
  MPI_Type_create_struct(1, s_blen, s_displ, s_oldtypes, &s_hindexed_bytes);
  MPI_Type_commit(&s_hindexed_bytes);

  // Warm-up
  MPI_Request req;
  MPI_Status stat;
  if (rank == 0) {
    if (synch) {
      MPI_Send((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD);
      MPI_Recv((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      MPI_Isend((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
                &req);
      MPI_Wait(&req, &stat);
      MPI_Irecv((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
                &req);
      MPI_Wait(&req, &stat);
    }
  } else if (rank == 1) {
    if (synch) {
      MPI_Recv((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD);
    } else {
      MPI_Irecv((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
                &req);
      MPI_Wait(&req, &stat);
      MPI_Isend((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
                &req);
      MPI_Wait(&req, &stat);
    }
  }

  // Timing
  double t1, t2;
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if (rank == 0) {
      if (synch) {
        MPI_Send((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD);
        MPI_Recv((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Isend((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
                  &req);
        MPI_Wait(&req, &stat);
        MPI_Irecv((void *)data_d, 1, s_hindexed_bytes, 1, 0, MPI_COMM_WORLD,
                  &req);
        MPI_Wait(&req, &stat);
      }
    } else if (rank == 1) {
      if (synch) {
        MPI_Recv((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
                  &req);
        MPI_Wait(&req, &stat);
        MPI_Isend((void *)data_d, 1, s_hindexed_bytes, 0, 0, MPI_COMM_WORLD,
                  &req);
        MPI_Wait(&req, &stat);
      }
    }
  }
  t2 = MPI_Wtime();
  GPU_ERRCHK(
      gpuMemcpy(data_after, data_d, sizeof(double) * N, gpuMemcpyDefault));
  double max_abs_err = 0.0;
  for (size_t k = 0; k < (size_t)N; ++k) {
    double ref = sin(6 * 100.0 * ((double)k) / ((double)N));
    double err = fabs(data_after[k] - ref);
    if (err > max_abs_err)
      max_abs_err = err;
  }
  if (out_max_abs_err)
    *out_max_abs_err = max_abs_err;

  MPI_Type_free(&hindexed_bytes);
  MPI_Type_free(&s_hindexed_bytes);
  GPU_ERRCHK(gpuFree(data_d));
  GPU_ERRCHK(gpuHostFree(data_after));
  GPU_ERRCHK(gpuHostFree(data_init));
  return (t2 - t1) / NITER;
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

  int devcount;
  GPU_ERRCHK(gpuGetDeviceCount(&devcount));
  if (rank == 0 || devcount > 1) {
    if (devcount > 1)
      GPU_ERRCHK(gpuSetDevice(rank % devcount));
  }
  fflush(stdout);
  if (rank == 0) {
    printf("bytes transferred (kB), async (ms), sync (ms), async_max_err, "
           "sync_max_err\n");
  }
  for (int N = 1 << 14; N < 1 << 18; N = N << 1) {
    double err_async = 0.0, err_sync = 0.0;
    double dt_struct_async =
        pingpong_hindexed_bytes(rank, N, 5, false, &err_async);
    double dt_struct_sync =
        pingpong_hindexed_bytes(rank, N, 5, true, &err_sync);
    if (rank == 0) {
      size_t partitions = 1 << 8;
      size_t partition_size = sizeof(double) * (size_t)N / partitions;
      size_t whitespace_size = partition_size / 5; // whitespace_ratio = 5
      size_t blocksize = partition_size - whitespace_size;
      double effective_bytes = (double)partitions * (blocksize + 0.5);
      printf("%g, %g, %g, %.3e, %.3e\n", effective_bytes / 1e3,
             dt_struct_async * 1000.0, dt_struct_sync * 1000.0, err_async,
             err_sync);
    }
  }

  MPI_Finalize();
  return 0;
}

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

// Collects blocks of columns to a buffer for sending over MPI (manual gpu-gpu)
extern "C" {
void gpu_packbuf(double *buf, double *pack_buf, size_t rows, size_t cols,
                 size_t blocksize);
void gpu_unpackbuf(double *buf, double *pack_buf, size_t rows, size_t cols,
                   size_t blocksize);
void add_one_unpacked(double *buf, size_t rows, size_t cols, size_t blocksize);
void add_one_packed(double *pack_buf, size_t rows, size_t cols,
                    size_t blocksize);
void fill(double *buf, double value, size_t rows, size_t cols);
}

#define NITER 100
const size_t gpu_blocksize = 512;

// Portable GPU error checking macro and function
#define GPU_ERRCHK(result) gpu_errchk(result, __FILE__, __LINE__)
static inline void gpu_errchk(gpuError_t result, const char *file,
                              int32_t line) {
  if (result != gpuSuccess) {
    printf("\n\n%s in %s at line %d\n", gpuGetErrorString(result), file, line);
    exit(EXIT_FAILURE);
  }
}

// Collects blocks of columns to a buffer for sending over MPI (manual gpu-gpu)
__global__ void k_gpu_packbuf(double *buf, double *pack_buf, size_t rows,
                              size_t cols, size_t blocksize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < blocksize * cols; i += stride) {
    size_t r = i / blocksize;
    size_t s = i % blocksize;
    pack_buf[r * blocksize + s] = buf[r * rows + s];
  }
}

extern "C" {
void gpu_packbuf(double *buf, double *pack_buf, size_t rows, size_t cols,
                 size_t blocksize) {
  size_t send_count = blocksize * cols;
  size_t gpu_gridsize = (send_count + gpu_blocksize - 1) / gpu_blocksize;
  k_gpu_packbuf<<<gpu_gridsize, gpu_blocksize>>>(buf, pack_buf, rows, cols,
                                                 blocksize);
}
}

// Unpacks contiguous buffer to non-contiguous buf
__global__ void k_gpu_unpackbuf(double *buf, double *pack_buf, size_t rows,
                                size_t cols, size_t blocksize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < blocksize * cols; i += stride) {
    size_t r = i / blocksize;
    size_t s = i % blocksize;
    buf[r * rows + s] = pack_buf[r * blocksize + s];
  }
}

extern "C" {
void gpu_unpackbuf(double *buf, double *pack_buf, size_t rows, size_t cols,
                   size_t blocksize) {
  size_t send_count = blocksize * cols;
  size_t gpu_gridsize = (send_count + gpu_blocksize - 1) / gpu_blocksize;
  k_gpu_unpackbuf<<<gpu_gridsize, gpu_blocksize>>>(buf, pack_buf, rows, cols,
                                                   blocksize);
}
}

__global__ void k_add_one_unpacked(double *buf, size_t rows, size_t cols,
                                   size_t blocksize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  constexpr double one = double(1.0) / NITER;
  for (size_t i = tid; i < blocksize * cols; i += stride) {
    size_t r = i / blocksize;
    size_t s = i % blocksize;
    buf[r * rows + s] += one;
  }
}

extern "C" {
void add_one_unpacked(double *buf, size_t rows, size_t cols, size_t blocksize) {
  size_t send_count = blocksize * cols;
  size_t gpu_gridsize = (send_count + gpu_blocksize - 1) / gpu_blocksize;
  k_add_one_unpacked<<<gpu_gridsize, gpu_blocksize>>>(buf, rows, cols,
                                                      blocksize);
}
}

__global__ void k_add_one_packed(double *pack_buf, size_t rows, size_t cols,
                                 size_t blocksize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  constexpr double one = double(1.0) / NITER;
  for (size_t i = tid; i < blocksize * cols; i += stride) {
    pack_buf[i] += one;
  }
}

extern "C" {
void add_one_packed(double *pack_buf, size_t rows, size_t cols,
                    size_t blocksize) {
  size_t send_count = blocksize * cols;
  size_t gpu_gridsize = (send_count + gpu_blocksize - 1) / gpu_blocksize;
  k_add_one_packed<<<gpu_gridsize, gpu_blocksize>>>(pack_buf, rows, cols,
                                                    blocksize);
}
}

__global__ void k_fill(double *buf, double value, size_t rows, size_t cols) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < rows * cols; i += stride) {
    buf[i] = value;
  }
}

extern "C" {
void fill(double *buf, double value, size_t rows, size_t cols) {
  size_t gpu_gridsize = (cols * rows + gpu_blocksize - 1) / gpu_blocksize;
  k_fill<<<gpu_gridsize, gpu_blocksize>>>(buf, value, rows, cols);
}
}

void getNodeInfo(int *nodeRank, int *nodeProcs, int *devCount) {
  MPI_Comm intranodecomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &intranodecomm);

  MPI_Comm_rank(intranodecomm, nodeRank);
  MPI_Comm_size(intranodecomm, nodeProcs);

  MPI_Comm_free(&intranodecomm);
  int count;
  gpuGetDeviceCount(&count);
  *devCount = count;
}

double pingpong_gpu_manual(int rank, double *databuf, size_t N, size_t rows,
                           size_t cols, size_t blocksize, double &checksum) {
  double *gpu_buffer;
  size_t send_count = blocksize * cols;
  double t1, t2;

  gpuMalloc((void **)&gpu_buffer, sizeof(double) * send_count);

  fill(databuf, double(rank), rows, cols);
  gpuDeviceSynchronize();

  // Warmup
  gpu_packbuf(databuf, gpu_buffer, rows, cols, blocksize);
  gpuDeviceSynchronize();
  if (rank == 0) {
    MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    add_one_packed(gpu_buffer, rows, cols, blocksize);
    gpuDeviceSynchronize();
    MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  gpu_unpackbuf(databuf, gpu_buffer, rows, cols, blocksize);

  fill(databuf, double(rank), rows, cols);
  gpuDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (int iter = 0; iter < NITER; ++iter) {
    gpu_packbuf(databuf, gpu_buffer, rows, cols, blocksize);
    gpuDeviceSynchronize();
    if (rank == 0) {
      MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      add_one_packed(gpu_buffer, rows, cols, blocksize);
      gpuDeviceSynchronize();
      MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    gpu_unpackbuf(databuf, gpu_buffer, rows, cols, blocksize);
    gpuDeviceSynchronize();
  }
  t2 = MPI_Wtime();

  if (rank == 0) {
    checksum = 0.0;
    double *h_databuf;

    GPU_ERRCHK(
        gpuHostMalloc((void **)&h_databuf, sizeof(double) * rows * cols));
    GPU_ERRCHK(gpuMemcpy(h_databuf, databuf, sizeof(double) * rows * cols,
                         gpuMemcpyDefault));
    for (size_t i = 0; i < rows * cols; ++i) {
      checksum += h_databuf[i];
    }
    GPU_ERRCHK(gpuHostFree(h_databuf));
  }

  gpuFree(gpu_buffer);
  return (t2 - t1) / NITER;
}

double pingpong_derived(int rank, double *databuf, size_t N, size_t rows,
                        size_t cols, size_t blocksize, double &checksum) {
  MPI_Datatype rowtype;
  double t1, t2;
  MPI_Type_vector(cols, blocksize, rows, MPI_DOUBLE, &rowtype);
  MPI_Type_commit(&rowtype);

  fill(databuf, double(rank), rows, cols);
  gpuDeviceSynchronize();

  // warmup
  if (rank == 0) {
    MPI_Send(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Recv(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    add_one_unpacked(databuf, rows, cols, blocksize);
    gpuDeviceSynchronize();
    MPI_Send(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD);
  }

  fill(databuf, double(rank), rows, cols);
  gpuDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if (rank == 0) {
      MPI_Send(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      add_one_unpacked(databuf, rows, cols, blocksize);
      gpuDeviceSynchronize();
      MPI_Send(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD);
    }
  }

  t2 = MPI_Wtime();

  if (rank == 0) {
    checksum = 0.0;
    double *h_databuf;

    GPU_ERRCHK(
        gpuHostMalloc((void **)&h_databuf, sizeof(double) * rows * cols));
    GPU_ERRCHK(gpuMemcpy(h_databuf, databuf, sizeof(double) * rows * cols,
                         gpuMemcpyDefault));
    for (size_t i = 0; i < rows * cols; ++i) {
      checksum += h_databuf[i];
    }
    GPU_ERRCHK(gpuHostFree(h_databuf));
  }

  MPI_Type_free(&rowtype);
  return (t2 - t1) / NITER;
}

#define N_data_a 16
#define N_data_b 15

typedef struct {
  uint8_t data_a[N_data_a];
  uint8_t data_b[N_data_b];
} data_t;

double pingpong_struct_mpi_byte(int rank, size_t N) {
  data_t *data;
  data_t *data_d;
  data = (data_t *)malloc(N * sizeof(data_t));

  // fill on cpu
  for (size_t i = 0; i < N; ++i) {
    for (size_t a = 0; a < N_data_a; ++a)
      data[i].data_a[a] = (uint8_t)'a';
    for (size_t b = 0; b < N_data_b; ++b)
      data[i].data_a[b] = (uint8_t)'b';
  }
  const int blen[2] = {N_data_a, N_data_b};

  const MPI_Aint array_of_displacements[2] = {
      0, (ptrdiff_t)((char *)&data[0].data_b[0] - (char *)&data[0])};
  const MPI_Datatype oldtypes[2] = {MPI_BYTE, MPI_BYTE};
  MPI_Datatype m_bytestruct;

  MPI_Type_create_struct(2, blen, array_of_displacements, oldtypes,
                         &m_bytestruct);
  MPI_Type_commit(&m_bytestruct);

  GPU_ERRCHK(gpuMalloc((void **)&data_d, sizeof(data_t) * N));
  GPU_ERRCHK(gpuMemcpy((void *)data_d, (void *)data, sizeof(data_t) * N,
                       gpuMemcpyDefault));

  MPI_Request req;
  MPI_Status stat;

  double t1, t2;
  // Warmup
  if (rank == 0) {
    MPI_Isend((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
    MPI_Irecv((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
  } else if (rank == 1) {
    MPI_Irecv((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
    MPI_Isend((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if (rank == 0) {
      MPI_Isend((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
      MPI_Irecv((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
    } else if (rank == 1) {
      MPI_Irecv((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
      MPI_Isend((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
    }
  }
  t2 = MPI_Wtime();
  MPI_Type_free(&m_bytestruct);
  GPU_ERRCHK(gpuFree(data_d));
  free(data);
  return (t2 - t1) / NITER;
}

double pingpong_struct_mpi_byte_synchr(int rank, size_t N) {
  data_t *data;
  data_t *data_d;
  data = (data_t *)malloc(N * sizeof(data_t));

  // fill on cpu
  for (size_t i = 0; i < N; ++i) {
    for (size_t a = 0; a < N_data_a; ++a)
      data[i].data_a[a] = (uint8_t)'a';
    for (size_t b = 0; b < N_data_b; ++b)
      data[i].data_a[b] = (uint8_t)'b';
  }
  const int blen[2] = {N_data_a, N_data_b};

  const MPI_Aint array_of_displacements[2] = {
      0, (ptrdiff_t)((char *)&data[0].data_b[0] - (char *)&data[0])};
  const MPI_Datatype oldtypes[2] = {MPI_BYTE, MPI_BYTE};
  MPI_Datatype m_bytestruct;

  MPI_Type_create_struct(2, blen, array_of_displacements, oldtypes,
                         &m_bytestruct);
  MPI_Type_commit(&m_bytestruct);

  GPU_ERRCHK(gpuMalloc((void **)&data_d, sizeof(data_t) * N));
  GPU_ERRCHK(gpuMemcpy((void *)data_d, (void *)data, sizeof(data_t) * N,
                       gpuMemcpyDefault));

  double t1, t2;
  // Warmup
  if (rank == 0) {
    MPI_Send((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD);
    MPI_Recv((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Recv((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Send((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if (rank == 0) {
      MPI_Send((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD);
      MPI_Recv((void *)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send((void *)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD);
    }
  }
  t2 = MPI_Wtime();
  MPI_Type_free(&m_bytestruct);
  GPU_ERRCHK(gpuFree(data_d));
  free(data);
  return (t2 - t1) / NITER;
}

double pingpong_hindexed_bytes(int rank, int N, size_t whitespace_ratio,
                               bool synch) {
  double *data_d;
  double *data_h;
  GPU_ERRCHK(gpuHostMalloc((void **)&data_h, sizeof(double) * N));
  fprintf(stderr, "Mallcoed %p \n", data_h);
  for (size_t k = 0; k < N; ++k)
    data_h[k] = sin(6 * 100.0 * ((double)k) / ((double)N));

  GPU_ERRCHK(gpuMalloc((void **)&data_d, sizeof(double) * N));
  GPU_ERRCHK(gpuMemcpy(data_d, data_h, sizeof(double) * N, gpuMemcpyDefault));

  constexpr size_t partitions = 1 << 8;
  size_t partition_size = sizeof(double) * N / partitions;
  size_t whitespace_size = partition_size / whitespace_ratio;
  size_t blocksize = partition_size - whitespace_size;

  MPI_Aint displ[partitions];
  int blens[partitions];
  for (size_t i = 0; i < partitions; ++i) {
    displ[i] = i * partition_size;
    blens[i] = blocksize + i % 2;
  }

  MPI_Datatype hindexed_bytes, s_hindexed_bytes;
  MPI_Type_create_hindexed(partitions, blens, displ, MPI_BYTE, &hindexed_bytes);
  MPI_Type_commit(&hindexed_bytes);

  MPI_Aint s_displ[1] = {0};
  int s_blen[1];
  s_blen[0] = 1;
  const MPI_Datatype s_oldtypes[1] = {hindexed_bytes};

  MPI_Type_create_struct(1, s_blen, s_displ, s_oldtypes, &s_hindexed_bytes);
  MPI_Type_commit(&s_hindexed_bytes);

  double t1, t2;
  MPI_Request req;
  MPI_Status stat;

  // Warmup
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

  MPI_Type_free(&hindexed_bytes);
  MPI_Type_free(&s_hindexed_bytes);
  GPU_ERRCHK(gpuFree(data_d));
  GPU_ERRCHK(gpuHostFree(data_h));
  return (t2 - t1) / NITER;
}

int main(int argc, char *argv[]) {
  fprintf(stderr, "Hello world!\n");
  int rank, size;
  double *matrix;
  constexpr size_t NCOLS = 1 << 12;
  constexpr size_t NROWS = 1 << 12;
  double dt_struct_async, dt_struct_sync;

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
  printf("There are %i devices for rank %i\n", devcount, rank);
  if (devcount > 1) { // Set device based on rank if multiple GPUs are visible
    GPU_ERRCHK(gpuSetDevice(rank % devcount));
  }
  fflush(stdout);

  gpuMalloc((void **)&matrix, sizeof(double) * NCOLS * NROWS);

  if (rank == 0)
    printf("bytes transferred (kB), async (ms), sync (ms)\n");

  for (int N = 1 << 14; N < 1 << 26; N = N << 1) {
    dt_struct_async = pingpong_hindexed_bytes(rank, N, 5, false);
    dt_struct_sync = pingpong_hindexed_bytes(rank, N, 5, true);

    if (rank == 0) {
      // Calculate effective data size based on the hindexed type parameters
      size_t partitions = 1 << 8;
      size_t partition_size = sizeof(double) * N / partitions;
      size_t whitespace_size = partition_size / 5; // whitespace_ratio = 5
      size_t blocksize = partition_size - whitespace_size;
      double effective_bytes =
          (double)partitions * (blocksize + 0.5); // avg of (blocksize + i%2)
      printf("%g, %g, %g\n", effective_bytes / 1e3, dt_struct_async * 1000,
             dt_struct_sync * 1000);
    }
  }
  gpuFree(matrix);
  MPI_Finalize();
  return 0;
}

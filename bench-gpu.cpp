#if 0
CC -O3 -g -xhip -Wno-unused-result bench-gpu.cpp -o bench-gpu-lumi
echo run like
echo HSA_XNACK=0 LD_PRELOAD=./libhooks.so SLURM_DEBUG=1 srun --nodes 2 --ntasks-per-node=1 -p dev-g -A project_462000007 -n 2 --gpus=2 -t 00:05:00 ./bench-gpu-lumi
exit
#endif

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#ifndef HOP_TARGET_CUDA
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HOP_TARGET_CUDA
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif


// Collects blocks of columns to a buffer for sending over MPI (manual gpu-gpu)
extern "C" {
  void gpu_packbuf(double* buf, double* pack_buf, size_t rows, size_t cols, size_t blocksize);
  void gpu_unpackbuf(double* buf, double* pack_buf, size_t rows, size_t cols, size_t blocksize);
  void add_one_unpacked(double* buf, size_t rows, size_t cols, size_t blocksize);
  void add_one_packed(double* pack_buf, size_t rows, size_t cols, size_t blocksize);
  void fill(double* buf, double value, size_t rows, size_t cols);
}

#define NITER 10
const size_t gpu_blocksize = 512;

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file,
                              int32_t line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}



// Collects blocks of columns to a buffer for sending over MPI (manual gpu-gpu)
__global__ void k_gpu_packbuf(double* buf, double* pack_buf, 
                            size_t rows, size_t cols, size_t blocksize) {
  size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < blocksize*cols; i += stride) {
    size_t r = i / blocksize;
    size_t s = i % blocksize;
    pack_buf[r*blocksize + s] = buf[r*rows + s];
  }

}

extern "C" { 
  void gpu_packbuf(double* buf, double* pack_buf, size_t rows, size_t cols, size_t blocksize) {
    size_t send_count = blocksize*cols;
    size_t gpu_gridsize = (send_count + gpu_blocksize - 1)/gpu_blocksize;
    k_gpu_packbuf<<<gpu_gridsize, gpu_blocksize>>>(buf, pack_buf, rows, cols, blocksize);
  }
}

// Unpacks contiguous buffer to non-contiguous buf
__global__ void k_gpu_unpackbuf(double* buf, double* pack_buf, size_t rows, size_t cols, size_t blocksize) {
  size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < blocksize*cols; i += stride) {
    size_t r = i/blocksize;
    size_t s = i % blocksize;
    buf[r*rows + s] = pack_buf[r*blocksize+s];
  }
}

extern "C" {
  void gpu_unpackbuf(double* buf, double* pack_buf, size_t rows, size_t cols, size_t blocksize) {
    size_t send_count = blocksize*cols;
    size_t gpu_gridsize = (send_count + gpu_blocksize - 1)/gpu_blocksize;
    k_gpu_unpackbuf<<<gpu_gridsize, gpu_blocksize>>>(buf, pack_buf, rows, cols, blocksize);
  }
}

__global__ void k_add_one_unpacked(double* buf, size_t rows, size_t cols, size_t blocksize) {
  size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  constexpr double one = double(1.0)/NITER;

  for (size_t i = tid; i < blocksize*cols; i+=stride) {
    size_t r = i/blocksize;
    size_t s = i % blocksize;
    buf[r*rows + s] += one;
  }
}

extern "C" {
  void add_one_unpacked(double* buf, size_t rows, size_t cols, size_t blocksize) {
    size_t send_count = blocksize*cols;
    size_t gpu_gridsize = (send_count + gpu_blocksize - 1)/gpu_blocksize;
    k_add_one_unpacked<<<gpu_gridsize, gpu_blocksize>>>(buf, rows, cols, blocksize);
  }
}

__global__ void k_add_one_packed(double* pack_buf, size_t rows, size_t cols, size_t blocksize) {
  size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  constexpr double one = double(1.0)/NITER;

  for (size_t i = tid; i < blocksize*cols; i+=stride) {
    pack_buf[i] += one;
  }
}

extern "C" {
  void add_one_packed(double* pack_buf, size_t rows, size_t cols, size_t blocksize) {
    size_t send_count = blocksize*cols;
    size_t gpu_gridsize = (send_count + gpu_blocksize - 1)/gpu_blocksize;
    k_add_one_packed<<<gpu_gridsize, gpu_blocksize>>>(pack_buf, rows,  cols, blocksize);
  }
}

__global__ void k_fill(double* buf, double value, size_t rows, size_t cols) {
  size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < rows*cols; i += stride) {
    buf[i] = value;
  }
}

extern "C" {
  void fill(double* buf, double value, size_t rows, size_t cols) {
    size_t gpu_gridsize = (cols*rows+gpu_blocksize -1) / gpu_blocksize;
    k_fill<<<gpu_gridsize, gpu_blocksize>>>(buf, value, rows, cols);
  }
}
/*
   This routine can be used to inspect the properties of a node
   Return arguments:

   nodeRank (int *)  -- My rank in the node communicator
   nodeProcs (int *) -- Total number of processes in this node
   devCount (int *)  -- Number of HIP devices available in the node
*/
void getNodeInfo(int *nodeRank, int *nodeProcs, int *devCount)
{
    MPI_Comm intranodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &intranodecomm);

    MPI_Comm_rank(intranodecomm, nodeRank);
    MPI_Comm_size(intranodecomm, nodeProcs);

    MPI_Comm_free(&intranodecomm);
    hipGetDeviceCount(devCount);
}

double pingpong_gpu_manual(int rank, double *databuf, 
                           size_t N, size_t rows, size_t cols, size_t blocksize, double& checksum) {
  double *gpu_buffer;
  size_t send_count = blocksize*cols;
  constexpr size_t gpu_blocksize = 512;
  size_t gpu_gridsize = (send_count + gpu_blocksize - 1)/gpu_blocksize;
  double t1, t2;

  hipMalloc((void**) &gpu_buffer, sizeof(double)*send_count);

  /* fill<<<(rows*cols+gpu_blocksize-1)/gpu_blocksize, gpu_blocksize>>>(databuf, double(rank), rows, cols); */
  fill(databuf, double(rank), rows, cols);
  hipStreamSynchronize(0);

  // Warmup
  /* gpu_packbuf<<<gpu_gridsize, gpu_blocksize>>>(databuf, gpu_buffer, rows, cols, blocksize); */
  gpu_packbuf(databuf, gpu_buffer, rows, cols, blocksize);
  hipStreamSynchronize(0);
  if(rank == 0) {
    MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* add_one_packed<<<gpu_gridsize, gpu_blocksize>>>(gpu_buffer, rows, cols, blocksize); */
    add_one_packed(gpu_buffer, rows, cols, blocksize);
    hipStreamSynchronize(0);
    MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  /* gpu_unpackbuf<<<gpu_gridsize, gpu_blocksize>>>(databuf, gpu_buffer, rows, cols, blocksize); */
  gpu_unpackbuf(databuf, gpu_buffer, rows, cols, blocksize);
  
  /* fill<<<(rows*cols+gpu_blocksize-1)/gpu_blocksize, gpu_blocksize>>>(databuf, double(rank), rows, cols); */
  fill(databuf, double(rank), rows, cols);
  hipStreamSynchronize(0);

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (int iter = 0; iter < NITER; ++iter) {
    /* gpu_packbuf<<<gpu_gridsize, gpu_blocksize>>>(databuf, gpu_buffer, rows, cols, blocksize); */
    gpu_packbuf(databuf, gpu_buffer, rows, cols, blocksize);
    hipStreamSynchronize(0);
    if(rank == 0) {
      MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* add_one_packed<<<gpu_gridsize, gpu_blocksize>>>(gpu_buffer, rows, cols, blocksize); */
      add_one_packed(gpu_buffer, rows, cols, blocksize);
      hipStreamSynchronize(0);
      MPI_Send(gpu_buffer, send_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
    /* gpu_unpackbuf<<<gpu_gridsize, gpu_blocksize>>>(databuf, gpu_buffer, rows, cols, blocksize); */
    gpu_unpackbuf(databuf, gpu_buffer, rows, cols, blocksize);
    hipStreamSynchronize(0);
  }
  t2 = MPI_Wtime();

  if (rank==0) {
    checksum = 0.0;
    double* h_databuf;

    HIP_ERRCHK(hipHostMalloc((void**)&h_databuf, sizeof(double)*rows*cols));
    HIP_ERRCHK(hipMemcpy(h_databuf, databuf, sizeof(double)*rows*cols, hipMemcpyDefault));
    for (size_t i = 0; i < rows*cols; ++i) {
      checksum += h_databuf[i];
    }
    HIP_ERRCHK(hipHostFree(h_databuf));
    /* thrust::device_ptr<double> t_container = thrust::device_pointer_cast(databuf); */
    /* checksum = thrust::reduce(t_container, t_container + rows*cols, double(0.0), thrust::plus<double>()); */
  }


  hipFree(gpu_buffer);
  return (t2-t1)/NITER;
}


// This should work with gpu and cpu right out of the box
double pingpong_derived(int rank, double *databuf, size_t N, size_t rows, size_t cols, size_t blocksize, double& checksum) { 
  // N should equal rows*cols
  MPI_Datatype rowtype;
  constexpr size_t gpu_blocksize = 512;
  double t1, t2;
  MPI_Type_vector(cols, blocksize, rows, MPI_DOUBLE, &rowtype);
  MPI_Type_commit(&rowtype);

  /* fill<<<(rows*cols+gpu_blocksize-1)/gpu_blocksize, gpu_blocksize>>>(databuf, double(rank), rows, cols); */
  fill(databuf, double(rank), rows, cols);
  hipStreamSynchronize(0);

  // warmup
  if (rank == 0) {
    MPI_Send(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Recv(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* add_one_unpacked<<<(blocksize*cols + gpu_blocksize -1)/gpu_blocksize, gpu_blocksize>>>(databuf, rows, cols, blocksize); */
    add_one_unpacked(databuf, rows, cols, blocksize);
    hipStreamSynchronize(0);
    MPI_Send(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD);
  }

  /* fill<<<(rows*cols+gpu_blocksize-1)/gpu_blocksize, gpu_blocksize>>>(databuf, double(rank), rows, cols); */
  fill(databuf, double(rank), rows, cols);
  hipStreamSynchronize(0);
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter<NITER; ++iter) {
    if (rank == 0) {
      MPI_Send(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(databuf, 1, rowtype, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      /* add_one_unpacked<<<(blocksize*cols + gpu_blocksize -1)/gpu_blocksize, gpu_blocksize>>>(databuf, rows, cols, blocksize); */
      add_one_unpacked(databuf, rows, cols, blocksize);
      hipStreamSynchronize(0);
      MPI_Send(databuf, 1, rowtype, 0, 0, MPI_COMM_WORLD);
    }
  }

  t2 = MPI_Wtime();

  if (rank==0) {
    checksum = 0.0;
    double* h_databuf;

    HIP_ERRCHK(hipHostMalloc((void**)&h_databuf, sizeof(double)*rows*cols));
    HIP_ERRCHK(hipMemcpy(h_databuf, databuf, sizeof(double)*rows*cols, hipMemcpyDefault));
    for (size_t i = 0; i < rows*cols; ++i) {
      checksum += h_databuf[i];
    }
    HIP_ERRCHK(hipHostFree(h_databuf));
  }

  MPI_Type_free(&rowtype);
  return (t2-t1)/NITER;
}

#define N_data_a 16 
#define N_data_b 15 

typedef struct {
   uint8_t data_a[N_data_a];
   uint8_t data_b[N_data_b];
} data_t;

double pingpong_struct_mpi_byte(int rank, size_t N) {
  data_t* data;
  data_t* data_d;
  data = (data_t*) malloc(N*sizeof(data_t));
  
  // fill on cpu
  for (size_t i = 0; i < N; ++i) {
    for (size_t a = 0; a < N_data_a; ++a) data[i].data_a[a] = (uint8_t) 'a';
    for (size_t b = 0; b < N_data_b; ++b) data[i].data_a[b] = (uint8_t) 'b';
  }
  const int blen[2] = {N_data_a, N_data_b};

  const MPI_Aint array_of_displacements[2] = {0, (ptrdiff_t)((char*)&data[0].data_b[0] - (char*)&data[0])};
  const MPI_Datatype oldtypes[2] = {MPI_BYTE, MPI_BYTE};
  MPI_Datatype m_bytestruct;

  MPI_Type_create_struct(2, blen, array_of_displacements, oldtypes, &m_bytestruct);
  MPI_Type_commit(&m_bytestruct);
  
  HIP_ERRCHK(hipMalloc((void**)&data_d, sizeof(data_t)*N));
  HIP_ERRCHK(hipMemcpy((void*)data_d, (void*)data, sizeof(data_t)*N, hipMemcpyDefault));

/* int MPI_Isend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag, */
              /* MPI_Comm comm, MPI_Request *req) { */
  MPI_Request req;
  MPI_Status stat;

  double t1, t2;
  if(rank==0) {
    MPI_Isend((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);

    MPI_Irecv((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
  }

  if(rank==1) {
    MPI_Irecv((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
    MPI_Isend((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &stat);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if(rank==0) {
      MPI_Isend((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);

      MPI_Irecv((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
    }

    if(rank==1) {
      MPI_Irecv((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);

      MPI_Isend((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, &stat);
    }
  }
  t2 = MPI_Wtime();
  MPI_Type_free(&m_bytestruct);
  HIP_ERRCHK(hipFree(data_d));
  free(data);
  return (t2-t1)/NITER;

}

double pingpong_struct_mpi_byte_synchr(int rank, size_t N) {
  data_t* data;
  data_t* data_d;
  data = (data_t*) malloc(N*sizeof(data_t));
  
  // fill on cpu
  for (size_t i = 0; i < N; ++i) {
    for (size_t a = 0; a < N_data_a; ++a) data[i].data_a[a] = (uint8_t) 'a';
    for (size_t b = 0; b < N_data_b; ++b) data[i].data_a[b] = (uint8_t) 'b';
  }
  const int blen[2] = {N_data_a, N_data_b};

  const MPI_Aint array_of_displacements[2] = {0, (ptrdiff_t)((char*)&data[0].data_b[0] - (char*)&data[0])};
  const MPI_Datatype oldtypes[2] = {MPI_BYTE, MPI_BYTE};
  MPI_Datatype m_bytestruct;

  MPI_Type_create_struct(2, blen, array_of_displacements, oldtypes, &m_bytestruct);
  MPI_Type_commit(&m_bytestruct);
  
  HIP_ERRCHK(hipMalloc((void**)&data_d, sizeof(data_t)*N));
  HIP_ERRCHK(hipMemcpy((void*)data_d, (void*)data, sizeof(data_t)*N, hipMemcpyDefault));

/* int MPI_Isend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag, */
              /* MPI_Comm comm, MPI_Request *req) { */

  double t1, t2;
  if(rank==0) {
    MPI_Send((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD);

    MPI_Recv((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if(rank==1) {
    MPI_Recv((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  for (size_t iter = 0; iter < NITER; ++iter) {
    if(rank==0) {
      MPI_Send((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD);
      MPI_Recv((void*)data_d, N, m_bytestruct, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if(rank==1) {
      MPI_Recv((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      MPI_Send((void*)data_d, N, m_bytestruct, 0, 0, MPI_COMM_WORLD);
    }
  }
  t2 = MPI_Wtime();
  MPI_Type_free(&m_bytestruct);
  HIP_ERRCHK(hipFree(data_d));
  free(data);
  return (t2-t1)/NITER;

}

int main(int argc, char *argv[]) {

    int rank, size;
    double *matrix;
    constexpr size_t NCOLS = 1<<12;
    constexpr size_t NROWS = 1<<12;
    constexpr size_t gpu_blocksize = 1<<9;
    MPI_Datatype rowtype;
    double t1, t2, dt_derived, dt_manual, dt_struct_async, dt_struct_sync;
    double checksum_derived, checksum_manual;

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
    HIP_ERRCHK(hipGetDeviceCount(&devcount));
    printf("There are %i devices for rank %i\n", devcount, rank);
    if (devcount == 2) {
      HIP_ERRCHK(hipSetDevice(rank));
    }
    fflush(stdout);


    hipMalloc((void**) &matrix, sizeof(double)*NCOLS*NROWS);

    if(rank==0) printf("N, time (ms)");

    // TODO: Do something to the data in rank 1 and test that result is correct
    for (size_t N = 1<<20; N< 1<<25; N=N<<1) {

      /* dt_derived = pingpong_derived(rank, matrix, NROWS*NCOLS, NROWS, NCOLS, BLOCKSIZE, checksum_derived); */
      /* dt_manual = pingpong_gpu_manual(rank, matrix, NROWS*NCOLS, NROWS, NCOLS, BLOCKSIZE, checksum_manual); */
      dt_struct_async = pingpong_struct_mpi_byte(rank, N);
      dt_struct_sync = pingpong_struct_mpi_byte_synchr(rank, N);


      /* size_t msg_size = sizeof(double)*NCOLS*(BLOCKSIZE); */
      if (rank == 0) {
      printf("%zu, %g, %g\n", N, dt_struct_async*1000, dt_struct_sync*1000);
      }
    }
    hipFree(matrix);
    MPI_Finalize();
    return 0;
}


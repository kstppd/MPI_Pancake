#if 0 
hipcc -O3 -DKOMPRESS -std=c++17 -Wno-unused-result -fPIC -shared -x hip mpi_pancake.cpp -ffast-math -march=native -fno-exceptions -I./hipCOMP-core/build/include/ -L./hipCOMP-core/build/lib/  -Wl,-rpath,./hipCOMP-core/build/lib/  -o libmpipancake.so
exit
#endif
// clang-format off
/*
mpi_hooks.cpp:
  LD_PRELOAD-able library that hooks Isend/Irecv and
  does gpu packing/unpacking. Assumes GPU aware MPI
  implementation. Developed for Vlasiator so a lot
  more things can be added to support more codes.
Author: Kostis Papadakis(kpapadakis@protonmail.com) 2025

Literature:
  [GPU custom packing](https://carlpearson.net/post/20201006-mpi-pack/)
  [Poisson Issues in 2D due to strided comms](https://blogs.fau.de/adityauj/2025/02/27/cuda-aware-mpi-part-4-optimizing-strided-data-communication-between-gpus/)
  [Poisson as above](https://developer.nvidia.com/blog/benchmarking-cuda-aware-mpi/)
  [MPI_ERR_TRUNCATE](https://stackoverflow.com/questions/37566517/mpi-err-truncate-message-truncated)
  [Paper that helped with kernels](https://sdm.lbl.gov/sdav/images/publications/Jen2012b/Jenkins_Cluster_MPI-GPU_12.pdf)
  [Very Good MPI Docs apart from manpages](https://rookiehpc.org/mpi/docs/mpi_datatype/index.html)

TODOs:
  [ ] Thread Waitall and lock map
  [ ] Compress dpack with some nvCOMP hipCOMP
  [X] Clean up memory after MPI_Finalize. We techincally leak everything but Vlasiator dies after so...

Status for Vlasiator comms:
  [-] `SUBARRAY CONTIGUOUS MPI_BYTE` comes from FsGrid so my hooks do not handle it for now
  [-] `MPI_BYTE` is also not yet handled although trivial will probably add support now (low usage)
  [-] `MPI_INT` not handled (KB)
  [+] `STRUCT MPI_BYTE` is coming from spatial cells (dccrg) and is intercepted and gpu packed
  [+] `STRUCT HINDEXED MPI_BYTE` is also coming from spatial cells (dccrg) and is intercepted and gpu packed
  [+] `STRUCT MIXED` is also coming from spatial cells (dccrg) and is intercepted and gpu packed. Mixed in Vlasiator
                     is a combo of HINDEXED+MPI_NAMED
                     
   This is what a smallish 3D run offloads to the network for 2 timesteps + initialization
    --- MPI Sniffer ---
      [MPI_Send   CALLS]
        MPI_INT                            -> 158088     calls, 617.53 KB
        ---------------------------------- Total: 617.53 KB
      [MPI_Isend  CALLS]
        STRUCT MPI_BYTE                    -> 32824004   calls, 0 B
        MPI_BYTE                           -> 385568     calls, 3.37 GB
        STRUCT HINDEXED MPI_BYTE           -> 2209231    calls, 793.11 GB
        SUBARRAY CONTIGUOUS MPI_BYTE       -> 1517952    calls, 374.29 GB
        STRUCT MIXED                       -> 2226741    calls, 192.43 GB
        ---------------------------------- Total: 1.33 TB
      [MPI_Recv   CALLS]
        (None)
      [MPI_Irecv  CALLS]
        MPI_INT                            -> 158088     calls, 617.53 KB
        MPI_BYTE                           -> 1320332    calls, 4.90 GB
        STRUCT MPI_BYTE                    -> 32824004   calls, 0 B
        STRUCT HINDEXED MPI_BYTE           -> 2209231    calls, 793.11 GB
        SUBARRAY CONTIGUOUS MPI_BYTE       -> 1517952    calls, 374.29 GB
        STRUCT MIXED                       -> 2226741    calls, 192.43 GB
        ---------------------------------- Total: 1.33 TB
    ````quote
    Scaling more 
    ```
    ==> logfile_100_base.txt <==
    (CELLS) tstep = 20 time = 7.49371 spatial cells [ 94452 37336 99392 ] blocks proton [ 40919164 16796718 37986670 ]
    (MAIN): All timesteps calculated.
             (TIME) total run time 446.759 s, total simulated time 7.49371 s
             (TIME) total 3.72299 node-hours
             (TIME) total 208.488 thread-hours
             (TIME) seconds per timestep 22.338, seconds per simulated second 59.6179
    Sat Sep 20 15:14:24 2025
    (MAIN): Completed requested simulation. Exiting.


    ==> logfile_100_hooks.txt <==
    (CELLS) tstep = 20 time = 7.49371 spatial cells [ 94452 37336 99392 ] blocks proton [ 40919168 16796703 37986642 ]
    (MAIN): All timesteps calculated.
             (TIME) total run time 218.188 s, total simulated time 7.49371 s
             (TIME) total 1.81823 node-hours
             (TIME) total 101.821 thread-hours
             (TIME) seconds per timestep 10.9094, seconds per simulated second 29.1161
    Sat Sep 20 15:21:34 2025
    (MAIN): Completed requested simulation. Exiting.
    ```
    ````
    For fun I ran the cpu branch on the same allocation. So same MPI config and it takes
    ```
    (MAIN): All timesteps calculated.
             (TIME) total run time 157.059 s, total simulated time 7.49371 s
             (TIME) total 1.30883 node-hours
             (TIME) total 73.2944 thread-hours
             (TIME) seconds per timestep 7.85297, seconds per simulated second 20.9588
    Sun Sep 21 21:20:33 2025
    (MAIN): Completed requested simulation. Exiting.
    ```
*/

#include <cstdio>
#include <dlfcn.h>
#include <unordered_map>
#include "mpi.h"

//Currenlty only HIP and based on https://github.com/ROCm/hipCOMP-core/blob/main/tests/test_lz4.cpp 
#ifdef KOMPRESS
#include "hipcomp.hpp"
#include "hipcomp/lz4.hpp"
using namespace hipcomp;
#endif


#if defined(USE_HIP) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuMalloc(ptr, size) hipMalloc(ptr, size)
#define gpuFree(ptr) hipFree(ptr)
#define gpuMemcpy(dst, src, sz, k) hipMemcpy(dst, src, sz, k)
#define gpuMemcpyAsync(dst, src, sz, k, st) hipMemcpyAsync(dst, src, sz, k, st)
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuPointerAttributes hipPointerAttribute_t
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#else
#include <cuda_runtime.h>
#define gpuMalloc(ptr, size) cudaMalloc(ptr, size)
#define gpuFree(ptr) cudaFree(ptr)
#define gpuMemcpy(dst, src, sz, k) cudaMemcpy(dst, src, sz, k)
#define gpuMemcpyAsync(dst, src, sz, k, st) cudaMemcpyAsync(dst, src, sz, k, st)
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuPointerAttributes cudaPointerAttributes
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#endif

// #define VERBOSE
#ifndef VERBOSE
#define LOG(...)                                                               \
  do {                                                                         \
  } while (0)
#else
#define LOG(fmt, ...)                                                          \
  do {                                                                         \
    fprintf(stderr, "[VERBOSE] %s:%d: " fmt "\n", __FILE__, __LINE__,          \
            ##__VA_ARGS__);                                                    \
  } while (0)
#endif

#define FATAL(fmt, ...)                                                        \
  do {                                                                         \
    fprintf(stderr, "[FATAL] %s:%d: " fmt "\n", __FILE__, __LINE__,          \
            ##__VA_ARGS__);                                                    \
    MPI_Abort(MPI_COMM_WORLD, 42);                                             \
  } while (0)


// Stolen from AST_Picasso@Graffathon 2025
struct BumpAllocator {
  void *mem = nullptr;
  std::size_t sp = 0;
  std::size_t cap = 0;
  BumpAllocator(void *buf, std::size_t bytes) : mem(buf), sp(0), cap(bytes) {}
  BumpAllocator(const BumpAllocator &) = delete;
  BumpAllocator &operator=(const BumpAllocator &) = delete;
  template <class T> T *allocate(std::size_t n, int align_force = -1) {
    if (n == 0) {
      return nullptr;
    }
    std::size_t need = n * sizeof(T);
    std::size_t al = (align_force > 0) ? (std::size_t)align_force
                                  : std::max<std::size_t>(alignof(T), 8);
    std::size_t base = (std::size_t)((char *)mem + sp);
    std::size_t pad = (base % al == 0) ? 0 : (al - (base % al));
    if (sp + pad + need > cap) {
      FATAL("MPI_Pancake::OOM");
      return nullptr;
    }
    void *p = (char *)mem + sp + pad;
    sp += pad + need;
    return (T *)p;
  }
  //realloc esssentially?
  template <class T> void unsafe_extend_allocation(std::size_t extraElems) {
    std::size_t extra = extraElems * sizeof(T);
    if (sp + extra > cap) {
      FATAL("F around Find out");
    };
    sp += extra;
  }
  void release() { sp = 0; }
};

#ifdef KOMPRESS
//Kompression stuff
inline std::size_t estimate_bytes_z(std::size_t input_bytes,
                               hipStream_t stream,
                               std::size_t chunk_size = (1u << 16)){
  LZ4Manager manager{chunk_size, HIPCOMP_TYPE_CHAR, stream};
  return manager.configure_compression(input_bytes).max_compressed_buffer_size;
}

inline std::size_t estimate_bytes_d(const uint8_t* d_comp,
                               hipStream_t stream,
                               std::size_t chunk_size = (1u << 16)){
  LZ4Manager manager{chunk_size, HIPCOMP_TYPE_CHAR, stream};
  return manager.configure_decompression(const_cast<uint8_t*>(d_comp)).decomp_data_size;
}

inline std::size_t compress_into(const uint8_t* d_in,
                                 uint8_t* d_comp,
                                 std::size_t input_bytes,
                                 hipStream_t stream,
                                 std::size_t chunk_size = (1u << 16)){
  LZ4Manager manager{chunk_size, HIPCOMP_TYPE_CHAR, stream};
  auto cfg = manager.configure_compression(input_bytes);
  manager.compress(d_in, d_comp, cfg);
  hipStreamSynchronize(stream);
  return manager.get_compressed_output_size(d_comp);
}

inline std::size_t decompress_into(const uint8_t* d_comp,
                                   uint8_t* d_out,
                                   hipStream_t stream,
                                   std::size_t chunk_size = (1u << 16)){
  LZ4Manager manager{chunk_size, HIPCOMP_TYPE_CHAR, stream};
  auto dcfg = manager.configure_decompression(const_cast<uint8_t*>(d_comp));
  manager.decompress(d_out, const_cast<uint8_t*>(d_comp), dcfg);
  hipStreamSynchronize(stream);
  return dcfg.decomp_data_size;
}

#endif //KOMPRESS

struct SOABlock {
  MPI_Aint disp;
  int len;
  std::size_t pack_off;
};

struct Pending {
  enum Op { SEND, RECV } op;
  MPI_Request rreq{};
  SOABlock *blocks = nullptr;
  std::size_t nblocks = 0;
  MPI_Aint extent = 0;
  std::size_t total_bytes = 0; 
  std::size_t pack_size = 0;  
  MPI_Aint *h_disp = nullptr;
  int *h_len = nullptr;
  std::size_t *h_pref = nullptr;
  int64_t *d_disp = nullptr; 
  int *d_len = nullptr;
  std::size_t *d_pref = nullptr;
  char *pack_buffer = nullptr;   
  char *stage = nullptr;
  char *d_pack_buffer = nullptr;
  void *user_buf = nullptr;
  int count = 0;
  MPI_Comm comm{};
#ifdef KOMPRESS
  std::size_t comp_bytes = 0;
  char *d_comp_pack_buffer = nullptr;
#endif
  struct Header {
    int nseg;
    std::size_t total_bytes;
    std::size_t extent;
  } header;
};

static int (*rMPI_Init)         (int *, char ***)                                                     = nullptr;
static int (*rMPI_Init_thread)  (int *, char ***, int, int *)                                         = nullptr;
static int (*rMPI_Isend)        (const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *)  = nullptr;
static int (*rMPI_Irecv)        (void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *)        = nullptr;
static int (*rMPI_Type_get_env) (MPI_Datatype, int *, int *, int *, int *)                            = nullptr; 
static int (*rMPI_Type_size)    (MPI_Datatype, int *)                                                 = nullptr;
static int (*rMPI_Wait)         (MPI_Request *, MPI_Status *)                                         = nullptr;
static int (*rMPI_Waitall)      (int, MPI_Request[], MPI_Status[])                                    = nullptr;
static int (*rMPI_Finalize)     (void)                                                                = nullptr;

// SETTINGS
static constexpr std::size_t POOL = 8ull * 1024ull * 1024ull * 1024ull;
static constexpr std::size_t INIT_BLOCKS = 8 * 512;
static BumpAllocator *host_arena = nullptr;
static BumpAllocator *dev_arena = nullptr;
static bool initialized = false;
static gpuStream_t s;
static std::unordered_map<MPI_Request, Pending *> pending;



__global__ void pack_kernel(const char *__restrict__ src,
                            const int64_t *__restrict__ disp,
                            const int *__restrict__ len,
                            const std::size_t *__restrict__ pref, int nseg,
                            std::size_t extent, std::size_t total_bytes,
                            int count, char *__restrict__ dst) {
  const auto s_nseg = nseg;
  const auto s_extent = extent;
  const auto s_total = total_bytes;
  const int cell = blockIdx.y;
  if (cell >= count) {
    return;
  }
  const char *base = src + (std::size_t)cell * s_extent;
  char *out = dst + (std::size_t)cell * s_total;
  for (int seg = blockIdx.x; seg < s_nseg; seg += gridDim.x) {
    const int64_t d = disp[seg];
    const int L = len[seg];
    const std::size_t P = pref[seg];
    const char *sb = base + (std::size_t)d;
    char *db = out + P;
    std::size_t head = 0;
    for (std::size_t v = head + (std::size_t)threadIdx.x; v < (std::size_t)L;
         v += (std::size_t)blockDim.x) {
      db[v] = sb[v];
    }
  }
}

__global__ void unpack_kernel(const char *__restrict__ src,
                              const int64_t *__restrict__ disp,
                              const int *__restrict__ len,
                              const std::size_t *__restrict__ pref, int nseg,
                              std::size_t extent, std::size_t total_bytes,
                              int count, char *__restrict__ dst) {
  const auto s_nseg = nseg;
  const auto s_extent = extent;
  const auto s_total = total_bytes;
  const int cell = blockIdx.y;
  if (cell >= count) {
    return;
  }
  const char *in = src + (std::size_t)cell * s_total;
  char *obj = dst + (std::size_t)cell * s_extent;
  for (int seg = blockIdx.x; seg < s_nseg; seg += gridDim.x) {
    const int64_t d = disp[seg];
    const int L = len[seg];
    const std::size_t P = pref[seg];
    const char *sb = in + P;
    char *db = obj + (std::size_t)d;
    std::size_t head = 0;

    for (std::size_t v = head + (std::size_t)threadIdx.x; v < (std::size_t)L;
         v += (std::size_t)blockDim.x) {
      db[v] = sb[v];
    }
  }
}



void do_pack(const char *user, int count, Pending *p, gpuStream_t s) {
  std::size_t avg = (p->nblocks ? p->total_bytes / p->nblocks : p->total_bytes);
  int tpb = (avg >= 4096 ? 256 : (avg >= 1024 ? 128 : 64));
  dim3 block(tpb);
  dim3 grid(std::min<int>((int)p->nblocks, 65535), count);
  pack_kernel<<<grid, block, 0, s>>>(
      user, p->d_disp, p->d_len, p->d_pref, p->header.nseg,
      p->header.extent, p->header.total_bytes, count, p->d_pack_buffer);
}

void do_unpack(char *user, int count, Pending *p, gpuStream_t s) {
  std::size_t avg = (p->nblocks ? p->total_bytes / p->nblocks : p->total_bytes);
  int tpb = (avg >= 4096 ? 256 : (avg >= 1024 ? 128 : 64));
  dim3 block(tpb);
  dim3 grid(std::min<int>((int)p->nblocks, 65535), count);
  unpack_kernel<<<grid, block, 0, s>>>(
      (const char *)p->d_pack_buffer, p->d_disp, p->d_len, p->d_pref,
      p->header.nseg, p->header.extent, p->header.total_bytes, count, user);
}



static void init() {
  if (initialized) {
    return;
  }
  rMPI_Init         =  (decltype(rMPI_Init))         dlsym(RTLD_NEXT, "MPI_Init");
  rMPI_Init_thread  =  (decltype(rMPI_Init_thread))  dlsym(RTLD_NEXT, "MPI_Init_thread");
  rMPI_Isend        =  (decltype(rMPI_Isend))        dlsym(RTLD_NEXT, "MPI_Isend");
  rMPI_Irecv        =  (decltype(rMPI_Irecv))        dlsym(RTLD_NEXT, "MPI_Irecv");
  rMPI_Type_get_env =  (decltype(rMPI_Type_get_env)) dlsym(RTLD_NEXT, "MPI_Type_get_envelope");
  rMPI_Type_size    =  (decltype(rMPI_Type_size))    dlsym(RTLD_NEXT, "MPI_Type_size");
  rMPI_Wait         =  (decltype(rMPI_Wait))         dlsym(RTLD_NEXT, "MPI_Wait");
  rMPI_Waitall      =  (decltype(rMPI_Waitall))      dlsym(RTLD_NEXT, "MPI_Waitall");
  rMPI_Finalize     =  (decltype(rMPI_Finalize))     dlsym(RTLD_NEXT, "MPI_Finalize");

  void *h = malloc(POOL);
  void *d = nullptr;
  gpuMalloc(&d, POOL);
  pending.reserve(1<<12);
  if (!h) {
    FATAL("ERROR:host pool alloc failed\n");
  }
  if (!d) {
    FATAL("ERROR:device pool alloc failed\n");
  }

  const bool are_all_hooks_ok = rMPI_Init &&
                                rMPI_Init_thread &&
                                rMPI_Isend &&
                                rMPI_Irecv &&
                                rMPI_Type_get_env &&
                                rMPI_Type_size
                                && rMPI_Wait &&
                                rMPI_Waitall &&
                                rMPI_Finalize;

  if (!are_all_hooks_ok){
    fprintf(stderr, "ERROR:Some hook could not be dlopened!\n");
    MPI_Abort(MPI_COMM_WORLD, 32);
  }
  host_arena = new BumpAllocator(h, POOL);
  dev_arena = new BumpAllocator(d, POOL);
  gpuStreamCreate(&s);
  initialized = true;
  return;
}
// clang-format on

// TODO Will look into this at some point!
static inline bool is_device_ptr(const void *p) { return true; }

static int get_combiner(MPI_Datatype t) {
  int ni = 0, na = 0, nt = 0, comb = 0;
  rMPI_Type_get_env(t, &ni, &na, &nt, &comb);
  return comb;
}

// This is what I ended up doing becasue at times we get MPI_DUP
//  which underneath is the usual STRUCT->HINDEXED->MPI_BYTE
static MPI_Datatype unwrap_datatype(MPI_Datatype t) {
  int ni = 0, na = 0, nt = 0, comb = 0;
  if (rMPI_Type_get_env(t, &ni, &na, &nt, &comb) != MPI_SUCCESS) {
    FATAL("ERROR: Could not unwrap datatype!");
  }
  if (comb == MPI_COMBINER_DUP) {
    auto *ints = host_arena->allocate<int>(ni);
    auto *addrs = host_arena->allocate<MPI_Aint>(na);
    auto *types = host_arena->allocate<MPI_Datatype>(nt);
    if (MPI_Type_get_contents(t, ni, na, nt, ints, addrs, types) !=
        MPI_SUCCESS) {
      FATAL("ERROR: Could not get MPI contents!");
    }
    return unwrap_datatype(types[0]);
  }
  return t;
}

/*
There is a big assumption made here. What comes in
 is guaranteed to be an MPI_STRUCT that has children (dynamically sized)
 of type MPI_HINDEXED OR MPI_BYTE/NAMED.
*/
static std::size_t flatten_blocks(MPI_Datatype dtype, SOABlock **out,
                                  MPI_Aint &extent, std::size_t &total_bytes) {
  std::size_t nb = 0;
  *out = host_arena->allocate<SOABlock>(INIT_BLOCKS);
  dtype = unwrap_datatype(dtype);
  MPI_Aint lb = 0;
  if (MPI_Type_get_extent(dtype, &lb, &extent) != MPI_SUCCESS) {
    FATAL("ERROR: Could not get MPI extent!");
  }

  int ni = 0, na = 0, nt = 0, comb = 0;
  if (rMPI_Type_get_env(dtype, &ni, &na, &nt, &comb) != MPI_SUCCESS) {
    FATAL("ERROR: Could not get MPI envelope!");
  }
  auto *ints = host_arena->allocate<int>(ni);
  auto *addrs = host_arena->allocate<MPI_Aint>(na);
  auto *types = host_arena->allocate<MPI_Datatype>(nt);
  if (MPI_Type_get_contents(dtype, ni, na, nt, ints, addrs, types) !=
      MPI_SUCCESS) {
    FATAL("ERROR: Could not get MPI content!");
  }

  const int nchildren = ints[0];
  const int *struct_blks = &ints[1];

  total_bytes = 0;

  for (int c = 0; c < nchildren; c++) {
    MPI_Datatype ctype = unwrap_datatype(types[c]);
    int si = 0, sa = 0, st = 0, sc = 0;
    if (rMPI_Type_get_env(ctype, &si, &sa, &st, &sc) != MPI_SUCCESS) {
      FATAL("ERROR: Could not get MPI envelope!");
    }

    if (sc == MPI_COMBINER_HINDEXED) {
      // This works for MPI_BYTE only
      auto *sints = host_arena->allocate<int>(si);
      auto *saddrs = host_arena->allocate<MPI_Aint>(sa);
      auto *stypes = host_arena->allocate<MPI_Datatype>(st);
      if (MPI_Type_get_contents(ctype, si, sa, st, sints, saddrs, stypes) !=
          MPI_SUCCESS) {
        FATAL("ERROR: Could not get MPI contents!");
      }

      const int subcount = sints[0];
      const int *lens = &sints[1];

      for (int i = 0; i < subcount; i++) {
        SOABlock b{addrs[c] + saddrs[i], lens[i], total_bytes};
        total_bytes += (std::size_t)lens[i];
        (*out)[nb++] = b;
      }

    } else if (sc == MPI_COMBINER_NAMED) {
      // This works for NAMED stuff only
      int child_sz = 0;
      if (rMPI_Type_size(ctype, &child_sz) != MPI_SUCCESS) {
        FATAL("ERROR: Could not get MPI type size!");
      }
      const int bl = struct_blks[c];
      if (bl > 0 && child_sz > 0) {
        const std::size_t L = (std::size_t)bl * (std::size_t)child_sz;
        SOABlock b{addrs[c], (int)L, total_bytes};
        total_bytes += L;
        (*out)[nb++] = b;
      }
    } else {
      FATAL("We should have never ended up here in Vlasiator! That means we "
            "got a "
            "type in here which is uknonwn!");
    }
  }

  if (nb > INIT_BLOCKS) {
    host_arena->unsafe_extend_allocation<SOABlock>(nb - INIT_BLOCKS);
  }
  return nb;
}

static void build_lookaside(Pending *p) {
  const std::size_t n = p->nblocks;
  p->h_disp = host_arena->allocate<MPI_Aint>(n, 16);
  p->h_len = host_arena->allocate<int>(n, 16);
  p->h_pref = host_arena->allocate<std::size_t>(n, 16);

  for (std::size_t i = 0; i < n; i++) {
    p->h_disp[i] = p->blocks[i].disp;
    p->h_len[i] = p->blocks[i].len;
    p->h_pref[i] = p->blocks[i].pack_off;
  }

  // device copies
  p->d_disp = dev_arena->allocate<int64_t>(n, 16);
  p->d_len = dev_arena->allocate<int>(n, 16);
  p->d_pref = dev_arena->allocate<std::size_t>(n, 16);
  int64_t *tmp = host_arena->allocate<int64_t>(n);
  for (std::size_t i = 0; i < n; i++)
    tmp[i] = (int64_t)p->h_disp[i];

  gpuMemcpyAsync(p->d_disp, tmp, n * sizeof(int64_t), gpuMemcpyHostToDevice, s);
  gpuMemcpyAsync(p->d_len, p->h_len, n * sizeof(int), gpuMemcpyHostToDevice, s);
  gpuMemcpyAsync(p->d_pref, p->h_pref, n * sizeof(std::size_t),
                 gpuMemcpyHostToDevice, s);
  gpuStreamSynchronize(s);

  p->header.nseg = (int)n;
  p->header.total_bytes = p->total_bytes;
  p->header.extent = (std::size_t)p->extent;
}

static void cpu_pack(const void *user_buf, int count, Pending *p) {
  p->pack_buffer = host_arena->allocate<char>(p->pack_size, 16);
  char *dst = p->pack_buffer;
  for (int c = 0; c < count; c++) {
    const char *base =
        (const char *)user_buf + (std::size_t)c * (std::size_t)p->extent;
    for (std::size_t i = 0; i < p->nblocks; i++) {
      const SOABlock &b = p->blocks[i];
      memcpy(dst + b.pack_off, base + b.disp, (std::size_t)b.len);
    }
    dst += p->total_bytes;
  }
}

static void cpu_unpack(void *user_buf, int count, Pending *p) {
  const char *src = p->stage;
  for (int c = 0; c < count; c++) {
    char *base = (char *)user_buf + (std::size_t)c * (std::size_t)p->extent;
    for (std::size_t i = 0; i < p->nblocks; i++) {
      const SOABlock &b = p->blocks[i];
      memcpy(base + b.disp, src + b.pack_off, (std::size_t)b.len);
    }
    src += p->total_bytes;
  }
}

static void gpu_pack(const void *user_buf, int count, Pending *p) {
  p->d_pack_buffer = dev_arena->allocate<char>(p->pack_size, 256);
  do_pack((const char *)user_buf, count, p, s);
#ifdef KOMPRESS
  std::size_t cap = estimate_bytes_z(p->pack_size, s);
  auto *d_comp = dev_arena->allocate<uint8_t>(cap, 256);
  const std::size_t comp_bytes =
      compress_into(reinterpret_cast<const uint8_t *>(p->d_pack_buffer), d_comp,
                    p->pack_size, s);
  p->d_comp_pack_buffer = reinterpret_cast<char *>(d_comp);
  p->comp_bytes = comp_bytes;
  // fprintf(stderr,"Compression of %f x
  // \n",(float)p->pack_size/(float)comp_bytes);
#endif
  gpuStreamSynchronize(s);
}

static void gpu_unpack(void *user_buf, int count, Pending *p) {
  do_unpack((char *)user_buf, count, p, s);
  gpuStreamSynchronize(s);
}


static void do_complete(Pending *p, MPI_Status *st_opt) {
  (void)st_opt;
  if (p->op == Pending::RECV) {
    if (is_device_ptr(p->user_buf)) {
#ifdef KOMPRESS
      p->d_pack_buffer = dev_arena->allocate<char>(p->pack_size, 256);
      const std::size_t decomp_bytes = decompress_into(
          reinterpret_cast<const uint8_t *>(p->d_comp_pack_buffer),
          reinterpret_cast<uint8_t *>(p->d_pack_buffer), s);
      if (decomp_bytes != p->pack_size) {
        FATAL("Decompressed size (%zu) VS wanted size (%zu).", decomp_bytes,
              p->pack_size);
      }
#endif
      gpu_unpack(p->user_buf, p->count, p);
    } else {
      cpu_unpack(p->user_buf, p->count, p);
    }
  }
}

static int complete_request(MPI_Request *req, MPI_Status *st) {
  if (*req == MPI_REQUEST_NULL)
    return MPI_SUCCESS;
  Pending *p = nullptr;
  auto it = pending.find(*req);
  if (it == pending.end()) {
    // not ours — delegate to real MPI_Wait
    return rMPI_Wait(req, st);
  }
  p = it->second;
  int ret = rMPI_Wait(&p->rreq, st);
  if (ret == MPI_SUCCESS) {
    if (!(st && st->MPI_ERROR != MPI_SUCCESS)) {
      do_complete(p, st);
    }
  }
  pending.erase(it);
  *req = MPI_REQUEST_NULL;
  if (pending.empty()) {
    host_arena->release();
    dev_arena->release();
  }
  return ret;
}

extern "C" {
int MPI_Isend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag,
              MPI_Comm comm, MPI_Request *req) {
  init();
  // This will also crash the code
  if (count == 0 || dtype == MPI_DATATYPE_NULL) {
    return rMPI_Isend(buf, count, dtype, dest, tag, comm, req);
  }
  int type_sz = 0;
  rMPI_Type_size(dtype, &type_sz);

  if (get_combiner(dtype) == MPI_COMBINER_STRUCT && type_sz > 0) {
    Pending *p = ::new (host_arena->allocate<Pending>(1)) Pending{};
    p->op = Pending::SEND;
    p->nblocks = flatten_blocks(dtype, &p->blocks, p->extent, p->total_bytes);
    p->pack_size = (std::size_t)count * p->total_bytes;

    build_lookaside(p);

    int ret = MPI_SUCCESS;
    if (is_device_ptr(buf)) {
      LOG("Dev SEND");
      gpu_pack(buf, count, p); //<== look inside it kompresses
#ifdef KOMPRESS
      ret = rMPI_Isend(p->d_comp_pack_buffer, (int)p->comp_bytes, MPI_BYTE,
                       dest, tag, comm, &p->rreq);

#else
      ret = rMPI_Isend(p->d_pack_buffer, (int)p->pack_size, MPI_BYTE, dest, tag,
                       comm, &p->rreq);
#endif
    } else {
      LOG("Host SEND");
      cpu_pack(buf, count, p);
      ret = rMPI_Isend(p->pack_buffer, (int)p->pack_size, MPI_BYTE, dest, tag,
                       comm, &p->rreq);
    }
    if (ret == MPI_SUCCESS) {
      *req = p->rreq;
      pending[p->rreq] = p;
    }
    return ret;
  }
  return rMPI_Isend(buf, count, dtype, dest, tag, comm, req);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype dtype, int src, int tag,
              MPI_Comm comm, MPI_Request *req) {
  init();
  if (count == 0 || dtype == MPI_DATATYPE_NULL) {
    return rMPI_Irecv(buf, count, dtype, src, tag, comm, req);
  }

  int type_sz = 0;
  rMPI_Type_size(dtype, &type_sz);

  if (get_combiner(dtype) == MPI_COMBINER_STRUCT && type_sz > 0) {
    Pending *p = ::new (host_arena->allocate<Pending>(1)) Pending{};
    p->op = Pending::RECV;
    p->user_buf = buf;
    p->count = count;
    p->comm = comm;
    p->nblocks = flatten_blocks(dtype, &p->blocks, p->extent, p->total_bytes);
    p->pack_size = (std::size_t)count * p->total_bytes;

    build_lookaside(p);
    int ret = MPI_SUCCESS;
    if (is_device_ptr(buf)) {
      LOG("Dev RECV");
#ifdef KOMPRESS
      //Worst case scenario
      const std::size_t max_comp_size = estimate_bytes_z(p->pack_size, s);
      p->d_comp_pack_buffer = dev_arena->allocate<char>(max_comp_size, 256);
      ret = rMPI_Irecv(p->d_comp_pack_buffer, (int)max_comp_size, MPI_BYTE, src,
                       tag, comm, &p->rreq);
#else
      p->d_pack_buffer = dev_arena->allocate<char>(p->pack_size, 256);
      ret = rMPI_Irecv(p->d_pack_buffer, (int)p->pack_size, MPI_BYTE, src, tag,
                       comm, &p->rreq);
#endif
    } else {
      LOG("Host RECV");
      p->stage = host_arena->allocate<char>(p->pack_size, 16);
      ret = rMPI_Irecv(p->stage, (int)p->pack_size, MPI_BYTE, src, tag, comm,
                       &p->rreq);
    }
    if (ret == MPI_SUCCESS) {
      *req = p->rreq;
      pending[p->rreq] = p;
    }
    return ret;
  }
  return rMPI_Irecv(buf, count, dtype, src, tag, comm, req);
}

int MPI_Wait(MPI_Request *req, MPI_Status *st) {
  init();
  return complete_request(req, st);
}

// // TODO is it ok if I thread this and lock the pendig map?
// int MPI_Waitall_(int n, MPI_Request reqs[], MPI_Status stats[]) {
//   init();
//   int err = MPI_SUCCESS;
//   for (int i = 0; i < n; i++) {
//     MPI_Status *st =
//         (stats == MPI_STATUSES_IGNORE) ? MPI_STATUS_IGNORE : &stats[i];
//     int e = complete_request(&reqs[i], st);
//     if (e != MPI_SUCCESS && err == MPI_SUCCESS)
//       err = e;
//   }
//   return err;
// }
//
int MPI_Waitall(int n, MPI_Request reqs[], MPI_Status stats[]) {
  init();
  struct Comm {
    int id;
    MPI_Request r;
  };
  Comm *comms_list = host_arena->allocate<Comm>(n);
  std::size_t comms_count = 0;
  for (int i = 0; i < n; ++i) {
    auto it = pending.find(reqs[i]);
    if (it != pending.end()) {
      comms_list[comms_count] = Comm{i, reqs[i]};
      comms_count++;
    }
  }
  int err = rMPI_Waitall(n, reqs, stats);
  for (std::size_t i = 0; i < comms_count; ++i) {
    const auto &e = comms_list[i];
    auto it = pending.find(e.r);
    if (it == pending.end()) {
      continue;
    }
    Pending *p = it->second;
    if (stats && stats != MPI_STATUSES_IGNORE) {
      const MPI_Status &st = stats[e.id];
      if (st.MPI_ERROR != MPI_SUCCESS) {
        pending.erase(it);
        continue;
      }
#ifdef KOMPRESS
      if (p->op == Pending::RECV) {
        int got = 0;
        MPI_Get_count(&st, MPI_BYTE, &got);
        p->comp_bytes = (std::size_t)got;
      }
#endif
      do_complete(p, const_cast<MPI_Status *>(&st));
    } else {
      //for MPI_IGNORES which have no stats
      do_complete(p, nullptr);
    }
    pending.erase(it);
    reqs[e.id] = MPI_REQUEST_NULL;
  }
  // Release pools now
  if (pending.empty()) {
    host_arena->release();
    dev_arena->release();
  }
  return err;
}

int MPI_Init(int *argc, char ***argv) {
  init();
  return rMPI_Init(argc, argv);
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
  if (required == MPI_THREAD_MULTIPLE) {
    FATAL("These hookds do not work with MPI_THREAD_MULTIPLE!");
  }
  init();
  return rMPI_Init_thread(argc, argv, required, provided);
}

// Kill time
int MPI_Finalize(void) {
  init();
  free(host_arena->mem);
  gpuFree(dev_arena->mem);
  // gpuStreamDestroy(s);
  return rMPI_Finalize();
}

} // extern "C"

#if 0
mpicxx -O3 -std=c++20 -fPIC mpi_sniffer.cpp  --shared  -o mpi_sniffer.so
exit
#endif
/*
mpi_hooks.cpp:
LD_PRELOAD-able library that tries to look into your comms
Author: Kostis Papadakis(kpapadakis@protonmail.com) 2025
*/
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Statistics {
  size_t num_calls = 0;
  size_t num_bytes = 0;
};

static std::mutex _m;
static bool initialized = false;
static std::unordered_map<uint64_t, Statistics> send_stats;
static std::unordered_map<uint64_t, Statistics> isend_stats;
static std::unordered_map<uint64_t, Statistics> recv_stats;
static std::unordered_map<uint64_t, Statistics> irecv_stats;
static std::unordered_map<uint64_t, std::string> _sig_map;
static std::vector<std::size_t> send_size;

//Hooks
static int (*real_MPI_Send)              (const void *, int, MPI_Datatype, int, int, MPI_Comm)                                                    = nullptr;
static int (*real_MPI_Isend)             (const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *)                                     = nullptr;
static int (*real_MPI_Recv)              (void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *)                                            = nullptr;
static int (*real_MPI_Irecv)             (void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *)                                           = nullptr;
static int (*real_MPI_Type_get_envelope) (MPI_Datatype, int *, int *, int *, int *)                                                               = nullptr;
static int (*real_MPI_Type_size)         (MPI_Datatype, int *)                                                                                    = nullptr;
static int (*real_MPI_Finalize)          (void)                                                                                                   = nullptr;
static int (*real_MPI_Comm_rank)         (MPI_Comm, int *)                                                                                        = nullptr;
static int (*real_MPI_Comm_size)         (MPI_Comm, int *)                                                                                        = nullptr;
static int (*real_MPI_Gather)            (const void *, int, MPI_Datatype, void *, int, MPI_Datatype, int, MPI_Comm)                              = nullptr;
static int (*real_MPI_Gatherv)           (const void *, int, MPI_Datatype, void *, const int *, const int *, MPI_Datatype, int, MPI_Comm)         = nullptr;

static void init() {
  if (initialized) {
    return;
  }
  std::lock_guard<std::mutex> lk(_m);
  if (initialized) {
    return;
  }
  real_MPI_Send = (decltype(real_MPI_Send))dlsym(RTLD_NEXT, "MPI_Send");
  real_MPI_Isend =     (decltype(real_MPI_Isend))dlsym(RTLD_NEXT, "MPI_Isend");
  real_MPI_Recv =      (decltype(real_MPI_Recv))dlsym(RTLD_NEXT, "MPI_Recv");
  real_MPI_Irecv =     (decltype(real_MPI_Irecv))dlsym(RTLD_NEXT, "MPI_Irecv");
  real_MPI_Type_get_envelope = (decltype(real_MPI_Type_get_envelope))dlsym(RTLD_NEXT, "MPI_Type_get_envelope");
  real_MPI_Type_size = (decltype(real_MPI_Type_size))dlsym(RTLD_NEXT, "MPI_Type_size");
  real_MPI_Finalize =  (decltype(real_MPI_Finalize))dlsym(RTLD_NEXT, "MPI_Finalize");
  real_MPI_Comm_rank = (decltype(real_MPI_Comm_rank))dlsym(RTLD_NEXT, "MPI_Comm_rank");
  real_MPI_Comm_size = (decltype(real_MPI_Comm_size))dlsym(RTLD_NEXT, "MPI_Comm_size");
  real_MPI_Gather    = (decltype(real_MPI_Gather))dlsym(RTLD_NEXT, "MPI_Gather");
  real_MPI_Gatherv   = (decltype(real_MPI_Gatherv))dlsym(RTLD_NEXT, "MPI_Gatherv");
  initialized = true;
}

static std::string_view classify_type(int combiner) {
  switch (combiner) {
  case MPI_COMBINER_NAMED:
    return "NAMED";
  case MPI_COMBINER_DUP:
    return "DUP";
  case MPI_COMBINER_CONTIGUOUS:
    return "CONTIGUOUS";
  case MPI_COMBINER_VECTOR:
    return "VECTOR";
  case MPI_COMBINER_HVECTOR:
    return "HVECTOR";
  case MPI_COMBINER_INDEXED:
    return "INDEXED";
  case MPI_COMBINER_HINDEXED:
    return "HINDEXED";
  case MPI_COMBINER_INDEXED_BLOCK:
    return "INDEXED_BLOCK";
  case MPI_COMBINER_STRUCT:
    return "STRUCT";
  case MPI_COMBINER_SUBARRAY:
    return "SUBARRAY";
  case MPI_COMBINER_DARRAY:
    return "DARRAY";
  case MPI_COMBINER_F90_REAL:
    return "F90_REAL";
  case MPI_COMBINER_F90_COMPLEX:
    return "F90_COMPLEX";
  case MPI_COMBINER_F90_INTEGER:
    return "F90_INTEGER";
  case MPI_COMBINER_RESIZED:
    return "RESIZED";
  case MPI_COMBINER_HINDEXED_BLOCK:
    return "HINDEXED_BLOCK";
  default:
    return "UNKNOWN";
  }
}

static int get_combiner(MPI_Datatype t) {
  int ni = 0, na = 0, nt = 0, comb = 0;
  MPI_Type_get_envelope(t, &ni, &na, &nt, &comb);
  return comb;
}

static int classify_type_int(MPI_Datatype dt) {
  int ni = 0, na = 0, nt = 0, comb = 0;
  real_MPI_Type_get_envelope(dt, &ni, &na, &nt, &comb);
  return comb;
}

static MPI_Datatype unwrap_dup(MPI_Datatype t) {
  int ni = 0, na = 0, nt = 0, comb = 0;
  real_MPI_Type_get_envelope(t, &ni, &na, &nt, &comb);
  if (comb != MPI_COMBINER_DUP)
    return t;
  std::vector<int> ints(ni);
  std::vector<MPI_Aint> addrs(na);
  std::vector<MPI_Datatype> types(nt);
  MPI_Type_get_contents(t, ni, na, nt, ints.data(), addrs.data(), types.data());
  return unwrap_dup(types[0]);
}

static std::string get_named_type_string(MPI_Datatype dt) {
  if (dt == MPI_CHAR)
    return "MPI_CHAR";
  if (dt == MPI_SIGNED_CHAR)
    return "MPI_SIGNED_CHAR";
  if (dt == MPI_UNSIGNED_CHAR)
    return "MPI_UNSIGNED_CHAR";
  if (dt == MPI_BYTE)
    return "MPI_BYTE";
  if (dt == MPI_SHORT)
    return "MPI_SHORT";
  if (dt == MPI_UNSIGNED_SHORT)
    return "MPI_UNSIGNED_SHORT";
  if (dt == MPI_INT)
    return "MPI_INT";
  if (dt == MPI_UNSIGNED)
    return "MPI_UNSIGNED";
  if (dt == MPI_LONG)
    return "MPI_LONG";
  if (dt == MPI_UNSIGNED_LONG)
    return "MPI_UNSIGNED_LONG";
  if (dt == MPI_LONG_LONG)
    return "MPI_LONG_LONG";
  if (dt == MPI_UNSIGNED_LONG_LONG)
    return "MPI_UNSIGNED_LONG_LONG";
  if (dt == MPI_FLOAT)
    return "MPI_FLOAT";
  if (dt == MPI_DOUBLE)
    return "MPI_DOUBLE";
  if (dt == MPI_LONG_DOUBLE)
    return "MPI_LONG_DOUBLE";
  if (dt == MPI_C_BOOL)
    return "MPI_C_BOOL";
  return "UNKNOWN_NAMED";
}

static inline uint64_t fnv1a64_has(const std::string &s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

static std::string datatype_signature(MPI_Datatype t) {
  t = unwrap_dup(t);

  int ni = 0, na = 0, nt = 0, comb = 0;
  real_MPI_Type_get_envelope(t, &ni, &na, &nt, &comb);

  if (comb == MPI_COMBINER_NAMED) {
    return (get_named_type_string(t));
  }

  std::vector<int> ints(ni);
  std::vector<MPI_Aint> addrs(na);
  std::vector<MPI_Datatype> types(nt);
  if (ni || na || nt) {
    MPI_Type_get_contents(t, ni, na, nt, ints.data(), addrs.data(),
                          types.data());
  }
  const std::string head = std::string(classify_type(comb));
  if (comb == MPI_COMBINER_STRUCT) {
    int nblocks = (ni > 0) ? ints[0] : 0;
    std::unordered_set<std::string> uniq;
    uniq.reserve((size_t)nblocks);
    for (int i = 0; i < nblocks; ++i) {
      uniq.insert(datatype_signature(types[i]));
    }
    if (uniq.empty())
      return head;
    if (uniq.size() == 1)
      return head + " " + *uniq.begin();
    return head + " MIXED";
  }
  if (nt > 0) {
    return head + " " + datatype_signature(types[0]);
  }
  return head;
}

static std::vector<uint8_t> pack_signature_map() {
  std::vector<uint8_t> buf;
  buf.reserve(_sig_map.size() * 32);
  for (auto &kv : _sig_map) {
    uint64_t h = kv.first;
    const std::string &s = kv.second;
    uint32_t len = (uint32_t)s.size();

    size_t old = buf.size();
    buf.resize(old + sizeof(uint64_t) + sizeof(uint32_t) + len);
    std::memcpy(&buf[old], &h, sizeof(uint64_t));
    std::memcpy(&buf[old + sizeof(uint64_t)], &len, sizeof(uint32_t));
    std::memcpy(&buf[old + sizeof(uint64_t) + sizeof(uint32_t)], s.data(), len);
  }
  return buf;
}

static std::string format_num_bytes(size_t num_bytes) {
  if (num_bytes == 0)
    return "0 B";
  const char *suffixes[] = {"B", "KB", "MB", "GB", "TB"};
  int i = 0;
  double d = (double)num_bytes;
  while (d >= 1024.0 && i < 4) {
    d /= 1024.0;
    ++i;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << d << " " << suffixes[i];
  return oss.str();
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  init();
  std::lock_guard<std::mutex> lk(_m);
  std::string sig = datatype_signature(datatype);
  uint64_t key = fnv1a64_has(sig);
  _sig_map.emplace(key, sig);
  int type_size_num_bytes = 0;
  real_MPI_Type_size(datatype, &type_size_num_bytes);
  send_stats[key].num_calls++;
  send_stats[key].num_bytes += (size_t)count * (size_t)type_size_num_bytes;
  return real_MPI_Send(buf, count, datatype, dest, tag, comm);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request) {
  init();
  std::lock_guard<std::mutex> lk(_m);
  std::string sig = datatype_signature(datatype);
  uint64_t key = fnv1a64_has(sig);
  _sig_map.emplace(key, sig);
  int type_size_num_bytes = 0;
  real_MPI_Type_size(datatype, &type_size_num_bytes);
  isend_stats[key].num_calls++;
  isend_stats[key].num_bytes += (size_t)count * (size_t)type_size_num_bytes;
  send_size.push_back((size_t)count * (size_t)type_size_num_bytes);
  return real_MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) {
  init();
  std::lock_guard<std::mutex> lk(_m);
  std::string sig = datatype_signature(datatype);
  uint64_t key = fnv1a64_has(sig);
  _sig_map.emplace(key, sig);
  int type_size_num_bytes = 0;
  real_MPI_Type_size(datatype, &type_size_num_bytes);
  recv_stats[key].num_calls++;
  recv_stats[key].num_bytes += (size_t)count * (size_t)type_size_num_bytes;
  return real_MPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) {
  init();
  std::lock_guard<std::mutex> lk(_m);
  std::string sig = datatype_signature(datatype);
  uint64_t key = fnv1a64_has(sig);
  _sig_map.emplace(key, sig);
  int type_size_num_bytes = 0;
  real_MPI_Type_size(datatype, &type_size_num_bytes);
  irecv_stats[key].num_calls++;
  irecv_stats[key].num_bytes += (size_t)count * (size_t)type_size_num_bytes;
  return real_MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

struct CommStat {
  int map_id;
  uint64_t type_key;
  uint64_t call_count;
  uint64_t byte_count;
};

int MPI_Finalize() {
  init();
  int rank = 0, world = 1;
  real_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  real_MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Datatype comm_stat_type;
  int blocklengths[4] = {1, 1, 1, 1};
  MPI_Aint displs[4];
  MPI_Datatype types[4] = {MPI_INT, MPI_UINT64_T, MPI_UINT64_T, MPI_UINT64_T};
  CommStat dummy{};
  MPI_Aint base;
  MPI_Get_address(&dummy, &base);
  MPI_Get_address(&dummy.map_id, &displs[0]);
  MPI_Get_address(&dummy.type_key, &displs[1]);
  MPI_Get_address(&dummy.call_count, &displs[2]);
  MPI_Get_address(&dummy.byte_count, &displs[3]);
  for (int i = 0; i < 4; ++i) {
    displs[i] = MPI_Aint_diff(displs[i], base);
  }
  MPI_Type_create_struct(4, blocklengths, displs, types, &comm_stat_type);
  MPI_Type_commit(&comm_stat_type);
  std::vector<CommStat> local_stats;
  const int local_size = send_size.size();
  std::vector<int> rank_sizes;
  if (rank == 0) {
    rank_sizes.resize(world);
  }

  MPI_Gather(&local_size, 1, MPI_INT, rank_sizes.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  std::vector<std::size_t> total_comms; 

  if (rank == 0) {
    std::vector<int> displacements;
    displacements.push_back(0);
    for (size_t i = 0; i < rank_sizes.size() - 1; ++i) {
      displacements.push_back(displacements.back() + rank_sizes[i]);
    }
    std::size_t total_comms_size = 0;
    for (const auto &size : rank_sizes) {
      total_comms_size += size;
    }
    total_comms.resize(total_comms_size);

    MPI_Gatherv(send_size.data(), local_size, MPI_UNSIGNED_LONG,
                total_comms.data(), rank_sizes.data(), displacements.data(),
                MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(send_size.data(), local_size, MPI_UNSIGNED_LONG, nullptr,
                nullptr, nullptr, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::ofstream fs("data.bin",
                     std::ios::out | std::ios::binary | std::ios::app);
    if (fs.is_open()) {
      fs.write(reinterpret_cast<const char *>(total_comms.data()),
               total_comms.size() * sizeof(std::size_t));
      fs.close();
    }
  }

  {
    std::lock_guard<std::mutex> lk(_m);
    for (auto &p : send_stats)
      local_stats.push_back(
          {0, p.first, (uint64_t)p.second.num_calls, (uint64_t)p.second.num_bytes});
    for (auto &p : isend_stats)
      local_stats.push_back(
          {1, p.first, (uint64_t)p.second.num_calls, (uint64_t)p.second.num_bytes});
    for (auto &p : recv_stats)
      local_stats.push_back(
          {2, p.first, (uint64_t)p.second.num_calls, (uint64_t)p.second.num_bytes});
    for (auto &p : irecv_stats)
      local_stats.push_back(
          {3, p.first, (uint64_t)p.second.num_calls, (uint64_t)p.second.num_bytes});
  }

  int local_n = (int)local_stats.size();
  std::vector<int> recv_counts(rank == 0 ? world : 0);
  real_MPI_Gather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0,
                  MPI_COMM_WORLD);

  std::vector<CommStat> global_stats;
  std::vector<int> displs_counts(rank == 0 ? world : 0);
  if (rank == 0) {
    int total = 0;
    for (int i = 0; i < world; ++i) {
      displs_counts[i] = total;
      total += (recv_counts.empty() ? 0 : recv_counts[i]);
    }
    global_stats.resize(total);
  }

  real_MPI_Gatherv(local_stats.data(), local_n, comm_stat_type,
                   global_stats.data(), recv_counts.data(),
                   displs_counts.data(), comm_stat_type, 0, MPI_COMM_WORLD);

  std::vector<uint8_t> local_sig_buf = pack_signature_map();
  int local_sig_size = (int)local_sig_buf.size();
  std::vector<int> sig_sizes(rank == 0 ? world : 0);
  real_MPI_Gather(&local_sig_size, 1, MPI_INT, sig_sizes.data(), 1, MPI_INT, 0,
                  MPI_COMM_WORLD);
  std::vector<int> sig_displs(rank == 0 ? world : 0);
  std::vector<uint8_t> all_sig_buf;
  if (rank == 0) {
    int total = 0;
    for (int i = 0; i < world; ++i) {
      sig_displs[i] = total;
      total += (sig_sizes.empty() ? 0 : sig_sizes[i]);
    }
    all_sig_buf.resize(total);
  }

  real_MPI_Gatherv(local_sig_buf.data(), local_sig_size, MPI_BYTE,
                   all_sig_buf.data(), sig_sizes.data(), sig_displs.data(),
                   MPI_BYTE, 0, MPI_COMM_WORLD);

  std::unordered_map<uint64_t, std::string> hash2sig;
  if (rank == 0) {
    for (int i = 0; i < world; ++i) {
      size_t off = (size_t)sig_displs[i];
      size_t end = off + (size_t)sig_sizes[i];
      while (off + sizeof(uint64_t) + sizeof(uint32_t) <= end) {
        uint64_t h;
        uint32_t len;
        std::memcpy(&h, &all_sig_buf[off], sizeof(uint64_t));
        off += sizeof(uint64_t);
        std::memcpy(&len, &all_sig_buf[off], sizeof(uint32_t));
        off += sizeof(uint32_t);
        if (off + len > end)
          break;
        std::string s((const char *)&all_sig_buf[off], (size_t)len);
        off += len;
        if (!hash2sig.count(h))
          hash2sig.emplace(h, std::move(s));
      }
    }
  }

  if (rank == 0) {
    std::unordered_map<uint64_t, Statistics> total_sends, total_isends, total_recvs,
        total_irecvs;
    for (const auto &st : global_stats) {
      switch (st.map_id) {
      case 0:
        total_sends[st.type_key].num_calls += (size_t)st.call_count;
        total_sends[st.type_key].num_bytes += (size_t)st.byte_count;
        break;
      case 1:
        total_isends[st.type_key].num_calls += (size_t)st.call_count;
        total_isends[st.type_key].num_bytes += (size_t)st.byte_count;
        break;
      case 2:
        total_recvs[st.type_key].num_calls += (size_t)st.call_count;
        total_recvs[st.type_key].num_bytes += (size_t)st.byte_count;
        break;
      case 3:
        total_irecvs[st.type_key].num_calls += (size_t)st.call_count;
        total_irecvs[st.type_key].num_bytes += (size_t)st.byte_count;
        break;
      }
    }

    //This will be ugly
    auto print_summary = [&](const char *title, const auto &map) {
      std::printf("%s\n", title);
      if (map.empty()) {
        std::printf("  (NO COMMS)\n");
        return;
      }
      size_t total_num_bytes = 0;
      for (const auto &[key, st] : map) {
        const char *name = nullptr;
        std::string fallback;
        auto it = hash2sig.find(key);
        if (it != hash2sig.end()) {
          name = it->second.c_str();
        } else {
          char buf[64];
          std::snprintf(buf, sizeof(buf), "<UNKNOWN 0x%016llx>",
                        (unsigned long long)key);
          fallback = buf;
          name = fallback.c_str();
        }
        std::printf("  %-60s -> %-10zu num_calls, %s\n", name, st.num_calls,
                    format_num_bytes(st.num_bytes).c_str());
        total_num_bytes += st.num_bytes;
      }
      std::printf(
          "  ------------------------------------------------------------ "
          "Total: %s\n",
          format_num_bytes(total_num_bytes).c_str());
    };

    std::printf("\n--- MPI Sniffer ---\n");
    print_summary("[MPI_Send   num_calls]", total_sends);
    print_summary("[MPI_Isend  num_calls]", total_isends);
    print_summary("[MPI_Recv   num_calls]", total_recvs);
    print_summary("[MPI_Irecv  num_calls]", total_irecvs);
    std::printf("--------------------------------------------------------------"
                "-------\n");
  }

  MPI_Type_free(&comm_stat_type);
  return real_MPI_Finalize();
}

// T1.6 — GPU adjacency (conflict-graph neighbour list) builder, device upload
// of the adjacency table (docs/design.md §2.2 "adj.u32", §2.3 kiss/adjacency.h, §5.2).
//
// This header is deliberately free of CUDA headers so plain .cpp translation
// units (tools/build_adj.cpp, tests/test_adjacency.cpp) can include it; the
// implementation lives in cuda/build_adjacency.cu (library kiss_cuda).
//
// It does NOT depend on cuda/kiss_cuda.h (T1.5): the packed words are uploaded
// here by AdjacencyBuilder itself. The raw entry point build_adjacency_rows_device()
// takes device pointers so T1.5's DeviceLeech::packed can be passed straight in
// (the §2.3 signature `build_adjacency(const DeviceLeech&, ...)` is a one-line
// wrapper around it once kiss_cuda.h exists).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/leech.h"
#include "kiss/types.h"

namespace kiss::cuda {

// Default number of rows per chunk streamed to disk: 4096 rows × 4600 × 4 B =
// 75.4 MB per device buffer (two buffers + two pinned host buffers ≈ 300 MB).
constexpr uint32_t ADJ_DEFAULT_CHUNK_ROWS = 4096;

// Raw device-side entry point (all pointers are DEVICE pointers).
//   d_packed : uint32[6][N] SoA (docs/design.md §2.2 layout, i.e. pack_vectors output)
//   d_out    : uint32[nrows][DEG]; row r receives the sorted neighbours of
//              vertex first_row + r (dot == 16)
//   d_counts : uint32[nrows]; number of hits found for each row. The kernel
//              always writes DEG entries per row, but only rows whose count is
//              exactly DEG are meaningful — the caller must check.
//   stream   : a cudaStream_t (passed as void* to keep this header CUDA-free);
//              nullptr = the legacy default stream.
// Requires first_row + nrows <= N. Throws std::runtime_error on CUDA errors
// (launch errors only; the kernel is asynchronous on `stream`).
void build_adjacency_rows_device(const uint32_t* d_packed, uint32_t first_row, uint32_t nrows,
                                 uint32_t* d_out, uint32_t* d_counts, void* stream);

// Context object: uploads the packed words once, owns the device/pinned
// buffers, and computes row ranges into HOST memory.
class AdjacencyBuilder {
 public:
  // packed.size() must be 6*N (pack_vectors(L.C)).
  explicit AdjacencyBuilder(const std::vector<uint32_t>& packed,
                            uint32_t chunk_rows = ADJ_DEFAULT_CHUNK_ROWS);
  ~AdjacencyBuilder();
  AdjacencyBuilder(const AdjacencyBuilder&) = delete;
  AdjacencyBuilder& operator=(const AdjacencyBuilder&) = delete;

  // Computes rows [first_row, first_row + nrows) into h_out (nrows*DEG uint32,
  // ordinary host memory). Synchronous. Throws std::runtime_error if any row
  // does not have exactly DEG hits (message names the row and the count) or on
  // a CUDA error. nrows may exceed chunk_rows (processed in pieces).
  void build_rows(uint32_t first_row, uint32_t nrows, uint32_t* h_out) const;

  uint32_t chunk_rows() const { return chunk_rows_; }
  const uint32_t* device_packed() const { return d_packed_; }

  // Streams the whole table (rows 0..N-1) to `out_file` in chunk_rows pieces:
  // kernel(chunk k+1) overlaps fwrite(chunk k). Writes to "<out_file>.tmp" and
  // renames on success, so a partial file is never mistaken for a valid one.
  // Returns wall-clock seconds spent (kernel+copy+write). Throws on error.
  double build_file(const std::filesystem::path& out_file) const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  uint32_t* d_packed_ = nullptr;
  uint32_t chunk_rows_ = ADJ_DEFAULT_CHUNK_ROWS;
};

struct BuildAdjacencyStats {
  double upload_s = 0;   // pack + upload of the 4.7 MB packed words
  double build_s = 0;    // AdjacencyBuilder::build_file
  double total_s = 0;
  uint32_t chunk_rows = 0;
  std::size_t bytes = 0;
};

// Convenience used by BOTH tools/build_adj and tests/test_adjacency: builds
// <out_file> (normally data/adj.u32) from L on the GPU. Throws on error.
BuildAdjacencyStats build_adjacency_file(const Leech& L, const std::filesystem::path& out_file,
                                         uint32_t chunk_rows = ADJ_DEFAULT_CHUNK_ROWS);

// True if at least one CUDA device is usable (cudaGetDeviceCount succeeds with n>0).
bool cuda_available();

// Uploads the whole memory-mapped table to the device: cudaMalloc of
// N*DEG*4 = 3,616,704,000 bytes + cudaMemcpy in 64 MB pieces. Returns the device
// pointer (uint32[N][DEG], row-major; free with adjacency_device_free). Throws
// std::runtime_error with a clear message (including free/total device memory)
// if cudaMalloc fails. This is the "Adjacency::to_device()" of docs/design.md §2.3 /
// the T1.6 card, provided as a free function so that libkiss (CPU) stays free
// of CUDA symbols.
uint32_t* adjacency_to_device(const Adjacency& adj);
void adjacency_device_free(uint32_t* d_adj);

// Free/total device memory in bytes (cudaMemGetInfo); both 0 if no device.
void device_mem_info(std::size_t* free_bytes, std::size_t* total_bytes);

}  // namespace kiss::cuda

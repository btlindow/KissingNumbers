// T1.6 — GPU adjacency builder (docs/design.md T1.6, §2.2, §5). See adjacency_cuda.h.
//
// Kernel design (one block per ROWS_PER_BLOCK consecutive rows):
//   * the block's row vectors (6 packed words each) sit in registers;
//   * 512 threads stride over all j in [0, N): 6 coalesced loads of the SoA
//     packed words + 6 __dp4a per row → int32 dot; hits (dot == 16) are
//     appended to a per-row shared buffer through a shared atomic counter;
//   * each row's ≤ 4608 hits are then sorted in shared memory with
//     cub::BlockRadixSort (keys < 2^18, so only 18 bits are sorted) and the
//     first DEG entries are written out; the hit count goes to d_counts[] and
//     the host verifies it is exactly DEG for every row.
// The sort's temp storage aliases the row's own hit buffer (the keys are in
// registers by then), so static shared memory is ROWS_PER_BLOCK × BUF words.
#include "adjacency_cuda.h"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "kiss/io.h"

namespace kiss::cuda {

namespace {

constexpr int THREADS = 512;
constexpr int ITEMS = 9;                     // 512 × 9 = 4608 ≥ DEG = 4600
constexpr int CAP = THREADS * ITEMS;         // hit-buffer capacity per row
constexpr int ROWS_PER_BLOCK = 2;            // rows sharing one pass over the packed words
constexpr int KEY_BITS = 18;                 // N = 196560 < 2^18
static_assert(CAP >= DEG, "hit buffer must hold a full row");
static_assert((1 << KEY_BITS) > N, "vertex index must fit in KEY_BITS");

using Sorter = cub::BlockRadixSort<uint32_t, THREADS, ITEMS>;
constexpr int SORT_WORDS = static_cast<int>((sizeof(typename Sorter::TempStorage) + 3) / 4);
constexpr int BUF = (SORT_WORDS > CAP ? SORT_WORDS : CAP) + 4;  // words per row buffer
constexpr int BUF_ALIGNED = (BUF + 3) & ~3;                        // keep 16-byte alignment
static_assert(static_cast<std::size_t>(ROWS_PER_BLOCK) * BUF_ALIGNED * 4 <= 48 * 1024,
              "static shared memory budget");

[[noreturn]] void fail(const char* what, cudaError_t e) {
  throw std::runtime_error(std::string("build_adjacency: ") + what + " failed: " + cudaGetErrorString(e));
}
inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) fail(what, e);
}

__global__ void __launch_bounds__(THREADS)
adjacency_rows_kernel(const uint32_t* __restrict__ packed, uint32_t first_row, uint32_t nrows,
                      uint32_t* __restrict__ out, uint32_t* __restrict__ counts) {
  __shared__ __align__(16) uint32_t buf[ROWS_PER_BLOCK][BUF_ALIGNED];
  __shared__ int cnt[ROWS_PER_BLOCK];

  const uint32_t base = first_row + blockIdx.x * ROWS_PER_BLOCK;   // absolute vertex index
  const uint32_t end = first_row + nrows;
  if (threadIdx.x < ROWS_PER_BLOCK) cnt[threadIdx.x] = 0;

  int a[ROWS_PER_BLOCK][6];
#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const uint32_t row = base + r < end ? base + r : base;   // tail rows: duplicate work, no write
#pragma unroll
    for (int w = 0; w < 6; ++w) a[r][w] = static_cast<int>(packed[static_cast<std::size_t>(w) * N + row]);
  }
  __syncthreads();

  for (uint32_t j = threadIdx.x; j < static_cast<uint32_t>(N); j += THREADS) {
    int b[6];
#pragma unroll
    for (int w = 0; w < 6; ++w) b[w] = static_cast<int>(packed[static_cast<std::size_t>(w) * N + j]);
#pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
      int acc = 0;
#pragma unroll
      for (int w = 0; w < 6; ++w) acc = __dp4a(a[r][w], b[w], acc);
      if (acc == 16) {
        const int idx = atomicAdd(&cnt[r], 1);
        if (idx < CAP) buf[r][idx] = j;
      }
    }
  }
  __syncthreads();

  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const uint32_t row = base + r;
    if (row >= end) break;
    const int c = cnt[r];
    const int valid = c < CAP ? c : CAP;
    uint32_t keys[ITEMS];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      const int idx = threadIdx.x * ITEMS + k;
      keys[k] = idx < valid ? buf[r][idx] : 0xFFFFFFFFu;
    }
    __syncthreads();   // everyone has its keys; buf[r] may now be reused as sort scratch
    Sorter(*reinterpret_cast<typename Sorter::TempStorage*>(buf[r])).Sort(keys, 0, KEY_BITS);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) buf[r][threadIdx.x * ITEMS + k] = keys[k];
    __syncthreads();
    uint32_t* dst = out + static_cast<std::size_t>(row - first_row) * DEG;
    for (int k = threadIdx.x; k < DEG; k += THREADS) dst[k] = buf[r][k];
    if (threadIdx.x == 0) counts[row - first_row] = static_cast<uint32_t>(c);
    __syncthreads();
  }
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void build_adjacency_rows_device(const uint32_t* d_packed, uint32_t first_row, uint32_t nrows,
                                 uint32_t* d_out, uint32_t* d_counts, void* stream) {
  if (nrows == 0) return;
  if (first_row + nrows > static_cast<uint32_t>(N) || first_row + nrows < first_row)
    throw std::runtime_error("build_adjacency_rows_device: row range out of bounds");
  const unsigned blocks = (nrows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  adjacency_rows_kernel<<<blocks, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
      d_packed, first_row, nrows, d_out, d_counts);
  check(cudaGetLastError(), "kernel launch");
}

// ---------------------------------------------------------------------------
// AdjacencyBuilder
// ---------------------------------------------------------------------------
struct AdjacencyBuilder::Impl {
  uint32_t* d_out[2] = {nullptr, nullptr};
  uint32_t* d_counts[2] = {nullptr, nullptr};
  uint32_t* h_out[2] = {nullptr, nullptr};      // pinned
  uint32_t* h_counts[2] = {nullptr, nullptr};   // pinned
  cudaStream_t stream[2] = {nullptr, nullptr};
  cudaEvent_t done[2] = {nullptr, nullptr};

  void release() {
    for (int s = 0; s < 2; ++s) {
      if (done[s]) cudaEventDestroy(done[s]);
      if (stream[s]) cudaStreamDestroy(stream[s]);
      if (h_out[s]) cudaFreeHost(h_out[s]);
      if (h_counts[s]) cudaFreeHost(h_counts[s]);
      if (d_out[s]) cudaFree(d_out[s]);
      if (d_counts[s]) cudaFree(d_counts[s]);
      done[s] = nullptr; stream[s] = nullptr; h_out[s] = nullptr; h_counts[s] = nullptr;
      d_out[s] = nullptr; d_counts[s] = nullptr;
    }
  }
};

AdjacencyBuilder::AdjacencyBuilder(const std::vector<uint32_t>& packed, uint32_t chunk_rows)
    : chunk_rows_(chunk_rows) {
  if (packed.size() != static_cast<std::size_t>(6) * N)
    throw std::runtime_error("AdjacencyBuilder: packed.size() must be 6*N (" +
                             std::to_string(6 * static_cast<std::size_t>(N)) + "), got " +
                             std::to_string(packed.size()));
  if (chunk_rows_ == 0) chunk_rows_ = ADJ_DEFAULT_CHUNK_ROWS;
  chunk_rows_ = (chunk_rows_ + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK * ROWS_PER_BLOCK;
  const std::size_t chunk_bytes = static_cast<std::size_t>(chunk_rows_) * DEG * sizeof(uint32_t);

  impl_ = new Impl();
  try {
    check(cudaMalloc(&d_packed_, packed.size() * sizeof(uint32_t)), "cudaMalloc(packed)");
    check(cudaMemcpy(d_packed_, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
          "cudaMemcpy(packed)");
    for (int s = 0; s < 2; ++s) {
      check(cudaMalloc(&impl_->d_out[s], chunk_bytes), "cudaMalloc(chunk)");
      check(cudaMalloc(&impl_->d_counts[s], chunk_rows_ * sizeof(uint32_t)), "cudaMalloc(counts)");
      check(cudaMallocHost(&impl_->h_out[s], chunk_bytes), "cudaMallocHost(chunk)");
      check(cudaMallocHost(&impl_->h_counts[s], chunk_rows_ * sizeof(uint32_t)), "cudaMallocHost(counts)");
      check(cudaStreamCreateWithFlags(&impl_->stream[s], cudaStreamNonBlocking), "cudaStreamCreate");
      check(cudaEventCreateWithFlags(&impl_->done[s], cudaEventDisableTiming), "cudaEventCreate");
    }
  } catch (...) {
    impl_->release();
    delete impl_;
    impl_ = nullptr;
    if (d_packed_) cudaFree(d_packed_);
    d_packed_ = nullptr;
    throw;
  }
}

AdjacencyBuilder::~AdjacencyBuilder() {
  if (impl_) {
    impl_->release();
    delete impl_;
  }
  if (d_packed_) cudaFree(d_packed_);
}

namespace {

// Validate the hit counts of a finished chunk; throws naming the first bad row.
void check_counts(const uint32_t* counts, uint32_t first_row, uint32_t nrows) {
  for (uint32_t r = 0; r < nrows; ++r)
    if (counts[r] != static_cast<uint32_t>(DEG))
      throw std::runtime_error("build_adjacency: row " + std::to_string(first_row + r) + " has " +
                               std::to_string(counts[r]) + " neighbours at dot==16, expected " +
                               std::to_string(DEG));
}

}  // namespace

void AdjacencyBuilder::build_rows(uint32_t first_row, uint32_t nrows, uint32_t* h_out) const {
  if (first_row + nrows > static_cast<uint32_t>(N) || first_row + nrows < first_row)
    throw std::runtime_error("AdjacencyBuilder::build_rows: row range out of bounds");
  uint32_t done_rows = 0;
  while (done_rows < nrows) {
    const uint32_t n = std::min(chunk_rows_, nrows - done_rows);
    const uint32_t r0 = first_row + done_rows;
    build_adjacency_rows_device(d_packed_, r0, n, impl_->d_out[0], impl_->d_counts[0], impl_->stream[0]);
    check(cudaMemcpyAsync(h_out + static_cast<std::size_t>(done_rows) * DEG, impl_->d_out[0],
                          static_cast<std::size_t>(n) * DEG * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                          impl_->stream[0]),
          "cudaMemcpyAsync(rows)");
    check(cudaMemcpyAsync(impl_->h_counts[0], impl_->d_counts[0], n * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, impl_->stream[0]),
          "cudaMemcpyAsync(counts)");
    check(cudaStreamSynchronize(impl_->stream[0]), "cudaStreamSynchronize");
    check_counts(impl_->h_counts[0], r0, n);
    done_rows += n;
  }
}

double AdjacencyBuilder::build_file(const std::filesystem::path& out_file) const {
  const auto t0 = std::chrono::steady_clock::now();
  const std::filesystem::path tmp = out_file.string() + ".tmp";
  std::FILE* f = kiss::fopen_path(tmp, "wb");
  if (!f) throw std::runtime_error("build_adjacency: cannot open " + tmp.string() + " for writing");

  const uint32_t nchunks = (static_cast<uint32_t>(N) + chunk_rows_ - 1) / chunk_rows_;
  auto rows_of = [&](uint32_t c) {
    return std::min(chunk_rows_, static_cast<uint32_t>(N) - c * chunk_rows_);
  };
  auto launch = [&](uint32_t c) {
    const int s = static_cast<int>(c & 1u);
    const uint32_t r0 = c * chunk_rows_, n = rows_of(c);
    build_adjacency_rows_device(d_packed_, r0, n, impl_->d_out[s], impl_->d_counts[s], impl_->stream[s]);
    check(cudaMemcpyAsync(impl_->h_out[s], impl_->d_out[s],
                          static_cast<std::size_t>(n) * DEG * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                          impl_->stream[s]),
          "cudaMemcpyAsync(rows)");
    check(cudaMemcpyAsync(impl_->h_counts[s], impl_->d_counts[s], n * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, impl_->stream[s]),
          "cudaMemcpyAsync(counts)");
    check(cudaEventRecord(impl_->done[s], impl_->stream[s]), "cudaEventRecord");
  };
  auto finish = [&](uint32_t c) {
    const int s = static_cast<int>(c & 1u);
    const uint32_t r0 = c * chunk_rows_, n = rows_of(c);
    check(cudaEventSynchronize(impl_->done[s]), "cudaEventSynchronize");
    check_counts(impl_->h_counts[s], r0, n);
    const std::size_t words = static_cast<std::size_t>(n) * DEG;
    if (std::fwrite(impl_->h_out[s], sizeof(uint32_t), words, f) != words)
      throw std::runtime_error("build_adjacency: short write to " + tmp.string());
  };

  try {
    for (uint32_t c = 0; c < nchunks; ++c) {
      launch(c);            // chunk c runs on the GPU while chunk c-1 is written
      if (c > 0) finish(c - 1);
    }
    finish(nchunks - 1);
    if (std::fflush(f) != 0 || std::fclose(f) != 0)
      throw std::runtime_error("build_adjacency: flush/close failed on " + tmp.string());
    f = nullptr;
  } catch (...) {
    if (f) std::fclose(f);
    cudaDeviceSynchronize();
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    throw;
  }
  std::error_code ec;
  std::filesystem::rename(tmp, out_file, ec);
  if (ec) throw std::runtime_error("build_adjacency: rename " + tmp.string() + " -> " + out_file.string() +
                                   " failed: " + ec.message());
  return seconds_since(t0);
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------
BuildAdjacencyStats build_adjacency_file(const Leech& L, const std::filesystem::path& out_file,
                                         uint32_t chunk_rows) {
  BuildAdjacencyStats st;
  const auto t0 = std::chrono::steady_clock::now();
  if (L.C.size() != static_cast<std::size_t>(N))
    throw std::runtime_error("build_adjacency_file: Leech has " + std::to_string(L.C.size()) +
                             " vectors, expected " + std::to_string(N));
  const std::vector<uint32_t> packed = pack_vectors(L.C);
  AdjacencyBuilder builder(packed, chunk_rows);
  st.upload_s = seconds_since(t0);
  st.chunk_rows = builder.chunk_rows();
  st.build_s = builder.build_file(out_file);
  st.bytes = Adjacency::EXPECTED_BYTES;
  st.total_s = seconds_since(t0);
  return st;
}

bool cuda_available() {
  int n = 0;
  const cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    return false;
  }
  return n > 0;
}

void device_mem_info(std::size_t* free_bytes, std::size_t* total_bytes) {
  std::size_t fr = 0, to = 0;
  if (cudaMemGetInfo(&fr, &to) != cudaSuccess) {
    cudaGetLastError();
    fr = to = 0;
  }
  if (free_bytes) *free_bytes = fr;
  if (total_bytes) *total_bytes = to;
}

uint32_t* adjacency_to_device(const Adjacency& adj) {
  const std::size_t bytes = adj.bytes();
  uint32_t* d = nullptr;
  const cudaError_t e = cudaMalloc(&d, bytes);
  if (e != cudaSuccess) {
    cudaGetLastError();
    std::size_t fr = 0, to = 0;
    device_mem_info(&fr, &to);
    throw std::runtime_error("adjacency_to_device: cudaMalloc of " + std::to_string(bytes) +
                             " bytes (3.62 GB adjacency table) failed: " + cudaGetErrorString(e) +
                             "; device memory free/total = " + std::to_string(fr) + "/" +
                             std::to_string(to) + " bytes. Is another process using the GPU?");
  }
  const std::size_t piece = static_cast<std::size_t>(64) << 20;   // 64 MB pieces from the mmap
  const auto* src = reinterpret_cast<const unsigned char*>(adj.data());
  for (std::size_t off = 0; off < bytes; off += piece) {
    const std::size_t n = std::min(piece, bytes - off);
    const cudaError_t ce = cudaMemcpy(reinterpret_cast<unsigned char*>(d) + off, src + off, n,
                                      cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
      cudaFree(d);
      throw std::runtime_error("adjacency_to_device: cudaMemcpy at offset " + std::to_string(off) +
                               " failed: " + cudaGetErrorString(ce));
    }
  }
  return d;
}

void adjacency_device_free(uint32_t* d_adj) {
  if (d_adj) cudaFree(d_adj);
}

}  // namespace kiss::cuda

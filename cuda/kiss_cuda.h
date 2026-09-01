// T1.5 — CUDA-side contract of the Leech-subset project (docs/design.md §2.3, §5.1).
//
// Everything here is plain C++ (no C linkage); host code that never launches
// kernels may include it from a .cpp file (it only needs <cuda_runtime.h>).
//
// Ownership: DeviceLeech is a pair of raw device pointers returned by
// upload_leech() and released by free_leech(); ScopedDeviceLeech is an
// optional RAII wrapper. All entry points are asynchronous with respect to the
// host on the given stream unless stated otherwise.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kiss/leech.h"
#include "kiss/types.h"

namespace kiss::cuda {

// ---------------------------------------------------------------------------
// Error handling: KISS_CUDA_CHECK(expr) throws std::runtime_error carrying
// file:line, the expression and cudaGetErrorString() when expr != cudaSuccess.
// ---------------------------------------------------------------------------
namespace detail {
inline void check(cudaError_t e, const char* expr, const char* file, int line) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                             " failed: " + cudaGetErrorString(e) + " (" +
                             cudaGetErrorName(e) + ")");
  }
}
}  // namespace detail

#define KISS_CUDA_CHECK(expr) ::kiss::cuda::detail::check((expr), #expr, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Constants shared with the search engine (T3.2).
// ---------------------------------------------------------------------------
constexpr int SMAX = 1024;                 // max |S| per chain (docs/design.md §5.2)
constexpr int INS_WORDS = (N + 31) / 32;   // uint32 words of one chain's inS bitmap = 6143
constexpr int PACKED_WORDS = 6;            // uint32 words per packed vector (24 int8)

// ---------------------------------------------------------------------------
// Device copy of the minimal vectors.
//   packed : uint32[6][N], structure-of-arrays (docs/design.md §2.2): word w of vector i
//            is packed[w*N + i] and holds coordinates 4w..4w+3 as int8 bytes,
//            little-endian. Produced by kiss::pack_vectors().
//   neg    : uint32[N], neg[i] = index of -C[i].
// ---------------------------------------------------------------------------
struct DeviceLeech {
  const uint32_t* packed = nullptr;  // [6][N]
  const uint32_t* neg = nullptr;     // [N]
};

// Packs L.C with kiss::pack_vectors, cudaMallocs and copies both arrays
// (5.5 MB). Synchronous. Throws std::runtime_error on any CUDA error.
DeviceLeech upload_leech(const Leech& L);

// cudaFree of both arrays; pointers are reset to nullptr. Safe on an empty
// (default-constructed / already freed) DeviceLeech. Does not throw.
void free_leech(DeviceLeech& d);

// RAII wrapper (optional convenience).
class ScopedDeviceLeech {
 public:
  explicit ScopedDeviceLeech(const Leech& L) : d_(upload_leech(L)) {}
  ~ScopedDeviceLeech() { free_leech(d_); }
  ScopedDeviceLeech(const ScopedDeviceLeech&) = delete;
  ScopedDeviceLeech& operator=(const ScopedDeviceLeech&) = delete;
  const DeviceLeech& get() const { return d_; }
  operator const DeviceLeech&() const { return d_; }

 private:
  DeviceLeech d_;
};

// ---------------------------------------------------------------------------
// T1.5: full tightness recompute for B chains (docs/design.md §5.1).
//
//   d_S     : uint32[B][SMAX] vertex indices (0..N-1); entries beyond
//             d_Ssize[b] are ignored. Duplicates are counted twice.
//   d_Ssize : uint32[B], each <= SMAX (values above SMAX are clamped).
//   d_tight : uint16[B][N] output, tight[b*N + v] = #{ s in S_b : <C[v],C[s]> == 16 }.
//
// Never materialises the N x |S| matrix. Work = B*N*|S| pairs, 6 dp4a each.
// Uses a stream-ordered scratch allocation (cudaMallocAsync) of
// min(B,256)*24 KB for the gathered packed S words. Fully asynchronous on
// `stream`; throws std::runtime_error on launch/allocation errors.
// ---------------------------------------------------------------------------
void tightness_full(const DeviceLeech& L, const uint32_t* d_S, const uint32_t* d_Ssize, int B,
                    uint16_t* d_tight, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Bonus for T3.2: number of free vertices per chain,
//   d_out[b] = #{ v : tight[b][v] == 0 && !inS_b(v) }.
//   d_inS_bits : uint32[B][INS_WORDS] bitmap, bit (v & 31) of word (v >> 5);
//                may be nullptr, in which case only tight == 0 is counted.
// Asynchronous on `stream`.
// ---------------------------------------------------------------------------
void count_free(const uint16_t* d_tight, const uint32_t* d_inS_bits, int B, uint32_t* d_out,
                cudaStream_t stream = 0);

// Note for T2.2 (pair-class histogram): not provided here. Derive it from the
// same kernel skeleton by replacing `cnt += (acc == 16)` with
// `hist[(acc + 32) >> 3]++` (7 register counters, acc in {-32,...,32} step 8)
// and writing uint16[7] per (chain, v); see docs/reports/T1.5.md §"T2.2".

}  // namespace kiss::cuda

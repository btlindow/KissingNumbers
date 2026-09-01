// T2.3 — triple-class histogram kernel (four-point numbers of the scheme).
// See triple_cuda.h and docs/reports/T2.3.md.
//
// Design (a direct extension of T2.2's pair_class_histogram):
//   1. pair_class_kernel: g[w] = 7 * class(x, w) + class(y, w) in 0..48.
//   2. Host counting sort of w by g[w] -> perm[N], offs[50].
//   3. gather_sorted_kernel: the minimal vectors in group order (SoA).
//   4. hist_kernel<ZB>: one 256-thread block per ZB consecutive z's (their
//      packed words in registers).  For each of the 49 groups the threads
//      stride over that group's contiguous w's, compute dot(w, z) with 6
//      dp4a per z and add 1 << (9 * class(w, z)) into a packed uint64 (seven
//      9-bit fields).  A thread sees at most ceil(93150 / 256) = 364 < 512
//      w's of one group, so no field overflows (the host enforces
//      group size <= 511 * 256).  After each group the fields are
//      warp-reduced and accumulated into s_hist[z][g * 7 + c] in shared
//      memory (ZB * 343 counters); written to d_M at the end.
#include "triple_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kiss::cuda {

namespace {

constexpr int TB = 256;         // threads per block
constexpr int ZB = 8;           // z vectors per block
constexpr int NGROUP = 49;      // (class(x,w), class(y,w)) pairs
constexpr int FIELD_BITS = 9;
constexpr unsigned FIELD_MASK = (1u << FIELD_BITS) - 1u;   // 511
constexpr int MAX_GROUP_SIZE = static_cast<int>(FIELD_MASK) * TB;  // 130816 > 93150

__device__ __forceinline__ int dot6(const int (&a)[6], const int (&b)[6]) {
  int acc = __dp4a(a[0], b[0], 0);
  acc = __dp4a(a[1], b[1], acc);
  acc = __dp4a(a[2], b[2], acc);
  acc = __dp4a(a[3], b[3], acc);
  acc = __dp4a(a[4], b[4], acc);
  acc = __dp4a(a[5], b[5], acc);
  return acc;
}

// dot in {-32,-16,-8,0,8,16,32} -> class 0..6 (kiss::ip_class order; the dots
// are not uniformly spaced: (d+32)/8 = {0,2,3,4,5,6,8} is compacted).
__device__ __forceinline__ int ip_class_dev(int d) {
  const int t = (d + 32) >> 3;
  return t - (t > 1) - (t > 7);
}

__device__ __forceinline__ unsigned warp_sum(unsigned v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_add_sync(0xffffffffu, v);
#else
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
#endif
}

// g[w] = 7 * class(x, w) + class(y, w).
__global__ void pair_class_kernel(const uint32_t* __restrict__ packed, uint32_t x, uint32_t y,
                                  uint8_t* __restrict__ g) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= N) return;
  int xw[6], yw[6], ww[6];
#pragma unroll
  for (int q = 0; q < 6; ++q) {
    xw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + x]);
    yw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + y]);
    ww[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + w]);
  }
  g[w] = static_cast<uint8_t>(7 * ip_class_dev(dot6(xw, ww)) + ip_class_dev(dot6(yw, ww)));
}

__global__ void gather_sorted_kernel(const uint32_t* __restrict__ packed,
                                     const uint32_t* __restrict__ perm,
                                     uint32_t* __restrict__ sorted) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  const uint32_t w = perm[t];
#pragma unroll
  for (int q = 0; q < 6; ++q)
    sorted[static_cast<std::size_t>(q) * N + t] = packed[static_cast<std::size_t>(q) * N + w];
}

template <int Z>
__global__ void __launch_bounds__(TB)
hist_kernel(const uint32_t* __restrict__ sorted,   // [6][N], w in group order
            const uint32_t* __restrict__ packed,   // [6][N], original order (for z)
            const uint32_t* __restrict__ offs,     // [50] group boundaries in `sorted`
            uint32_t z0, uint32_t nz,
            uint32_t* __restrict__ M) {            // [nz][343]
  __shared__ uint32_t s_hist[Z][NCLASS3];
  __shared__ int s_offs[NGROUP + 1];

  for (int t = threadIdx.x; t < Z * NCLASS3; t += TB) (&s_hist[0][0])[t] = 0u;
  if (threadIdx.x < NGROUP + 1) s_offs[threadIdx.x] = static_cast<int>(offs[threadIdx.x]);

  const uint32_t zb = z0 + static_cast<uint32_t>(blockIdx.x) * Z;
  const uint32_t zend = z0 + nz;
  int zw[Z][6];
#pragma unroll
  for (int k = 0; k < Z; ++k) {
    const uint32_t z = zb + static_cast<uint32_t>(k);
    const bool valid = z < zend;
#pragma unroll
    for (int q = 0; q < 6; ++q)
      zw[k][q] = valid ? static_cast<int>(packed[static_cast<std::size_t>(q) * N + z]) : 0;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  for (int g = 0; g < NGROUP; ++g) {
    const int beg = s_offs[g], end = s_offs[g + 1];
    if (beg == end) continue;
    unsigned long long h[Z];
#pragma unroll
    for (int k = 0; k < Z; ++k) h[k] = 0ull;
    for (int t = beg + static_cast<int>(threadIdx.x); t < end; t += TB) {
      int ww[6];
#pragma unroll
      for (int q = 0; q < 6; ++q) ww[q] = static_cast<int>(sorted[static_cast<std::size_t>(q) * N + t]);
#pragma unroll
      for (int k = 0; k < Z; ++k) {
        const int idx = ip_class_dev(dot6(ww, zw[k]));
        h[k] += 1ull << (idx * FIELD_BITS);
      }
    }
#pragma unroll
    for (int k = 0; k < Z; ++k) {
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        const unsigned f = static_cast<unsigned>(h[k] >> (c * FIELD_BITS)) & FIELD_MASK;
        const unsigned s = warp_sum(f);
        if (lane == 0 && s) atomicAdd(&s_hist[k][g * 7 + c], s);
      }
    }
  }
  __syncthreads();

  for (int t = threadIdx.x; t < Z * NCLASS3; t += TB) {
    const int k = t / NCLASS3, e = t - k * NCLASS3;
    const uint32_t z = zb + static_cast<uint32_t>(k);
    if (z < zend) M[static_cast<std::size_t>(z - z0) * NCLASS3 + e] = s_hist[k][e];
  }
}

struct DeviceBuffer {
  void* p = nullptr;
  explicit DeviceBuffer(std::size_t bytes) { KISS_CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuffer() { if (p) cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  template <class T> T* as() const { return static_cast<T*>(p); }
};

}  // namespace

void triple_class_histogram(const DeviceLeech& L, uint32_t x, uint32_t y, uint32_t z0, uint32_t nz,
                            uint32_t* d_M, cudaStream_t stream) {
  if (!L.packed) throw std::runtime_error("triple_class_histogram: DeviceLeech.packed is null");
  if (x >= static_cast<uint32_t>(N) || y >= static_cast<uint32_t>(N))
    throw std::runtime_error("triple_class_histogram: x or y out of range");
  if (z0 > static_cast<uint32_t>(N) || nz > static_cast<uint32_t>(N) - z0)
    throw std::runtime_error("triple_class_histogram: z range out of bounds");
  if (nz == 0) return;

  const std::size_t nw = static_cast<std::size_t>(N);
  DeviceBuffer d_g(nw);                                // uint8[N]
  DeviceBuffer d_perm(nw * sizeof(uint32_t));          // uint32[N]
  DeviceBuffer d_sorted(6 * nw * sizeof(uint32_t));    // uint32[6][N]
  DeviceBuffer d_offs((NGROUP + 1) * sizeof(uint32_t));

  pair_class_kernel<<<(N + TB - 1) / TB, TB, 0, stream>>>(L.packed, x, y, d_g.as<uint8_t>());
  KISS_CUDA_CHECK(cudaGetLastError());
  std::vector<uint8_t> g(nw);
  KISS_CUDA_CHECK(cudaMemcpyAsync(g.data(), d_g.p, nw, cudaMemcpyDeviceToHost, stream));
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));

  uint32_t count[NGROUP] = {0};
  for (std::size_t w = 0; w < nw; ++w) {
    if (g[w] >= NGROUP)
      throw std::runtime_error("triple_class_histogram: inner product out of range at w=" +
                               std::to_string(w));
    ++count[g[w]];
  }
  std::vector<uint32_t> offs(NGROUP + 1, 0u);
  for (int c = 0; c < NGROUP; ++c) {
    if (count[c] > static_cast<uint32_t>(MAX_GROUP_SIZE))
      throw std::runtime_error("triple_class_histogram: group " + std::to_string(c) + " has " +
                               std::to_string(count[c]) + " members > " +
                               std::to_string(MAX_GROUP_SIZE));
    offs[static_cast<std::size_t>(c) + 1] = offs[static_cast<std::size_t>(c)] + count[c];
  }
  std::vector<uint32_t> perm(nw);
  {
    uint32_t next[NGROUP];
    for (int c = 0; c < NGROUP; ++c) next[c] = offs[static_cast<std::size_t>(c)];
    for (std::size_t w = 0; w < nw; ++w) perm[next[g[w]]++] = static_cast<uint32_t>(w);
  }
  KISS_CUDA_CHECK(cudaMemcpyAsync(d_perm.p, perm.data(), nw * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));
  KISS_CUDA_CHECK(cudaMemcpyAsync(d_offs.p, offs.data(), (NGROUP + 1) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));

  gather_sorted_kernel<<<(N + TB - 1) / TB, TB, 0, stream>>>(L.packed, d_perm.as<uint32_t>(),
                                                              d_sorted.as<uint32_t>());
  KISS_CUDA_CHECK(cudaGetLastError());

  const unsigned blocks = (nz + ZB - 1) / ZB;
  hist_kernel<ZB><<<blocks, TB, 0, stream>>>(d_sorted.as<uint32_t>(), L.packed,
                                             d_offs.as<uint32_t>(), z0, nz, d_M);
  KISS_CUDA_CHECK(cudaGetLastError());
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace kiss::cuda

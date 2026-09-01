// B2 — quadruple-class histogram kernel (five-point numbers of the scheme).
// See quad_cuda.h and docs/reports/B2.md.
//
// Design (a direct extension of T2.3's triple_stats.cu; same skeleton, the
// host counting sort now uses the 343 cells (class(x,v), class(y,v),
// class(z,v)) instead of 49):
//   1. tri_class_kernel: g[v] = 49*class(x,v) + 7*class(y,v) + class(z,v).
//   2. Host counting sort of v by g[v] -> perm[N], offs[344].
//   3. gather_sorted_kernel: the minimal vectors in group order (SoA).
//   4. hist_kernel<WB>: one 256-thread block per WB consecutive w's (their
//      packed words in registers).  For each of the 343 groups the threads
//      stride over that group's contiguous v's, compute dot(v, w) with 6
//      dp4a per w and add 1 << (9 * class(v, w)) into a packed uint64 (seven
//      9-bit fields).  A thread sees at most ceil(93150 / 256) = 364 < 512
//      v's of one group, so no field overflows (the host enforces
//      group size <= 511 * 256).  After each group the fields are
//      warp-reduced and accumulated into s_hist[w][g * 7 + d] in shared
//      memory (WB * 2401 counters, 38.4 KB at WB = 4); written to d_M at
//      the end.
#include "quad_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kiss::cuda {

namespace {

constexpr int TB = 256;         // threads per block
constexpr int WB = 4;           // w vectors per block (shared: WB * 2401 * 4 B)
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

// g[v] = 49 * class(x, v) + 7 * class(y, v) + class(z, v).
__global__ void tri_class_kernel(const uint32_t* __restrict__ packed, uint32_t x, uint32_t y,
                                 uint32_t z, uint16_t* __restrict__ g) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= N) return;
  int xw[6], yw[6], zw[6], vw[6];
#pragma unroll
  for (int q = 0; q < 6; ++q) {
    xw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + x]);
    yw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + y]);
    zw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + z]);
    vw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + v]);
  }
  g[v] = static_cast<uint16_t>(49 * ip_class_dev(dot6(xw, vw)) + 7 * ip_class_dev(dot6(yw, vw)) +
                               ip_class_dev(dot6(zw, vw)));
}

__global__ void gather_sorted_kernel(const uint32_t* __restrict__ packed,
                                     const uint32_t* __restrict__ perm,
                                     uint32_t* __restrict__ sorted) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  const uint32_t v = perm[t];
#pragma unroll
  for (int q = 0; q < 6; ++q)
    sorted[static_cast<std::size_t>(q) * N + t] = packed[static_cast<std::size_t>(q) * N + v];
}

template <int W>
__global__ void __launch_bounds__(TB)
hist_kernel(const uint32_t* __restrict__ sorted,   // [6][N], v in group order
            const uint32_t* __restrict__ packed,   // [6][N], original order (for w)
            const uint32_t* __restrict__ offs,     // [344] group boundaries in `sorted`
            uint32_t w0, uint32_t nw,
            uint32_t* __restrict__ M) {            // [nw][2401]
  __shared__ uint32_t s_hist[W][NCLASS4];
  __shared__ int s_offs[NGROUP4 + 1];

  for (int t = threadIdx.x; t < W * NCLASS4; t += TB) (&s_hist[0][0])[t] = 0u;
  for (int t = threadIdx.x; t < NGROUP4 + 1; t += TB) s_offs[t] = static_cast<int>(offs[t]);

  const uint32_t wb = w0 + static_cast<uint32_t>(blockIdx.x) * W;
  const uint32_t wend = w0 + nw;
  int ww[W][6];
#pragma unroll
  for (int k = 0; k < W; ++k) {
    const uint32_t w = wb + static_cast<uint32_t>(k);
    const bool valid = w < wend;
#pragma unroll
    for (int q = 0; q < 6; ++q)
      ww[k][q] = valid ? static_cast<int>(packed[static_cast<std::size_t>(q) * N + w]) : 0;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  for (int g = 0; g < NGROUP4; ++g) {
    const int beg = s_offs[g], end = s_offs[g + 1];
    if (beg == end) continue;
    unsigned long long h[W];
#pragma unroll
    for (int k = 0; k < W; ++k) h[k] = 0ull;
    for (int t = beg + static_cast<int>(threadIdx.x); t < end; t += TB) {
      int vv[6];
#pragma unroll
      for (int q = 0; q < 6; ++q) vv[q] = static_cast<int>(sorted[static_cast<std::size_t>(q) * N + t]);
#pragma unroll
      for (int k = 0; k < W; ++k) {
        const int idx = ip_class_dev(dot6(vv, ww[k]));
        h[k] += 1ull << (idx * FIELD_BITS);
      }
    }
#pragma unroll
    for (int k = 0; k < W; ++k) {
#pragma unroll
      for (int d = 0; d < 7; ++d) {
        const unsigned f = static_cast<unsigned>(h[k] >> (d * FIELD_BITS)) & FIELD_MASK;
        const unsigned s = warp_sum(f);
        if (lane == 0 && s) atomicAdd(&s_hist[k][g * 7 + d], s);
      }
    }
  }
  __syncthreads();

  for (int t = threadIdx.x; t < W * NCLASS4; t += TB) {
    const int k = t / NCLASS4, e = t - k * NCLASS4;
    const uint32_t w = wb + static_cast<uint32_t>(k);
    if (w < wend) M[static_cast<std::size_t>(w - w0) * NCLASS4 + e] = s_hist[k][e];
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

void quad_class_histogram(const DeviceLeech& L, uint32_t x, uint32_t y, uint32_t z, uint32_t w0,
                          uint32_t nw, uint32_t* d_M, cudaStream_t stream) {
  if (!L.packed) throw std::runtime_error("quad_class_histogram: DeviceLeech.packed is null");
  if (x >= static_cast<uint32_t>(N) || y >= static_cast<uint32_t>(N) || z >= static_cast<uint32_t>(N))
    throw std::runtime_error("quad_class_histogram: x, y or z out of range");
  if (w0 > static_cast<uint32_t>(N) || nw > static_cast<uint32_t>(N) - w0)
    throw std::runtime_error("quad_class_histogram: w range out of bounds");
  if (nw == 0) return;

  const std::size_t nv = static_cast<std::size_t>(N);
  DeviceBuffer d_g(nv * sizeof(uint16_t));             // uint16[N]
  DeviceBuffer d_perm(nv * sizeof(uint32_t));          // uint32[N]
  DeviceBuffer d_sorted(6 * nv * sizeof(uint32_t));    // uint32[6][N]
  DeviceBuffer d_offs((NGROUP4 + 1) * sizeof(uint32_t));

  tri_class_kernel<<<(N + TB - 1) / TB, TB, 0, stream>>>(L.packed, x, y, z, d_g.as<uint16_t>());
  KISS_CUDA_CHECK(cudaGetLastError());
  std::vector<uint16_t> g(nv);
  KISS_CUDA_CHECK(cudaMemcpyAsync(g.data(), d_g.p, nv * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                  stream));
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));

  uint32_t count[NGROUP4] = {0};
  for (std::size_t v = 0; v < nv; ++v) {
    if (g[v] >= NGROUP4)
      throw std::runtime_error("quad_class_histogram: inner product out of range at v=" +
                               std::to_string(v));
    ++count[g[v]];
  }
  std::vector<uint32_t> offs(NGROUP4 + 1, 0u);
  for (int c = 0; c < NGROUP4; ++c) {
    if (count[c] > static_cast<uint32_t>(MAX_GROUP_SIZE))
      throw std::runtime_error("quad_class_histogram: group " + std::to_string(c) + " has " +
                               std::to_string(count[c]) + " members > " +
                               std::to_string(MAX_GROUP_SIZE));
    offs[static_cast<std::size_t>(c) + 1] = offs[static_cast<std::size_t>(c)] + count[c];
  }
  std::vector<uint32_t> perm(nv);
  {
    uint32_t next[NGROUP4];
    for (int c = 0; c < NGROUP4; ++c) next[c] = offs[static_cast<std::size_t>(c)];
    for (std::size_t v = 0; v < nv; ++v) perm[next[g[v]]++] = static_cast<uint32_t>(v);
  }
  KISS_CUDA_CHECK(cudaMemcpyAsync(d_perm.p, perm.data(), nv * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));
  KISS_CUDA_CHECK(cudaMemcpyAsync(d_offs.p, offs.data(), (NGROUP4 + 1) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));

  gather_sorted_kernel<<<(N + TB - 1) / TB, TB, 0, stream>>>(L.packed, d_perm.as<uint32_t>(),
                                                              d_sorted.as<uint32_t>());
  KISS_CUDA_CHECK(cudaGetLastError());

  const unsigned blocks = (nw + WB - 1) / WB;
  hist_kernel<WB><<<blocks, TB, 0, stream>>>(d_sorted.as<uint32_t>(), L.packed,
                                             d_offs.as<uint32_t>(), w0, nw, d_M);
  KISS_CUDA_CHECK(cudaGetLastError());
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace kiss::cuda

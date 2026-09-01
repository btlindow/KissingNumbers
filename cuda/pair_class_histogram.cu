// T2.2 — pair-class histogram kernel (docs/design.md T2.2, §5.1 variant; see
// docs/reports/T2.2.md).
//
// Design (derived from the T1.5 tightness kernel)
//   1. class_kernel: cls[z] = class(<C[x],C[z]>) for all z (one dp4a pass);
//      class = kiss::ip_class order 0..6 <-> dot -32,-16,-8,0,8,16,32.
//   2. Host counting sort of z by cls[z] -> perm[N], offs[8] (class boundaries).
//   3. gather_sorted_kernel: sorted[w*N + t] = packed[w*N + perm[t]] — the
//      minimal vectors in class order, SoA, so step 4 reads them coalesced.
//   4. hist_kernel<YB>: one block (256 threads) per YB consecutive y's. The
//      6 packed words of each y sit in registers. For each class g of x the
//      threads stride over the z's of that class (contiguous in `sorted`),
//      compute dot(z, y) with 6 dp4a for every y of the block and update a
//      packed 64-bit histogram h[y] += 1 << (9 * class(z, y)): seven 9-bit
//      fields, one per class of (z, y). A thread sees at most
//      ceil(93150 / 256) = 364 < 512 z's of one class, so no field can
//      overflow (the host wrapper enforces max class size <= 511 * 256).
//      After each class the seven fields are warp-reduced (redux.sync on
//      sm_80+) and accumulated into s_hist[y][g*7 + j] in shared memory;
//      at the end the YB matrices are written to d_M.
//   Per (z, y) pair: 6 IDP4A + ~5 integer ops; the whole N x N pass is
//   ~3.9e10 pairs and issue-bound (T1.5's arithmetic bound), i.e. ~0.2-0.5 s.
#include "scheme_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kiss::cuda {

namespace {

constexpr int TB = 256;         // threads per block
constexpr int YB = 8;           // y vectors per block (48 registers of y words)
constexpr int FIELD_BITS = 9;   // per-class counter width inside the packed uint64
constexpr unsigned FIELD_MASK = (1u << FIELD_BITS) - 1u;   // 511
constexpr int MAX_CLASS_SIZE = static_cast<int>(FIELD_MASK) * TB;  // 130816 > 93150

__device__ __forceinline__ int dot6(const int (&a)[6], const int (&b)[6]) {
  int acc = __dp4a(a[0], b[0], 0);
  acc = __dp4a(a[1], b[1], acc);
  acc = __dp4a(a[2], b[2], acc);
  acc = __dp4a(a[3], b[3], acc);
  acc = __dp4a(a[4], b[4], acc);
  acc = __dp4a(a[5], b[5], acc);
  return acc;
}

// Class index of an inner product d in {-32,-16,-8,0,8,16,32} -> 0..6, the
// canonical order of kiss::ip_class. The dots are NOT uniformly spaced, so
// (d+32)/8 = {0,2,3,4,5,6,8} has to be compacted: t - [t>1] - [t>7].
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

// cls[z] = class of (x, z).
__global__ void class_kernel(const uint32_t* __restrict__ packed, uint32_t x,
                             uint8_t* __restrict__ cls) {
  const int z = blockIdx.x * blockDim.x + threadIdx.x;
  if (z >= N) return;
  int xw[6], zw[6];
#pragma unroll
  for (int q = 0; q < 6; ++q) {
    xw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + x]);
    zw[q] = static_cast<int>(packed[static_cast<std::size_t>(q) * N + z]);
  }
  cls[z] = static_cast<uint8_t>(ip_class_dev(dot6(xw, zw)));
}

// sorted[w*N + t] = packed[w*N + perm[t]].
__global__ void gather_sorted_kernel(const uint32_t* __restrict__ packed,
                                     const uint32_t* __restrict__ perm,
                                     uint32_t* __restrict__ sorted) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  const uint32_t z = perm[t];
#pragma unroll
  for (int q = 0; q < 6; ++q)
    sorted[static_cast<std::size_t>(q) * N + t] = packed[static_cast<std::size_t>(q) * N + z];
}

template <int Y>
__global__ void __launch_bounds__(TB)
hist_kernel(const uint32_t* __restrict__ sorted,   // [6][N], z in class order
            const uint32_t* __restrict__ packed,   // [6][N], original order (for y)
            const uint32_t* __restrict__ offs,     // [8] class boundaries in `sorted`
            uint32_t y0, uint32_t ny,
            uint32_t* __restrict__ M) {            // [ny][49]
  __shared__ uint32_t s_hist[Y][NCLASS2];
  __shared__ int s_offs[NCLASS + 1];

  for (int t = threadIdx.x; t < Y * NCLASS2; t += TB) (&s_hist[0][0])[t] = 0u;
  if (threadIdx.x < NCLASS + 1) s_offs[threadIdx.x] = static_cast<int>(offs[threadIdx.x]);

  const uint32_t yb = y0 + static_cast<uint32_t>(blockIdx.x) * Y;
  const uint32_t yend = y0 + ny;
  int yw[Y][6];
#pragma unroll
  for (int k = 0; k < Y; ++k) {
    const uint32_t y = yb + static_cast<uint32_t>(k);
    const bool valid = y < yend;
#pragma unroll
    for (int q = 0; q < 6; ++q)
      yw[k][q] = valid ? static_cast<int>(packed[static_cast<std::size_t>(q) * N + y]) : 0;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  for (int g = 0; g < NCLASS; ++g) {
    unsigned long long h[Y];
#pragma unroll
    for (int k = 0; k < Y; ++k) h[k] = 0ull;
    const int beg = s_offs[g], end = s_offs[g + 1];
    for (int t = beg + static_cast<int>(threadIdx.x); t < end; t += TB) {
      int zw[6];
#pragma unroll
      for (int q = 0; q < 6; ++q) zw[q] = static_cast<int>(sorted[static_cast<std::size_t>(q) * N + t]);
#pragma unroll
      for (int k = 0; k < Y; ++k) {
        const int idx = ip_class_dev(dot6(zw, yw[k]));  // dot in {-32,-16,-8,0,8,16,32} -> 0..6
        h[k] += 1ull << (idx * FIELD_BITS);
      }
    }
    // Block reduction of the seven fields of every y (all 32 lanes take part).
#pragma unroll
    for (int k = 0; k < Y; ++k) {
#pragma unroll
      for (int j = 0; j < NCLASS; ++j) {
        const unsigned f = static_cast<unsigned>(h[k] >> (j * FIELD_BITS)) & FIELD_MASK;
        const unsigned s = warp_sum(f);
        if (lane == 0 && s) atomicAdd(&s_hist[k][g * NCLASS + j], s);
      }
    }
  }
  __syncthreads();

  for (int t = threadIdx.x; t < Y * NCLASS2; t += TB) {
    const int k = t / NCLASS2, e = t - k * NCLASS2;
    const uint32_t y = yb + static_cast<uint32_t>(k);
    if (y < yend) M[static_cast<std::size_t>(y - y0) * NCLASS2 + e] = s_hist[k][e];
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

void pair_class_histogram(const DeviceLeech& L, uint32_t x, uint32_t y0, uint32_t ny,
                          uint32_t* d_M, cudaStream_t stream) {
  if (!L.packed) throw std::runtime_error("pair_class_histogram: DeviceLeech.packed is null");
  if (x >= static_cast<uint32_t>(N)) throw std::runtime_error("pair_class_histogram: x out of range");
  if (y0 > static_cast<uint32_t>(N) || ny > static_cast<uint32_t>(N) - y0)
    throw std::runtime_error("pair_class_histogram: y range out of bounds");
  if (ny == 0) return;

  const std::size_t nz = static_cast<std::size_t>(N);
  DeviceBuffer d_cls(nz);                              // uint8[N]
  DeviceBuffer d_perm(nz * sizeof(uint32_t));          // uint32[N]
  DeviceBuffer d_sorted(6 * nz * sizeof(uint32_t));    // uint32[6][N]
  DeviceBuffer d_offs((NCLASS + 1) * sizeof(uint32_t));

  // 1. classes of (x, z)
  class_kernel<<<(N + TB - 1) / TB, TB, 0, stream>>>(L.packed, x, d_cls.as<uint8_t>());
  KISS_CUDA_CHECK(cudaGetLastError());
  std::vector<uint8_t> cls(nz);
  KISS_CUDA_CHECK(cudaMemcpyAsync(cls.data(), d_cls.p, nz, cudaMemcpyDeviceToHost, stream));
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));

  // 2. counting sort (stable) by class
  uint32_t count[NCLASS] = {0};
  for (std::size_t z = 0; z < nz; ++z) {
    if (cls[z] >= NCLASS)
      throw std::runtime_error("pair_class_histogram: inner product out of range at z=" +
                               std::to_string(z));
    ++count[cls[z]];
  }
  std::vector<uint32_t> offs(NCLASS + 1, 0u);
  for (int c = 0; c < NCLASS; ++c) {
    if (count[c] > static_cast<uint32_t>(MAX_CLASS_SIZE))
      throw std::runtime_error("pair_class_histogram: class " + std::to_string(c) + " has " +
                               std::to_string(count[c]) + " members > " +
                               std::to_string(MAX_CLASS_SIZE));
    offs[static_cast<std::size_t>(c) + 1] = offs[static_cast<std::size_t>(c)] + count[c];
  }
  std::vector<uint32_t> perm(nz);
  {
    uint32_t next[NCLASS];
    for (int c = 0; c < NCLASS; ++c) next[c] = offs[static_cast<std::size_t>(c)];
    for (std::size_t z = 0; z < nz; ++z) perm[next[cls[z]]++] = static_cast<uint32_t>(z);
  }
  KISS_CUDA_CHECK(cudaMemcpyAsync(d_perm.p, perm.data(), nz * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));
  KISS_CUDA_CHECK(cudaMemcpyAsync(d_offs.p, offs.data(), (NCLASS + 1) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));

  // 3. gather in class order
  gather_sorted_kernel<<<(N + TB - 1) / TB, TB, 0, stream>>>(L.packed, d_perm.as<uint32_t>(),
                                                              d_sorted.as<uint32_t>());
  KISS_CUDA_CHECK(cudaGetLastError());

  // 4. histograms
  const unsigned blocks = (ny + YB - 1) / YB;
  hist_kernel<YB><<<blocks, TB, 0, stream>>>(d_sorted.as<uint32_t>(), L.packed,
                                             d_offs.as<uint32_t>(), y0, ny, d_M);
  KISS_CUDA_CHECK(cudaGetLastError());
  // The scratch buffers are freed by the DeviceBuffer destructors; synchronise
  // first so that the kernel has finished using them.
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace kiss::cuda

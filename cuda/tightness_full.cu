// T1.5 — fused tightness kernel (docs/design.md §5.1).
//
// Design
//   grid  = (ceil(N / (256*VPT)), Bchunk), block = 256 threads.
//   Each block first copies the chain's S as packed words (6 uint32 per
//   member, AoS, <= 24 KB) into shared memory, then each thread owns VPT
//   candidate vertices (v, v+256, ...) whose 6 packed words it keeps in
//   registers (SoA global layout → the 32 lanes of a warp read 6 coalesced
//   128-byte lines). Loop over S: acc = dp4a(w0,s0,0) … dp4a(w5,s5,acc),
//   cnt += (acc == 16). The S loop is unrolled by 4 members; the shared words
//   of 2 consecutive members (48 bytes) are fetched with 3 LDS.128 broadcasts.
//   The N x |S| matrix is never materialised; output is tight[b][v] (uint16).
//
//   The gather of S's packed words is done once per chain by a small
//   pre-kernel into a stream-ordered scratch buffer ([Bchunk][SMAX][6]); the
//   main kernel then stages S with coalesced loads. Gathering directly from
//   `packed` inside every block (6·|S| scattered 4-byte loads per block, each
//   pulling a 32-byte L2 sector) measured ~12% slower at B=64, |S|=500
//   (15.1 ms vs 13.5 ms with VPT=2); see docs/reports/T1.5.md.
//
//   VPT = 4 candidates per thread: 54 registers, no spills, 4 blocks/SM
//   (smem-limited: 4 x 24 KB). Measured best-of-10 at B=64, |S|=500:
//   VPT=1 15.7 ms, VPT=2 12.7-14.4 ms, VPT=4 12.0-13.6 ms (noisy shared box).
#include "kiss_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace kiss::cuda {

namespace {

constexpr int TB = 256;                       // threads per block
constexpr int VPT = 4;                        // candidates per thread (see header comment)
constexpr int B_CHUNK = 256;                  // chains per launch (scratch = 6 MB)
constexpr int S_WORDS = PACKED_WORDS * SMAX;  // 6144 words = 24 KB per chain

// Gather S_b's packed words: Sp[b][i*6 + w] = packed[w*N + S[b][i]].
__global__ void gather_S_kernel(const uint32_t* __restrict__ packed,
                                const uint32_t* __restrict__ S,
                                const uint32_t* __restrict__ Ssize,
                                uint32_t* __restrict__ Sp) {
  const int b = blockIdx.x;
  const int n = min(static_cast<int>(Ssize[b]), SMAX);
  const uint32_t* Sb = S + static_cast<std::size_t>(b) * SMAX;
  uint32_t* out = Sp + static_cast<std::size_t>(b) * S_WORDS;
  for (int j = threadIdx.x; j < PACKED_WORDS * n; j += blockDim.x) {
    const int i = j / PACKED_WORDS;
    const int w = j - i * PACKED_WORDS;
    out[j] = packed[static_cast<std::size_t>(w) * N + Sb[i]];
  }
}

__device__ __forceinline__ int dot6(const int (&w)[6], int s0, int s1, int s2, int s3, int s4,
                                    int s5) {
  int acc = __dp4a(w[0], s0, 0);
  acc = __dp4a(w[1], s1, acc);
  acc = __dp4a(w[2], s2, acc);
  acc = __dp4a(w[3], s3, acc);
  acc = __dp4a(w[4], s4, acc);
  acc = __dp4a(w[5], s5, acc);
  return acc;
}

template <int V>
__global__ void __launch_bounds__(TB)
tightness_kernel(const uint32_t* __restrict__ packed,
                 const uint32_t* __restrict__ Sp,     // [B][SMAX*6] gathered words
                 const uint32_t* __restrict__ Ssize,  // [B]
                 uint16_t* __restrict__ tight) {      // [B][N]
  __shared__ __align__(16) uint32_t sS[S_WORDS];

  const int b = blockIdx.y;
  const int n = min(static_cast<int>(Ssize[b]), SMAX);

  // Stage S_b (6n words, coalesced).
  {
    const uint32_t* src = Sp + static_cast<std::size_t>(b) * S_WORDS;
    for (int j = threadIdx.x; j < PACKED_WORDS * n; j += TB) sS[j] = src[j];
  }
  __syncthreads();

  // Candidate words in registers (SoA → coalesced across the warp).
  int w[V][6];
  int cnt[V];
  const int v0 = blockIdx.x * (TB * V) + threadIdx.x;
#pragma unroll
  for (int k = 0; k < V; ++k) {
    const int v = v0 + k * TB;
    cnt[k] = 0;
#pragma unroll
    for (int q = 0; q < 6; ++q)
      w[k][q] = (v < N) ? static_cast<int>(packed[static_cast<std::size_t>(q) * N + v]) : 0;
  }

  // Main loop: 4 members per iteration = 2 member-pairs = 6 uint4 shared loads.
  const uint4* s4 = reinterpret_cast<const uint4*>(sS);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint4* p = s4 + (i / 2) * 3;
    const uint4 a0 = p[0], a1 = p[1], a2 = p[2], a3 = p[3], a4 = p[4], a5 = p[5];
#pragma unroll
    for (int k = 0; k < V; ++k) {
      cnt[k] += (dot6(w[k], a0.x, a0.y, a0.z, a0.w, a1.x, a1.y) == 16);
      cnt[k] += (dot6(w[k], a1.z, a1.w, a2.x, a2.y, a2.z, a2.w) == 16);
      cnt[k] += (dot6(w[k], a3.x, a3.y, a3.z, a3.w, a4.x, a4.y) == 16);
      cnt[k] += (dot6(w[k], a4.z, a4.w, a5.x, a5.y, a5.z, a5.w) == 16);
    }
  }
  // Tail (n mod 4 members).
  for (; i < n; ++i) {
    const uint32_t* s = sS + i * PACKED_WORDS;
    const int s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], s4v = s[4], s5 = s[5];
#pragma unroll
    for (int k = 0; k < V; ++k) cnt[k] += (dot6(w[k], s0, s1, s2, s3, s4v, s5) == 16);
  }

#pragma unroll
  for (int k = 0; k < V; ++k) {
    const int v = v0 + k * TB;
    if (v < N) tight[static_cast<std::size_t>(b) * N + v] = static_cast<uint16_t>(cnt[k]);
  }
}

__global__ void count_free_kernel(const uint16_t* __restrict__ tight,
                                  const uint32_t* __restrict__ inS,  // may be nullptr
                                  uint32_t* __restrict__ out) {
  __shared__ uint32_t warp_sum[32];
  const int b = blockIdx.x;
  const uint16_t* t = tight + static_cast<std::size_t>(b) * N;
  const uint32_t* bits = inS ? inS + static_cast<std::size_t>(b) * INS_WORDS : nullptr;
  uint32_t c = 0;
  for (int v = threadIdx.x; v < N; v += blockDim.x) {
    const bool in = bits ? ((bits[v >> 5] >> (v & 31)) & 1u) : false;
    c += (t[v] == 0 && !in);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_sum[wid] = c;
  __syncthreads();
  if (wid == 0) {
    const int nw = (blockDim.x + 31) >> 5;
    c = (lane < nw) ? warp_sum[lane] : 0u;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if (lane == 0) out[b] = c;
  }
}

template <int V>
void launch_tightness(const uint32_t* packed, const uint32_t* Sp, const uint32_t* Ssize, int Bc,
                      uint16_t* tight, cudaStream_t stream) {
  const dim3 grid((N + TB * V - 1) / (TB * V), Bc);
  tightness_kernel<V><<<grid, TB, 0, stream>>>(packed, Sp, Ssize, tight);
  KISS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void tightness_full(const DeviceLeech& L, const uint32_t* d_S, const uint32_t* d_Ssize, int B,
                    uint16_t* d_tight, cudaStream_t stream) {
  if (B <= 0) return;
  if (!L.packed) throw std::runtime_error("tightness_full: DeviceLeech.packed is null");
  const int Bc = std::min(B, B_CHUNK);
  uint32_t* Sp = nullptr;
  KISS_CUDA_CHECK(cudaMallocAsync(&Sp, static_cast<std::size_t>(Bc) * S_WORDS * sizeof(uint32_t),
                                  stream));
  try {
    for (int b0 = 0; b0 < B; b0 += B_CHUNK) {
      const int nb = std::min(B_CHUNK, B - b0);
      gather_S_kernel<<<nb, TB, 0, stream>>>(L.packed, d_S + static_cast<std::size_t>(b0) * SMAX,
                                             d_Ssize + b0, Sp);
      KISS_CUDA_CHECK(cudaGetLastError());
      launch_tightness<VPT>(L.packed, Sp, d_Ssize + b0, nb,
                            d_tight + static_cast<std::size_t>(b0) * N, stream);
    }
  } catch (...) {
    cudaFreeAsync(Sp, stream);
    throw;
  }
  KISS_CUDA_CHECK(cudaFreeAsync(Sp, stream));
}

void count_free(const uint16_t* d_tight, const uint32_t* d_inS_bits, int B, uint32_t* d_out,
                cudaStream_t stream) {
  if (B <= 0) return;
  count_free_kernel<<<B, 1024, 0, stream>>>(d_tight, d_inS_bits, d_out);
  KISS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kiss::cuda

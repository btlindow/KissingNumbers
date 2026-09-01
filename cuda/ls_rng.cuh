// T3.2b — tiny per-lane PRNG for the ILS loop kernel.
//
// T3.2a keeps one Philox state per chain (LSState::rng, used by its own
// kernels). The loop kernel wants 32 independent streams per chain (every lane
// samples candidates in parallel), so it carries its own state: one 64-bit
// xorshift64* generator per lane, seeded by splitmix64 from (seed, chain, lane).
// 256 bytes per chain, 2 registers per lane in the kernel.
#pragma once

#include <cstdint>

namespace kiss::cuda {

__host__ __device__ inline uint64_t ls_splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Deterministic non-zero seed for (seed, stream).
__host__ __device__ inline uint64_t ls_rng_seed(uint64_t seed, uint64_t stream) {
  uint64_t s = ls_splitmix64(ls_splitmix64(seed) ^ (stream * 0xD1B54A32D192ED03ull + 0x8CB92BA72F3D8DD7ull));
  return s ? s : 0x9E3779B97F4A7C15ull;
}

#ifdef __CUDACC__
// xorshift64*: 32 high bits of the multiplied state.
__device__ __forceinline__ uint32_t ls_rng_next(uint64_t& s) {
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32);
}

// Uniform in [0, n) (n > 0), multiply-shift.
__device__ __forceinline__ uint32_t ls_rng_below(uint64_t& s, uint32_t n) {
  return static_cast<uint32_t>((static_cast<uint64_t>(ls_rng_next(s)) * n) >> 32);
}

// Warp-uniform draw: every lane advances its own stream, lane 0's value is broadcast.
__device__ __forceinline__ uint32_t ls_rng_uniform_warp(uint64_t& s) {
  return __shfl_sync(0xFFFFFFFFu, ls_rng_next(s), 0);
}
__device__ __forceinline__ uint32_t ls_rng_below_warp(uint64_t& s, uint32_t n) {
  return __shfl_sync(0xFFFFFFFFu, ls_rng_below(s, n), 0);
}
#endif

}  // namespace kiss::cuda

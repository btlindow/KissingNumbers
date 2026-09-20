// Bit primitives used by the bitset code paths (frames, swaps, orbits, cliques).
//
// The project was written against gcc and called `__builtin_popcountll` and
// friends directly. MSVC has no such builtins, so the four that are actually
// used are collected here behind one name each. Semantics match the gcc
// builtins exactly, including the precondition that `ctz64` and `clz32` are
// **undefined for x == 0** — callers already guarantee a nonzero argument
// (docs/design.md §2.4: bitset scans test the word before scanning it).
//
// `atomic_or64_relaxed` is here for the same reason: the parallel BitGraph
// build in src/family.cpp ORs into a word another thread owns, and gcc spelled
// that `__atomic_fetch_or(..., __ATOMIC_RELAXED)`.
//
// Host-only. Device code uses the CUDA intrinsics (`__popcll`, `__ffsll`)
// directly; nothing here is marked `__device__`.
#pragma once

#include <atomic>
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace kiss {

#if defined(__GNUC__) || defined(__clang__)

inline int popcount32(uint32_t x) { return __builtin_popcount(x); }
inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
inline int clz32(uint32_t x) { return __builtin_clz(x); }

#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))

// `__popcnt` lowers to the POPCNT instruction unconditionally — MSVC does not
// gate it behind an architecture flag. Every x86-64 part this project can run
// on (the GPU pins it to a machine with an Ampere card) has SSE4.2.
inline int popcount32(uint32_t x) { return static_cast<int>(__popcnt(x)); }
inline int popcount64(uint64_t x) { return static_cast<int>(__popcnt64(x)); }

inline int ctz64(uint64_t x) {
  unsigned long i;  // NOLINT(runtime/int) — _BitScanForward64's out parameter
  _BitScanForward64(&i, x);
  return static_cast<int>(i);
}

inline int clz32(uint32_t x) {
  unsigned long i;  // NOLINT(runtime/int)
  _BitScanReverse(&i, x);
  return 31 - static_cast<int>(i);
}

#else

// Portable fallback (32-bit MSVC, or any other compiler). Correct, not fast.
inline int popcount32(uint32_t x) {
  int c = 0;
  for (; x; x &= x - 1) ++c;
  return c;
}
inline int popcount64(uint64_t x) {
  int c = 0;
  for (; x; x &= x - 1) ++c;
  return c;
}
inline int ctz64(uint64_t x) {
  int c = 0;
  for (; (x & 1u) == 0; x >>= 1) ++c;
  return c;
}
inline int clz32(uint32_t x) {
  int c = 0;
  for (; (x & 0x80000000u) == 0; x <<= 1) ++c;
  return c;
}

#endif

// Bitwise OR into *p, relaxed. Only the atomicity of the read-modify-write is
// required; the bits are independent, so no ordering is needed between them.
inline void atomic_or64_relaxed(uint64_t* p, uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
  __atomic_fetch_or(p, v, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
  // MSVC has no relaxed form; _InterlockedOr64 is seq_cst, which is stronger
  // than needed and therefore still correct.
  ::_InterlockedOr64(reinterpret_cast<volatile long long*>(p), static_cast<long long>(v));
#else
  reinterpret_cast<std::atomic<uint64_t>*>(p)->fetch_or(v, std::memory_order_relaxed);
#endif
}

}  // namespace kiss

// T2.3 — triple-class histogram kernel: the four-point numbers of the
// inner-product scheme on the Leech minimal vectors (docs/design.md T2.3,
// docs/reports/T2.3.md).  Companion to scheme_cuda.h (T2.2, not edited here).
//
// Inner-product classes are the canonical ones of kiss::ip_class:
//   class c = 0..6  <->  dot = -32,-16,-8,0,8,16,32   (class 6 = identity).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kiss/types.h"
#include "kiss_cuda.h"

namespace kiss::cuda {

constexpr int NCLASS3 = 7 * 7 * 7;   // entries of one 7x7x7 triple-class histogram

// ---------------------------------------------------------------------------
// Triple-class histogram: for fixed x, y and every z in [z0, z0 + nz),
//
//   d_M[(z - z0) * 343 + a * 49 + b * 7 + c]
//        = #{ w in C : class(x, w) = a, class(y, w) = b, class(z, w) = c }.
//
// For z with class(x, z) = j and class(y, z) = k this is the four-point number
// q[i][j][k][a][b][c] (i = class(x, y)) IF it depends on z only through
// (j, k) — "triple regularity" of the scheme with respect to the base pair
// (x, y); tools/triple_stats.cpp checks that over all z.
//
// Work: nz * N inner products (6 dp4a each); the full N x N pass is ~3.9e10
// dots, ~0.2-0.4 s on the RTX 3070 Laptop.  Output for all z is
// N * 343 * 4 bytes = 270 MB of device memory (caller-owned).
//
// Synchronous.  The w vectors are counting-sorted on the host by the pair
// (class(x,w), class(y,w)) (49 groups) so that the kernel streams each group
// contiguously; ~5.5 MB of device scratch is cudaMalloc'ed and freed inside.
// Throws std::runtime_error on any CUDA error, if x, y >= N, if the z range
// is out of bounds, or if a group has more than 511 * 256 members (cannot
// happen: the largest group is 93150, for y = +-x).
// ---------------------------------------------------------------------------
void triple_class_histogram(const DeviceLeech& L, uint32_t x, uint32_t y, uint32_t z0, uint32_t nz,
                            uint32_t* d_M, cudaStream_t stream = 0);

// Convenience: all z (z0 = 0, nz = N); d_M is uint32[N][343].
inline void triple_class_histogram_all(const DeviceLeech& L, uint32_t x, uint32_t y, uint32_t* d_M,
                                       cudaStream_t stream = 0) {
  triple_class_histogram(L, x, y, 0u, static_cast<uint32_t>(N), d_M, stream);
}

}  // namespace kiss::cuda

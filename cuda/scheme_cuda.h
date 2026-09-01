// T2.2 — pair-class histogram kernel for the association-scheme numbers
// (docs/design.md T2.2, §5.1 "variant for T2.2"; docs/reports/T2.2.md).
//
// Companion to kiss_cuda.h (which is owned by T1.5 and not edited here).
//
// Inner-product classes are the canonical ones of docs/design.md §2.2 / kiss::ip_class:
//   class index c = 0..6  <->  dot = -32,-16,-8,0,8,16,32  (NOT (dot+32)/8:
//   the dots are not uniformly spaced; see ip_class_dev in the .cu).
// Class 6 (dot 32) is the identity relation (z = y); class 0 is the antipode.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kiss/types.h"
#include "kiss_cuda.h"

namespace kiss::cuda {

constexpr int NCLASS = 7;                  // inner-product classes
constexpr int NCLASS2 = NCLASS * NCLASS;   // entries of one 7x7 pair-class matrix

// ---------------------------------------------------------------------------
// Pair-class histogram: for a fixed x and every y in [y0, y0 + ny),
//
//   d_M[(y - y0) * 49 + i * 7 + j] = #{ z in C : class(x, z) = i, class(z, y) = j }.
//
// If the inner-product relations form an association scheme, this matrix
// depends only on class(x, y) = k and equals the intersection numbers
// p^k_{ij}; the host (tools/scheme_numbers.cpp) checks that.
//
// Work: ny * N inner products (6 dp4a each); all of N x N is ~3.9e10 dots,
// ~0.2-0.5 s on the RTX 3070 Laptop.
//
// Synchronous: the z vectors are counting-sorted by class(x, z) on the host
// (one 196 KB device->host copy) so that the kernel streams them in class
// order with coalesced loads; ~5 MB of device scratch is cudaMalloc'ed and
// freed inside. Throws std::runtime_error on any CUDA error or if x >= N,
// y0 + ny > N, or a class of x has more than 511 * 256 members (cannot
// happen for the Leech minimal vectors: the largest class has 93150).
// ---------------------------------------------------------------------------
void pair_class_histogram(const DeviceLeech& L, uint32_t x, uint32_t y0, uint32_t ny,
                          uint32_t* d_M, cudaStream_t stream = 0);

// Convenience: all y (y0 = 0, ny = N); d_M is uint32[N][49].
inline void pair_class_histogram_all(const DeviceLeech& L, uint32_t x, uint32_t* d_M,
                                     cudaStream_t stream = 0) {
  pair_class_histogram(L, x, 0u, static_cast<uint32_t>(N), d_M, stream);
}

}  // namespace kiss::cuda

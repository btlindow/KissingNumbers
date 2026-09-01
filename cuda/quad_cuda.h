// B2 — quadruple-class histogram kernel: the five-point numbers of the
// inner-product scheme on the Leech minimal vectors (docs/reports/B2.md).
// Direct extension of triple_cuda.h (T2.3) from two fixed base points to
// three: where triple_stats fixes (x, y) and histograms every z over w,
// quad_stats fixes (x, y, z) and histograms every w over v.
//
// Inner-product classes are the canonical ones of kiss::ip_class:
//   class c = 0..6  <->  dot = -32,-16,-8,0,8,16,32   (class 6 = identity).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kiss/types.h"
#include "kiss_cuda.h"

namespace kiss::cuda {

constexpr int NGROUP4 = 7 * 7 * 7;      // (class(x,v), class(y,v), class(z,v)) cells
constexpr int NCLASS4 = NGROUP4 * 7;    // entries of one 7x7x7x7 histogram

// ---------------------------------------------------------------------------
// Quadruple-class histogram: for fixed x, y, z and every w in [w0, w0 + nw),
//
//   d_M[(w - w0) * 2401 + (a * 49 + b * 7 + c) * 7 + d]
//     = #{ v in C : class(x,v) = a, class(y,v) = b, class(z,v) = c, class(w,v) = d }.
//
// H_w is invariant under Stab_{Co_0}(x, y, z), so the number of distinct H_w
// (paired with w's own cell (class(x,w), class(y,w), class(z,w)), which the
// v = w term of H_w already determines) is a LOWER bound on the number of
// orbits of Stab(x, y, z) on C.  Summed over one z per Stab(x,y)-orbit this
// lower-bounds dim T(x,y) = #orbitals of Stab(x,y) — the block dimension a
// four-point SDP in the T2.3 style would need.  tools/quad_stats.cpp computes
// the matching upper bound from explicit stabiliser elements.
//
// Work: nw * N inner products (6 dp4a each); a full N x N pass is ~3.9e10
// dots, ~0.3 s on the RTX 3070 Laptop plus the (now 343-group) reduction.
// Output is nw * 2401 * 4 bytes of device memory (caller-owned); callers
// chunk w (full N would need 1.9 GB).
//
// Synchronous.  The v vectors are counting-sorted on the host by
// (class(x,v), class(y,v), class(z,v)) (343 groups) so the kernel streams
// each group contiguously; ~5.5 MB of device scratch is cudaMalloc'ed and
// freed inside.  Throws std::runtime_error on any CUDA error, if x, y or z
// >= N, if the w range is out of bounds, or if a group exceeds 511 * 256
// members (cannot happen: the largest possible group is 93150).
// ---------------------------------------------------------------------------
void quad_class_histogram(const DeviceLeech& L, uint32_t x, uint32_t y, uint32_t z, uint32_t w0,
                          uint32_t nw, uint32_t* d_M, cudaStream_t stream = 0);

}  // namespace kiss::cuda

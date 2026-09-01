// T1.2 — shared scalar/vector types for the Leech-subset project (docs/design.md §2.3).
//
// Coordinates are integers in the √8 scaling of README §1.2: every minimal
// vector has squared norm 32, and inner products between distinct minimal
// vectors lie in {-32,-16,-8,0,8,16}. All coordinates fit in int8.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kiss {

constexpr int N = 196560;   // number of Leech minimal vectors (kissing number of Λ24)
constexpr int DIM = 24;     // ambient dimension
constexpr int DEG = 4600;   // neighbours of a vector at inner product 16 (cos = 1/2)

using Vec = std::array<int8_t, DIM>;

// Exact integer inner product (|result| ≤ 24·127² < 2^31).
inline int dot(const Vec& a, const Vec& b) {
  int s = 0;
  for (int k = 0; k < DIM; ++k) s += static_cast<int>(a[static_cast<std::size_t>(k)]) *
                                     static_cast<int>(b[static_cast<std::size_t>(k)]);
  return s;
}

// Squared norm ⟨a,a⟩.
inline int norm2(const Vec& a) { return dot(a, a); }

// Coordinate-wise negation.
inline Vec negate(const Vec& a) {
  Vec r;
  for (int k = 0; k < DIM; ++k) r[static_cast<std::size_t>(k)] =
      static_cast<int8_t>(-a[static_cast<std::size_t>(k)]);
  return r;
}

// Inner-product classes in canonical order (docs/design.md §2.2): {-32,-16,-8,0,8,16,32}
// ↔ class index 0..6. Returns -1 for any other value.
inline int ip_class(int ip) {
  switch (ip) {
    case -32: return 0;
    case -16: return 1;
    case -8:  return 2;
    case 0:   return 3;
    case 8:   return 4;
    case 16:  return 5;
    case 32:  return 6;
    default:  return -1;
  }
}

}  // namespace kiss

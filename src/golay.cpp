// T1.1 — extended binary Golay code G24. See include/kiss/golay.h.
//
// Written independently of python/kiss_ref/golay.py (docs/design.md §2.4: the two halves
// of T1.1 must not share code); the only cross-check is the sorted-mask
// fingerprint compared in tests/test_golay.cpp.
#include "kiss/golay.h"

#include <algorithm>

#include "kiss/bits.h"

namespace kiss {

namespace {

// Carry-less product a(x)·b(x) over GF(2). Inputs are bit-mask polynomials;
// deg a + deg b must be < 32 (here <= 11 + 11 = 22).
inline uint32_t clmul(uint32_t a, uint32_t b) {
  uint32_t r = 0;
  for (int i = 0; i < 32 && (b >> i); ++i) {
    if ((b >> i) & 1u) r ^= a << i;
  }
  return r;
}

std::vector<uint32_t> build_codewords() {
  std::vector<uint32_t> words;
  words.reserve(4096);
  for (uint32_t m = 0; m < 4096; ++m) {
    // m(x)·g(x): deg <= 22, so it already lives in coordinates 0..22 with no
    // reduction mod x^23 - 1 needed.
    uint32_t c = clmul(m, GOLAY_GENERATOR);
    // Overall parity bit in coordinate 23 makes every weight even.
    if (popcount32(c) & 1) c |= 1u << 23;
    words.push_back(c);
  }
  std::sort(words.begin(), words.end());
  return words;
}

const std::vector<uint32_t>& cached_codewords() {
  static const std::vector<uint32_t> words = build_codewords();
  return words;
}

// 2^24-bit membership table: word w>>5, bit w&31.
const std::vector<uint32_t>& membership_table() {
  static const std::vector<uint32_t> table = [] {
    std::vector<uint32_t> t((1u << 24) / 32, 0u);
    for (uint32_t w : cached_codewords()) t[w >> 5] |= 1u << (w & 31u);
    return t;
  }();
  return table;
}

}  // namespace

std::vector<uint32_t> golay_codewords() { return cached_codewords(); }

std::array<int, 25> golay_weight_distribution(const std::vector<uint32_t>& words) {
  std::array<int, 25> dist{};
  for (uint32_t w : words) {
    int k = popcount32(w & 0xFFFFFFu);
    ++dist[static_cast<std::size_t>(k)];
  }
  return dist;
}

bool golay_is_codeword(uint32_t mask) {
  if (mask >> 24) return false;
  const auto& t = membership_table();
  return (t[mask >> 5] >> (mask & 31u)) & 1u;
}

std::vector<uint32_t> golay_octads() {
  std::vector<uint32_t> oct;
  oct.reserve(759);
  for (uint32_t w : cached_codewords()) {
    if (popcount32(w) == 8) oct.push_back(w);
  }
  return oct;  // already ascending since the source is sorted
}

}  // namespace kiss

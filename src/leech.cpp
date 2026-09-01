// T1.2 — Leech minimal vectors, canonical order, membership test.
// See include/kiss/leech.h. Written independently of python/kiss_ref/leech.py.
#include "kiss/leech.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kiss/golay.h"
#include "kiss/io.h"

namespace kiss {

namespace {

constexpr std::size_t kN = static_cast<std::size_t>(N);
constexpr std::size_t kDim = static_cast<std::size_t>(DIM);

inline int popcount32(uint32_t x) { return __builtin_popcount(x); }

// Shape (a): ±2 on the octad coordinates, even number of minus signs.
void gen_octad_shape(std::vector<Vec>& out) {
  for (uint32_t oct : golay_octads()) {
    int pos[8];
    int k = 0;
    for (int i = 0; i < DIM; ++i)
      if ((oct >> i) & 1u) pos[k++] = i;
    // Signs of coordinates pos[0..6] are free; pos[7] fixed by even parity.
    for (uint32_t s = 0; s < 128; ++s) {
      Vec v{};
      int minus = 0;
      for (int j = 0; j < 7; ++j) {
        bool neg = (s >> j) & 1u;
        v[static_cast<std::size_t>(pos[j])] = neg ? -2 : 2;
        minus += neg;
      }
      v[static_cast<std::size_t>(pos[7])] = (minus & 1) ? -2 : 2;
      out.push_back(v);
    }
  }
}

// Shape (b): (−3 at i, +1 elsewhere), then negate the coordinates in c.
// See the header / report for why this convention satisfies the membership
// test as-is (all entries ≡ 1 mod 4 before negation; sum = 20 ≡ 4 mod 8).
void gen_three_one_shape(std::vector<Vec>& out) {
  const std::vector<uint32_t> words = golay_codewords();
  for (int i = 0; i < DIM; ++i) {
    for (uint32_t c : words) {
      Vec v;
      for (int j = 0; j < DIM; ++j) {
        int x = (j == i) ? -3 : 1;
        if ((c >> j) & 1u) x = -x;
        v[static_cast<std::size_t>(j)] = static_cast<int8_t>(x);
      }
      out.push_back(v);
    }
  }
}

// Shape (c): (±4, ±4, 0^22) on every pair of coordinates.
void gen_four_four_shape(std::vector<Vec>& out) {
  for (int i = 0; i < DIM; ++i)
    for (int j = i + 1; j < DIM; ++j)
      for (int si = 0; si < 2; ++si)
        for (int sj = 0; sj < 2; ++sj) {
          Vec v{};
          v[static_cast<std::size_t>(i)] = si ? -4 : 4;
          v[static_cast<std::size_t>(j)] = sj ? -4 : 4;
          out.push_back(v);
        }
}

void build_neg(Leech& L) {
  L.neg.assign(L.C.size(), 0u);
  for (std::size_t i = 0; i < L.C.size(); ++i) {
    const int32_t j = L.index_of(negate(L.C[i]));
    if (j < 0) throw std::logic_error("generate_leech: set is not negation-closed");
    L.neg[i] = static_cast<uint32_t>(j);
  }
}

}  // namespace

int32_t Leech::index_of(const Vec& v) const {
  // std::array<int8_t,24>::operator< is exactly the lexicographic signed
  // comparison with coordinate 0 most significant (docs/design.md §2.2).
  auto it = std::lower_bound(C.begin(), C.end(), v);
  if (it == C.end() || *it != v) return -1;
  return static_cast<int32_t>(it - C.begin());
}

bool Leech::is_lattice_vector(const std::array<int, DIM>& x) const {
  // 1. common parity. (x & 1) and (x & 3) are the residues mod 2 / mod 4 also
  // for negative ints on a two's-complement machine.
  const int m = x[0] & 1;
  for (int k = 1; k < DIM; ++k)
    if ((x[static_cast<std::size_t>(k)] & 1) != m) return false;
  // 2. residue class 2 (m = 0) or 3 (m = 1) mod 4 is a codeword.
  const int r = m ? 3 : 2;
  uint32_t mask = 0;
  long sum = 0;
  for (int k = 0; k < DIM; ++k) {
    const int xk = x[static_cast<std::size_t>(k)];
    if ((xk & 3) == r) mask |= 1u << k;
    sum += xk;
  }
  if (!golay_is_codeword(mask)) return false;
  // 3. sum ≡ 4m (mod 8).
  return (sum & 7) == 4L * m;
}

int leech_shape(const Vec& v) {
  int n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  for (int8_t c : v) {
    switch (c) {
      case 0: ++n0; break;
      case 1: case -1: ++n1; break;
      case 2: case -2: ++n2; break;
      case 3: case -3: ++n3; break;
      case 4: case -4: ++n4; break;
      default: return -1;
    }
  }
  if (n2 == 8 && n0 == 16) return 0;
  if (n3 == 1 && n1 == 23) return 1;
  if (n4 == 2 && n0 == 22) return 2;
  return -1;
}

Leech generate_leech() {
  Leech L;
  L.C.reserve(kN);
  gen_octad_shape(L.C);
  gen_three_one_shape(L.C);
  gen_four_four_shape(L.C);
  std::sort(L.C.begin(), L.C.end());
  if (L.C.size() != kN) throw std::logic_error("generate_leech: wrong count");
  build_neg(L);
  return L;
}

std::vector<uint32_t> pack_vectors(const std::vector<Vec>& C) {
  const std::size_t n = C.size();
  std::vector<uint32_t> packed(6 * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t w = 0; w < 6; ++w) {
      uint32_t word = 0;
      for (std::size_t b = 0; b < 4; ++b)
        word |= static_cast<uint32_t>(static_cast<uint8_t>(C[i][4 * w + b])) << (8 * b);
      packed[w * n + i] = word;
    }
  return packed;
}

std::vector<Vec> unpack_vectors(const std::vector<uint32_t>& packed) {
  if (packed.size() % 6 != 0) throw std::runtime_error("unpack_vectors: size not a multiple of 6");
  const std::size_t n = packed.size() / 6;
  std::vector<Vec> C(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t w = 0; w < 6; ++w) {
      const uint32_t word = packed[w * n + i];
      for (std::size_t b = 0; b < 4; ++b)
        C[i][4 * w + b] = static_cast<int8_t>(static_cast<uint8_t>(word >> (8 * b)));
    }
  return C;
}

Leech load_leech(const std::filesystem::path& data_dir) {
  const auto i8_path = data_dir / "leech_min.i8";
  const auto neg_path = data_dir / "neg.u32";
  const std::vector<uint8_t> raw = read_binary_file(i8_path);
  if (raw.size() != kN * kDim)
    throw std::runtime_error("load_leech: " + i8_path.string() + " has " +
                             std::to_string(raw.size()) + " bytes, expected " +
                             std::to_string(kN * kDim));
  const std::vector<uint8_t> rawneg = read_binary_file(neg_path);
  if (rawneg.size() != kN * 4)
    throw std::runtime_error("load_leech: " + neg_path.string() + " has " +
                             std::to_string(rawneg.size()) + " bytes, expected " +
                             std::to_string(kN * 4));
  Leech L;
  L.C.resize(kN);
  std::memcpy(L.C.data(), raw.data(), raw.size());  // int8 rows, row-major
  L.neg.resize(kN);
  std::memcpy(L.neg.data(), rawneg.data(), rawneg.size());  // little-endian host
  for (std::size_t i = 1; i < kN; ++i)
    if (!(L.C[i - 1] < L.C[i]))
      throw std::runtime_error("load_leech: rows not strictly increasing at row " +
                               std::to_string(i));
  for (std::size_t i = 0; i < kN; ++i) {
    const uint32_t j = L.neg[i];
    if (j >= kN || L.C[j] != negate(L.C[i]))
      throw std::runtime_error("load_leech: neg[" + std::to_string(i) + "] = " +
                               std::to_string(j) + " is wrong");
  }
  return L;
}

void save_leech(const Leech& L, const std::filesystem::path& data_dir) {
  if (L.C.size() != kN || L.neg.size() != kN)
    throw std::runtime_error("save_leech: Leech object has wrong size");
  std::filesystem::create_directories(data_dir);
  write_binary_file(data_dir / "leech_min.i8", L.C.data(), kN * kDim);
  write_binary_file(data_dir / "neg.u32", L.neg.data(), kN * 4);
  const std::vector<uint32_t> packed = pack_vectors(L.C);
  write_binary_file(data_dir / "leech_packed.u32", packed.data(), packed.size() * 4);
  // Human-readable copy, same order, no header (plain 24-column table).
  const auto txt = data_dir / "leech_min.txt";
  std::FILE* f = std::fopen(txt.c_str(), "wb");
  if (!f) throw std::runtime_error("save_leech: cannot open " + txt.string());
  std::string buf;
  buf.reserve(kN * 60);
  char tmp[8];
  for (const Vec& v : L.C) {
    for (int k = 0; k < DIM; ++k) {
      const int len = std::snprintf(tmp, sizeof tmp, k ? " %d" : "%d",
                                    static_cast<int>(v[static_cast<std::size_t>(k)]));
      buf.append(tmp, static_cast<std::size_t>(len));
    }
    buf.push_back('\n');
  }
  const bool ok = std::fwrite(buf.data(), 1, buf.size(), f) == buf.size();
  const bool closed = std::fclose(f) == 0;
  if (!ok || !closed) throw std::runtime_error("save_leech: write failed for " + txt.string());
}

}  // namespace kiss

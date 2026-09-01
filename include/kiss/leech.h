// T1.2 — Leech lattice minimal vectors in canonical order (docs/design.md §2.2/§2.3,
// README §1.2).
//
// Integer (√8) scaling. The 196560 minimal vectors come in three shapes:
//   (a) (±2^8, 0^16): ±2 on an octad, even number of minus signs   759·128 = 97152
//   (b) (∓3, ±1^23): (−3 at i, +1 elsewhere), then negate the
//       coordinates of a Golay codeword c                          24·4096 = 98304
//   (c) (±4, ±4, 0^22): two ±4 on any pair of coordinates          C(24,2)·4 = 1104
// Canonical order: rows sorted lexicographically ascending as signed integers,
// coordinate 0 most significant. That order is the vertex index used
// everywhere (data/leech_min.i8, neg.u32, leech_packed.u32, adj.u32).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "kiss/types.h"

namespace kiss {

struct Leech {
  std::vector<Vec> C;          // N rows, canonical (sorted) order
  std::vector<uint32_t> neg;   // neg[i] = index of -C[i]

  // Binary search in C; -1 if v is not a minimal vector.
  int32_t index_of(const Vec& v) const;

  // README §1.2 membership test for an arbitrary integer vector (any norm):
  //   1. all coordinates have the same parity m;
  //   2. {i : x_i ≡ 2 (mod 4)} (m = 0) resp. {i : x_i ≡ 3 (mod 4)} (m = 1)
  //      is a Golay codeword (the complementary class is then one too);
  //   3. Σ x_i ≡ 4m (mod 8).
  bool is_lattice_vector(const std::array<int, DIM>& x) const;
};

// Deterministic generator: builds the three shapes from golay_codewords(),
// sorts canonically and fills neg. Never touches the disk. ~40 ms.
Leech generate_leech();

// Reads <data_dir>/leech_min.i8 and <data_dir>/neg.u32, validates sizes,
// canonical ordering and the neg table. Throws std::runtime_error on failure.
Leech load_leech(const std::filesystem::path& data_dir);

// ---- additions beyond §2.3 (documented in docs/reports/T1.2.md) ------------

// Shape classification of a minimal vector: 0 = (±2^8,0^16), 1 = (∓3,±1^23),
// 2 = (±4,±4,0^22); -1 if v does not match any of the three shapes.
int leech_shape(const Vec& v);

// dp4a layout of docs/design.md §2.2: uint32[6][n] structure-of-arrays; word w of vector
// i (at index w*n + i) holds coordinates 4w..4w+3 as int8 bytes, little-endian
// (coordinate 4w in the low byte).
std::vector<uint32_t> pack_vectors(const std::vector<Vec>& C);
// Inverse of pack_vectors (packed.size() must be a multiple of 6).
std::vector<Vec> unpack_vectors(const std::vector<uint32_t>& packed);

// Writes leech_min.i8, neg.u32, leech_packed.u32 and leech_min.txt into
// data_dir (created if missing). Throws std::runtime_error on I/O failure.
void save_leech(const Leech& L, const std::filesystem::path& data_dir);

}  // namespace kiss

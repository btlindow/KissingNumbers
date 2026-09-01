// T4.1 / T3.4 — automorphisms of the Leech lattice acting on the minimal
// vectors C (docs/design.md T3.4, T4.1; README §3 W3).
//
// Three representations, all in the integer √8 scaling of kiss/types.h:
//
//  * Monomial   — an element of the monomial group 2^12:M24 = N(2^12):
//                 a coordinate permutation followed by negating the
//                 coordinates of a Golay codeword.
//  * Aut        — a general element of Co_0 as a 24×24 rational matrix with a
//                 common denominator (Conway's ξ has denominator 2).
//  * IndexPerm  — the induced permutation of the 196560 vertex indices
//                 (uint32[N], 786 KB). This is the working representation for
//                 random elements (product replacement) and for T3.4's orbits.
//
// Conventions:
//   perm[i] = image of coordinate i.  (m·v)[perm[i]] = s_{perm[i]} · v[i] with
//   s_j = −1 iff bit j of `signs` is set, i.e. m = D_signs · P_perm.
//   compose(a, b) is a∘b (apply b first) in every representation, so that
//   index_permutation(L, compose(a,b)) == compose(index_permutation(L,a),
//   index_permutation(L,b)) and (a∘b)[i] = a[b[i]] for index permutations.
//   Nothing here asserts a Monomial/Aut is actually an automorphism; the
//   proof is index_permutation(), which throws if any image leaves C.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "kiss/leech.h"
#include "kiss/types.h"

namespace kiss {

// ---------------------------------------------------------------------------
// Monomial group 2^12:M24
// ---------------------------------------------------------------------------
struct Monomial {
  std::array<uint8_t, DIM> perm;  // coordinate i -> perm[i]
  uint32_t signs;                 // Golay codeword mask: negate those (output) coordinates
};

Monomial monomial_identity();
Monomial sign_flip(uint32_t codeword);                              // perm = id
Monomial coordinate_permutation(const std::array<uint8_t, DIM>& perm);  // signs = 0
bool is_permutation(const std::array<uint8_t, DIM>& perm);
Vec apply(const Monomial& m, const Vec& v);
Monomial compose(const Monomial& a, const Monomial& b);  // a∘b
Monomial inverse(const Monomial& m);
bool operator==(const Monomial& a, const Monomial& b);
// True iff perm maps every Golay codeword to a codeword (i.e. perm ∈ M24).
bool preserves_golay_code(const std::array<uint8_t, DIM>& perm);

// Parse data/group/m24_generators.txt: '#' comments, one permutation per line
// (images of 0..23, optionally followed by a '#' comment). Throws on a line
// that is not a permutation. Does NOT check membership in M24.
std::vector<Monomial> load_m24_generators(const std::filesystem::path& path);

// ---------------------------------------------------------------------------
// General automorphism as a rational matrix
// ---------------------------------------------------------------------------
struct Aut {
  std::array<int32_t, DIM * DIM> num;  // row-major; (A v)[r] = Σ_c num[r*DIM+c] v[c] / den
  int32_t den;
};

Aut aut_identity();
Aut aut_from_monomial(const Monomial& m);          // den = 1
// Exact application: throws std::runtime_error if some coordinate of the
// image is not divisible by den or does not fit in int8.
Vec apply(const Aut& a, const Vec& v);
// Same, but reports failure instead of throwing (out is untouched on failure).
bool try_apply(const Aut& a, const Vec& v, Vec& out);
Aut compose(const Aut& a, const Aut& b);            // a∘b, reduced by gcd
Aut transpose(const Aut& a);                        // = inverse for orthogonal a (den unchanged)
bool operator==(const Aut& a, const Aut& b);

using Sextet = std::array<std::array<uint8_t, 4>, 6>;
// The sextet through `tetrad` (S(5,8,24): T∪{p} lies in exactly one octad).
// Tetrads are sorted, tetrad 0 is `tetrad` sorted, the others in order of
// their smallest element. Throws if tetrad is not 4 distinct coordinates.
Sextet find_sextet(const std::array<uint8_t, 4>& tetrad);
// Conway's ξ: on tetrad T with sign s_T, x_i -> s_T · ((Σ_{j∈T} x_j)/2 − x_i),
// i.e. the block s_T·(J − 2I)/2. Only patterns with an odd number of negative
// tetrads are automorphisms (verified in tests/test_group.cpp); the canonical
// choice is XI_TETRAD_SIGNS.
constexpr std::array<int, 6> XI_TETRAD_SIGNS = {-1, 1, 1, 1, 1, 1};
Aut make_xi(const Sextet& sextet, const std::array<int, 6>& tetrad_signs = XI_TETRAD_SIGNS);

// data/group/xi.txt format: '#' comments, a line "den D", then 24 rows of 24 ints.
Aut load_aut(const std::filesystem::path& path);
void save_aut(const std::filesystem::path& path, const Aut& a, const std::string& header_comment = "");
// data/group/sextet.txt: '#' comments, 6 lines of 4 ints.
Sextet load_sextet(const std::filesystem::path& path);

// ---------------------------------------------------------------------------
// Index permutations of the 196560 minimal vectors
// ---------------------------------------------------------------------------
using IndexPerm = std::vector<uint32_t>;  // size N; g[i] = index of g·C[i]

// Induced permutation. Throws std::runtime_error naming the first index whose
// image is not a minimal vector (that is the automorphism test).
IndexPerm index_permutation(const Leech& L, const Monomial& m);
IndexPerm index_permutation(const Leech& L, const Aut& a);
IndexPerm index_identity();
IndexPerm compose(const IndexPerm& a, const IndexPerm& b);   // (a∘b)[i] = a[b[i]]
void compose_into(IndexPerm& out, const IndexPerm& a, const IndexPerm& b);
IndexPerm inverse(const IndexPerm& g);
bool is_index_permutation(const IndexPerm& g);              // size N and bijective
// dot(C[g i], C[g j]) == dot(C[i], C[j]) on `samples` random pairs (i, j).
// Returns the number of failing pairs (0 = pass).
int inner_product_violations(const Leech& L, const IndexPerm& g, int samples, uint64_t seed);
// Apply to a set given by indices / vectors.
std::vector<uint32_t> apply(const IndexPerm& g, const std::vector<uint32_t>& S_idx);
std::vector<Vec> apply(const Leech& L, const IndexPerm& g, const std::vector<Vec>& S);

// Binary uint32[N] files.
void write_index_perm(const std::filesystem::path& path, const IndexPerm& g);
IndexPerm read_index_perm(const std::filesystem::path& path);  // throws on wrong size / not a bijection

// ---------------------------------------------------------------------------
// Generating sets and random elements
// ---------------------------------------------------------------------------
// Monomial generators of 2^12:M24: the M24 generators of m24_generators.txt
// plus sign flips. With basis_signs = false a single octad sign flip (M24 is
// transitive on octads and the octads span the code, so this already
// generates 2^12:M24 — 4 generators); with basis_signs = true sign flips on
// a GF(2)-basis of the code (12 codewords, 15 generators).
std::vector<Monomial> monomial_generators(const std::filesystem::path& group_dir, bool basis_signs = false);
// The same as index permutations (each verified by index_permutation()).
std::vector<IndexPerm> monomial_generator_perms(const Leech& L, const std::filesystem::path& group_dir,
                                                bool basis_signs = false);
// {monomial generators} ∪ {ξ} as index permutations — a generating set of Co_0
// (5 elements by default).
std::vector<IndexPerm> co0_generator_perms(const Leech& L, const std::filesystem::path& group_dir,
                                           bool basis_signs = false);

// Product replacement (Celler, Leedham-Green, Murray, Niemeyer, O'Brien 1995)
// with an accumulator ("rattle", Leedham-Green & Murray 2002; GAP's
// PseudoRandom). State: `slots` elements initialised by cycling through the
// generators, plus an accumulator a (initially the identity). One step picks
// distinct slots i ≠ j and replaces x_i by one of x_i·x_j, x_i·x_j^{-1},
// x_j·x_i, x_j^{-1}·x_i (uniformly), then sets a ← a·x_i; the new a is the
// output. Without the accumulator, consecutive outputs repeat exactly whenever
// an involution slot is multiplied in twice (observed 20 repeats in 1000).
// Discard `burn_in` steps before using outputs as random elements.
class ProductReplacement {
 public:
  ProductReplacement(const std::vector<IndexPerm>& generators, int slots, uint64_t seed);
  void burn_in(int steps);
  const IndexPerm& next();           // one step; returns the accumulator (valid until the next call)
  const IndexPerm& slot(int i) const { return state_[static_cast<std::size_t>(i)]; }
  int slots() const { return static_cast<int>(state_.size()); }
  uint64_t steps() const { return steps_; }
 private:
  std::vector<IndexPerm> state_;
  IndexPerm acc_, tmp_, inv_;
  std::mt19937_64 rng_;
  uint64_t steps_ = 0;
};

}  // namespace kiss

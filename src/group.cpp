// T4.1 / T3.4 — Leech automorphisms (see include/kiss/group.h).
#include "kiss/group.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "kiss/golay.h"
#include "kiss/io.h"
#include "kiss/bits.h"

namespace kiss {

namespace {

std::size_t idx(int i) { return static_cast<std::size_t>(i); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

// Strip '#' comments; return the remaining tokens.
std::vector<std::string> tokens_of(const std::string& line) {
  std::string s = line;
  const auto hash = s.find('#');
  if (hash != std::string::npos) s.erase(hash);
  std::istringstream is(s);
  std::vector<std::string> out;
  std::string t;
  while (is >> t) out.push_back(t);
  return out;
}

long parse_long(const std::string& t, const std::filesystem::path& path, int lineno) {
  std::size_t used = 0;
  long v = 0;
  try {
    v = std::stol(t, &used);
  } catch (const std::exception&) {
    fail(path.string() + ":" + std::to_string(lineno) + ": not an integer: '" + t + "'");
  }
  if (used != t.size()) fail(path.string() + ":" + std::to_string(lineno) + ": not an integer: '" + t + "'");
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// Monomial
// ---------------------------------------------------------------------------
bool is_permutation(const std::array<uint8_t, DIM>& perm) {
  uint32_t seen = 0;
  for (int i = 0; i < DIM; ++i) {
    const uint8_t p = perm[idx(i)];
    if (p >= DIM || (seen >> p) & 1u) return false;
    seen |= 1u << p;
  }
  return true;
}

Monomial monomial_identity() {
  Monomial m;
  for (int i = 0; i < DIM; ++i) m.perm[idx(i)] = static_cast<uint8_t>(i);
  m.signs = 0;
  return m;
}

Monomial sign_flip(uint32_t codeword) {
  Monomial m = monomial_identity();
  m.signs = codeword & 0xFFFFFFu;
  return m;
}

Monomial coordinate_permutation(const std::array<uint8_t, DIM>& perm) {
  if (!is_permutation(perm)) fail("coordinate_permutation: not a permutation");
  Monomial m;
  m.perm = perm;
  m.signs = 0;
  return m;
}

Vec apply(const Monomial& m, const Vec& v) {
  Vec w;
  for (int i = 0; i < DIM; ++i) {
    const uint8_t j = m.perm[idx(i)];
    const int8_t x = v[idx(i)];
    w[j] = static_cast<int8_t>(((m.signs >> j) & 1u) ? -x : x);
  }
  return w;
}

Monomial compose(const Monomial& a, const Monomial& b) {
  // a∘b = D_a P_a D_b P_b = D_a D_{P_a(b.signs)} P_a P_b.
  Monomial c;
  uint32_t moved = 0;
  for (int i = 0; i < DIM; ++i) {
    c.perm[idx(i)] = a.perm[b.perm[idx(i)]];
    if ((b.signs >> i) & 1u) moved |= 1u << a.perm[idx(i)];
  }
  c.signs = a.signs ^ moved;
  return c;
}

Monomial inverse(const Monomial& m) {
  // m^{-1} = P^{-1} D = D' P^{-1} with D' bit i = D bit perm[i].
  Monomial r;
  r.signs = 0;
  for (int i = 0; i < DIM; ++i) {
    const uint8_t j = m.perm[idx(i)];
    r.perm[j] = static_cast<uint8_t>(i);
    if ((m.signs >> j) & 1u) r.signs |= 1u << i;
  }
  return r;
}

bool operator==(const Monomial& a, const Monomial& b) { return a.perm == b.perm && a.signs == b.signs; }

bool preserves_golay_code(const std::array<uint8_t, DIM>& perm) {
  if (!is_permutation(perm)) return false;
  for (uint32_t w : golay_codewords()) {
    uint32_t img = 0;
    for (int i = 0; i < DIM; ++i)
      if ((w >> i) & 1u) img |= 1u << perm[idx(i)];
    if (!golay_is_codeword(img)) return false;
  }
  return true;
}

std::vector<Monomial> load_m24_generators(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) fail("load_m24_generators: cannot open " + path.string());
  std::vector<Monomial> out;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto t = tokens_of(line);
    if (t.empty()) continue;
    if (t.size() != DIM) fail(path.string() + ":" + std::to_string(lineno) + ": expected 24 values");
    std::array<uint8_t, DIM> perm;
    for (int i = 0; i < DIM; ++i) {
      const long v = parse_long(t[idx(i)], path, lineno);
      if (v < 0 || v >= DIM) fail(path.string() + ":" + std::to_string(lineno) + ": value out of range");
      perm[idx(i)] = static_cast<uint8_t>(v);
    }
    if (!is_permutation(perm)) fail(path.string() + ":" + std::to_string(lineno) + ": not a permutation");
    out.push_back(coordinate_permutation(perm));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Aut
// ---------------------------------------------------------------------------
Aut aut_identity() {
  Aut a;
  a.num.fill(0);
  for (int i = 0; i < DIM; ++i) a.num[idx(i * DIM + i)] = 1;
  a.den = 1;
  return a;
}

Aut aut_from_monomial(const Monomial& m) {
  Aut a;
  a.num.fill(0);
  a.den = 1;
  for (int i = 0; i < DIM; ++i) {
    const int j = m.perm[idx(i)];
    a.num[idx(j * DIM + i)] = ((m.signs >> j) & 1u) ? -1 : 1;
  }
  return a;
}

bool try_apply(const Aut& a, const Vec& v, Vec& out) {
  Vec w;
  for (int r = 0; r < DIM; ++r) {
    int64_t s = 0;
    const int32_t* row = &a.num[idx(r * DIM)];
    for (int c = 0; c < DIM; ++c) s += static_cast<int64_t>(row[c]) * v[idx(c)];
    if (s % a.den != 0) return false;
    s /= a.den;
    if (s < -128 || s > 127) return false;
    w[idx(r)] = static_cast<int8_t>(s);
  }
  out = w;
  return true;
}

Vec apply(const Aut& a, const Vec& v) {
  Vec w;
  if (!try_apply(a, v, w)) fail("apply(Aut): image is not an integer vector in int8 range");
  return w;
}

Aut compose(const Aut& a, const Aut& b) {
  Aut c;
  int64_t g = static_cast<int64_t>(a.den) * b.den;
  std::array<int64_t, DIM * DIM> tmp;
  for (int r = 0; r < DIM; ++r)
    for (int k = 0; k < DIM; ++k) {
      int64_t s = 0;
      for (int m = 0; m < DIM; ++m)
        s += static_cast<int64_t>(a.num[idx(r * DIM + m)]) * b.num[idx(m * DIM + k)];
      tmp[idx(r * DIM + k)] = s;
      g = std::gcd(g, s < 0 ? -s : s);
    }
  if (g == 0) g = 1;
  for (int i = 0; i < DIM * DIM; ++i) {
    const int64_t v = tmp[idx(i)] / g;
    if (v < INT32_MIN || v > INT32_MAX) fail("compose(Aut): overflow");
    c.num[idx(i)] = static_cast<int32_t>(v);
  }
  c.den = static_cast<int32_t>(static_cast<int64_t>(a.den) * b.den / g);
  return c;
}

Aut transpose(const Aut& a) {
  Aut t;
  t.den = a.den;
  for (int r = 0; r < DIM; ++r)
    for (int c = 0; c < DIM; ++c) t.num[idx(c * DIM + r)] = a.num[idx(r * DIM + c)];
  return t;
}

bool operator==(const Aut& a, const Aut& b) { return a.den == b.den && a.num == b.num; }

Sextet find_sextet(const std::array<uint8_t, 4>& tetrad) {
  std::array<uint8_t, 4> t0 = tetrad;
  std::sort(t0.begin(), t0.end());
  uint32_t m0 = 0;
  for (uint8_t i : t0) {
    if (i >= DIM || (m0 >> i) & 1u) fail("find_sextet: tetrad must be 4 distinct coordinates");
    m0 |= 1u << i;
  }
  const auto octads = golay_octads();
  Sextet s;
  s[0] = t0;
  uint32_t used = m0;
  int n = 1;
  while (used != 0xFFFFFFu) {
    int p = 0;
    while ((used >> p) & 1u) ++p;
    const uint32_t m = m0 | (1u << p);
    int found = 0;
    uint32_t oct = 0;
    for (uint32_t o : octads)
      if ((o & m) == m) { ++found; oct = o; }
    if (found != 1) fail("find_sextet: S(5,8,24) violated");
    const uint32_t rest = oct & ~m0;
    int k = 0;
    for (int i = 0; i < DIM; ++i)
      if ((rest >> i) & 1u) s[idx(n)][idx(k++)] = static_cast<uint8_t>(i);
    if (k != 4 || (rest & used)) fail("find_sextet: tetrad overlap");
    used |= rest;
    ++n;
  }
  return s;
}

Aut make_xi(const Sextet& sextet, const std::array<int, 6>& tetrad_signs) {
  Aut a;
  a.num.fill(0);
  a.den = 2;
  uint32_t seen = 0;
  for (int t = 0; t < 6; ++t) {
    const int s = tetrad_signs[idx(t)];
    if (s != 1 && s != -1) fail("make_xi: tetrad signs must be ±1");
    for (uint8_t r : sextet[idx(t)]) {
      if (r >= DIM || (seen >> r) & 1u) fail("make_xi: sextet is not a partition of 0..23");
      seen |= 1u << r;
      for (uint8_t c : sextet[idx(t)]) a.num[idx(r * DIM + c)] = (r == c) ? -s : s;
    }
  }
  return a;
}

Aut load_aut(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) fail("load_aut: cannot open " + path.string());
  Aut a;
  a.num.fill(0);
  a.den = 0;
  std::string line;
  int lineno = 0, row = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto t = tokens_of(line);
    if (t.empty()) continue;
    if (a.den == 0) {
      if (t.size() != 2 || t[0] != "den") fail(path.string() + ":" + std::to_string(lineno) + ": expected 'den D'");
      const long d = parse_long(t[1], path, lineno);
      if (d <= 0 || d > 1024) fail(path.string() + ": bad denominator");
      a.den = static_cast<int32_t>(d);
      continue;
    }
    if (row >= DIM) fail(path.string() + ":" + std::to_string(lineno) + ": too many rows");
    if (t.size() != DIM) fail(path.string() + ":" + std::to_string(lineno) + ": expected 24 values");
    for (int c = 0; c < DIM; ++c) a.num[idx(row * DIM + c)] = static_cast<int32_t>(parse_long(t[idx(c)], path, lineno));
    ++row;
  }
  if (a.den == 0 || row != DIM) fail(path.string() + ": incomplete matrix");
  return a;
}

void save_aut(const std::filesystem::path& path, const Aut& a, const std::string& header_comment) {
  std::ofstream out(path);
  if (!out) fail("save_aut: cannot open " + path.string());
  if (!header_comment.empty()) {
    std::istringstream is(header_comment);
    std::string l;
    while (std::getline(is, l)) out << "# " << l << "\n";
  }
  out << "den " << a.den << "\n";
  for (int r = 0; r < DIM; ++r) {
    for (int c = 0; c < DIM; ++c) out << (c ? " " : "") << a.num[idx(r * DIM + c)];
    out << "\n";
  }
  if (!out) fail("save_aut: write failed");
}

Sextet load_sextet(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) fail("load_sextet: cannot open " + path.string());
  Sextet s;
  std::string line;
  int lineno = 0, row = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto t = tokens_of(line);
    if (t.empty()) continue;
    if (row >= 6 || t.size() != 4) fail(path.string() + ":" + std::to_string(lineno) + ": expected 6 rows of 4");
    for (int c = 0; c < 4; ++c) {
      const long v = parse_long(t[idx(c)], path, lineno);
      if (v < 0 || v >= DIM) fail(path.string() + ": coordinate out of range");
      s[idx(row)][idx(c)] = static_cast<uint8_t>(v);
    }
    ++row;
  }
  if (row != 6) fail(path.string() + ": expected 6 tetrads");
  return s;
}

// ---------------------------------------------------------------------------
// IndexPerm
// ---------------------------------------------------------------------------
namespace {

template <class F>
IndexPerm induced(const Leech& L, F&& image, const char* what) {
  if (L.C.size() != static_cast<std::size_t>(N)) fail("index_permutation: Leech has wrong size");
  IndexPerm g(static_cast<std::size_t>(N));
  int64_t bad = -1;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    Vec w;
    int32_t j = -1;
    if (image(L.C[idx(i)], w)) j = L.index_of(w);
    if (j < 0) {
#pragma omp critical
      bad = (bad < 0 || i < bad) ? i : bad;
      j = 0;
    }
    g[idx(i)] = static_cast<uint32_t>(j);
  }
  if (bad >= 0)
    fail(std::string("index_permutation(") + what + "): image of C[" + std::to_string(bad) +
         "] is not a minimal vector — not an automorphism");
  return g;
}

}  // namespace

IndexPerm index_permutation(const Leech& L, const Monomial& m) {
  if (!is_permutation(m.perm)) fail("index_permutation(Monomial): perm is not a permutation");
  return induced(L, [&](const Vec& v, Vec& w) { w = apply(m, v); return true; }, "Monomial");
}

IndexPerm index_permutation(const Leech& L, const Aut& a) {
  return induced(L, [&](const Vec& v, Vec& w) { return try_apply(a, v, w); }, "Aut");
}

IndexPerm index_identity() {
  IndexPerm g(static_cast<std::size_t>(N));
  std::iota(g.begin(), g.end(), 0u);
  return g;
}

void compose_into(IndexPerm& out, const IndexPerm& a, const IndexPerm& b) {
  out.resize(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) out[i] = a[b[i]];
}

IndexPerm compose(const IndexPerm& a, const IndexPerm& b) {
  IndexPerm c;
  compose_into(c, a, b);
  return c;
}

IndexPerm inverse(const IndexPerm& g) {
  IndexPerm r(g.size());
  for (std::size_t i = 0; i < g.size(); ++i) r[g[i]] = static_cast<uint32_t>(i);
  return r;
}

bool is_index_permutation(const IndexPerm& g) {
  if (g.size() != static_cast<std::size_t>(N)) return false;
  std::vector<uint8_t> seen(static_cast<std::size_t>(N), 0);
  for (uint32_t v : g) {
    if (v >= static_cast<uint32_t>(N) || seen[v]) return false;
    seen[v] = 1;
  }
  return true;
}

int inner_product_violations(const Leech& L, const IndexPerm& g, int samples, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint32_t> U(0, N - 1);
  int bad = 0;
  for (int s = 0; s < samples; ++s) {
    const uint32_t i = U(rng), j = U(rng);
    if (dot(L.C[g[i]], L.C[g[j]]) != dot(L.C[i], L.C[j])) ++bad;
  }
  return bad;
}

std::vector<uint32_t> apply(const IndexPerm& g, const std::vector<uint32_t>& S_idx) {
  std::vector<uint32_t> out(S_idx.size());
  for (std::size_t k = 0; k < S_idx.size(); ++k) out[k] = g[S_idx[k]];
  return out;
}

std::vector<Vec> apply(const Leech& L, const IndexPerm& g, const std::vector<Vec>& S) {
  std::vector<Vec> out(S.size());
  for (std::size_t k = 0; k < S.size(); ++k) {
    const int32_t i = L.index_of(S[k]);
    if (i < 0) fail("apply(IndexPerm): set element " + std::to_string(k) + " is not a minimal vector");
    out[k] = L.C[g[static_cast<uint32_t>(i)]];
  }
  return out;
}

void write_index_perm(const std::filesystem::path& path, const IndexPerm& g) {
  write_binary_file(path, g.data(), g.size() * sizeof(uint32_t));
}

IndexPerm read_index_perm(const std::filesystem::path& path) {
  const auto bytes = read_binary_file(path);
  if (bytes.size() != static_cast<std::size_t>(N) * sizeof(uint32_t))
    fail("read_index_perm: " + path.string() + " has wrong size");
  IndexPerm g(static_cast<std::size_t>(N));
  std::memcpy(g.data(), bytes.data(), bytes.size());
  if (!is_index_permutation(g)) fail("read_index_perm: " + path.string() + " is not a permutation of 0..N-1");
  return g;
}

// ---------------------------------------------------------------------------
// Generating sets
// ---------------------------------------------------------------------------
std::vector<Monomial> monomial_generators(const std::filesystem::path& group_dir, bool basis_signs) {
  std::vector<Monomial> gens = load_m24_generators(group_dir / "m24_generators.txt");
  if (!basis_signs) {
    gens.push_back(sign_flip(golay_octads().front()));
    return gens;
  }
  // Sign flips on a GF(2)-basis of the Golay code: greedy row reduction.
  std::vector<uint32_t> basis;  // echelon form (pivot = highest bit)
  for (uint32_t w : golay_codewords()) {
    uint32_t r = w;
    for (uint32_t b : basis)
      if (r & (1u << (31 - kiss::clz32(b)))) r ^= b;
    if (r == 0) continue;
    // keep the echelon invariant: reduce existing rows by r
    for (uint32_t& b : basis)
      if (b & (1u << (31 - kiss::clz32(r)))) b ^= r;
    basis.push_back(r);
    gens.push_back(sign_flip(w));  // the original codeword, not the reduced one
    if (basis.size() == 12) break;
  }
  if (basis.size() != 12) fail("monomial_generators: Golay basis extraction failed");
  return gens;
}

std::vector<IndexPerm> monomial_generator_perms(const Leech& L, const std::filesystem::path& group_dir,
                                                bool basis_signs) {
  std::vector<IndexPerm> out;
  for (const Monomial& m : monomial_generators(group_dir, basis_signs)) out.push_back(index_permutation(L, m));
  return out;
}

std::vector<IndexPerm> co0_generator_perms(const Leech& L, const std::filesystem::path& group_dir,
                                           bool basis_signs) {
  std::vector<IndexPerm> out = monomial_generator_perms(L, group_dir, basis_signs);
  out.push_back(index_permutation(L, load_aut(group_dir / "xi.txt")));
  return out;
}

// ---------------------------------------------------------------------------
// Product replacement
// ---------------------------------------------------------------------------
ProductReplacement::ProductReplacement(const std::vector<IndexPerm>& generators, int slots, uint64_t seed)
    : rng_(seed) {
  if (generators.empty()) fail("ProductReplacement: no generators");
  for (const IndexPerm& g : generators)
    if (g.size() != static_cast<std::size_t>(N)) fail("ProductReplacement: generator of wrong size");
  if (slots < 2) slots = 2;
  if (slots < static_cast<int>(generators.size())) slots = static_cast<int>(generators.size());
  state_.reserve(static_cast<std::size_t>(slots));
  for (int k = 0; k < slots; ++k) state_.push_back(generators[idx(k) % generators.size()]);
  acc_ = index_identity();
  tmp_.resize(static_cast<std::size_t>(N));
  inv_.resize(static_cast<std::size_t>(N));
}

void ProductReplacement::burn_in(int steps) {
  for (int s = 0; s < steps; ++s) next();
}

const IndexPerm& ProductReplacement::next() {
  const int k = slots();
  std::uniform_int_distribution<int> U(0, k - 1);
  const int i = U(rng_);
  int j = U(rng_);
  while (j == i) j = U(rng_);
  const int variant = static_cast<int>(rng_() & 3u);
  const IndexPerm* xj = &state_[idx(j)];
  if (variant & 1) {  // use x_j^{-1}
    const IndexPerm& g = state_[idx(j)];
    for (std::size_t t = 0; t < g.size(); ++t) inv_[g[t]] = static_cast<uint32_t>(t);
    xj = &inv_;
  }
  if (variant & 2) compose_into(tmp_, state_[idx(i)], *xj);   // x_i ← x_i · x_j^{±1}
  else             compose_into(tmp_, *xj, state_[idx(i)]);   // x_i ← x_j^{±1} · x_i
  state_[idx(i)].swap(tmp_);
  compose_into(tmp_, acc_, state_[idx(i)]);                // a ← a · x_i
  acc_.swap(tmp_);
  ++steps_;
  return acc_;
}

}  // namespace kiss

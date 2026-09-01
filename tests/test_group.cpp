// T4.1 — acceptance tests for kiss/group.h.
//
//  1. Monomial group: the M24 generators of data/group/m24_generators.txt are
//     permutations, preserve the Golay code and induce permutations of C;
//     a transposition / a non-codeword sign flip is rejected by
//     index_permutation; compose/inverse agree between Monomial and IndexPerm.
//  2. ξ: sextet through {0,1,2,3} (matches data/group/sextet.txt), make_xi ==
//     data/group/xi.txt, ξ maps all 196560 minimal vectors bijectively onto C,
//     ξ² = 1, ξ preserves inner products; exactly the 16 sign patterns with an
//     odd number of negated tetrads work, the uniform pattern is rejected.
//  3. Product replacement on {α, γ, δ, octad sign flip, ξ}: 1000 elements each
//     pass a 10^4-sample inner-product-preservation check and are bijections.
//  4. 200 of them applied to S (data/S496.txt, else data/S488.txt, else a
//     random greedy independent set): every image is an independent set of
//     the same size (Gram ≤ 8 checked here directly); overlap histogram.
//  5. Observations: |gS ∩ S| for each generator and −I; random monomial
//     elements fixing S; shape counts of S.
// Usage: test_group [--set path] [--count 1000] [--apply 200] [--seed 1]
// Prints `RESULT ok=1 ...`; exit code 0 iff all pass.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "kiss/golay.h"
#include "kiss/group.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      ++g_failures;                                                 \
      std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                     \
      std::printf("\n");                                            \
    }                                                               \
  } while (0)

using kiss::Aut;
using kiss::DIM;
using kiss::IndexPerm;
using kiss::Leech;
using kiss::Monomial;
using kiss::N;
using kiss::Vec;

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

template <class F>
bool throws(F&& f) {
  try { f(); } catch (const std::runtime_error&) { return true; }
  return false;
}

std::vector<uint32_t> indices_of(const Leech& L, const std::vector<Vec>& S) {
  std::vector<uint32_t> idx;
  for (const Vec& v : S) {
    const int32_t i = L.index_of(v);
    if (i < 0) throw std::runtime_error("set element is not a minimal vector");
    idx.push_back(static_cast<uint32_t>(i));
  }
  return idx;
}

// Independent set check done here on purpose (no dependency on kiss/verify.h):
// norms 32, distinct, all off-diagonal Gram entries ≤ 8.
bool is_independent(const std::vector<Vec>& S, std::string* why = nullptr) {
  for (std::size_t a = 0; a < S.size(); ++a) {
    if (kiss::norm2(S[a]) != 32) { if (why) *why = "norm != 32 at " + std::to_string(a); return false; }
    for (std::size_t b = a + 1; b < S.size(); ++b) {
      const int d = kiss::dot(S[a], S[b]);
      if (d > 8) { if (why) *why = "dot=" + std::to_string(d) + " at (" + std::to_string(a) + "," + std::to_string(b) + ")"; return false; }
    }
  }
  return true;
}

// Random greedy maximal independent set (fallback when no fixture exists).
std::vector<Vec> greedy_independent_set(const Leech& L, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<uint32_t> order(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) order[static_cast<std::size_t>(i)] = static_cast<uint32_t>(i);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<Vec> S;
  for (uint32_t i : order) {
    const Vec& v = L.C[i];
    bool ok = true;
    for (const Vec& s : S)
      if (kiss::dot(s, v) == 16) { ok = false; break; }
    if (ok) S.push_back(v);
    if (S.size() >= 400) break;  // keep the fallback cheap; size does not matter
  }
  return S;
}

std::size_t overlap(const std::vector<uint32_t>& A_sorted, const std::vector<uint32_t>& B) {
  std::size_t n = 0;
  for (uint32_t b : B) n += std::binary_search(A_sorted.begin(), A_sorted.end(), b) ? 1 : 0;
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  std::string set_path;
  int count = 1000, apply_n = 200;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--set" && i + 1 < argc) set_path = argv[++i];
    else if (a == "--count" && i + 1 < argc) count = std::atoi(argv[++i]);
    else if (a == "--apply" && i + 1 < argc) apply_n = std::atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
  }
  const std::filesystem::path group_dir = "data/group";
  const auto t_start = std::chrono::steady_clock::now();
  Leech L;
  try { L = kiss::load_leech("data"); } catch (const std::exception&) { L = kiss::generate_leech(); }
  std::printf("leech loaded: %.1f ms\n", ms_since(t_start));

  // ---- 1. Monomial group ---------------------------------------------------
  auto t1 = std::chrono::steady_clock::now();
  std::vector<Monomial> m24;
  try { m24 = kiss::load_m24_generators(group_dir / "m24_generators.txt"); }
  catch (const std::exception& e) { std::printf("FAIL: %s\n", e.what()); std::printf("RESULT ok=0\n"); return 1; }
  CHECK(m24.size() == 3, "got %zu generators", m24.size());
  std::vector<IndexPerm> m24_idx;
  for (std::size_t k = 0; k < m24.size(); ++k) {
    CHECK(kiss::is_permutation(m24[k].perm), "gen %zu", k);
    CHECK(kiss::preserves_golay_code(m24[k].perm), "gen %zu does not preserve the Golay code", k);
    CHECK(m24[k].signs == 0, "gen %zu", k);
    try {
      m24_idx.push_back(kiss::index_permutation(L, m24[k]));
      CHECK(kiss::is_index_permutation(m24_idx.back()), "gen %zu not bijective on C", k);
      CHECK(kiss::inner_product_violations(L, m24_idx.back(), 20000, 7 + k) == 0, "gen %zu", k);
    } catch (const std::exception& e) {
      CHECK(false, "gen %zu: %s", k, e.what());
      m24_idx.push_back(kiss::index_identity());
    }
  }
  // Element orders of the coordinate permutations: 23, 2, 5.
  {
    const int expect[3] = {23, 2, 5};
    for (std::size_t k = 0; k < 3 && k < m24.size(); ++k) {
      Monomial p = m24[k];
      int ord = 1;
      while (!(p == kiss::monomial_identity()) && ord < 100) { p = kiss::compose(p, m24[k]); ++ord; }
      CHECK(ord == expect[k], "gen %zu has order %d", k, ord);
    }
  }
  // Rejections.
  {
    Monomial t = kiss::monomial_identity();
    t.perm[0] = 1; t.perm[1] = 0;
    CHECK(!kiss::preserves_golay_code(t.perm), "transposition (0 1)");
    CHECK(throws([&] { kiss::index_permutation(L, t); }), "transposition must be rejected on C");
    CHECK(throws([&] { kiss::index_permutation(L, kiss::sign_flip(0x7u)); }), "non-codeword sign flip must be rejected");
    // A weight-8 non-octad: octad with one bit moved.
    const uint32_t oct = kiss::golay_octads().front();
    int hi = 23; while (!((oct >> hi) & 1u)) --hi;
    int lo = 0; while ((oct >> lo) & 1u) ++lo;
    const uint32_t nonoct = (oct & ~(1u << hi)) | (1u << lo);
    CHECK(!kiss::golay_is_codeword(nonoct), "construction");
    CHECK(throws([&] { kiss::index_permutation(L, kiss::sign_flip(nonoct)); }), "weight-8 non-octad sign flip must be rejected");
  }
  // Sign flips on codewords, all-ones = −I.
  const Monomial oct_flip = kiss::sign_flip(kiss::golay_octads().front());
  const Monomial neg_I = kiss::sign_flip(0xFFFFFFu);
  IndexPerm oct_idx, neg_idx;
  try {
    oct_idx = kiss::index_permutation(L, oct_flip);
    neg_idx = kiss::index_permutation(L, neg_I);
    CHECK(neg_idx == L.neg, "−I must induce the neg table");
  } catch (const std::exception& e) { CHECK(false, "%s", e.what()); }
  // Compose / inverse consistency, Monomial vs IndexPerm.
  {
    std::vector<Monomial> pool = m24;
    pool.push_back(oct_flip);
    std::vector<IndexPerm> pool_idx = m24_idx;
    pool_idx.push_back(oct_idx);
    for (std::size_t a = 0; a < pool.size(); ++a) {
      CHECK(kiss::compose(pool[a], kiss::inverse(pool[a])) == kiss::monomial_identity(), "inverse %zu", a);
      CHECK(kiss::compose(kiss::inverse(pool[a]), pool[a]) == kiss::monomial_identity(), "inverse %zu", a);
      CHECK(kiss::index_permutation(L, kiss::inverse(pool[a])) == kiss::inverse(pool_idx[a]), "inverse idx %zu", a);
      CHECK(kiss::index_permutation(L, kiss::aut_from_monomial(pool[a])) == pool_idx[a], "aut_from_monomial %zu", a);
      for (std::size_t b = 0; b < pool.size(); ++b) {
        const Monomial ab = kiss::compose(pool[a], pool[b]);
        CHECK(kiss::index_permutation(L, ab) == kiss::compose(pool_idx[a], pool_idx[b]), "compose %zu %zu", a, b);
        // apply(a, apply(b, v)) == apply(a∘b, v) on a few vectors
        for (int i = 0; i < N; i += 9973) {
          const Vec v = L.C[static_cast<std::size_t>(i)];
          CHECK(kiss::apply(pool[a], kiss::apply(pool[b], v)) == kiss::apply(ab, v), "apply compose %zu %zu %d", a, b, i);
        }
      }
    }
    const auto mg = kiss::monomial_generators(group_dir, true);
    CHECK(mg.size() == 15, "monomial generators (basis signs): %zu", mg.size());
    uint32_t span_dim = 0;
    { std::set<uint32_t> span = {0};
      for (std::size_t k = 3; k < mg.size(); ++k) { std::set<uint32_t> nxt = span; for (uint32_t s : span) nxt.insert(s ^ mg[k].signs); span.swap(nxt); }
      span_dim = static_cast<uint32_t>(span.size()); }
    CHECK(span_dim == 4096, "basis sign flips span %u words", span_dim);
    CHECK(kiss::monomial_generators(group_dir).size() == 4, "monomial generators (default)");
  }
  std::printf("monomial section: %.1f ms\n", ms_since(t1));

  // ---- 2. ξ ------------------------------------------------------------------
  auto t2 = std::chrono::steady_clock::now();
  const kiss::Sextet sextet = kiss::find_sextet({0, 1, 2, 3});
  {
    std::printf("sextet:");
    for (const auto& t : sextet) std::printf(" {%d,%d,%d,%d}", t[0], t[1], t[2], t[3]);
    std::printf("\n");
    try { CHECK(kiss::load_sextet(group_dir / "sextet.txt") == sextet, "sextet.txt differs"); }
    catch (const std::exception& e) { CHECK(false, "%s", e.what()); }
    uint32_t all = 0;
    for (const auto& t : sextet) for (uint8_t i : t) all |= 1u << i;
    CHECK(all == 0xFFFFFFu, "sextet is not a partition");
    for (int a = 0; a < 6; ++a)
      for (int b = a + 1; b < 6; ++b) {
        uint32_t m = 0;
        for (uint8_t i : sextet[static_cast<std::size_t>(a)]) m |= 1u << i;
        for (uint8_t i : sextet[static_cast<std::size_t>(b)]) m |= 1u << i;
        CHECK(kiss::golay_is_codeword(m) && __builtin_popcount(m) == 8, "tetrads %d,%d do not form an octad", a, b);
      }
  }
  const Aut xi = kiss::make_xi(sextet);
  IndexPerm xi_idx;
  try {
    const Aut xi_file = kiss::load_aut(group_dir / "xi.txt");
    CHECK(xi_file == xi, "data/group/xi.txt differs from make_xi(sextet, {-1,1,1,1,1,1})");
    CHECK(xi.den == 2, "den");
    xi_idx = kiss::index_permutation(L, xi);
    CHECK(kiss::is_index_permutation(xi_idx), "ξ is not a bijection on C");
    CHECK(kiss::inner_product_violations(L, xi_idx, 100000, 99) == 0, "ξ inner products");
    CHECK(kiss::compose(xi_idx, xi_idx) == kiss::index_identity(), "ξ² != 1 on C");
    CHECK(kiss::compose(xi, xi) == kiss::aut_identity(), "ξ² != I as a matrix");
    CHECK(kiss::transpose(xi) == xi, "ξ symmetric");
    // Not monomial: every row has 4 nonzeros.
    for (int r = 0; r < DIM; ++r) {
      int nz = 0;
      for (int c = 0; c < DIM; ++c) nz += xi.num[static_cast<std::size_t>(r * DIM + c)] != 0;
      CHECK(nz == 4, "row %d has %d nonzeros", r, nz);
    }
    // Moves vertices of every shape to other shapes somewhere (sanity: not a
    // coordinate permutation in disguise).
    std::array<std::array<int, 3>, 3> shape_map{};
    for (int i = 0; i < N; i += 97) {
      const int s0 = kiss::leech_shape(L.C[static_cast<std::size_t>(i)]);
      const int s1 = kiss::leech_shape(L.C[xi_idx[static_cast<std::size_t>(i)]]);
      if (s0 >= 0 && s1 >= 0) ++shape_map[static_cast<std::size_t>(s0)][static_cast<std::size_t>(s1)];
    }
    std::printf("xi shape transitions (sampled, rows=from octad/31/44):");
    for (int a = 0; a < 3; ++a) { std::printf(" |"); for (int b = 0; b < 3; ++b) std::printf(" %d", shape_map[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)]); }
    std::printf("\n");
    CHECK(shape_map[0][1] + shape_map[0][2] + shape_map[1][0] + shape_map[2][0] > 0, "ξ preserves shapes?!");
    // Exact-divisibility rejection.
    Vec e0{}; e0[0] = 1;
    CHECK(throws([&] { kiss::apply(xi, e0); }), "apply(Aut) must throw on a non-integral image");
    Vec v0; CHECK(!kiss::try_apply(xi, e0, v0), "try_apply");
    // Round trip through save_aut/load_aut.
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / ("kiss_xi_" + std::to_string(::getpid()) + ".txt");
    kiss::save_aut(tmp, xi, "round trip\nsecond line");
    CHECK(kiss::load_aut(tmp) == xi, "save/load round trip");
    std::filesystem::remove(tmp);
  } catch (const std::exception& e) { CHECK(false, "ξ: %s", e.what()); xi_idx = kiss::index_identity(); }
  // All 32 sign patterns (global sign fixed by s_0 = +1 → 32 patterns ↔ 64/2).
  {
    int good = 0, bad = 0, odd_good = 0;
    for (int mask = 0; mask < 64; ++mask) {
      std::array<int, 6> s;
      int neg = 0;
      for (int t = 0; t < 6; ++t) { s[static_cast<std::size_t>(t)] = ((mask >> t) & 1) ? -1 : 1; neg += (mask >> t) & 1; }
      const Aut a = kiss::make_xi(sextet, s);
      bool ok = false;
      try { const IndexPerm g = kiss::index_permutation(L, a); ok = kiss::is_index_permutation(g); } catch (const std::runtime_error&) {}
      if (ok) { ++good; if (neg % 2 == 1) ++odd_good; } else ++bad;
      if (mask == 0) CHECK(!ok, "uniform-sign ξ must NOT be an automorphism (Python found 49104/196560 images in C)");
    }
    std::printf("xi sign patterns: %d automorphisms, %d not; all automorphisms have an odd number of negated tetrads: %d\n",
                good, bad, odd_good == good);
    CHECK(good == 32 && odd_good == 32, "expected exactly the 32 odd patterns");
  }
  std::printf("xi section: %.1f ms\n", ms_since(t2));

  // ---- 3. Product replacement ---------------------------------------------
  auto t3 = std::chrono::steady_clock::now();
  std::vector<IndexPerm> gens;
  try { gens = kiss::co0_generator_perms(L, group_dir); } catch (const std::exception& e) { CHECK(false, "%s", e.what()); }
  CHECK(gens.size() == 5, "co0 generators: %zu", gens.size());
  std::vector<IndexPerm> elems;
  int ip_fail = 0, bij_fail = 0;
  std::set<uint32_t> images_of_0;
  {
    kiss::ProductReplacement pr(gens, 10, seed);
    pr.burn_in(100);
    for (int k = 0; k < count; ++k) {
      const IndexPerm& g = pr.next();
      if (kiss::inner_product_violations(L, g, 10000, seed * 7919 + static_cast<uint64_t>(k)) != 0) ++ip_fail;
      if (!kiss::is_index_permutation(g)) ++bij_fail;
      images_of_0.insert(g[0]);
      if (k < apply_n) elems.push_back(g);
    }
    CHECK(pr.slots() == 10, "slots");
    CHECK(pr.steps() == static_cast<uint64_t>(100 + count), "steps");
  }
  CHECK(ip_fail == 0, "%d elements violate inner-product preservation", ip_fail);
  CHECK(bij_fail == 0, "%d elements not bijective", bij_fail);
  std::printf("product replacement: %d elements, ip_fail=%d bij_fail=%d distinct images of vertex 0: %zu, %.1f ms\n",
              count, ip_fail, bij_fail, images_of_0.size(), ms_since(t3));
  // Random elements should scatter vertex 0 (transitivity): expect ~count distinct images.
  CHECK(images_of_0.size() > static_cast<std::size_t>(count) * 9 / 10, "vertex 0 only reaches %zu images", images_of_0.size());
  // Determinism: same seed → same first element.
  {
    kiss::ProductReplacement a(gens, 10, seed), b(gens, 10, seed);
    a.burn_in(100); b.burn_in(100);
    CHECK(a.next() == b.next(), "product replacement is not deterministic for a fixed seed");
  }

  // ---- 4. Apply to S ----------------------------------------------------------
  auto t4 = std::chrono::steady_clock::now();
  std::vector<Vec> S;
  std::string set_name;
  if (set_path.empty()) {
    for (const char* cand : {"data/S496.txt", "data/S488.txt"})
      if (std::filesystem::exists(cand)) { set_path = cand; break; }
  }
  if (!set_path.empty()) {
    try { S = kiss::read_set(set_path); set_name = set_path; }
    catch (const std::exception& e) { CHECK(false, "read_set(%s): %s", set_path.c_str(), e.what()); }
  }
  if (S.empty()) {
    S = greedy_independent_set(L, seed);
    set_name = "greedy(seed=" + std::to_string(seed) + ")";
  }
  std::string why;
  CHECK(is_independent(S, &why), "S itself is not independent: %s", why.c_str());
  std::vector<uint32_t> S_idx;
  try { S_idx = indices_of(L, S); } catch (const std::exception& e) { CHECK(false, "%s", e.what()); }
  std::vector<uint32_t> S_sorted = S_idx;
  std::sort(S_sorted.begin(), S_sorted.end());
  CHECK(std::adjacent_find(S_sorted.begin(), S_sorted.end()) == S_sorted.end(), "S has repeated vectors");
  std::printf("S = %s, |S| = %zu\n", set_name.c_str(), S.size());
  std::map<std::size_t, int> hist;
  double overlap_sum = 0;
  int image_fail = 0;
  for (std::size_t k = 0; k < elems.size(); ++k) {
    const std::vector<Vec> gS = kiss::apply(L, elems[k], S);
    std::string w;
    bool ok = gS.size() == S.size() && is_independent(gS, &w);
    // distinctness
    std::vector<uint32_t> gidx = kiss::apply(elems[k], S_idx);
    std::sort(gidx.begin(), gidx.end());
    if (std::adjacent_find(gidx.begin(), gidx.end()) != gidx.end()) { ok = false; w = "repeated"; }
    if (!ok) { ++image_fail; std::printf("  image %zu not independent: %s\n", k, w.c_str()); }
    const std::size_t ov = overlap(S_sorted, gidx);
    ++hist[ov];
    overlap_sum += static_cast<double>(ov);
  }
  CHECK(image_fail == 0, "%d images are not independent sets of size |S|", image_fail);
  const double expected = static_cast<double>(S.size()) * static_cast<double>(S.size()) / N;
  std::printf("overlap |gS ∩ S| over %zu random elements: mean=%.3f expected(random)=%.3f histogram:",
              elems.size(), elems.empty() ? 0.0 : overlap_sum / static_cast<double>(elems.size()), expected);
  for (const auto& [ov, c] : hist) std::printf(" %zu:%d", ov, c);
  std::printf("\n");
  std::printf("apply section: %.1f ms\n", ms_since(t4));

  // ---- 5. Observations ----------------------------------------------------------
  {
    std::array<int, 3> shapes{};
    for (const Vec& v : S) { const int s = kiss::leech_shape(v); if (s >= 0) ++shapes[static_cast<std::size_t>(s)]; }
    std::printf("S shape counts (octad, 31, 44): %d %d %d\n", shapes[0], shapes[1], shapes[2]);
    const char* names[] = {"alpha", "gamma", "delta", "octad_flip", "neg_I", "xi"};
    std::vector<const IndexPerm*> gl = {&m24_idx[0], &m24_idx[1], &m24_idx[2], &oct_idx, &neg_idx, &xi_idx};
    std::printf("generator invariance of S:");
    for (std::size_t k = 0; k < gl.size(); ++k) {
      const std::size_t ov = overlap(S_sorted, kiss::apply(*gl[k], S_idx));
      std::printf(" %s:|gS∩S|=%zu%s", names[k], ov, ov == S.size() ? "(INVARIANT)" : "");
    }
    std::printf("\n");
    // Random monomial elements.
    const int mono_n = 2000;
    kiss::ProductReplacement pm(kiss::monomial_generator_perms(L, group_dir), 10, seed + 1);
    pm.burn_in(100);
    int fixed = 0; std::map<std::size_t, int> mh; double msum = 0;
    for (int k = 0; k < mono_n; ++k) {
      const std::size_t ov = overlap(S_sorted, kiss::apply(pm.next(), S_idx));
      if (ov == S.size()) ++fixed;
      ++mh[ov]; msum += static_cast<double>(ov);
    }
    std::printf("random monomial elements (%d): fixing S as a set: %d; overlap mean=%.3f histogram:", mono_n, fixed, msum / mono_n);
    for (const auto& [ov, c] : mh) std::printf(" %zu:%d", ov, c);
    std::printf("\n");
  }

  const double total = ms_since(t_start);
  std::printf("RESULT ok=%d m24_gens=%zu xi_bijective=%d xi_sign_patterns_ok=32 pr_elements=%d ip_fail=%d bij_fail=%d "
              "applied=%zu image_fail=%d set=%s size=%zu overlap_mean=%.3f ms=%.1f\n",
              g_failures == 0 ? 1 : 0, m24.size(), kiss::is_index_permutation(xi_idx) ? 1 : 0, count, ip_fail, bij_fail,
              elems.size(), image_fail, set_name.c_str(), S.size(),
              elems.empty() ? 0.0 : overlap_sum / static_cast<double>(elems.size()), total);
  return g_failures == 0 ? 0 : 1;
}

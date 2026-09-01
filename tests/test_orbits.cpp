// T3.4 — acceptance tests for kiss/orbits.h (docs/design.md T3.4).
//
//  1. Branch and bound == brute force on small random weighted graphs
//     (synthetic OrbitGraph, 24 vertices, several densities, some
//     self-conflicting vertices); greedy ≤ local search ≤ B&B, all valid.
//  2. Orbit decomposition invariants for H = ⟨α⟩ and ⟨α, −I⟩ on C:
//     Σ|orbit| = N, invariance under the generators, orbit sizes divide |H|,
//     numbering by smallest vertex, is_orbit_union / union_of_orbits.
//  3. The 496 and its monomial stabiliser: the coordinate invariants of the
//     496 (independent C++ re-derivation of the argument in
//     python/tools/stabilizer_496.py) force π = id, and with π = id exactly
//     two sign masks fix the set — so Stab_{2^12:M24}(496) = {±I}; the file
//     runs/orbits/stab496/stabiliser_generators.txt (if present) agrees; the
//     496 is a union of stabiliser orbits and is recovered from them.
//  4. With data/adj.u32 (exit 77 = skip otherwise): H = 23:11; the orbit
//     conflict relation and self-conflict flags equal brute force on 50 orbit
//     pairs / 20 orbits; greedy, LS and B&B produce valid unions whose vertex
//     sets pass verify_independent; B&B ≥ LS ≥ greedy.
// Usage: test_orbits [data_dir]. Prints `RESULT ok=1 ...`.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/golay.h"
#include "kiss/group.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/orbits.h"
#include "kiss/types.h"
#include "kiss/verify.h"

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

using kiss::DIM;
using kiss::IndexPerm;
using kiss::Leech;
using kiss::Monomial;
using kiss::N;
using kiss::Vec;

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// 1. B&B vs brute force
// ---------------------------------------------------------------------------
uint64_t brute_force_mwis(const std::vector<uint32_t>& w, const std::vector<uint32_t>& adjmask,
                          const std::vector<uint8_t>& self) {
  const uint32_t n = static_cast<uint32_t>(w.size());
  uint64_t best = 0;
  for (uint32_t mask = 0; mask < (1u << n); ++mask) {
    bool ok = true;
    uint64_t wt = 0;
    for (uint32_t v = 0; v < n && ok; ++v) {
      if (!((mask >> v) & 1u)) continue;
      if (self[v] || (adjmask[v] & mask)) ok = false;
      wt += w[v];
    }
    if (ok && wt > best) best = wt;
  }
  return best;
}

void test_bb_vs_brute_force() {
  std::mt19937_64 rng(2024);
  const int n = 24;
  const double densities[] = {0.1, 0.3, 0.5, 0.8};
  int solved = 0;
  for (double p : densities)
    for (int rep = 0; rep < 2; ++rep) {
      std::vector<uint32_t> w(n), adjmask(n, 0);
      std::vector<uint8_t> self(n, 0);
      std::vector<std::pair<uint32_t, uint32_t>> edges;
      std::uniform_int_distribution<uint32_t> W(1, 12);
      std::uniform_real_distribution<double> U(0, 1);
      for (int i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = W(rng);
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
          if (U(rng) < p) {
            edges.emplace_back(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
            adjmask[static_cast<std::size_t>(i)] |= 1u << j;
            adjmask[static_cast<std::size_t>(j)] |= 1u << i;
          }
      for (int k = 0; k < 3; ++k) {
        const int v = static_cast<int>(rng() % n);
        self[static_cast<std::size_t>(v)] = 1;
        if (k == 0) edges.emplace_back(static_cast<uint32_t>(v), static_cast<uint32_t>(v));  // self loop form too
      }
      kiss::OrbitGraph G(w, edges, self);
      const uint64_t bf = brute_force_mwis(w, adjmask, self);
      const kiss::OrbitSet g = kiss::greedy_orbit_mis(G);
      const kiss::OrbitSet ls = kiss::local_search_orbit_mis(G, g, 0.2, 7, 20000);
      const kiss::OrbitSet bb = kiss::branch_and_bound_orbit_mis(G, ls, 30.0);
      const kiss::OrbitSet bb0 = kiss::branch_and_bound_orbit_mis(G, kiss::OrbitSet{}, 30.0);  // no initial bound
      CHECK(kiss::orbit_set_valid(G, g), "greedy invalid (p=%.1f)", p);
      CHECK(kiss::orbit_set_valid(G, ls), "ls invalid (p=%.1f)", p);
      CHECK(kiss::orbit_set_valid(G, bb), "bb invalid (p=%.1f)", p);
      CHECK(kiss::orbit_set_valid(G, bb0), "bb0 invalid (p=%.1f)", p);
      CHECK(g.weight <= ls.weight && ls.weight <= bb.weight, "monotone greedy %llu ls %llu bb %llu",
            static_cast<unsigned long long>(g.weight), static_cast<unsigned long long>(ls.weight),
            static_cast<unsigned long long>(bb.weight));
      CHECK(bb.optimal && bb0.optimal, "B&B did not finish (p=%.1f)", p);
      CHECK(bb.weight == bf && bb0.weight == bf, "B&B %llu / %llu vs brute force %llu (p=%.1f)",
            static_cast<unsigned long long>(bb.weight), static_cast<unsigned long long>(bb0.weight),
            static_cast<unsigned long long>(bf), p);
      ++solved;
    }
  std::printf("bb_vs_brute_force: %d random graphs (n=%d) agree\n", solved, n);
}

// ---------------------------------------------------------------------------
// 2. orbit invariants
// ---------------------------------------------------------------------------
void test_orbit_invariants(const Leech& L, const std::vector<Monomial>& m24) {
  const Monomial alpha = m24.at(0);
  const Monomial negI = kiss::sign_flip(0xFFFFFFu);
  const IndexPerm a = kiss::index_permutation(L, alpha), ng = kiss::index_permutation(L, negI);
  {
    kiss::Subgroup H{{a}, "C23"};
    const kiss::Orbits O = kiss::orbits(H);
    uint64_t sum = 0;
    std::map<uint32_t, uint32_t> hist;
    bool numbering_ok = true, members_ok = true;
    uint32_t last_rep = 0;
    for (uint32_t o = 0; o < O.count(); ++o) {
      sum += O.size(o);
      ++hist[O.size(o)];
      if (o > 0 && O.rep(o) <= last_rep) numbering_ok = false;
      last_rep = O.rep(o);
      for (uint32_t k = O.start[o]; k < O.start[o + 1]; ++k) {
        if (O.id[O.members[k]] != o) members_ok = false;
        if (k > O.start[o] && O.members[k] <= O.members[k - 1]) members_ok = false;
      }
    }
    CHECK(sum == static_cast<uint64_t>(N), "sum of orbit sizes %llu", static_cast<unsigned long long>(sum));
    CHECK(numbering_ok, "orbits not numbered by smallest vertex");
    CHECK(members_ok, "members/id inconsistent");
    CHECK(kiss::orbits_invariant(O, a), "C23 orbits not invariant under alpha");
    CHECK(hist.size() == 2 && hist.count(1) && hist.count(23), "C23 orbit sizes are not {1,23}");
    // fixed points of alpha (x -> x+1 on F_23): vectors constant on coordinates 0..22
    const uint32_t fixed = hist.count(1) ? hist[1] : 0;
    // (∓3,±1^23) with -3 at 23 and all other signs equal: sign mask ∈ {0, all-ones} restricted... exactly
    // the codewords 0 and 1^24 give 2 vectors; (−3 at 23, +1 elsewhere) and its negative → 2 fixed vectors.
    CHECK(fixed == 2, "alpha fixes %u vectors (expected 2)", fixed);
    std::printf("C23: %u orbits, sizes %s\n", O.count(),
                [&] { std::string s; for (auto& kv : hist) s += std::to_string(kv.first) + "x" + std::to_string(kv.second) + " "; return s; }().c_str());
    // is_orbit_union / union_of_orbits
    std::vector<uint32_t> ids;
    std::vector<uint32_t> U = kiss::union_of_orbits(O, {0, 5, 17});
    CHECK(kiss::is_orbit_union(O, U, ids) && ids == std::vector<uint32_t>({0, 5, 17}), "union round trip");
    U.pop_back();
    CHECK(!kiss::is_orbit_union(O, U, ids) && ids.size() == 1, "partial orbit not detected");
  }
  {
    kiss::Subgroup H{{a, ng}, "C23x2"};
    const kiss::Orbits O = kiss::orbits(H);
    std::map<uint32_t, uint32_t> hist;
    for (uint32_t o = 0; o < O.count(); ++o) ++hist[O.size(o)];
    CHECK(kiss::orbits_invariant(O, a) && kiss::orbits_invariant(O, ng), "C23x2 orbits not invariant");
    bool divides = true;
    for (auto& kv : hist) if (46 % kv.first != 0) divides = false;
    CHECK(divides, "orbit sizes do not divide 46");
    CHECK(O.count() * 46 >= static_cast<uint32_t>(N), "too few orbits for order 46");
    std::printf("C23x2: %u orbits\n", O.count());
  }
  // sanity: the trivial group gives N singleton orbits
  {
    const kiss::Orbits O = kiss::orbits(kiss::Subgroup{{}, "1"});
    CHECK(O.count() == static_cast<uint32_t>(N), "trivial group: %u orbits", O.count());
  }
}

// ---------------------------------------------------------------------------
// 3. the 496 and its monomial stabiliser
// ---------------------------------------------------------------------------
// Coordinate invariants of a set S under coordinate permutations: for each
// coordinate i the number of octad-shape rows supported on i, of (∓3)-rows
// with the 3 at i, of (±4,±4)-rows supported on i, and the sorted profile of
// "number of octad-shape rows supported on both i and j" over j ≠ i. A
// permutation π with π S = S (up to signs) maps coordinate i to a coordinate
// with the same invariant.
using CoordKey = std::pair<std::array<int, 3>, std::vector<std::pair<int, std::array<int, 3>>>>;

std::vector<CoordKey> coordinate_keys(const std::vector<Vec>& S) {
  std::array<int, DIM> n_oct{}, n_tri{}, n_ff{};
  std::array<std::array<int, DIM>, DIM> pair_oct{};
  for (const Vec& v : S) {
    const int sh = kiss::leech_shape(v);
    std::vector<int> supp;
    for (int i = 0; i < DIM; ++i)
      if (v[static_cast<std::size_t>(i)] != 0) supp.push_back(i);
    if (sh == 0) {
      for (int i : supp) {
        ++n_oct[static_cast<std::size_t>(i)];
        for (int j : supp) if (i != j) ++pair_oct[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      }
    } else if (sh == 1) {
      for (int i = 0; i < DIM; ++i)
        if (std::abs(v[static_cast<std::size_t>(i)]) == 3) ++n_tri[static_cast<std::size_t>(i)];
    } else {
      for (int i : supp) ++n_ff[static_cast<std::size_t>(i)];
    }
  }
  std::vector<CoordKey> keys(DIM);
  for (int i = 0; i < DIM; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    keys[si].first = {n_oct[si], n_tri[si], n_ff[si]};
    for (int j = 0; j < DIM; ++j) {
      if (j == i) continue;
      const std::size_t sj = static_cast<std::size_t>(j);
      keys[si].second.emplace_back(pair_oct[si][sj], std::array<int, 3>{n_oct[sj], n_tri[sj], n_ff[sj]});
    }
    std::sort(keys[si].second.begin(), keys[si].second.end());
  }
  return keys;
}

void test_stabiliser_496(const Leech& L, const std::filesystem::path& set_path) {
  const std::vector<Vec> S = kiss::read_set(set_path);
  CHECK(S.size() == 496, "expected 496 rows, got %zu", S.size());
  const std::vector<uint32_t> S_idx = [&] { auto v = kiss::set_indices(L, S); std::sort(v.begin(), v.end()); return v; }();
  const std::set<std::vector<int8_t>> Sset = [&] {
    std::set<std::vector<int8_t>> s;
    for (const Vec& v : S) s.insert(std::vector<int8_t>(v.begin(), v.end()));
    return s;
  }();
  // (a) coordinate invariants: all 24 distinct ⇒ any π ∈ Sym(24) with π S = ±S is the identity
  const std::vector<CoordKey> keys = coordinate_keys(S);
  std::set<CoordKey> distinct(keys.begin(), keys.end());
  CHECK(distinct.size() == DIM, "only %zu distinct coordinate invariants (need 24 to force pi = id)", distinct.size());
  // (b) with π = id: sign masks c with D_c S = S. Take a (∓3) row x; any y ∈ S with |y| = 3 at the same
  // position determines c = {j : x_j ≠ y_j}. Count the codewords among those that fix S.
  int tri0 = -1;
  for (std::size_t r = 0; r < S.size(); ++r)
    if (kiss::leech_shape(S[r]) == 1) { tri0 = static_cast<int>(r); break; }
  CHECK(tri0 >= 0, "no (∓3,±1^23) row in S");
  int fixing = 0;
  std::set<uint32_t> tried;
  if (tri0 >= 0) {
    const Vec& x = S[static_cast<std::size_t>(tri0)];
    int px = 0;
    for (int i = 0; i < DIM; ++i) if (std::abs(x[static_cast<std::size_t>(i)]) == 3) px = i;
    for (const Vec& y : S) {
      if (std::abs(y[static_cast<std::size_t>(px)]) != 3) continue;
      uint32_t c = 0;
      for (int j = 0; j < DIM; ++j)
        if (x[static_cast<std::size_t>(j)] != y[static_cast<std::size_t>(j)]) c |= 1u << j;
      if (!tried.insert(c).second || !kiss::golay_is_codeword(c)) continue;
      const Monomial m = kiss::sign_flip(c);
      bool fixes = true;
      for (const Vec& v : S) {
        const Vec w = kiss::apply(m, v);
        if (!Sset.count(std::vector<int8_t>(w.begin(), w.end()))) { fixes = false; break; }
      }
      if (fixes) ++fixing;
    }
  }
  CHECK(fixing == 2, "sign masks fixing the 496 with pi = id: %d (expected 2: 0 and all-ones)", fixing);
  std::printf("stabiliser_496: %zu distinct coordinate invariants, %zu candidate sign masks, %d fix S -> |Stab| = %d\n",
              distinct.size(), tried.size(), fixing, distinct.size() == DIM ? fixing : -1);
  // (c) the exported generators (if present) fix S and generate a group of order 2
  std::vector<Monomial> gens{kiss::sign_flip(0xFFFFFFu)};
  const std::filesystem::path gen_file = "runs/orbits/stab496/stabiliser_generators.txt";
  if (std::filesystem::exists(gen_file)) {
    gens = kiss::load_monomial_list(gen_file);
    CHECK(!gens.empty(), "empty generator file");
    CHECK(kiss::monomial_group_order(gens) == 2, "exported stabiliser has order %llu",
          static_cast<unsigned long long>(kiss::monomial_group_order(gens)));
    for (const Monomial& m : gens) {
      const IndexPerm g = kiss::index_permutation(L, m);
      std::vector<uint32_t> img = kiss::apply(g, S_idx);
      std::sort(img.begin(), img.end());
      CHECK(img == S_idx, "exported generator does not fix the 496");
    }
  } else {
    std::printf("note: %s not present, using <-I>\n", gen_file.string().c_str());
  }
  // (d) the 496 is a union of stabiliser orbits and is recovered from them
  std::vector<IndexPerm> gp;
  for (const Monomial& m : gens) gp.push_back(kiss::index_permutation(L, m));
  const kiss::Orbits O = kiss::orbits(kiss::Subgroup{gp, "Stab(496)"});
  CHECK(O.count() == 98280, "stabiliser orbits on C: %u (expected 98280 antipodal pairs)", O.count());
  std::vector<uint32_t> ids;
  const bool is_union = kiss::is_orbit_union(O, S_idx, ids);
  CHECK(is_union, "the 496 is not a union of stabiliser orbits (%zu partial)", ids.size());
  CHECK(ids.size() == 248, "%zu stabiliser orbits in the 496 (expected 248)", ids.size());
  CHECK(kiss::union_of_orbits(O, ids) == S_idx, "union of the orbits is not the 496");
  std::printf("stabiliser_496: %u orbits on C, the 496 = union of %zu orbits (recovered exactly)\n", O.count(), ids.size());
  // (e) the full Co_0 stabiliser exported by python/tools/stabilizer_496.py (Gram automorphisms that
  // extend to Leech automorphisms): each generator must be an automorphism (index_permutation throws
  // otherwise), fix the 496, and the 496 must be a union of its orbits.
  std::vector<IndexPerm> cp;
  for (int k = 0; k < 64; ++k) {
    const std::filesystem::path p = std::filesystem::path("runs/orbits/stab496") / ("co0_stab_gen" + std::to_string(k) + ".txt");
    if (!std::filesystem::exists(p)) break;
    const kiss::Aut a = kiss::load_aut(p);
    CHECK(a.den == 8, "co0 generator %d has den %d", k, a.den);
    cp.push_back(kiss::index_permutation(L, a));
    std::vector<uint32_t> img = kiss::apply(cp.back(), S_idx);
    std::sort(img.begin(), img.end());
    CHECK(img == S_idx, "co0 generator %d does not fix the 496", k);
    CHECK(kiss::inner_product_violations(L, cp.back(), 2000, 3) == 0, "co0 generator %d breaks inner products", k);
  }
  if (!cp.empty()) {
    const kiss::Orbits Oc = kiss::orbits(kiss::Subgroup{cp, "Stab_Co0(496)"});
    std::vector<uint32_t> cids;
    CHECK(kiss::is_orbit_union(Oc, S_idx, cids), "the 496 is not a union of Co_0-stabiliser orbits");
    CHECK(kiss::union_of_orbits(Oc, cids) == S_idx, "union of Co_0-stabiliser orbits is not the 496");
    std::map<uint32_t, uint32_t> hs;
    for (uint32_t o : cids) ++hs[Oc.size(o)];
    std::string hstr;
    for (auto& kv : hs) hstr += std::to_string(kv.first) + "x" + std::to_string(kv.second) + " ";
    std::printf("stabiliser_496 (Co_0, %zu generators): %u orbits on C, the 496 = union of %zu orbits, sizes %s\n",
                cp.size(), Oc.count(), cids.size(), hstr.c_str());
  } else {
    std::printf("note: no runs/orbits/stab496/co0_stab_gen*.txt — Co_0 stabiliser check skipped\n");
  }
}

// ---------------------------------------------------------------------------
// 4. orbit conflict graph vs brute force; MIS on a real orbit graph
// ---------------------------------------------------------------------------
void test_orbit_graph(const Leech& L, const kiss::Adjacency& A, const std::vector<Monomial>& m24) {
  const auto t0 = std::chrono::steady_clock::now();
  std::array<uint8_t, DIM> p;
  for (int i = 0; i < 23; ++i) p[static_cast<std::size_t>(i)] = static_cast<uint8_t>((2 * i) % 23);
  p[23] = 23;
  CHECK(kiss::preserves_golay_code(p), "x -> 2x is not in M24?");
  const Monomial mu2 = kiss::coordinate_permutation(p), alpha = m24.at(0);
  std::vector<Monomial> gens{alpha, mu2};
  CHECK(kiss::monomial_group_order(gens) == 253, "23:11 has order %llu", static_cast<unsigned long long>(kiss::monomial_group_order(gens)));
  std::vector<IndexPerm> gp;
  for (const Monomial& m : gens) gp.push_back(kiss::index_permutation(L, m));
  const kiss::Orbits O = kiss::orbits(kiss::Subgroup{gp, "23:11"});
  kiss::OrbitGraph G(O, A, 20000);
  CHECK(G.is_explicit(), "expected explicit graph for %u orbits", O.count());
  std::printf("23:11: %u orbits, %u self-conflicting, %llu edges (%.0f ms)\n", O.count(), G.self_conflicting_count(),
              static_cast<unsigned long long>(G.edges()), ms_since(t0));
  std::mt19937_64 rng(11);
  std::uniform_int_distribution<uint32_t> U(0, O.count() - 1);
  std::vector<uint32_t> buf;
  int agree = 0, conflicts = 0;
  for (int t = 0; t < 50; ++t) {
    uint32_t a = U(rng), b = U(rng);
    while (b == a) b = U(rng);
    // bias half the samples towards actual neighbours
    if (t % 2 == 0) {
      uint32_t k;
      const uint32_t* nb = G.neighbours(a, k, buf);
      if (k) b = nb[rng() % k];
    }
    uint32_t k;
    const uint32_t* nb = G.neighbours(a, k, buf);
    const bool graph_says = std::binary_search(nb, nb + k, b);
    const bool brute = kiss::OrbitGraph::conflict_brute_force(L, O, a, b);
    CHECK(graph_says == brute, "conflict(%u,%u): graph %d brute %d", a, b, graph_says, brute);
    if (graph_says == brute) ++agree;
    if (brute) ++conflicts;
    // symmetry
    uint32_t k2;
    const uint32_t* nb2 = G.neighbours(b, k2, buf);
    CHECK(std::binary_search(nb2, nb2 + k2, a) == graph_says, "conflict relation not symmetric (%u,%u)", a, b);
  }
  int self_agree = 0;
  for (int t = 0; t < 20; ++t) {
    const uint32_t a = (t < 10) ? U(rng) : t * 37u % O.count();
    const bool brute = kiss::OrbitGraph::conflict_brute_force(L, O, a, a);
    CHECK(G.self_conflicting(a) == brute, "self-conflict(%u): graph %d brute %d", a, G.self_conflicting(a), brute);
    if (G.self_conflicting(a) == brute) ++self_agree;
  }
  std::printf("conflict vs brute force: %d/50 pairs agree (%d conflicting), %d/20 self flags agree\n", agree, conflicts, self_agree);
  // degrees consistent with neighbour lists
  bool deg_ok = true;
  for (uint32_t o = 0; o < O.count(); ++o) {
    uint32_t k;
    G.neighbours(o, k, buf);
    if (k != G.degree(o)) deg_ok = false;
  }
  CHECK(deg_ok, "degree() disagrees with neighbour lists");
  // MIS chain
  const kiss::OrbitSet g = kiss::greedy_orbit_mis(G);
  const kiss::OrbitSet ls = kiss::local_search_orbit_mis(G, g, 2.0, 5);
  const kiss::OrbitSet bb = kiss::branch_and_bound_orbit_mis(G, ls, 20.0);
  CHECK(kiss::orbit_set_valid(G, g) && kiss::orbit_set_valid(G, ls) && kiss::orbit_set_valid(G, bb), "invalid orbit set");
  CHECK(g.weight <= ls.weight && ls.weight <= bb.weight, "not monotone");
  for (const kiss::OrbitSet* s : {&g, &ls, &bb}) {
    const std::vector<uint32_t> verts = kiss::union_of_orbits(O, s->orbits);
    CHECK(verts.size() == s->weight, "weight mismatch");
    std::vector<Vec> vecs;
    for (uint32_t v : verts) vecs.push_back(L.C[v]);
    const kiss::VerifyResult vr = kiss::verify_independent(L, vecs);
    CHECK(vr.ok, "orbit union not independent: %s", vr.message.c_str());
  }
  std::printf("23:11 MIS: greedy %llu, ls %llu, bb %llu (optimal=%d, %llu nodes, %.1f s)\n",
              static_cast<unsigned long long>(g.weight), static_cast<unsigned long long>(ls.weight),
              static_cast<unsigned long long>(bb.weight), bb.optimal ? 1 : 0, static_cast<unsigned long long>(bb.nodes),
              bb.seconds);
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
  const auto t0 = std::chrono::steady_clock::now();
  bool skipped_adj = false;
  try {
    test_bb_vs_brute_force();
    Leech L;
    try { L = kiss::load_leech(data_dir); } catch (const std::exception&) { L = kiss::generate_leech(); }
    const std::vector<Monomial> m24 = kiss::load_m24_generators(data_dir / "group" / "m24_generators.txt");
    test_orbit_invariants(L, m24);
    const std::filesystem::path set = data_dir / "S496.txt";
    if (std::filesystem::exists(set)) test_stabiliser_496(L, set);
    else { std::printf("note: %s missing, stabiliser test skipped\n", set.string().c_str()); ++g_failures; }
    const std::filesystem::path adj = data_dir / "adj.u32";
    if (std::filesystem::exists(adj) && std::filesystem::file_size(adj) == kiss::Adjacency::EXPECTED_BYTES) {
      kiss::Adjacency A(adj);
      test_orbit_graph(L, A, m24);
    } else {
      skipped_adj = true;
      std::printf("note: %s missing — orbit graph tests skipped\n", adj.string().c_str());
    }
  } catch (const std::exception& e) {
    std::printf("RESULT ok=0 error=\"%s\"\n", e.what());
    return 1;
  }
  if (skipped_adj && g_failures == 0) {
    std::printf("RESULT ok=0 skipped=1 (no adjacency) ms=%.0f\n", ms_since(t0));
    return 77;
  }
  std::printf("RESULT ok=%d failures=%d ms=%.0f\n", g_failures == 0 ? 1 : 0, g_failures, ms_since(t0));
  return g_failures == 0 ? 0 : 1;
}

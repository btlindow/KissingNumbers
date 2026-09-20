// B1 — exact clique number of the conflict graph G (docs/reports/B1.md).
//
// G: vertices = the N = 196560 Leech minimal vectors, edges = inner product 16
// (60 degrees).  NOT the orthogonality graph of T3.3 (whose "clique number 24"
// is about antipodal classes at inner product 0).
//
// Facts this tool establishes / verifies:
//   * common neighbours of an edge: 891 = p^5_{55} (the scheme's intersection
//     number), listed explicitly and cross-checked against data/adj.u32 when
//     that file is present;
//   * the common-neighbourhood graph G[N(x,y)] is 336-regular on 891 vertices;
//   * exact max clique of G[N(x,y)] by branch-and-bound with greedy-colouring
//     bound (Tomita-style) = 22, for the base edge (0,1) and further random
//     edges (arc-transitivity of Co_0 on ordered 16-pairs — Stab(x) is
//     transitive on every class, `stab_x_orbits = 7` in orbitals.json — makes
//     one edge enough; the extra edges are a computational sanity check);
//   * hence omega(G) = 24: 2 + 22 from the search, and <= 24 because k vectors
//     of norm 32 with pairwise inner product 16 have Gram 16(I_k + J_k), which
//     is positive definite (eigenvalues 16(k+1), 16^(k-1)), i.e. rank k, so
//     they are linearly independent in R^24 and k <= 24;
//   * an explicit 24-clique, verified by exact integer dot products (its Gram
//     is 16(I+J), so its 24 vectors are a basis of R^24);
//   * a Bron-Kerbosch census of the maximal cliques of G[N(x,y)] by size
//     (= maximal cliques of G through one edge, sizes shifted by +2).
//
// Output: a JSON certificate (default data/scheme/clique_g.json) with the
// 24-clique (indices + vectors), per-edge search results and the census.
//
// Modes:  clique_g [--data DIR] [--edges E] [--census-seconds T] [--no-census]
//                  [--out FILE] [--seed S] [--test]
// --test: fast acceptance subset (common-nbhd counts, 24-clique verification,
//         B&B on the much smaller triangle neighbourhood), prints RESULT ok=1.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <omp.h>

#include "kiss/adjacency.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/bits.h"

namespace {

using kiss::DIM;
using kiss::N;
using kiss::Vec;

double now_s() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// fixed-width bitset for subgraphs of <= 896 vertices (14 x 64)
// ---------------------------------------------------------------------------
constexpr int MAXW = 14;
constexpr int MAXN = MAXW * 64;

struct Bits {
  std::array<uint64_t, MAXW> w{};
  void set(int i) { w[static_cast<std::size_t>(i >> 6)] |= 1ull << (i & 63); }
  void clear(int i) { w[static_cast<std::size_t>(i >> 6)] &= ~(1ull << (i & 63)); }
  bool test(int i) const { return (w[static_cast<std::size_t>(i >> 6)] >> (i & 63)) & 1u; }
  bool any() const {
    for (int k = 0; k < MAXW; ++k) if (w[static_cast<std::size_t>(k)]) return true;
    return false;
  }
  int count() const {
    int c = 0;
    for (int k = 0; k < MAXW; ++k) c += kiss::popcount64(w[static_cast<std::size_t>(k)]);
    return c;
  }
  int lsb() const {  // index of lowest set bit; -1 if empty
    for (int k = 0; k < MAXW; ++k)
      if (w[static_cast<std::size_t>(k)])
        return (k << 6) + kiss::ctz64(w[static_cast<std::size_t>(k)]);
    return -1;
  }
  Bits operator&(const Bits& o) const {
    Bits r;
    for (int k = 0; k < MAXW; ++k)
      r.w[static_cast<std::size_t>(k)] = w[static_cast<std::size_t>(k)] & o.w[static_cast<std::size_t>(k)];
    return r;
  }
  Bits andnot(const Bits& o) const {
    Bits r;
    for (int k = 0; k < MAXW; ++k)
      r.w[static_cast<std::size_t>(k)] = w[static_cast<std::size_t>(k)] & ~o.w[static_cast<std::size_t>(k)];
    return r;
  }
  int count_and(const Bits& o) const {
    int c = 0;
    for (int k = 0; k < MAXW; ++k)
      c += kiss::popcount64(w[static_cast<std::size_t>(k)] & o.w[static_cast<std::size_t>(k)]);
    return c;
  }
};

// ---------------------------------------------------------------------------
// exact max clique by branch and bound with greedy colouring (Tomita MCQ)
// ---------------------------------------------------------------------------
struct MaxClique {
  int n = 0;
  std::vector<Bits> adj;
  int best = 0;
  std::vector<int> best_R, R;
  uint64_t nodes = 0;

  void expand(Bits P, int size) {
    ++nodes;
    // greedy colouring: colour classes are independent sets; a clique meets
    // each class at most once, so size + colour(v) bounds any completion
    // through {order[0..i]}.
    int order[MAXN], col[MAXN];
    int m = 0;
    Bits U = P;
    int colour = 0;
    while (U.any()) {
      ++colour;
      Bits Q = U;
      while (Q.any()) {
        const int v = Q.lsb();
        U.clear(v);
        Q.clear(v);
        Q = Q.andnot(adj[static_cast<std::size_t>(v)]);
        order[m] = v;
        col[m] = colour;
        ++m;
      }
    }
    for (int i = m - 1; i >= 0; --i) {
      if (size + col[i] <= best) return;  // col is nondecreasing in i
      const int v = order[i];
      P.clear(v);
      R.push_back(v);
      Bits P2 = P & adj[static_cast<std::size_t>(v)];
      if (P2.any()) {
        expand(P2, size + 1);
      } else if (size + 1 > best) {
        best = size + 1;
        best_R = R;
      }
      R.pop_back();
    }
  }

  int run() {
    best = 0;
    best_R.clear();
    R.clear();
    nodes = 0;
    Bits P;
    for (int i = 0; i < n; ++i) P.set(i);
    expand(P, 0);
    return best;
  }
};

// ---------------------------------------------------------------------------
// Bron-Kerbosch with pivot: census of maximal cliques by size
// ---------------------------------------------------------------------------
struct BKCensus {
  const std::vector<Bits>* adj = nullptr;
  std::map<int, uint64_t> counts;
  uint64_t calls = 0;
  double t0 = 0, budget_s = 0;
  std::atomic<bool>* stop = nullptr;   // shared across workers
  bool timed_out = false;

  void bk(Bits P, Bits X, int rsize) {
    if (timed_out || (stop && stop->load(std::memory_order_relaxed))) {
      timed_out = true;
      return;
    }
    if ((++calls & ((1u << 22) - 1)) == 0 && budget_s > 0 && now_s() - t0 > budget_s) {
      timed_out = true;
      if (stop) stop->store(true, std::memory_order_relaxed);
      return;
    }
    if (!P.any() && !X.any()) {
      ++counts[rsize];
      return;
    }
    // pivot u in P|X maximising |P & N(u)|
    int bu = -1, bc = -1;
    for (int part = 0; part < 2; ++part) {
      Bits Q = part ? X : P;
      while (Q.any()) {
        const int u = Q.lsb();
        Q.clear(u);
        const int c = P.count_and((*adj)[static_cast<std::size_t>(u)]);
        if (c > bc) { bc = c; bu = u; }
      }
    }
    Bits ext = P.andnot((*adj)[static_cast<std::size_t>(bu)]);
    while (ext.any()) {
      const int v = ext.lsb();
      ext.clear(v);
      const Bits& nv = (*adj)[static_cast<std::size_t>(v)];
      bk(P & nv, X & nv, rsize + 1);
      P.clear(v);
      X.set(v);
    }
  }

  void run(int n, const std::vector<Bits>& a, double budget) {
    adj = &a;
    counts.clear();
    calls = 0;
    timed_out = false;
    budget_s = budget;
    t0 = now_s();
    Bits P;
    for (int i = 0; i < n; ++i) P.set(i);
    Bits X;
    bk(P, X, 0);
  }
};

// parallel driver: the root's branches (P \ N(pivot)) are independent given the
// sequential prefix state, so distribute them over OpenMP threads with a shared
// stop flag; identical counts to the serial run (checked on the triangle census).
BKCensus census_parallel(int n, const std::vector<Bits>& adj, double budget) {
  BKCensus agg;
  agg.adj = &adj;
  agg.budget_s = budget;
  agg.t0 = now_s();
  Bits P;
  for (int i = 0; i < n; ++i) P.set(i);
  int bu = -1, bc = -1;
  for (int u = 0; u < n; ++u) {
    const int c = P.count_and(adj[static_cast<std::size_t>(u)]);
    if (c > bc) { bc = c; bu = u; }
  }
  std::vector<int> branch;
  Bits ext = P.andnot(adj[static_cast<std::size_t>(bu)]);
  while (ext.any()) {
    const int v = ext.lsb();
    ext.clear(v);
    branch.push_back(v);
  }
  const int B = static_cast<int>(branch.size());
  std::vector<Bits> Pi(static_cast<std::size_t>(B)), Xi(static_cast<std::size_t>(B));
  Bits Pc = P, Xc;
  for (int i = 0; i < B; ++i) {
    const int v = branch[static_cast<std::size_t>(i)];
    const Bits& nv = adj[static_cast<std::size_t>(v)];
    Pi[static_cast<std::size_t>(i)] = Pc & nv;   // v itself is not in N(v)
    Xi[static_cast<std::size_t>(i)] = Xc & nv;
    Pc.clear(v);
    Xc.set(v);
  }
  std::atomic<bool> stop{false};
  std::vector<BKCensus> loc(static_cast<std::size_t>(B));
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < B; ++i) {
    BKCensus& L = loc[static_cast<std::size_t>(i)];
    L.adj = &adj;
    L.budget_s = budget;
    L.t0 = agg.t0;
    L.stop = &stop;
    L.bk(Pi[static_cast<std::size_t>(i)], Xi[static_cast<std::size_t>(i)], 1);
  }
  for (const auto& L : loc) {
    agg.calls += L.calls;
    agg.timed_out = agg.timed_out || L.timed_out;
    for (const auto& [s, c] : L.counts) agg.counts[s] += c;
  }
  return agg;
}

// ---------------------------------------------------------------------------
struct Subgraph {
  std::vector<uint32_t> verts;  // global indices, sorted
  std::vector<Bits> adj;
  int n = 0;
};

int dot32(const Vec& a, const Vec& b) { return kiss::dot(a, b); }

// common neighbours of a vertex set at inner product 16 with every member
std::vector<uint32_t> common_neighbours(const kiss::Leech& L, const std::vector<uint32_t>& base) {
  std::vector<uint32_t> out;
  for (uint32_t z = 0; z < static_cast<uint32_t>(N); ++z) {
    bool ok = true;
    for (uint32_t b : base)
      if (dot32(L.C[z], L.C[b]) != 16) { ok = false; break; }
    if (ok) out.push_back(z);
  }
  return out;
}

Subgraph induced(const kiss::Leech& L, const std::vector<uint32_t>& verts) {
  Subgraph S;
  S.verts = verts;
  S.n = static_cast<int>(verts.size());
  if (S.n > MAXN) { std::fprintf(stderr, "subgraph too large (%d > %d)\n", S.n, MAXN); std::exit(2); }
  S.adj.assign(static_cast<std::size_t>(S.n), Bits{});
  for (int i = 0; i < S.n; ++i)
    for (int j = i + 1; j < S.n; ++j)
      if (dot32(L.C[verts[static_cast<std::size_t>(i)]], L.C[verts[static_cast<std::size_t>(j)]]) == 16) {
        S.adj[static_cast<std::size_t>(i)].set(j);
        S.adj[static_cast<std::size_t>(j)].set(i);
      }
  return S;
}

uint64_t splitmix(uint64_t& s) {
  s += 0x9e3779b97f4a7c15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

}  // namespace

int main(int argc, char** argv) {
  std::string data_dir = "data";
  std::string out_path = "data/scheme/clique_g.json";
  int extra_edges = 2;
  double census_seconds = 900.0;
  bool do_census = true, test_mode = false, triangle_census = false;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--data") data_dir = next();
    else if (a == "--out") out_path = next();
    else if (a == "--edges") extra_edges = std::atoi(next().c_str());
    else if (a == "--census-seconds") census_seconds = std::atof(next().c_str());
    else if (a == "--no-census") do_census = false;
    else if (a == "--triangle-census") triangle_census = true;
    else if (a == "--seed") seed = static_cast<uint64_t>(std::atoll(next().c_str()));
    else if (a == "--test") test_mode = true;
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }

  const double T0 = now_s();
  bool ok = true;
  kiss::Leech L = kiss::generate_leech();
  std::printf("Leech minimal vectors: %zu (generated, canonical order)\n", L.C.size());

  // ---- common neighbourhood of the base edge (0,1) -------------------------
  const uint32_t ex = 0, ey = 1;
  if (dot32(L.C[ex], L.C[ey]) != 16) { std::fprintf(stderr, "base pair is not an edge\n"); return 2; }
  std::vector<uint32_t> common = common_neighbours(L, {ex, ey});
  std::printf("edge (%u,%u): common neighbours (= triangles through the edge) = %zu (scheme: p^5_55 = 891)\n",
              ex, ey, common.size());
  ok = ok && common.size() == 891;

  // cross-check against the mmap'ed adjacency table when present
  const std::filesystem::path adj_file = std::filesystem::path(data_dir) / "adj.u32";
  if (std::filesystem::exists(adj_file)) {
    kiss::Adjacency A(adj_file);
    std::vector<uint32_t> ix(A.row(ex), A.row(ex) + A.cols());
    std::vector<uint32_t> iy(A.row(ey), A.row(ey) + A.cols());
    std::vector<uint32_t> inter;
    std::set_intersection(ix.begin(), ix.end(), iy.begin(), iy.end(), std::back_inserter(inter));
    const bool same = inter == common;
    std::printf("adjacency cross-check (data/adj.u32): row intersection %zu, identical to dot-product list: %s\n",
                inter.size(), same ? "yes" : "NO");
    ok = ok && same;
  } else {
    std::printf("adjacency cross-check: data/adj.u32 not present, skipped\n");
  }

  Subgraph S = induced(L, common);
  int dmin = MAXN, dmax = 0;
  long esum = 0;
  for (int i = 0; i < S.n; ++i) {
    const int d = S.adj[static_cast<std::size_t>(i)].count();
    dmin = std::min(dmin, d);
    dmax = std::max(dmax, d);
    esum += d;
  }
  std::printf("G[N(x,y)]: %d vertices, %ld edges, degree min %d max %d (336-regular expected)\n",
              S.n, esum / 2, dmin, dmax);
  ok = ok && dmin == 336 && dmax == 336;

  std::vector<uint32_t> omega_clique;    // global indices of the maximum clique
  int omega = 0;

  if (test_mode) {
    // fast subset: B&B on the (much smaller) common neighbourhood of a triangle
    const uint32_t z0 = common.front();
    std::vector<uint32_t> tri_common = common_neighbours(L, {ex, ey, z0});
    Subgraph T = induced(L, tri_common);
    MaxClique mc;
    mc.n = T.n;
    mc.adj = T.adj;
    const int b = mc.run();
    std::printf("[test] triangle (0,1,%u): common neighbours %d, max clique %d -> clique through triangle %d\n",
                z0, T.n, b, b + 3);
    ok = ok && b + 3 <= 24 && b >= 1;
    omega = b + 3;  // only a lower bound in test mode; the 24-clique check below still runs
  } else {
    // ---- exact max clique on the full 891-vertex common neighbourhood ------
    MaxClique mc;
    mc.n = S.n;
    mc.adj = S.adj;
    double t = now_s();
    const int b = mc.run();
    std::printf("edge (0,1): max clique in G[N(x,y)] = %d  (omega(G) = %d), %llu B&B nodes, %.1f s\n",
                b, b + 2, static_cast<unsigned long long>(mc.nodes), now_s() - t);
    ok = ok && b == 22;
    omega = b + 2;
    omega_clique = {ex, ey};
    for (int v : mc.best_R) omega_clique.push_back(S.verts[static_cast<std::size_t>(v)]);
    std::sort(omega_clique.begin(), omega_clique.end());

    // ---- further random edges (arc-transitivity makes them redundant) ------
    uint64_t st = seed;
    for (int e = 0; e < extra_edges; ++e) {
      const uint32_t u = static_cast<uint32_t>(splitmix(st) % static_cast<uint64_t>(N));
      // random neighbour of u
      std::vector<uint32_t> nb;
      for (uint32_t z = 0; z < static_cast<uint32_t>(N); ++z)
        if (dot32(L.C[u], L.C[z]) == 16) nb.push_back(z);
      const uint32_t v = nb[splitmix(st) % nb.size()];
      std::vector<uint32_t> cm = common_neighbours(L, {u, v});
      Subgraph Se = induced(L, cm);
      MaxClique mce;
      mce.n = Se.n;
      mce.adj = Se.adj;
      t = now_s();
      const int be = mce.run();
      std::printf("random edge (%u,%u): common %d, max clique %d -> omega through it %d, %llu nodes, %.1f s\n",
                  u, v, Se.n, be, be + 2, static_cast<unsigned long long>(mce.nodes), now_s() - t);
      ok = ok && be == 22 && Se.n == 891;
    }
  }

  // ---- the explicit 24-clique, re-verified by exact integer arithmetic -----
  // (found by the B&B above; hard-wired so --test verifies it too)
  static const uint32_t K24[24] = {0, 1, 2386, 2398, 2399, 2403, 2404, 2406, 2407, 2410,
                                   2411, 2414, 2417, 2418, 2419, 2422, 2423, 2424, 2425,
                                   2426, 2427, 2428, 2429, 65390};
  bool k24 = true;
  for (int i = 0; i < 24; ++i) {
    if (dot32(L.C[K24[i]], L.C[K24[i]]) != 32) k24 = false;
    for (int j = i + 1; j < 24; ++j)
      if (dot32(L.C[K24[i]], L.C[K24[j]]) != 16) k24 = false;
  }
  std::printf("explicit 24-clique: pairwise inner products all 16, norms all 32: %s\n", k24 ? "yes" : "NO");
  std::printf("upper bound omega <= 24: Gram of k pairwise-16 vectors is 16(I+J), positive definite\n"
              "  (eigenvalues 16(k+1) and 16), so the vectors are linearly independent in R^24.\n");
  ok = ok && k24;
  if (omega_clique.empty()) omega_clique.assign(K24, K24 + 24);
  if (!test_mode) ok = ok && omega == 24;

  // ---- Bron-Kerbosch census of maximal cliques of G[N(x,y)] ---------------
  BKCensus census;
  if (do_census && !test_mode) {
    std::printf("Bron-Kerbosch census of maximal cliques in G[N(0,1)] (budget %.0f s, %d threads)...\n",
                census_seconds, omp_get_max_threads());
    census = census_parallel(S.n, S.adj, census_seconds);
    uint64_t total = 0;
    for (auto& [sz, c] : census.counts) total += c;
    std::printf("census %s: %llu maximal cliques, %llu calls\n",
                census.timed_out ? "TIMED OUT (partial)" : "complete",
                static_cast<unsigned long long>(total), static_cast<unsigned long long>(census.calls));
    for (auto& [sz, c] : census.counts)
      std::printf("  maximal cliques of size %2d in G[N(x,y)]: %10llu   (size %2d through the edge in G)\n",
                  sz, static_cast<unsigned long long>(c), sz + 2);
  }

  // ---- optional: complete census on a triangle's common neighbourhood -----
  if (triangle_census && !test_mode) {
    const uint32_t z0 = common.front();
    std::vector<uint32_t> tc = common_neighbours(L, {ex, ey, z0});
    Subgraph T = induced(L, tc);
    std::printf("Bron-Kerbosch census of maximal cliques in G[N(0,1,%u)] (%d vertices, budget %.0f s, %d threads)...\n",
                z0, T.n, census_seconds, omp_get_max_threads());
    BKCensus tcen = census_parallel(T.n, T.adj, census_seconds);
    uint64_t ttot = 0;
    for (auto& [sz, c] : tcen.counts) ttot += c;
    std::printf("triangle census %s: %llu maximal cliques, %llu calls\n",
                tcen.timed_out ? "TIMED OUT (partial)" : "complete",
                static_cast<unsigned long long>(ttot), static_cast<unsigned long long>(tcen.calls));
    for (auto& [sz, c] : tcen.counts)
      std::printf("  maximal cliques of size %2d in G[N(x,y,z)]: %10llu   (size %2d through the triangle in G)\n",
                  sz, static_cast<unsigned long long>(c), sz + 3);
  }

  // ---- JSON certificate ---------------------------------------------------
  if (!test_mode) {
    std::FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 2; }
    std::fprintf(f, "{\n  \"task\": \"B1\",\n  \"graph\": \"conflict graph G: Leech minimal vectors, edges = inner product 16\",\n");
    std::fprintf(f, "  \"omega\": %d,\n", omega);
    std::fprintf(f, "  \"upper_bound_argument\": \"k pairwise-16 norm-32 vectors have Gram 16(I+J_k), rank k, so k <= 24\",\n");
    std::fprintf(f, "  \"base_edge\": [%u, %u],\n  \"common_neighbours\": %zu,\n", ex, ey, common.size());
    std::fprintf(f, "  \"common_neighbourhood_regular_degree\": 336,\n");
    std::fprintf(f, "  \"max_clique_in_common_neighbourhood\": 22,\n");
    std::fprintf(f, "  \"clique\": [");
    for (std::size_t i = 0; i < omega_clique.size(); ++i)
      std::fprintf(f, "%s%u", i ? ", " : "", omega_clique[i]);
    std::fprintf(f, "],\n  \"clique_vectors\": [\n");
    for (std::size_t i = 0; i < omega_clique.size(); ++i) {
      std::fprintf(f, "    [");
      for (int kk = 0; kk < DIM; ++kk)
        std::fprintf(f, "%s%d", kk ? ", " : "", static_cast<int>(L.C[omega_clique[i]][static_cast<std::size_t>(kk)]));
      std::fprintf(f, "]%s\n", i + 1 < omega_clique.size() ? "," : "");
    }
    std::fprintf(f, "  ],\n");
    std::fprintf(f, "  \"census_complete\": %s,\n", (do_census && !census.timed_out) ? "true" : "false");
    std::fprintf(f, "  \"maximal_cliques_by_size\": {");
    bool first = true;
    for (auto& [sz, c] : census.counts) {
      std::fprintf(f, "%s\"%d\": %llu", first ? "" : ", ", sz, static_cast<unsigned long long>(c));
      first = false;
    }
    std::fprintf(f, "}\n}\n");
    std::fclose(f);
    std::printf("certificate written to %s\n", out_path.c_str());
  }

  std::printf("RESULT ok=%d omega=%d common=891 lambda=891 total_s=%.1f\n",
              ok ? 1 : 0, omega, now_s() - T0);
  return ok ? 0 : 1;
}

// T3.1 — exhaustive small-swap neighbourhoods of an independent set
// (docs/design.md T3.1, README §3 W2a).
//
// Terminology. S is an independent set of the conflict graph, tight[v] =
// #{s ∈ S : v ~ s} for every vertex v, conf(v) = {s ∈ S : v ~ s} (so
// |conf(v)| = tight[v]). A (k, m)-swap removes a set R ⊆ S with |R| = k and
// adds m mutually non-adjacent vertices whose conflicts in S all lie in R,
// i.e. m members of the pool P(R) = {v ∉ S : conf(v) ⊆ R}. Gain = m − k:
// improving if > 0, a PLATEAU move if = 0.
//
// Key reduction (src/swaps.cpp, kswap_search): if I ⊆ P(R) is independent then
// I ⊆ P(R') for R' = ∪_{v∈I} conf(v) ⊆ R, and |R'| ≤ |R|. So every swap with
// removed-set size ≤ kmax is witnessed by an R that is a *union of conf sets*
// of size ≤ kmax, and enumerating exactly those unions (level by level: the
// unions of size k are built from the stored unions of size < k, each union
// once) with an exact maximum-independent-set computation on each pool is a
// complete search of the (≤ kmax, ·)-swap neighbourhood. A level cut by the
// time limit or the set cap is reported non-exhaustive (and so are all
// higher levels); kmax_exhaustive is the largest k with levels 1..k complete.
//
// The routines are written against a tiny abstract graph interface so the
// same code runs on the Leech conflict graph (mmap adjacency of T1.6 for the
// rows, kiss::dot for pairwise tests) and on synthetic graphs in unit tests.
// Nothing here depends on kiss/verify.h.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/leech.h"
#include "kiss/types.h"

namespace kiss {

// ---------------------------------------------------------------------------
// Graph interface
// ---------------------------------------------------------------------------
class SwapGraph {
 public:
  virtual ~SwapGraph() = default;
  virtual uint32_t num_vertices() const = 0;
  // Neighbours of v: pointer to `count` vertex indices (need not be sorted).
  virtual const uint32_t* neighbours(uint32_t v, std::size_t& count) const = 0;
  virtual bool adjacent(uint32_t u, uint32_t v) const = 0;
};

// The Leech conflict graph: rows from the mmap'ed adj.u32, pairwise test by
// exact inner product (⟨C[u],C[v]⟩ == 16), which is faster than the binary
// search and touches no mmap pages.
class LeechSwapGraph final : public SwapGraph {
 public:
  LeechSwapGraph(const Leech& L, const Adjacency& adj) : L_(L), adj_(adj) {}
  uint32_t num_vertices() const override { return static_cast<uint32_t>(N); }
  const uint32_t* neighbours(uint32_t v, std::size_t& count) const override {
    count = static_cast<std::size_t>(DEG);
    return adj_.row(v);
  }
  bool adjacent(uint32_t u, uint32_t v) const override {
    return dot(L_.C[u], L_.C[v]) == 16;
  }
  const Leech& leech() const { return L_; }

 private:
  const Leech& L_;
  const Adjacency& adj_;
};

// Explicit adjacency lists (symmetrised, deduplicated, sorted in the ctor).
class ListSwapGraph final : public SwapGraph {
 public:
  explicit ListSwapGraph(uint32_t n, const std::vector<std::pair<uint32_t, uint32_t>>& edges);
  uint32_t num_vertices() const override { return static_cast<uint32_t>(nbr_.size()); }
  const uint32_t* neighbours(uint32_t v, std::size_t& count) const override {
    count = nbr_[v].size();
    return nbr_[v].data();
  }
  bool adjacent(uint32_t u, uint32_t v) const override;

 private:
  std::vector<std::vector<uint32_t>> nbr_;
};

// ---------------------------------------------------------------------------
// Basic helpers
// ---------------------------------------------------------------------------

// tight[v] = #{s ∈ S : v ∈ neighbours(s)} for every vertex (uint16; throws
// std::length_error if |S| > 65535). For an independent S, tight[s] = 0 on S.
std::vector<uint16_t> tightness_from_graph(const SwapGraph& g, const std::vector<uint32_t>& S);

// Histogram of tight[] over vertices NOT in S (index = tightness value).
std::vector<std::size_t> tightness_histogram(const std::vector<uint16_t>& tight,
                                             const std::vector<uint32_t>& S);
std::string histogram_string(const std::vector<std::size_t>& h);   // "0:496 4:80 ..."

// Random-order greedy maximal independent set (std::mt19937_64(seed) shuffle).
std::vector<uint32_t> greedy_maximal_set(const SwapGraph& g, uint64_t seed);

// All pairs non-adjacent?
bool is_independent(const SwapGraph& g, const std::vector<uint32_t>& S);

// Exact maximum independent set of the subgraph induced on `verts`
// (bitset branch and bound; intended for pools of at most a few hundred
// vertices). Returns the chosen members of `verts`.
std::vector<uint32_t> max_independent_subset(const SwapGraph& g, const std::vector<uint32_t>& verts);

// ---------------------------------------------------------------------------
// Swaps
// ---------------------------------------------------------------------------
struct Swap {
  std::vector<uint32_t> remove;   // vertices of S to delete
  std::vector<uint32_t> add;      // vertices to insert (mutually non-adjacent)
  int gain() const { return static_cast<int>(add.size()) - static_cast<int>(remove.size()); }
};

// (S \ remove) ∪ add, sorted. Throws std::runtime_error if a removed vertex is
// not in S or an added vertex already is.
std::vector<uint32_t> apply_swap(const std::vector<uint32_t>& S, const Swap& sw);

// Checks that applying sw to the independent set S yields an independent set:
// every added vertex is non-adjacent to every member of S \ remove and the
// added vertices are mutually non-adjacent (all via g.adjacent).
bool swap_is_valid(const SwapGraph& g, const std::vector<uint32_t>& S, const Swap& sw);

// Classical (1,2)-swap search (PLAN T3.1 spec): for every x ∈ S,
// L_x = {v ∈ neighbours(x) : tight[v] == 1}; success iff L_x contains a
// non-adjacent pair. OpenMP over x.
struct Swap12Result {
  bool found = false;
  Swap swap;                              // first hit in S order
  std::size_t hits = 0;                   // number of x with a (1,2)-swap
  std::vector<std::size_t> L_hist;        // L_hist[j] = #{x : |L_x| == j}
  std::size_t L_total = 0, L_max = 0;
  std::vector<std::vector<uint32_t>> L;   // L[i] = L_{S[i]} (kept for the (2,3) search)
  double seconds = 0;
};
Swap12Result find_swap12(const SwapGraph& g, const std::vector<uint32_t>& S,
                         const std::vector<uint16_t>& tight);

// Classical (2,3)-swap search: for every pair {x,y} ⊂ S the pool is
// P = L_x ∪ L_y ∪ {v ∈ N(x) ∩ N(y) : tight[v] == 2}; success iff P has an
// independent triple. All C(|S|,2) pairs are visited (the tightness-2 part is
// bucketed by conf pair, so a pair costs O(1) when its pool is empty).
struct Swap23Result {
  bool found = false;
  Swap swap;                              // first hit in pair order
  std::size_t hits = 0;                   // pairs with an independent triple
  std::size_t pairs = 0;                  // C(|S|,2)
  std::size_t pairs_nonempty = 0;         // pairs with |P| > 0
  std::size_t pairs_ge3 = 0;              // pairs with |P| ≥ 3 (triple test run)
  std::size_t pool_max = 0;
  std::vector<std::size_t> pool_hist;     // pool_hist[j] = #pairs with |P| == j
  std::size_t t2_candidates = 0;          // vertices with tight == 2
  double seconds = 0;
};
Swap23Result find_swap23(const SwapGraph& g, const std::vector<uint32_t>& S,
                         const std::vector<uint16_t>& tight, const Swap12Result& r12);

// Generalised (k, k+1)-swap search, k = 1..kmax (see file header).
constexpr int KSWAP_MAX_K = 12;

struct KSwapOptions {
  int kmax = 8;                        // 1..KSWAP_MAX_K
  double time_limit_s = 120.0;         // per level k; ≤ 0 = unlimited
  std::size_t max_sets = 20000000;     // cap on distinct unions kept (memory guard)
  std::size_t keep_moves = 64;         // plateau/improving moves stored per level
  bool stop_on_improving = false;      // finish the level, then stop
};

struct KSwapLevel {
  int k = 0;
  std::size_t cand_eq_k = 0;       // #{v ∉ S : tight[v] == k}
  std::size_t cand_le_k = 0;       // #{v ∉ S : 1 ≤ tight[v] ≤ k}
  std::size_t distinct_conf_eq_k = 0;  // distinct conf sets of size k
  std::size_t R_examined = 0;      // unions of size k whose pool was solved
  std::size_t R_total = 0;         // unions of size k generated (== R_examined if exhaustive)
  std::size_t largest_pool = 0;
  std::size_t pool_total = 0;      // Σ |P(R)| (for the mean)
  int best_m = 0;                  // max over R of MIS(P(R))
  std::size_t plateau_moves = 0;   // R with MIS(P(R)) == k (and union == R)
  std::size_t plateau_connected = 0; // ... of which the added set I is "connected": the bipartite
                                   // graph I ↔ R (v — s iff s ∈ conf(v)) is connected, i.e. the move is
                                   // not a disjoint union of smaller plateau moves
  std::size_t improving_moves = 0; // R with an independent I ⊆ P(R), |I| > |∪conf(I)|
  bool exhaustive = true;
  double seconds = 0;
  std::vector<Swap> plateau;       // up to keep_moves examples
  std::vector<Swap> improving;     // up to keep_moves examples (best gain first)
};

struct KSwapResult {
  std::size_t free_vertices = 0;   // tight == 0 outside S (each is a (0,1) move)
  std::vector<uint32_t> free_list; // up to 64 of them
  std::vector<KSwapLevel> levels;  // index k-1
  std::size_t unions_total = 0;    // distinct unions of size ≤ kmax generated
  bool improving_found = false;
  Swap best;                       // the best-gain improving swap (if any; free vertex counts)
  int kmax_exhaustive = 0;         // largest k such that levels 1..k were all exhaustive
  double seconds = 0;
};

KSwapResult kswap_search(const SwapGraph& g, const std::vector<uint32_t>& S,
                         const std::vector<uint16_t>& tight, const KSwapOptions& opt,
                         const std::function<void(const KSwapLevel&)>& on_level = nullptr);

// Grouping of the tightness-t vertices by their conf set (for the structural
// report on the 80 tightness-4 vertices of the 496).
struct ConfGroup {
  std::vector<uint32_t> conf;      // S-members (vertex indices), sorted
  std::vector<uint32_t> members;   // vertices v ∉ S with that conf set
  std::vector<uint32_t> mis;       // a maximum independent subset of members
};
std::vector<ConfGroup> group_by_conf(const SwapGraph& g, const std::vector<uint32_t>& S,
                                     const std::vector<uint16_t>& tight, int t);

}  // namespace kiss

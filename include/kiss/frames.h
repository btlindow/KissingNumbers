// T3.3 — frame-structured search on the antipodal classes of the Leech
// minimal vectors (docs/design.md T3.3, README §3 W2c).
//
// Objects. The 196560 minimal vectors form NCLASS = 98280 antipodal classes
// {v, −v}; class relations are sign-invariant functions of dot(rep a, rep b):
//   orthogonal  ⇔ dot = 0     (46575 classes per class; the ORTHOGONALITY graph)
//   conflict    ⇔ |dot| = 16  (4600 classes per class; the CONFLICT graph —
//                              the class-level quotient of the T1.6 graph)
// A class set X is INDEPENDENT iff no two of its classes conflict; then the
// 2|X| vectors form an independent set of the kissing problem (README §2). A
// FRAME is a set of 24 mutually orthogonal classes (48 vectors — a 4-frame of
// Λ24 in the VOA/Z4-code literature; NOT a Conway cross, which is a frame of
// norm-8 vectors). A k-SUBFRAME is a k-clique of the orthogonality graph.
// Every orthogonal clique is independent; two cliques are COMPATIBLE iff no
// cross pair conflicts (they may share classes).
//
// Tools built here: dense bitsets, induced graphs on class pools, clique
// enumeration (maximal cliques by Bron–Kerbosch with pivoting, k-cliques,
// max clique by colouring branch-and-bound), frame enumeration/counting
// through a class in the full 46575-class orthogonal neighbourhood (ordered
// DFS on bitsets, parallel over the first level, plus an unbiased
// random-descent estimator of the same count), the cross structure of a vector set (its frames / maximal
// cliques), the "frame extension" search around an independent set (cliques
// through a low-conflict class maximising |Q \ X| − |conflicts of Q in X|),
// a clique-move local search and the greedy frame-union builder.
// Everything is CPU (OpenMP); vector relations are exact integer dots.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/bits.h"

namespace kiss {

constexpr int NCLASS = N / 2;          // 98280 antipodal classes
constexpr int FRAME_SIZE = DIM;        // 24 classes per frame
constexpr int ORTHO_DEG = 46575;       // classes orthogonal to a class
constexpr int CONFLICT_DEG = DEG;      // classes conflicting with a class (4600)

// ---------------------------------------------------------------------------
// Antipodal classes
// ---------------------------------------------------------------------------
struct Classes {
  std::vector<uint32_t> rep;      // NCLASS entries: rep[c] = smaller index of {v, neg v}
  std::vector<uint32_t> cls_of;   // N entries: class of a vertex
  int size() const { return static_cast<int>(rep.size()); }
};
// Class ids in increasing order of their representative (the vertex with
// v < neg[v]); so class 0 is the class of vertex 0.
Classes make_classes(const Leech& L);

inline int class_dot(const Leech& L, const Classes& K, uint32_t a, uint32_t b) {
  return dot(L.C[K.rep[a]], L.C[K.rep[b]]);
}
inline bool class_orthogonal(const Leech& L, const Classes& K, uint32_t a, uint32_t b) {
  return class_dot(L, K, a, b) == 0;
}
inline bool class_conflict(const Leech& L, const Classes& K, uint32_t a, uint32_t b) {
  const int d = class_dot(L, K, a, b);
  return d == 16 || d == -16;
}

// Distinct classes of a vector set (sorted). Throws if a row is not in C.
std::vector<uint32_t> set_classes(const Leech& L, const Classes& K, const std::vector<Vec>& S);
// Both signs of every class (2|X| vectors, class order, +rep then −rep).
std::vector<Vec> classes_to_vectors(const Leech& L, const Classes& K, const std::vector<uint32_t>& X);
// No two classes conflict (exact dots, O(|X|²)).
bool classes_independent(const Leech& L, const Classes& K, const std::vector<uint32_t>& X);
// All pairs orthogonal.
bool classes_orthogonal_clique(const Leech& L, const Classes& K, const std::vector<uint32_t>& Q);
// No cross pair (a ∈ A, b ∈ B) conflicts (shared classes allowed).
bool cliques_compatible(const Leech& L, const Classes& K, const std::vector<uint32_t>& A,
                        const std::vector<uint32_t>& B);
// out[c] = #{x ∈ X : class_conflict(c, x)} for every class c (= tightness of
// rep[c] against the antipodal vector set of X). OpenMP over c.
std::vector<uint16_t> class_conflicts(const Leech& L, const Classes& K, const std::vector<uint32_t>& X);
// The 4600 classes conflicting with c (sorted), by 98280 exact dots.
std::vector<uint32_t> conflicting_classes(const Leech& L, const Classes& K, uint32_t c);

// ---------------------------------------------------------------------------
// Dense bitset
// ---------------------------------------------------------------------------
class Bitset {
 public:
  Bitset() = default;
  explicit Bitset(int n) : n_(n), w_(static_cast<std::size_t>((n + 63) / 64), 0) {}
  int size() const { return n_; }
  int nwords() const { return static_cast<int>(w_.size()); }
  uint64_t* words() { return w_.data(); }
  const uint64_t* words() const { return w_.data(); }
  bool test(int i) const { return (w_[static_cast<std::size_t>(i >> 6)] >> (i & 63)) & 1u; }
  void set(int i) { w_[static_cast<std::size_t>(i >> 6)] |= uint64_t{1} << (i & 63); }
  void reset(int i) { w_[static_cast<std::size_t>(i >> 6)] &= ~(uint64_t{1} << (i & 63)); }
  void clear();
  void fill();                       // all n_ bits
  bool empty() const;
  int count() const;
  void and_with(const Bitset& o);
  void and_not(const Bitset& o);
  void or_with(const Bitset& o);
  // this = a ∩ b ∩ {i > v}; returns the popcount.
  int intersect_above(const Bitset& a, const Bitset& b, int v);
  int first() const { return next(0); }
  int next(int i) const;             // smallest set bit ≥ i, or −1
  std::vector<int> to_list() const;
  template <class F>
  void for_each(F f) const {
    for (std::size_t k = 0; k < w_.size(); ++k) {
      uint64_t x = w_[k];
      while (x) {
        const int b = kiss::ctz64(x);
        f(static_cast<int>(k * 64) + b);
        x &= x - 1;
      }
    }
  }

 private:
  int n_ = 0;
  std::vector<uint64_t> w_;
};

// All classes orthogonal to c as a bitset over 0..NCLASS−1 (98280 dots).
void orthogonal_classes(const Leech& L, const Classes& K, uint32_t c, Bitset& out);

// ---------------------------------------------------------------------------
// Induced graphs on class pools
// ---------------------------------------------------------------------------
struct PoolGraph {
  std::vector<uint32_t> pool;   // class ids (pool index → class)
  std::vector<Bitset> adj;      // adj[i] over pool indices; no loops
  int size() const { return static_cast<int>(pool.size()); }
  int degree(int i) const { return adj[static_cast<std::size_t>(i)].count(); }
  bool adjacent(int i, int j) const { return adj[static_cast<std::size_t>(i)].test(j); }
  std::map<int, long> degree_histogram() const;
};
// Orthogonality graph induced on `pool` (OpenMP, |pool|² dots).
PoolGraph induced_orthogonality(const Leech& L, const Classes& K, const std::vector<uint32_t>& pool);
// Same for an arbitrary relation on the dot value (rel(dot) == true ⇒ edge).
PoolGraph induced_graph(const Leech& L, const Classes& K, const std::vector<uint32_t>& pool,
                        const std::function<bool(int)>& rel);
// Synthetic graph from an edge list (unit tests).
PoolGraph pool_graph_from_edges(int n, const std::vector<std::pair<int, int>>& edges);

// Maximal cliques (Bron–Kerbosch, pivot = max |P ∩ N(u)|). cb returns false
// to stop. Returns the number reported; sizes histogram in by_size;
// complete = false if stopped by cb or the time limit (0 = none).
struct CliqueStats {
  long count = 0;
  std::map<int, long> by_size;
  long nodes = 0;
  bool complete = true;
  double seconds = 0;
};
CliqueStats maximal_cliques(const PoolGraph& G,
                            const std::function<bool(const std::vector<int>&)>& cb = {},
                            double time_limit_s = 0);
// All k-cliques, each once (ordered DFS with |P| ≥ remaining pruning); `through`
// ≥ 0 restricts to cliques containing that pool vertex.
CliqueStats k_cliques(const PoolGraph& G, int k, const std::function<bool(const std::vector<int>&)>& cb = {},
                      double time_limit_s = 0, int through = -1);
// Maximum clique by greedy-colouring branch and bound (Tomita-style). Exact
// unless the time limit stops it (complete = false, best so far returned).
struct MaxCliqueResult {
  std::vector<int> clique;
  long nodes = 0;
  bool complete = true;
};
MaxCliqueResult max_clique(const PoolGraph& G, double time_limit_s = 0, int stop_at_size = 0);

// ---------------------------------------------------------------------------
// Cross structure of a vector set (T1.3-convert §3, now in C++)
// ---------------------------------------------------------------------------
struct CrossReport {
  std::vector<uint32_t> classes;             // distinct classes of S
  bool antipodal = false;
  std::map<int, long> degree_hist;           // orthogonality degrees
  std::map<int, long> maximal_by_size;
  long maximal_total = 0;
  int clique_number = 0;
  std::vector<std::vector<uint32_t>> frames; // 24-cliques (class ids)
  long cliques8 = -1;                        // all 8-cliques (−1 if not counted)
  double seconds = 0;
};
CrossReport cross_structure(const Leech& L, const Classes& K, const std::vector<Vec>& S, bool count8 = true);

// ---------------------------------------------------------------------------
// Frames through a class in the full orthogonality graph
// ---------------------------------------------------------------------------
// The orthogonal neighbourhood of a class as an explicit graph (46575
// vertices, 46575² dots, ~270 MB of bitsets). `restrict_to` (bitset over
// classes) intersects the neighbourhood, e.g. with the (±4,±4) classes.
struct Neighbourhood {
  uint32_t centre = 0;
  std::vector<uint32_t> verts;   // class ids, ascending
  std::vector<Bitset> adj;       // over neighbourhood indices
  int size() const { return static_cast<int>(verts.size()); }
};
Neighbourhood build_neighbourhood(const Leech& L, const Classes& K, uint32_t c, const Bitset* restrict_to = nullptr);

struct FrameCount {
  long frames = 0;        // (k−1)-cliques found in the neighbourhood = k-cliques through the centre
  long nodes = 0;         // DFS nodes visited
  bool complete = true;   // false ⇒ frames is a lower bound (time limit)
  double seconds = 0;
  long first_level_done = 0, first_level_total = 0;
};
// Counts cliques of size `k` through the centre (k = FRAME_SIZE for frames):
// ordered DFS on bitsets, OpenMP over the first level. cb (optional, called
// with neighbourhood indices of the k−1 other members, from any thread —
// must be thread-safe) may return false to stop.
FrameCount count_cliques_through(const Neighbourhood& Nb, int k, double time_limit_s = 0,
                                 const std::function<bool(const std::vector<int>&)>& cb = {});
// Unbiased estimate of the same count by `probes` random unordered descents
// (v_1 uniform in the neighbourhood, v_2 uniform in N(v_1), ...; estimate =
// Π|P_i| / (k−1)!): leaves = k-cliques through the centre with its standard
// error; cliques_by_depth[d] = estimated d-cliques through the centre (d ≤ k−1;
// the nodes of the ordered DFS at depth d, an upper bound on them); nodes =
// their sum; mean_candidates_by_depth[d] = mean |P_d| along the descents.
struct TreeEstimate {
  double leaves = 0, leaves_se = 0, nodes = 0;
  long probes = 0;
  std::map<int, double> mean_candidates_by_depth;
  std::map<int, double> cliques_by_depth;
};
TreeEstimate estimate_cliques_through(const Neighbourhood& Nb, int k, long probes, uint64_t seed);

// Shape mask helpers: bitset of all classes whose representative has the
// given leech_shape (0 = octad, 1 = (∓3,±1^23), 2 = (±4,±4)).
Bitset classes_of_shape(const Leech& L, const Classes& K, int shape);

// Random frame containing `start` (randomised DFS with backtracking and
// restarts, restricted to `allowed` if given; at each level the best of 16
// sampled candidates — the one keeping the most candidates — is taken and a
// level is abandoned after 6 tries, see src/frames.cpp). Returns the 24 class
// ids, or an empty vector if none was found within node_limit DFS nodes. At
// the default limit: 0/200 failures, 0.024 s per call (unrestricted).
std::vector<uint32_t> random_frame(const Leech& L, const Classes& K, uint32_t start, uint64_t seed,
                                   const Bitset* allowed = nullptr, long node_limit = 100000);

// ---------------------------------------------------------------------------
// Extension search around an independent class set X (the 496 = 248 classes)
// ---------------------------------------------------------------------------
// For a seed class c ∉ X: pool = {c} ∪ {x ∈ X : x ⊥ c} ∪ {d ∉ X : d ⊥ c, conf(d) ≤ tmax}.
// Enumerate cliques Q ∋ c in the pool with |Q| ≥ kmin; gain(Q) = |Q \ X| −
// |∪_{q ∈ Q\X} conf(q)| (the (removed, added) counts of the class-level swap
// X → X \ Conf(Q) ∪ Q). Branch and bound with the marginal-conflict bound
// (see src/frames.cpp); prunes nodes that cannot reach gain ≥ min_gain.
struct ExtendOptions {
  int tmax = 8;          // conflict cap for pool members outside X
  int kmin = 2;          // minimum clique size reported
  int min_gain = -1000;  // report cliques with gain ≥ this (bound prunes below it)
  double time_limit_s = 60;
  long node_limit = 0;   // 0 = none
  bool frames_only = false;   // only 24-cliques (kmin = 24)
};
struct ExtendHit {
  uint32_t seed = 0;
  std::vector<uint32_t> clique;   // class ids incl. seed
  int added = 0;                  // |Q \ X|
  int removed = 0;                // |Conf(Q)|
  int gain() const { return added - removed; }
};
struct ExtendResult {
  uint32_t seed = 0;
  int pool_size = 0, pool_outside = 0;
  long nodes = 0;
  long cliques = 0;                 // cliques of size ≥ kmin examined (leaves counted)
  bool complete = true;
  int best_gain = -1000000;
  ExtendHit best;                    // the best-gain clique (largest |Q| among ties)
  std::map<int, long> gain_hist;     // gain → number of cliques (size ≥ kmin) with that gain
  std::map<int, int> best_gain_by_size;   // |Q| → best gain among cliques of that size
  long frames = 0;                   // 24-cliques through the seed in the pool
  int best_frame_gain = -1000000;
  double seconds = 0;
};
ExtendResult extend_from_class(const Leech& L, const Classes& K, const std::vector<uint32_t>& X,
                               const std::vector<uint16_t>& conf, uint32_t c, const ExtendOptions& opt);
// Apply a hit: X \ Conf(Q) ∪ Q (checked independent; throws if not).
std::vector<uint32_t> apply_extend_hit(const Leech& L, const Classes& K, const std::vector<uint32_t>& X,
                                       const ExtendHit& hit);

// ---------------------------------------------------------------------------
// Greedy frame-union builder and clique-move local search
// ---------------------------------------------------------------------------
struct GreedyOptions {
  bool frames_first = true;   // first clique is a random full frame
  int min_clique = 1;         // stop adding when the largest free clique is smaller (1 = fill to maximality)
  double max_clique_time_s = 5;
  int large_pool = 3000;      // above this, look for a frame by randomised DFS instead of exact max clique
};
struct GreedyResult {
  std::vector<uint32_t> classes;        // the independent class set built
  std::vector<int> clique_sizes;        // sizes of the cliques added, in order
  int free_after_frames = 0;            // free classes after the last full frame
  double seconds = 0;
};
GreedyResult greedy_frame_union(const Leech& L, const Classes& K, uint64_t seed, const GreedyOptions& opt);

struct LocalSearchOptions {
  double time_limit_s = 60;
  int tmax = 6;              // pool conflict cap around the seed class
  int max_drop = 2;          // accept moves with gain ≥ −max_drop
  int tabu_tenure = 50;      // removed classes cannot be re-added for this many moves
  long node_limit = 200000;  // per clique search
  double move_time_s = 2;
  int seed_tmax = 8;         // candidate seed classes: conflict count ≤ this
  uint64_t rng_seed = 1;
  int target = 249;          // stop when |X| ≥ target
};
struct LocalSearchResult {
  std::vector<uint32_t> best;   // best independent class set seen
  int start_size = 0;
  long moves = 0, improving = 0, plateau = 0, worsening = 0, rejected = 0;
  double seconds = 0;
  std::map<int, long> size_visits;   // |X| after each accepted move
};
LocalSearchResult clique_local_search(const Leech& L, const Classes& K, const std::vector<uint32_t>& X0,
                                      const LocalSearchOptions& opt,
                                      const std::function<void(const std::string&)>& log = {});

// Number of perfect matchings of 2m points, (2m−1)!! — the count of frames
// made of (±4,±4) classes on 2m coordinates.
long double double_factorial_odd(int twom);

}  // namespace kiss

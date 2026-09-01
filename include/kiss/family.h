// T4.2 — disjoint families S_1..S_k of independent sets (README §1.3, §3 W3;
// docs/design.md T4.2; data/families/SCHEMA.md).
//
// Three ingredients:
//  * bitmask / sorted-index utilities for overlap tests between sets of
//    vertex indices (196560-bit masks, 24.6 KB each);
//  * the pool + disjointness graph + clique pipeline:
//    P random images g·S (product replacement, kiss/group.h), a dense
//    bit-adjacency graph with an edge iff two images are disjoint, and
//    greedy / local-search / exact clique search on it;
//  * the "sweep": a monotone greedy pass over the implicit pool
//    {x_m·x_l·x_j·h_i(S)} (E³·I samples for E random elements and I random
//    images) — a sample rejected against the current union U stays rejected
//    when U grows, so one pass accepts a maximal chain of pairwise disjoint
//    images; the rejection test is an early-exit scan of the image's 248
//    antipodal-pair ids against a bitset (g⁻¹U) that is refreshed per
//    (m,l,j), i.e. ~10 lookups per sample.
//  * family directories (S_xx.txt + family.json as in SCHEMA.md): write,
//    read, verify (each set independent, pairwise disjoint).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "kiss/group.h"
#include "kiss/leech.h"
#include "kiss/types.h"

namespace kiss {

// ---------------------------------------------------------------------------
// masks and overlaps
// ---------------------------------------------------------------------------
constexpr int MASK_WORDS = (N + 63) / 64;   // 3072
using Mask = std::vector<uint64_t>;          // MASK_WORDS words, bit v = vertex v in the set

Mask make_mask(const std::vector<uint32_t>& idx);          // throws std::out_of_range if idx >= N
bool mask_disjoint(const Mask& a, const Mask& b);
int mask_overlap(const Mask& a, const Mask& b);            // popcount(a & b)
int overlap_sorted(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);  // both sorted ascending

// ---------------------------------------------------------------------------
// pool of images g·S and their disjointness graph
// ---------------------------------------------------------------------------
struct ImagePool {
  std::vector<std::vector<uint32_t>> sets;   // each sorted ascending
  std::size_t size() const { return sets.size(); }
};

// `count` images of S_idx under successive outputs of ProductReplacement on
// `gens` (seeded, after `burnin` steps). Deterministic for a fixed seed.
ImagePool random_image_pool(const std::vector<uint32_t>& S_idx, const std::vector<IndexPerm>& gens,
                            std::size_t count, uint64_t seed, int slots = 10, int burnin = 100);

// Dense symmetric graph on n vertices as row bitsets (no self loops).
struct BitGraph {
  int n = 0;
  int words = 0;
  std::vector<uint64_t> bits;   // n * words
  explicit BitGraph(int n_ = 0);
  const uint64_t* row(int i) const { return bits.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(words); }
  uint64_t* row(int i) { return bits.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(words); }
  bool adj(int i, int j) const { return (row(i)[j >> 6] >> (j & 63)) & 1u; }
  void set_edge(int i, int j);                // both directions
  void clear_edge(int i, int j);
  int degree(int i) const;
  long long edges() const;                    // unordered
  double density() const;                     // edges / C(n,2)
};

// Edge iff the two images are disjoint. `disjointness_graph` uses an inverted
// vertex→images index (work Σ_v c_v², c_v = #images containing v);
// `disjointness_graph_masks` is the O(P²·MASK_WORDS) reference via mask ANDs.
BitGraph disjointness_graph(const ImagePool& pool);
BitGraph disjointness_graph_masks(const ImagePool& pool);

// Clique search. All return vertex lists (empty if nothing found).
bool is_clique(const BitGraph& G, const std::vector<int>& c);
// Greedy growth (vertex with most neighbours inside the candidate set) from
// random start vertices; best clique over `restarts`, stops early at size k.
std::vector<int> greedy_clique(const BitGraph& G, int k, uint64_t seed, int restarts);
// Swap-based local search (add if possible, else (1,1)-swap with tabu, else
// perturb). Returns the best clique seen (size ≥ k stops early).
std::vector<int> local_search_clique(const BitGraph& G, int k, uint64_t seed, long long max_steps,
                                     double time_limit_s, const std::vector<int>* start = nullptr);
// Exact branch and bound for a k-clique (candidate-set bitsets, size bound).
// Empty result with *nodes < node_limit ⇒ no k-clique exists.
std::vector<int> exact_clique(const BitGraph& G, int k, long long node_limit, long long* nodes = nullptr);

double log10_binomial(double n, int k);
// log10 of the expected number of k-cliques in G(n, p): log10 C(n,k) + C(k,2)·log10 p.
double log10_expected_cliques(double n, int k, double p);

// ---------------------------------------------------------------------------
// sweep: greedy chain over the implicit pool {x_m x_l x_j h_i(S)}
// ---------------------------------------------------------------------------
struct SweepOptions {
  int elements = 512;        // E random elements (forward + inverse index permutations, 2·786 KB each)
  int images = 16384;        // I random images of S (248 pair ids each)
  int target = 42;           // stop when this many pairwise disjoint sets are found
  double time_limit_s = 3600;
  uint64_t seed = 1;
  int slots = 10;
  int burnin = 100;
  bool verbose = true;       // print a line per accepted set (stdout)
  // Called (under the sweep's lock, from the accepting thread) after every
  // acceptance with the sets found so far — e.g. to checkpoint to disk.
  std::function<void(const std::vector<std::vector<uint32_t>>&)> on_accept;
};
struct SweepResult {
  std::vector<std::vector<uint32_t>> sets;   // sorted index lists, pairwise disjoint, in acceptance order
  unsigned long long samples = 0;             // rejection tests performed
  double seconds = 0;
  bool exhausted = false;                     // the whole E³·I space was swept
  bool timed_out = false;
  std::vector<std::pair<unsigned long long, double>> log;   // (samples, seconds) at each acceptance
};
SweepResult sweep_family(const Leech& L, const std::vector<uint32_t>& S_idx, const std::vector<IndexPerm>& gens,
                         const SweepOptions& opt);
// (1 − pairs·k/98280)^pairs: probability that a uniformly random image of an
// antipodal set of `pairs` antipodal pairs avoids the union of k such sets.
double sweep_success_probability(int k, int pairs = 248);

// ---------------------------------------------------------------------------
// chain: the orbit S, gS, g²S, ..., g^{k-1}S of a single random element
// ---------------------------------------------------------------------------
// g^a S ∩ g^b S = g^a (S ∩ g^{b-a} S), so the k images S, gS, ..., g^{k-1}S are
// pairwise disjoint iff g^i S ∩ S = ∅ for i = 1..k-1 — k−1 conditions instead
// of C(k,2), and for an antipodal S the conditions i and m−i coincide when g
// has order m on antipodal pairs (so an element of order 42 needs 21). The
// "chain length" L(g) = min{i ≥ 1 : g^i S ∩ S ≠ ∅} is computed with early
// exit (~2 steps on average) for random elements from product replacement,
// one sampler per OpenMP thread, until L(g) ≥ target or the time limit.
struct ChainOptions {
  int target = 42;
  double time_limit_s = 3600;
  uint64_t seed = 1;
  int slots = 10;
  int burnin = 100;
  // elements > 0: pool mode — E random elements x_0..x_{E-1} from product
  // replacement, candidates g = x_m ∘ x_l ∘ x_j over E³ triples (m in random
  // order, one (m,l) composition per E candidates; the chain test costs two
  // lookups per vector and step), ~10⁵ candidates/s/thread.
  // elements = 0: streaming mode — every candidate is a fresh product-
  // replacement output (~10² elements/s/thread, dominated by the composition).
  int elements = 1024;
  unsigned long long max_elements = 0;   // 0 = unlimited (pool mode: E³)
  int stat_min_length = 12;              // record (order, length) statistics for chains ≥ this
  bool verbose = true;
};
struct ChainResult {
  std::vector<std::vector<uint32_t>> sets;   // target sets g^i S (sorted index lists), empty if not found
  IndexPerm g;                               // the element (empty if not found)
  int chain_length = 0;                      // L(g) of the winner, capped at max_chain
  int order = 0;                             // order of g on antipodal pairs (0 = not computed)
  unsigned long long elements = 0;           // candidates tested
  bool exhausted = false;                    // pool mode: all E³ candidates tested
  double seconds = 0;
  bool timed_out = false;
  std::vector<unsigned long long> length_hist;   // [i] = #elements with L(g) = i, i = 1..target-1; [target] = successes
  // for elements with L(g) ≥ stat_min_length: order on antipodal pairs → histogram of L (same indexing)
  std::vector<std::pair<int, std::vector<unsigned long long>>> by_order;
};
ChainResult chain_family(const Leech& L, const std::vector<uint32_t>& S_idx, const std::vector<IndexPerm>& gens,
                         const ChainOptions& opt);
// L(g) = min{i ≥ 1 : g^i S ∩ S ≠ ∅}, capped at `cap` (returns cap if no hit up to cap−1).
int chain_length(const Leech& L, const std::vector<uint32_t>& S_idx, const IndexPerm& g, int cap);
// Order of g acting on antipodal pairs {v, −v} (lcm of the cycle lengths of all pairs).
int pair_order(const Leech& L, const IndexPerm& g);
// The exact 24×24 rational matrix of an index permutation (denominator 8,
// from the images of the (±4,±4,0^22) vectors); index_permutation(L, aut)
// reproduces g (that is the proof it is an automorphism).
Aut aut_from_index_perm(const Leech& L, const IndexPerm& g);

// ---------------------------------------------------------------------------
// family directories (SCHEMA.md)
// ---------------------------------------------------------------------------
struct Family {
  int dim = 0, d = 0;
  std::vector<std::string> files;            // as written in family.json
  std::vector<std::vector<Vec>> sets;
  std::string json;                          // raw family.json text
};
struct FamilyCheck {
  bool ok = false;
  std::string message;                       // "ok" or first failure
  std::vector<std::size_t> sizes;
  std::size_t total = 0;                     // Σ sizes
};
// Every set independent (verify_independent) and pairwise disjoint.
FamilyCheck verify_family(const Leech& L, const std::vector<std::vector<Vec>>& sets);
// Parse dir/family.json ("sets": [{"file", "size"}]) and read the set files
// (paths relative to dir). Throws std::runtime_error on any inconsistency.
Family read_family_dir(const std::filesystem::path& dir);
// Write S_01.txt.. and family.json. If template_json is non-empty its "T" and
// "extra" blocks are copied (groups: the largest |T_i| first, truncated to the
// number of sets) and the count is recomputed; otherwise "T"/"extra" are null.
// `provenance` is a free-form string stored under provenance.note; `header`
// lines (may contain '\n') are written as comments at the top of every set file.
void write_family_dir(const std::filesystem::path& dir, const std::vector<std::vector<Vec>>& sets, int dim,
                      const std::filesystem::path& template_json, const std::string& provenance,
                      const std::string& header);

}  // namespace kiss

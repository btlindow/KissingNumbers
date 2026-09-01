// T3.1b — plateau-graph exploration of an independent set (extends T3.1).
//
// The plateau graph has as nodes the independent sets of a fixed size and as
// edges the (k,k)-plateau moves of kswap_search (kiss/swaps.h) with k ≤ kmax.
// plateau_walk() performs a best-first search over this graph from S0: every
// expanded node gets the exhaustive (k, k+1)-swap search; an improving swap is
// applied at once (the improved set becomes a new node with top priority,
// the hook lets the caller write / verify it); otherwise the node's connected
// plateau moves — and, when it has at most `combo_max_moves` of them, every
// non-empty subset of pairwise-disjoint moves applied simultaneously — give
// its neighbours. Nodes are deduplicated by a canonical hash of the sorted
// index list. Two isomorphism invariants are kept per node:
//   fp0 = tightness histogram + caller-supplied invariant (e.g. Gram histogram),
//         known on discovery, used for the priority (nodes whose fp0 has been
//         expanded least often go first → breadth over inequivalent classes);
//   fp  = fp0 + per-k plateau-move counts, known after expansion, used to
//         count the distinct classes actually visited.
// Everything is written against SwapGraph so the walk runs on synthetic graphs
// in the unit test as well as on the Leech conflict graph.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "kiss/swaps.h"

namespace kiss {

// SHA-256 (hex) of the sorted index list as little-endian uint32 bytes. The
// input need not be sorted.
std::string canonical_hash(const std::vector<uint32_t>& S);

// The bipartite graph add ↔ remove (v — s iff g.adjacent(v, s)) is connected,
// i.e. the move is not a disjoint union of smaller plateau moves. Moves with
// ≤ 1 added vertex are connected.
bool plateau_move_is_connected(const SwapGraph& g, const Swap& sw);

// remove sets disjoint and add sets disjoint.
bool swaps_disjoint(const Swap& a, const Swap& b);
// Union of the two (sorted output); the caller guarantees disjointness.
Swap merge_swaps(const Swap& a, const Swap& b);
// All non-empty subsets of pairwise-disjoint moves, each merged into one Swap
// (singles first, in input order, then larger subsets). If moves.size() >
// max_moves only the singles are returned (the subsets would explode and are
// reachable by paths of singles anyway). The result is NOT validated against
// the graph; use swap_is_valid.
std::vector<Swap> disjoint_move_combinations(const std::vector<Swap>& moves, int max_moves);

struct PlateauOptions {
  int kmax = 12;                        // plateau moves of size ≤ kmax (1..KSWAP_MAX_K)
  double node_time_s = 60.0;            // per-level time limit of kswap_search at each node
  std::size_t max_sets = 10000000;      // kswap_search union cap
  std::size_t max_nodes = 200;          // nodes to expand
  double wall_s = 7200.0;               // wall-clock budget (checked between nodes)
  int combo_max_moves = 4;              // build subsets of disjoint moves when a node has ≤ this many
  std::size_t keep_moves = 4096;        // plateau moves kept per level by kswap_search
};

struct PlateauNode {
  std::size_t id = 0;
  std::size_t parent = 0;               // == id for the root
  std::size_t depth = 0;
  std::string via;                      // how it was reached: "root", "(k,k)", "combo<m>", "improve(k,m)"
  std::vector<uint32_t> S;              // sorted
  std::string hash;                     // canonical_hash(S)
  std::string fp0;                      // discovery-time invariant
  std::string fp;                       // fp0 + plateau counts (after expansion)
  std::vector<std::size_t> hist;        // tightness histogram outside S
  int min_tight = -1;                   // minimum tightness outside S (-1 if S is the whole graph)
  std::size_t free_vertices = 0, t1 = 0, t2 = 0, t3 = 0;
  // after expansion:
  bool expanded = false;
  std::vector<std::size_t> plateau_per_k;     // index k-1, all plateau moves (per union R)
  std::vector<std::size_t> connected_per_k;   // index k-1
  std::vector<Swap> moves;              // the connected plateau moves used for neighbours
  bool improving = false;               // kswap_search found an improving swap (applied)
  std::size_t improved_node = 0;        // id of the improved node (if improving)
  int kmax_exhaustive = 0;
  std::size_t unions = 0;
  double seconds = 0;                   // kswap_search time
  std::size_t neighbours = 0, neighbours_new = 0;
  bool fp0_new = false, fp_new = false; // first expanded node with that invariant
};

struct PlateauWalkResult {
  std::vector<PlateauNode> nodes;       // all discovered nodes (expanded or frontier), id = index
  std::size_t expanded = 0;
  std::size_t fingerprints = 0;         // distinct fp over expanded nodes
  std::size_t fingerprints0 = 0;        // distinct fp0 over expanded nodes
  std::size_t fingerprints0_all = 0;    // distinct fp0 over all discovered nodes
  std::size_t best_size = 0;
  std::vector<uint32_t> best_set;
  bool improving_found = false;
  std::size_t improvements = 0;         // number of improving swaps applied
  std::size_t frontier_left = 0;
  bool wall_hit = false, node_cap_hit = false;
  double seconds = 0;
};

using PlateauInvariant = std::function<std::string(const std::vector<uint32_t>&)>;
struct PlateauHooks {
  // Called after each node is expanded (stats filled, neighbours generated).
  std::function<void(const PlateauNode&, const PlateauWalkResult&)> on_node;
  // Called as soon as an improving swap has been applied and validated, before
  // anything else happens (write the set to disk here).
  std::function<void(const PlateauNode& from, const Swap& sw, const std::vector<uint32_t>& S2)> on_improve;
  // Called when a node is discovered (root, neighbour or improved set).
  std::function<void(const PlateauNode&)> on_discover;
};

PlateauWalkResult plateau_walk(const SwapGraph& g, const std::vector<uint32_t>& S0, const PlateauOptions& opt,
                               const PlateauInvariant& extra = nullptr, const PlateauHooks& hooks = {});

}  // namespace kiss

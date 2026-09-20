// T3.1b — plateau-walk acceptance test.
//
//   test_plateau [data_dir]
//
// 1. Synthetic: the path P_7 from S = {1,3,5}. With kmax = 1 the start node
//    has two (1,1) plateau moves (0 for 1, 6 for 5); the walk must reach the
//    maximum independent set {0,2,4,6} (size 4) via an improving (1,2)-swap at
//    the node {0,3,6}; every visited node must be independent; hashes,
//    connectivity and move combinations are checked directly.
// 2. Leech (skipped with exit 77 if data/adj.u32 is absent): the 496 with
//    kmax = 4 is a single isolated node (no plateau move of size ≤ 11), the 488
//    with kmax = 2 has 24 (2,2) moves; three nodes are expanded, all of size
//    488 and independent.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/plateau.h"
#include "kiss/swaps.h"
#include "kiss/verify.h"

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string data_dir = argc > 1 ? argv[1] : "data";
  bool leech_ran = false;
  std::size_t syn_best = 0, syn_nodes = 0;

  // ---- 1. synthetic ----------------------------------------------------------
  {
    std::printf("synthetic P_7\n");
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t i = 0; i + 1 < 7; ++i) edges.emplace_back(i, i + 1);
    const kiss::ListSwapGraph g(7, edges);
    const std::vector<uint32_t> S = {1, 3, 5};

    check(kiss::canonical_hash({5, 1, 3}) == kiss::canonical_hash(S), "canonical_hash is order-independent");
    check(kiss::canonical_hash({1, 3}) != kiss::canonical_hash(S), "canonical_hash distinguishes sets");

    kiss::Swap a, b;
    a.remove = {1}; a.add = {0};
    b.remove = {5}; b.add = {6};
    check(kiss::swaps_disjoint(a, b), "moves disjoint");
    check(kiss::plateau_move_is_connected(g, a), "single move connected");
    const kiss::Swap ab = kiss::merge_swaps(a, b);
    check(!kiss::plateau_move_is_connected(g, ab), "merged move is composite");
    check(kiss::swap_is_valid(g, S, ab), "merged move valid");
    kiss::Swap c;
    c.remove = {3}; c.add = {2, 4};   // (1,2) — connected through 3
    check(kiss::plateau_move_is_connected(g, c), "(1,2) connected");
    const std::vector<kiss::Swap> combos = kiss::disjoint_move_combinations({a, b}, 4);
    check(combos.size() == 3, "2 disjoint moves -> 3 combinations (" + std::to_string(combos.size()) + ")");
    check(kiss::disjoint_move_combinations({a, b}, 1).size() == 2, "combo cap 1 -> singles only");
    kiss::Swap a2 = a;   // overlaps a
    a2.add = {0};
    a2.remove = {1};
    check(kiss::disjoint_move_combinations({a, a2}, 4).size() == 2, "overlapping moves are not combined");

    kiss::PlateauOptions opt;
    opt.kmax = 1;
    opt.node_time_s = 10;
    opt.max_nodes = 100;
    opt.wall_s = 60;
    std::size_t improvements = 0;
    kiss::PlateauHooks hooks;
    hooks.on_improve = [&](const kiss::PlateauNode& from, const kiss::Swap& sw, const std::vector<uint32_t>& S2) {
      ++improvements;
      std::printf("  improvement at node %zu (depth %zu): (%zu,%zu) -> size %zu\n", from.id, from.depth, sw.remove.size(),
                  sw.add.size(), S2.size());
    };
    hooks.on_node = [&](const kiss::PlateauNode& nd, const kiss::PlateauWalkResult&) {
      std::printf("  node %zu depth %zu via %s size %zu hist %s moves %zu nbrs %zu new %zu improving %d fp0 %s\n", nd.id, nd.depth,
                  nd.via.c_str(), nd.S.size(), kiss::histogram_string(nd.hist).c_str(), nd.moves.size(), nd.neighbours,
                  nd.neighbours_new, nd.improving ? 1 : 0, nd.fp0_new ? "new" : "seen");
    };
    const kiss::PlateauWalkResult W = kiss::plateau_walk(g, S, opt, nullptr, hooks);
    syn_best = W.best_size;
    syn_nodes = W.expanded;
    check(W.nodes[0].moves.size() == 2, "root has 2 plateau moves (" + std::to_string(W.nodes[0].moves.size()) + ")");
    check(W.nodes[0].neighbours == 3, "root has 3 neighbours incl. the combo (" + std::to_string(W.nodes[0].neighbours) + ")");
    check(W.improving_found && W.best_size == 4, "walk reaches size 4");
    check(improvements >= 1, "on_improve called");
    check(W.best_set == std::vector<uint32_t>({0, 2, 4, 6}), "best set is {0,2,4,6}");
    bool all_indep = true, sizes_ok = true;
    for (const auto& nd : W.nodes) {
      if (!kiss::is_independent(g, nd.S)) all_indep = false;
      if (nd.S.size() != 3 && nd.S.size() != 4) sizes_ok = false;
    }
    check(all_indep, "all discovered nodes independent (" + std::to_string(W.nodes.size()) + " nodes)");
    check(sizes_ok, "node sizes are 3 or 4");
    check(W.frontier_left == 0 && !W.wall_hit && !W.node_cap_hit, "walk exhausted the component");
    check(W.fingerprints >= 3, "several fingerprint classes (" + std::to_string(W.fingerprints) + ")");
    // the improved node {0,2,4,6} was expanded and has no moves (all outside vertices have tightness 2)
    bool improved_expanded = false;
    for (const auto& nd : W.nodes)
      if (nd.S.size() == 4 && nd.expanded && nd.moves.empty() && nd.min_tight == 2) improved_expanded = true;
    check(improved_expanded, "improved node expanded: no moves, min tightness 2");
  }

  // ---- 2. Leech --------------------------------------------------------------
  const std::filesystem::path adj_file = std::filesystem::path(data_dir) / "adj.u32";
  if (!std::filesystem::exists(adj_file)) {
    std::printf("leech: %s absent — skipping (exit 77)\n", adj_file.string().c_str());
    if (failures) { std::printf("RESULT ok=0 failures=%d\n", failures); return 1; }
    std::printf("RESULT ok=1 synthetic=1 leech=0 syn_best=%zu syn_nodes=%zu failures=0\n", syn_best, syn_nodes);
    return 77;
  }
  std::size_t n496 = 0, n488 = 0, fp488 = 0;
  try {
    std::printf("leech\n");
    const kiss::Leech L = kiss::load_leech(data_dir);
    const kiss::Adjacency adj(adj_file);
    const kiss::LeechSwapGraph g(L, adj);
    auto load = [&](const char* name) {
      const std::vector<kiss::Vec> V = kiss::read_set(std::filesystem::path(data_dir) / name);
      std::vector<uint32_t> S = kiss::set_indices(L, V);
      std::sort(S.begin(), S.end());
      return S;
    };
    // the 496, kmax 4: isolated node
    {
      const std::vector<uint32_t> S = load("S496.txt");
      kiss::PlateauOptions opt;
      opt.kmax = 4;
      opt.node_time_s = 60;
      opt.max_nodes = 5;
      opt.wall_s = 120;
      const kiss::PlateauWalkResult W = kiss::plateau_walk(g, S, opt);
      n496 = W.expanded;
      check(W.expanded == 1 && W.nodes.size() == 1 && W.nodes[0].moves.empty(), "496 at kmax 4: isolated node");
      check(W.nodes[0].min_tight == 4 && W.nodes[0].t1 == 0 && W.nodes[0].t2 == 0 && W.nodes[0].t3 == 0,
            "496: min tightness 4, no tightness 1-3 vertices");
      check(!W.improving_found && W.best_size == 496, "496: no improvement at kmax 4");
      check(W.nodes[0].kmax_exhaustive == 4, "496: exhaustive to k=4");
    }
    // the 488, kmax 2: 24 (2,2) moves
    {
      const std::vector<uint32_t> S = load("S488.txt");
      kiss::PlateauOptions opt;
      opt.kmax = 2;
      opt.node_time_s = 60;
      opt.max_nodes = 3;
      opt.wall_s = 300;
      const kiss::PlateauWalkResult W = kiss::plateau_walk(g, S, opt);
      n488 = W.expanded;
      fp488 = W.fingerprints;
      check(W.nodes[0].moves.size() == 24, "488 at kmax 2: 24 connected (2,2) moves (" + std::to_string(W.nodes[0].moves.size()) + ")");
      check(W.nodes[0].neighbours_new == 24, "488: 24 new neighbours");
      check(W.expanded == 3 && W.node_cap_hit, "488: 3 nodes expanded (cap)");
      bool ok = true;
      for (const auto& nd : W.nodes)
        if (nd.S.size() != 488 || (nd.expanded && !kiss::is_independent(g, nd.S))) ok = false;
      check(ok, "488: all nodes size 488, expanded ones independent (" + std::to_string(W.nodes.size()) + " discovered)");
      check(!W.improving_found && W.best_size == 488, "488: no improvement at kmax 2");
      std::printf("  488 fingerprints over 3 nodes: %zu\n", W.fingerprints);
    }
    leech_ran = true;
  } catch (const std::exception& e) {
    std::printf("leech: exception %s\n", e.what());
    ++failures;
  }

  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("RESULT ok=%d synthetic=1 leech=%d syn_best=%zu syn_nodes=%zu n496=%zu n488=%zu fp488=%zu failures=%d ms=%.0f\n",
              failures == 0 ? 1 : 0, leech_ran ? 1 : 0, syn_best, syn_nodes, n496, n488, fp488, failures, ms);
  return failures == 0 ? 0 : 1;
}

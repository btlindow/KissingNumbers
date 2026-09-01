// T3.1 acceptance test — swap neighbourhoods (docs/design.md T3.1).
//
//  1. Synthetic graphs with planted (1,2)- and (2,3)-swaps: the classical
//     routines and the generalised k-search find them (and nothing spurious),
//     and every reported swap validates.
//  2. On data/S496.txt (needs data/adj.u32): the tightness histogram computed
//     from the adjacency rows equals T1.5's, and (1,2)/(2,3) report no hit.
//  3. A random greedy maximal set: the (k,k+1) routine finds an improvement at
//     k = 1 (and the classical (1,2) agrees); the improvement is applied and
//     re-verified as an independent set by exact inner products.
//
// Usage: test_swaps [data_dir]   (exit 77 = skip if data/adj.u32 is absent)
#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/swaps.h"
#include "kiss/types.h"

namespace {

int failures = 0;
void expect(bool cond, const std::string& what) {
  if (!cond) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::string vs(const std::vector<uint32_t>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
  return s + "]";
}

// ---- 1. synthetic ---------------------------------------------------------
void synthetic() {
  using E = std::pair<uint32_t, uint32_t>;
  // Graph A: S = {0}; 1,2 adjacent to 0 and not to each other → (1,2)-swap.
  // Vertex 3 is adjacent to 0, 1 and 2 (tight 1 but blocks nothing), vertex 4
  // adjacent to 1 and 2 only (tight 0 w.r.t. S? no: make it adjacent to 0 too
  // so S is maximal), vertex 5 adjacent to 0 and 3.
  {
    const kiss::ListSwapGraph g(6, std::vector<E>{{0, 1}, {0, 2}, {0, 3}, {1, 3}, {2, 3}, {0, 4}, {1, 4}, {2, 4}, {0, 5}, {3, 5}});
    const std::vector<uint32_t> S = {0};
    expect(kiss::is_independent(g, S), "A: S independent");
    const auto tight = kiss::tightness_from_graph(g, S);
    expect(tight == std::vector<uint16_t>({0, 1, 1, 1, 1, 1}), "A: tightness");
    expect(kiss::histogram_string(kiss::tightness_histogram(tight, S)) == "1:5", "A: histogram string");
    const auto r12 = kiss::find_swap12(g, S, tight);
    expect(r12.found && r12.hits == 1, "A: (1,2) found");
    expect(r12.L_max == 5 && r12.L_total == 5, "A: |L_0| = 5");
    expect(r12.found && kiss::swap_is_valid(g, S, r12.swap), "A: (1,2) swap valid");
    if (r12.found) {
      const auto S2 = kiss::apply_swap(S, r12.swap);
      expect(S2.size() == 2 && kiss::is_independent(g, S2), "A: applied (1,2) gives independent 2-set");
    }
    kiss::KSwapOptions opt;
    opt.kmax = 2;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(K.improving_found && K.best.gain() >= 1, "A: kswap improving");
    expect(K.levels.size() == 2 && K.levels[0].improving_moves == 1 && K.levels[0].R_examined == 1,
           "A: kswap level 1 has exactly one R with an improvement");
    expect(K.levels[0].best_m == 3, "A: kswap best_m at k=1 is 3 (MIS {1,2,5} of the pool {1,2,3,4,5})");
    expect(K.kmax_exhaustive == 2, "A: kswap exhaustive");
    expect(K.free_vertices == 0, "A: no free vertices");
    expect(kiss::swap_is_valid(g, S, K.best), "A: kswap best swap valid");
    // MIS helper on the pool {1,2,3,4,5}: edges 1-3,2-3,1-4,2-4,3-5 → max independent {1,2,5}
    const auto mis = kiss::max_independent_subset(g, {1, 2, 3, 4, 5});
    expect(mis.size() == 3, "A: MIS of pool = 3, got " + std::to_string(mis.size()));
    expect(mis == std::vector<uint32_t>({1, 2, 5}), "A: MIS is {1,2,5}, got " + vs(mis));
  }
  // Graph B: S = {0,1}. p=2 ~ 0 only, q=3 ~ 1 only, r=4 ~ 0 and 1; p,q,r mutually
  // non-adjacent → (2,3)-swap on {0,1}; no (1,2)-swap (|L_0| = |L_1| = 1).
  // Extra: 5 ~ 0,1,2 (tight 2, conflicts with p), 6 ~ 0,1,3,4 (tight 2).
  {
    const kiss::ListSwapGraph g(7, std::vector<E>{{0, 2}, {1, 3}, {0, 4}, {1, 4}, {0, 5}, {1, 5}, {2, 5}, {0, 6}, {1, 6}, {3, 6}, {4, 6}});
    const std::vector<uint32_t> S = {0, 1};
    const auto tight = kiss::tightness_from_graph(g, S);
    expect(tight == std::vector<uint16_t>({0, 0, 1, 1, 2, 2, 2}), "B: tightness");
    const auto r12 = kiss::find_swap12(g, S, tight);
    expect(!r12.found && r12.hits == 0, "B: no (1,2)");
    expect(r12.L_hist.size() == 2 && r12.L_hist[1] == 2, "B: |L_x| = 1 for both x");
    const auto r23 = kiss::find_swap23(g, S, tight, r12);
    expect(r23.found && r23.hits == 1 && r23.pairs == 1 && r23.pool_max == 5 && r23.t2_candidates == 3,
           "B: (2,3) found with pool {2,3,4,5,6}");
    expect(r23.found && kiss::swap_is_valid(g, S, r23.swap), "B: (2,3) swap valid");
    if (r23.found) {
      expect(r23.swap.add == std::vector<uint32_t>({2, 3, 4}), "B: added triple is {2,3,4}, got " + vs(r23.swap.add));
      const auto S2 = kiss::apply_swap(S, r23.swap);
      expect(S2.size() == 3 && kiss::is_independent(g, S2), "B: applied (2,3) gives independent 3-set");
    }
    kiss::KSwapOptions opt;
    opt.kmax = 3;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(K.improving_found, "B: kswap improving");
    expect(K.levels[0].improving_moves == 0 && K.levels[0].R_examined == 2, "B: k=1: two R, no improvement");
    expect(K.levels[0].plateau_moves == 2, "B: k=1: two plateau (1,1) moves (0->2, 1->3)");
    expect(K.levels[1].improving_moves == 1 && K.levels[1].R_examined == 1 && K.levels[1].best_m == 3,
           "B: k=2: one R = {0,1}, MIS 3");
    expect(K.levels[2].R_examined == 0, "B: k=3: no unions of size 3 (|S| = 2)");
    expect(K.kmax_exhaustive == 3, "B: exhaustive");
    expect(K.best.gain() == 1 && K.best.remove == std::vector<uint32_t>({0, 1}) && kiss::swap_is_valid(g, S, K.best),
           "B: best kswap = remove {0,1}, valid");
  }
  // Graph C: a maximal set with NO improvement of any kind: K4 on {0,1,2,3}, S = {0}.
  {
    const kiss::ListSwapGraph g(4, std::vector<E>{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}});
    const std::vector<uint32_t> S = {0};
    const auto tight = kiss::tightness_from_graph(g, S);
    const auto r12 = kiss::find_swap12(g, S, tight);
    expect(!r12.found, "C: no (1,2) in K4");
    kiss::KSwapOptions opt;
    opt.kmax = 2;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(!K.improving_found && K.levels[0].plateau_moves == 1 && K.levels[0].best_m == 1, "C: only a plateau move");
  }
  // Graph D: a free vertex (tight 0) is reported as an improvement.
  {
    const kiss::ListSwapGraph g(3, std::vector<E>{{0, 1}});
    const std::vector<uint32_t> S = {0};
    const auto tight = kiss::tightness_from_graph(g, S);
    kiss::KSwapOptions opt;
    opt.kmax = 1;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(K.free_vertices == 1 && K.improving_found && K.best.add == std::vector<uint32_t>({2}), "D: free vertex");
  }
  // Graph E: connected plateau move. S = {0,1}; 2 ~ 0 only, 3 ~ 1 only, 2 ~ 3;
  // 4, 5 ~ {0,1}, 4 ≁ 5, and 4, 5 adjacent to 2 and 3. R = {0,1}: pool {2,3,4,5},
  // MIS = {4,5} (size 2 = k, plateau), both added vertices touch both removed
  // ones → connected. k = 1: R={0} pool {2}, R={1} pool {3} → two (1,1) plateaus.
  {
    const kiss::ListSwapGraph g(6, std::vector<E>{{0, 2}, {1, 3}, {2, 3}, {0, 4}, {1, 4}, {0, 5}, {1, 5}, {2, 4}, {2, 5}, {3, 4}, {3, 5}});
    const std::vector<uint32_t> S = {0, 1};
    const auto tight = kiss::tightness_from_graph(g, S);
    kiss::KSwapOptions opt;
    opt.kmax = 2;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(!K.improving_found, "E: no improvement");
    expect(K.levels[0].plateau_moves == 2 && K.levels[0].plateau_connected == 2, "E: k=1 two (1,1) plateaus, trivially connected");
    expect(K.levels[1].R_examined == 1 && K.levels[1].plateau_moves == 1 && K.levels[1].plateau_connected == 1,
           "E: k=2 one connected (2,2) plateau move");
    expect(K.levels[1].plateau.size() == 1 && K.levels[1].plateau[0].add == std::vector<uint32_t>({4, 5}), "E: plateau adds {4,5}");
  }
  // Graph F: composite plateau move. S = {0,1}; 2 ~ 0 only, 3 ~ 1 only, 2 ≁ 3.
  // R = {0,1} (generated as conf(2) ∪ conf(3)): pool {2,3}, MIS 2 = k → plateau,
  // but it is the disjoint union of the (1,1) moves 0→2 and 1→3 → not connected.
  {
    const kiss::ListSwapGraph g(4, std::vector<E>{{0, 2}, {1, 3}});
    const std::vector<uint32_t> S = {0, 1};
    const auto tight = kiss::tightness_from_graph(g, S);
    kiss::KSwapOptions opt;
    opt.kmax = 2;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(!K.improving_found, "F: no improvement");
    expect(K.levels[0].plateau_moves == 2 && K.levels[0].plateau_connected == 2, "F: k=1 two (1,1) plateaus");
    expect(K.levels[1].R_examined == 1 && K.levels[1].plateau_moves == 1 && K.levels[1].plateau_connected == 0,
           "F: k=2 one composite (2,2) plateau move, not connected");
  }
  std::printf("synthetic        : done, failures so far %d\n", failures);
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
  try {
    synthetic();

    const std::filesystem::path adj_file = data_dir / "adj.u32";
    if (!std::filesystem::exists(adj_file) || std::filesystem::file_size(adj_file) != kiss::Adjacency::EXPECTED_BYTES) {
      std::printf("adjacency        : %s absent — Leech-graph checks skipped\n", adj_file.c_str());
      if (failures == 0) {
        std::printf("RESULT ok=1 synthetic=1 leech=0 failures=0 (skipped: no adjacency file)\n");
        return 77;
      }
      std::printf("RESULT ok=0 failures=%d\n", failures);
      return 1;
    }
    kiss::Leech L;
    try {
      L = kiss::load_leech(data_dir);
    } catch (const std::exception& e) {
      std::printf("leech            : load failed (%s); generating\n", e.what());
      L = kiss::generate_leech();
    }
    const kiss::Adjacency adj(adj_file);
    const kiss::LeechSwapGraph g(L, adj);

    // ---- 2. the 496 --------------------------------------------------------
    std::string hist496;
    bool have496 = false;
    const std::filesystem::path f496 = data_dir / "S496.txt";
    if (std::filesystem::exists(f496)) {
      have496 = true;
      const std::vector<kiss::Vec> V = kiss::read_set(f496);
      std::vector<uint32_t> S;
      for (const auto& v : V) {
        const int32_t i = L.index_of(v);
        expect(i >= 0, "496: row not in C");
        S.push_back(static_cast<uint32_t>(i));
      }
      std::sort(S.begin(), S.end());
      expect(S.size() == 496, "496: size");
      expect(kiss::is_independent(g, S), "496: independent by dot==16");
      const auto tight = kiss::tightness_from_graph(g, S);
      hist496 = kiss::histogram_string(kiss::tightness_histogram(tight, S));
      const std::string expected =
          "4:80 6:640 7:256 8:2704 9:8064 10:31424 11:52672 12:49552 13:31360 14:13440 15:2560 16:1480 17:256 18:704 19:64 20:528 22:128 24:152";
      std::printf("496 histogram    : %s\n", hist496.c_str());
      expect(hist496 == expected, "496: tightness histogram equals T1.5's (outside S; T1.5's 0:496 are the S members)");
      std::size_t inS_nonzero = 0;
      for (uint32_t s : S) inS_nonzero += tight[s] != 0;
      expect(inS_nonzero == 0, "496: tight==0 on S");
      const auto r12 = kiss::find_swap12(g, S, tight);
      const auto r23 = kiss::find_swap23(g, S, tight, r12);
      expect(!r12.found && r12.L_total == 0, "496: no tightness-1 vertices, no (1,2)-swap");
      expect(!r23.found && r23.pairs == 496 * 495 / 2 && r23.pairs_nonempty == 0, "496: no (2,3)-swap, all pools empty");
      std::printf("496 classical    : (1,2) hits=%zu (2,3) hits=%zu pairs=%zu %.3f s\n", r12.hits, r23.hits, r23.pairs, r12.seconds + r23.seconds);
      kiss::KSwapOptions opt;
      opt.kmax = 4;
      opt.time_limit_s = 60;
      const auto K = kiss::kswap_search(g, S, tight, opt);
      expect(K.levels[0].R_examined == 0 && K.levels[1].R_examined == 0 && K.levels[2].R_examined == 0, "496: no unions of size < 4");
      expect(K.levels[3].cand_eq_k == 80, "496: 80 tightness-4 candidates");
      expect(K.kmax_exhaustive == 4, "496: k<=4 exhaustive");
      expect(K.levels[3].distinct_conf_eq_k == 80 && K.levels[3].largest_pool == 1 && K.levels[3].plateau_moves == 0,
             "496: the 80 tightness-4 vertices have 80 distinct conf sets, pools of size 1, no (4,4) plateau");
      expect(!K.improving_found, "496: no improving swap with k<=4");
      std::printf("496 kswap        : k=4 distinct_conf=%zu R=%zu largest_pool=%zu best_m=%d plateau=%zu %.2f s\n",
                  K.levels[3].distinct_conf_eq_k, K.levels[3].R_examined, K.levels[3].largest_pool, K.levels[3].best_m,
                  K.levels[3].plateau_moves, K.seconds);
    } else {
      std::printf("496              : %s absent, skipped\n", f496.c_str());
    }

    // ---- 3. greedy random maximal set ---------------------------------------
    const std::vector<uint32_t> S = kiss::greedy_maximal_set(g, 20260825);
    expect(kiss::is_independent(g, S), "greedy: independent");
    const auto tight = kiss::tightness_from_graph(g, S);
    const auto hist = kiss::tightness_histogram(tight, S);
    expect(!hist.empty() && hist[0] == 0, "greedy: maximal (no free vertex)");
    std::printf("greedy           : |S|=%zu histogram %s\n", S.size(), kiss::histogram_string(hist).c_str());
    const auto r12 = kiss::find_swap12(g, S, tight);
    expect(r12.found, "greedy: classical (1,2) finds a swap");
    kiss::KSwapOptions opt;
    opt.kmax = 2;
    opt.time_limit_s = 60;
    const auto K = kiss::kswap_search(g, S, tight, opt);
    expect(K.improving_found, "greedy: kswap finds an improvement");
    expect(K.levels[0].improving_moves > 0, "greedy: improvement at k = 1");
    expect(K.levels[0].improving_moves == r12.hits, "greedy: k=1 improving R count == classical (1,2) hits");
    expect(K.levels[0].R_examined == r12.L_hist.size() - r12.L_hist[0] || K.levels[0].R_examined <= S.size(),
           "greedy: k=1 R count == #x with |L_x| > 0");
    expect(K.levels[0].R_examined + r12.L_hist[0] == S.size(), "greedy: every x with L_x nonempty is a union of size 1");
    expect(K.best.gain() >= 1 && kiss::swap_is_valid(g, S, K.best), "greedy: best swap valid");
    const auto S2 = kiss::apply_swap(S, K.best);
    expect(S2.size() == S.size() + static_cast<std::size_t>(K.best.gain()), "greedy: new size");
    expect(kiss::is_independent(g, S2), "greedy: improved set independent");
    // exact inner-product recheck of the improved set (independent of the graph object)
    std::size_t bad = 0;
    for (std::size_t i = 0; i < S2.size(); ++i)
      for (std::size_t j = i + 1; j < S2.size(); ++j) bad += kiss::dot(L.C[S2[i]], L.C[S2[j]]) > 8;
    expect(bad == 0, "greedy: improved set Gram <= 8 by dot");
    std::printf("greedy swaps     : (1,2) hits=%zu  kswap k=1 improving=%zu plateau=%zu; k=2 R=%zu improving=%zu plateau=%zu exhaustive=%d; best gain %d -> |S|=%zu  %.2f s\n",
                r12.hits, K.levels[0].improving_moves, K.levels[0].plateau_moves, K.levels[1].R_examined,
                K.levels[1].improving_moves, K.levels[1].plateau_moves, K.levels[1].exhaustive ? 1 : 0, K.best.gain(),
                S2.size(), K.seconds);

    std::printf("RESULT ok=%d synthetic=1 leech=1 s496=%d hist496_match=%d greedy_size=%zu greedy_gain=%d failures=%d\n",
                failures == 0 ? 1 : 0, have496 ? 1 : 0, hist496.empty() ? 0 : 1, S.size(), K.best.gain(), failures);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("exception: %s\nRESULT ok=0 failures=%d\n", e.what(), failures + 1);
    return 1;
  }
}

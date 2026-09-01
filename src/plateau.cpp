// T3.1b — plateau-graph walk (see include/kiss/plateau.h).
#include "kiss/plateau.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>

#include "kiss/io.h"

namespace kiss {

namespace {

using clock_t_ = std::chrono::steady_clock;
double secs_since(clock_t_::time_point t0) {
  return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

std::string counts_string(const std::vector<std::size_t>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (!v[i]) continue;
    if (!s.empty()) s += ',';
    s += std::to_string(i + 1) + ":" + std::to_string(v[i]);
  }
  return s.empty() ? "-" : s;
}

}  // namespace

std::string canonical_hash(const std::vector<uint32_t>& S) {
  std::vector<uint32_t> s = S;
  std::sort(s.begin(), s.end());
  std::vector<uint8_t> bytes(s.size() * 4);
  for (std::size_t i = 0; i < s.size(); ++i)
    for (int b = 0; b < 4; ++b) bytes[i * 4 + static_cast<std::size_t>(b)] = static_cast<uint8_t>(s[i] >> (8 * b));
  return sha256_hex(bytes.data(), bytes.size());
}

bool plateau_move_is_connected(const SwapGraph& g, const Swap& sw) {
  const std::size_t m = sw.add.size();
  if (m <= 1) return true;
  // union-find over the added vertices, joined through shared removed members
  std::vector<std::size_t> parent(m);
  for (std::size_t i = 0; i < m; ++i) parent[i] = i;
  auto find = [&](std::size_t a) {
    while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  std::size_t comps = m;
  for (uint32_t s : sw.remove) {
    std::size_t first = m;
    for (std::size_t i = 0; i < m; ++i) {
      if (!g.adjacent(sw.add[i], s)) continue;
      if (first == m) { first = i; continue; }
      const std::size_t a = find(first), b = find(i);
      if (a != b) { parent[a] = b; --comps; }
    }
  }
  return comps == 1;
}

bool swaps_disjoint(const Swap& a, const Swap& b) {
  for (uint32_t x : a.remove)
    if (std::find(b.remove.begin(), b.remove.end(), x) != b.remove.end()) return false;
  for (uint32_t x : a.add)
    if (std::find(b.add.begin(), b.add.end(), x) != b.add.end()) return false;
  return true;
}

Swap merge_swaps(const Swap& a, const Swap& b) {
  Swap m = a;
  m.remove.insert(m.remove.end(), b.remove.begin(), b.remove.end());
  m.add.insert(m.add.end(), b.add.begin(), b.add.end());
  std::sort(m.remove.begin(), m.remove.end());
  std::sort(m.add.begin(), m.add.end());
  return m;
}

std::vector<Swap> disjoint_move_combinations(const std::vector<Swap>& moves, int max_moves) {
  std::vector<Swap> out(moves);
  const std::size_t m = moves.size();
  if (max_moves < 2 || m > static_cast<std::size_t>(max_moves) || m < 2) return out;
  // subsets by increasing size (popcount), pairwise disjoint only
  std::vector<uint8_t> disj(m * m, 0);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = i + 1; j < m; ++j) disj[i * m + j] = disj[j * m + i] = swaps_disjoint(moves[i], moves[j]) ? 1 : 0;
  for (std::size_t size = 2; size <= m; ++size)
    for (uint32_t mask = 1; mask < (1u << m); ++mask) {
      if (static_cast<std::size_t>(__builtin_popcount(mask)) != size) continue;
      bool ok = true;
      for (std::size_t i = 0; i < m && ok; ++i) {
        if (!(mask >> i & 1u)) continue;
        for (std::size_t j = i + 1; j < m; ++j)
          if ((mask >> j & 1u) && !disj[i * m + j]) { ok = false; break; }
      }
      if (!ok) continue;
      Swap acc;
      for (std::size_t i = 0; i < m; ++i)
        if (mask >> i & 1u) acc = merge_swaps(acc, moves[i]);
      out.push_back(std::move(acc));
    }
  return out;
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------
PlateauWalkResult plateau_walk(const SwapGraph& g, const std::vector<uint32_t>& S0, const PlateauOptions& opt,
                               const PlateauInvariant& extra, const PlateauHooks& hooks) {
  const auto t0 = clock_t_::now();
  if (opt.kmax < 1 || opt.kmax > KSWAP_MAX_K) throw std::invalid_argument("plateau_walk: kmax out of range");
  PlateauWalkResult R;
  std::unordered_map<std::string, std::size_t> seen;            // hash -> id
  std::unordered_map<std::string, std::size_t> fp0_expanded;    // fp0 -> #expanded nodes with it
  std::unordered_map<std::string, std::size_t> fp0_all;         // fp0 -> #discovered nodes with it
  std::unordered_map<std::string, std::size_t> fp_first;        // fp -> first expanded id
  std::vector<std::size_t> frontier;

  // Discover a node (returns its id; existing id if already known).
  auto discover = [&](std::vector<uint32_t> S, std::size_t parent, std::size_t depth, const std::string& via,
                      bool& is_new) -> std::size_t {
    std::sort(S.begin(), S.end());
    const std::string h = canonical_hash(S);
    auto it = seen.find(h);
    if (it != seen.end()) { is_new = false; return it->second; }
    is_new = true;
    PlateauNode nd;
    nd.id = R.nodes.size();
    nd.parent = parent;
    nd.depth = depth;
    nd.via = via;
    nd.hash = h;
    const std::vector<uint16_t> tight = tightness_from_graph(g, S);
    nd.hist = tightness_histogram(tight, S);
    nd.free_vertices = nd.hist.empty() ? 0 : nd.hist[0];
    nd.t1 = nd.hist.size() > 1 ? nd.hist[1] : 0;
    nd.t2 = nd.hist.size() > 2 ? nd.hist[2] : 0;
    nd.t3 = nd.hist.size() > 3 ? nd.hist[3] : 0;
    for (std::size_t t = 0; t < nd.hist.size(); ++t)
      if (nd.hist[t]) { nd.min_tight = static_cast<int>(t); break; }
    nd.fp0 = "tight=" + histogram_string(nd.hist);
    if (extra) nd.fp0 += "|" + extra(S);
    nd.S = std::move(S);
    ++fp0_all[nd.fp0];
    seen.emplace(h, nd.id);
    R.nodes.push_back(std::move(nd));
    frontier.push_back(R.nodes.back().id);
    if (hooks.on_discover) hooks.on_discover(R.nodes.back());
    return R.nodes.back().id;
  };

  bool is_new = false;
  discover(S0, 0, 0, "root", is_new);
  R.best_size = R.nodes[0].S.size();
  R.best_set = R.nodes[0].S;

  while (!frontier.empty()) {
    if (R.expanded >= opt.max_nodes) { R.node_cap_hit = true; break; }
    if (opt.wall_s > 0 && secs_since(t0) > opt.wall_s) { R.wall_hit = true; break; }
    // ---- pick: largest size, then least-expanded fp0, then shallowest, then oldest
    std::size_t pick = 0;
    for (std::size_t i = 1; i < frontier.size(); ++i) {
      const PlateauNode& a = R.nodes[frontier[i]];
      const PlateauNode& b = R.nodes[frontier[pick]];
      if (a.S.size() != b.S.size()) { if (a.S.size() > b.S.size()) pick = i; continue; }
      const std::size_t ca = fp0_expanded[a.fp0], cb = fp0_expanded[b.fp0];
      if (ca != cb) { if (ca < cb) pick = i; continue; }
      if (a.depth != b.depth) { if (a.depth < b.depth) pick = i; continue; }
      if (a.id < b.id) pick = i;
    }
    const std::size_t id = frontier[pick];
    frontier[pick] = frontier.back();
    frontier.pop_back();

    // ---- expand
    KSwapOptions ko;
    ko.kmax = opt.kmax;
    ko.time_limit_s = opt.node_time_s;
    ko.max_sets = opt.max_sets;
    ko.keep_moves = opt.keep_moves;
    std::vector<uint32_t> S = R.nodes[id].S;   // copy: R.nodes may reallocate below
    const std::vector<uint16_t> tight = tightness_from_graph(g, S);
    const KSwapResult K = kswap_search(g, S, tight, ko);
    {
      PlateauNode& nd = R.nodes[id];
      nd.expanded = true;
      nd.kmax_exhaustive = K.kmax_exhaustive;
      nd.unions = K.unions_total;
      nd.seconds = K.seconds;
      nd.plateau_per_k.assign(static_cast<std::size_t>(opt.kmax), 0);
      nd.connected_per_k.assign(static_cast<std::size_t>(opt.kmax), 0);
      for (const KSwapLevel& L : K.levels) {
        nd.plateau_per_k[static_cast<std::size_t>(L.k - 1)] = L.plateau_moves;
        nd.connected_per_k[static_cast<std::size_t>(L.k - 1)] = L.plateau_connected;
        for (const Swap& sw : L.plateau)
          if (sw.gain() == 0 && plateau_move_is_connected(g, sw) && swap_is_valid(g, S, sw)) nd.moves.push_back(sw);
      }
      nd.fp = nd.fp0 + "|plateau=" + counts_string(nd.plateau_per_k) + "|connected=" + counts_string(nd.connected_per_k) +
              "|kexh=" + std::to_string(K.kmax_exhaustive);
      nd.fp0_new = fp0_expanded[nd.fp0]++ == 0;
      nd.fp_new = fp_first.emplace(nd.fp, id).second;
    }
    const std::size_t depth = R.nodes[id].depth;

    // ---- improvement
    if (K.improving_found) {
      const Swap& sw = K.best;
      if (sw.gain() > 0 && swap_is_valid(g, S, sw)) {
        const std::vector<uint32_t> S2 = apply_swap(S, sw);
        if (hooks.on_improve) hooks.on_improve(R.nodes[id], sw, S2);
        R.improving_found = true;
        ++R.improvements;
        bool nw = false;
        const std::size_t nid = discover(S2, id, depth + 1,
                                         "improve(" + std::to_string(sw.remove.size()) + "," + std::to_string(sw.add.size()) + ")", nw);
        R.nodes[id].improving = true;
        R.nodes[id].improved_node = nid;
        if (S2.size() > R.best_size) { R.best_size = S2.size(); R.best_set = S2; }
      }
    }

    // ---- neighbours
    {
      const std::vector<Swap> combos = disjoint_move_combinations(R.nodes[id].moves, opt.combo_max_moves);
      const std::size_t singles = R.nodes[id].moves.size();   // combos lists the singles first
      std::size_t gen = 0, fresh = 0;
      for (std::size_t ci = 0; ci < combos.size(); ++ci) {
        const Swap& sw = combos[ci];
        if (!swap_is_valid(g, S, sw)) continue;
        ++gen;
        bool nw = false;
        const std::string via = (ci < singles ? "(" : "combo(") + std::to_string(sw.remove.size()) + "," +
                                std::to_string(sw.add.size()) + ")";
        discover(apply_swap(S, sw), id, depth + 1, via, nw);
        fresh += nw ? 1 : 0;
      }
      PlateauNode& nd = R.nodes[id];
      nd.neighbours = gen;
      nd.neighbours_new = fresh;
    }
    ++R.expanded;
    if (hooks.on_node) hooks.on_node(R.nodes[id], R);
  }

  R.fingerprints = fp_first.size();
  R.fingerprints0 = fp0_expanded.size();
  R.fingerprints0_all = fp0_all.size();
  R.frontier_left = frontier.size();
  R.seconds = secs_since(t0);
  return R;
}

}  // namespace kiss

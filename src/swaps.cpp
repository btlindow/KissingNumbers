// T3.1 — swap neighbourhoods (see include/kiss/swaps.h).
#include "kiss/swaps.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "kiss/bits.h"

namespace kiss {

namespace {

using clock_t_ = std::chrono::steady_clock;
double secs_since(clock_t_::time_point t0) {
  return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

// Position of each vertex in S (-1 if absent).
std::vector<int32_t> positions(uint32_t n, const std::vector<uint32_t>& S) {
  std::vector<int32_t> pos(n, -1);
  for (std::size_t i = 0; i < S.size(); ++i) {
    if (S[i] >= n) throw std::out_of_range("swaps: S index out of range");
    pos[S[i]] = static_cast<int32_t>(i);
  }
  return pos;
}

// ---------------------------------------------------------------------------
// Bitset MIS (branch and bound). Vertices 0..p-1, W = words per row.
// ---------------------------------------------------------------------------
struct Mis {
  int p = 0, W = 0;
  std::vector<uint64_t> adj;      // p × W
  std::vector<uint64_t> stack;    // (p + 2) × W scratch rows
  std::vector<int> cur, best;

  static int popc(const uint64_t* a, int W) {
    int c = 0;
    for (int w = 0; w < W; ++w) c += kiss::popcount64(a[w]);
    return c;
  }

  void rec(int depth, const uint64_t* cand) {
    const int cnt = popc(cand, W);
    if (cnt == 0) {
      if (cur.size() > best.size()) best = cur;
      return;
    }
    if (static_cast<int>(cur.size()) + cnt <= static_cast<int>(best.size())) return;
    // pick the candidate of maximum degree inside cand; if all degrees are 0
    // the whole candidate set is independent.
    int pick = -1, pickdeg = -1;
    for (int w = 0; w < W; ++w) {
      uint64_t bits = cand[w];
      while (bits) {
        const int v = w * 64 + kiss::ctz64(bits);
        bits &= bits - 1;
        const uint64_t* row = &adj[static_cast<std::size_t>(v) * static_cast<std::size_t>(W)];
        int d = 0;
        for (int x = 0; x < W; ++x) d += kiss::popcount64(row[x] & cand[x]);
        if (d > pickdeg) { pickdeg = d; pick = v; }
      }
    }
    if (pickdeg == 0) {
      // all independent
      std::vector<int> all = cur;
      for (int w = 0; w < W; ++w) {
        uint64_t bits = cand[w];
        while (bits) {
          all.push_back(w * 64 + kiss::ctz64(bits));
          bits &= bits - 1;
        }
      }
      if (all.size() > best.size()) best = std::move(all);
      return;
    }
    uint64_t* next = &stack[static_cast<std::size_t>(depth) * static_cast<std::size_t>(W)];
    const uint64_t* row = &adj[static_cast<std::size_t>(pick) * static_cast<std::size_t>(W)];
    // include pick
    for (int w = 0; w < W; ++w) next[w] = cand[w] & ~row[w];
    next[pick >> 6] &= ~(uint64_t{1} << (pick & 63));
    cur.push_back(pick);
    rec(depth + 1, next);
    cur.pop_back();
    // exclude pick
    for (int w = 0; w < W; ++w) next[w] = cand[w];
    next[pick >> 6] &= ~(uint64_t{1} << (pick & 63));
    rec(depth + 1, next);
  }

  // Returns indices (into verts) of a maximum independent subset.
  std::vector<int> solve(const SwapGraph& g, const std::vector<uint32_t>& verts) {
    p = static_cast<int>(verts.size());
    best.clear();
    cur.clear();
    if (p == 0) return best;
    W = (p + 63) / 64;
    adj.assign(static_cast<std::size_t>(p) * static_cast<std::size_t>(W), 0);
    for (int i = 0; i < p; ++i)
      for (int j = i + 1; j < p; ++j)
        if (g.adjacent(verts[static_cast<std::size_t>(i)], verts[static_cast<std::size_t>(j)])) {
          adj[static_cast<std::size_t>(i) * static_cast<std::size_t>(W) + static_cast<std::size_t>(j >> 6)] |=
              uint64_t{1} << (j & 63);
          adj[static_cast<std::size_t>(j) * static_cast<std::size_t>(W) + static_cast<std::size_t>(i >> 6)] |=
              uint64_t{1} << (i & 63);
        }
    stack.assign(static_cast<std::size_t>(p + 2) * static_cast<std::size_t>(W), 0);
    std::vector<uint64_t> all(static_cast<std::size_t>(W), 0);
    for (int i = 0; i < p; ++i) all[static_cast<std::size_t>(i >> 6)] |= uint64_t{1} << (i & 63);
    rec(0, all.data());
    std::sort(best.begin(), best.end());
    return best;
  }
};

// ---------------------------------------------------------------------------
// Small sorted set of S-positions (|R| ≤ KSWAP_MAX_K) used as hash key.
// ---------------------------------------------------------------------------
constexpr uint16_t EMPTY = 0xFFFF;

struct SetKey {
  std::array<uint16_t, KSWAP_MAX_K> e;
  int size() const {
    int n = 0;
    while (n < KSWAP_MAX_K && e[static_cast<std::size_t>(n)] != EMPTY) ++n;
    return n;
  }
  bool operator==(const SetKey& o) const { return e == o.e; }
};
struct SetKeyHash {
  std::size_t operator()(const SetKey& k) const {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint16_t x : k.e) {
      h ^= x;
      h *= 0x100000001b3ull;
      h ^= h >> 29;
    }
    return static_cast<std::size_t>(h);
  }
};
SetKey make_key(const uint16_t* a, int n) {
  SetKey k;
  k.e.fill(EMPTY);
  for (int i = 0; i < n; ++i) k.e[static_cast<std::size_t>(i)] = a[i];
  return k;
}
// |b \ a| for sorted a (na) and b (nb), early exit when > limit.
int diff_count(const uint16_t* a, int na, const uint16_t* b, int nb, int limit) {
  int i = 0, j = 0, d = 0;
  while (j < nb) {
    if (i < na && a[i] < b[j]) { ++i; continue; }
    if (i < na && a[i] == b[j]) { ++i; ++j; continue; }
    ++j;
    if (++d > limit) return d;
  }
  return d;
}
// merge sorted a and b into out; returns size (caller guarantees ≤ KSWAP_MAX_K).
int merge_sets(const uint16_t* a, int na, const uint16_t* b, int nb, uint16_t* out) {
  int i = 0, j = 0, n = 0;
  while (i < na || j < nb) {
    if (j >= nb || (i < na && a[i] < b[j])) out[n++] = a[i++];
    else if (i >= na || b[j] < a[i]) out[n++] = b[j++];
    else { out[n++] = a[i]; ++i; ++j; }
  }
  return n;
}

}  // namespace

// ---------------------------------------------------------------------------
// ListSwapGraph
// ---------------------------------------------------------------------------
ListSwapGraph::ListSwapGraph(uint32_t n, const std::vector<std::pair<uint32_t, uint32_t>>& edges)
    : nbr_(n) {
  for (const auto& e : edges) {
    if (e.first >= n || e.second >= n) throw std::out_of_range("ListSwapGraph: edge out of range");
    if (e.first == e.second) continue;
    nbr_[e.first].push_back(e.second);
    nbr_[e.second].push_back(e.first);
  }
  for (auto& l : nbr_) {
    std::sort(l.begin(), l.end());
    l.erase(std::unique(l.begin(), l.end()), l.end());
  }
}
bool ListSwapGraph::adjacent(uint32_t u, uint32_t v) const {
  const auto& l = nbr_[u];
  return std::binary_search(l.begin(), l.end(), v);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
std::vector<uint16_t> tightness_from_graph(const SwapGraph& g, const std::vector<uint32_t>& S) {
  if (S.size() > 65535) throw std::length_error("tightness_from_graph: |S| > 65535");
  const uint32_t n = g.num_vertices();
  std::vector<uint16_t> tight(n, 0);
  for (uint32_t s : S) {
    if (s >= n) throw std::out_of_range("tightness_from_graph: S index out of range");
    std::size_t cnt = 0;
    const uint32_t* row = g.neighbours(s, cnt);
    for (std::size_t i = 0; i < cnt; ++i) ++tight[row[i]];
  }
  return tight;
}

std::vector<std::size_t> tightness_histogram(const std::vector<uint16_t>& tight,
                                             const std::vector<uint32_t>& S) {
  std::vector<uint8_t> inS(tight.size(), 0);
  for (uint32_t s : S) inS[s] = 1;
  std::vector<std::size_t> h;
  for (std::size_t v = 0; v < tight.size(); ++v) {
    if (inS[v]) continue;
    const std::size_t t = tight[v];
    if (t >= h.size()) h.resize(t + 1, 0);
    ++h[t];
  }
  return h;
}

std::string histogram_string(const std::vector<std::size_t>& h) {
  std::string s;
  for (std::size_t i = 0; i < h.size(); ++i) {
    if (!h[i]) continue;
    if (!s.empty()) s += ' ';
    s += std::to_string(i) + ":" + std::to_string(h[i]);
  }
  return s;
}

std::vector<uint32_t> greedy_maximal_set(const SwapGraph& g, uint64_t seed) {
  const uint32_t n = g.num_vertices();
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint16_t> tight(n, 0);
  std::vector<uint32_t> S;
  for (uint32_t v : order) {
    if (tight[v] != 0) continue;
    S.push_back(v);
    std::size_t cnt = 0;
    const uint32_t* row = g.neighbours(v, cnt);
    for (std::size_t i = 0; i < cnt; ++i) ++tight[row[i]];
  }
  std::sort(S.begin(), S.end());
  return S;
}

bool is_independent(const SwapGraph& g, const std::vector<uint32_t>& S) {
  bool ok = true;
#pragma omp parallel for schedule(dynamic, 8)
  for (std::size_t i = 0; i < S.size(); ++i) {
    if (!ok) continue;
    for (std::size_t j = i + 1; j < S.size(); ++j)
      if (S[i] == S[j] || g.adjacent(S[i], S[j])) { ok = false; break; }
  }
  return ok;
}

std::vector<uint32_t> max_independent_subset(const SwapGraph& g, const std::vector<uint32_t>& verts) {
  Mis m;
  const std::vector<int> idx = m.solve(g, verts);
  std::vector<uint32_t> out;
  out.reserve(idx.size());
  for (int i : idx) out.push_back(verts[static_cast<std::size_t>(i)]);
  return out;
}

std::vector<uint32_t> apply_swap(const std::vector<uint32_t>& S, const Swap& sw) {
  std::vector<uint32_t> out = S;
  std::sort(out.begin(), out.end());
  for (uint32_t r : sw.remove) {
    auto it = std::lower_bound(out.begin(), out.end(), r);
    if (it == out.end() || *it != r) throw std::runtime_error("apply_swap: removed vertex not in S");
    out.erase(it);
  }
  for (uint32_t a : sw.add) {
    auto it = std::lower_bound(out.begin(), out.end(), a);
    if (it != out.end() && *it == a) throw std::runtime_error("apply_swap: added vertex already in S");
    out.insert(it, a);
  }
  return out;
}

bool swap_is_valid(const SwapGraph& g, const std::vector<uint32_t>& S, const Swap& sw) {
  std::vector<uint32_t> keep;
  keep.reserve(S.size());
  for (uint32_t s : S)
    if (std::find(sw.remove.begin(), sw.remove.end(), s) == sw.remove.end()) keep.push_back(s);
  if (keep.size() + sw.remove.size() != S.size()) return false;   // a removed vertex was not in S
  for (std::size_t i = 0; i < sw.add.size(); ++i) {
    const uint32_t a = sw.add[i];
    if (std::find(keep.begin(), keep.end(), a) != keep.end()) return false;
    for (uint32_t s : keep)
      if (g.adjacent(a, s)) return false;
    for (std::size_t j = i + 1; j < sw.add.size(); ++j)
      if (sw.add[j] == a || g.adjacent(a, sw.add[j])) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Classical (1,2)
// ---------------------------------------------------------------------------
Swap12Result find_swap12(const SwapGraph& g, const std::vector<uint32_t>& S,
                         const std::vector<uint16_t>& tight) {
  const auto t0 = clock_t_::now();
  Swap12Result r;
  const std::size_t n = S.size();
  const std::vector<int32_t> pos = positions(g.num_vertices(), S);
  r.L.assign(n, {});
  std::vector<uint8_t> hit(n, 0);
  std::vector<std::array<uint32_t, 2>> pair(n);

#pragma omp parallel for schedule(dynamic, 4)
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t cnt = 0;
    const uint32_t* row = g.neighbours(S[i], cnt);
    std::vector<uint32_t>& L = r.L[i];
    for (std::size_t j = 0; j < cnt; ++j)
      if (tight[row[j]] == 1 && pos[row[j]] < 0) L.push_back(row[j]);
    std::sort(L.begin(), L.end());
    for (std::size_t a = 0; a < L.size() && !hit[i]; ++a)
      for (std::size_t b = a + 1; b < L.size(); ++b)
        if (!g.adjacent(L[a], L[b])) {
          hit[i] = 1;
          pair[i] = {L[a], L[b]};
          break;
        }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t l = r.L[i].size();
    if (l >= r.L_hist.size()) r.L_hist.resize(l + 1, 0);
    ++r.L_hist[l];
    r.L_total += l;
    r.L_max = std::max(r.L_max, l);
    if (hit[i]) {
      ++r.hits;
      if (!r.found) {
        r.found = true;
        r.swap.remove = {S[i]};
        r.swap.add = {pair[i][0], pair[i][1]};
      }
    }
  }
  r.seconds = secs_since(t0);
  return r;
}

// ---------------------------------------------------------------------------
// Classical (2,3)
// ---------------------------------------------------------------------------
Swap23Result find_swap23(const SwapGraph& g, const std::vector<uint32_t>& S,
                         const std::vector<uint16_t>& tight, const Swap12Result& r12) {
  const auto t0 = clock_t_::now();
  Swap23Result r;
  const std::size_t n = S.size();
  if (r12.L.size() != n) throw std::runtime_error("find_swap23: r12 does not match S");
  const std::vector<int32_t> pos = positions(g.num_vertices(), S);

  // Bucket the tightness-2 vertices by their conf pair (i<j positions in S).
  std::unordered_map<uint64_t, std::vector<uint32_t>> t2;
  {
    // conf of tight-2 vertices: scan rows of S in position order.
    std::unordered_map<uint32_t, uint32_t> first;   // v -> first conf position
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t cnt = 0;
      const uint32_t* row = g.neighbours(S[i], cnt);
      for (std::size_t j = 0; j < cnt; ++j) {
        const uint32_t v = row[j];
        if (tight[v] != 2 || pos[v] >= 0) continue;
        auto it = first.find(v);
        if (it == first.end()) {
          first.emplace(v, static_cast<uint32_t>(i));
        } else {
          const uint64_t key = (static_cast<uint64_t>(it->second) << 32) | static_cast<uint64_t>(i);
          t2[key].push_back(v);
        }
      }
    }
    r.t2_candidates = first.size();
  }

  const std::size_t npairs = n * (n - 1) / 2;
  r.pairs = npairs;
  std::vector<uint8_t> hit(npairs, 0);
  std::vector<std::array<uint32_t, 3>> triple(npairs);
  std::vector<uint32_t> psize(npairs, 0);

#pragma omp parallel
  {
    Mis mis;
    std::vector<uint32_t> P;
#pragma omp for schedule(dynamic, 256)
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const std::size_t pi = i * n - i * (i + 1) / 2 + (j - i - 1);
        P.clear();
        P.insert(P.end(), r12.L[i].begin(), r12.L[i].end());
        P.insert(P.end(), r12.L[j].begin(), r12.L[j].end());
        auto it = t2.find((static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(j));
        if (it != t2.end()) P.insert(P.end(), it->second.begin(), it->second.end());
        psize[pi] = static_cast<uint32_t>(P.size());
        if (P.size() < 3) continue;
        std::sort(P.begin(), P.end());
        const std::vector<int> m = mis.solve(g, P);
        if (m.size() >= 3) {
          hit[pi] = 1;
          triple[pi] = {P[static_cast<std::size_t>(m[0])], P[static_cast<std::size_t>(m[1])],
                        P[static_cast<std::size_t>(m[2])]};
        }
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      const std::size_t pi = i * n - i * (i + 1) / 2 + (j - i - 1);
      const std::size_t p = psize[pi];
      if (p >= r.pool_hist.size()) r.pool_hist.resize(p + 1, 0);
      ++r.pool_hist[p];
      if (p > 0) ++r.pairs_nonempty;
      if (p >= 3) ++r.pairs_ge3;
      r.pool_max = std::max(r.pool_max, p);
      if (hit[pi]) {
        ++r.hits;
        if (!r.found) {
          r.found = true;
          r.swap.remove = {S[i], S[j]};
          r.swap.add = {triple[pi][0], triple[pi][1], triple[pi][2]};
        }
      }
    }
  r.seconds = secs_since(t0);
  return r;
}

// ---------------------------------------------------------------------------
// Generalised (k, k+1) search
// ---------------------------------------------------------------------------
namespace {

struct CandTable {
  std::vector<uint32_t> vert;      // candidate index c -> vertex
  std::vector<uint32_t> off;       // conf of c = conf_flat[off[c] .. off[c+1])
  std::vector<uint16_t> conf_flat; // S positions, sorted per candidate
  std::vector<std::vector<uint32_t>> by_pos;  // S position -> candidates containing it
  int tight(uint32_t c) const { return static_cast<int>(off[c + 1] - off[c]); }
  const uint16_t* conf(uint32_t c) const { return &conf_flat[off[c]]; }
};

CandTable build_candidates(const SwapGraph& g, const std::vector<uint32_t>& S,
                           const std::vector<uint16_t>& tight, const std::vector<int32_t>& pos, int kmax) {
  const uint32_t n = g.num_vertices();
  CandTable T;
  std::vector<int32_t> cid(n, -1);
  for (uint32_t v = 0; v < n; ++v)
    if (pos[v] < 0 && tight[v] >= 1 && tight[v] <= kmax) {
      cid[v] = static_cast<int32_t>(T.vert.size());
      T.vert.push_back(v);
    }
  const std::size_t nc = T.vert.size();
  T.off.assign(nc + 1, 0);
  for (std::size_t c = 0; c < nc; ++c) T.off[c + 1] = T.off[c] + tight[T.vert[c]];
  T.conf_flat.assign(T.off[nc], 0);
  std::vector<uint32_t> fill(nc, 0);
  T.by_pos.assign(S.size(), {});
  for (std::size_t i = 0; i < S.size(); ++i) {   // increasing i ⇒ conf lists sorted
    std::size_t cnt = 0;
    const uint32_t* row = g.neighbours(S[i], cnt);
    for (std::size_t j = 0; j < cnt; ++j) {
      const int32_t c = cid[row[j]];
      if (c < 0) continue;
      const auto cu = static_cast<uint32_t>(c);
      T.conf_flat[T.off[cu] + fill[cu]++] = static_cast<uint16_t>(i);
      T.by_pos[i].push_back(cu);
    }
  }
  for (std::size_t c = 0; c < nc; ++c)
    if (fill[c] != static_cast<uint32_t>(tight[T.vert[c]]))
      throw std::runtime_error("kswap: tight[] inconsistent with the graph rows");
  return T;
}

// Is the plateau move (R = uni, I = members) connected, i.e. is the bipartite
// graph I ↔ R (v — s iff s ∈ conf(v)) connected? Union-find over I, joining
// members that share an S position.
bool plateau_is_connected(const CandTable& T, const std::vector<uint32_t>& members,
                          const std::vector<uint16_t>& uni) {
  const std::size_t m = members.size();
  if (m <= 1) return true;
  std::vector<int> parent(m);
  for (std::size_t i = 0; i < m; ++i) parent[i] = static_cast<int>(i);
  auto find = [&](int a) {
    while (parent[static_cast<std::size_t>(a)] != a) {
      parent[static_cast<std::size_t>(a)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
      a = parent[static_cast<std::size_t>(a)];
    }
    return a;
  };
  std::vector<int> owner(uni.size(), -1);   // S position (index into uni) -> first member containing it
  int comps = static_cast<int>(m);
  for (std::size_t i = 0; i < m; ++i) {
    const uint32_t c = members[i];
    const uint16_t* cf = T.conf(c);
    const int t = T.tight(c);
    for (int j = 0; j < t; ++j) {
      const auto it = std::lower_bound(uni.begin(), uni.end(), cf[j]);
      const std::size_t pi = static_cast<std::size_t>(it - uni.begin());
      if (owner[pi] < 0) { owner[pi] = static_cast<int>(i); continue; }
      const int a = find(owner[pi]), b = find(static_cast<int>(i));
      if (a != b) { parent[static_cast<std::size_t>(a)] = b; --comps; }
    }
  }
  return comps == 1;
}

}  // namespace

KSwapResult kswap_search(const SwapGraph& g, const std::vector<uint32_t>& S,
                         const std::vector<uint16_t>& tight, const KSwapOptions& opt,
                         const std::function<void(const KSwapLevel&)>& on_level) {
  const auto t_all = clock_t_::now();
  if (opt.kmax < 1 || opt.kmax > KSWAP_MAX_K) throw std::invalid_argument("kswap_search: kmax out of range");
  if (S.size() >= EMPTY) throw std::length_error("kswap_search: |S| too large for SetKey");
  const int kmax = opt.kmax;
  KSwapResult R;
  const std::vector<int32_t> pos = positions(g.num_vertices(), S);

  // free vertices
  for (uint32_t v = 0; v < g.num_vertices(); ++v)
    if (pos[v] < 0 && tight[v] == 0) {
      ++R.free_vertices;
      if (R.free_list.size() < 64) R.free_list.push_back(v);
    }
  if (R.free_vertices) {
    R.improving_found = true;
    R.best.remove.clear();
    R.best.add = {R.free_list[0]};
  }

  const CandTable T = build_candidates(g, S, tight, pos, kmax);
  const std::size_t nc = T.vert.size();
  // candidates sorted by tightness, and per-tightness prefix: cand_by_t[t]
  std::vector<std::vector<uint32_t>> cand_by_t(static_cast<std::size_t>(kmax) + 1);
  for (std::size_t c = 0; c < nc; ++c) cand_by_t[static_cast<std::size_t>(T.tight(static_cast<uint32_t>(c)))].push_back(static_cast<uint32_t>(c));

  // Union enumeration state.
  std::unordered_set<SetKey, SetKeyHash> visited;
  std::vector<std::vector<SetKey>> bucket(static_cast<std::size_t>(kmax) + 1);
  R.levels.assign(static_cast<std::size_t>(kmax), KSwapLevel{});
  for (int k = 1; k <= kmax; ++k) {
    KSwapLevel& L = R.levels[static_cast<std::size_t>(k - 1)];
    L.k = k;
    L.cand_eq_k = cand_by_t[static_cast<std::size_t>(k)].size();
    L.cand_le_k = (k > 1 ? R.levels[static_cast<std::size_t>(k - 2)].cand_le_k : 0) + L.cand_eq_k;
  }
  // seeds: every distinct conf set
  for (std::size_t c = 0; c < nc; ++c) {
    const int t = T.tight(static_cast<uint32_t>(c));
    const SetKey key = make_key(T.conf(static_cast<uint32_t>(c)), t);
    if (visited.insert(key).second) {
      bucket[static_cast<std::size_t>(t)].push_back(key);
      ++R.levels[static_cast<std::size_t>(t - 1)].distinct_conf_eq_k;
    }
  }

  bool stop = false;
  int best_gain = R.free_vertices ? 1 : 0;
  const int nth = omp_get_max_threads();
  std::vector<std::vector<uint32_t>> stamps(static_cast<std::size_t>(nth), std::vector<uint32_t>(nc, 0));
  std::vector<uint32_t> gens(static_cast<std::size_t>(nth), 0);
  for (int k = 1; k <= kmax && !stop; ++k) {
    KSwapLevel& L = R.levels[static_cast<std::size_t>(k - 1)];
    const auto t_lvl = clock_t_::now();
    auto over_budget = [&]() {
      return (opt.time_limit_s > 0 && secs_since(t_lvl) > opt.time_limit_s) || visited.size() > opt.max_sets;
    };
    std::vector<SetKey>& sets = bucket[static_cast<std::size_t>(k)];

    // ---- 1. generate the unions of size exactly k from the smaller ones -----
    // Every union U of conf sets with |U| = k is either a single conf set (a
    // seed, already in bucket[k]) or R ∪ conf(c) for a union R with |R| = r < k
    // and a candidate c with |conf(c) \ R| = k − r =: d. Candidates with
    // tight == d must be disjoint from R (scan cand_by_t[d]); candidates with
    // d < tight ≤ k must meet R in tight − d members (found through by_pos of
    // R's members); tight < d or tight > k cannot give |R ∪ conf(c)| = k.
    // Generating level by level keeps memory proportional to the unions of
    // size ≤ k actually needed and makes "exhaustive up to k" meaningful even
    // when a later level is cut by the time limit.
    const std::size_t gen_chunk = 1024;
    for (int r = 1; r < k && !stop; ++r) {
      const int d = k - r;
      const std::vector<SetKey>& src = bucket[static_cast<std::size_t>(r)];
      for (std::size_t start = 0; start < src.size() && !stop; start += gen_chunk) {
        const std::size_t end = std::min(src.size(), start + gen_chunk);
        std::vector<std::vector<SetKey>> newkeys(static_cast<std::size_t>(nth));
#pragma omp parallel
        {
          const int tid = omp_get_thread_num();
          std::vector<uint32_t>& stamp = stamps[static_cast<std::size_t>(tid)];
          uint32_t& gen = gens[static_cast<std::size_t>(tid)];
          std::vector<SetKey>& out = newkeys[static_cast<std::size_t>(tid)];
          uint16_t buf[KSWAP_MAX_K];
#pragma omp for schedule(dynamic, 16)
          for (std::size_t si = start; si < end; ++si) {
            const uint16_t* Rs = src[si].e.data();
            ++gen;
            // (a) candidates of tightness d disjoint from R
            for (uint32_t c : cand_by_t[static_cast<std::size_t>(d)]) {
              if (diff_count(Rs, r, T.conf(c), d, d - 1) < d) continue;   // meets R → union smaller than k
              const int nn = merge_sets(Rs, r, T.conf(c), d, buf);
              out.push_back(make_key(buf, nn));
            }
            // (b) candidates of tightness d < t ≤ k meeting R in exactly t − d members
            for (int i = 0; i < r; ++i)
              for (uint32_t c : T.by_pos[Rs[i]]) {
                if (stamp[c] == gen) continue;
                stamp[c] = gen;
                const int t = T.tight(c);
                if (t <= d || t > k) continue;
                if (diff_count(Rs, r, T.conf(c), t, d) != d) continue;
                const int nn = merge_sets(Rs, r, T.conf(c), t, buf);
                out.push_back(make_key(buf, nn));
              }
          }
        }
        for (auto& nk : newkeys)
          for (const SetKey& key : nk)
            if (visited.insert(key).second) sets.push_back(key);
        if (over_budget()) { stop = true; L.exhaustive = false; }
      }
    }
    L.R_total = sets.size();

    // ---- 2. pool + exact MIS for every union of size k ----------------------
    const std::size_t chunk = 4096;
    for (std::size_t start = 0; start < sets.size() && !stop; start += chunk) {
      const std::size_t end = std::min(sets.size(), start + chunk);
      const std::size_t m = end - start;
      std::vector<uint32_t> pool_size(m, 0);
      std::vector<int> mis_size(m, 0);
      std::vector<std::vector<uint32_t>> mis_set(m);  // candidate ids of the MIS (only if mis ≥ k)
#pragma omp parallel
      {
        Mis mis;
        std::vector<uint32_t> pool, poolc;
#pragma omp for schedule(dynamic, 16)
        for (std::size_t si = start; si < end; ++si) {
          const uint16_t* Rs = sets[si].e.data();
          // pool: candidates with conf ⊆ R, taken from by_pos of their first conf member
          pool.clear();
          poolc.clear();
          for (int i = 0; i < k; ++i)
            for (uint32_t c : T.by_pos[Rs[i]]) {
              const uint16_t* cf = T.conf(c);
              const int t = T.tight(c);
              if (cf[0] != Rs[i] || t > k) continue;
              if (diff_count(Rs, k, cf, t, 0) == 0) { pool.push_back(T.vert[c]); poolc.push_back(c); }
            }
          pool_size[si - start] = static_cast<uint32_t>(pool.size());
          if (!pool.empty()) {
            const std::vector<int> ms = mis.solve(g, pool);
            mis_size[si - start] = static_cast<int>(ms.size());
            if (static_cast<int>(ms.size()) >= k) {
              std::vector<uint32_t>& out = mis_set[si - start];
              for (int x : ms) out.push_back(poolc[static_cast<std::size_t>(x)]);
            }
          }
        }
      }
      // merge results (serial)
      for (std::size_t si = start; si < end; ++si) {
        const std::size_t x = si - start;
        ++L.R_examined;
        L.largest_pool = std::max<std::size_t>(L.largest_pool, pool_size[x]);
        L.pool_total += pool_size[x];
        L.best_m = std::max(L.best_m, mis_size[x]);
        if (mis_size[x] >= k) {
          // actual union of the MIS's conf sets
          std::vector<uint16_t> uni;
          for (uint32_t c : mis_set[x]) uni.insert(uni.end(), T.conf(c), T.conf(c) + T.tight(c));
          std::sort(uni.begin(), uni.end());
          uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
          Swap sw;
          for (uint16_t p : uni) sw.remove.push_back(S[p]);
          for (uint32_t c : mis_set[x]) sw.add.push_back(T.vert[c]);
          std::sort(sw.add.begin(), sw.add.end());
          if (sw.gain() > 0) {
            ++L.improving_moves;
            R.improving_found = true;
            if (sw.gain() > best_gain) { best_gain = sw.gain(); R.best = sw; }
            if (L.improving.size() < opt.keep_moves) L.improving.push_back(sw);
          } else {   // gain == 0 and union == R (a smaller union would have gain > 0)
            ++L.plateau_moves;
            if (plateau_is_connected(T, mis_set[x], uni)) {
              ++L.plateau_connected;
              if (L.plateau.size() < opt.keep_moves) L.plateau.push_back(sw);
            } else if (L.plateau.size() < opt.keep_moves && L.plateau_connected == 0 && L.plateau_moves == 1) {
              L.plateau.push_back(sw);   // keep at least one example even if all moves are composite
            }
          }
        }
      }
      if (end < sets.size() && over_budget()) { stop = true; L.exhaustive = false; }
    }
    if (!stop && opt.stop_on_improving && L.improving_moves) stop = true;
    std::sort(L.improving.begin(), L.improving.end(),
              [](const Swap& a, const Swap& b) { return a.gain() > b.gain(); });
    L.seconds = secs_since(t_lvl);
    if (on_level) on_level(L);
    if (stop)
      for (int k2 = k + 1; k2 <= kmax; ++k2) R.levels[static_cast<std::size_t>(k2 - 1)].exhaustive = false;
  }
  R.unions_total = visited.size();
  R.kmax_exhaustive = 0;
  for (int k = 1; k <= kmax; ++k) {
    if (!R.levels[static_cast<std::size_t>(k - 1)].exhaustive) break;
    R.kmax_exhaustive = k;
  }
  R.seconds = secs_since(t_all);
  return R;
}

std::vector<ConfGroup> group_by_conf(const SwapGraph& g, const std::vector<uint32_t>& S,
                                     const std::vector<uint16_t>& tight, int t) {
  const std::vector<int32_t> pos = positions(g.num_vertices(), S);
  std::unordered_map<SetKey, std::vector<uint32_t>, SetKeyHash> groups;
  std::vector<std::vector<uint16_t>> conf;
  std::vector<uint32_t> verts;
  std::unordered_map<uint32_t, std::size_t> id;
  for (uint32_t v = 0; v < g.num_vertices(); ++v)
    if (pos[v] < 0 && tight[v] == t) { id[v] = verts.size(); verts.push_back(v); conf.emplace_back(); }
  for (std::size_t i = 0; i < S.size(); ++i) {
    std::size_t cnt = 0;
    const uint32_t* row = g.neighbours(S[i], cnt);
    for (std::size_t j = 0; j < cnt; ++j) {
      auto it = id.find(row[j]);
      if (it != id.end()) conf[it->second].push_back(static_cast<uint16_t>(i));
    }
  }
  if (t > KSWAP_MAX_K) throw std::invalid_argument("group_by_conf: t > KSWAP_MAX_K");
  for (std::size_t c = 0; c < verts.size(); ++c)
    groups[make_key(conf[c].data(), static_cast<int>(conf[c].size()))].push_back(verts[c]);
  std::vector<ConfGroup> out;
  for (auto& kv : groups) {
    ConfGroup gr;
    for (int i = 0; i < kv.first.size(); ++i) gr.conf.push_back(S[kv.first.e[static_cast<std::size_t>(i)]]);
    gr.members = kv.second;
    std::sort(gr.members.begin(), gr.members.end());
    gr.mis = max_independent_subset(g, gr.members);
    out.push_back(std::move(gr));
  }
  std::sort(out.begin(), out.end(), [](const ConfGroup& a, const ConfGroup& b) {
    if (a.members.size() != b.members.size()) return a.members.size() > b.members.size();
    return a.conf < b.conf;
  });
  return out;
}

}  // namespace kiss

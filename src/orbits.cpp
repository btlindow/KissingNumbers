// T3.4 — orbits, orbit conflict graph, weighted MIS of orbits (see include/kiss/orbits.h).
#include "kiss/orbits.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "kiss/golay.h"
#include "kiss/bits.h"

namespace kiss {

namespace {

std::size_t idx(uint64_t i) { return static_cast<std::size_t>(i); }
[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// Orbits
// ---------------------------------------------------------------------------
Orbits orbits(const Subgroup& H) {
  for (const IndexPerm& g : H.gens)
    if (g.size() != static_cast<std::size_t>(N)) fail("orbits: generator of wrong size");
  std::vector<uint32_t> parent(idx(N));
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&](uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (const IndexPerm& g : H.gens)
    for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
      uint32_t a = find(v), b = find(g[v]);
      if (a != b) {
        if (a < b) std::swap(a, b);
        parent[a] = b;  // keep the smaller root
      }
    }
  Orbits O;
  O.id.assign(idx(N), 0);
  // number orbits by smallest vertex: roots are minimal in their class, so the
  // first vertex whose root is itself starts a new orbit.
  std::vector<uint32_t> root_to_id(idx(N), UINT32_MAX);
  uint32_t n = 0;
  std::vector<uint32_t> sizes;
  for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
    const uint32_t r = find(v);
    if (root_to_id[r] == UINT32_MAX) {
      root_to_id[r] = n++;
      sizes.push_back(0);
    }
    O.id[v] = root_to_id[r];
    ++sizes[root_to_id[r]];
  }
  O.start.assign(idx(n) + 1, 0);
  for (uint32_t o = 0; o < n; ++o) O.start[o + 1] = O.start[o] + sizes[o];
  O.members.assign(idx(N), 0);
  std::vector<uint32_t> fill(O.start.begin(), O.start.end() - 1);
  for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) O.members[fill[O.id[v]]++] = v;
  return O;
}

bool orbits_invariant(const Orbits& O, const IndexPerm& g) {
  if (g.size() != O.id.size()) return false;
  for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v)
    if (O.id[g[v]] != O.id[v]) return false;
  return true;
}

bool is_orbit_union(const Orbits& O, const std::vector<uint32_t>& S_idx, std::vector<uint32_t>& orbit_ids) {
  std::vector<uint32_t> hit(idx(O.count()), 0);
  for (uint32_t v : S_idx) {
    if (v >= static_cast<uint32_t>(N)) fail("is_orbit_union: index out of range");
    ++hit[O.id[v]];
  }
  orbit_ids.clear();
  bool ok = true;
  std::vector<uint32_t> partial;
  for (uint32_t o = 0; o < O.count(); ++o) {
    if (hit[o] == 0) continue;
    if (hit[o] == O.size(o)) orbit_ids.push_back(o);
    else { ok = false; partial.push_back(o); }
  }
  if (!ok) orbit_ids = partial;
  return ok;
}

std::vector<uint32_t> union_of_orbits(const Orbits& O, const std::vector<uint32_t>& orbit_ids) {
  std::vector<uint32_t> out;
  for (uint32_t o : orbit_ids)
    for (uint32_t k = O.start[o]; k < O.start[o + 1]; ++k) out.push_back(O.members[k]);
  std::sort(out.begin(), out.end());
  return out;
}

// ---------------------------------------------------------------------------
// OrbitGraph
// ---------------------------------------------------------------------------
void OrbitGraph::neighbours_of_row(uint32_t o, std::vector<uint32_t>& out) const {
  const uint32_t r = orb_->rep(o);
  const uint32_t* row = adj_->row(r);
  out.resize(static_cast<std::size_t>(DEG));
  for (int k = 0; k < DEG; ++k) out[idx(k)] = orb_->id[row[k]];
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

OrbitGraph::OrbitGraph(const Orbits& O, const Adjacency& A, uint32_t max_explicit)
    : orb_(&O), adj_(&A), n_(O.count()) {
  weight_.resize(idx(n_));
  for (uint32_t o = 0; o < n_; ++o) weight_[o] = O.size(o);
  self_.assign(idx(n_), 0);
  degree_.assign(idx(n_), 0);
  explicit_ = n_ <= max_explicit;
  std::vector<std::vector<uint32_t>> rows;
  if (explicit_) rows.resize(idx(n_));
  uint64_t edges2 = 0;
  uint32_t nself = 0;
#pragma omp parallel
  {
    std::vector<uint32_t> buf;
    uint64_t my_edges = 0;
    uint32_t my_self = 0;
#pragma omp for schedule(dynamic, 64)
    for (int64_t oi = 0; oi < static_cast<int64_t>(n_); ++oi) {
      const uint32_t o = static_cast<uint32_t>(oi);
      neighbours_of_row(o, buf);
      auto it = std::lower_bound(buf.begin(), buf.end(), o);
      if (it != buf.end() && *it == o) {
        self_[o] = 1;
        ++my_self;
        buf.erase(it);
      }
      degree_[o] = static_cast<uint32_t>(buf.size());
      my_edges += buf.size();
      if (explicit_) rows[o] = buf;
    }
#pragma omp critical
    {
      edges2 += my_edges;
      nself += my_self;
    }
  }
  edges_ = edges2 / 2;
  n_self_ = nself;
  if (explicit_) {
    csr_start_.assign(idx(n_) + 1, 0);
    for (uint32_t o = 0; o < n_; ++o) csr_start_[o + 1] = csr_start_[o] + rows[o].size();
    csr_.resize(idx(csr_start_[n_]));
    for (uint32_t o = 0; o < n_; ++o) std::copy(rows[o].begin(), rows[o].end(), csr_.begin() + static_cast<std::ptrdiff_t>(csr_start_[o]));
  }
}

OrbitGraph::OrbitGraph(const std::vector<uint32_t>& weights, const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                       const std::vector<uint8_t>& self)
    : n_(static_cast<uint32_t>(weights.size())), weight_(weights) {
  explicit_ = true;
  self_.assign(idx(n_), 0);
  for (uint32_t o = 0; o < n_ && o < self.size(); ++o) self_[o] = self[o] ? 1 : 0;
  std::vector<std::vector<uint32_t>> rows(idx(n_));
  for (const auto& e : edges) {
    if (e.first >= n_ || e.second >= n_) fail("OrbitGraph: edge endpoint out of range");
    if (e.first == e.second) { self_[e.first] = 1; continue; }
    rows[e.first].push_back(e.second);
    rows[e.second].push_back(e.first);
  }
  degree_.assign(idx(n_), 0);
  csr_start_.assign(idx(n_) + 1, 0);
  for (uint32_t o = 0; o < n_; ++o) {
    auto& r = rows[o];
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    degree_[o] = static_cast<uint32_t>(r.size());
    csr_start_[o + 1] = csr_start_[o] + r.size();
    edges_ += r.size();
    if (self_[o]) ++n_self_;
  }
  edges_ /= 2;
  csr_.resize(idx(csr_start_[n_]));
  for (uint32_t o = 0; o < n_; ++o) std::copy(rows[o].begin(), rows[o].end(), csr_.begin() + static_cast<std::ptrdiff_t>(csr_start_[o]));
}

const uint32_t* OrbitGraph::neighbours(uint32_t o, uint32_t& count, std::vector<uint32_t>& buf) const {
  if (explicit_) {
    count = static_cast<uint32_t>(csr_start_[o + 1] - csr_start_[o]);
    return csr_.data() + csr_start_[o];
  }
  neighbours_of_row(o, buf);
  auto it = std::lower_bound(buf.begin(), buf.end(), o);
  if (it != buf.end() && *it == o) buf.erase(it);
  count = static_cast<uint32_t>(buf.size());
  return buf.data();
}

bool OrbitGraph::conflict_brute_force(const Leech& L, const Orbits& O, uint32_t a, uint32_t b) {
  for (uint32_t i = O.start[a]; i < O.start[a + 1]; ++i)
    for (uint32_t j = O.start[b]; j < O.start[b + 1]; ++j)
      if (dot(L.C[O.members[i]], L.C[O.members[j]]) == 16) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Weighted MIS: greedy, local search
// ---------------------------------------------------------------------------
namespace {

struct LSState {
  const OrbitGraph& G;
  std::vector<uint8_t> chosen;
  std::vector<uint32_t> cnt;      // # chosen neighbours
  std::vector<uint32_t> free_list;
  uint64_t weight = 0;
  std::vector<uint32_t> buf;

  explicit LSState(const OrbitGraph& g) : G(g), chosen(idx(g.n()), 0), cnt(idx(g.n()), 0) {}

  void add(uint32_t o) {
    chosen[o] = 1;
    weight += G.weight(o);
    uint32_t k;
    const uint32_t* nb = G.neighbours(o, k, buf);
    for (uint32_t t = 0; t < k; ++t) ++cnt[nb[t]];
  }
  void remove(uint32_t o) {
    chosen[o] = 0;
    weight -= G.weight(o);
    uint32_t k;
    const uint32_t* nb = G.neighbours(o, k, buf);
    for (uint32_t t = 0; t < k; ++t) {
      const uint32_t u = nb[t];
      if (--cnt[u] == 0 && !G.self_conflicting(u)) free_list.push_back(u);
    }
  }
  // add every free orbit, heaviest first (lazy list validated on pop)
  void fill(const std::vector<uint64_t>* tabu_until = nullptr, uint64_t now = 0) {
    std::sort(free_list.begin(), free_list.end(),
              [&](uint32_t a, uint32_t b) { return G.weight(a) != G.weight(b) ? G.weight(a) > G.weight(b) : a < b; });
    std::vector<uint32_t> keep;
    for (uint32_t v : free_list) {
      if (chosen[v] || cnt[v] != 0 || G.self_conflicting(v)) continue;
      if (tabu_until && (*tabu_until)[v] > now) { keep.push_back(v); continue; }
      add(v);
    }
    free_list = keep;  // tabu ones stay as candidates
  }
  OrbitSet to_set() const {
    OrbitSet s;
    for (uint32_t o = 0; o < G.n(); ++o)
      if (chosen[o]) s.orbits.push_back(o);
    s.weight = weight;
    return s;
  }
  void load(const OrbitSet& s) {
    std::fill(chosen.begin(), chosen.end(), 0);
    std::fill(cnt.begin(), cnt.end(), 0);
    free_list.clear();
    weight = 0;
    for (uint32_t o : s.orbits) add(o);
  }
};

}  // namespace

OrbitSet greedy_orbit_mis(const OrbitGraph& G) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<uint32_t> order;
  order.reserve(idx(G.n()));
  for (uint32_t o = 0; o < G.n(); ++o)
    if (!G.self_conflicting(o)) order.push_back(o);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    if (G.weight(a) != G.weight(b)) return G.weight(a) > G.weight(b);
    if (G.degree(a) != G.degree(b)) return G.degree(a) < G.degree(b);
    return a < b;
  });
  LSState st(G);
  for (uint32_t o : order)
    if (st.cnt[o] == 0) st.add(o);
  OrbitSet s = st.to_set();
  s.seconds = seconds_since(t0);
  return s;
}

OrbitSet local_search_orbit_mis(const OrbitGraph& G, const OrbitSet& init, double seconds, uint64_t seed,
                                uint64_t max_iters) {
  const auto t0 = std::chrono::steady_clock::now();
  std::mt19937_64 rng(seed);
  LSState st(G);
  st.load(init);
  OrbitSet best = st.to_set();
  best.weight = st.weight;
  // candidate pool: non-self-conflicting orbits
  std::vector<uint32_t> pool;
  for (uint32_t o = 0; o < G.n(); ++o)
    if (!G.self_conflicting(o)) pool.push_back(o);
  if (pool.empty()) { best.seconds = seconds_since(t0); return best; }
  std::uniform_int_distribution<std::size_t> U(0, pool.size() - 1);
  std::vector<uint64_t> tabu_until(idx(G.n()), 0);
  const uint64_t tenure = 10;
  uint64_t it = 0, since_improve = 0;
  std::vector<uint32_t> nbbuf;
  while (true) {
    if ((it & 63) == 0 && seconds_since(t0) >= seconds) break;
    if (max_iters && it >= max_iters) break;
    ++it;
    // pick a non-chosen candidate with small loss (sample a few)
    uint32_t bestv = UINT32_MAX;
    int64_t best_gain = INT64_MIN;
    for (int s = 0; s < 6; ++s) {
      const uint32_t v = pool[U(rng)];
      if (st.chosen[v] || tabu_until[v] > it) continue;
      uint32_t k;
      const uint32_t* nb = G.neighbours(v, k, st.buf);
      int64_t loss = 0;
      for (uint32_t t = 0; t < k; ++t)
        if (st.chosen[nb[t]]) loss += G.weight(nb[t]);
      const int64_t gain = static_cast<int64_t>(G.weight(v)) - loss;
      if (gain > best_gain) { best_gain = gain; bestv = v; }
    }
    if (bestv == UINT32_MAX) continue;
    // force-add bestv: remove its chosen neighbours (they become tabu)
    {
      uint32_t k;
      const uint32_t* nb = G.neighbours(bestv, k, nbbuf);
      nbbuf.assign(nb, nb + k);
    }
    for (uint32_t u : nbbuf)
      if (st.chosen[u]) { st.remove(u); tabu_until[u] = it + tenure; }
    st.add(bestv);
    st.fill(&tabu_until, it);
    if (st.weight > best.weight) {
      best = st.to_set();
      since_improve = 0;
    } else {
      ++since_improve;
      if (since_improve >= 2000) {  // restart from best
        st.load(best);
        since_improve = 0;
      }
    }
  }
  best.seconds = seconds_since(t0);
  best.nodes = it;
  return best;
}

// ---------------------------------------------------------------------------
// Branch and bound
// ---------------------------------------------------------------------------
namespace {

struct BB {
  uint32_t m = 0, W = 0;
  std::vector<uint32_t> vert;       // compact index -> orbit id
  std::vector<uint32_t> w;          // weights by compact index
  std::vector<uint64_t> adj;        // m × W words
  std::vector<uint32_t> best_set;   // compact indices
  uint64_t best = 0;
  uint64_t nodes = 0;
  bool timed_out = false;
  double limit = 0;
  std::chrono::steady_clock::time_point t0;
  std::vector<uint32_t> cur;

  const uint64_t* row(uint32_t i) const { return adj.data() + static_cast<std::size_t>(i) * W; }

  void expand(std::vector<uint64_t>& P, uint64_t curw) {
    if (++nodes % 256 == 0 && seconds_since(t0) > limit) { timed_out = true; return; }
    // list P
    std::vector<uint32_t> verts;
    for (uint32_t wi = 0; wi < W; ++wi) {
      uint64_t x = P[wi];
      while (x) {
        const int b = kiss::ctz64(x);
        x &= x - 1;
        verts.push_back(wi * 64 + static_cast<uint32_t>(b));
      }
    }
    if (verts.empty()) {
      if (curw > best) { best = curw; best_set = cur; }
      return;
    }
    // greedy clique cover in index order (index order = weight desc)
    std::vector<std::vector<uint64_t>> cliques;
    std::vector<uint32_t> cmax;
    std::vector<uint32_t> cof(verts.size());
    for (std::size_t vi = 0; vi < verts.size(); ++vi) {
      const uint32_t v = verts[vi];
      const uint64_t* av = row(v);
      std::size_t c = 0;
      for (; c < cliques.size(); ++c) {
        bool all = true;
        for (uint32_t wi = 0; wi < W; ++wi)
          if (cliques[c][wi] & ~av[wi]) { all = false; break; }
        if (all) break;
      }
      if (c == cliques.size()) {
        cliques.emplace_back(W, 0);
        cmax.push_back(w[v]);
      }
      cliques[c][v >> 6] |= 1ull << (v & 63);
      cof[vi] = static_cast<uint32_t>(c);
    }
    // order by clique, prefix bounds
    std::vector<uint32_t> ord(verts.size());
    std::iota(ord.begin(), ord.end(), 0u);
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return cof[a] < cof[b]; });
    std::vector<uint64_t> bound(verts.size());
    uint64_t acc = 0;
    uint32_t lastc = UINT32_MAX;
    for (std::size_t k = 0; k < ord.size(); ++k) {
      const uint32_t c = cof[ord[k]];
      if (c != lastc) { acc += cmax[c]; lastc = c; }
      bound[k] = acc;
    }
    std::vector<uint64_t> Pn(W);
    for (std::size_t k = ord.size(); k-- > 0;) {
      if (curw + bound[k] <= best) return;
      const uint32_t v = verts[ord[k]];
      const uint64_t* av = row(v);
      for (uint32_t wi = 0; wi < W; ++wi) Pn[wi] = P[wi] & ~av[wi];
      Pn[v >> 6] &= ~(1ull << (v & 63));
      cur.push_back(v);
      expand(Pn, curw + w[v]);
      cur.pop_back();
      if (timed_out) return;
      P[v >> 6] &= ~(1ull << (v & 63));
    }
  }
};

}  // namespace

OrbitSet branch_and_bound_orbit_mis(const OrbitGraph& G, const OrbitSet& init, double seconds,
                                    uint32_t max_vertices) {
  if (!G.is_explicit()) fail("branch_and_bound_orbit_mis: orbit graph is not explicit");
  const auto t0 = std::chrono::steady_clock::now();
  BB bb;
  bb.t0 = t0;
  bb.limit = seconds;
  for (uint32_t o = 0; o < G.n(); ++o)
    if (!G.self_conflicting(o)) bb.vert.push_back(o);
  std::sort(bb.vert.begin(), bb.vert.end(), [&](uint32_t a, uint32_t b) {
    if (G.weight(a) != G.weight(b)) return G.weight(a) > G.weight(b);
    if (G.degree(a) != G.degree(b)) return G.degree(a) > G.degree(b);
    return a < b;
  });
  bb.m = static_cast<uint32_t>(bb.vert.size());
  if (bb.m > max_vertices) fail("branch_and_bound_orbit_mis: too many orbits (" + std::to_string(bb.m) + ")");
  bb.W = (bb.m + 63) / 64;
  std::vector<uint32_t> compact(idx(G.n()), UINT32_MAX);
  for (uint32_t i = 0; i < bb.m; ++i) compact[bb.vert[i]] = i;
  bb.w.resize(idx(bb.m));
  bb.adj.assign(static_cast<std::size_t>(bb.m) * bb.W, 0);
  std::vector<uint32_t> buf;
  for (uint32_t i = 0; i < bb.m; ++i) {
    const uint32_t o = bb.vert[i];
    bb.w[i] = G.weight(o);
    uint32_t k;
    const uint32_t* nb = G.neighbours(o, k, buf);
    uint64_t* r = bb.adj.data() + static_cast<std::size_t>(i) * bb.W;
    for (uint32_t t = 0; t < k; ++t) {
      const uint32_t j = compact[nb[t]];
      if (j != UINT32_MAX) r[j >> 6] |= 1ull << (j & 63);
    }
  }
  bb.best = init.weight;
  for (uint32_t o : init.orbits) bb.best_set.push_back(compact[o]);
  std::vector<uint64_t> P(bb.W, 0);
  for (uint32_t i = 0; i < bb.m; ++i) P[i >> 6] |= 1ull << (i & 63);
  bb.expand(P, 0);
  OrbitSet s;
  for (uint32_t i : bb.best_set) s.orbits.push_back(bb.vert[i]);
  std::sort(s.orbits.begin(), s.orbits.end());
  s.weight = bb.best;
  s.optimal = !bb.timed_out;
  s.nodes = bb.nodes;
  s.seconds = seconds_since(t0);
  return s;
}

bool orbit_set_valid(const OrbitGraph& G, const OrbitSet& s) {
  std::vector<uint8_t> in(idx(G.n()), 0);
  uint64_t wsum = 0;
  for (uint32_t o : s.orbits) {
    if (o >= G.n() || in[o] || G.self_conflicting(o)) return false;
    in[o] = 1;
    wsum += G.weight(o);
  }
  if (wsum != s.weight) return false;
  std::vector<uint32_t> buf;
  for (uint32_t o : s.orbits) {
    uint32_t k;
    const uint32_t* nb = G.neighbours(o, k, buf);
    for (uint32_t t = 0; t < k; ++t)
      if (in[nb[t]]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Monomial helpers
// ---------------------------------------------------------------------------
int monomial_order(const Monomial& m) {
  const Monomial id = monomial_identity();
  Monomial p = m;
  int k = 1;
  while (!(p == id)) {
    p = compose(m, p);
    ++k;
    if (k > 100000) fail("monomial_order: runaway");
  }
  return k;
}

namespace {
struct MonoKey {
  std::size_t operator()(const Monomial& m) const {
    uint64_t h = m.signs * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < DIM; ++i) h = (h ^ m.perm[idx(i)]) * 0x100000001B3ull;
    return static_cast<std::size_t>(h);
  }
};
}  // namespace

uint64_t monomial_group_order(const std::vector<Monomial>& gens, uint64_t cap) {
  std::unordered_set<Monomial, MonoKey> seen;
  std::vector<Monomial> frontier{monomial_identity()};
  seen.insert(frontier[0]);
  while (!frontier.empty()) {
    std::vector<Monomial> next;
    for (const Monomial& x : frontier)
      for (const Monomial& g : gens) {
        const Monomial y = compose(g, x);
        if (seen.insert(y).second) {
          next.push_back(y);
          if (seen.size() > cap) return 0;
        }
      }
    frontier.swap(next);
  }
  return seen.size();
}

uint64_t index_perm_order(const IndexPerm& g) {
  std::vector<uint8_t> seen(g.size(), 0);
  uint64_t l = 1;
  for (std::size_t i = 0; i < g.size(); ++i) {
    if (seen[i]) continue;
    uint64_t len = 0;
    std::size_t j = i;
    while (!seen[j]) { seen[j] = 1; j = g[j]; ++len; }
    l = std::lcm(l, len);
  }
  return l;
}

Monomial random_monomial_word(const std::vector<Monomial>& gens, int len, std::mt19937_64& rng) {
  if (gens.empty()) fail("random_monomial_word: no generators");
  std::uniform_int_distribution<std::size_t> U(0, gens.size() - 1);
  Monomial m = monomial_identity();
  for (int k = 0; k < len; ++k) m = compose(gens[U(rng)], m);
  return m;
}

std::vector<Monomial> load_monomial_list(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) fail("load_monomial_list: cannot open " + path.string());
  std::vector<Monomial> out;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    std::istringstream is(line);
    std::vector<long> t;
    long v;
    while (is >> v) t.push_back(v);
    if (t.empty()) continue;
    if (t.size() != DIM + 1) fail(path.string() + ":" + std::to_string(lineno) + ": expected 24 images + sign mask");
    Monomial m;
    for (int i = 0; i < DIM; ++i) {
      if (t[idx(i)] < 0 || t[idx(i)] >= DIM) fail(path.string() + ":" + std::to_string(lineno) + ": image out of range");
      m.perm[idx(i)] = static_cast<uint8_t>(t[idx(i)]);
    }
    if (!is_permutation(m.perm)) fail(path.string() + ":" + std::to_string(lineno) + ": not a permutation");
    if (t[DIM] < 0 || t[DIM] >= (1L << 24)) fail(path.string() + ":" + std::to_string(lineno) + ": sign mask out of range");
    m.signs = static_cast<uint32_t>(t[DIM]);
    if (!golay_is_codeword(m.signs)) fail(path.string() + ":" + std::to_string(lineno) + ": sign mask is not a Golay codeword");
    out.push_back(m);
  }
  return out;
}

void save_monomial_list(const std::filesystem::path& path, const std::vector<Monomial>& ms,
                        const std::string& header_comment) {
  std::ofstream out(path);
  if (!out) fail("save_monomial_list: cannot open " + path.string());
  if (!header_comment.empty()) {
    std::istringstream is(header_comment);
    std::string l;
    while (std::getline(is, l)) out << "# " << l << "\n";
  }
  for (const Monomial& m : ms) {
    for (int i = 0; i < DIM; ++i) out << (i ? " " : "") << static_cast<int>(m.perm[idx(i)]);
    out << " " << m.signs << "\n";
  }
  if (!out) fail("save_monomial_list: write failed");
}

}  // namespace kiss

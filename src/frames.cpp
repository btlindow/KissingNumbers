// T3.3 — frame-structured search: implementation of kiss/frames.h.
//
// Algorithms (all exact integer arithmetic; OpenMP where noted):
//  * make_classes: class id = rank of the representative v < neg[v].
//  * Bitset: 64-bit words, popcount by kiss::popcount64 (kiss/bits.h).
//  * induced_graph: |pool|² dots, OpenMP over rows (no symmetry, no races).
//  * maximal_cliques: Bron–Kerbosch with the Tomita pivot (u ∈ P ∪ X maximising
//    |P ∩ N(u)|), bitsets, explicit recursion.
//  * k_cliques: ordered DFS (candidates > last member) with |R| + |P| ≥ k.
//  * max_clique: greedy-colouring branch and bound (Tomita & Seki MCQ):
//    candidates coloured greedily in degree order, expanded from the highest
//    colour down, pruned when |R| + colour ≤ best.
//  * count_cliques_through: the neighbourhood of a class as a dense bitset
//    graph; DFS over index-ordered candidates with P_{d+1} = P_d ∩ N(v) ∩ {>v};
//    at the last level only a popcount is needed (leaves are not visited one by
//    one); OpenMP dynamic over the first-level vertex.
//  * estimate_cliques_through: unbiased random-descent estimator (unordered:
//    v_1 uniform in the neighbourhood, v_2 uniform in N(v_1), ...; the product
//    of the candidate-set sizes divided by (k−1)! is an unbiased estimate of
//    the k-cliques through the centre). Knuth's estimator on the *ordered*
//    DFS tree was tried first and is useless here (almost every uniform
//    descent dies in an ordering dead end: rel. se ≈ 100%).
//  * extend_from_class: DFS over cliques of OUTSIDERS (classes ∉ X with
//    conf ≤ tmax, orthogonal to the seed) with the X-members orthogonal to
//    the clique carried as a bitset XP (they never change the gain; a clique
//    Q_out extends to a frame iff XP contains a (24 − |Q_out|)-clique, counted
//    by k_cliques on the induced subgraph). Bound: with confU = ∪ conf of the
//    clique so far, marginal cost m_u = |conf(u) \ confU| for every candidate
//    u, and m_(1) ≤ m_(2) ≤ … the sorted costs, any extension by r candidates
//    has gain ≤ new + r − |confU| − m_(r) (the union grows by at least the
//    largest marginal cost of the added set, which is ≥ m_(r)); maximise over
//    r ≤ min(|P|, 24 − |Q_out| − 0). Prune when that maximum < min_gain.
//  * greedy_frame_union / clique_local_search: see the header.
#include "kiss/frames.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include <omp.h>

#include "kiss/verify.h"
#include "kiss/bits.h"

namespace kiss {

namespace {

using clock_t_ = std::chrono::steady_clock;
double secs_since(clock_t_::time_point t0) {
  return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------
Classes make_classes(const Leech& L) {
  const std::size_t n = L.C.size();
  if (n != static_cast<std::size_t>(N) || L.neg.size() != n)
    throw std::runtime_error("make_classes: Leech has the wrong size");
  Classes K;
  K.rep.reserve(static_cast<std::size_t>(NCLASS));
  K.cls_of.assign(n, 0);
  for (std::size_t v = 0; v < n; ++v) {
    const uint32_t nv = L.neg[v];
    if (v < nv) {
      const uint32_t id = static_cast<uint32_t>(K.rep.size());
      K.rep.push_back(static_cast<uint32_t>(v));
      K.cls_of[v] = id;
      K.cls_of[nv] = id;
    }
  }
  if (K.rep.size() != static_cast<std::size_t>(NCLASS))
    throw std::runtime_error("make_classes: expected 98280 classes");
  return K;
}

std::vector<uint32_t> set_classes(const Leech& L, const Classes& K, const std::vector<Vec>& S) {
  std::vector<uint32_t> out;
  out.reserve(S.size());
  for (std::size_t k = 0; k < S.size(); ++k) {
    const int32_t i = L.index_of(S[k]);
    if (i < 0) throw std::runtime_error("set_classes: row " + std::to_string(k) + " is not a minimal vector");
    out.push_back(K.cls_of[static_cast<std::size_t>(i)]);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<Vec> classes_to_vectors(const Leech& L, const Classes& K, const std::vector<uint32_t>& X) {
  std::vector<Vec> S;
  S.reserve(2 * X.size());
  for (uint32_t c : X) {
    const uint32_t r = K.rep[c];
    S.push_back(L.C[r]);
    S.push_back(L.C[L.neg[r]]);
  }
  return S;
}

bool classes_independent(const Leech& L, const Classes& K, const std::vector<uint32_t>& X) {
  for (std::size_t i = 0; i < X.size(); ++i)
    for (std::size_t j = i + 1; j < X.size(); ++j)
      if (X[i] == X[j] || class_conflict(L, K, X[i], X[j])) return false;
  return true;
}

bool classes_orthogonal_clique(const Leech& L, const Classes& K, const std::vector<uint32_t>& Q) {
  for (std::size_t i = 0; i < Q.size(); ++i)
    for (std::size_t j = i + 1; j < Q.size(); ++j)
      if (Q[i] == Q[j] || !class_orthogonal(L, K, Q[i], Q[j])) return false;
  return true;
}

bool cliques_compatible(const Leech& L, const Classes& K, const std::vector<uint32_t>& A,
                        const std::vector<uint32_t>& B) {
  for (uint32_t a : A)
    for (uint32_t b : B)
      if (a != b && class_conflict(L, K, a, b)) return false;
  return true;
}

std::vector<uint16_t> class_conflicts(const Leech& L, const Classes& K, const std::vector<uint32_t>& X) {
  std::vector<Vec> xs;
  xs.reserve(X.size());
  for (uint32_t x : X) xs.push_back(L.C[K.rep[x]]);
  std::vector<uint16_t> out(static_cast<std::size_t>(NCLASS), 0);
  const long nc = NCLASS;
#pragma omp parallel for schedule(static)
  for (long c = 0; c < nc; ++c) {
    const Vec& v = L.C[K.rep[static_cast<std::size_t>(c)]];
    unsigned cnt = 0;
    for (const Vec& x : xs) {
      const int d = dot(v, x);
      cnt += (d == 16 || d == -16) ? 1u : 0u;
    }
    out[static_cast<std::size_t>(c)] = static_cast<uint16_t>(cnt);
  }
  return out;
}

std::vector<uint32_t> conflicting_classes(const Leech& L, const Classes& K, uint32_t c) {
  std::vector<uint32_t> out;
  out.reserve(static_cast<std::size_t>(CONFLICT_DEG));
  const Vec& v = L.C[K.rep[c]];
  for (int b = 0; b < NCLASS; ++b) {
    const int d = dot(v, L.C[K.rep[static_cast<std::size_t>(b)]]);
    if (d == 16 || d == -16) out.push_back(static_cast<uint32_t>(b));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Bitset
// ---------------------------------------------------------------------------
void Bitset::clear() { std::fill(w_.begin(), w_.end(), 0); }
void Bitset::fill() {
  std::fill(w_.begin(), w_.end(), ~uint64_t{0});
  if (n_ & 63) w_.back() &= (uint64_t{1} << (n_ & 63)) - 1;
}
bool Bitset::empty() const {
  for (uint64_t x : w_)
    if (x) return false;
  return true;
}
int Bitset::count() const {
  int c = 0;
  for (uint64_t x : w_) c += kiss::popcount64(x);
  return c;
}
void Bitset::and_with(const Bitset& o) {
  for (std::size_t k = 0; k < w_.size(); ++k) w_[k] &= o.w_[k];
}
void Bitset::and_not(const Bitset& o) {
  for (std::size_t k = 0; k < w_.size(); ++k) w_[k] &= ~o.w_[k];
}
void Bitset::or_with(const Bitset& o) {
  for (std::size_t k = 0; k < w_.size(); ++k) w_[k] |= o.w_[k];
}
int Bitset::intersect_above(const Bitset& a, const Bitset& b, int v) {
  const std::size_t nw = w_.size();
  const std::size_t k0 = static_cast<std::size_t>((v + 1) >> 6);
  int c = 0;
  for (std::size_t k = 0; k < k0 && k < nw; ++k) w_[k] = 0;
  for (std::size_t k = k0; k < nw; ++k) {
    uint64_t x = a.w_[k] & b.w_[k];
    if (k == k0 && ((v + 1) & 63)) x &= ~uint64_t{0} << ((v + 1) & 63);
    w_[k] = x;
    c += kiss::popcount64(x);
  }
  return c;
}
int Bitset::next(int i) const {
  if (i >= n_) return -1;
  std::size_t k = static_cast<std::size_t>(i >> 6);
  uint64_t x = w_[k] & (~uint64_t{0} << (i & 63));
  while (true) {
    if (x) return static_cast<int>(k * 64) + kiss::ctz64(x);
    if (++k >= w_.size()) return -1;
    x = w_[k];
  }
}
std::vector<int> Bitset::to_list() const {
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(count()));
  for_each([&](int i) { out.push_back(i); });
  return out;
}

void orthogonal_classes(const Leech& L, const Classes& K, uint32_t c, Bitset& out) {
  if (out.size() != NCLASS) out = Bitset(NCLASS);
  out.clear();
  const Vec& v = L.C[K.rep[c]];
  uint64_t* w = out.words();
  for (int b = 0; b < NCLASS; ++b)
    if (dot(v, L.C[K.rep[static_cast<std::size_t>(b)]]) == 0) w[b >> 6] |= uint64_t{1} << (b & 63);
  out.reset(static_cast<int>(c));  // dot(v,v) = 32 anyway; defensive
}

// ---------------------------------------------------------------------------
// Pool graphs
// ---------------------------------------------------------------------------
std::map<int, long> PoolGraph::degree_histogram() const {
  std::map<int, long> h;
  for (int i = 0; i < size(); ++i) ++h[degree(i)];
  return h;
}

PoolGraph induced_graph(const Leech& L, const Classes& K, const std::vector<uint32_t>& pool,
                        const std::function<bool(int)>& rel) {
  PoolGraph G;
  G.pool = pool;
  const int n = G.size();
  G.adj.assign(static_cast<std::size_t>(n), Bitset(n));
  std::vector<Vec> vs;
  vs.reserve(pool.size());
  for (uint32_t c : pool) vs.push_back(L.C[K.rep[c]]);
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < n; ++i) {
    Bitset& row = G.adj[static_cast<std::size_t>(i)];
    const Vec& a = vs[static_cast<std::size_t>(i)];
    for (int j = 0; j < n; ++j)
      if (j != i && rel(dot(a, vs[static_cast<std::size_t>(j)]))) row.set(j);
  }
  return G;
}

PoolGraph induced_orthogonality(const Leech& L, const Classes& K, const std::vector<uint32_t>& pool) {
  return induced_graph(L, K, pool, [](int d) { return d == 0; });
}

PoolGraph pool_graph_from_edges(int n, const std::vector<std::pair<int, int>>& edges) {
  PoolGraph G;
  G.pool.resize(static_cast<std::size_t>(n));
  std::iota(G.pool.begin(), G.pool.end(), 0u);
  G.adj.assign(static_cast<std::size_t>(n), Bitset(n));
  for (const auto& e : edges) {
    if (e.first == e.second) continue;
    G.adj[static_cast<std::size_t>(e.first)].set(e.second);
    G.adj[static_cast<std::size_t>(e.second)].set(e.first);
  }
  return G;
}

// ---------------------------------------------------------------------------
// Maximal cliques (Bron–Kerbosch with pivot)
// ---------------------------------------------------------------------------
namespace {

struct BKState {
  const PoolGraph& G;
  const std::function<bool(const std::vector<int>&)>& cb;
  CliqueStats& st;
  clock_t_::time_point t0;
  double limit;
  bool stop = false;
  std::vector<int> R;
  std::vector<Bitset> Pst, Xst, tmp;

  void rec(int d) {
    if (stop) return;
    ++st.nodes;
    if ((st.nodes & 4095) == 0 && limit > 0 && secs_since(t0) > limit) {
      stop = true;
      st.complete = false;
      return;
    }
    if (static_cast<int>(Pst.size()) <= d + 1) {   // grow before taking references (reallocation)
      Pst.emplace_back(G.size());
      Xst.emplace_back(G.size());
      tmp.emplace_back(G.size());
    }
    Bitset& P = Pst[static_cast<std::size_t>(d)];
    Bitset& X = Xst[static_cast<std::size_t>(d)];
    if (P.empty()) {
      if (X.empty()) {
        ++st.count;
        ++st.by_size[static_cast<int>(R.size())];
        if (cb && !cb(R)) {
          stop = true;
          st.complete = false;
        }
      }
      return;
    }
    // pivot u ∈ P ∪ X maximising |P ∩ N(u)|
    int bestu = -1, bestc = -1;
    Bitset& t = tmp[static_cast<std::size_t>(d)];
    auto consider = [&](int u) {
      t = P;
      t.and_with(G.adj[static_cast<std::size_t>(u)]);
      const int c = t.count();
      if (c > bestc) {
        bestc = c;
        bestu = u;
      }
    };
    P.for_each(consider);
    X.for_each(consider);
    t = P;
    t.and_not(G.adj[static_cast<std::size_t>(bestu)]);
    const std::vector<int> cand = t.to_list();
    // below only indexed accesses: rec(d + 1) may grow the vectors and invalidate P, X, t
    for (int v : cand) {
      Bitset& P2 = Pst[static_cast<std::size_t>(d + 1)];
      Bitset& X2 = Xst[static_cast<std::size_t>(d + 1)];
      P2 = Pst[static_cast<std::size_t>(d)];
      P2.and_with(G.adj[static_cast<std::size_t>(v)]);
      X2 = Xst[static_cast<std::size_t>(d)];
      X2.and_with(G.adj[static_cast<std::size_t>(v)]);
      R.push_back(v);
      rec(d + 1);
      R.pop_back();
      if (stop) return;
      Pst[static_cast<std::size_t>(d)].reset(v);
      Xst[static_cast<std::size_t>(d)].set(v);
    }
  }
};

}  // namespace

CliqueStats maximal_cliques(const PoolGraph& G, const std::function<bool(const std::vector<int>&)>& cb,
                            double time_limit_s) {
  CliqueStats st;
  const auto t0 = clock_t_::now();
  BKState s{G, cb, st, t0, time_limit_s, false, {}, {}, {}, {}};
  s.Pst.emplace_back(G.size());
  s.Xst.emplace_back(G.size());
  s.tmp.emplace_back(G.size());
  s.Pst[0].fill();
  s.rec(0);
  st.seconds = secs_since(t0);
  return st;
}

// ---------------------------------------------------------------------------
// k-cliques (ordered DFS)
// ---------------------------------------------------------------------------
namespace {

struct KState {
  const PoolGraph& G;
  int k;
  const std::function<bool(const std::vector<int>&)>& cb;
  CliqueStats& st;
  clock_t_::time_point t0;
  double limit;
  bool stop = false;
  std::vector<int> R;
  std::vector<Bitset> Pst;

  void rec(int d) {
    if (stop) return;
    ++st.nodes;
    if ((st.nodes & 4095) == 0 && limit > 0 && secs_since(t0) > limit) {
      stop = true;
      st.complete = false;
      return;
    }
    const int need = k - static_cast<int>(R.size());
    Bitset& P = Pst[static_cast<std::size_t>(d)];   // Pst is preallocated (k + 1 levels): no reallocation
    if (need == 0) {
      ++st.count;
      if (cb && !cb(R)) {
        stop = true;
        st.complete = false;
      }
      return;
    }
    if (need == 1 && !cb) {
      st.count += P.count();
      return;
    }
    for (int v = P.first(); v >= 0; v = P.next(v + 1)) {
      const int c = Pst[static_cast<std::size_t>(d + 1)].intersect_above(P, G.adj[static_cast<std::size_t>(v)], v);
      if (c < need - 1) continue;
      R.push_back(v);
      rec(d + 1);
      R.pop_back();
      if (stop) return;
    }
  }
};

}  // namespace

CliqueStats k_cliques(const PoolGraph& G, int k, const std::function<bool(const std::vector<int>&)>& cb,
                      double time_limit_s, int through) {
  CliqueStats st;
  const auto t0 = clock_t_::now();
  if (k <= 0) return st;
  KState s{G, k, cb, st, t0, time_limit_s, false, {}, {}};
  s.Pst.assign(static_cast<std::size_t>(k + 1), Bitset(G.size()));
  if (through >= 0) {
    s.R.push_back(through);
    s.Pst[0] = G.adj[static_cast<std::size_t>(through)];
  } else {
    s.Pst[0].fill();
  }
  s.rec(0);
  st.by_size[k] = st.count;
  st.seconds = secs_since(t0);
  return st;
}

// ---------------------------------------------------------------------------
// Maximum clique (colouring branch and bound)
// ---------------------------------------------------------------------------
namespace {

struct MCState {
  const PoolGraph& G;
  MaxCliqueResult& res;
  clock_t_::time_point t0;
  double limit;
  int stop_at;
  bool stop = false;
  std::vector<int> R;

  // Greedy colouring of P (order given); returns vertices reordered by colour with colour numbers.
  void colour_sort(const std::vector<int>& P, std::vector<int>& order, std::vector<int>& col) {
    order.clear();
    col.clear();
    std::vector<std::vector<int>> classes;
    for (int v : P) {
      const Bitset& nv = G.adj[static_cast<std::size_t>(v)];
      std::size_t k = 0;
      for (; k < classes.size(); ++k) {
        bool ok = true;
        for (int u : classes[k])
          if (nv.test(u)) {
            ok = false;
            break;
          }
        if (ok) break;
      }
      if (k == classes.size()) classes.emplace_back();
      classes[k].push_back(v);
    }
    for (std::size_t k = 0; k < classes.size(); ++k)
      for (int v : classes[k]) {
        order.push_back(v);
        col.push_back(static_cast<int>(k) + 1);
      }
  }

  void expand(const std::vector<int>& P) {
    if (stop) return;
    ++res.nodes;
    if ((res.nodes & 1023) == 0 && limit > 0 && secs_since(t0) > limit) {
      stop = true;
      res.complete = false;
      return;
    }
    std::vector<int> order, col;
    colour_sort(P, order, col);
    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
      if (static_cast<int>(R.size()) + col[static_cast<std::size_t>(i)] <= static_cast<int>(res.clique.size())) return;
      const int v = order[static_cast<std::size_t>(i)];
      R.push_back(v);
      std::vector<int> P2;
      const Bitset& nv = G.adj[static_cast<std::size_t>(v)];
      for (int j = 0; j < i; ++j) {
        const int u = order[static_cast<std::size_t>(j)];
        if (nv.test(u)) P2.push_back(u);
      }
      if (P2.empty()) {
        if (R.size() > res.clique.size()) {
          res.clique = R;
          if (stop_at > 0 && static_cast<int>(R.size()) >= stop_at) {
            stop = true;
            res.complete = false;
          }
        }
      } else {
        expand(P2);
      }
      R.pop_back();
      if (stop) return;
    }
  }
};

}  // namespace

MaxCliqueResult max_clique(const PoolGraph& G, double time_limit_s, int stop_at_size) {
  MaxCliqueResult res;
  const auto t0 = clock_t_::now();
  const int n = G.size();
  if (n == 0) return res;
  // initial order: non-increasing degree (MCQ)
  std::vector<int> P(static_cast<std::size_t>(n));
  std::iota(P.begin(), P.end(), 0);
  std::vector<int> deg(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) deg[static_cast<std::size_t>(i)] = G.degree(i);
  std::sort(P.begin(), P.end(), [&](int a, int b) {
    return deg[static_cast<std::size_t>(a)] > deg[static_cast<std::size_t>(b)];
  });
  MCState s{G, res, t0, time_limit_s, stop_at_size, false, {}};
  s.expand(P);
  return res;
}

// ---------------------------------------------------------------------------
// Cross structure
// ---------------------------------------------------------------------------
CrossReport cross_structure(const Leech& L, const Classes& K, const std::vector<Vec>& S, bool count8) {
  const auto t0 = clock_t_::now();
  CrossReport r;
  r.classes = set_classes(L, K, S);
  r.antipodal = is_antipodal(S);
  const PoolGraph G = induced_orthogonality(L, K, r.classes);
  r.degree_hist = G.degree_histogram();
  CliqueStats st = maximal_cliques(G, [&](const std::vector<int>& R) {
    if (static_cast<int>(R.size()) == FRAME_SIZE) {
      std::vector<uint32_t> f;
      for (int i : R) f.push_back(G.pool[static_cast<std::size_t>(i)]);
      std::sort(f.begin(), f.end());
      r.frames.push_back(f);
    }
    return true;
  });
  r.maximal_by_size = st.by_size;
  r.maximal_total = st.count;
  r.clique_number = st.by_size.empty() ? 0 : st.by_size.rbegin()->first;
  std::sort(r.frames.begin(), r.frames.end());
  if (count8) r.cliques8 = k_cliques(G, 8).count;
  r.seconds = secs_since(t0);
  return r;
}

// ---------------------------------------------------------------------------
// Neighbourhood of a class, frame counting and estimation
// ---------------------------------------------------------------------------
Neighbourhood build_neighbourhood(const Leech& L, const Classes& K, uint32_t c, const Bitset* restrict_to) {
  Neighbourhood Nb;
  Nb.centre = c;
  Bitset o(NCLASS);
  orthogonal_classes(L, K, c, o);
  if (restrict_to) o.and_with(*restrict_to);
  o.for_each([&](int b) { Nb.verts.push_back(static_cast<uint32_t>(b)); });
  const int m = Nb.size();
  Nb.adj.assign(static_cast<std::size_t>(m), Bitset(m));
  std::vector<Vec> vs;
  vs.reserve(Nb.verts.size());
  for (uint32_t b : Nb.verts) vs.push_back(L.C[K.rep[b]]);
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < m; ++i) {
    uint64_t* w = Nb.adj[static_cast<std::size_t>(i)].words();
    const Vec& a = vs[static_cast<std::size_t>(i)];
    for (int j = 0; j < m; ++j)
      if (j != i && dot(a, vs[static_cast<std::size_t>(j)]) == 0) w[j >> 6] |= uint64_t{1} << (j & 63);
  }
  return Nb;
}

namespace {

// Per-thread DFS for count_cliques_through.
struct ThroughDFS {
  const Neighbourhood& Nb;
  int k;
  const std::function<bool(const std::vector<int>&)>* cb;
  std::vector<Bitset> P;
  std::vector<int> R;
  long frames = 0, nodes = 0;
  std::atomic<bool>* stop;
  clock_t_::time_point t0;
  double limit;

  ThroughDFS(const Neighbourhood& nb, int kk, const std::function<bool(const std::vector<int>&)>* c,
             std::atomic<bool>* s, clock_t_::time_point t, double lim)
      : Nb(nb), k(kk), cb(c), stop(s), t0(t), limit(lim) {
    P.assign(static_cast<std::size_t>(k + 1), Bitset(Nb.size()));
  }

  // R has d members (neighbourhood indices); need = k − 1 − d more from P[d].
  void rec(int d) {
    const int need = k - 1 - d;
    Bitset& Pd = P[static_cast<std::size_t>(d)];
    if (need == 0) {
      ++frames;
      if (cb && !(*cb)(R)) stop->store(true);
      return;
    }
    if (need == 1 && !cb) {
      frames += Pd.count();
      return;
    }
    Bitset& Pn = P[static_cast<std::size_t>(d + 1)];
    for (int v = Pd.first(); v >= 0; v = Pd.next(v + 1)) {
      ++nodes;
      if ((nodes & 8191) == 0) {
        if (stop->load(std::memory_order_relaxed)) return;
        if (limit > 0 && secs_since(t0) > limit) {
          stop->store(true);
          return;
        }
      }
      const int c = Pn.intersect_above(Pd, Nb.adj[static_cast<std::size_t>(v)], v);
      if (c < need - 1) continue;
      R.push_back(v);
      rec(d + 1);
      R.pop_back();
      if (stop->load(std::memory_order_relaxed)) return;
    }
  }
};

}  // namespace

FrameCount count_cliques_through(const Neighbourhood& Nb, int k, double time_limit_s,
                                 const std::function<bool(const std::vector<int>&)>& cb) {
  FrameCount fc;
  const auto t0 = clock_t_::now();
  const int m = Nb.size();
  fc.first_level_total = m;
  if (k < 2) return fc;
  std::atomic<bool> stop{false};
  std::atomic<long> done{0};
  long frames = 0, nodes = 0;
  const std::function<bool(const std::vector<int>&)>* cbp = cb ? &cb : nullptr;
#pragma omp parallel
  {
    ThroughDFS dfs(Nb, k, cbp, &stop, t0, time_limit_s);
    long my_frames = 0, my_nodes = 0;
#pragma omp for schedule(dynamic, 1)
    for (int v = 0; v < m; ++v) {
      if (stop.load(std::memory_order_relaxed)) continue;
      // Only first-level vertices; candidates > v adjacent to v.
      const int need = k - 1;
      dfs.frames = 0;
      dfs.nodes = 1;
      if (need == 1) {
        dfs.frames = 1;
      } else {
        Bitset all(m);
        all.fill();
        const int c = dfs.P[1].intersect_above(all, Nb.adj[static_cast<std::size_t>(v)], v);
        if (c >= need - 1) {
          dfs.R.assign(1, v);
          dfs.rec(1);
          dfs.R.clear();
        }
      }
      my_frames += dfs.frames;
      my_nodes += dfs.nodes;
      if (!stop.load(std::memory_order_relaxed)) done.fetch_add(1);
    }
#pragma omp critical
    {
      frames += my_frames;
      nodes += my_nodes;
    }
  }
  fc.frames = frames;
  fc.nodes = nodes;
  fc.complete = !stop.load();
  fc.first_level_done = done.load();
  fc.seconds = secs_since(t0);
  return fc;
}

TreeEstimate estimate_cliques_through(const Neighbourhood& Nb, int k, long probes, uint64_t seed) {
  // Unordered random descent: v_1 uniform in P_0 = all, P_1 = P_0 ∩ N(v_1), v_2
  // uniform in P_1, ... The number of ordered (k−1)-tuples of mutually adjacent
  // vertices is (k−1)!·#(k−1)-cliques and E[|P_0|·|P_1|···|P_{k−2}|] equals that
  // number (each tuple is reached with probability 1/(|P_0|···|P_{k−3}|), then
  // |P_{k−2}| counts the last coordinate exactly). Likewise, for every d, the
  // mean of |P_0|···|P_{d−1}|/d! estimates the number of d-cliques through the
  // centre (= nodes at depth d of the ordered DFS before pruning).
  TreeEstimate est;
  est.probes = probes;
  const int m = Nb.size();
  if (k < 2 || probes <= 0) return est;
  const int depth = k - 1;   // members to choose besides the centre
  std::vector<double> leaf_samples(static_cast<std::size_t>(probes), 0.0);
  std::vector<double> cliques_by_depth(static_cast<std::size_t>(depth + 1), 0.0);   // Σ over probes of Π|P_i| / d!
  std::vector<double> cand_by_depth(static_cast<std::size_t>(depth + 1), 0.0);
#pragma omp parallel
  {
    std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ull * static_cast<uint64_t>(omp_get_thread_num() + 1)));
    std::vector<Bitset> P(static_cast<std::size_t>(depth + 1), Bitset(m));
    std::vector<double> my_cl(static_cast<std::size_t>(depth + 1), 0.0), my_cand(static_cast<std::size_t>(depth + 1), 0.0);
#pragma omp for schedule(dynamic, 256)
    for (long p = 0; p < probes; ++p) {
      P[0].fill();
      double prod = 1.0, fact = 1.0;   // Π|P_i| over chosen levels, d!
      int cnt = m;
      double leaves = 0;
      my_cl[0] += 1.0;
      for (int d = 0; d < depth; ++d) {
        my_cand[static_cast<std::size_t>(d)] += cnt;
        // d-cliques counted so far: prod/fact with prod = |P_0|…|P_{d-1}|
        if (d == depth - 1) {
          leaves = prod * cnt;   // exact last coordinate
          break;
        }
        if (cnt == 0) break;
        std::uniform_int_distribution<int> U(0, cnt - 1);
        int r = U(rng);
        int v = P[static_cast<std::size_t>(d)].first();
        while (r-- > 0) v = P[static_cast<std::size_t>(d)].next(v + 1);
        prod *= cnt;
        fact *= (d + 1);
        my_cl[static_cast<std::size_t>(d + 1)] += prod / fact;
        Bitset& Pn = P[static_cast<std::size_t>(d + 1)];
        Pn = P[static_cast<std::size_t>(d)];
        Pn.and_with(Nb.adj[static_cast<std::size_t>(v)]);
        Pn.reset(v);
        cnt = Pn.count();
      }
      leaf_samples[static_cast<std::size_t>(p)] = leaves;
    }
#pragma omp critical
    for (int d = 0; d <= depth; ++d) {
      cliques_by_depth[static_cast<std::size_t>(d)] += my_cl[static_cast<std::size_t>(d)];
      cand_by_depth[static_cast<std::size_t>(d)] += my_cand[static_cast<std::size_t>(d)];
    }
  }
  double s = 0, s2 = 0;
  for (long p = 0; p < probes; ++p) {
    const double x = leaf_samples[static_cast<std::size_t>(p)];
    s += x;
    s2 += x * x;
  }
  const double n = static_cast<double>(probes);
  double kf = 1;
  for (int i = 2; i <= depth; ++i) kf *= i;
  est.leaves = s / n / kf;
  const double var = std::max(0.0, s2 / n - (s / n) * (s / n));
  est.leaves_se = std::sqrt(var / n) / kf;
  est.nodes = 0;
  for (int d = 0; d <= depth; ++d) {
    est.cliques_by_depth[d] = cliques_by_depth[static_cast<std::size_t>(d)] / n;
    est.nodes += est.cliques_by_depth[d];
    if (d < depth) est.mean_candidates_by_depth[d] = cand_by_depth[static_cast<std::size_t>(d)] / n;
  }
  return est;
}

Bitset classes_of_shape(const Leech& L, const Classes& K, int shape) {
  Bitset b(NCLASS);
  for (int c = 0; c < NCLASS; ++c)
    if (leech_shape(L.C[K.rep[static_cast<std::size_t>(c)]]) == shape) b.set(c);
  return b;
}

namespace {

// |P ∩ N(v)| by exact dots over the SET BITS of P only (not all NCLASS
// classes): the candidate sets shrink by ≈ 0.45 per level, so this is ~25×
// cheaper than building the full 98280-bit orthogonal set of v and masking.
// If `out` is given it receives P ∩ N(v) (must be a different bitset than P).
int intersect_orthogonal(const Leech& L, const Classes& K, const Bitset& P, uint32_t v, Bitset* out) {
  const Vec& vv = L.C[K.rep[v]];
  if (out) out->clear();
  int c = 0;
  P.for_each([&](int u) {
    if (dot(vv, L.C[K.rep[static_cast<std::size_t>(u)]]) == 0) {
      ++c;
      if (out) out->set(u);
    }
  });
  return c;
}

}  // namespace

std::vector<uint32_t> random_frame(const Leech& L, const Classes& K, uint32_t start, uint64_t seed,
                                   const Bitset* allowed, long node_limit) {
  // Randomised DFS with backtracking and restarts. A uniformly random next
  // class almost never completes (the candidate set shrinks by ≈ 0.45 per
  // level and dies around depth 14), so at every level up to SAMPLE random
  // untried candidates are examined and the one keeping the most candidates is
  // taken. Deep failures are cheap to recover from, shallow ones are not (at
  // depth 1 there are 21582 candidates and each trial costs |P| dots), so a
  // level is abandoned after TRY_CAP candidates and the whole descent is
  // restarted from `start` once the root gives up — restarts, not exhaustive
  // backtracking, are what makes this reliable.
  // The per-level stacks are preallocated to the full depth (FRAME_SIZE + 1),
  // so nothing reallocates while references into them are live.
  constexpr int SAMPLE = 16;
  constexpr int TRY_CAP = 6;
  std::mt19937_64 rng(seed);
  std::vector<Bitset> P(static_cast<std::size_t>(FRAME_SIZE + 1), Bitset(NCLASS));
  std::vector<Bitset> tried(static_cast<std::size_t>(FRAME_SIZE + 1), Bitset(NCLASS));
  std::vector<int> ntried(static_cast<std::size_t>(FRAME_SIZE + 1), 0);
  std::vector<uint32_t> R;
  Bitset rem(NCLASS);
  orthogonal_classes(L, K, start, P[1]);   // never modified below
  if (allowed) P[1].and_with(*allowed);
  long nodes = 0;
  while (nodes <= node_limit) {
    for (int i = 1; i <= FRAME_SIZE; ++i) {
      tried[static_cast<std::size_t>(i)].clear();
      ntried[static_cast<std::size_t>(i)] = 0;
    }
    R.assign(1, start);
    int d = 1;
    while (d >= 1) {
      if (static_cast<int>(R.size()) == FRAME_SIZE) return R;
      Bitset& Pd = P[static_cast<std::size_t>(d)];
      Bitset& Td = tried[static_cast<std::size_t>(d)];
      rem = Pd;
      rem.and_not(Td);
      const int cnt = rem.count();
      if (cnt == 0 || cnt < FRAME_SIZE - d || ntried[static_cast<std::size_t>(d)] >= TRY_CAP) {
        R.pop_back();
        --d;
        continue;
      }
      if (++nodes > node_limit) return {};
      const std::vector<int> rl = rem.to_list();
      int bestv = -1, bestc = -1;
      const int ns = std::min(SAMPLE, cnt);
      for (int t = 0; t < ns; ++t) {
        const int v = rl[static_cast<std::size_t>(rng() % static_cast<uint64_t>(cnt))];
        const int c = intersect_orthogonal(L, K, Pd, static_cast<uint32_t>(v), nullptr);
        if (c > bestc) {
          bestc = c;
          bestv = v;
        }
      }
      Td.set(bestv);
      ++ntried[static_cast<std::size_t>(d)];
      if (bestc < FRAME_SIZE - d - 1) continue;
      intersect_orthogonal(L, K, Pd, static_cast<uint32_t>(bestv), &P[static_cast<std::size_t>(d + 1)]);
      R.push_back(static_cast<uint32_t>(bestv));
      ++d;
      tried[static_cast<std::size_t>(d)].clear();
      ntried[static_cast<std::size_t>(d)] = 0;
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Extension search
// ---------------------------------------------------------------------------
namespace {

struct ExtState {
  const PoolGraph& G;          // pool graph: index 0 = seed; [1, nx] = X-members; [nx+1, n) = outsiders
  int nx;                       // number of X-members in the pool
  int xbits;                    // |X|
  const std::vector<Bitset>& conf;   // conf[i] over X indices (outsiders only; others empty)
  const std::vector<int>& confsize;  // |conf[i]|
  const ExtendOptions& opt;
  ExtendResult& res;
  clock_t_::time_point t0;
  bool stop = false;
  std::vector<int> R;           // outsider members (pool indices), R[0] = 0 (seed)
  std::vector<Bitset> P;        // outsider candidates > last, per depth
  std::vector<Bitset> XP;       // X-members orthogonal to all of R, per depth
  std::vector<Bitset> CU;       // conflict union per depth
  std::vector<int> hist;        // scratch: marginal-cost histogram

  bool frame_extension_possible(int d) const {
    return static_cast<int>(R.size()) + XP[static_cast<std::size_t>(d)].count() >= FRAME_SIZE;
  }

  // number of (24 − |R|)-cliques among XP[d] (frames through the clique)
  long count_frames(int d) {
    const int need = FRAME_SIZE - static_cast<int>(R.size());
    std::vector<int> xs = XP[static_cast<std::size_t>(d)].to_list();
    if (static_cast<int>(xs.size()) < need) return 0;
    if (need == 0) return 1;
    // induced subgraph on xs
    PoolGraph H;
    H.pool.resize(xs.size());
    H.adj.assign(xs.size(), Bitset(static_cast<int>(xs.size())));
    for (std::size_t i = 0; i < xs.size(); ++i) {
      H.pool[i] = static_cast<uint32_t>(xs[i]);
      const Bitset& row = G.adj[static_cast<std::size_t>(xs[i])];
      for (std::size_t j = 0; j < xs.size(); ++j)
        if (i != j && row.test(xs[j])) H.adj[i].set(static_cast<int>(j));
    }
    return k_cliques(H, need).count;
  }

  void record(int d) {
    const int size_out = static_cast<int>(R.size());
    const int removed = CU[static_cast<std::size_t>(d)].count();
    const int gain = size_out - removed;
    ++res.cliques;
    if (size_out >= opt.kmin || (size_out + XP[static_cast<std::size_t>(d)].count() >= opt.kmin)) {
      ++res.gain_hist[gain];
      auto it = res.best_gain_by_size.find(size_out);
      if (it == res.best_gain_by_size.end() || it->second < gain) res.best_gain_by_size[size_out] = gain;
      if (gain > res.best_gain || (gain == res.best_gain && size_out > res.best.added)) {
        res.best_gain = gain;
        res.best.seed = G.pool[0];
        res.best.clique.clear();
        for (int i : R) res.best.clique.push_back(G.pool[static_cast<std::size_t>(i)]);
        res.best.added = size_out;
        res.best.removed = removed;
      }
    }
    if (frame_extension_possible(d)) {
      const long f = count_frames(d);
      if (f > 0) {
        res.frames += f;
        if (gain > res.best_frame_gain) res.best_frame_gain = gain;
      }
    }
  }

  // Upper bound on the gain of any extension of the current node.
  int bound(int d) {
    const Bitset& Pd = P[static_cast<std::size_t>(d)];
    const Bitset& cu = CU[static_cast<std::size_t>(d)];
    const int new_now = static_cast<int>(R.size());
    const int conf_now = cu.count();
    const int rmax = std::min(Pd.count(), FRAME_SIZE - new_now);
    if (rmax <= 0) return new_now - conf_now;
    std::fill(hist.begin(), hist.end(), 0);
    Pd.for_each([&](int u) {
      // marginal cost |conf[u] \ cu|
      const uint64_t* a = conf[static_cast<std::size_t>(u)].words();
      const uint64_t* b = cu.words();
      int mc = 0;
      for (int w = 0; w < cu.nwords(); ++w) mc += kiss::popcount64(a[w] & ~b[w]);
      ++hist[static_cast<std::size_t>(mc)];
    });
    int best = new_now - conf_now;
    int r = 0;
    for (std::size_t mc = 0; mc < hist.size() && r < rmax; ++mc) {
      if (hist[mc] == 0) continue;
      // taking r' ∈ (r, r + hist[mc]] candidates: the r'-th smallest cost is mc; best at r' = min(r + hist[mc], rmax)
      const int r2 = std::min(r + hist[mc], rmax);
      best = std::max(best, new_now + r2 - conf_now - static_cast<int>(mc));
      r = r2;
    }
    return best;
  }

  void rec(int d) {
    if (stop) return;
    ++res.nodes;
    if ((res.nodes & 1023) == 0) {
      if (opt.time_limit_s > 0 && secs_since(t0) > opt.time_limit_s) stop = true;
      if (opt.node_limit > 0 && res.nodes > opt.node_limit) stop = true;
      if (stop) {
        res.complete = false;
        return;
      }
    }
    record(d);
    if (static_cast<int>(R.size()) >= FRAME_SIZE) return;
    if (bound(d) < opt.min_gain) return;
    Bitset& Pd = P[static_cast<std::size_t>(d)];   // P/XP/CU preallocated (FRAME_SIZE + 2 levels)
    for (int v = Pd.first(); v >= 0; v = Pd.next(v + 1)) {
      const Bitset& nv = G.adj[static_cast<std::size_t>(v)];
      P[static_cast<std::size_t>(d + 1)].intersect_above(Pd, nv, v);
      XP[static_cast<std::size_t>(d + 1)] = XP[static_cast<std::size_t>(d)];
      XP[static_cast<std::size_t>(d + 1)].and_with(nv);
      CU[static_cast<std::size_t>(d + 1)] = CU[static_cast<std::size_t>(d)];
      CU[static_cast<std::size_t>(d + 1)].or_with(conf[static_cast<std::size_t>(v)]);
      R.push_back(v);
      rec(d + 1);
      R.pop_back();
      if (stop) return;
    }
  }
};

}  // namespace

ExtendResult extend_from_class(const Leech& L, const Classes& K, const std::vector<uint32_t>& X,
                               const std::vector<uint16_t>& conf, uint32_t c, const ExtendOptions& opt_in) {
  ExtendOptions opt = opt_in;
  if (opt.frames_only) opt.kmin = FRAME_SIZE;
  ExtendResult res;
  res.seed = c;
  const auto t0 = clock_t_::now();
  // membership of X
  Bitset inX(NCLASS);
  for (uint32_t x : X) inX.set(static_cast<int>(x));
  if (inX.test(static_cast<int>(c))) throw std::invalid_argument("extend_from_class: seed is in X");
  const Vec& vc = L.C[K.rep[c]];
  std::vector<uint32_t> pool{c};
  for (uint32_t x : X)
    if (dot(vc, L.C[K.rep[x]]) == 0) pool.push_back(x);
  const int nx = static_cast<int>(pool.size()) - 1;
  for (int b = 0; b < NCLASS; ++b) {
    if (static_cast<uint32_t>(b) == c || inX.test(b)) continue;
    if (conf[static_cast<std::size_t>(b)] > opt.tmax) continue;
    if (dot(vc, L.C[K.rep[static_cast<std::size_t>(b)]]) == 0) pool.push_back(static_cast<uint32_t>(b));
  }
  const PoolGraph G = induced_orthogonality(L, K, pool);
  const int n = G.size();
  res.pool_size = n;
  res.pool_outside = n - nx - 1;
  const int xbits = static_cast<int>(X.size());
  // conflict sets of outsiders (and the seed) over X indices
  std::vector<Bitset> cset(static_cast<std::size_t>(n), Bitset(xbits));
  std::vector<int> csize(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    if (i >= 1 && i <= nx) continue;
    const Vec& v = L.C[K.rep[pool[static_cast<std::size_t>(i)]]];
    for (int j = 0; j < xbits; ++j) {
      const int d = dot(v, L.C[K.rep[X[static_cast<std::size_t>(j)]]]);
      if (d == 16 || d == -16) cset[static_cast<std::size_t>(i)].set(j);
    }
    csize[static_cast<std::size_t>(i)] = cset[static_cast<std::size_t>(i)].count();
  }
  ExtState s{G, nx, xbits, cset, csize, opt, res, t0, false, {}, {}, {}, {}, {}};
  s.hist.assign(static_cast<std::size_t>(xbits + 1), 0);
  s.P.assign(static_cast<std::size_t>(FRAME_SIZE + 2), Bitset(n));
  s.XP.assign(static_cast<std::size_t>(FRAME_SIZE + 2), Bitset(n));
  s.CU.assign(static_cast<std::size_t>(FRAME_SIZE + 2), Bitset(xbits));
  s.R.push_back(0);
  // candidates: outsiders adjacent to the seed (all outsiders are, by construction) with index > nx
  for (int i = nx + 1; i < n; ++i)
    if (G.adjacent(0, i)) s.P[0].set(i);
  for (int i = 1; i <= nx; ++i) s.XP[0].set(i);
  s.CU[0] = cset[0];
  s.rec(0);
  res.seconds = secs_since(t0);
  return res;
}

std::vector<uint32_t> apply_extend_hit(const Leech& L, const Classes& K, const std::vector<uint32_t>& X,
                                       const ExtendHit& hit) {
  std::vector<uint32_t> out;
  for (uint32_t x : X) {
    bool keep = true;
    for (uint32_t q : hit.clique)
      if (q == x || class_conflict(L, K, q, x)) {
        keep = false;
        break;
      }
    if (keep) out.push_back(x);
  }
  for (uint32_t q : hit.clique) out.push_back(q);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  if (!classes_independent(L, K, out)) throw std::runtime_error("apply_extend_hit: result is not independent");
  return out;
}

// ---------------------------------------------------------------------------
// Greedy frame unions
// ---------------------------------------------------------------------------
namespace {

struct ConflictCache {
  const Leech& L;
  const Classes& K;
  std::unordered_map<uint32_t, std::vector<uint32_t>> m;
  const std::vector<uint32_t>& get(uint32_t c) {
    auto it = m.find(c);
    if (it != m.end()) return it->second;
    if (m.size() > 30000) m.clear();
    return m.emplace(c, conflicting_classes(L, K, c)).first->second;
  }
};

struct ClassState {
  std::vector<uint16_t> conf;   // conflicts into U
  Bitset inU;
  std::vector<uint32_t> U;
  ClassState() : conf(static_cast<std::size_t>(NCLASS), 0), inU(NCLASS) {}
  void add(ConflictCache& cc, uint32_t c) {
    inU.set(static_cast<int>(c));
    U.push_back(c);
    for (uint32_t w : cc.get(c)) ++conf[w];
  }
  void remove(ConflictCache& cc, uint32_t c) {
    inU.reset(static_cast<int>(c));
    U.erase(std::find(U.begin(), U.end(), c));
    for (uint32_t w : cc.get(c)) --conf[w];
  }
  std::vector<uint32_t> free_classes() const {
    std::vector<uint32_t> f;
    for (int c = 0; c < NCLASS; ++c)
      if (conf[static_cast<std::size_t>(c)] == 0 && !inU.test(c)) f.push_back(static_cast<uint32_t>(c));
    return f;
  }
};

}  // namespace

GreedyResult greedy_frame_union(const Leech& L, const Classes& K, uint64_t seed, const GreedyOptions& opt) {
  const auto t0 = clock_t_::now();
  GreedyResult r;
  std::mt19937_64 rng(seed);
  ConflictCache cc{L, K, {}};
  ClassState st;
  bool last_was_frame = false;
  if (opt.frames_first) {
    const uint32_t s = static_cast<uint32_t>(rng() % static_cast<uint64_t>(NCLASS));
    std::vector<uint32_t> F = random_frame(L, K, s, rng());
    if (static_cast<int>(F.size()) != FRAME_SIZE) throw std::runtime_error("greedy_frame_union: no random frame found");
    for (uint32_t c : F) st.add(cc, c);
    r.clique_sizes.push_back(FRAME_SIZE);
    last_was_frame = true;
    r.free_after_frames = static_cast<int>(st.free_classes().size());
  }
  while (true) {
    std::vector<uint32_t> fr = st.free_classes();
    if (fr.empty()) break;
    std::vector<uint32_t> Q;
    if (static_cast<int>(fr.size()) > opt.large_pool) {
      Bitset allowed(NCLASS);
      for (uint32_t c : fr) allowed.set(static_cast<int>(c));
      for (int attempt = 0; attempt < 4 && Q.empty(); ++attempt) {
        const uint32_t s = fr[rng() % fr.size()];
        Q = random_frame(L, K, s, rng(), &allowed, 100000);
      }
    }
    if (Q.empty()) {
      const PoolGraph G = induced_orthogonality(L, K, fr);
      const MaxCliqueResult mc = max_clique(G, opt.max_clique_time_s, FRAME_SIZE);
      for (int i : mc.clique) Q.push_back(G.pool[static_cast<std::size_t>(i)]);
    }
    if (Q.empty()) break;
    if (static_cast<int>(Q.size()) < opt.min_clique) break;
    for (uint32_t c : Q) st.add(cc, c);
    r.clique_sizes.push_back(static_cast<int>(Q.size()));
    if (static_cast<int>(Q.size()) == FRAME_SIZE) {
      last_was_frame = true;
      r.free_after_frames = static_cast<int>(st.free_classes().size());
    } else if (last_was_frame) {
      last_was_frame = false;
    }
  }
  r.classes = st.U;
  std::sort(r.classes.begin(), r.classes.end());
  r.seconds = secs_since(t0);
  return r;
}

// ---------------------------------------------------------------------------
// Clique-move local search
// ---------------------------------------------------------------------------
LocalSearchResult clique_local_search(const Leech& L, const Classes& K, const std::vector<uint32_t>& X0,
                                      const LocalSearchOptions& opt,
                                      const std::function<void(const std::string&)>& log) {
  const auto t0 = clock_t_::now();
  LocalSearchResult res;
  std::mt19937_64 rng(opt.rng_seed);
  ConflictCache cc{L, K, {}};
  ClassState st;
  for (uint32_t c : X0) st.add(cc, c);
  res.start_size = static_cast<int>(st.U.size());
  res.best = st.U;
  std::vector<long> tabu_until(static_cast<std::size_t>(NCLASS), -1);
  long move = 0;
  auto drain = [&]() {
    // add free classes greedily (random order)
    std::vector<uint32_t> fr = st.free_classes();
    std::shuffle(fr.begin(), fr.end(), rng);
    for (uint32_t c : fr)
      if (st.conf[c] == 0 && !st.inU.test(static_cast<int>(c))) st.add(cc, c);
  };
  drain();
  while (secs_since(t0) < opt.time_limit_s) {
    if (static_cast<int>(st.U.size()) >= opt.target) break;
    // seed candidates: classes ∉ U, conf ≤ seed_tmax, not tabu; prefer the minimum conflict level
    std::vector<uint32_t> cand;
    int minconf = 1 << 20;
    for (int c = 0; c < NCLASS; ++c) {
      if (st.inU.test(c)) continue;
      const int cf = st.conf[static_cast<std::size_t>(c)];
      if (cf > opt.seed_tmax || tabu_until[static_cast<std::size_t>(c)] > move) continue;
      if (cf < minconf) {
        minconf = cf;
        cand.clear();
      }
      if (cf == minconf) cand.push_back(static_cast<uint32_t>(c));
    }
    if (cand.empty()) {
      // nothing admissible; relax the tabu
      for (auto& t : tabu_until) t = -1;
      ++move;
      continue;
    }
    const uint32_t c = cand[rng() % cand.size()];
    ExtendOptions eo;
    eo.tmax = opt.tmax;
    eo.kmin = 1;
    eo.min_gain = -opt.max_drop;
    eo.time_limit_s = opt.move_time_s;
    eo.node_limit = opt.node_limit;
    // the search must not re-add tabu classes: mask them out via a conf trick (conf > tmax)
    std::vector<uint16_t> conf_masked = st.conf;
    for (int b = 0; b < NCLASS; ++b)
      if (tabu_until[static_cast<std::size_t>(b)] > move) conf_masked[static_cast<std::size_t>(b)] = 65535;
    std::vector<uint32_t> U_sorted = st.U;
    std::sort(U_sorted.begin(), U_sorted.end());
    const ExtendResult er = extend_from_class(L, K, U_sorted, conf_masked, c, eo);
    ++move;
    ++res.moves;
    if (er.best_gain < -opt.max_drop || er.best.clique.empty()) {
      ++res.rejected;
      tabu_until[c] = move + opt.tabu_tenure;
      continue;
    }
    // apply: remove conflicts of the clique, add the clique
    std::vector<uint32_t> removed;
    for (uint32_t x : U_sorted)
      for (uint32_t q : er.best.clique)
        if (class_conflict(L, K, q, x)) {
          removed.push_back(x);
          break;
        }
    for (uint32_t x : removed) {
      st.remove(cc, x);
      tabu_until[x] = move + opt.tabu_tenure;
    }
    for (uint32_t q : er.best.clique)
      if (!st.inU.test(static_cast<int>(q))) st.add(cc, q);
    drain();
    const int g = er.best_gain;
    if (g > 0) ++res.improving;
    else if (g == 0) ++res.plateau;
    else ++res.worsening;
    ++res.size_visits[static_cast<int>(st.U.size())];
    if (st.U.size() > res.best.size()) {
      res.best = st.U;
      std::sort(res.best.begin(), res.best.end());
      if (log)
        log("move " + std::to_string(move) + ": new best |X| = " + std::to_string(res.best.size()) +
            " (gain " + std::to_string(g) + ", removed " + std::to_string(removed.size()) + ", added " +
            std::to_string(er.best.added) + ", t = " + std::to_string(secs_since(t0)) + " s)");
    }
  }
  if (!classes_independent(L, K, res.best)) throw std::runtime_error("clique_local_search: best set not independent");
  res.seconds = secs_since(t0);
  return res;
}

long double double_factorial_odd(int twom) {
  long double r = 1;
  for (int k = twom - 1; k > 1; k -= 2) r *= k;
  return r;
}

}  // namespace kiss

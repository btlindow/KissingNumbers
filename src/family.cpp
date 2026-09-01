// T4.2 — disjoint families: masks, image pools, disjointness graph, clique
// search, the sweep, family directories. See include/kiss/family.h.
#include "kiss/family.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include "kiss/io.h"
#include "kiss/verify.h"

namespace kiss {

namespace {

using clock_t_ = std::chrono::steady_clock;

double seconds_since(clock_t_::time_point t0) {
  return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

}  // namespace

// ---------------------------------------------------------------------------
// masks
// ---------------------------------------------------------------------------
Mask make_mask(const std::vector<uint32_t>& idx) {
  Mask m(static_cast<std::size_t>(MASK_WORDS), 0);
  for (uint32_t v : idx) {
    if (v >= static_cast<uint32_t>(N)) throw std::out_of_range("make_mask: index >= N");
    m[v >> 6] |= uint64_t{1} << (v & 63);
  }
  return m;
}

bool mask_disjoint(const Mask& a, const Mask& b) {
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t w = 0; w < n; ++w)
    if (a[w] & b[w]) return false;
  return true;
}

int mask_overlap(const Mask& a, const Mask& b) {
  const std::size_t n = std::min(a.size(), b.size());
  int c = 0;
  for (std::size_t w = 0; w < n; ++w) c += popcount64(a[w] & b[w]);
  return c;
}

int overlap_sorted(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  std::size_t i = 0, j = 0;
  int c = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) ++i;
    else if (b[j] < a[i]) ++j;
    else { ++c; ++i; ++j; }
  }
  return c;
}

// ---------------------------------------------------------------------------
// pool
// ---------------------------------------------------------------------------
ImagePool random_image_pool(const std::vector<uint32_t>& S_idx, const std::vector<IndexPerm>& gens,
                            std::size_t count, uint64_t seed, int slots, int burnin) {
  ImagePool pool;
  pool.sets.reserve(count);
  ProductReplacement pr(gens, slots, seed);
  pr.burn_in(burnin);
  for (std::size_t k = 0; k < count; ++k) {
    const IndexPerm& g = pr.next();
    std::vector<uint32_t> img = apply(g, S_idx);
    std::sort(img.begin(), img.end());
    pool.sets.push_back(std::move(img));
  }
  return pool;
}

// ---------------------------------------------------------------------------
// BitGraph
// ---------------------------------------------------------------------------
BitGraph::BitGraph(int n_) : n(n_), words((n_ + 63) / 64), bits(static_cast<std::size_t>(n_) * static_cast<std::size_t>((n_ + 63) / 64), 0) {}

void BitGraph::set_edge(int i, int j) {
  if (i == j) return;
  row(i)[j >> 6] |= uint64_t{1} << (j & 63);
  row(j)[i >> 6] |= uint64_t{1} << (i & 63);
}

void BitGraph::clear_edge(int i, int j) {
  row(i)[j >> 6] &= ~(uint64_t{1} << (j & 63));
  row(j)[i >> 6] &= ~(uint64_t{1} << (i & 63));
}

int BitGraph::degree(int i) const {
  int d = 0;
  const uint64_t* r = row(i);
  for (int w = 0; w < words; ++w) d += popcount64(r[w]);
  return d;
}

long long BitGraph::edges() const {
  long long e = 0;
  for (int i = 0; i < n; ++i) e += degree(i);
  return e / 2;
}

double BitGraph::density() const {
  if (n < 2) return 0.0;
  return static_cast<double>(edges()) / (static_cast<double>(n) * (n - 1) / 2.0);
}

BitGraph disjointness_graph(const ImagePool& pool) {
  const int P = static_cast<int>(pool.size());
  BitGraph G(P);
  // complete graph minus diagonal, then clear conflicting pairs
  for (int i = 0; i < P; ++i) {
    uint64_t* r = G.row(i);
    for (int w = 0; w < G.words; ++w) r[w] = ~uint64_t{0};
    // clear bits >= P in the last word
    const int extra = G.words * 64 - P;
    if (extra > 0) r[G.words - 1] &= (~uint64_t{0}) >> extra;
    r[i >> 6] &= ~(uint64_t{1} << (i & 63));
  }
  // inverted index vertex -> images (CSR)
  std::vector<uint32_t> cnt(static_cast<std::size_t>(N) + 1, 0);
  for (const auto& s : pool.sets)
    for (uint32_t v : s) {
      if (v >= static_cast<uint32_t>(N)) throw std::out_of_range("disjointness_graph: index >= N");
      ++cnt[v + 1];
    }
  for (std::size_t v = 0; v < static_cast<std::size_t>(N); ++v) cnt[v + 1] += cnt[v];
  std::vector<uint32_t> lists(cnt[static_cast<std::size_t>(N)]);
  {
    std::vector<uint32_t> pos(cnt.begin(), cnt.end() - 1);
    for (int i = 0; i < P; ++i)
      for (uint32_t v : pool.sets[static_cast<std::size_t>(i)]) lists[pos[v]++] = static_cast<uint32_t>(i);
  }
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < P; ++i) {
    uint64_t* r = G.row(i);
    for (uint32_t v : pool.sets[static_cast<std::size_t>(i)])
      for (uint32_t p = cnt[v]; p < cnt[v + 1]; ++p) {
        const uint32_t j = lists[p];
        r[j >> 6] &= ~(uint64_t{1} << (j & 63));
      }
  }
  return G;
}

BitGraph disjointness_graph_masks(const ImagePool& pool) {
  const int P = static_cast<int>(pool.size());
  std::vector<Mask> masks;
  masks.reserve(pool.size());
  for (const auto& s : pool.sets) masks.push_back(make_mask(s));
  BitGraph G(P);
#pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < P; ++i)
    for (int j = i + 1; j < P; ++j)
      if (mask_disjoint(masks[static_cast<std::size_t>(i)], masks[static_cast<std::size_t>(j)])) {
        // set both directions without racing: row i is owned by this thread,
        // row j by another — use atomic OR on row j.
        G.row(i)[j >> 6] |= uint64_t{1} << (j & 63);
        uint64_t* wj = G.row(j) + (i >> 6);
        __atomic_fetch_or(wj, uint64_t{1} << (i & 63), __ATOMIC_RELAXED);
      }
  return G;
}

// ---------------------------------------------------------------------------
// clique search
// ---------------------------------------------------------------------------
bool is_clique(const BitGraph& G, const std::vector<int>& c) {
  for (std::size_t a = 0; a < c.size(); ++a) {
    if (c[a] < 0 || c[a] >= G.n) return false;
    for (std::size_t b = a + 1; b < c.size(); ++b)
      if (c[a] == c[b] || !G.adj(c[a], c[b])) return false;
  }
  return true;
}

namespace {

int popcount_and(const uint64_t* a, const uint64_t* b, int words) {
  int c = 0;
  for (int w = 0; w < words; ++w) c += popcount64(a[w] & b[w]);
  return c;
}

std::vector<int> bits_to_list(const std::vector<uint64_t>& cand) {
  std::vector<int> out;
  for (std::size_t w = 0; w < cand.size(); ++w) {
    uint64_t x = cand[w];
    while (x) {
      const int b = __builtin_ctzll(x);
      out.push_back(static_cast<int>(w * 64) + b);
      x &= x - 1;
    }
  }
  return out;
}

}  // namespace

std::vector<int> greedy_clique(const BitGraph& G, int k, uint64_t seed, int restarts) {
  std::mt19937_64 rng(seed);
  std::vector<int> best;
  if (G.n == 0) return best;
  std::vector<uint64_t> cand(static_cast<std::size_t>(G.words));
  for (int r = 0; r < restarts; ++r) {
    std::vector<int> clique;
    int v = static_cast<int>(rng() % static_cast<uint64_t>(G.n));
    clique.push_back(v);
    std::memcpy(cand.data(), G.row(v), sizeof(uint64_t) * static_cast<std::size_t>(G.words));
    for (;;) {
      const std::vector<int> cl = bits_to_list(cand);
      if (cl.empty()) break;
      // vertex with most neighbours inside cand; ties broken randomly
      int bestv = -1, bestd = -1, ties = 0;
      for (int u : cl) {
        const int d = popcount_and(cand.data(), G.row(u), G.words);
        if (d > bestd) { bestd = d; bestv = u; ties = 1; }
        else if (d == bestd && static_cast<int>(rng() % static_cast<uint64_t>(++ties)) == 0) bestv = u;
      }
      clique.push_back(bestv);
      const uint64_t* rb = G.row(bestv);
      for (int w = 0; w < G.words; ++w) cand[static_cast<std::size_t>(w)] &= rb[w];
    }
    if (clique.size() > best.size()) best = clique;
    if (static_cast<int>(best.size()) >= k) break;
  }
  return best;
}

std::vector<int> local_search_clique(const BitGraph& G, int k, uint64_t seed, long long max_steps,
                                     double time_limit_s, const std::vector<int>* start) {
  // Dynamic local search in the style of DLS-MC (Pullan & Hoos 2006): add a
  // non-tabu vertex adjacent to the whole clique if one exists (least penalty),
  // else a (1,1)-swap (least penalty, tabu on the removed vertex), else a
  // perturbation (drop a vertex); penalties of clique vertices grow at every
  // stuck point and decay periodically.
  const auto t0 = clock_t_::now();
  std::mt19937_64 rng(seed);
  const int n = G.n;
  std::vector<int> best;
  if (n == 0) return best;
  std::vector<int> miss(static_cast<std::size_t>(n), 0);       // #clique vertices not adjacent to v
  std::vector<char> in(static_cast<std::size_t>(n), 0);
  std::vector<long long> tabu(static_cast<std::size_t>(n), -1);
  std::vector<int> penalty(static_cast<std::size_t>(n), 0);
  std::vector<int> clique;

  auto add = [&](int v) {
    in[static_cast<std::size_t>(v)] = 1;
    clique.push_back(v);
    const uint64_t* r = G.row(v);
    for (int u = 0; u < n; ++u)
      if (!((r[u >> 6] >> (u & 63)) & 1u)) ++miss[static_cast<std::size_t>(u)];
    --miss[static_cast<std::size_t>(v)];  // v is "not adjacent" to itself; keep miss[v] = #others
  };
  auto remove = [&](int v) {
    in[static_cast<std::size_t>(v)] = 0;
    clique.erase(std::find(clique.begin(), clique.end(), v));
    const uint64_t* r = G.row(v);
    for (int u = 0; u < n; ++u)
      if (!((r[u >> 6] >> (u & 63)) & 1u)) --miss[static_cast<std::size_t>(u)];
    ++miss[static_cast<std::size_t>(v)];
  };
  auto pick_min_penalty = [&](const std::vector<int>& cands) -> int {
    int bestv = -1, bestp = 0, ties = 0;
    for (int v : cands) {
      const int pv = penalty[static_cast<std::size_t>(v)];
      if (bestv < 0 || pv < bestp) { bestv = v; bestp = pv; ties = 1; }
      else if (pv == bestp && static_cast<int>(rng() % static_cast<uint64_t>(++ties)) == 0) bestv = v;
    }
    return bestv;
  };

  if (start && !start->empty() && is_clique(G, *start)) {
    for (int v : *start) add(v);
  } else {
    add(static_cast<int>(rng() % static_cast<uint64_t>(n)));
  }
  best = clique;
  std::vector<int> adds, swaps;
  const long long penalty_delay = std::max<long long>(2, n / 200);
  long long since_improvement = 0, stuck_events = 0;
  for (long long step = 0; step < max_steps; ++step) {
    if ((step & 1023) == 0 && seconds_since(t0) > time_limit_s) break;
    adds.clear();
    swaps.clear();
    for (int v = 0; v < n; ++v) {
      if (in[static_cast<std::size_t>(v)] || tabu[static_cast<std::size_t>(v)] >= step) continue;
      const int m = miss[static_cast<std::size_t>(v)];
      if (m == 0) adds.push_back(v);
      else if (m == 1) swaps.push_back(v);
    }
    if (!adds.empty()) {
      add(pick_min_penalty(adds));
      if (clique.size() > best.size()) { best = clique; since_improvement = 0; }
      if (static_cast<int>(best.size()) >= k) break;
      continue;
    }
    ++since_improvement;
    if (!swaps.empty()) {
      const int v = pick_min_penalty(swaps);
      const uint64_t* r = G.row(v);
      int u = -1;
      for (int c : clique)
        if (!((r[c >> 6] >> (c & 63)) & 1u)) { u = c; break; }
      remove(u);
      tabu[static_cast<std::size_t>(u)] = step + 7 + static_cast<long long>(rng() % 10);
      add(v);
    } else {
      // stuck: penalise the current clique, perturb
      ++stuck_events;
      for (int c : clique) ++penalty[static_cast<std::size_t>(c)];
      if (stuck_events % penalty_delay == 0)
        for (int& p : penalty) p = std::max(0, p - 1);
      if (clique.size() > 1) {
        const int u = clique[rng() % clique.size()];
        remove(u);
        tabu[static_cast<std::size_t>(u)] = step + 7 + static_cast<long long>(rng() % 10);
      } else {
        while (!clique.empty()) remove(clique.back());
        add(static_cast<int>(rng() % static_cast<uint64_t>(n)));
      }
    }
    if (since_improvement > 20LL * n) {
      // restart from a random vertex, keep the penalties
      while (!clique.empty()) remove(clique.back());
      add(static_cast<int>(rng() % static_cast<uint64_t>(n)));
      since_improvement = 0;
    }
  }
  return best;
}

namespace {

struct ExactState {
  const BitGraph* G;
  int k;
  long long node_limit;
  long long nodes = 0;
  bool limit_hit = false;
  std::vector<int> clique;
  std::vector<int> order;   // vertices by decreasing degree

  bool rec(std::vector<uint64_t>& cand, int cand_size) {
    ++nodes;
    if (static_cast<int>(clique.size()) == k) return true;
    if (nodes > node_limit) { limit_hit = true; return false; }
    if (static_cast<int>(clique.size()) + cand_size < k) return false;
    std::vector<int> cl = bits_to_list(cand);
    std::vector<uint64_t> next(cand.size());
    for (std::size_t t = 0; t < cl.size(); ++t) {
      if (static_cast<int>(clique.size()) + static_cast<int>(cl.size() - t) < k) return false;
      const int v = cl[t];
      // remove v from cand for the remaining iterations
      cand[static_cast<std::size_t>(v >> 6)] &= ~(uint64_t{1} << (v & 63));
      const uint64_t* r = G->row(v);
      int ns = 0;
      for (std::size_t w = 0; w < cand.size(); ++w) { next[w] = cand[w] & r[w]; ns += popcount64(next[w]); }
      clique.push_back(v);
      if (rec(next, ns)) return true;
      clique.pop_back();
      if (limit_hit) return false;
    }
    return false;
  }
};

}  // namespace

std::vector<int> exact_clique(const BitGraph& G, int k, long long node_limit, long long* nodes) {
  ExactState st;
  st.G = &G;
  st.k = k;
  st.node_limit = node_limit;
  std::vector<int> result;
  if (k <= 0) return result;
  if (G.n == 0 || k > G.n) { if (nodes) *nodes = 1; return result; }
  std::vector<uint64_t> cand(static_cast<std::size_t>(G.words), 0);
  for (int v = 0; v < G.n; ++v) cand[static_cast<std::size_t>(v >> 6)] |= uint64_t{1} << (v & 63);
  const bool found = st.rec(cand, G.n);
  if (nodes) *nodes = st.nodes;
  if (found) result = st.clique;
  return result;
}

double log10_binomial(double n, int k) {
  if (k < 0 || static_cast<double>(k) > n) return -INFINITY;
  return (std::lgamma(n + 1.0) - std::lgamma(static_cast<double>(k) + 1.0) - std::lgamma(n - k + 1.0)) / std::log(10.0);
}

double log10_expected_cliques(double n, int k, double p) {
  if (p <= 0) return k >= 2 ? -INFINITY : log10_binomial(n, k);
  return log10_binomial(n, k) + (static_cast<double>(k) * (k - 1) / 2.0) * std::log10(p);
}

double sweep_success_probability(int k, int pairs) {
  const double np = N / 2.0;
  const double f = 1.0 - static_cast<double>(pairs) * k / np;
  if (f <= 0) return 0.0;
  return std::pow(f, pairs);
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------
SweepResult sweep_family(const Leech& L, const std::vector<uint32_t>& S_idx, const std::vector<IndexPerm>& gens,
                         const SweepOptions& opt) {
  const auto t0 = clock_t_::now();
  SweepResult res;
  if (opt.elements < 1 || opt.images < 1) throw std::invalid_argument("sweep_family: elements/images must be >= 1");
  // antipodal pair ids
  std::vector<uint32_t> pair_id(static_cast<std::size_t>(N));
  std::vector<uint32_t> rep;
  rep.reserve(static_cast<std::size_t>(N / 2));
  for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v)
    if (v < L.neg[v]) { pair_id[v] = static_cast<uint32_t>(rep.size()); rep.push_back(v); }
  for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v)
    if (v > L.neg[v]) pair_id[v] = pair_id[L.neg[v]];
  const uint32_t NP = static_cast<uint32_t>(rep.size());
  const std::size_t PW = (NP + 63) / 64;
  // S as pair reps (S must be antipodal-closed for the pair-level test to be exact)
  std::vector<uint32_t> S_pairs;
  {
    std::vector<char> seen(NP, 0);
    for (uint32_t v : S_idx) {
      if (v >= static_cast<uint32_t>(N)) throw std::out_of_range("sweep_family: index >= N");
      const uint32_t p = pair_id[v];
      if (!seen[p]) { seen[p] = 1; S_pairs.push_back(p); }
    }
    for (uint32_t v : S_idx)
      if (!std::binary_search(S_idx.begin(), S_idx.end(), L.neg[v]))
        throw std::invalid_argument("sweep_family: S must be antipodal-closed and sorted");
  }
  const int np = static_cast<int>(S_pairs.size());
  // random elements and images
  ProductReplacement pr(gens, opt.slots, opt.seed);
  pr.burn_in(opt.burnin);
  const int E = opt.elements, I = opt.images;
  std::vector<IndexPerm> fwd(static_cast<std::size_t>(E)), inv(static_cast<std::size_t>(E));
  for (int e = 0; e < E; ++e) {
    fwd[static_cast<std::size_t>(e)] = pr.next();
    inv[static_cast<std::size_t>(e)] = inverse(fwd[static_cast<std::size_t>(e)]);
  }
  // images as pair ids; the first HEAD ids of every image are kept in a
  // separate compact array (I·64 bytes) so that the early-exit rejection scan
  // is cache-resident; the full lists are only read by the ~(1-f)^HEAD survivors.
  constexpr int HEAD = 16;
  const int head_n = std::min(HEAD, np);
  std::vector<uint32_t> images(static_cast<std::size_t>(I) * static_cast<std::size_t>(np));
  std::vector<uint32_t> heads(static_cast<std::size_t>(I) * HEAD, 0);
  for (int i = 0; i < I; ++i) {
    const IndexPerm& g = pr.next();
    for (int t = 0; t < np; ++t)
      images[static_cast<std::size_t>(i) * np + t] = pair_id[g[rep[S_pairs[static_cast<std::size_t>(t)]]]];
    for (int t = 0; t < head_n; ++t) heads[static_cast<std::size_t>(i) * HEAD + t] = images[static_cast<std::size_t>(i) * np + t];
  }
  if (opt.verbose)
    std::printf("sweep: E=%d elements, I=%d images, %d pairs per set, sample space %.3g, setup %.1f s\n", E, I, np,
                static_cast<double>(E) * E * E * I, seconds_since(t0));
  // shared state
  std::vector<uint64_t> U_bits(PW, 0);
  std::vector<uint32_t> U_reps;                 // vertex reps of pairs in U
  std::atomic<unsigned long long> version{0};
  std::atomic<bool> stop{false};
  std::atomic<bool> timed_out{false};
  std::atomic<unsigned long long> samples_total{0};
  std::mutex mu;
  std::vector<int> order(static_cast<std::size_t>(E));
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937_64(opt.seed ^ 0x9e3779b97f4a7c15ull));
  std::atomic<int> m_done{0};

  auto accept = [&](const std::vector<uint32_t>& img_pairs, unsigned long long local_samples) -> bool {
    std::lock_guard<std::mutex> lock(mu);
    if (stop.load()) return false;
    for (uint32_t p : img_pairs)
      if ((U_bits[p >> 6] >> (p & 63)) & 1u) return false;   // stale-M false positive
    std::vector<uint32_t> set;
    set.reserve(static_cast<std::size_t>(2 * np));
    for (uint32_t p : img_pairs) {
      U_bits[p >> 6] |= uint64_t{1} << (p & 63);
      U_reps.push_back(rep[p]);
      set.push_back(rep[p]);
      set.push_back(L.neg[rep[p]]);
    }
    std::sort(set.begin(), set.end());
    res.sets.push_back(std::move(set));
    version.fetch_add(1);
    const unsigned long long s = samples_total.load() + local_samples;
    res.log.emplace_back(s, seconds_since(t0));
    if (opt.verbose)
      std::printf("  set %2zu accepted at %.1f s, samples %.3g (expected 1/p_%zu = %.3g)\n", res.sets.size(),
                  seconds_since(t0), static_cast<double>(s), res.sets.size() - 1,
                  1.0 / std::max(sweep_success_probability(static_cast<int>(res.sets.size()) - 1, np), 1e-300));
    if (static_cast<int>(res.sets.size()) >= opt.target) stop.store(true);
    if (opt.on_accept) opt.on_accept(res.sets);
    return true;
  };

#pragma omp parallel
  {
    std::vector<uint32_t> Uloc, V1, V2;
    std::vector<uint64_t> M(PW, 0);
    std::vector<uint32_t> cand(static_cast<std::size_t>(np));
    unsigned long long local = 0;
    unsigned long long seen_version = ~0ull;
#pragma omp for schedule(dynamic, 1)
    for (int mi = 0; mi < E; ++mi) {
      if (stop.load(std::memory_order_relaxed)) continue;
      const int m = order[static_cast<std::size_t>(mi)];
      const IndexPerm& im = inv[static_cast<std::size_t>(m)];
      for (int l = 0; l < E && !stop.load(std::memory_order_relaxed); ++l) {
        if (seconds_since(t0) > opt.time_limit_s) { timed_out.store(true); stop.store(true); break; }
        // refresh the union snapshot and V1 if it changed
        if (version.load() != seen_version) {
          std::lock_guard<std::mutex> lock(mu);
          Uloc = U_reps;
          seen_version = version.load();
          V1.resize(Uloc.size());
          for (std::size_t t = 0; t < Uloc.size(); ++t) V1[t] = im[Uloc[t]];
        }
        const IndexPerm& il = inv[static_cast<std::size_t>(l)];
        V2.resize(V1.size());
        for (std::size_t t = 0; t < V1.size(); ++t) V2[t] = il[V1[t]];
        for (int j = 0; j < E; ++j) {
          if (stop.load(std::memory_order_relaxed)) break;
          const IndexPerm& ij = inv[static_cast<std::size_t>(j)];
          std::fill(M.begin(), M.end(), 0);
          for (uint32_t v : V2) {
            const uint32_t p = pair_id[ij[v]];
            M[p >> 6] |= uint64_t{1} << (p & 63);
          }
          const uint32_t* hd = heads.data();
          for (int i = 0; i < I; ++i, hd += HEAD) {
            bool hit = false;
            for (int t = 0; t < head_n; ++t) {
              const uint32_t p = hd[t];
              if ((M[p >> 6] >> (p & 63)) & 1u) { hit = true; break; }
            }
            ++local;
            if (hit) continue;
            const uint32_t* img = images.data() + static_cast<std::size_t>(i) * np;
            for (int t = head_n; t < np; ++t) {
              const uint32_t p = img[t];
              if ((M[p >> 6] >> (p & 63)) & 1u) { hit = true; break; }
            }
            if (hit) continue;
            // candidate: g = x_m ∘ x_l ∘ x_j applied to image i
            const IndexPerm& fm = fwd[static_cast<std::size_t>(m)];
            const IndexPerm& fl = fwd[static_cast<std::size_t>(l)];
            const IndexPerm& fj = fwd[static_cast<std::size_t>(j)];
            for (int t = 0; t < np; ++t) cand[static_cast<std::size_t>(t)] = pair_id[fm[fl[fj[rep[img[t]]]]]];
            samples_total.fetch_add(local);
            local = 0;
            accept(cand, 0);
            if (stop.load(std::memory_order_relaxed)) break;
          }
        }
      }
      m_done.fetch_add(1);
    }
    samples_total.fetch_add(local);
  }
  res.samples = samples_total.load();
  res.seconds = seconds_since(t0);
  res.timed_out = timed_out.load();
  res.exhausted = !res.timed_out && static_cast<int>(res.sets.size()) < opt.target && m_done.load() == E;
  return res;
}

// ---------------------------------------------------------------------------
// minimal JSON DOM (family.json only)
// ---------------------------------------------------------------------------
namespace {

struct Json {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  bool b = false;
  std::string s;                                   // number text or string value
  std::vector<Json> a;
  std::vector<std::pair<std::string, Json>> o;
  const Json* get(const std::string& key) const {
    for (const auto& kv : o)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  Json& set(const std::string& key, Json v) {
    for (auto& kv : o)
      if (kv.first == key) { kv.second = std::move(v); return kv.second; }
    o.emplace_back(key, std::move(v));
    return o.back().second;
  }
  long long as_int() const {
    if (type != Number) throw std::runtime_error("json: not a number");
    return std::stoll(s);
  }
  static Json number(long long v) { Json j; j.type = Number; j.s = std::to_string(v); return j; }
  static Json string(const std::string& v) { Json j; j.type = String; j.s = v; return j; }
  static Json array() { Json j; j.type = Array; return j; }
  static Json object() { Json j; j.type = Object; return j; }
};

struct JsonParser {
  const std::string& t;
  std::size_t p = 0;
  explicit JsonParser(const std::string& text) : t(text) {}
  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("family.json parse error: ") + what + " at offset " + std::to_string(p));
  }
  void ws() { while (p < t.size() && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) ++p; }
  bool lit(const char* s) {
    const std::size_t n = std::strlen(s);
    if (t.compare(p, n, s) == 0) { p += n; return true; }
    return false;
  }
  std::string str() {
    if (p >= t.size() || t[p] != '"') fail("expected string");
    ++p;
    std::string out;
    while (p < t.size() && t[p] != '"') {
      if (t[p] == '\\') {
        ++p;
        if (p >= t.size()) fail("bad escape");
        switch (t[p]) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': out += '?'; p += 4; break;   // not needed for our files
          default: fail("bad escape");
        }
        ++p;
      } else {
        out += t[p++];
      }
    }
    if (p >= t.size()) fail("unterminated string");
    ++p;
    return out;
  }
  Json value() {
    ws();
    if (p >= t.size()) fail("unexpected end");
    Json j;
    const char c = t[p];
    if (c == '{') {
      ++p;
      j.type = Json::Object;
      ws();
      if (p < t.size() && t[p] == '}') { ++p; return j; }
      for (;;) {
        ws();
        std::string k = str();
        ws();
        if (p >= t.size() || t[p] != ':') fail("expected ':'");
        ++p;
        Json v = value();
        j.o.emplace_back(std::move(k), std::move(v));
        ws();
        if (p < t.size() && t[p] == ',') { ++p; continue; }
        if (p < t.size() && t[p] == '}') { ++p; return j; }
        fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++p;
      j.type = Json::Array;
      ws();
      if (p < t.size() && t[p] == ']') { ++p; return j; }
      for (;;) {
        j.a.push_back(value());
        ws();
        if (p < t.size() && t[p] == ',') { ++p; continue; }
        if (p < t.size() && t[p] == ']') { ++p; return j; }
        fail("expected ',' or ']'");
      }
    }
    if (c == '"') { j.type = Json::String; j.s = str(); return j; }
    if (lit("null")) return j;
    if (lit("true")) { j.type = Json::Bool; j.b = true; return j; }
    if (lit("false")) { j.type = Json::Bool; j.b = false; return j; }
    // number
    const std::size_t start = p;
    while (p < t.size() && (std::isdigit(static_cast<unsigned char>(t[p])) || t[p] == '-' || t[p] == '+' ||
                            t[p] == '.' || t[p] == 'e' || t[p] == 'E'))
      ++p;
    if (p == start) fail("unexpected character");
    j.type = Json::Number;
    j.s = t.substr(start, p - start);
    return j;
  }
};

Json json_parse(const std::string& text) {
  JsonParser ps(text);
  Json j = ps.value();
  ps.ws();
  if (ps.p != text.size()) ps.fail("trailing characters");
  return j;
}

void json_dump(const Json& j, std::string& out) {
  switch (j.type) {
    case Json::Null: out += "null"; break;
    case Json::Bool: out += j.b ? "true" : "false"; break;
    case Json::Number: out += j.s; break;
    case Json::String:
      out += '"';
      for (char c : j.s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
      }
      out += '"';
      break;
    case Json::Array:
      out += '[';
      for (std::size_t i = 0; i < j.a.size(); ++i) { if (i) out += ','; json_dump(j.a[i], out); }
      out += ']';
      break;
    case Json::Object:
      out += '{';
      for (std::size_t i = 0; i < j.o.size(); ++i) {
        if (i) out += ',';
        json_dump(Json::string(j.o[i].first), out);
        out += ':';
        json_dump(j.o[i].second, out);
      }
      out += '}';
      break;
  }
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// chain
// ---------------------------------------------------------------------------
int chain_length(const Leech& L, const std::vector<uint32_t>& S_idx, const IndexPerm& g, int cap) {
  std::vector<uint64_t> Sm(static_cast<std::size_t>(MASK_WORDS), 0);
  for (uint32_t v : S_idx) {
    if (v >= static_cast<uint32_t>(N)) throw std::out_of_range("chain_length: index >= N");
    Sm[v >> 6] |= uint64_t{1} << (v & 63);
  }
  (void)L;
  std::vector<uint32_t> cur = S_idx;
  for (int i = 1; i < cap; ++i) {
    bool hit = false;
    for (uint32_t& v : cur) {
      v = g[v];
      if ((Sm[v >> 6] >> (v & 63)) & 1u) hit = true;
    }
    if (hit) return i;
  }
  return cap;
}

int pair_order(const Leech& L, const IndexPerm& g) {
  std::vector<char> seen(static_cast<std::size_t>(N), 0);
  long long ord = 1;
  for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
    if (seen[v]) continue;
    long long len = 0;
    uint32_t w = v;
    do {
      seen[w] = 1;
      seen[L.neg[w]] = 1;
      w = g[w];
      ++len;
    } while (w != v && w != L.neg[v]);
    ord = std::lcm(ord, len);
  }
  return static_cast<int>(ord);
}

Aut aut_from_index_perm(const Leech& L, const IndexPerm& g) {
  if (!is_index_permutation(g)) throw std::invalid_argument("aut_from_index_perm: not an index permutation");
  auto f = [&](int i, int j) {   // index of 4e_i + 4e_j
    Vec v{};
    v[static_cast<std::size_t>(i)] = 4;
    v[static_cast<std::size_t>(j)] = 4;
    const int32_t idx = L.index_of(v);
    if (idx < 0) throw std::runtime_error("aut_from_index_perm: (4,4) vector not in C");
    return static_cast<uint32_t>(idx);
  };
  Aut a;
  a.den = 8;
  for (int i = 0; i < DIM; ++i) {
    // 8 e_i = f_ij + f_ik − f_jk  ⇒  8·A e_i = A f_ij + A f_ik − A f_jk
    const int j = (i + 1) % DIM, k = (i + 2) % DIM;
    const Vec& x = L.C[g[f(i, j)]];
    const Vec& y = L.C[g[f(i, k)]];
    const Vec& z = L.C[g[f(j, k)]];
    for (int r = 0; r < DIM; ++r)
      a.num[static_cast<std::size_t>(r * DIM + i)] =
          static_cast<int32_t>(x[static_cast<std::size_t>(r)]) + y[static_cast<std::size_t>(r)] - z[static_cast<std::size_t>(r)];
  }
  return a;
}

ChainResult chain_family(const Leech& L, const std::vector<uint32_t>& S_idx, const std::vector<IndexPerm>& gens,
                         const ChainOptions& opt) {
  const auto t0 = clock_t_::now();
  ChainResult res;
  if (opt.target < 2) throw std::invalid_argument("chain_family: target must be >= 2");
  if (opt.elements < 0) throw std::invalid_argument("chain_family: elements must be >= 0");
  std::vector<uint64_t> Sm(static_cast<std::size_t>(MASK_WORDS), 0);
  for (uint32_t v : S_idx) {
    if (v >= static_cast<uint32_t>(N)) throw std::out_of_range("chain_family: index >= N");
    Sm[v >> 6] |= uint64_t{1} << (v & 63);
  }
  const int T = opt.target;
  res.length_hist.assign(static_cast<std::size_t>(T) + 1, 0);
  std::map<int, std::vector<unsigned long long>> by_order;
  std::atomic<bool> stop{false};
  std::atomic<bool> timed_out{false};
  std::atomic<unsigned long long> elements{0};
  std::atomic<int> m_done{0};
  std::mutex mu;
  const int nthreads = omp_get_max_threads();
  const int E = opt.elements;

  // pool mode: E random elements
  std::vector<IndexPerm> pool;
  if (E > 0) {
    ProductReplacement pr(gens, opt.slots, opt.seed);
    pr.burn_in(opt.burnin);
    pool.reserve(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) pool.push_back(pr.next());
    if (opt.verbose)
      std::printf("chain: pool of %d elements (%.1f MB) in %.1f s; candidates g = x_m x_l x_j, %.3g triples\n", E,
                  static_cast<double>(E) * N * 4 / 1e6, seconds_since(t0), static_cast<double>(E) * E * E);
  }
  std::vector<int> order(static_cast<std::size_t>(std::max(E, 1)));
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937_64(opt.seed ^ 0x5851f42d4c957f2dull));

  // chain test of v -> a[b[v]] (b = identity if null); returns L(g) capped at T
  auto test = [&](const uint32_t* a, const uint32_t* b, std::vector<uint32_t>& cur) -> int {
    std::copy(S_idx.begin(), S_idx.end(), cur.begin());
    for (int i = 1; i < T; ++i) {
      bool hit = false;
      if (b) {
        for (uint32_t& v : cur) {
          v = a[b[v]];
          if ((Sm[v >> 6] >> (v & 63)) & 1u) hit = true;
        }
      } else {
        for (uint32_t& v : cur) {
          v = a[v];
          if ((Sm[v >> 6] >> (v & 63)) & 1u) hit = true;
        }
      }
      if (hit) return i;
    }
    return T;
  };

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::vector<unsigned long long> hist(static_cast<std::size_t>(T) + 1, 0);
    std::map<int, std::vector<unsigned long long>> local_by_order;
    std::vector<uint32_t> cur(S_idx.size());
    unsigned long long local = 0;
    auto record = [&](int len, const IndexPerm& g) {
      ++hist[static_cast<std::size_t>(len)];
      if (len >= opt.stat_min_length && len < T) {
        const int ord = pair_order(L, g);
        auto& hv = local_by_order[ord];
        if (hv.empty()) hv.assign(static_cast<std::size_t>(T) + 1, 0);
        ++hv[static_cast<std::size_t>(len)];
      }
      if (len >= T) {
        std::lock_guard<std::mutex> lock(mu);
        if (!stop.load()) {
          res.g = g;
          res.chain_length = T;
          res.elements = elements.load() + local;
          stop.store(true);
        }
      }
    };
    auto check_limits = [&]() {
      if (seconds_since(t0) > opt.time_limit_s) { timed_out.store(true); stop.store(true); }
      if (opt.max_elements && elements.load() + local >= opt.max_elements) stop.store(true);
    };
    if (E == 0) {
      ProductReplacement pr(gens, opt.slots, opt.seed + 1000003ull * static_cast<uint64_t>(tid));
      pr.burn_in(opt.burnin);
      while (!stop.load(std::memory_order_relaxed)) {
        if ((local & 63) == 0) check_limits();
        if (stop.load(std::memory_order_relaxed)) break;
        const IndexPerm& g = pr.next();
        ++local;
        record(test(g.data(), nullptr, cur), g);
      }
    } else {
      IndexPerm y(static_cast<std::size_t>(N)), g;
#pragma omp for schedule(dynamic, 1)
      for (int mi = 0; mi < E; ++mi) {
        if (stop.load(std::memory_order_relaxed)) continue;
        const int m = order[static_cast<std::size_t>(mi)];
        for (int l = 0; l < E && !stop.load(std::memory_order_relaxed); ++l) {
          check_limits();
          if (stop.load(std::memory_order_relaxed)) break;
          compose_into(y, pool[static_cast<std::size_t>(m)], pool[static_cast<std::size_t>(l)]);
          for (int j = 0; j < E; ++j) {
            const IndexPerm& xj = pool[static_cast<std::size_t>(j)];
            ++local;
            const int len = test(y.data(), xj.data(), cur);
            if (len >= T || len >= opt.stat_min_length) {
              compose_into(g, y, xj);
              record(len, g);
              if (stop.load(std::memory_order_relaxed)) break;
            } else {
              ++hist[static_cast<std::size_t>(len)];
            }
          }
          // flush the per-thread count so that other threads' max_elements checks see it
          elements.fetch_add(local);
          local = 0;
        }
        m_done.fetch_add(1);
      }
    }
    elements.fetch_add(local);
    std::lock_guard<std::mutex> lock(mu);
    for (std::size_t i = 0; i < hist.size(); ++i) res.length_hist[i] += hist[i];
    for (const auto& kv : local_by_order) {
      auto& hv = by_order[kv.first];
      if (hv.empty()) hv.assign(static_cast<std::size_t>(T) + 1, 0);
      for (std::size_t i = 0; i < hv.size(); ++i) hv[i] += kv.second[i];
    }
  }
  res.timed_out = timed_out.load() && res.g.empty();
  res.exhausted = E > 0 && res.g.empty() && !res.timed_out && m_done.load() == E;
  if (res.g.empty()) res.elements = elements.load();
  if (!res.g.empty()) {
    res.order = pair_order(L, res.g);
    // extend the chain length beyond the target (informational, capped at 4·target)
    res.chain_length = chain_length(L, S_idx, res.g, 4 * T);
    std::vector<uint32_t> cur = S_idx;
    for (int i = 0; i < T; ++i) {
      std::vector<uint32_t> s = cur;
      std::sort(s.begin(), s.end());
      res.sets.push_back(std::move(s));
      for (uint32_t& v : cur) v = res.g[v];
    }
    auto& hv = by_order[res.order];
    if (hv.empty()) hv.assign(static_cast<std::size_t>(T) + 1, 0);
    ++hv[static_cast<std::size_t>(T)];
  }
  for (auto& kv : by_order) res.by_order.emplace_back(kv.first, std::move(kv.second));
  res.seconds = seconds_since(t0);
  if (opt.verbose) {
    std::printf("chain: %llu candidates on %d threads in %.1f s (%.0f/s); L(g) histogram:", res.elements, nthreads,
                res.seconds, res.seconds > 0 ? static_cast<double>(res.elements) / res.seconds : 0.0);
    for (int i = 1; i <= T; ++i)
      if (res.length_hist[static_cast<std::size_t>(i)]) std::printf(" %d:%llu", i, res.length_hist[static_cast<std::size_t>(i)]);
    std::printf("\n");
    if (!res.by_order.empty()) {
      std::printf("chain: candidates with L(g) >= %d by order on antipodal pairs (order: L=count ...):\n", opt.stat_min_length);
      for (const auto& kv : res.by_order) {
        std::printf("  %3d:", kv.first);
        for (int i = 1; i <= T; ++i)
          if (kv.second[static_cast<std::size_t>(i)]) std::printf(" %d=%llu", i, kv.second[static_cast<std::size_t>(i)]);
        std::printf("\n");
      }
    }
    if (!res.g.empty())
      std::printf("chain: found g with L(g) = %d (order %d on antipodal pairs) after %llu candidates, %.1f s\n",
                  res.chain_length, res.order, res.elements, res.seconds);
  }
  return res;
}

// ---------------------------------------------------------------------------
// family directories
// ---------------------------------------------------------------------------
FamilyCheck verify_family(const Leech& L, const std::vector<std::vector<Vec>>& sets) {
  FamilyCheck r;
  r.total = 0;
  std::vector<Mask> masks;
  for (std::size_t i = 0; i < sets.size(); ++i) {
    const VerifyResult v = verify_independent(L, sets[i]);
    r.sizes.push_back(sets[i].size());
    r.total += sets[i].size();
    if (!v.ok) {
      r.ok = false;
      r.message = "set " + std::to_string(i + 1) + ": " + v.message;
      return r;
    }
    std::vector<uint32_t> idx;
    idx.reserve(sets[i].size());
    for (const Vec& x : sets[i]) idx.push_back(static_cast<uint32_t>(L.index_of(x)));
    masks.push_back(make_mask(idx));
  }
  for (std::size_t i = 0; i < masks.size(); ++i)
    for (std::size_t j = i + 1; j < masks.size(); ++j) {
      const int ov = mask_overlap(masks[i], masks[j]);
      if (ov) {
        r.ok = false;
        r.message = "sets " + std::to_string(i + 1) + " and " + std::to_string(j + 1) + " overlap in " +
                    std::to_string(ov) + " vectors";
        return r;
      }
    }
  r.ok = true;
  r.message = "ok";
  return r;
}

Family read_family_dir(const std::filesystem::path& dir) {
  Family fam;
  fam.json = read_text(dir / "family.json");
  const Json j = json_parse(fam.json);
  if (j.type != Json::Object) throw std::runtime_error("family.json: top level is not an object");
  if (const Json* d = j.get("dim")) fam.dim = static_cast<int>(d->as_int());
  if (const Json* d = j.get("d")) fam.d = static_cast<int>(d->as_int());
  const Json* sets = j.get("sets");
  if (!sets || sets->type != Json::Array) throw std::runtime_error("family.json: missing \"sets\" array");
  for (const Json& e : sets->a) {
    const Json* f = e.get("file");
    if (!f || f->type != Json::String) throw std::runtime_error("family.json: set entry without \"file\"");
    std::vector<Vec> S = read_set(dir / f->s);
    if (const Json* sz = e.get("size"))
      if (static_cast<std::size_t>(sz->as_int()) != S.size())
        throw std::runtime_error("family.json: " + f->s + " has " + std::to_string(S.size()) + " rows, size says " +
                                 sz->s);
    fam.files.push_back(f->s);
    fam.sets.push_back(std::move(S));
  }
  return fam;
}

void write_family_dir(const std::filesystem::path& dir, const std::vector<std::vector<Vec>>& sets, int dim,
                      const std::filesystem::path& template_json, const std::string& provenance,
                      const std::string& header) {
  std::filesystem::create_directories(dir);
  Json fam = Json::object();
  fam.set("schema_version", Json::number(1));
  fam.set("dim", Json::number(dim));
  fam.set("d", Json::number(dim - 24));
  fam.set("coordinates", Json::string("leech-sqrt8-integer"));
  Json sets_j = Json::array();
  for (std::size_t i = 0; i < sets.size(); ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "S_%02zu.txt", i + 1);
    const std::filesystem::path p = dir / name;
    std::string hdr = "S_" + std::to_string(i + 1) + " of " + std::to_string(sets.size()) + ", |S| = " +
                      std::to_string(sets[i].size()) + ", dim " + std::to_string(dim);
    if (!header.empty()) hdr += "\n" + header;
    write_set(p, sets[i], hdr);
    Json e = Json::object();
    e.set("file", Json::string(name));
    e.set("size", Json::number(static_cast<long long>(sets[i].size())));
    e.set("sha256", Json::string(sha256_file(p)));
    sets_j.a.push_back(std::move(e));
  }
  fam.set("sets", std::move(sets_j));
  long long lifted = 0, n_extra = 0;
  bool have_template = false;
  if (!template_json.empty()) {
    const Json tpl = json_parse(read_text(template_json));
    const Json* T = tpl.get("T");
    const Json* X = tpl.get("extra");
    if (!T || T->type != Json::Object || !X || X->type != Json::Object)
      throw std::runtime_error("template " + template_json.string() + " has no T/extra objects");
    Json Tc = *T;
    const Json* groups = T->get("groups");
    if (!groups || groups->type != Json::Array) throw std::runtime_error("template T has no groups");
    std::vector<Json> gs = groups->a;
    std::stable_sort(gs.begin(), gs.end(), [](const Json& a, const Json& b) { return a.a.size() > b.a.size(); });
    if (gs.size() < sets.size())
      throw std::runtime_error("template has " + std::to_string(gs.size()) + " groups for " +
                               std::to_string(sets.size()) + " sets");
    gs.resize(sets.size());
    Json gj = Json::array();
    for (std::size_t i = 0; i < gs.size(); ++i) {
      lifted += static_cast<long long>(gs[i].a.size() - 1) * static_cast<long long>(sets[i].size());
      gj.a.push_back(gs[i]);
    }
    Tc.set("groups", std::move(gj));
    fam.set("T", std::move(Tc));
    fam.set("extra", *X);
    if (const Json* c = X->get("count")) n_extra = c->as_int();
    have_template = true;
  } else {
    fam.set("T", Json());
    fam.set("extra", Json());
  }
  if (have_template) {
    fam.set("count", Json::number(n_extra + N + lifted));
    Json terms = Json::object();
    terms.set("extra", Json::number(n_extra));
    terms.set("leech", Json::number(N));
    terms.set("lifted", Json::number(lifted));
    fam.set("count_terms", std::move(terms));
    fam.set("count_formula", Json::string(std::to_string(n_extra) + " + 196560 + sum_i (|T_i| - 1) * |S_i|"));
  } else {
    fam.set("count", Json());
    fam.set("count_formula", Json::string("no T/extra template given: sets only"));
  }
  Json prov = Json::object();
  prov.set("tool", Json::string("tools/disjoint_family (kiss::write_family_dir)"));
  prov.set("note", Json::string(provenance));
  if (have_template) prov.set("template", Json::string(template_json.string()));
  fam.set("provenance", std::move(prov));
  std::string out;
  json_dump(fam, out);
  out += '\n';
  std::ofstream f(dir / "family.json");
  if (!f) throw std::runtime_error("cannot write " + (dir / "family.json").string());
  f << out;
}

}  // namespace kiss

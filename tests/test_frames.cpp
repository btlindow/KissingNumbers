// T3.3 acceptance test — frame-structured search (docs/design.md T3.3).
//
//  1. Bitset operations against a naive std::vector<bool> model.
//  2. Clique routines (maximal cliques, k-cliques, max clique) on random
//     synthetic graphs against brute-force subset enumeration.
//  3. Class machinery on C: 98280 classes, conflict counts and clique
//     compatibility against brute-force dots; orthogonal degree 46575 and
//     conflict degree 4600 of class 0.
//  4. (±4,±4)-only frames on M coordinates are perfect matchings: the exact
//     counts through class 0 equal (M−3)!! and all M-cliques equal (M−1)!!
//     for M = 6, 8, 10; the random-descent estimator agrees within 5 se.
//  5. The 496 (data/S496.txt): 248 classes, 3 pairwise-disjoint full frames,
//     2718 maximal cliques with T1.3-convert's size histogram, 7,069,615
//     8-cliques.
//  6. extend_from_class on a tightness-4 class of the 496 against a brute
//     force over all cliques of the outsider pool (best gain, clique count,
//     gain histogram); apply_extend_hit gives an independent set.
//  7. random_frame gives a frame; greedy_frame_union gives an independent
//     set that is a union of the cliques it reports.
//
// Usage: test_frames [S496.txt]   (default data/S496.txt; the vectors are regenerated, no data/ needed otherwise)
#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "kiss/frames.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/verify.h"
#include "kiss/bits.h"

namespace {

int failures = 0;
void expect(bool cond, const std::string& what) {
  if (!cond) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::string hist(const std::map<int, long>& h) {
  std::string s;
  for (const auto& kv : h) s += (s.empty() ? "" : " ") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
  return s;
}

// ---- 1. bitset ---------------------------------------------------------------
void test_bitset() {
  std::mt19937_64 rng(7);
  for (int n : {1, 63, 64, 65, 200, 1000}) {
    kiss::Bitset a(n), b(n), c(n);
    std::vector<bool> A(static_cast<std::size_t>(n)), B(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (rng() & 1) { a.set(i); A[static_cast<std::size_t>(i)] = true; }
      if (rng() & 1) { b.set(i); B[static_cast<std::size_t>(i)] = true; }
    }
    int ca = 0;
    for (bool x : A) ca += x;
    expect(a.count() == ca, "bitset count n=" + std::to_string(n));
    for (int i = 0; i < n; ++i) expect(a.test(i) == A[static_cast<std::size_t>(i)], "bitset test");
    // next / to_list
    std::vector<int> lst = a.to_list();
    std::vector<int> ref;
    for (int i = 0; i < n; ++i)
      if (A[static_cast<std::size_t>(i)]) ref.push_back(i);
    expect(lst == ref, "bitset to_list n=" + std::to_string(n));
    int j = a.first();
    std::vector<int> viaNext;
    while (j >= 0) { viaNext.push_back(j); j = a.next(j + 1); }
    expect(viaNext == ref, "bitset next n=" + std::to_string(n));
    // intersect_above
    for (int v : {-1, 0, n / 2, n - 1}) {
      const int cnt = c.intersect_above(a, b, v);
      int rc = 0;
      bool ok = true;
      for (int i = 0; i < n; ++i) {
        const bool want = A[static_cast<std::size_t>(i)] && B[static_cast<std::size_t>(i)] && i > v;
        rc += want;
        if (c.test(i) != want) ok = false;
      }
      expect(ok && cnt == rc, "bitset intersect_above n=" + std::to_string(n) + " v=" + std::to_string(v));
    }
    kiss::Bitset d = a;
    d.and_not(b);
    kiss::Bitset e = a;
    e.or_with(b);
    kiss::Bitset f = a;
    f.and_with(b);
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      const bool x = A[static_cast<std::size_t>(i)], y = B[static_cast<std::size_t>(i)];
      if (d.test(i) != (x && !y) || e.test(i) != (x || y) || f.test(i) != (x && y)) ok = false;
    }
    expect(ok, "bitset and_not/or_with/and_with n=" + std::to_string(n));
    kiss::Bitset g(n);
    g.fill();
    expect(g.count() == n, "bitset fill n=" + std::to_string(n));
    g.clear();
    expect(g.empty() && g.count() == 0, "bitset clear");
  }
  std::printf("1. bitset: ok\n");
}

// ---- 2. clique routines vs brute force ---------------------------------------
void test_cliques() {
  std::mt19937_64 rng(11);
  for (int trial = 0; trial < 6; ++trial) {
    const int n = 10 + trial * 2;   // up to 20 vertices: 2^20 subsets
    const double p = 0.5 + 0.06 * trial;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<bool>> adj(static_cast<std::size_t>(n), std::vector<bool>(static_cast<std::size_t>(n), false));
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (std::uniform_real_distribution<double>(0, 1)(rng) < p) {
          edges.emplace_back(i, j);
          adj[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = adj[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = true;
        }
    const kiss::PoolGraph G = kiss::pool_graph_from_edges(n, edges);
    // brute force: all cliques by subset
    std::map<int, long> all_by_size, maximal_by_size;
    long through0_by_size[32] = {0};
    int omega = 0;
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
      bool clique = true;
      for (int i = 0; i < n && clique; ++i)
        if (mask >> i & 1u)
          for (int j = i + 1; j < n; ++j)
            if ((mask >> j & 1u) && !adj[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) { clique = false; break; }
      if (!clique) continue;
      const int sz = kiss::popcount32(mask);
      ++all_by_size[sz];
      if (mask & 1u) ++through0_by_size[sz];
      omega = std::max(omega, sz);
      bool maximal = true;
      for (int v = 0; v < n && maximal; ++v) {
        if (mask >> v & 1u) continue;
        bool ext = true;
        for (int i = 0; i < n; ++i)
          if ((mask >> i & 1u) && !adj[static_cast<std::size_t>(i)][static_cast<std::size_t>(v)]) { ext = false; break; }
        if (ext) maximal = false;
      }
      if (maximal) ++maximal_by_size[sz];
    }
    const kiss::CliqueStats mc = kiss::maximal_cliques(G, [&](const std::vector<int>& R) {
      for (std::size_t i = 0; i < R.size(); ++i)
        for (std::size_t j = i + 1; j < R.size(); ++j)
          expect(G.adjacent(R[i], R[j]), "maximal clique reported is a clique");
      return true;
    });
    expect(mc.by_size == maximal_by_size && mc.complete,
           "maximal cliques trial " + std::to_string(trial) + ": got " + hist(mc.by_size) + " want " + hist(maximal_by_size));
    for (int k = 1; k <= omega + 1; ++k) {
      const kiss::CliqueStats kc = kiss::k_cliques(G, k);
      const long want = all_by_size.count(k) ? all_by_size[k] : 0;
      expect(kc.count == want, "k-cliques trial " + std::to_string(trial) + " k=" + std::to_string(k) + ": got " +
                                   std::to_string(kc.count) + " want " + std::to_string(want));
      long cnt_cb = 0;
      const kiss::CliqueStats kc2 = kiss::k_cliques(G, k, [&](const std::vector<int>& R) {
        ++cnt_cb;
        expect(static_cast<int>(R.size()) == k, "k-clique callback size");
        return true;
      });
      expect(cnt_cb == want && kc2.count == want, "k-cliques with callback trial " + std::to_string(trial) + " k=" + std::to_string(k));
      const kiss::CliqueStats kt = kiss::k_cliques(G, k, {}, 0, 0);
      expect(kt.count == through0_by_size[k], "k-cliques through vertex 0 trial " + std::to_string(trial) + " k=" +
                                                   std::to_string(k) + ": got " + std::to_string(kt.count) + " want " +
                                                   std::to_string(through0_by_size[k]));
    }
    const kiss::MaxCliqueResult mx = kiss::max_clique(G);
    expect(static_cast<int>(mx.clique.size()) == omega && mx.complete,
           "max clique trial " + std::to_string(trial) + ": got " + std::to_string(mx.clique.size()) + " want " + std::to_string(omega));
    for (std::size_t i = 0; i < mx.clique.size(); ++i)
      for (std::size_t j = i + 1; j < mx.clique.size(); ++j) expect(G.adjacent(mx.clique[i], mx.clique[j]), "max clique is a clique");
    const kiss::MaxCliqueResult mx2 = kiss::max_clique(G, 0, 2);
    expect(mx2.clique.size() >= 2 || omega < 2, "max clique stop_at_size");
  }
  std::printf("2. clique routines vs brute force: ok\n");
}

// ---- 3. classes ----------------------------------------------------------------
void test_classes(const kiss::Leech& L, const kiss::Classes& K) {
  expect(K.size() == kiss::NCLASS, "98280 classes");
  bool ok = true;
  for (int c = 0; c < kiss::NCLASS; ++c) {
    const uint32_t r = K.rep[static_cast<std::size_t>(c)];
    if (K.cls_of[r] != static_cast<uint32_t>(c) || K.cls_of[L.neg[r]] != static_cast<uint32_t>(c) || r > L.neg[r]) ok = false;
  }
  expect(ok, "class representatives and cls_of consistent");
  expect(K.rep[0] == 0 && kiss::leech_shape(L.C[0]) == 2, "class 0 = vertex 0, a (±4,±4) vector");
  kiss::Bitset o;
  kiss::orthogonal_classes(L, K, 0, o);
  expect(o.count() == kiss::ORTHO_DEG, "orthogonal degree of class 0 = 46575, got " + std::to_string(o.count()));
  const std::vector<uint32_t> cc = kiss::conflicting_classes(L, K, 0);
  expect(static_cast<int>(cc.size()) == kiss::CONFLICT_DEG, "conflict degree of class 0 = 4600, got " + std::to_string(cc.size()));
  // random X: class_conflicts vs brute force on a sample of classes
  std::mt19937_64 rng(3);
  std::vector<uint32_t> X;
  for (int i = 0; i < 300; ++i) X.push_back(static_cast<uint32_t>(rng() % kiss::NCLASS));
  std::sort(X.begin(), X.end());
  X.erase(std::unique(X.begin(), X.end()), X.end());
  const std::vector<uint16_t> conf = kiss::class_conflicts(L, K, X);
  ok = true;
  for (int t = 0; t < 2000; ++t) {
    const uint32_t c = static_cast<uint32_t>(rng() % kiss::NCLASS);
    int cnt = 0;
    for (uint32_t x : X) {
      const int d = kiss::dot(L.C[K.rep[c]], L.C[K.rep[x]]);
      cnt += (d == 16 || d == -16);
    }
    if (conf[c] != cnt) ok = false;
  }
  expect(ok, "class_conflicts vs brute force");
  // conflicts are the tightness of the representative against the antipodal vector set (both signs)
  {
    std::vector<uint32_t> idx;
    for (uint32_t x : X) { idx.push_back(K.rep[x]); idx.push_back(L.neg[K.rep[x]]); }
    const std::vector<uint16_t> tight = kiss::tightness_cpu(L, idx);
    ok = true;
    for (int c = 0; c < kiss::NCLASS; ++c)
      if (tight[K.rep[static_cast<std::size_t>(c)]] != conf[static_cast<std::size_t>(c)]) ok = false;
    expect(ok, "class_conflicts == tightness_cpu of the representatives against ±X");
  }
  // compatibility vs brute force on random small A, B
  ok = true;
  for (int t = 0; t < 200; ++t) {
    std::vector<uint32_t> A, B;
    for (int i = 0; i < 5; ++i) { A.push_back(static_cast<uint32_t>(rng() % kiss::NCLASS)); B.push_back(static_cast<uint32_t>(rng() % kiss::NCLASS)); }
    bool want = true;
    for (uint32_t a : A)
      for (uint32_t b : B) {
        const int d = kiss::dot(L.C[K.rep[a]], L.C[K.rep[b]]);
        if (a != b && (d == 16 || d == -16)) want = false;
      }
    if (kiss::cliques_compatible(L, K, A, B) != want) ok = false;
  }
  expect(ok, "cliques_compatible vs brute force");
  // set_classes / classes_to_vectors round trip
  const std::vector<uint32_t> Y = {0u, 5u, 77u, 98279u};
  const std::vector<kiss::Vec> V = kiss::classes_to_vectors(L, K, Y);
  expect(V.size() == 8 && kiss::is_antipodal(V) && kiss::set_classes(L, K, V) == Y, "classes_to_vectors / set_classes round trip");
  std::printf("3. classes: ok\n");
}

// ---- 4. matching frames ----------------------------------------------------
void test_matchings(const kiss::Leech& L, const kiss::Classes& K) {
  const kiss::Bitset shape2 = kiss::classes_of_shape(L, K, 2);
  expect(shape2.count() == 552, "552 (±4,±4) classes, got " + std::to_string(shape2.count()));
  for (int M : {6, 8, 10}) {
    kiss::Bitset allowed = shape2;
    for (int b = 0; b < kiss::NCLASS; ++b) {
      if (!allowed.test(b)) continue;
      const kiss::Vec& v = L.C[K.rep[static_cast<std::size_t>(b)]];
      for (int i = M; i < kiss::DIM; ++i)
        if (v[static_cast<std::size_t>(i)] != 0) { allowed.reset(b); break; }
    }
    expect(allowed.count() == M * (M - 1), "(±4,±4) classes on M coordinates = M(M−1)");
    const kiss::Neighbourhood Nb = kiss::build_neighbourhood(L, K, 0, &allowed);
    expect(Nb.size() == (M - 2) * (M - 3) + 1, "neighbourhood size (M−2)(M−3)+1 for M=" + std::to_string(M));
    const kiss::FrameCount fc = kiss::count_cliques_through(Nb, M);
    const long want = static_cast<long>(kiss::double_factorial_odd(M - 2));   // (M−3)!!
    expect(fc.complete && fc.frames == want, "matching frames through class 0 on M=" + std::to_string(M) +
                                                 " coordinates: got " + std::to_string(fc.frames) + " want (M−3)!! = " + std::to_string(want));
    // callback version: every reported clique is orthogonal and (with the centre) a matching
    long via_cb = 0;
    const kiss::FrameCount fc2 = kiss::count_cliques_through(Nb, M, 0, [&](const std::vector<int>& R) {
      ++via_cb;
      std::vector<uint32_t> Q{0u};
      for (int i : R) Q.push_back(Nb.verts[static_cast<std::size_t>(i)]);
      if (!kiss::classes_orthogonal_clique(L, K, Q)) expect(false, "reported frame is an orthogonal clique");
      return true;
    });
    expect(fc2.frames == want && via_cb == want, "matching frames via callback M=" + std::to_string(M));
    // all M-cliques in the pool = (M−1)!!
    std::vector<uint32_t> pool = allowed.to_list().empty() ? std::vector<uint32_t>{} : std::vector<uint32_t>{};
    allowed.for_each([&](int b) { pool.push_back(static_cast<uint32_t>(b)); });
    const kiss::PoolGraph G = kiss::induced_orthogonality(L, K, pool);
    const kiss::CliqueStats kc = kiss::k_cliques(G, M);
    const long want_all = static_cast<long>(kiss::double_factorial_odd(M));
    expect(kc.count == want_all, "all matching frames on M=" + std::to_string(M) + ": got " + std::to_string(kc.count) +
                                     " want (M−1)!! = " + std::to_string(want_all));
    // estimator within 5 se
    const kiss::TreeEstimate est = kiss::estimate_cliques_through(Nb, M, 100000, 5);
    expect(std::abs(est.leaves - static_cast<double>(want)) <= 5 * est.leaves_se + 1e-9,
           "estimator M=" + std::to_string(M) + ": " + std::to_string(est.leaves) + " ± " + std::to_string(est.leaves_se) +
               " vs " + std::to_string(want));
    std::printf("   M=%d: through class 0 %ld = (M-3)!!, all %ld = (M-1)!!, estimate %.2f ± %.2f\n", M, fc.frames, kc.count,
                est.leaves, est.leaves_se);
  }
  expect(kiss::double_factorial_odd(24) == 316234143225.0L, "23!! = 316234143225");
  std::printf("4. matching frames: ok\n");
}

// ---- 5. the 496 ---------------------------------------------------------------
std::vector<uint32_t> test_496(const kiss::Leech& L, const kiss::Classes& K, const std::string& file) {
  const std::vector<kiss::Vec> S = kiss::read_set(file);
  expect(S.size() == 496 && kiss::verify_independent(L, S).ok, "S496 loads and verifies");
  const kiss::CrossReport r = kiss::cross_structure(L, K, S, true);
  expect(r.classes.size() == 248 && r.antipodal, "248 antipodal classes");
  expect(r.clique_number == 24, "clique number 24");
  expect(r.frames.size() == 3, "3 full frames, got " + std::to_string(r.frames.size()));
  const std::map<int, long> want_max = {{8, 1280}, {10, 800}, {12, 448}, {16, 171}, {18, 4}, {20, 8}, {22, 4}, {24, 3}};
  expect(r.maximal_total == 2718 && r.maximal_by_size == want_max, "2718 maximal cliques with T1.3's histogram, got " + hist(r.maximal_by_size));
  expect(r.cliques8 == 7069615, "7,069,615 8-cliques, got " + std::to_string(r.cliques8));
  const std::map<int, long> want_deg = {{79, 16}, {87, 88}, {95, 4}, {99, 32}, {103, 72}, {107, 32}, {111, 4}};
  expect(r.degree_hist == want_deg, "orthogonality degree histogram");
  std::set<uint32_t> seen;
  for (const auto& f : r.frames) {
    expect(kiss::classes_orthogonal_clique(L, K, f), "frame is an orthogonal clique");
    for (uint32_t c : f) expect(seen.insert(c).second, "frames pairwise disjoint");
  }
  expect(kiss::classes_independent(L, K, r.classes), "the 248 classes are independent");
  std::printf("5. the 496: %zu classes, %zu frames, %ld maximal cliques, %ld 8-cliques (%.2f s): ok\n", r.classes.size(),
              r.frames.size(), r.maximal_total, r.cliques8, r.seconds);
  return r.classes;
}

// ---- 6. extension search vs brute force -----------------------------------------
void test_extend(const kiss::Leech& L, const kiss::Classes& K, const std::vector<uint32_t>& X) {
  const std::vector<uint16_t> conf = kiss::class_conflicts(L, K, X);
  kiss::Bitset inX(kiss::NCLASS);
  for (uint32_t x : X) inX.set(static_cast<int>(x));
  std::map<int, long> ch;
  for (int b = 0; b < kiss::NCLASS; ++b)
    if (!inX.test(b)) ++ch[conf[static_cast<std::size_t>(b)]];
  expect(ch[4] == 40 && ch[6] == 320 && ch[7] == 128 && ch[8] == 1352 && ch.count(0) == 0 && ch.count(1) == 0 && ch.count(2) == 0 && ch.count(3) == 0,
         "class conflict histogram of the 496 = half of T1.5's tightness histogram, got " + hist(ch));
  // two tightness-4 seeds: one from a 4-component (small pool) and one from a 12-component (frame pool)
  std::vector<uint32_t> seeds;
  for (int b = 0; b < kiss::NCLASS && seeds.size() < 40; ++b)
    if (!inX.test(b) && conf[static_cast<std::size_t>(b)] == 4) seeds.push_back(static_cast<uint32_t>(b));
  int tested = 0;
  for (uint32_t c : seeds) {
    kiss::ExtendOptions eo;
    eo.tmax = 4;
    eo.kmin = 1;
    eo.min_gain = -1000;
    eo.time_limit_s = 0;
    const kiss::ExtendResult er = kiss::extend_from_class(L, K, X, conf, c, eo);
    expect(er.complete, "extend complete");
    // brute force: outsider pool, all cliques through c via k_cliques on the induced graph
    std::vector<uint32_t> outs{c};
    for (int b = 0; b < kiss::NCLASS; ++b)
      if (!inX.test(b) && static_cast<uint32_t>(b) != c && conf[static_cast<std::size_t>(b)] <= 4 &&
          kiss::class_orthogonal(L, K, c, static_cast<uint32_t>(b)))
        outs.push_back(static_cast<uint32_t>(b));
    expect(static_cast<int>(outs.size()) - 1 == er.pool_outside, "pool_outside");
    if (outs.size() > 17) continue;   // keep the brute force small (a 23-outsider pool is a complete graph: 2^23 cliques)
    const kiss::PoolGraph G = kiss::induced_orthogonality(L, K, outs);
    std::vector<std::vector<uint32_t>> cs(outs.size());
    for (std::size_t i = 0; i < outs.size(); ++i) {
      for (std::size_t j = 0; j < X.size(); ++j)
        if (kiss::class_conflict(L, K, outs[i], X[j])) cs[i].push_back(static_cast<uint32_t>(j));
    }
    long total = 0;
    int best = -1000000;
    std::map<int, long> gh;
    for (int k = 1; k <= static_cast<int>(outs.size()); ++k) {
      const kiss::CliqueStats kc = kiss::k_cliques(G, k, [&](const std::vector<int>& R) {
        std::set<uint32_t> U;
        for (int i : R) U.insert(cs[static_cast<std::size_t>(i)].begin(), cs[static_cast<std::size_t>(i)].end());
        const int gain = k - static_cast<int>(U.size());
        ++gh[gain];
        best = std::max(best, gain);
        ++total;
        return true;
      }, 0, 0);
      if (kc.count == 0) break;
    }
    expect(er.cliques == total, "extend clique count vs brute force: " + std::to_string(er.cliques) + " vs " + std::to_string(total));
    expect(er.best_gain == best, "extend best gain vs brute force: " + std::to_string(er.best_gain) + " vs " + std::to_string(best));
    expect(er.gain_hist == gh, "extend gain histogram vs brute force: " + hist(er.gain_hist) + " vs " + hist(gh));
    expect(er.best.gain() == er.best_gain && er.best.added == static_cast<int>(er.best.clique.size()), "best hit consistent");
    const std::vector<uint32_t> X2 = kiss::apply_extend_hit(L, K, X, er.best);
    expect(static_cast<int>(X2.size()) == static_cast<int>(X.size()) + er.best_gain, "apply_extend_hit size = |X| + gain");
    expect(kiss::classes_independent(L, K, X2), "apply_extend_hit independent");
    // with the bound active the best gain is unchanged
    kiss::ExtendOptions eb = eo;
    eb.min_gain = best;
    const kiss::ExtendResult eb_r = kiss::extend_from_class(L, K, X, conf, c, eb);
    expect(eb_r.best_gain == best, "bounded search finds the same best gain");
    expect(eb_r.nodes <= er.nodes, "bound prunes");
    std::printf("   seed %u: pool %d outsiders, %ld cliques, best gain %d (brute force %d), bounded nodes %ld/%ld\n", c,
                er.pool_outside, er.cliques, er.best_gain, best, eb_r.nodes, er.nodes);
    if (++tested >= 3) break;
  }
  expect(tested >= 1, "at least one extension seed brute-forced");
  std::printf("6. extension search vs brute force: ok\n");
}

// ---- 7. random frames / greedy ---------------------------------------------------
void test_greedy(const kiss::Leech& L, const kiss::Classes& K) {
  const std::vector<uint32_t> F = kiss::random_frame(L, K, 12345u, 9);
  expect(F.size() == 24 && F[0] == 12345u && kiss::classes_orthogonal_clique(L, K, F), "random_frame gives a frame through the start");
  kiss::GreedyOptions go;
  go.max_clique_time_s = 2;
  const kiss::GreedyResult g = kiss::greedy_frame_union(L, K, 42, go);
  expect(kiss::classes_independent(L, K, g.classes), "greedy union independent");
  long sum = 0;
  for (int s : g.clique_sizes) sum += s;
  expect(sum == static_cast<long>(g.classes.size()) && !g.clique_sizes.empty() && g.clique_sizes[0] == 24,
         "greedy clique sizes sum to |U| and start with a frame");
  const std::vector<uint16_t> conf = kiss::class_conflicts(L, K, g.classes);
  bool maximal = true;
  kiss::Bitset inU(kiss::NCLASS);
  for (uint32_t c : g.classes) inU.set(static_cast<int>(c));
  for (int b = 0; b < kiss::NCLASS; ++b)
    if (!inU.test(b) && conf[static_cast<std::size_t>(b)] == 0) maximal = false;
  expect(maximal, "greedy union is maximal (no free class)");
  std::printf("7. random frame + greedy union (%zu classes = %zu vectors, %zu cliques, %.1f s): ok\n", g.classes.size(),
              2 * g.classes.size(), g.clique_sizes.size(), g.seconds);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string s496 = argc > 1 ? argv[1] : "data/S496.txt";
    test_bitset();
    test_cliques();
    const kiss::Leech L = kiss::generate_leech();
    const kiss::Classes K = kiss::make_classes(L);
    test_classes(L, K);
    test_matchings(L, K);
    std::vector<uint32_t> X;
    if (std::filesystem::exists(s496)) {
      X = test_496(L, K, s496);
      test_extend(L, K, X);
    } else {
      std::printf("5/6. %s not found: skipped\n", s496.c_str());
    }
    test_greedy(L, K);
  } catch (const std::exception& e) {
    std::printf("exception: %s\n", e.what());
    ++failures;
  }
  std::printf("RESULT test_frames ok=%d failures=%d\n", failures == 0 ? 1 : 0, failures);
  return failures == 0 ? 0 : 1;
}

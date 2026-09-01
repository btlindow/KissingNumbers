// T4.2 — acceptance tests for kiss/family.h.
//
//  1. masks: mask_overlap / mask_disjoint / overlap_sorted agree with a brute
//     force set intersection on random index sets.
//  2. pool: disjointness_graph (inverted index) == disjointness_graph_masks
//     (mask ANDs) on a pool of random images of S; spot checks vs overlap_sorted.
//  3. planted clique: greedy / local search / exact B&B recover a planted
//     k-clique in a sparse random graph; exact search proves non-existence on
//     a small instance.
//  4. family directory round trip: a family of disjoint images written with
//     write_family_dir (T/extra copied from data/families/dim27/family.json if
//     present), read back, verified; corrupted families are rejected.
//  5. data/families/dim26 (if present) verifies with total 992.
//  6. sweep: a short sweep finds 6 pairwise disjoint images.
//  7. chain: chain_family (pool mode) finds g with S, gS, .., g^5 S pairwise
//     disjoint; chain_length / pair_order sanity; aut_from_index_perm gives a
//     rational matrix whose index permutation reproduces g.
// Usage: test_family [--set data/S496.txt] [--pool 400] [--seed 7]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "kiss/family.h"
#include "kiss/group.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/verify.h"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      ++g_failures;                                                 \
      std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                     \
      std::printf("\n");                                            \
    }                                                               \
  } while (0)

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<uint32_t> random_indices(std::mt19937_64& rng, int n, int lo, int hi) {
  std::set<uint32_t> s;
  while (static_cast<int>(s.size()) < n) s.insert(static_cast<uint32_t>(lo + rng() % static_cast<uint64_t>(hi - lo)));
  return std::vector<uint32_t>(s.begin(), s.end());
}

}  // namespace

int main(int argc, char** argv) {
  const auto t0 = std::chrono::steady_clock::now();
  std::string set_file = "data/S496.txt";
  int pool_size = 400;
  uint64_t seed = 7;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--set") set_file = argv[i + 1];
    else if (a == "--pool") pool_size = std::atoi(argv[i + 1]);
    else if (a == "--seed") seed = std::strtoull(argv[i + 1], nullptr, 10);
  }
  std::mt19937_64 rng(seed);

  // ---- 1. masks vs brute force -------------------------------------------
  {
    int checked = 0;
    for (int t = 0; t < 200; ++t) {
      const int lo = static_cast<int>(rng() % 150000);
      const auto a = random_indices(rng, 300 + static_cast<int>(rng() % 300), lo, lo + 4000);
      const auto b = random_indices(rng, 300 + static_cast<int>(rng() % 300), lo, lo + 4000);
      std::vector<uint32_t> inter;
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
      const kiss::Mask ma = kiss::make_mask(a), mb = kiss::make_mask(b);
      CHECK(kiss::mask_overlap(ma, mb) == static_cast<int>(inter.size()), "mask_overlap %d vs %zu",
            kiss::mask_overlap(ma, mb), inter.size());
      CHECK(kiss::mask_disjoint(ma, mb) == inter.empty(), "mask_disjoint");
      CHECK(kiss::overlap_sorted(a, b) == static_cast<int>(inter.size()), "overlap_sorted");
      ++checked;
    }
    const auto a = random_indices(rng, 500, 0, kiss::N);
    const auto b = random_indices(rng, 500, 0, kiss::N);
    std::vector<uint32_t> inter;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
    CHECK(kiss::mask_overlap(kiss::make_mask(a), kiss::make_mask(b)) == static_cast<int>(inter.size()), "full range");
    bool threw = false;
    try { kiss::make_mask({static_cast<uint32_t>(kiss::N)}); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw, "make_mask must reject index N");
    std::printf("masks: %d random pairs agree with brute force\n", checked);
  }

  // ---- Leech, S, generators ------------------------------------------------
  kiss::Leech L;
  try { L = kiss::load_leech("data"); } catch (const std::exception&) { L = kiss::generate_leech(); }
  std::vector<kiss::Vec> S;
  try { S = kiss::read_set(set_file); } catch (const std::exception&) { S.clear(); }
  if (S.empty() || !kiss::verify_independent(L, S).ok) {
    // fallback: random greedy antipodal independent set
    std::printf("note: %s unavailable, using a random greedy antipodal set\n", set_file.c_str());
    std::vector<uint32_t> tight(static_cast<std::size_t>(kiss::N), 0);
    std::vector<uint32_t> order(static_cast<std::size_t>(kiss::N));
    for (uint32_t i = 0; i < static_cast<uint32_t>(kiss::N); ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<uint32_t> chosen;
    for (uint32_t v : order) {
      if (tight[v]) continue;
      const uint32_t nv = L.neg[v];
      bool ok = true;
      for (uint32_t u : chosen)
        if (kiss::dot(L.C[u], L.C[v]) == 16 || kiss::dot(L.C[u], L.C[nv]) == 16) { ok = false; break; }
      if (!ok) continue;
      chosen.push_back(v);
      chosen.push_back(nv);
      tight[v] = tight[nv] = 1;
      if (chosen.size() >= 200) break;
    }
    for (uint32_t v : chosen) S.push_back(L.C[v]);
  }
  std::vector<uint32_t> S_idx = kiss::set_indices(L, S);
  std::sort(S_idx.begin(), S_idx.end());
  std::printf("S = %s: %zu vectors, antipodal=%d\n", set_file.c_str(), S.size(), kiss::is_antipodal(S) ? 1 : 0);
  const auto gens = kiss::co0_generator_perms(L, "data/group");

  // ---- 2. pool + disjointness graph two ways ------------------------------
  kiss::ImagePool pool;
  {
    const auto t1 = std::chrono::steady_clock::now();
    pool = kiss::random_image_pool(S_idx, gens, static_cast<std::size_t>(pool_size), seed);
    const double t_pool = ms_since(t1);
    const auto t2 = std::chrono::steady_clock::now();
    const kiss::BitGraph G1 = kiss::disjointness_graph(pool);
    const double t_g1 = ms_since(t2);
    const auto t3 = std::chrono::steady_clock::now();
    const kiss::BitGraph G2 = kiss::disjointness_graph_masks(pool);
    const double t_g2 = ms_since(t3);
    CHECK(G1.n == pool_size && G2.n == pool_size, "graph sizes");
    CHECK(G1.bits == G2.bits, "inverted-index graph != mask graph");
    int mism = 0;
    for (int t = 0; t < 3000; ++t) {
      const int i = static_cast<int>(rng() % static_cast<uint64_t>(pool_size));
      const int j = static_cast<int>(rng() % static_cast<uint64_t>(pool_size));
      if (i == j) { CHECK(!G1.adj(i, i), "self loop"); continue; }
      const bool disj = kiss::overlap_sorted(pool.sets[static_cast<std::size_t>(i)], pool.sets[static_cast<std::size_t>(j)]) == 0;
      if (disj != G1.adj(i, j)) ++mism;
    }
    CHECK(mism == 0, "%d adjacency mismatches vs overlap_sorted", mism);
    for (const auto& s : pool.sets) CHECK(s.size() == S.size() && std::is_sorted(s.begin(), s.end()), "image shape");
    std::printf("pool: %d images in %.0f ms; graph density %.3f (inverted index %.0f ms, masks %.0f ms), identical\n",
                pool_size, t_pool, G1.density(), t_g1, t_g2);
    CHECK(G1.density() > 0.3 && G1.density() < 0.8, "density %.3f implausible", G1.density());
  }

  // ---- 3. planted clique ---------------------------------------------------
  {
    const int n = 600, kp = 14;
    const double p = 0.3;
    kiss::BitGraph G(n);
    std::bernoulli_distribution coin(p);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (coin(rng)) G.set_edge(i, j);
    std::vector<int> planted;
    {
      std::vector<int> all(static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i) all[static_cast<std::size_t>(i)] = i;
      std::shuffle(all.begin(), all.end(), rng);
      planted.assign(all.begin(), all.begin() + kp);
    }
    for (int a = 0; a < kp; ++a)
      for (int b = a + 1; b < kp; ++b) G.set_edge(planted[static_cast<std::size_t>(a)], planted[static_cast<std::size_t>(b)]);
    CHECK(kiss::is_clique(G, planted), "planted is a clique");
    std::sort(planted.begin(), planted.end());
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<int> g = kiss::greedy_clique(G, kp, seed, 200);
    const double t_g = ms_since(t1);
    const auto t2 = std::chrono::steady_clock::now();
    std::vector<int> ls = kiss::local_search_clique(G, kp, seed, 200000, 60.0);
    const double t_ls = ms_since(t2);
    const auto t3 = std::chrono::steady_clock::now();
    long long nodes = 0;
    std::vector<int> bb = kiss::exact_clique(G, kp, 50000000, &nodes);
    const double t_bb = ms_since(t3);
    CHECK(kiss::is_clique(G, g) && kiss::is_clique(G, ls) && kiss::is_clique(G, bb), "returned cliques valid");
    std::sort(g.begin(), g.end());
    std::sort(ls.begin(), ls.end());
    std::sort(bb.begin(), bb.end());
    CHECK(static_cast<int>(ls.size()) >= kp, "local search found %zu < %d", ls.size(), kp);
    CHECK(static_cast<int>(bb.size()) == kp, "exact B&B found %zu (nodes %lld)", bb.size(), nodes);
    CHECK(ls == planted || static_cast<int>(ls.size()) >= kp, "ls result");
    std::printf("planted %d-clique in G(%d,%.1f): greedy %zu (%.0f ms), ls %zu (%.0f ms, planted=%d), "
                "bb %zu (%.0f ms, %lld nodes, planted=%d)\n",
                kp, n, p, g.size(), t_g, ls.size(), t_ls, ls == planted ? 1 : 0, bb.size(), t_bb, nodes,
                bb == planted ? 1 : 0);
    // exact non-existence on a small instance
    kiss::BitGraph H(40);
    for (int i = 0; i < 40; ++i)
      for (int j = i + 1; j < 40; ++j)
        if (coin(rng)) H.set_edge(i, j);
    long long nh = 0;
    const std::vector<int> none = kiss::exact_clique(H, 12, 10000000, &nh);
    CHECK(none.empty() && nh < 10000000, "12-clique in G(40,0.3) must not exist (nodes %lld)", nh);
    const std::vector<int> five = kiss::exact_clique(G, 5, 10000000, &nh);
    CHECK(five.size() == 5 && kiss::is_clique(G, five), "5-clique exists");
    // expected-count formula sanity: E[#2-cliques in G(n,p)] = C(n,2) p
    const double lg2 = kiss::log10_expected_cliques(600, 2, 0.3);
    CHECK(std::abs(std::pow(10.0, lg2) - 600 * 599 / 2.0 * 0.3) < 1e-6 * 600 * 599 / 2.0 * 0.3, "log10_expected_cliques k=2");
  }

  // ---- 4. family directory round trip --------------------------------------
  {
    const kiss::BitGraph G = kiss::disjointness_graph(pool);
    std::vector<int> clique = kiss::greedy_clique(G, 5, seed, 50);
    CHECK(clique.size() >= 3, "need >= 3 disjoint images in the pool, got %zu", clique.size());
    if (clique.size() > 5) clique.resize(5);
    std::vector<std::vector<kiss::Vec>> sets;
    for (int v : clique) {
      std::vector<kiss::Vec> s;
      for (uint32_t i : pool.sets[static_cast<std::size_t>(v)]) s.push_back(L.C[i]);
      sets.push_back(std::move(s));
    }
    const kiss::FamilyCheck direct = kiss::verify_family(L, sets);
    CHECK(direct.ok, "direct verify_family: %s", direct.message.c_str());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("kiss_test_family_" + std::to_string(seed));
    std::filesystem::remove_all(dir);
    const std::filesystem::path tpl = std::filesystem::exists("data/families/dim27/family.json") ? "data/families/dim27/family.json" : "";
    kiss::write_family_dir(dir, sets, 27, tpl, "test_family round trip", "test header line 1\nline 2");
    const kiss::Family back = kiss::read_family_dir(dir);
    CHECK(back.dim == 27 && back.d == 3, "dim/d read back (%d, %d)", back.dim, back.d);
    CHECK(back.sets.size() == sets.size(), "set count %zu vs %zu", back.sets.size(), sets.size());
    for (std::size_t i = 0; i < sets.size() && i < back.sets.size(); ++i)
      CHECK(back.sets[i] == sets[i], "set %zu differs after round trip", i);
    const kiss::FamilyCheck chk = kiss::verify_family(L, back.sets);
    CHECK(chk.ok && chk.total == sets.size() * S.size(), "round trip verify: %s", chk.message.c_str());
    if (!tpl.empty()) {
      CHECK(back.json.find("\"T\":{") != std::string::npos && back.json.find("\"groups\":[") != std::string::npos,
            "template T block copied");
      const std::string want = "\"count\":" + std::to_string(12 + kiss::N + 2 * 496 + 2 * 496 + (sets.size() >= 3 ? 496 : 0) +
                                                                (sets.size() >= 4 ? 496 : 0) + (sets.size() >= 5 ? 496 : 0));
      CHECK(S.size() != 496 || back.json.find(want) != std::string::npos, "count recomputed (%s)", want.c_str());
    }
    // corrupted: duplicate set
    std::vector<std::vector<kiss::Vec>> bad = sets;
    bad[1] = bad[0];
    const kiss::FamilyCheck c1 = kiss::verify_family(L, bad);
    CHECK(!c1.ok && c1.message.find("overlap") != std::string::npos, "duplicate set must fail: %s", c1.message.c_str());
    // corrupted: one shared vector
    bad = sets;
    bad[1][0] = bad[0][0];
    const kiss::FamilyCheck c2 = kiss::verify_family(L, bad);
    CHECK(!c2.ok, "shared vector must fail");
    // corrupted: a 60-degree pair inside a set
    bad = sets;
    {
      const uint32_t v = pool.sets[static_cast<std::size_t>(clique[0])][0];
      // find a neighbour at inner product 16
      for (uint32_t u = 0; u < static_cast<uint32_t>(kiss::N); ++u)
        if (kiss::dot(L.C[u], L.C[v]) == 16) { bad[0].push_back(L.C[u]); break; }
    }
    const kiss::FamilyCheck c3 = kiss::verify_family(L, bad);
    CHECK(!c3.ok && c3.message.find("set 1") != std::string::npos, "60-degree pair must fail: %s", c3.message.c_str());
    // corrupted file on disk: size mismatch in family.json
    {
      std::string js = back.json;
      const std::size_t pos = js.find("\"size\":");
      if (pos != std::string::npos) {
        js.replace(pos, 7, "\"size\":1");
        std::ofstream f(dir / "family.json");
        f << js;
      }
      bool threw = false;
      try { kiss::read_family_dir(dir); } catch (const std::runtime_error&) { threw = true; }
      CHECK(threw, "size mismatch must throw");
    }
    std::filesystem::remove_all(dir);
    std::printf("family round trip: %zu sets, verify=%s, corrupted variants rejected\n", sets.size(), chk.message.c_str());
  }

  // ---- 5. committed baseline family ------------------------------------------
  if (std::filesystem::exists("data/families/dim26/family.json")) {
    const kiss::Family fam = kiss::read_family_dir("data/families/dim26");
    const kiss::FamilyCheck chk = kiss::verify_family(L, fam.sets);
    CHECK(chk.ok && fam.sets.size() == 2 && chk.total == 992, "data/families/dim26: %s (sets %zu, total %zu)",
          chk.message.c_str(), fam.sets.size(), chk.total);
    std::printf("data/families/dim26: %zu sets, total %zu, %s\n", fam.sets.size(), chk.total, chk.message.c_str());
  }

  // ---- 6. sweep ------------------------------------------------------------------
  if (kiss::is_antipodal(S)) {
    kiss::SweepOptions opt;
    opt.elements = 24;
    opt.images = 256;
    opt.target = 6;
    opt.time_limit_s = 120;
    opt.seed = seed;
    opt.verbose = false;
    const auto t1 = std::chrono::steady_clock::now();
    const kiss::SweepResult r = kiss::sweep_family(L, S_idx, gens, opt);
    const double t_sw = ms_since(t1);
    CHECK(static_cast<int>(r.sets.size()) == 6, "sweep found %zu of 6 (samples %llu)", r.sets.size(), r.samples);
    std::vector<std::vector<kiss::Vec>> sets;
    for (const auto& idx : r.sets) {
      std::vector<kiss::Vec> s;
      for (uint32_t i : idx) s.push_back(L.C[i]);
      CHECK(idx.size() == S.size(), "sweep set size %zu", idx.size());
      sets.push_back(std::move(s));
    }
    const kiss::FamilyCheck chk = kiss::verify_family(L, sets);
    CHECK(chk.ok, "sweep family: %s", chk.message.c_str());
    std::printf("sweep: %zu disjoint images in %.0f ms, %llu samples, verify=%s\n", r.sets.size(), t_sw, r.samples,
                chk.message.c_str());
  }

  // ---- 7. chain ------------------------------------------------------------------
  {
    // sanity on the identity and a generator
    const kiss::IndexPerm id = kiss::index_identity();
    CHECK(kiss::pair_order(L, id) == 1, "pair_order(id) = %d", kiss::pair_order(L, id));
    CHECK(kiss::chain_length(L, S_idx, id, 10) == 1, "chain_length(id) = %d", kiss::chain_length(L, S_idx, id, 10));
    const kiss::IndexPerm& g0 = gens.front();
    const int ord0 = kiss::pair_order(L, g0);
    CHECK(ord0 >= 1 && ord0 <= 120, "pair_order(gen0) = %d", ord0);
    {
      // g0^ord0 fixes every antipodal pair
      std::vector<uint32_t> v = {0, 1, 12345, static_cast<uint32_t>(kiss::N - 1)};
      for (uint32_t x : v) {
        uint32_t w = x;
        for (int i = 0; i < ord0; ++i) w = g0[w];
        CHECK(w == x || w == L.neg[x], "g0^%d moves pair of %u", ord0, x);
      }
      const kiss::Aut A0 = kiss::aut_from_index_perm(L, g0);
      CHECK(kiss::index_permutation(L, A0) == g0, "aut_from_index_perm(gen0) does not reproduce gen0");
    }
    const auto t1 = std::chrono::steady_clock::now();
    kiss::ChainOptions opt;
    opt.target = 6;
    opt.elements = 32;
    opt.time_limit_s = 120;
    opt.seed = seed;
    opt.verbose = false;
    const kiss::ChainResult r = kiss::chain_family(L, S_idx, gens, opt);
    const double t_ch = ms_since(t1);
    CHECK(r.sets.size() == 6 && !r.g.empty(), "chain found %zu of 6 (%llu candidates)", r.sets.size(), r.elements);
    if (!r.g.empty()) {
      CHECK(r.chain_length >= 6, "chain_length %d", r.chain_length);
      CHECK(kiss::chain_length(L, S_idx, r.g, 6) == 6, "chain_length recheck");
      CHECK(r.order >= 6 || r.order == 0, "order %d < chain length", r.order);
      std::vector<std::vector<kiss::Vec>> sets;
      for (const auto& idx : r.sets) {
        std::vector<kiss::Vec> s;
        for (uint32_t i : idx) s.push_back(L.C[i]);
        sets.push_back(std::move(s));
      }
      const kiss::FamilyCheck chk = kiss::verify_family(L, sets);
      CHECK(chk.ok, "chain family: %s", chk.message.c_str());
      // S_{i+1} = g S_i
      for (std::size_t i = 0; i + 1 < r.sets.size(); ++i) {
        std::vector<uint32_t> img = kiss::apply(r.g, r.sets[i]);
        std::sort(img.begin(), img.end());
        CHECK(img == r.sets[i + 1], "chain set %zu is not g * set %zu", i + 1, i);
      }
      const kiss::Aut A = kiss::aut_from_index_perm(L, r.g);
      CHECK(A.den == 8, "den %d", A.den);
      CHECK(kiss::index_permutation(L, A) == r.g, "aut_from_index_perm(g) does not reproduce g");
      std::printf("chain: %zu disjoint images (g of order %d on pairs, L(g) = %d) in %.0f ms, %llu candidates, verify=%s\n",
                  r.sets.size(), r.order, r.chain_length, t_ch, r.elements, chk.message.c_str());
    }
  }

  std::printf("RESULT ok=%d failures=%d pool=%d set=%s size=%zu ms=%.1f\n", g_failures == 0 ? 1 : 0, g_failures, pool_size,
              set_file.c_str(), S.size(), ms_since(t0));
  return g_failures == 0 ? 0 : 1;
}

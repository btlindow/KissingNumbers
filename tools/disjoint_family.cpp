// T4.2 — search for k pairwise disjoint images g·S of an independent set S
// and write them as a family directory (data/families/SCHEMA.md).
//
//   disjoint_family --set data/S496.txt --k 42 --out runs/families/dim31
//                   [--method sweep|greedy|ls|bb] [--pool 5000] [--seed 1]
//                   [--dim 31] [--template data/families/dim31/family.json]
//                   [--time 3600] [--elements 512] [--images 16384]
//                   [--restarts 20] [--steps 200000] [--nodes 100000000]
//                   [--data data] [--group data/group]
//   disjoint_family --verify data/families/dim31      (read + verify a family directory only)
//
// Methods:
//   greedy / ls / bb  the pipeline — a pool of `--pool` random
//                     images, the disjointness graph (edge iff disjoint), then
//                     greedy growth / swap-based local search / exact branch
//                     and bound for a k-clique. Also prints the graph density
//                     p and log10 of the expected number of k-cliques in
//                     G(pool, p), and the pool size at which that expectation
//                     reaches 1.
//   sweep             greedy chain over the implicit pool {x_m x_l x_j h_i(S)}
//                     (kiss/family.h); reaches far larger k for the same time.
//   chain (default)   orbit S, gS, ..., g^{k-1}S of a single random element g
//                     with g^i S ∩ S = ∅ for i = 1..k-1 (kiss/family.h): k-1
//                     conditions instead of C(k,2); reaches k = 42 in minutes.
//                     Writes g as g.txt (24×24 rational matrix, den 8) and
//                     g.u32 (index permutation) next to the family.
// --dim defaults to the dimension whose record uses k sets (2→26, 5→27, 8→28,
// 14→29, 24→30, 42→31); --template defaults to data/families/dim<dim>/family.json
// if that file exists (its T/extra blocks are copied into the written family.json).
// Last line: RESULT ok=<verified> k=<target> found=<sets written> ... target_met=<0|1>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "kiss/family.h"
#include "kiss/group.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/verify.h"

namespace {

int dim_for_k(int k) {
  switch (k) {
    case 2: return 26;
    case 5: return 27;
    case 8: return 28;
    case 14: return 29;
    case 24: return 30;
    case 42: return 31;
    default: return 0;
  }
}

}  // namespace

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::string set_file, out_dir, method = "chain", data_dir = "data", group_dir, template_json, verify_dir;
  int k = 42, dim = 0, pool = 5000, elements = 512, images = 16384, restarts = 20, slots = 10, burnin = 100;
  long long max_elements = 0;
  long long steps = 200000, nodes = 100000000;
  double time_limit = 3600;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
      return argv[++i];
    };
    if (a == "--set") set_file = val("--set");
    else if (a == "--verify") verify_dir = val("--verify");
    else if (a == "--k") k = std::atoi(val("--k").c_str());
    else if (a == "--out") out_dir = val("--out");
    else if (a == "--method") method = val("--method");
    else if (a == "--pool") pool = std::atoi(val("--pool").c_str());
    else if (a == "--seed") seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
    else if (a == "--dim") dim = std::atoi(val("--dim").c_str());
    else if (a == "--template") template_json = val("--template");
    else if (a == "--time") time_limit = std::atof(val("--time").c_str());
    else if (a == "--elements") elements = std::atoi(val("--elements").c_str());
    else if (a == "--images") images = std::atoi(val("--images").c_str());
    else if (a == "--restarts") restarts = std::atoi(val("--restarts").c_str());
    else if (a == "--steps") steps = std::atoll(val("--steps").c_str());
    else if (a == "--nodes") nodes = std::atoll(val("--nodes").c_str());
    else if (a == "--slots") slots = std::atoi(val("--slots").c_str());
    else if (a == "--burnin") burnin = std::atoi(val("--burnin").c_str());
    else if (a == "--max-elements") max_elements = std::atoll(val("--max-elements").c_str());
    else if (a == "--data") data_dir = val("--data");
    else if (a == "--group") group_dir = val("--group");
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (!verify_dir.empty()) {
    try {
      kiss::Leech L;
      try { L = kiss::load_leech(data_dir); } catch (const std::exception&) { L = kiss::generate_leech(); }
      const kiss::Family fam = kiss::read_family_dir(verify_dir);
      const kiss::FamilyCheck chk = kiss::verify_family(L, fam.sets);
      long long count = -1;   // the top-level "count" (extra.count also matches the key: take the largest value)
      for (std::size_t pos = fam.json.find("\"count\":"); pos != std::string::npos; pos = fam.json.find("\"count\":", pos + 8))
        count = std::max(count, std::atoll(fam.json.c_str() + pos + 8));
      std::printf("%s: dim %d, %zu sets, total %zu, count %lld, verify: %s\n", verify_dir.c_str(), fam.dim, fam.sets.size(),
                  chk.total, count, chk.message.c_str());
      std::printf("RESULT ok=%d dir=%s dim=%d sets=%zu total=%zu count=%lld seconds=%.1f\n", chk.ok ? 1 : 0,
                  verify_dir.c_str(), fam.dim, fam.sets.size(), chk.total, count,
                  std::chrono::duration<double>(clock::now() - t0).count());
      return chk.ok ? 0 : 1;
    } catch (const std::exception& e) {
      std::printf("RESULT ok=0 dir=%s error=\"%s\"\n", verify_dir.c_str(), e.what());
      return 1;
    }
  }
  if (set_file.empty() || out_dir.empty()) {
    std::fprintf(stderr, "usage: disjoint_family --set S.txt --k K --out dir/ [--method chain|sweep|greedy|ls|bb] ...\n");
    return 2;
  }
  if (group_dir.empty()) group_dir = (std::filesystem::path(data_dir) / "group").string();
  if (dim == 0) dim = dim_for_k(k);
  if (template_json.empty() && dim) {
    const std::filesystem::path cand = std::filesystem::path(data_dir) / "families" / ("dim" + std::to_string(dim)) / "family.json";
    if (std::filesystem::exists(cand)) template_json = cand.string();
  }
  try {
    kiss::Leech L;
    try { L = kiss::load_leech(data_dir); } catch (const std::exception&) { L = kiss::generate_leech(); }
    const std::vector<kiss::Vec> S = kiss::read_set(set_file);
    const kiss::VerifyResult vs = kiss::verify_independent(L, S);
    if (!vs.ok) {
      std::printf("RESULT ok=0 error=\"input set: %s\"\n", vs.message.c_str());
      return 1;
    }
    std::vector<uint32_t> S_idx = kiss::set_indices(L, S);
    std::sort(S_idx.begin(), S_idx.end());
    const auto gens = kiss::co0_generator_perms(L, group_dir);
    std::printf("S = %s (%zu vectors, antipodal=%d), k = %d, dim = %d, method = %s, seed = %llu, template = %s\n",
                set_file.c_str(), S.size(), kiss::is_antipodal(S) ? 1 : 0, k, dim, method.c_str(),
                static_cast<unsigned long long>(seed), template_json.empty() ? "(none)" : template_json.c_str());

    std::vector<std::vector<uint32_t>> found_idx;
    std::string extra_fields, provenance;
    if (method == "sweep") {
      kiss::SweepOptions opt;
      opt.elements = elements;
      opt.images = images;
      opt.target = k;
      opt.time_limit_s = time_limit;
      opt.seed = seed;
      opt.slots = slots;
      opt.burnin = burnin;
      // checkpoint: rewrite the family directory after every acceptance
      opt.on_accept = [&](const std::vector<std::vector<uint32_t>>& found) {
        std::vector<std::vector<kiss::Vec>> sets;
        for (const auto& idx : found) {
          std::vector<kiss::Vec> v;
          v.reserve(idx.size());
          for (uint32_t i : idx) v.push_back(L.C[i]);
          sets.push_back(std::move(v));
        }
        try {
          kiss::write_family_dir(out_dir, sets, dim, template_json,
                                 "checkpoint of a running sweep (seed " + std::to_string(seed) + "); S = " + set_file,
                                 "image of " + set_file + " under a random Leech automorphism (tools/disjoint_family, checkpoint)");
        } catch (const std::exception& e) {
          std::fprintf(stderr, "checkpoint failed: %s\n", e.what());
        }
        std::fflush(stdout);
      };
      std::printf("expected samples per step 1/p_j (j sets already in the union): ");
      for (int j = 0; j < k; j += std::max(1, k / 8))
        std::printf("j=%d:%.2g ", j, 1.0 / std::max(kiss::sweep_success_probability(j), 1e-300));
      std::printf("j=%d:%.2g\n", k - 1, 1.0 / std::max(kiss::sweep_success_probability(k - 1), 1e-300));
      const kiss::SweepResult r = kiss::sweep_family(L, S_idx, gens, opt);
      found_idx = r.sets;
      char buf[256];
      std::snprintf(buf, sizeof buf, " samples=%llu elements=%d images=%d exhausted=%d timed_out=%d",
                    r.samples, elements, images, r.exhausted ? 1 : 0, r.timed_out ? 1 : 0);
      extra_fields = buf;
      provenance = "sweep over {x_m x_l x_j h_i(S)}: seed " + std::to_string(seed) + ", E=" + std::to_string(elements) +
                   " elements, I=" + std::to_string(images) + " images, " + std::to_string(r.samples) + " samples, " +
                   std::to_string(r.seconds) + " s; sets in acceptance order; S = " + set_file;
    } else if (method == "chain") {
      kiss::ChainOptions opt;
      opt.target = k;
      opt.time_limit_s = time_limit;
      opt.seed = seed;
      opt.slots = slots;
      opt.burnin = burnin;
      opt.max_elements = static_cast<unsigned long long>(std::max(0LL, max_elements));
      opt.elements = elements;
      const kiss::ChainResult r = kiss::chain_family(L, S_idx, gens, opt);
      found_idx = r.sets;
      char buf[256];
      std::snprintf(buf, sizeof buf, " candidates=%llu elements=%d chain_length=%d order=%d timed_out=%d exhausted=%d",
                    r.elements, elements, r.chain_length, r.order, r.timed_out ? 1 : 0, r.exhausted ? 1 : 0);
      extra_fields = buf;
      if (!r.g.empty()) {
        std::filesystem::create_directories(out_dir);
        const kiss::Aut A = kiss::aut_from_index_perm(L, r.g);
        const kiss::IndexPerm back = kiss::index_permutation(L, A);   // throws unless A is an automorphism
        if (back != r.g) { std::printf("RESULT ok=0 error=\"g.txt does not reproduce g\"\n"); return 1; }
        kiss::save_aut(std::filesystem::path(out_dir) / "g.txt",
                       A, "g in Co_0 (24x24 rational matrix, rows = images of the coordinate vectors scaled by den): S_i = g^{i-1} S, i = 1.." +
                              std::to_string(k) + "; order " + std::to_string(r.order) + " on antipodal pairs; L(g) = " +
                              std::to_string(r.chain_length) + "; tools/disjoint_family --method chain --seed " + std::to_string(seed));
        kiss::write_index_perm(std::filesystem::path(out_dir) / "g.u32", r.g);
      }
      provenance = "chain S_i = g^{i-1} S (i = 1.." + std::to_string(k) + ") of a random element g of Co_0 (g = x_m x_l x_j over a pool of " +
                   std::to_string(elements) + " product-replacement elements, seed " + std::to_string(seed) + ", " +
                   std::to_string(r.elements) + " candidates tested, " + std::to_string(r.seconds) + " s); g in g.txt (rational matrix) / g.u32 (index permutation); order of g on antipodal pairs " +
                   std::to_string(r.order) + ", L(g) = " + std::to_string(r.chain_length) + "; S = " + set_file;
    } else if (method == "greedy" || method == "ls" || method == "bb") {
      const auto t1 = clock::now();
      const kiss::ImagePool P = kiss::random_image_pool(S_idx, gens, static_cast<std::size_t>(pool), seed, slots, burnin);
      const double t_pool = std::chrono::duration<double>(clock::now() - t1).count();
      const auto t2 = clock::now();
      const kiss::BitGraph G = kiss::disjointness_graph(P);
      const double t_graph = std::chrono::duration<double>(clock::now() - t2).count();
      const double p = G.density();
      const double lg = kiss::log10_expected_cliques(static_cast<double>(pool), k, p);
      // pool size where E[#k-cliques] = 1: log10 C(n,k) ≈ -C(k,2) log10 p
      double n_needed = static_cast<double>(pool);
      {
        double lo = 1, hi = 1e12;
        for (int it = 0; it < 200; ++it) {
          const double mid = std::sqrt(lo * hi);
          if (kiss::log10_expected_cliques(mid, k, p) < 0) lo = mid; else hi = mid;
        }
        n_needed = hi;
      }
      std::printf("pool %d images in %.1f s; disjointness graph: density p = %.4f, %lld edges in %.1f s; "
                  "log10 E[#%d-cliques in G(%d,p)] = %.1f; pool needed for E = 1: %.3g; "
                  "2 log_{1/p} n = %.1f (typical clique number of G(n,p))\n",
                  pool, t_pool, p, G.edges(), t_graph, k, pool, lg, n_needed, 2 * std::log(pool) / std::log(1 / p));
      const auto t3 = clock::now();
      std::vector<int> clique;
      long long nodes_used = 0;
      if (method == "greedy") {
        clique = kiss::greedy_clique(G, k, seed, restarts);
      } else if (method == "ls") {
        const std::vector<int> start = kiss::greedy_clique(G, k, seed, restarts);
        clique = kiss::local_search_clique(G, k, seed, steps, time_limit, &start);
      } else {
        clique = kiss::exact_clique(G, k, nodes, &nodes_used);
        if (clique.empty())
          std::printf("exact: no %d-clique found (%s, %lld nodes)\n", k,
                      nodes_used < nodes ? "exhaustive: none exists in this pool" : "node limit", nodes_used);
      }
      const double t_search = std::chrono::duration<double>(clock::now() - t3).count();
      if (!kiss::is_clique(G, clique)) { std::printf("RESULT ok=0 error=\"search returned a non-clique\"\n"); return 1; }
      std::printf("%s: clique of size %zu in %.1f s%s\n", method.c_str(), clique.size(), t_search,
                  static_cast<int>(clique.size()) > k ? " (truncated to k)" : "");
      if (static_cast<int>(clique.size()) > k) clique.resize(static_cast<std::size_t>(k));
      for (int v : clique) found_idx.push_back(P.sets[static_cast<std::size_t>(v)]);
      char buf[256];
      std::snprintf(buf, sizeof buf, " density=%.4f log10_expected=%.1f pool_needed=%.3g nodes=%lld", p, lg, n_needed,
                    nodes_used);
      extra_fields = buf;
      provenance = method + " clique on the disjointness graph of " + std::to_string(pool) + " random images (seed " +
                   std::to_string(seed) + ", density " + std::to_string(p) + "); S = " + set_file;
    } else {
      std::fprintf(stderr, "unknown method %s\n", method.c_str());
      return 2;
    }
    // write + verify
    std::vector<std::vector<kiss::Vec>> sets;
    for (const auto& idx : found_idx) {
      std::vector<kiss::Vec> v;
      v.reserve(idx.size());
      for (uint32_t i : idx) v.push_back(L.C[i]);
      sets.push_back(std::move(v));
    }
    kiss::write_family_dir(out_dir, sets, dim, template_json, provenance,
                           method == "chain" ? "S_i = g^(i-1) S for the automorphism g of g.txt, S = " + set_file + " (tools/disjoint_family --method chain)"
                                             : "image of " + set_file + " under a random Leech automorphism (tools/disjoint_family)");
    const kiss::Family back = kiss::read_family_dir(out_dir);
    const kiss::FamilyCheck chk = kiss::verify_family(L, back.sets);
    const double secs = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("written %s: %zu sets, total %zu vectors, verify: %s\n", out_dir.c_str(), back.sets.size(), chk.total,
                chk.message.c_str());
    std::printf("RESULT ok=%d k=%d found=%zu pool=%d method=%s seed=%llu dim=%d total=%zu target_met=%d%s seconds=%.1f\n",
                chk.ok ? 1 : 0, k, back.sets.size(), method == "sweep" ? elements * elements * elements : method == "chain" ? 0 : pool,
                method.c_str(), static_cast<unsigned long long>(seed), dim, chk.total,
                static_cast<int>(back.sets.size()) >= k ? 1 : 0, extra_fields.c_str(), secs);
    return chk.ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("RESULT ok=0 error=\"%s\"\n", e.what());
    return 1;
  }
}

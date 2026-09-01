// T4.1 — random elements of Co_0 by product replacement, as index permutations.
//
//   random_aut --seed S --count K --out dir/ [--slots 10] [--burnin 100]
//              [--data data] [--group data/group] [--check 0|1]
//
// Writes dir/aut_<k>.u32 (uint32[196560], k = 0..K-1) and prints
//   RESULT ok=1 seed=.. count=.. slots=.. burnin=.. out=.. ip_fail=.. ms=..
// With --check 1 (default) every element is tested for inner-product
// preservation on 1000 random pairs (ip_fail counts failing elements).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "kiss/group.h"
#include "kiss/leech.h"

int main(int argc, char** argv) {
  uint64_t seed = 1;
  int count = 10, slots = 10, burnin = 100, check = 1;
  std::string out_dir, data_dir = "data", group_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
      return argv[++i];
    };
    if (a == "--seed") seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
    else if (a == "--count") count = std::atoi(val("--count").c_str());
    else if (a == "--slots") slots = std::atoi(val("--slots").c_str());
    else if (a == "--burnin") burnin = std::atoi(val("--burnin").c_str());
    else if (a == "--out") out_dir = val("--out");
    else if (a == "--data") data_dir = val("--data");
    else if (a == "--group") group_dir = val("--group");
    else if (a == "--check") check = std::atoi(val("--check").c_str());
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (out_dir.empty()) { std::fprintf(stderr, "usage: random_aut --seed S --count K --out dir/\n"); return 2; }
  if (group_dir.empty()) group_dir = (std::filesystem::path(data_dir) / "group").string();
  const auto t0 = std::chrono::steady_clock::now();
  try {
    std::filesystem::create_directories(out_dir);
    kiss::Leech L;
    try { L = kiss::load_leech(data_dir); } catch (const std::exception&) { L = kiss::generate_leech(); }
    const auto gens = kiss::co0_generator_perms(L, group_dir);
    kiss::ProductReplacement pr(gens, slots, seed);
    pr.burn_in(burnin);
    int ip_fail = 0;
    for (int k = 0; k < count; ++k) {
      const kiss::IndexPerm& g = pr.next();
      if (check && kiss::inner_product_violations(L, g, 1000, seed * 1000003ull + static_cast<uint64_t>(k)) != 0)
        ++ip_fail;
      char name[64];
      std::snprintf(name, sizeof name, "aut_%04d.u32", k);
      kiss::write_index_perm(std::filesystem::path(out_dir) / name, g);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("RESULT ok=%d seed=%llu count=%d slots=%d burnin=%d gens=%zu out=%s ip_fail=%d ms=%.1f\n",
                ip_fail == 0 ? 1 : 0, static_cast<unsigned long long>(seed), count, pr.slots(), burnin,
                gens.size(), out_dir.c_str(), ip_fail, ms);
    return ip_fail == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("RESULT ok=0 error=\"%s\"\n", e.what());
    return 1;
  }
}

// T1.4 — acceptance tests for kiss/verify.h.
//
//  1. A greedy random maximal independent set (built here from generate_leech,
//     fixed seed) verifies; its size is reported (~238 expected).
//  2. Corrupted variants are rejected with the offending row(s)/pair named:
//     a member replaced by a 60°-neighbour of another member (pair + ip 16),
//     a duplicate row, a wrong-norm row, a norm-32 vector that is not in C
//     (octad vector with one ±2 moved off the octad).
//  3. tightness_cpu: on the greedy set no vertex outside S is free; on a
//     random 100-subset it equals a serial brute-force double loop with the
//     opposite loop order, Σ tight = 100·4600; single-member and empty cases.
//  4. set_line_numbers / line-labelled messages.
//  5. Fixtures data/S496.txt and data/S488.txt (skipped with a note if absent):
//     verify, size, antipodal, Gram histogram, tightness histogram (free = 0),
//     and the 60°-corruption of the 496 is rejected naming the pair.
// Prints `RESULT ok=1 ...`; exit code 0 iff all pass. Optional argv[1] = data dir.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/verify.h"
#include "kiss/platform.h"

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

using kiss::DIM;
using kiss::N;
using kiss::Vec;

bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

// Greedy random maximal independent set: visit vertices in a random order,
// add a vertex unless it is at 60° to a member already chosen.
std::vector<uint32_t> greedy_mis(const kiss::Leech& L, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> perm(static_cast<std::size_t>(N));
  std::iota(perm.begin(), perm.end(), 0u);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<uint8_t> blocked(static_cast<std::size_t>(N), 0);
  std::vector<uint32_t> S;
  for (uint32_t v : perm) {
    if (blocked[v]) continue;
    S.push_back(v);
    const Vec& x = L.C[v];
#pragma omp parallel for schedule(static)
    for (long j = 0; j < N; ++j)
      if (kiss::dot(x, L.C[static_cast<std::size_t>(j)]) == 16) blocked[static_cast<std::size_t>(j)] = 1;
  }
  return S;
}

std::vector<Vec> gather(const kiss::Leech& L, const std::vector<uint32_t>& idx) {
  std::vector<Vec> S;
  S.reserve(idx.size());
  for (uint32_t i : idx) S.push_back(L.C[i]);
  return S;
}

// First vector y (canonical order) with ⟨x,y⟩ = 16.
Vec neighbour60(const kiss::Leech& L, const Vec& x) {
  for (const Vec& y : L.C)
    if (kiss::dot(x, y) == 16) return y;
  return Vec{};
}

// Norm-32 vector NOT in C: take an octad-shape vector and move one ±2 to a
// coordinate outside the octad (the support is no longer a Golay codeword).
Vec octad_moved(const kiss::Leech& L) {
  for (const Vec& v : L.C) {
    if (kiss::leech_shape(v) != 0) continue;
    Vec w = v;
    std::size_t from = 0, to = 0;
    while (w[from] == 0) ++from;
    while (w[to] != 0) ++to;
    w[to] = w[from];
    w[from] = 0;
    return w;
  }
  return Vec{};
}

std::string hist_str(const std::map<int, long>& h) {
  std::string s;
  for (const auto& kv : h) {
    if (!s.empty()) s += ',';
    s += std::to_string(kv.first) + ":" + std::to_string(kv.second);
  }
  return s;
}

std::string gram_str(const std::array<long, 8>& gh) {
  const int vals[7] = {-32, -16, -8, 0, 8, 16, 32};
  std::string s;
  for (std::size_t c = 0; c < 7; ++c) {
    if (!gh[c]) continue;
    if (!s.empty()) s += ',';
    s += std::to_string(vals[c]) + ":" + std::to_string(gh[c]);
  }
  if (gh[7]) s += ",other:" + std::to_string(gh[7]);
  return s;
}

// Tightness histogram over vertices outside S; returns the free count.
long tight_profile(const kiss::Leech& L, const std::vector<uint32_t>& idx,
                   const std::vector<uint16_t>& tight, std::map<int, long>& hist) {
  std::vector<uint8_t> in_s(L.C.size(), 0);
  for (uint32_t i : idx) in_s[i] = 1;
  hist.clear();
  long free_count = 0;
  for (std::size_t v = 0; v < tight.size(); ++v) {
    if (in_s[v]) continue;
    ++hist[tight[v]];
    if (tight[v] == 0) ++free_count;
  }
  return free_count;
}

}  // namespace

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const fs::path data_dir = argc > 1 ? fs::path(argv[1]) : fs::path("data");

  const kiss::Leech L = kiss::generate_leech();

  // ---- 1. greedy random maximal independent set ---------------------------
  const std::vector<uint32_t> g_idx = greedy_mis(L, 20260825u);
  const std::vector<Vec> G = gather(L, g_idx);
  const kiss::VerifyResult rg = kiss::verify_independent(L, G);
  CHECK(rg.ok, "greedy set rejected: %s", rg.message.c_str());
  CHECK(rg.size == G.size(), "size %zu != %zu", rg.size, G.size());
  CHECK(G.size() > 150 && G.size() < 400, "greedy size %zu implausible", G.size());
  std::printf("greedy MIS          : size=%zu antipodal=%d gram=%s verify=%s\n", G.size(),
              kiss::is_antipodal(G) ? 1 : 0, gram_str(kiss::gram_histogram(G)).c_str(),
              rg.message.c_str());
  {
    const std::vector<uint32_t> back = kiss::set_indices(L, G);
    CHECK(back == g_idx, "set_indices does not round-trip");
    const kiss::VerifyResult re = kiss::verify_independent(L, {});
    CHECK(re.ok && re.size == 0, "empty set: ok=%d size=%zu", re.ok ? 1 : 0, re.size);
  }

  // ---- 2. corrupted variants ------------------------------------------------
  {
    // (a) 60°-neighbour: row 1 replaced by a y with ⟨G[0],y⟩ = 16.
    std::vector<Vec> B = G;
    const Vec y = neighbour60(L, G[0]);
    CHECK(kiss::dot(G[0], y) == 16, "no 60-degree neighbour found");
    B[1] = y;
    const kiss::VerifyResult r = kiss::verify_independent(L, B);
    CHECK(!r.ok, "60-degree corruption accepted");
    CHECK(contains(r.message, "row 0") && contains(r.message, "row 1") &&
              contains(r.message, "inner product 16"),
          "pair not named: %s", r.message.c_str());
    CHECK(r.size == G.size(), "size on failure %zu", r.size);
    std::printf("corrupt 60-degree   : %s\n", r.message.c_str());
    // Same with line labels: rows k ↦ line 10+2k.
    std::vector<long> lines;
    for (std::size_t k = 0; k < B.size(); ++k) lines.push_back(10 + 2 * static_cast<long>(k));
    const kiss::VerifyResult rl = kiss::verify_independent(L, B, lines);
    CHECK(!rl.ok && contains(rl.message, "line 10 (row 0)") && contains(rl.message, "line 12 (row 1)"),
          "line labels missing: %s", rl.message.c_str());
  }
  {
    // (b) duplicate row.
    std::vector<Vec> B = G;
    B[5] = B[3];
    const kiss::VerifyResult r = kiss::verify_independent(L, B);
    CHECK(!r.ok && contains(r.message, "distinct") && contains(r.message, "row 3") &&
              contains(r.message, "row 5"),
          "duplicate not named: %s", r.message.c_str());
    std::printf("corrupt duplicate   : %s\n", r.message.c_str());
  }
  {
    // (c) wrong norm: (8, 0^23) is a lattice vector of norm 64.
    std::vector<Vec> B = G;
    Vec w{};
    w[0] = 8;
    B[2] = w;
    const kiss::VerifyResult r = kiss::verify_independent(L, B);
    CHECK(!r.ok && contains(r.message, "norm") && contains(r.message, "row 2") && contains(r.message, "64"),
          "wrong norm not named: %s", r.message.c_str());
    std::printf("corrupt norm        : %s\n", r.message.c_str());
    // Also a norm-32-looking but non-int8-shaped vector: (4,4,0..) with a 1 → norm 33.
  }
  {
    // (d) norm 32 but not in C.
    std::vector<Vec> B = G;
    const Vec w = octad_moved(L);
    CHECK(kiss::norm2(w) == 32, "octad_moved norm %d", kiss::norm2(w));
    CHECK(L.index_of(w) < 0, "octad_moved is in C");
    std::array<int, DIM> wi{};
    for (std::size_t k = 0; k < static_cast<std::size_t>(DIM); ++k) wi[k] = w[k];
    CHECK(!L.is_lattice_vector(wi), "octad_moved passes membership");
    B[4] = w;
    const kiss::VerifyResult r = kiss::verify_independent(L, B);
    CHECK(!r.ok && contains(r.message, "membership") && contains(r.message, "row 4"),
          "non-member not named: %s", r.message.c_str());
    std::printf("corrupt not-in-C    : %s\n", r.message.c_str());
    bool threw = false;
    try {
      kiss::set_indices(L, B);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "set_indices accepted a non-member");
  }
  {
    // (e) check order: a set with a norm problem AND a Gram problem reports the norm first.
    std::vector<Vec> B = G;
    B[1] = neighbour60(L, G[0]);
    Vec w{};
    w[0] = 8;
    B[7] = w;
    const kiss::VerifyResult r = kiss::verify_independent(L, B);
    CHECK(!r.ok && contains(r.message, "norm:"), "check order: %s", r.message.c_str());
  }

  // ---- 3. tightness_cpu ------------------------------------------------------
  double tight_ms = 0;
  {
    const auto t1 = clock::now();
    const std::vector<uint16_t> tight = kiss::tightness_cpu(L, g_idx);
    tight_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    CHECK(tight.size() == static_cast<std::size_t>(N), "tight size %zu", tight.size());
    std::map<int, long> hist;
    const long free_count = tight_profile(L, g_idx, tight, hist);
    CHECK(free_count == 0, "greedy set not maximal: %ld free vertices", free_count);
    for (uint32_t s : g_idx) CHECK(tight[s] == 0, "member %u has tight %u", s, tight[s]);
    long sum = 0;
    for (uint16_t t : tight) sum += t;
    CHECK(sum == static_cast<long>(g_idx.size()) * kiss::DEG, "sum tight %ld != |S|*4600", sum);
    std::printf("tightness greedy    : free=%ld hist(outside S)=%s sum=%ld=%zu*4600 [%.1f ms]\n",
                free_count, hist_str(hist).c_str(), sum, g_idx.size(), tight_ms);
  }
  {
    // Random 100-subset vs serial brute force (opposite loop order).
    std::mt19937 rng(7u);
    std::vector<uint32_t> perm(static_cast<std::size_t>(N));
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    const std::vector<uint32_t> sub(perm.begin(), perm.begin() + 100);
    const std::vector<uint16_t> tight = kiss::tightness_cpu(L, sub);
    std::vector<int> brute(static_cast<std::size_t>(N), 0);
    for (uint32_t s : sub) {
      const Vec& x = L.C[s];
      for (std::size_t v = 0; v < static_cast<std::size_t>(N); ++v) {
        int ip = 0;
        for (std::size_t k = 0; k < static_cast<std::size_t>(DIM); ++k)
          ip += static_cast<int>(x[k]) * static_cast<int>(L.C[v][k]);
        if (ip == 16) ++brute[v];
      }
    }
    long mism = 0, sum = 0;
    int maxt = 0;
    for (std::size_t v = 0; v < static_cast<std::size_t>(N); ++v) {
      if (static_cast<int>(tight[v]) != brute[v]) ++mism;
      sum += tight[v];
      maxt = std::max(maxt, static_cast<int>(tight[v]));
    }
    CHECK(mism == 0, "tightness mismatches vs brute force: %ld", mism);
    CHECK(sum == 100L * kiss::DEG, "sum %ld != 460000", sum);
    std::printf("tightness random100 : mismatches=%ld sum=%ld max=%d\n", mism, sum, maxt);
    // Single member: exactly the 4600 neighbours, member itself 0.
    const std::vector<uint16_t> one = kiss::tightness_cpu(L, {sub[0]});
    long cnt = 0;
    for (std::size_t v = 0; v < one.size(); ++v) {
      cnt += one[v];
      if (one[v]) CHECK(kiss::dot(L.C[v], L.C[sub[0]]) == 16, "one: wrong vertex %zu", v);
    }
    CHECK(cnt == kiss::DEG && one[sub[0]] == 0, "single member: count %ld", cnt);
    // Empty S: all zero.
    const std::vector<uint16_t> zero = kiss::tightness_cpu(L, {});
    CHECK(std::all_of(zero.begin(), zero.end(), [](uint16_t t) { return t == 0; }), "empty S nonzero");
    // Bad index throws.
    bool threw = false;
    try {
      kiss::tightness_cpu(L, {static_cast<uint32_t>(N)});
    } catch (const std::out_of_range&) {
      threw = true;
    }
    CHECK(threw, "out-of-range index accepted");
  }

  // ---- 4. set_line_numbers + file round trip ---------------------------------
  {
    const fs::path tmp = fs::temp_directory_path() /
                         ("kiss_test_verify_" + std::to_string(kiss::process_id()));
    fs::create_directories(tmp);
    {
      std::ofstream out(tmp / "s.txt");
      out << "# header\n\n";
      for (std::size_t k = 0; k < 5; ++k) {
        if (k == 2) out << "   \n# mid comment\n";
        for (std::size_t c = 0; c < static_cast<std::size_t>(DIM); ++c)
          out << (c ? " " : "") << static_cast<int>(G[k][c]);
        out << (k == 3 ? "  # trailing\n" : "\n");
      }
    }
    const std::vector<long> lines = kiss::set_line_numbers(tmp / "s.txt");
    const std::vector<long> expect = {3, 4, 7, 8, 9};
    CHECK(lines == expect, "line numbers: got %zu entries", lines.size());
    const std::vector<Vec> S5 = kiss::read_set(tmp / "s.txt");
    CHECK(S5.size() == 5, "read 5 rows: %zu", S5.size());
    const kiss::VerifyResult r5 = kiss::verify_independent(L, S5, lines);
    CHECK(r5.ok && r5.size == 5, "5-row file: %s", r5.message.c_str());
    bool threw = false;
    try {
      kiss::set_line_numbers(tmp / "missing.txt");
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "missing file accepted");
    fs::remove_all(tmp);
  }

  // ---- 5. fixtures ------------------------------------------------------------
  std::map<std::string, std::size_t> fixture_size;
  std::string fixture_note;
  for (const char* nm : {"S496.txt", "S488.txt"}) {
    const fs::path p = data_dir / nm;
    if (!fs::exists(p)) {
      std::printf("fixture %-12s: SKIP (file not found: %s)\n", nm, p.string().c_str());
      fixture_note += std::string(fixture_note.empty() ? "" : ",") + nm + ":missing";
      continue;
    }
    std::vector<Vec> S;
    std::vector<long> lines;
    try {
      S = kiss::read_set(p);
      lines = kiss::set_line_numbers(p);
    } catch (const std::exception& e) {
      CHECK(false, "fixture %s unreadable: %s", nm, e.what());
      continue;
    }
    const kiss::VerifyResult r = kiss::verify_independent(L, S, lines);
    CHECK(r.ok, "fixture %s rejected: %s", nm, r.message.c_str());
    const std::size_t expect = std::string(nm) == "S496.txt" ? 496 : 488;
    CHECK(r.size == expect, "fixture %s size %zu != %zu", nm, r.size, expect);
    fixture_size[nm] = r.size;
    const bool anti = kiss::is_antipodal(S);
    const std::array<long, 8> gh = kiss::gram_histogram(S);
    std::printf("fixture %-12s: ok=%d size=%zu antipodal=%d gram=%s\n", nm, r.ok ? 1 : 0, r.size,
                anti ? 1 : 0, gram_str(gh).c_str());
    if (!r.ok) continue;
    const std::vector<uint32_t> idx = kiss::set_indices(L, S);
    const auto t1 = clock::now();
    const std::vector<uint16_t> tight = kiss::tightness_cpu(L, idx);
    const double ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    std::map<int, long> hist;
    const long free_count = tight_profile(L, idx, tight, hist);
    long sum = 0;
    for (uint16_t t : tight) sum += t;
    CHECK(free_count == 0, "fixture %s: %ld free vertices (not maximal)", nm, free_count);
    CHECK(sum == static_cast<long>(idx.size()) * kiss::DEG, "fixture %s: sum tight %ld", nm, sum);
    for (uint32_t s : idx) CHECK(tight[s] == 0, "fixture %s: member %u tight %u", nm, s, tight[s]);
    std::printf("fixture %-12s: tightness free=%ld hist(outside S)=%s [%.1f ms]\n", nm, free_count,
                hist_str(hist).c_str(), ms);
    // 60°-corruption: replace row 1 by a neighbour of row 0.
    std::vector<Vec> B = S;
    B[1] = neighbour60(L, S[0]);
    const kiss::VerifyResult rb = kiss::verify_independent(L, B, lines);
    CHECK(!rb.ok && contains(rb.message, "row 0") && contains(rb.message, "row 1") &&
              contains(rb.message, "inner product 16"),
          "fixture %s corruption not caught: %s", nm, rb.message.c_str());
    std::printf("fixture %-12s: corrupted -> %s\n", nm, rb.message.c_str());
  }

  const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::printf("RESULT ok=%d greedy_size=%zu s496=%zu s488=%zu fixtures=%s threads=%d "
              "tight_ms=%.1f total_ms=%.0f failures=%d\n",
              g_failures == 0 ? 1 : 0, G.size(), fixture_size.count("S496.txt") ? fixture_size["S496.txt"] : 0,
              fixture_size.count("S488.txt") ? fixture_size["S488.txt"] : 0,
              fixture_note.empty() ? "present" : fixture_note.c_str(), omp_get_max_threads(),
              tight_ms, total_ms, g_failures);
  return g_failures == 0 ? 0 : 1;
}

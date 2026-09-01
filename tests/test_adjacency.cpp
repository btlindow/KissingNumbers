// T1.6 acceptance test — data/adj.u32 and the Adjacency mmap reader (docs/design.md T1.6).
//
//   test_adjacency [data_dir]           (default: data; ctest passes `data`)
//
// If <data_dir>/adj.u32 is missing (or has the wrong size) the test builds it
// first with kiss::cuda::build_adjacency_file — the same library function
// tools/build_adj uses — so ctest is self-contained. Without a CUDA device and
// without the file it exits 77 (ctest SKIP_RETURN_CODE).
//
// Checks (CPU unless noted):
//   1. every row has exactly DEG entries, strictly increasing, all < N (full
//      pass over the 3.6 GB mmap);
//   2. symmetry j ∈ row(i) ⇔ i ∈ row(j) for 10^6 random (row, slot) pairs and
//      for 200 full rows;
//   3. 200 random rows equal a CPU brute-force row (kiss::dot == 16);
//   4. for 10^5 random edges (i,j) and all edges of the 200 full rows:
//      C[i] − C[j] ∈ C (index_of ≥ 0) and neg[j] ∉ row(i);
//   5. file size exact; sha256 of the file equals <data_dir>/adj.sha256;
//   6. (GPU) adjacency_to_device succeeds (3.62 GB cudaMalloc + copy), freed.
// Last line: RESULT ok=<0|1> ... (docs/design.md §2.4).
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "adjacency_cuda.h"
#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"

namespace {

constexpr int kSkip = 77;
using clock_t_ = std::chrono::steady_clock;

double secs(clock_t_::time_point t0) { return std::chrono::duration<double>(clock_t_::now() - t0).count(); }

// C[i] − C[j] as a Vec (entries stay within int8: coordinates are in [-4,4]).
kiss::Vec diff(const kiss::Vec& a, const kiss::Vec& b) {
  kiss::Vec r;
  for (int k = 0; k < kiss::DIM; ++k)
    r[static_cast<std::size_t>(k)] = static_cast<int8_t>(a[static_cast<std::size_t>(k)] - b[static_cast<std::size_t>(k)]);
  return r;
}

// Brute-force neighbour list of vertex i (sorted by construction).
std::vector<uint32_t> brute_row(const kiss::Leech& L, uint32_t i) {
  std::vector<uint32_t> r;
  r.reserve(kiss::DEG);
  const kiss::Vec& a = L.C[i];
  for (uint32_t j = 0; j < static_cast<uint32_t>(kiss::N); ++j)
    if (kiss::dot(a, L.C[j]) == 16) r.push_back(j);
  return r;
}

std::string read_sha_sidecar(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::string tok;
  if (!(in >> tok)) return "";
  return tok;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
  const auto adj_file = data_dir / "adj.u32";
  const auto sha_file = data_dir / "adj.sha256";
  const auto t_all = clock_t_::now();
  int failures = 0;
  auto expect = [&](bool cond, const char* what) {
    if (!cond) {
      ++failures;
      std::printf("FAIL: %s\n", what);
    }
  };

  try {
    // ---- Leech data ------------------------------------------------------
    kiss::Leech L;
    if (std::filesystem::exists(data_dir / "leech_min.i8") && std::filesystem::exists(data_dir / "neg.u32"))
      L = kiss::load_leech(data_dir);
    else
      L = kiss::generate_leech();
    std::printf("leech             : %zu vectors\n", L.C.size());

    // ---- Build the table if needed ----------------------------------------
    const bool have_cuda = kiss::cuda::cuda_available();
    bool present = std::filesystem::exists(adj_file) &&
                   std::filesystem::file_size(adj_file) == kiss::Adjacency::EXPECTED_BYTES;
    int built = 0;
    double build_s = 0;
    if (!present) {
      if (!have_cuda) {
        std::printf("test_adjacency: %s absent and no CUDA device to build it — skipping\n",
                    adj_file.string().c_str());
        std::printf("RESULT ok=1 skipped=1 reason=no_adj_no_cuda\n");
        return kSkip;
      }
      std::printf("adjacency         : %s absent/invalid, building on the GPU ...\n", adj_file.string().c_str());
      const kiss::cuda::BuildAdjacencyStats st = kiss::cuda::build_adjacency_file(L, adj_file);
      build_s = st.build_s;
      built = 1;
      std::printf("build             : chunk_rows=%u upload_s=%.3f build_s=%.3f total_s=%.3f\n",
                  st.chunk_rows, st.upload_s, st.build_s, st.total_s);
      present = true;
    }

    // ---- 5a. file size ------------------------------------------------------
    const std::size_t fsize = std::filesystem::file_size(adj_file);
    std::printf("file size         : %zu bytes (expected %zu)\n", fsize, kiss::Adjacency::EXPECTED_BYTES);
    expect(fsize == kiss::Adjacency::EXPECTED_BYTES, "file size != 3,616,704,000");

    const kiss::Adjacency adj(adj_file);
    expect(adj.rows() == static_cast<std::size_t>(kiss::N), "rows() != N");

    // ---- 1. full pass: DEG strictly increasing entries < N per row ----------
    auto t = clock_t_::now();
    adj.advise_sequential();
    long long bad_rows = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad_rows)
    for (int i = 0; i < kiss::N; ++i) {
      const uint32_t* r = adj.row(static_cast<uint32_t>(i));
      bool ok = r[0] < static_cast<uint32_t>(kiss::N) && r[0] != static_cast<uint32_t>(i);
      for (int k = 1; k < kiss::DEG && ok; ++k)
        ok = r[k] > r[k - 1] && r[k] < static_cast<uint32_t>(kiss::N) && r[k] != static_cast<uint32_t>(i);
      if (!ok) ++bad_rows;
    }
    const double pass_s = secs(t);
    std::printf("full pass         : %d rows, %lld malformed (strictly increasing, < N, != self), %.2f s, %d threads\n",
                kiss::N, bad_rows, pass_s, omp_get_max_threads());
    expect(bad_rows == 0, "some rows are not strictly increasing DEG entries");

    // ---- 2/3/4. randomised checks --------------------------------------------
    std::mt19937_64 rng(20260825);
    std::uniform_int_distribution<uint32_t> pick_v(0, static_cast<uint32_t>(kiss::N - 1));
    std::uniform_int_distribution<uint32_t> pick_k(0, static_cast<uint32_t>(kiss::DEG - 1));

    // 2a. symmetry on 10^6 random (row, slot) pairs
    t = clock_t_::now();
    const int NPAIRS = 1000000;
    std::vector<uint32_t> pi(NPAIRS), pk(NPAIRS);
    for (int n = 0; n < NPAIRS; ++n) { pi[n] = pick_v(rng); pk[n] = pick_k(rng); }
    long long asym_pairs = 0;
#pragma omp parallel for schedule(static) reduction(+ : asym_pairs)
    for (int n = 0; n < NPAIRS; ++n) {
      const uint32_t i = pi[n], j = adj.row(i)[pk[n]];
      if (!adj.adjacent(j, i)) ++asym_pairs;
    }
    std::printf("symmetry (pairs)  : %d random (i, j∈row(i)), %lld with i∉row(j), %.2f s\n", NPAIRS, asym_pairs, secs(t));
    expect(asym_pairs == 0, "symmetry violated on random pairs");

    // 2b/3/4. 200 random full rows: brute force, symmetry, neg, difference in C
    t = clock_t_::now();
    const int NROWS = 200;
    std::vector<uint32_t> rows(NROWS);
    for (int n = 0; n < NROWS; ++n) rows[n] = pick_v(rng);
    long long brute_mismatch = 0, asym_rows = 0, neg_in_row = 0, diff_not_in_C = 0, not_adjacent_full = 0;
#pragma omp parallel for schedule(dynamic) \
    reduction(+ : brute_mismatch, asym_rows, neg_in_row, diff_not_in_C, not_adjacent_full)
    for (int n = 0; n < NROWS; ++n) {
      const uint32_t i = rows[n];
      const uint32_t* r = adj.row(i);
      const std::vector<uint32_t> ref = brute_row(L, i);
      if (ref.size() != static_cast<std::size_t>(kiss::DEG) || !std::equal(ref.begin(), ref.end(), r))
        ++brute_mismatch;
      for (int k = 0; k < kiss::DEG; ++k) {
        const uint32_t j = r[k];
        if (!adj.adjacent(j, i)) ++asym_rows;
        if (adj.adjacent(i, L.neg[j])) ++neg_in_row;
        if (L.index_of(diff(L.C[i], L.C[j])) < 0) ++diff_not_in_C;
        if (!adj.adjacent(i, j)) ++not_adjacent_full;
      }
    }
    std::printf("full rows         : %d random rows: brute-force mismatches=%lld, asymmetric edges=%lld, "
                "neg[j]∈row(i)=%lld, C[i]-C[j]∉C=%lld, adjacent(i,j) false=%lld, %.2f s\n",
                NROWS, brute_mismatch, asym_rows, neg_in_row, diff_not_in_C, not_adjacent_full, secs(t));
    expect(brute_mismatch == 0, "row differs from CPU brute force");
    expect(asym_rows == 0, "symmetry violated on full rows");
    expect(neg_in_row == 0, "neg[j] found in row(i)");
    expect(diff_not_in_C == 0, "C[i]-C[j] not a minimal vector");
    expect(not_adjacent_full == 0, "adjacent(i,j) false for j in row(i)");

    // 4b. 10^5 random edges: difference in C and neg[j] ∉ row(i)
    t = clock_t_::now();
    const int NEDGES = 100000;
    std::vector<uint32_t> ei(NEDGES), ek(NEDGES);
    for (int n = 0; n < NEDGES; ++n) { ei[n] = pick_v(rng); ek[n] = pick_k(rng); }
    long long e_diff_bad = 0, e_neg_bad = 0, e_dot_bad = 0;
#pragma omp parallel for schedule(static) reduction(+ : e_diff_bad, e_neg_bad, e_dot_bad)
    for (int n = 0; n < NEDGES; ++n) {
      const uint32_t i = ei[n], j = adj.row(i)[ek[n]];
      if (L.index_of(diff(L.C[i], L.C[j])) < 0) ++e_diff_bad;
      if (adj.adjacent(i, L.neg[j])) ++e_neg_bad;
      if (kiss::dot(L.C[i], L.C[j]) != 16) ++e_dot_bad;
    }
    std::printf("random edges      : %d edges: C[i]-C[j]∉C=%lld, neg[j]∈row(i)=%lld, dot!=16: %lld, %.2f s\n",
                NEDGES, e_diff_bad, e_neg_bad, e_dot_bad, secs(t));
    expect(e_diff_bad == 0, "C[i]-C[j] not in C on random edges");
    expect(e_neg_bad == 0, "neg[j] in row(i) on random edges");
    expect(e_dot_bad == 0, "dot != 16 on random edges");

    // Non-adjacency sanity: neg[i] and i itself are never adjacent; a few random
    // non-edges have dot != 16.
    long long nonadj_bad = 0;
    for (int n = 0; n < 1000; ++n) {
      const uint32_t i = pick_v(rng), j = pick_v(rng);
      if (adj.adjacent(i, L.neg[i]) || adj.adjacent(i, i)) ++nonadj_bad;
      if (adj.adjacent(i, j) != (kiss::dot(L.C[i], L.C[j]) == 16)) ++nonadj_bad;
    }
    expect(nonadj_bad == 0, "adjacent() disagrees with dot()==16 on random vertex pairs");

    // ---- 5b. sha256 vs the committed sidecar -----------------------------------
    t = clock_t_::now();
    const std::string sha = kiss::sha256_file(adj_file);
    const double sha_s = secs(t);
    const std::string ref_sha = read_sha_sidecar(sha_file);
    int sha_checked = 0;
    if (ref_sha.empty()) {
      std::printf("sha256            : %s  (no %s to compare against — NOT checked)\n", sha.c_str(),
                  sha_file.string().c_str());
    } else {
      sha_checked = 1;
      std::printf("sha256            : %s  (%s: %s) %.1f s\n", sha.c_str(), sha_file.string().c_str(),
                  sha == ref_sha ? "match" : "MISMATCH", sha_s);
      expect(sha == ref_sha, "sha256 of adj.u32 differs from data/adj.sha256");
    }

    // ---- 6. to_device --------------------------------------------------------
    int to_device = -1;   // -1 skipped, 0 failed, 1 ok
    double upload_s = 0;
    if (have_cuda) {
      std::size_t fr0 = 0, to0 = 0, fr1 = 0, fr2 = 0;
      kiss::cuda::device_mem_info(&fr0, &to0);
      t = clock_t_::now();
      try {
        uint32_t* d = kiss::cuda::adjacency_to_device(adj);
        upload_s = secs(t);
        kiss::cuda::device_mem_info(&fr1, nullptr);
        kiss::cuda::adjacency_device_free(d);
        kiss::cuda::device_mem_info(&fr2, nullptr);
        to_device = 1;
        std::printf("to_device         : ok, %.2f s (%.2f GB/s); device free %.2f -> %.2f -> %.2f GiB (total %.2f)\n",
                    upload_s, static_cast<double>(adj.bytes()) / upload_s / 1e9,
                    static_cast<double>(fr0) / (1 << 30), static_cast<double>(fr1) / (1 << 30),
                    static_cast<double>(fr2) / (1 << 30), static_cast<double>(to0) / (1 << 30));
      } catch (const std::exception& e) {
        to_device = 0;
        std::printf("to_device         : FAILED: %s\n", e.what());
      }
      expect(to_device == 1, "adjacency_to_device failed");
    } else {
      std::printf("to_device         : skipped (no CUDA device)\n");
    }

    const bool ok = failures == 0;
    std::printf("RESULT ok=%d rows=%d deg=%d bytes=%zu built=%d build_s=%.3f bad_rows=%lld asym_pairs=%lld "
                "brute_rows=%d brute_mismatch=%lld neg_in_row=%lld diff_not_in_C=%lld sha256=%s sha_checked=%d "
                "to_device=%d upload_s=%.2f pass_s=%.2f total_s=%.1f failures=%d\n",
                ok ? 1 : 0, kiss::N, kiss::DEG, fsize, built, build_s, bad_rows, asym_pairs, NROWS, brute_mismatch,
                neg_in_row + e_neg_bad, diff_not_in_C + e_diff_bad, sha.c_str(), sha_checked, to_device, upload_s,
                pass_s, secs(t_all), failures);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\nRESULT ok=0 failures=%d\n", e.what(), failures + 1);
    return 1;
  }
}

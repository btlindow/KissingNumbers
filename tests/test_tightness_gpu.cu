// T1.5 — acceptance test for kiss::cuda::tightness_full / count_free.
//
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) if no CUDA device, 1 on
// failure. Last line: RESULT ok=<0|1> ... (docs/design.md §2.4).
//
// Fixtures
//   * the 496 (data/S496.txt, produced by T1.3; polled for --wait-s496 <sec>,
//     default 0) — fallback: a greedy random maximal independent set built here;
//   * 65 random S with sizes cycling through {1,2,7,64,255,256,500,1000,1024}
//     plus 3 special chains (|S|=0; duplicates; 1024 neighbours of one vertex),
//     launched as batches of B = 1, 3 and 64 chains (a different S per chain).
// Every chain is compared with a CPU reference (plain double loop over
// kiss::dot, OpenMP) — exact equality on all N entries — and count_free is
// compared with the host count. Throughput is measured with cudaEvents for
// B=64, |S|=500 (upload excluded).
//
// Usage: test_tightness_gpu [--data DIR] [--wait-s496 SEC] [--bench-iters N] [--no-bench]
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss_cuda.h"

using kiss::Leech;
using kiss::N;
using kiss::Vec;
using kiss::cuda::INS_WORDS;
using kiss::cuda::SMAX;

namespace {

constexpr int kSkip = 77;
int g_failures = 0;

void fail(const std::string& msg) {
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

// ---- CPU reference (deliberately naive; independent of kiss/verify.h) ------
std::vector<uint16_t> tightness_ref(const Leech& L, const std::vector<uint32_t>& S) {
  std::vector<Vec> SV(S.size());
  for (std::size_t i = 0; i < S.size(); ++i) SV[i] = L.C[S[i]];
  std::vector<uint16_t> t(static_cast<std::size_t>(N));
#pragma omp parallel for schedule(static)
  for (int v = 0; v < N; ++v) {
    int c = 0;
    const Vec& x = L.C[static_cast<std::size_t>(v)];
    for (const Vec& s : SV) c += (kiss::dot(x, s) == 16);
    t[static_cast<std::size_t>(v)] = static_cast<uint16_t>(c);
  }
  return t;
}

std::vector<uint32_t> in_s_bitmap(const std::vector<uint32_t>& S) {
  std::vector<uint32_t> bits(static_cast<std::size_t>(INS_WORDS), 0u);
  for (uint32_t v : S) bits[v >> 5] |= 1u << (v & 31);
  return bits;
}

uint32_t count_free_ref(const std::vector<uint16_t>& t, const std::vector<uint32_t>& bits) {
  uint32_t c = 0;
  for (int v = 0; v < N; ++v)
    c += (t[static_cast<std::size_t>(v)] == 0 &&
          !((bits[static_cast<std::size_t>(v >> 5)] >> (v & 31)) & 1u));
  return c;
}

// ---- one batched launch + verification --------------------------------------
struct BatchResult {
  std::vector<std::vector<uint16_t>> tight;  // [B][N]
  std::vector<uint32_t> free_gpu;            // [B]
  float gpu_ms = 0.f;                        // tightness_full only
};

BatchResult run_batch(const kiss::cuda::DeviceLeech& dl, const std::vector<std::vector<uint32_t>>& sets,
                      bool with_count_free = true) {
  const int B = static_cast<int>(sets.size());
  std::vector<uint32_t> hS(static_cast<std::size_t>(B) * SMAX, 0xffffffffu);  // poison unused slots
  std::vector<uint32_t> hSize(static_cast<std::size_t>(B));
  std::vector<uint32_t> hBits(static_cast<std::size_t>(B) * INS_WORDS);
  for (int b = 0; b < B; ++b) {
    const auto& S = sets[static_cast<std::size_t>(b)];
    if (S.size() > static_cast<std::size_t>(SMAX)) throw std::runtime_error("set too large");
    std::copy(S.begin(), S.end(), hS.begin() + static_cast<std::ptrdiff_t>(b) * SMAX);
    hSize[static_cast<std::size_t>(b)] = static_cast<uint32_t>(S.size());
    const auto bits = in_s_bitmap(S);
    std::copy(bits.begin(), bits.end(), hBits.begin() + static_cast<std::ptrdiff_t>(b) * INS_WORDS);
  }
  uint32_t *dS = nullptr, *dSize = nullptr, *dBits = nullptr, *dFree = nullptr;
  uint16_t* dT = nullptr;
  KISS_CUDA_CHECK(cudaMalloc(&dS, hS.size() * sizeof(uint32_t)));
  KISS_CUDA_CHECK(cudaMalloc(&dSize, hSize.size() * sizeof(uint32_t)));
  KISS_CUDA_CHECK(cudaMalloc(&dBits, hBits.size() * sizeof(uint32_t)));
  KISS_CUDA_CHECK(cudaMalloc(&dFree, static_cast<std::size_t>(B) * sizeof(uint32_t)));
  KISS_CUDA_CHECK(cudaMalloc(&dT, static_cast<std::size_t>(B) * N * sizeof(uint16_t)));
  KISS_CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  KISS_CUDA_CHECK(cudaMemcpy(dSize, hSize.data(), hSize.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  KISS_CUDA_CHECK(cudaMemcpy(dBits, hBits.data(), hBits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  KISS_CUDA_CHECK(cudaMemset(dT, 0xff, static_cast<std::size_t>(B) * N * sizeof(uint16_t)));  // poison

  cudaEvent_t e0, e1;
  KISS_CUDA_CHECK(cudaEventCreate(&e0));
  KISS_CUDA_CHECK(cudaEventCreate(&e1));
  KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
  kiss::cuda::tightness_full(dl, dS, dSize, B, dT, 0);
  KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
  if (with_count_free) kiss::cuda::count_free(dT, dBits, B, dFree, 0);
  KISS_CUDA_CHECK(cudaDeviceSynchronize());

  BatchResult r;
  KISS_CUDA_CHECK(cudaEventElapsedTime(&r.gpu_ms, e0, e1));
  std::vector<uint16_t> hT(static_cast<std::size_t>(B) * N);
  KISS_CUDA_CHECK(cudaMemcpy(hT.data(), dT, hT.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  r.tight.resize(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b)
    r.tight[static_cast<std::size_t>(b)].assign(hT.begin() + static_cast<std::ptrdiff_t>(b) * N,
                                                hT.begin() + static_cast<std::ptrdiff_t>(b + 1) * N);
  r.free_gpu.assign(static_cast<std::size_t>(B), 0u);
  if (with_count_free)
    KISS_CUDA_CHECK(cudaMemcpy(r.free_gpu.data(), dFree, static_cast<std::size_t>(B) * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));
  KISS_CUDA_CHECK(cudaEventDestroy(e0));
  KISS_CUDA_CHECK(cudaEventDestroy(e1));
  KISS_CUDA_CHECK(cudaFree(dS));
  KISS_CUDA_CHECK(cudaFree(dSize));
  KISS_CUDA_CHECK(cudaFree(dBits));
  KISS_CUDA_CHECK(cudaFree(dFree));
  KISS_CUDA_CHECK(cudaFree(dT));
  return r;
}

// Compare one chain with the CPU reference; returns number of mismatching entries.
long long verify_chain(const Leech& L, const std::string& name, const std::vector<uint32_t>& S,
                       const std::vector<uint16_t>& gpu, uint32_t free_gpu, double* cpu_ms) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<uint16_t> ref = tightness_ref(L, S);
  *cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  long long bad = 0;
  int first = -1;
  for (int v = 0; v < N; ++v)
    if (ref[static_cast<std::size_t>(v)] != gpu[static_cast<std::size_t>(v)]) {
      if (first < 0) first = v;
      ++bad;
    }
  if (bad)
    fail(name + ": |S|=" + std::to_string(S.size()) + " " + std::to_string(bad) +
         " mismatches, first at v=" + std::to_string(first) + " cpu=" +
         std::to_string(ref[static_cast<std::size_t>(first)]) + " gpu=" +
         std::to_string(gpu[static_cast<std::size_t>(first)]));
  const uint32_t free_ref = count_free_ref(ref, in_s_bitmap(S));
  if (free_ref != free_gpu)
    fail(name + ": count_free cpu=" + std::to_string(free_ref) + " gpu=" + std::to_string(free_gpu));
  return bad;
}

// Greedy random maximal independent set (fallback fixture).
std::vector<uint32_t> greedy_mis(const Leech& L, std::mt19937_64& rng) {
  std::vector<uint32_t> order(static_cast<std::size_t>(N));
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint32_t> S;
  std::vector<Vec> SV;
  for (uint32_t v : order) {
    const Vec& x = L.C[v];
    bool ok = true;
    for (const Vec& s : SV)
      if (kiss::dot(x, s) == 16) { ok = false; break; }
    if (ok) { S.push_back(v); SV.push_back(x); }
  }
  return S;
}

std::vector<uint32_t> random_set(std::mt19937_64& rng, int size) {
  std::uniform_int_distribution<uint32_t> d(0, N - 1);
  std::vector<uint32_t> S(static_cast<std::size_t>(size));
  for (auto& v : S) v = d(rng);
  return S;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  int wait_s496 = 0, bench_iters = 5;
  bool bench = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--wait-s496") && i + 1 < argc) wait_s496 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bench-iters") && i + 1 < argc) bench_iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-bench")) bench = false;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }

  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::printf("test_tightness_gpu: no CUDA device (%s) — skipping\n", cudaGetErrorString(e));
    std::printf("RESULT ok=1 skipped=1 devices=0\n");
    return kSkip;
  }

  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    std::mt19937_64 rng(20260825ull);
    double cpu_ms = 0;

    // Warm-up: the first tightness_full call pays for lazy module loading and
    // the cudaMallocAsync pool (~6 ms one-off); do it on an empty chain so the
    // timings below are steady-state. Also exercises the |S| = 0 path.
    {
      const BatchResult r = run_batch(dl, {std::vector<uint32_t>{}});
      verify_chain(L, "warm-up empty chain", {}, r.tight[0], r.free_gpu[0], &cpu_ms);
      std::printf("warm-up         : first call %.3f ms (B=1, |S|=0)\n", r.gpu_ms);
    }

    // ---- fixture: the 496 (or greedy MIS fallback) -------------------------
    const std::filesystem::path p496 = data_dir / "S496.txt";
    {
      const auto t0 = std::chrono::steady_clock::now();
      while (!std::filesystem::exists(p496)) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= wait_s496) break;
        if (static_cast<int>(el) % 30 == 0)
          std::printf("waiting for %s (%.0f/%d s)\n", p496.c_str(), el, wait_s496);
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
    }
    std::string fixture;
    std::vector<uint32_t> S496;
    if (std::filesystem::exists(p496)) {
      const std::vector<Vec> rows = kiss::read_set(p496);
      int missing = 0;
      for (const Vec& r : rows) {
        const int32_t idx = L.index_of(r);
        if (idx < 0) ++missing; else S496.push_back(static_cast<uint32_t>(idx));
      }
      if (missing) fail(p496.string() + ": " + std::to_string(missing) + " rows are not minimal vectors");
      fixture = "S496.txt";
    } else {
      S496 = greedy_mis(L, rng);
      fixture = "greedy_mis";
    }
    {
      long long conflicts = 0, dup = 0;
      for (std::size_t i = 0; i < S496.size(); ++i)
        for (std::size_t j = i + 1; j < S496.size(); ++j) {
          if (S496[i] == S496[j]) ++dup;
          else if (kiss::dot(L.C[S496[i]], L.C[S496[j]]) > 8) ++conflicts;
        }
      std::printf("fixture         : %s |S|=%zu conflicts=%lld duplicates=%lld\n", fixture.c_str(),
                  S496.size(), conflicts, dup);
      if (conflicts || dup) fail("fixture is not an independent set");
    }
    std::map<int, long long> hist;
    uint32_t free496 = 0;
    {
      const BatchResult r = run_batch(dl, {S496});
      verify_chain(L, fixture, S496, r.tight[0], r.free_gpu[0], &cpu_ms);
      free496 = r.free_gpu[0];
      for (int v = 0; v < N; ++v) ++hist[r.tight[0][static_cast<std::size_t>(v)]];
      std::printf("fixture GPU time: %.3f ms (B=1, |S|=%zu)\n", r.gpu_ms, S496.size());
      std::printf("fixture tightness histogram (value:count):");
      for (const auto& kv : hist) std::printf(" %d:%lld", kv.first, kv.second);
      std::printf("\n");
      std::printf("fixture free    : tight==0 && !inS = %u (expect 0 for a maximal set)\n", free496);
      if (free496 != 0) fail("fixture is not maximal (free vertices exist)");
    }

    // ---- random batches: B = 1, 3, 64 ---------------------------------------
    const int sizes[] = {1, 2, 7, 64, 255, 256, 500, 1000, 1024};
    std::vector<std::vector<uint32_t>> all_sets;
    for (int j = 0; j < 68; ++j) all_sets.push_back(random_set(rng, sizes[j % 9]));
    // special chains at the end of the B=64 batch
    all_sets[65].clear();                                  // |S| = 0
    {                                                      // duplicates: 250 vertices twice
      std::vector<uint32_t> d = random_set(rng, 250);
      d.insert(d.end(), d.begin(), d.end());
      all_sets[66] = d;
    }
    {                                                      // 1024 neighbours of one vertex → tight[u] = 1024
      const uint32_t u = std::uniform_int_distribution<uint32_t>(0, N - 1)(rng);
      std::vector<uint32_t> nb;
      for (int v = 0; v < N && static_cast<int>(nb.size()) < SMAX; ++v)
        if (kiss::dot(L.C[u], L.C[static_cast<std::size_t>(v)]) == 16) nb.push_back(static_cast<uint32_t>(v));
      all_sets[67] = nb;
      std::printf("special chain   : u=%u with %zu neighbours in S\n", u, nb.size());
    }
    const int batch_sizes[] = {1, 3, 64};
    long long total_bad = 0;
    int chains = 0;
    std::size_t off = 0;
    int max_tight = 0;
    for (int B : batch_sizes) {
      std::vector<std::vector<uint32_t>> sets(all_sets.begin() + static_cast<std::ptrdiff_t>(off),
                                              all_sets.begin() + static_cast<std::ptrdiff_t>(off + B));
      off += static_cast<std::size_t>(B);
      const BatchResult r = run_batch(dl, sets);
      long long pairs = 0;
      for (int b = 0; b < B; ++b) {
        const auto& S = sets[static_cast<std::size_t>(b)];
        pairs += static_cast<long long>(N) * static_cast<long long>(S.size());
        total_bad += verify_chain(L, "B=" + std::to_string(B) + " chain " + std::to_string(b), S,
                                  r.tight[static_cast<std::size_t>(b)], r.free_gpu[static_cast<std::size_t>(b)],
                                  &cpu_ms);
        for (int v = 0; v < N; ++v)
          max_tight = std::max(max_tight, static_cast<int>(r.tight[static_cast<std::size_t>(b)][static_cast<std::size_t>(v)]));
        ++chains;
      }
      std::printf("batch B=%-2d      : %.3f ms GPU, %.3e pairs, %.3e pairs/s, mismatches so far %lld\n", B,
                  r.gpu_ms, static_cast<double>(pairs), static_cast<double>(pairs) / (r.gpu_ms * 1e-3),
                  total_bad);
    }
    std::printf("max tightness seen in random batches: %d (expect 1024 from the special chain)\n", max_tight);
    if (max_tight != SMAX) fail("special chain did not reach tightness 1024");

    // ---- throughput: B=64, |S|=500 -------------------------------------------
    double best_ms = 0, mean_ms = 0, pairs_per_s = 0;
    if (bench) {
      const int B = 64, sz = 500;
      std::vector<std::vector<uint32_t>> sets;
      for (int b = 0; b < B; ++b) sets.push_back(random_set(rng, sz));
      std::vector<uint32_t> hS(static_cast<std::size_t>(B) * SMAX, 0u), hSize(static_cast<std::size_t>(B), sz);
      for (int b = 0; b < B; ++b)
        std::copy(sets[static_cast<std::size_t>(b)].begin(), sets[static_cast<std::size_t>(b)].end(),
                  hS.begin() + static_cast<std::ptrdiff_t>(b) * SMAX);
      uint32_t *dS = nullptr, *dSize = nullptr;
      uint16_t* dT = nullptr;
      KISS_CUDA_CHECK(cudaMalloc(&dS, hS.size() * sizeof(uint32_t)));
      KISS_CUDA_CHECK(cudaMalloc(&dSize, hSize.size() * sizeof(uint32_t)));
      KISS_CUDA_CHECK(cudaMalloc(&dT, static_cast<std::size_t>(B) * N * sizeof(uint16_t)));
      KISS_CUDA_CHECK(cudaMemcpy(dS, hS.data(), hS.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
      KISS_CUDA_CHECK(cudaMemcpy(dSize, hSize.data(), hSize.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
      cudaEvent_t e0, e1;
      KISS_CUDA_CHECK(cudaEventCreate(&e0));
      KISS_CUDA_CHECK(cudaEventCreate(&e1));
      kiss::cuda::tightness_full(dl, dS, dSize, B, dT, 0);  // warm-up
      KISS_CUDA_CHECK(cudaDeviceSynchronize());
      best_ms = 1e30;
      for (int it = 0; it < bench_iters; ++it) {
        KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
        kiss::cuda::tightness_full(dl, dS, dSize, B, dT, 0);
        KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
        KISS_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        KISS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best_ms = std::min(best_ms, static_cast<double>(ms));
        mean_ms += ms;
      }
      mean_ms /= std::max(1, bench_iters);
      const double pairs = static_cast<double>(B) * N * sz;
      pairs_per_s = pairs / (best_ms * 1e-3);
      std::printf("bench B=%d |S|=%d: best %.3f ms, mean %.3f ms over %d iters → %.3e pairs/s (%.3e dp4a/s)\n",
                  B, sz, best_ms, mean_ms, bench_iters, pairs_per_s, 6.0 * pairs_per_s);
      // spot-check the benchmark output on one chain (cheap insurance)
      std::vector<uint16_t> hT(static_cast<std::size_t>(N));
      KISS_CUDA_CHECK(cudaMemcpy(hT.data(), dT + static_cast<std::ptrdiff_t>(B - 1) * N, hT.size() * sizeof(uint16_t),
                                 cudaMemcpyDeviceToHost));
      total_bad += verify_chain(L, "bench chain 63", sets.back(), hT, count_free_ref(hT, in_s_bitmap(sets.back())),
                                &cpu_ms);
      KISS_CUDA_CHECK(cudaEventDestroy(e0));
      KISS_CUDA_CHECK(cudaEventDestroy(e1));
      KISS_CUDA_CHECK(cudaFree(dS));
      KISS_CUDA_CHECK(cudaFree(dSize));
      KISS_CUDA_CHECK(cudaFree(dT));
    }

    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    const bool ok = (g_failures == 0);
    std::printf("RESULT ok=%d fixture=%s fixture_size=%zu fixture_free=%u chains=%d mismatches=%lld "
                "max_tight=%d bench_best_ms=%.3f bench_pairs_per_s=%.3e cpu_ref_ms=%.0f omp_threads=%d "
                "total_ms=%.0f failures=%d\n",
                ok ? 1 : 0, fixture.c_str(), S496.size(), free496, chains + 1, total_bad, max_tight, best_ms,
                pairs_per_s, cpu_ms, omp_get_max_threads(), total_ms, g_failures);
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "test_tightness_gpu: exception: %s\n", ex.what());
    std::printf("RESULT ok=0 exception=1\n");
    return 1;
  }
}

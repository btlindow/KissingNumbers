// T2.3 — acceptance test for kiss::cuda::triple_class_histogram.
//
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) if no CUDA device, 1 on
// failure. Last line: RESULT ok=<0|1> ... (docs/design.md §2.4).
//
// For x = C[0] and y = one representative per class(x, y) (7 passes):
//   1. full pass over all N z's: every histogram sums to N, and the three
//      one-point marginals equal the class sizes of x, y and z respectively;
//   2. 16 random z's plus the special z's {x, y, -x, -y} per pass: the
//      343-histogram from the kernel == a CPU double loop over kiss::dot
//      with an explicit class lookup, exactly;
//   3. the sub-range entry point (z0, nz) reproduces the full pass.
//
// Usage: test_triple_gpu [--data DIR] [--x IDX] [--samples K]
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss_cuda.h"
#include "triple_cuda.h"

using kiss::Leech;
using kiss::N;
using kiss::cuda::NCLASS3;

namespace {

constexpr int kSkip = 77;
int g_failures = 0;

void fail(const std::string& msg) {
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

using Hist = std::array<uint32_t, NCLASS3>;

int class_ref(int d) {
  static const int dots[7] = {-32, -16, -8, 0, 8, 16, 32};
  for (int c = 0; c < 7; ++c)
    if (dots[c] == d) return c;
  throw std::runtime_error("inner product " + std::to_string(d) + " is not a class");
}

Hist cpu_hist(const Leech& L, uint32_t x, uint32_t y, uint32_t z) {
  Hist h{};
  for (int w = 0; w < N; ++w) {
    const auto& W = L.C[static_cast<std::size_t>(w)];
    const int a = class_ref(kiss::dot(L.C[x], W));
    const int b = class_ref(kiss::dot(L.C[y], W));
    const int c = class_ref(kiss::dot(L.C[z], W));
    ++h[static_cast<std::size_t>(a * 49 + b * 7 + c)];
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  uint32_t x = 0;
  int samples = 16;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--x") && i + 1 < argc) x = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--samples") && i + 1 < argc) samples = std::atoi(argv[++i]);
  }
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::printf("no CUDA device: skipping\nRESULT ok=0 skipped=1\n");
    return kSkip;
  }
  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    if (x >= static_cast<uint32_t>(N)) throw std::runtime_error("x out of range");

    std::array<uint32_t, 7> v{};
    std::vector<uint8_t> cls_x(static_cast<std::size_t>(N));
    std::array<uint32_t, 7> rep{};
    std::array<bool, 7> have{};
    for (int y = 0; y < N; ++y) {
      const int c = class_ref(kiss::dot(L.C[x], L.C[static_cast<std::size_t>(y)]));
      cls_x[static_cast<std::size_t>(y)] = static_cast<uint8_t>(c);
      ++v[static_cast<std::size_t>(c)];
      if (!have[static_cast<std::size_t>(c)]) { have[static_cast<std::size_t>(c)] = true; rep[static_cast<std::size_t>(c)] = static_cast<uint32_t>(y); }
    }
    std::printf("x=%u valencies:", x);
    for (int c = 0; c < 7; ++c) std::printf(" %u", v[static_cast<std::size_t>(c)]);
    std::printf("\n");

    uint32_t* dM = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(N) * NCLASS3 * sizeof(uint32_t);
    KISS_CUDA_CHECK(cudaMalloc(&dM, bytes));
    std::vector<uint32_t> hM(static_cast<std::size_t>(N) * NCLASS3);
    std::mt19937_64 rng(20260826ull);
    std::uniform_int_distribution<uint32_t> dist(0, N - 1);

    long long bad_sum = 0, bad_marg = 0, mismatches = 0, checked = 0, bad_sub = 0;
    double gpu_ms_total = 0, cpu_ms_total = 0;
    for (int i = 0; i < 7; ++i) {
      const uint32_t y = rep[static_cast<std::size_t>(i)];
      cudaEvent_t e0, e1;
      KISS_CUDA_CHECK(cudaEventCreate(&e0));
      KISS_CUDA_CHECK(cudaEventCreate(&e1));
      KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
      kiss::cuda::triple_class_histogram_all(dl, x, y, dM, 0);
      KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
      KISS_CUDA_CHECK(cudaDeviceSynchronize());
      float ms = 0;
      KISS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      gpu_ms_total += ms;
      KISS_CUDA_CHECK(cudaEventDestroy(e0));
      KISS_CUDA_CHECK(cudaEventDestroy(e1));
      KISS_CUDA_CHECK(cudaMemcpy(hM.data(), dM, bytes, cudaMemcpyDeviceToHost));

      // 1. sums and marginals for all z
      std::vector<uint8_t> cls_y(static_cast<std::size_t>(N));
      for (int w = 0; w < N; ++w) cls_y[static_cast<std::size_t>(w)] = static_cast<uint8_t>(class_ref(kiss::dot(L.C[y], L.C[static_cast<std::size_t>(w)])));
      for (int z = 0; z < N; ++z) {
        const uint32_t* h = hM.data() + static_cast<std::size_t>(z) * NCLASS3;
        uint64_t tot = 0;
        std::array<uint64_t, 7> ma{}, mb{}, mc{};
        for (int a = 0; a < 7; ++a)
          for (int b = 0; b < 7; ++b)
            for (int c = 0; c < 7; ++c) {
              const uint32_t u = h[a * 49 + b * 7 + c];
              tot += u; ma[static_cast<std::size_t>(a)] += u; mb[static_cast<std::size_t>(b)] += u; mc[static_cast<std::size_t>(c)] += u;
            }
        if (tot != static_cast<uint64_t>(N)) ++bad_sum;
        for (int c = 0; c < 7; ++c)
          if (ma[static_cast<std::size_t>(c)] != v[static_cast<std::size_t>(c)] || mb[static_cast<std::size_t>(c)] != v[static_cast<std::size_t>(c)] || mc[static_cast<std::size_t>(c)] != v[static_cast<std::size_t>(c)]) { ++bad_marg; break; }
      }

      // 2. CPU reference on samples
      std::vector<uint32_t> zs{x, y, L.neg[x], L.neg[y]};
      for (int s = 0; s < samples; ++s) zs.push_back(dist(rng));
      const auto c0 = std::chrono::steady_clock::now();
      for (uint32_t z : zs) {
        const Hist ref = cpu_hist(L, x, y, z);
        ++checked;
        if (!std::equal(ref.begin(), ref.end(), hM.data() + static_cast<std::size_t>(z) * NCLASS3)) {
          ++mismatches;
          fail("kernel != CPU for x=" + std::to_string(x) + " y=" + std::to_string(y) + " z=" + std::to_string(z));
        }
      }
      cpu_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

      // 3. sub-range
      if (i == 3) {
        const uint32_t z0 = 123456, nz = 777;
        std::vector<uint32_t> sub(static_cast<std::size_t>(nz) * NCLASS3);
        uint32_t* dS = nullptr;
        KISS_CUDA_CHECK(cudaMalloc(&dS, sub.size() * sizeof(uint32_t)));
        kiss::cuda::triple_class_histogram(dl, x, y, z0, nz, dS, 0);
        KISS_CUDA_CHECK(cudaMemcpy(sub.data(), dS, sub.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        KISS_CUDA_CHECK(cudaFree(dS));
        for (uint32_t t = 0; t < nz; ++t)
          if (!std::equal(sub.begin() + static_cast<std::ptrdiff_t>(t) * NCLASS3, sub.begin() + static_cast<std::ptrdiff_t>(t + 1) * NCLASS3,
                          hM.begin() + static_cast<std::ptrdiff_t>(z0 + t) * NCLASS3)) ++bad_sub;
        if (bad_sub) fail("sub-range differs from the full pass in " + std::to_string(bad_sub) + " rows");
      }
      std::printf("class %d y=%u: gpu %.1f ms, samples %zu\n", i, y, ms, zs.size());
    }
    KISS_CUDA_CHECK(cudaFree(dM));
    if (bad_sum) fail(std::to_string(bad_sum) + " histograms do not sum to N");
    if (bad_marg) fail(std::to_string(bad_marg) + " histograms have wrong one-point marginals");

    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("full passes: 7, %.1f ms GPU total (%.3e pairs/s); histogram sums bad %lld, marginals bad %lld; "
                "CPU reference %lld z's, %lld mismatches (%.0f ms CPU); sub-range bad rows %lld\n",
                gpu_ms_total, 7.0 * N * N / (gpu_ms_total * 1e-3), bad_sum, bad_marg, checked, mismatches, cpu_ms_total, bad_sub);
    std::printf("RESULT ok=%d x=%u passes=7 gpu_ms=%.1f samples=%lld mismatches=%lld bad_sum=%lld bad_marg=%lld bad_sub=%lld total_ms=%.0f failures=%d\n",
                g_failures == 0 ? 1 : 0, x, gpu_ms_total, checked, mismatches, bad_sum, bad_marg, bad_sub, total_ms, g_failures);
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "test_triple_gpu: %s\n", ex.what());
    std::printf("RESULT ok=0 error=1\n");
    return 1;
  }
}

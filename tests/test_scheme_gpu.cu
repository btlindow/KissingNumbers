// T2.2 — acceptance test for kiss::cuda::pair_class_histogram.
//
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) if no CUDA device, 1 on
// failure. Last line: RESULT ok=<0|1> ... (docs/design.md §2.4).
//
// Checks (x = C[0] by default, --x IDX to change):
//   1. full pass over all N y's: every row sum  sum_j M_y[i][j] = v_i
//      (the class sizes of x) and every column sum sum_i M_y[i][j] = v_j
//      (the class sizes of y; equal to those of x by the histogram of
//      README §1.2 which is checked for x on the host) — for ALL y;
//   2. 64 random y plus the special y's {x, -x, first y of each class}:
//      M_y from the kernel == a CPU double loop over kiss::dot, exactly;
//   3. the sub-range entry point (y0, ny) reproduces the full pass;
//   4. informational: number of distinct M_y per class(x,y) (the scheme
//      property; asserted by tools/scheme_numbers, only printed here).
//
// Usage: test_scheme_gpu [--data DIR] [--x IDX] [--samples K]
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss_cuda.h"
#include "scheme_cuda.h"

using kiss::Leech;
using kiss::N;
using kiss::Vec;
using kiss::cuda::NCLASS;
using kiss::cuda::NCLASS2;

namespace {

constexpr int kSkip = 77;
int g_failures = 0;

void fail(const std::string& msg) {
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

using Mat = std::array<uint32_t, NCLASS2>;

// Class of an inner product, spelled out (independent of kiss::ip_class and
// of the kernel's arithmetic): position of d in {-32,-16,-8,0,8,16,32}.
int class_ref(int d) {
  static const int dots[NCLASS] = {-32, -16, -8, 0, 8, 16, 32};
  for (int c = 0; c < NCLASS; ++c)
    if (dots[c] == d) return c;
  throw std::runtime_error("class_ref: inner product " + std::to_string(d) + " is not a class value");
}

// CPU reference: M[i*7+j] = #{ z : class(x,z) = i, class(z,y) = j } by a plain
// double loop over kiss::dot with a scalar inner product (no packing, no dp4a).
Mat histogram_ref(const Leech& L, uint32_t x, uint32_t y) {
  Mat M{};
  const Vec& X = L.C[x];
  const Vec& Y = L.C[y];
  for (int z = 0; z < N; ++z) {
    const Vec& Z = L.C[static_cast<std::size_t>(z)];
    const int i = class_ref(kiss::dot(X, Z));
    const int j = class_ref(kiss::dot(Z, Y));
    ++M[static_cast<std::size_t>(i * NCLASS + j)];
  }
  return M;
}

std::string mat_str(const Mat& M) {
  std::string s;
  for (int i = 0; i < NCLASS; ++i) {
    s += "[";
    for (int j = 0; j < NCLASS; ++j) s += std::to_string(M[static_cast<std::size_t>(i * NCLASS + j)]) + (j + 1 < NCLASS ? " " : "");
    s += "]";
  }
  return s;
}

std::vector<uint32_t> run_range(const kiss::cuda::DeviceLeech& dl, uint32_t x, uint32_t y0, uint32_t ny,
                                float* ms) {
  uint32_t* dM = nullptr;
  KISS_CUDA_CHECK(cudaMalloc(&dM, static_cast<std::size_t>(ny) * NCLASS2 * sizeof(uint32_t)));
  KISS_CUDA_CHECK(cudaMemset(dM, 0xff, static_cast<std::size_t>(ny) * NCLASS2 * sizeof(uint32_t)));
  cudaEvent_t e0, e1;
  KISS_CUDA_CHECK(cudaEventCreate(&e0));
  KISS_CUDA_CHECK(cudaEventCreate(&e1));
  KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
  kiss::cuda::pair_class_histogram(dl, x, y0, ny, dM, 0);
  KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
  KISS_CUDA_CHECK(cudaDeviceSynchronize());
  KISS_CUDA_CHECK(cudaEventElapsedTime(ms, e0, e1));
  std::vector<uint32_t> h(static_cast<std::size_t>(ny) * NCLASS2);
  KISS_CUDA_CHECK(cudaMemcpy(h.data(), dM, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  KISS_CUDA_CHECK(cudaEventDestroy(e0));
  KISS_CUDA_CHECK(cudaEventDestroy(e1));
  KISS_CUDA_CHECK(cudaFree(dM));
  return h;
}

Mat row_of(const std::vector<uint32_t>& all, uint32_t y) {
  Mat M;
  std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(y) * NCLASS2, NCLASS2, M.begin());
  return M;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  uint32_t x = 0;
  int samples = 64;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--x") && i + 1 < argc) x = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--samples") && i + 1 < argc) samples = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }

  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::printf("test_scheme_gpu: no CUDA device (%s) — skipping\n", cudaGetErrorString(e));
    std::printf("RESULT ok=1 skipped=1 devices=0\n");
    return kSkip;
  }

  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    if (x >= static_cast<uint32_t>(N)) { std::fprintf(stderr, "x out of range\n"); return 1; }

    // Host: class of (x, y) for all y, valencies v_c, first y of each class.
    std::vector<uint8_t> cls(static_cast<std::size_t>(N));
    std::array<uint32_t, NCLASS> val{};
    std::array<int64_t, NCLASS> first_y;
    first_y.fill(-1);
    for (int y = 0; y < N; ++y) {
      const int d = kiss::dot(L.C[x], L.C[static_cast<std::size_t>(y)]);
      const int c = kiss::ip_class(d);
      if (c < 0) { fail("inner product " + std::to_string(d) + " not a class value"); continue; }
      cls[static_cast<std::size_t>(y)] = static_cast<uint8_t>(c);
      ++val[static_cast<std::size_t>(c)];
      if (first_y[static_cast<std::size_t>(c)] < 0) first_y[static_cast<std::size_t>(c)] = y;
    }
    {
      const std::array<uint32_t, NCLASS> expect{1, 4600, 47104, 93150, 47104, 4600, 1};
      std::printf("valencies of x=%u :", x);
      for (int c = 0; c < NCLASS; ++c) std::printf(" %u", val[static_cast<std::size_t>(c)]);
      std::printf("\n");
      if (val != expect) fail("valencies differ from README section 1.2");
    }

    // Warm-up (module load) on a tiny range, then the full pass.
    float ms_warm = 0.f, ms_full = 0.f, ms_range = 0.f;
    (void)run_range(dl, x, 0, 8, &ms_warm);
    std::printf("warm-up         : %.3f ms (ny=8)\n", ms_warm);
    const std::vector<uint32_t> full = run_range(dl, x, 0, static_cast<uint32_t>(N), &ms_full);
    const double pairs = static_cast<double>(N) * static_cast<double>(N);
    std::printf("full pass       : %.1f ms for %.3e pairs -> %.3e pairs/s\n", ms_full, pairs,
                pairs / (ms_full * 1e-3));

    // 1. row/column sums for all y.
    {
      long long bad_rows = 0, bad_cols = 0;
      for (int y = 0; y < N; ++y) {
        const uint32_t* M = full.data() + static_cast<std::ptrdiff_t>(y) * NCLASS2;
        for (int i = 0; i < NCLASS; ++i) {
          uint64_t rs = 0, cs = 0;
          for (int j = 0; j < NCLASS; ++j) { rs += M[i * NCLASS + j]; cs += M[j * NCLASS + i]; }
          if (rs != val[static_cast<std::size_t>(i)]) ++bad_rows;
          if (cs != val[static_cast<std::size_t>(i)]) ++bad_cols;
        }
      }
      std::printf("row/col sums    : bad rows %lld, bad cols %lld (over all %d y)\n", bad_rows, bad_cols, N);
      if (bad_rows || bad_cols) fail("row/column sums do not equal the valencies");
    }

    // 2. CPU reference on a sample.
    std::vector<uint32_t> ys;
    ys.push_back(x);
    ys.push_back(L.neg[x]);
    for (int c = 0; c < NCLASS; ++c)
      if (first_y[static_cast<std::size_t>(c)] >= 0) ys.push_back(static_cast<uint32_t>(first_y[static_cast<std::size_t>(c)]));
    {
      std::mt19937_64 rng(20260825ull);
      std::uniform_int_distribution<uint32_t> d(0, N - 1);
      for (int s = 0; s < samples; ++s) ys.push_back(d(rng));
    }
    long long mismatches = 0;
    double cpu_ms = 0;
    for (uint32_t y : ys) {
      const auto t0 = std::chrono::steady_clock::now();
      const Mat ref = histogram_ref(L, x, y);
      cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      const Mat gpu = row_of(full, y);
      if (ref != gpu) {
        ++mismatches;
        fail("y=" + std::to_string(y) + " (class " + std::to_string(cls[y]) + "): cpu " + mat_str(ref) +
             " gpu " + mat_str(gpu));
      }
    }
    std::printf("cpu reference   : %zu y's checked (%d random + specials), %lld mismatches, %.0f ms CPU\n",
                ys.size(), samples, mismatches, cpu_ms);
    std::printf("M_y for y = x   : %s\n", mat_str(row_of(full, x)).c_str());
    std::printf("M_y for y = -x  : %s\n", mat_str(row_of(full, L.neg[x])).c_str());

    // 3. sub-range entry point.
    {
      const uint32_t y0 = 123456, ny = 1001;
      const std::vector<uint32_t> part = run_range(dl, x, y0, ny, &ms_range);
      long long bad = 0;
      for (uint32_t k = 0; k < ny; ++k)
        if (row_of(part, k) != row_of(full, y0 + k)) ++bad;
      std::printf("sub-range       : y0=%u ny=%u %.3f ms, %lld rows differ from the full pass\n", y0, ny,
                  ms_range, bad);
      if (bad) fail("sub-range result differs from the full pass");
    }

    // 4. scheme property (informational).
    int distinct_total = 0;
    {
      std::array<std::map<Mat, uint32_t>, NCLASS> per_class;
      for (int y = 0; y < N; ++y) ++per_class[cls[static_cast<std::size_t>(y)]][row_of(full, static_cast<uint32_t>(y))];
      std::printf("distinct M_y per class(x,y):");
      for (int c = 0; c < NCLASS; ++c) {
        std::printf(" %zu", per_class[static_cast<std::size_t>(c)].size());
        distinct_total += static_cast<int>(per_class[static_cast<std::size_t>(c)].size());
      }
      std::printf("  (scheme property <=> all 1)\n");
    }

    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("RESULT ok=%d x=%u full_ms=%.1f pairs_per_s=%.3e samples=%zu mismatches=%lld "
                "scheme=%d distinct_total=%d cpu_ref_ms=%.0f total_ms=%.0f failures=%d\n",
                g_failures == 0 ? 1 : 0, x, ms_full, pairs / (ms_full * 1e-3), ys.size(), mismatches,
                distinct_total == NCLASS ? 1 : 0, distinct_total, cpu_ms, total_ms, g_failures);
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "exception: %s\n", ex.what());
    std::printf("RESULT ok=0 exception=1\n");
    return 1;
  }
}

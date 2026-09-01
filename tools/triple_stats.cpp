// triple_stats (T2.3): four-point numbers of the inner-product scheme on the
// Leech minimal vectors, by brute force on the GPU, with the
// triple-regularity check that the three-point SDP needs.
//
//   x = C[0] (or --x IDX).  For each class i a representative y with
//   class(x, y) = i (the first such y, plus --reps-1 random ones) and for
//   EVERY z the kernel returns
//     H_z[a][b][c] = #{ w : class(x,w) = a, class(y,w) = b, class(z,w) = c }.
//   Group the z by (j, k) = (class(x,z), class(y,z)) ("cell" (i,j,k)).  The
//   scheme is "triply regular" w.r.t. (x, y) iff H_z depends only on (j, k);
//   then q[i][j][k][a][b][c] := H_z is a four-point number.  In general the
//   distinct histograms inside a cell, with the number of z carrying each
//   ("variants"), are recorded: H_z is invariant under the stabiliser of
//   (x, y) in Co_0, so #variants is a LOWER bound on the number of orbits of
//   Stab(x,y) on the z of that cell, i.e. on the number of Co_0-orbits of
//   ordered triples with that class pattern.  tools/triple_orbitals.cpp
//   computes the orbits themselves; docs/reports/T2.3.md section 1 puts the
//   two together.
//
//   FINDING (2026-08-26): every cell is regular except (i,j,k) = (3,3,3)
//   (x, y, z mutually orthogonal), which has two variants — so the
//   triple-class matrices do not span an algebra and the three-point SDP has
//   to be formulated on the orbitals of Stab(x) (148 of them), not on the
//   147 class triples.
//
// Checks on the z-summed totals T[i][jk][abc] = sum_{z in cell} H_z (these
// hold whether or not the cells are regular; n_jk = p^i_{jk} = #z in the cell):
//   (a) sum_{abc} T = N n_jk;  (b) one-point marginals sum_{bc} T = v_a n_jk etc.;
//   (c) two-point marginals sum_c T = p^i_{ab} n_jk, sum_b T = sum_z p^j_{ac}, ...;
//   (d) T[i][jk][abc] = T[i][ab][jkc]  (swap the roles of z and w).
// A second representative y per class (and --extra-x further base points)
// must give the same variants (histograms and z-counts) in every cell.
//
// Output: data/scheme/triples.json (or --out PATH): per cell (i,j,k) the
// variants (sparse histograms with z-counts), plus the verification record.
// Last line: RESULT ok=<0|1> regular=<0|1> reps_agree=<0|1> checks=<0|1> ...
// (ok does NOT require regular=1: non-regularity is a finding, not a failure).
//
// Usage: triple_stats [--data DIR] [--x IDX] [--out PATH] [--reps K] [--extra-x K] [--seed S]
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
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

constexpr int NC = 7;
constexpr int kIdentity = 6;
const std::array<int, NC> kClassDot{-32, -16, -8, 0, 8, 16, 32};

using Hist = std::array<uint32_t, NCLASS3>;            // [a*49 + b*7 + c]
using Variants = std::map<Hist, uint32_t>;             // histogram -> number of z carrying it
using Cell = std::array<Variants, NC * NC>;            // per (j,k): j*7+k
using Tot = std::array<uint64_t, NCLASS3>;
using CellTot = std::array<Tot, NC * NC>;

std::vector<uint8_t> classes_of(const Leech& L, uint32_t x) {
  std::vector<uint8_t> cls(static_cast<std::size_t>(N));
  for (int y = 0; y < N; ++y) {
    const int c = kiss::ip_class(kiss::dot(L.C[x], L.C[static_cast<std::size_t>(y)]));
    if (c < 0) throw std::runtime_error("inner product outside the seven classes");
    cls[static_cast<std::size_t>(y)] = static_cast<uint8_t>(c);
  }
  return cls;
}

struct Pass {
  Cell var{};                                // variants per (j,k) cell
  CellTot tot{};                             // z-summed histograms per cell
  std::array<uint32_t, NC * NC> group_size{};
  bool regular = true;
  float gpu_ms = 0.f;
  double host_ms = 0.;
};

Pass run_pass(const kiss::cuda::DeviceLeech& dl, uint32_t* dM, std::vector<uint32_t>& hM,
              const std::vector<uint8_t>& cls_x, const std::vector<uint8_t>& cls_y,
              uint32_t x, uint32_t y) {
  Pass p;
  cudaEvent_t e0, e1;
  KISS_CUDA_CHECK(cudaEventCreate(&e0));
  KISS_CUDA_CHECK(cudaEventCreate(&e1));
  KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
  kiss::cuda::triple_class_histogram_all(dl, x, y, dM, 0);
  KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
  KISS_CUDA_CHECK(cudaDeviceSynchronize());
  KISS_CUDA_CHECK(cudaEventElapsedTime(&p.gpu_ms, e0, e1));
  KISS_CUDA_CHECK(cudaEventDestroy(e0));
  KISS_CUDA_CHECK(cudaEventDestroy(e1));
  const std::size_t bytes = static_cast<std::size_t>(N) * NCLASS3 * sizeof(uint32_t);
  KISS_CUDA_CHECK(cudaMemcpy(hM.data(), dM, bytes, cudaMemcpyDeviceToHost));

  const auto t0 = std::chrono::steady_clock::now();
  for (auto& t : p.tot) t.fill(0);
  for (int z = 0; z < N; ++z) {
    const int g = cls_x[static_cast<std::size_t>(z)] * NC + cls_y[static_cast<std::size_t>(z)];
    Hist h;
    const uint32_t* row = hM.data() + static_cast<std::ptrdiff_t>(z) * NCLASS3;
    std::copy_n(row, NCLASS3, h.begin());
    ++p.var[static_cast<std::size_t>(g)][h];
    ++p.group_size[static_cast<std::size_t>(g)];
    Tot& t = p.tot[static_cast<std::size_t>(g)];
    for (int e = 0; e < NCLASS3; ++e) t[static_cast<std::size_t>(e)] += row[e];
  }
  for (int g = 0; g < NC * NC; ++g)
    if (p.var[static_cast<std::size_t>(g)].size() > 1) p.regular = false;
  p.host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return p;
}

// Intersection numbers p^i_{ab} recovered from the totals (sum over c of any
// cell of i, divided by the cell size), and the valencies.
struct Derived {
  std::array<uint32_t, NC> v{};
  std::array<std::array<std::array<uint64_t, NC>, NC>, NC> p{};   // p[i][a][b]
};

struct Checks {
  long long bad_total = 0, bad_marg1 = 0, bad_marg2 = 0, bad_sym = 0, bad_p = 0;
  bool ok() const { return !(bad_total || bad_marg1 || bad_marg2 || bad_sym || bad_p); }
};

inline uint64_t tv(const std::array<CellTot, NC>& T, int i, int j, int k, int a, int b, int c) {
  return T[static_cast<std::size_t>(i)][static_cast<std::size_t>(j * NC + k)][static_cast<std::size_t>(a * 49 + b * 7 + c)];
}

Checks check_totals(const std::array<CellTot, NC>& T, const std::array<std::array<uint32_t, NC * NC>, NC>& gsz,
                    const Derived& d) {
  Checks r;
  auto n_of = [&](int i, int j, int k) -> uint64_t { return gsz[static_cast<std::size_t>(i)][static_cast<std::size_t>(j * NC + k)]; };
  for (int i = 0; i < NC; ++i)
    for (int j = 0; j < NC; ++j)
      for (int k = 0; k < NC; ++k) {
        const uint64_t n = n_of(i, j, k);
        if (n == 0) continue;
        if (n != d.p[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]) ++r.bad_p;
        uint64_t total = 0;
        for (int a = 0; a < NC; ++a)
          for (int b = 0; b < NC; ++b)
            for (int c = 0; c < NC; ++c) total += tv(T, i, j, k, a, b, c);
        if (total != static_cast<uint64_t>(N) * n) ++r.bad_total;
        for (int a = 0; a < NC; ++a) {
          uint64_t sa = 0, sb = 0, sc = 0;
          for (int u = 0; u < NC; ++u)
            for (int t = 0; t < NC; ++t) {
              sa += tv(T, i, j, k, a, u, t);
              sb += tv(T, i, j, k, u, a, t);
              sc += tv(T, i, j, k, u, t, a);
            }
          const uint64_t va = static_cast<uint64_t>(d.v[static_cast<std::size_t>(a)]) * n;
          if (sa != va || sb != va || sc != va) ++r.bad_marg1;
        }
        for (int a = 0; a < NC; ++a)
          for (int b = 0; b < NC; ++b) {
            uint64_t s_ab = 0, s_ac = 0, s_bc = 0;
            for (int t = 0; t < NC; ++t) {
              s_ab += tv(T, i, j, k, a, b, t);   // -> n * p^i_{ab}
              s_ac += tv(T, i, j, k, a, t, b);   // -> n * p^j_{ab}
              s_bc += tv(T, i, j, k, t, a, b);   // -> n * p^k_{ab}
            }
            if (s_ab != n * d.p[static_cast<std::size_t>(i)][static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] ||
                s_ac != n * d.p[static_cast<std::size_t>(j)][static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] ||
                s_bc != n * d.p[static_cast<std::size_t>(k)][static_cast<std::size_t>(a)][static_cast<std::size_t>(b)]) ++r.bad_marg2;
            for (int c = 0; c < NC; ++c) {
              const uint64_t lhs = tv(T, i, j, k, a, b, c);
              const uint64_t rhs = tv(T, i, a, b, j, k, c);
              if (lhs != rhs) ++r.bad_sym;
            }
          }
      }
  return r;
}

std::string sparse_hist(const Hist& h) {
  std::string s = "[";
  bool first = true;
  for (int a = 0; a < NC; ++a)
    for (int b = 0; b < NC; ++b)
      for (int c = 0; c < NC; ++c) {
        const uint32_t cnt = h[static_cast<std::size_t>(a * 49 + b * 7 + c)];
        if (!cnt) continue;
        s += (first ? "" : ", ") + std::string("[") + std::to_string(a) + ", " + std::to_string(b) + ", " +
             std::to_string(c) + ", " + std::to_string(cnt) + "]";
        first = false;
      }
  return s + "]";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  std::filesystem::path out = "data/scheme/triples.json";
  uint32_t x = 0;
  int reps = 2, extra_x = 0;
  uint64_t seed = 20260826ull;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--x") && i + 1 < argc) x = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--extra-x") && i + 1 < argc) extra_x = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::fprintf(stderr, "triple_stats: no CUDA device (%s)\n", cudaGetErrorString(e));
    std::printf("RESULT ok=0 devices=0\n");
    return 1;
  }
  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    if (x >= static_cast<uint32_t>(N)) throw std::runtime_error("x out of range");
    std::mt19937_64 rng(seed);

    uint32_t* dM = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(N) * NCLASS3 * sizeof(uint32_t);
    KISS_CUDA_CHECK(cudaMalloc(&dM, bytes));
    std::vector<uint32_t> hM(static_cast<std::size_t>(N) * NCLASS3);

    std::vector<uint32_t> xs{x};
    {
      std::uniform_int_distribution<uint32_t> d(0, N - 1);
      for (int t = 0; t < extra_x; ++t) xs.push_back(d(rng));
    }

    std::array<Cell, NC> var0{};                          // variants from x, first representatives
    std::array<CellTot, NC> tot0{};
    std::array<std::array<uint32_t, NC * NC>, NC> gsz0{};
    Derived der;
    bool all_regular = true, all_agree = true;
    int passes = 0;
    double gpu_total = 0;
    std::vector<std::string> pass_records;
    std::array<uint32_t, NC> rep0{};

    for (std::size_t xi = 0; xi < xs.size(); ++xi) {
      const uint32_t xb = xs[xi];
      const std::vector<uint8_t> cls_x = classes_of(L, xb);
      std::array<std::vector<uint32_t>, NC> members;
      for (int y = 0; y < N; ++y) members[cls_x[static_cast<std::size_t>(y)]].push_back(static_cast<uint32_t>(y));
      if (xi == 0)
        for (int c = 0; c < NC; ++c) der.v[static_cast<std::size_t>(c)] = static_cast<uint32_t>(members[static_cast<std::size_t>(c)].size());
      std::printf("base x=%u : C[x] =", xb);
      for (int c = 0; c < kiss::DIM; ++c) std::printf(" %d", L.C[xb][static_cast<std::size_t>(c)]);
      std::printf("\n");

      for (int i = 0; i < NC; ++i) {
        std::vector<uint32_t> ys{members[static_cast<std::size_t>(i)].front()};
        std::uniform_int_distribution<std::size_t> d(0, members[static_cast<std::size_t>(i)].size() - 1);
        for (int r = 1; r < reps; ++r) ys.push_back(members[static_cast<std::size_t>(i)][d(rng)]);
        for (std::size_t r = 0; r < ys.size(); ++r) {
          const uint32_t y = ys[r];
          const std::vector<uint8_t> cls_y = classes_of(L, y);
          const Pass p = run_pass(dl, dM, hM, cls_x, cls_y, xb, y);
          ++passes;
          gpu_total += p.gpu_ms;
          const bool first = (xi == 0 && r == 0);
          bool agree = true;
          if (first) {
            var0[static_cast<std::size_t>(i)] = p.var;
            tot0[static_cast<std::size_t>(i)] = p.tot;
            gsz0[static_cast<std::size_t>(i)] = p.group_size;
            rep0[static_cast<std::size_t>(i)] = y;
          } else {
            for (int g = 0; g < NC * NC; ++g)
              if (p.var[static_cast<std::size_t>(g)] != var0[static_cast<std::size_t>(i)][static_cast<std::size_t>(g)]) agree = false;
          }
          all_regular = all_regular && p.regular;
          all_agree = all_agree && agree;
          std::string dist;
          for (int g = 0; g < NC * NC; ++g)
            if (p.group_size[static_cast<std::size_t>(g)] > 0) dist += std::to_string(p.var[static_cast<std::size_t>(g)].size());
          std::printf("  class i=%d (dot %3d) y=%6u [rep %zu]: gpu %.1f ms, host %.0f ms, distinct histograms per (j,k) cell: %s -> %s%s\n",
                      i, kClassDot[static_cast<std::size_t>(i)], y, r, p.gpu_ms, p.host_ms, dist.c_str(),
                      p.regular ? "regular" : "NOT REGULAR", first ? "" : (agree ? ", variants identical to rep 0 of x" : ", variants DIFFER from rep 0 of x"));
          pass_records.push_back("{\"x\": " + std::to_string(xb) + ", \"i\": " + std::to_string(i) + ", \"y\": " +
                                 std::to_string(y) + ", \"regular\": " + (p.regular ? "true" : "false") +
                                 ", \"variants_agree_with_first\": " + (agree ? "true" : "false") + ", \"gpu_ms\": " +
                                 std::to_string(p.gpu_ms) + "}");
        }
      }
    }
    KISS_CUDA_CHECK(cudaFree(dM));

    // p^i_{ab} from the totals of the first nonempty cell of i (sum over c) / cell size.
    for (int i = 0; i < NC; ++i)
      for (int a = 0; a < NC; ++a)
        for (int b = 0; b < NC; ++b) {
          uint64_t s = 0, n = 0;
          for (int g = 0; g < NC * NC; ++g)
            if (gsz0[static_cast<std::size_t>(i)][static_cast<std::size_t>(g)]) {
              n = gsz0[static_cast<std::size_t>(i)][static_cast<std::size_t>(g)];
              for (int c = 0; c < NC; ++c) s += tv(tot0, i, g / NC, g % NC, a, b, c);
              break;
            }
          der.p[static_cast<std::size_t>(i)][static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = n ? s / n : 0;
          if (n && s % n) throw std::runtime_error("total not divisible by the cell size");
        }
    const Checks ck = check_totals(tot0, gsz0, der);
    int ncells = 0, nvariants = 0;
    std::vector<std::string> split;
    for (int i = 0; i < NC; ++i)
      for (int g = 0; g < NC * NC; ++g)
        if (gsz0[static_cast<std::size_t>(i)][static_cast<std::size_t>(g)]) {
          ++ncells;
          const std::size_t nv = var0[static_cast<std::size_t>(i)][static_cast<std::size_t>(g)].size();
          nvariants += static_cast<int>(nv);
          if (nv > 1) {
            std::string s = "(" + std::to_string(i) + "," + std::to_string(g / NC) + "," + std::to_string(g % NC) + "):";
            for (const auto& kv : var0[static_cast<std::size_t>(i)][static_cast<std::size_t>(g)]) s += " " + std::to_string(kv.second);
            split.push_back(s);
          }
        }
    std::printf("nonempty cells (i,j,k): %d; variants (distinct histograms) in total: %d = lower bound on the number of "
                "Stab(x)-orbitals\n", ncells, nvariants);
    for (const auto& s : split) std::printf("  split cell %s  (z-counts of the variants)\n", s.c_str());
    std::printf("checks on z-summed totals: sum=N*n %lld bad, one-point marginals %lld bad, two-point marginals = p %lld bad, "
                "cell size = p^i_jk %lld bad, z<->w symmetry %lld bad -> %s\n",
                ck.bad_total, ck.bad_marg1, ck.bad_marg2, ck.bad_p, ck.bad_sym, ck.ok() ? "OK" : "FAIL");

    // JSON (sparse).
    std::filesystem::create_directories(out.parent_path().empty() ? "." : out.parent_path());
    std::ofstream f(out);
    if (!f) throw std::runtime_error("cannot write " + out.string());
    f << "{\n  \"task\": \"T2.3\",\n";
    f << "  \"description\": \"Four-point statistics of the Leech minimal-vector scheme: for x = C[x], one representative y per class i = class(x,y) "
         "and every z, H_z[a][b][c] = #{w : class(x,w)=a, class(y,w)=b, class(z,w)=c}; cells (i,j,k) with j = class(x,z), k = class(y,z). "
         "Per cell the distinct histograms (variants) with the number of z carrying each; a cell is regular iff it has one variant. "
         "Classes c <-> inner products classes[c] (sqrt(8) scaling).\",\n";
    f << "  \"N\": " << N << ",\n  \"x\": " << x << ",\n";
    f << "  \"classes\": [";
    for (int c = 0; c < NC; ++c) f << kClassDot[static_cast<std::size_t>(c)] << (c + 1 < NC ? ", " : "");
    f << "],\n  \"identity_class\": " << kIdentity << ",\n  \"conflict_class\": 5,\n";
    f << "  \"valencies\": [";
    for (int c = 0; c < NC; ++c) f << der.v[static_cast<std::size_t>(c)] << (c + 1 < NC ? ", " : "");
    f << "],\n  \"representatives\": [";
    for (int c = 0; c < NC; ++c) f << rep0[static_cast<std::size_t>(c)] << (c + 1 < NC ? ", " : "");
    f << "],\n  \"reps_per_class\": " << reps << ",\n  \"extra_x\": [";
    for (std::size_t t = 1; t < xs.size(); ++t) f << xs[t] << (t + 1 < xs.size() ? ", " : "");
    f << "],\n  \"passes\": " << passes << ",\n";
    f << "  \"regular\": " << (all_regular ? "true" : "false") << ",\n";
    f << "  \"reps_agree\": " << (all_agree ? "true" : "false") << ",\n";
    f << "  \"checks_ok\": " << (ck.ok() ? "true" : "false") << ",\n";
    f << "  \"nonempty_cells\": " << ncells << ",\n";
    f << "  \"total_variants\": " << nvariants << ",\n";
    f << "  \"pass_records\": [\n";
    for (std::size_t t = 0; t < pass_records.size(); ++t) f << "    " << pass_records[t] << (t + 1 < pass_records.size() ? ",\n" : "\n");
    f << "  ],\n";
    f << "  \"cells_format\": \"list of {ijk: [i,j,k], n_z: p^i_jk, regular: bool, variants: [{z_count, q: [[a,b,c,count], ...]}]} (count > 0 only; "
         "q is the histogram of one z of the variant; for a regular cell it is the four-point number q[i][j][k])\",\n";
    f << "  \"cells\": [\n";
    bool firstc = true;
    for (int i = 0; i < NC; ++i)
      for (int j = 0; j < NC; ++j)
        for (int k = 0; k < NC; ++k) {
          const uint32_t n = gsz0[static_cast<std::size_t>(i)][static_cast<std::size_t>(j * NC + k)];
          if (!n) continue;
          const Variants& vs = var0[static_cast<std::size_t>(i)][static_cast<std::size_t>(j * NC + k)];
          f << (firstc ? "" : ",\n") << "    {\"ijk\": [" << i << ", " << j << ", " << k << "], \"n_z\": " << n
            << ", \"regular\": " << (vs.size() == 1 ? "true" : "false") << ", \"variants\": [";
          firstc = false;
          bool firstv = true;
          for (const auto& kv : vs) {
            f << (firstv ? "" : ", ") << "{\"z_count\": " << kv.second << ", \"q\": " << sparse_hist(kv.first) << "}";
            firstv = false;
          }
          f << "]}";
        }
    f << "\n  ]\n}\n";
    f.close();
    std::printf("wrote %s\n", out.string().c_str());

    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    const bool ok = all_agree && ck.ok();
    std::printf("RESULT ok=%d regular=%d reps_agree=%d checks=%d x=%u passes=%d cells=%d variants=%d split_cells=%zu reps=%d extra_x=%d gpu_ms=%.1f total_ms=%.0f out=%s\n",
                ok ? 1 : 0, all_regular ? 1 : 0, all_agree ? 1 : 0, ck.ok() ? 1 : 0, x, passes, ncells, nvariants, split.size(), reps, extra_x,
                gpu_total, total_ms, out.string().c_str());
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "triple_stats: %s\n", ex.what());
    std::printf("RESULT ok=0 error=1\n");
    return 1;
  }
}

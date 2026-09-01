// scheme_numbers (T2.2): intersection numbers p^k_{ij} of the inner-product
// relations on the Leech minimal vectors, computed by brute force on the GPU
// and checked for the association-scheme property.
//
//   x = C[0] (or --x IDX).  For EVERY y the kernel returns
//     M_y[i][j] = #{ z : class(x,z) = i, class(z,y) = j }
//   (class c = (dot+32)/8, c = 0..6 <-> dot = -32,-16,-8,0,8,16,32; class 6 is
//   the identity relation).  The relations form an association scheme iff M_y
//   depends only on k = class(x,y); then p^k_{ij} = M_y[i][j] for any y in
//   class k.  This is verified here for x = C[0] and all 196560 y; that it
//   then holds for every x follows from vertex-transitivity of Co_0 on C
//   (Conway 1969 / Conway–Sloane ch. 10), which is VERIFIED here rather than
//   cited: the orbit of C[x] under T4.1's five explicit generators of Co_0
//   (data/group: M24 generators, an octad sign flip, Conway's xi) is computed
//   by breadth-first search over index permutations and must be all of C;
//   each generator is an orthogonal map (monomial by construction, xi checked
//   exactly: xi^T xi = den^2 I) that maps C to C (index_permutation throws
//   otherwise), hence preserves every inner-product class.  --no-orbit skips
//   this; --extra-x K additionally re-runs the whole computation for K further
//   random x's as a direct spot check.
//
// Identities checked (v_i = class sizes, e = 6 the identity class):
//   (a) sum_j p^k_{ij} = v_i           (b) v_k p^k_{ij} = v_i p^i_{kj}
//   (c) p^k_{ij} = p^k_{ji}            (d) p^e_{ij} = delta_ij v_i
//   (e) sum_i p^k_{ij} = v_j           (f) p^k_{ie} = delta_ik
//
// Output: data/scheme/intersection_numbers.json (or --out PATH) with
// p[k][i][j], valencies, class labels; if the scheme property fails, the
// distinct matrices found in each class are written too ("variants").
// Last line: RESULT ok=<0|1> scheme=<0|1> identities=<0|1> ...
//
// Usage: scheme_numbers [--data DIR] [--group DIR] [--x IDX] [--out PATH] [--extra-x K] [--no-orbit]
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

#include "kiss/group.h"
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

constexpr int kIdentity = 6;  // class index of dot = 32
const std::array<int, NCLASS> kClassDot{-32, -16, -8, 0, 8, 16, 32};

using Mat = std::array<uint32_t, NCLASS2>;   // M[i*7+j]
using Tensor = std::array<Mat, NCLASS>;      // p[k][i*7+j]

struct ClassInfo {
  std::vector<uint8_t> cls;                      // class(x, y) for all y
  std::array<uint32_t, NCLASS> val{};            // valencies
};

ClassInfo classes_of(const Leech& L, uint32_t x) {
  ClassInfo ci;
  ci.cls.resize(static_cast<std::size_t>(N));
  for (int y = 0; y < N; ++y) {
    const int c = kiss::ip_class(kiss::dot(L.C[x], L.C[static_cast<std::size_t>(y)]));
    if (c < 0) throw std::runtime_error("inner product outside the seven classes at y=" + std::to_string(y));
    ci.cls[static_cast<std::size_t>(y)] = static_cast<uint8_t>(c);
    ++ci.val[static_cast<std::size_t>(c)];
  }
  return ci;
}

struct Histograms {
  std::vector<uint32_t> M;   // [N][49]
  float gpu_ms = 0.f;
};

Histograms run_all(const kiss::cuda::DeviceLeech& dl, uint32_t x) {
  Histograms h;
  uint32_t* dM = nullptr;
  const std::size_t bytes = static_cast<std::size_t>(N) * NCLASS2 * sizeof(uint32_t);
  KISS_CUDA_CHECK(cudaMalloc(&dM, bytes));
  cudaEvent_t e0, e1;
  KISS_CUDA_CHECK(cudaEventCreate(&e0));
  KISS_CUDA_CHECK(cudaEventCreate(&e1));
  KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
  kiss::cuda::pair_class_histogram_all(dl, x, dM, 0);
  KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
  KISS_CUDA_CHECK(cudaDeviceSynchronize());
  KISS_CUDA_CHECK(cudaEventElapsedTime(&h.gpu_ms, e0, e1));
  h.M.resize(static_cast<std::size_t>(N) * NCLASS2);
  KISS_CUDA_CHECK(cudaMemcpy(h.M.data(), dM, bytes, cudaMemcpyDeviceToHost));
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

struct Grouped {
  Tensor p{};                                              // matrix of the first y of each class
  std::array<std::map<Mat, uint32_t>, NCLASS> variants;    // distinct matrices per class -> count
  std::array<uint32_t, NCLASS> example_y{};                // a y with the "p" matrix of each class
  bool scheme = true;
};

Grouped group_by_class(const Histograms& h, const ClassInfo& ci) {
  Grouped g;
  std::array<bool, NCLASS> seen{};
  for (int y = 0; y < N; ++y) {
    const int k = ci.cls[static_cast<std::size_t>(y)];
    const Mat M = row_of(h.M, static_cast<uint32_t>(y));
    ++g.variants[static_cast<std::size_t>(k)][M];
    if (!seen[static_cast<std::size_t>(k)]) {
      seen[static_cast<std::size_t>(k)] = true;
      g.p[static_cast<std::size_t>(k)] = M;
      g.example_y[static_cast<std::size_t>(k)] = static_cast<uint32_t>(y);
    }
  }
  for (int k = 0; k < NCLASS; ++k)
    if (g.variants[static_cast<std::size_t>(k)].size() != 1) g.scheme = false;
  return g;
}

inline uint32_t P(const Tensor& p, int k, int i, int j) {
  return p[static_cast<std::size_t>(k)][static_cast<std::size_t>(i * NCLASS + j)];
}

struct Identities {
  long long bad_a = 0, bad_b = 0, bad_c = 0, bad_d = 0, bad_e = 0, bad_f = 0;
  bool ok() const { return !(bad_a || bad_b || bad_c || bad_d || bad_e || bad_f); }
};

Identities check_identities(const Tensor& p, const std::array<uint32_t, NCLASS>& v) {
  Identities r;
  for (int k = 0; k < NCLASS; ++k)
    for (int i = 0; i < NCLASS; ++i) {
      uint64_t rs = 0, cs = 0;
      for (int j = 0; j < NCLASS; ++j) {
        rs += P(p, k, i, j);
        cs += P(p, k, j, i);
        if (static_cast<uint64_t>(v[static_cast<std::size_t>(k)]) * P(p, k, i, j) !=
            static_cast<uint64_t>(v[static_cast<std::size_t>(i)]) * P(p, i, k, j)) ++r.bad_b;
        if (P(p, k, i, j) != P(p, k, j, i)) ++r.bad_c;
        if (P(p, kIdentity, i, j) != (i == j ? v[static_cast<std::size_t>(i)] : 0u)) ++r.bad_d;
      }
      if (rs != v[static_cast<std::size_t>(i)]) ++r.bad_a;
      if (cs != v[static_cast<std::size_t>(i)]) ++r.bad_e;
      if (P(p, k, i, kIdentity) != (i == k ? 1u : 0u)) ++r.bad_f;
    }
  return r;
}

std::string mat_json(const Mat& M, const std::string& indent) {
  std::string s = "[\n";
  for (int i = 0; i < NCLASS; ++i) {
    s += indent + "  [";
    for (int j = 0; j < NCLASS; ++j) s += std::to_string(M[static_cast<std::size_t>(i * NCLASS + j)]) + (j + 1 < NCLASS ? ", " : "");
    s += std::string("]") + (i + 1 < NCLASS ? ",\n" : "\n");
  }
  return s + indent + "]";
}

std::string mat_line(const Mat& M) {
  std::string s;
  for (int i = 0; i < NCLASS; ++i) {
    s += "[";
    for (int j = 0; j < NCLASS; ++j) s += std::to_string(M[static_cast<std::size_t>(i * NCLASS + j)]) + (j + 1 < NCLASS ? " " : "");
    s += "]";
  }
  return s;
}

// Exact orthogonality of a rational automorphism: num^T num == den^2 I.
bool is_orthogonal(const kiss::Aut& a) {
  for (int r = 0; r < kiss::DIM; ++r)
    for (int c = 0; c < kiss::DIM; ++c) {
      long long acc = 0;
      for (int k = 0; k < kiss::DIM; ++k)
        acc += static_cast<long long>(a.num[static_cast<std::size_t>(k * kiss::DIM + r)]) *
               a.num[static_cast<std::size_t>(k * kiss::DIM + c)];
      if (acc != (r == c ? static_cast<long long>(a.den) * a.den : 0ll)) return false;
    }
  return true;
}

struct OrbitInfo {
  std::size_t orbit_size = 0;
  std::size_t n_generators = 0;
  bool xi_orthogonal = false;
  double ms = 0;
  bool transitive() const { return orbit_size == static_cast<std::size_t>(N) && xi_orthogonal; }
};

// Orbit of x under the group generated by T4.1's Co_0 generators (BFS).
OrbitInfo orbit_of(const Leech& L, const std::filesystem::path& group_dir, uint32_t x) {
  const auto t0 = std::chrono::steady_clock::now();
  OrbitInfo info;
  info.xi_orthogonal = is_orthogonal(kiss::load_aut(group_dir / "xi.txt"));
  // Every generator is verified to map C into C by index_permutation() (throws otherwise);
  // the monomial ones are signed coordinate permutations, hence orthogonal.
  const std::vector<kiss::IndexPerm> gens = kiss::co0_generator_perms(L, group_dir);
  info.n_generators = gens.size();
  std::vector<uint8_t> seen(static_cast<std::size_t>(N), 0);
  std::vector<uint32_t> queue;
  queue.reserve(static_cast<std::size_t>(N));
  queue.push_back(x);
  seen[x] = 1;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const uint32_t v = queue[head];
    for (const kiss::IndexPerm& g : gens) {
      const uint32_t w = g[v];
      if (!seen[w]) { seen[w] = 1; queue.push_back(w); }
    }
  }
  info.orbit_size = queue.size();
  info.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return info;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  std::filesystem::path group_dir = "data/group";
  std::filesystem::path out = "data/scheme/intersection_numbers.json";
  uint32_t x = 0;
  int extra_x = 0;
  bool do_orbit = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--group") && i + 1 < argc) group_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--no-orbit")) do_orbit = false;
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--x") && i + 1 < argc) x = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--extra-x") && i + 1 < argc) extra_x = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::fprintf(stderr, "scheme_numbers: no CUDA device (%s)\n", cudaGetErrorString(e));
    std::printf("RESULT ok=0 devices=0\n");
    return 1;
  }
  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    if (x >= static_cast<uint32_t>(N)) throw std::runtime_error("x out of range");

    const ClassInfo ci = classes_of(L, x);
    std::printf("x = %u : C[x] =", x);
    for (int c = 0; c < kiss::DIM; ++c) std::printf(" %d", L.C[x][static_cast<std::size_t>(c)]);
    std::printf("\nvalencies v_c (c = 0..6 <-> dot -32,-16,-8,0,8,16,32):");
    for (int c = 0; c < NCLASS; ++c) std::printf(" %u", ci.val[static_cast<std::size_t>(c)]);
    std::printf("\n");

    const Histograms h = run_all(dl, x);
    std::printf("GPU pass over all %d y: %.1f ms (%.3e pairs/s)\n", N, h.gpu_ms,
                static_cast<double>(N) * N / (h.gpu_ms * 1e-3));
    const Grouped g = group_by_class(h, ci);

    std::printf("distinct M_y per class k = class(x,y):");
    for (int k = 0; k < NCLASS; ++k) std::printf(" %zu", g.variants[static_cast<std::size_t>(k)].size());
    std::printf("  -> scheme property %s\n", g.scheme ? "HOLDS" : "FAILS");
    for (int k = 0; k < NCLASS; ++k) {
      std::printf("p^%d (dot %3d, v=%5u, example y=%u): %s\n", k, kClassDot[static_cast<std::size_t>(k)],
                  ci.val[static_cast<std::size_t>(k)], g.example_y[static_cast<std::size_t>(k)],
                  mat_line(g.p[static_cast<std::size_t>(k)]).c_str());
      if (g.variants[static_cast<std::size_t>(k)].size() != 1) {
        for (const auto& [M, cnt] : g.variants[static_cast<std::size_t>(k)])
          std::printf("    variant x%u: %s\n", cnt, mat_line(M).c_str());
      }
    }
    const Identities id = check_identities(g.p, ci.val);
    std::printf("identities: (a) row sums = v_i: %lld bad, (b) v_k p^k_ij = v_i p^i_kj: %lld bad, "
                "(c) p^k_ij = p^k_ji: %lld bad, (d) p^6_ij = delta v_i: %lld bad, "
                "(e) column sums = v_j: %lld bad, (f) p^k_i6 = delta_ik: %lld bad -> %s\n",
                id.bad_a, id.bad_b, id.bad_c, id.bad_d, id.bad_e, id.bad_f, id.ok() ? "OK" : "FAIL");

    // Vertex-transitivity, computed: orbit of x under the Co_0 generators.
    OrbitInfo orbit;
    if (do_orbit) {
      orbit = orbit_of(L, group_dir, x);
      std::printf("orbit of x=%u under %zu Co_0 generators (%s): %zu of %d vectors -> transitive: %s; "
                  "xi orthogonal (exact): %s (%.0f ms)\n",
                  x, orbit.n_generators, group_dir.string().c_str(), orbit.orbit_size, N,
                  orbit.orbit_size == static_cast<std::size_t>(N) ? "YES" : "NO",
                  orbit.xi_orthogonal ? "yes" : "NO", orbit.ms);
    }

    // Spot check of vertex-transitivity: the same numbers for other x.
    int extra_agree = 0;
    std::vector<uint32_t> extra_xs;
    if (extra_x > 0) {
      std::mt19937_64 rng(20260825ull);
      std::uniform_int_distribution<uint32_t> d(0, N - 1);
      for (int t = 0; t < extra_x; ++t) {
        const uint32_t x2 = d(rng);
        extra_xs.push_back(x2);
        const ClassInfo ci2 = classes_of(L, x2);
        const Histograms h2 = run_all(dl, x2);
        const Grouped g2 = group_by_class(h2, ci2);
        const bool same = g2.scheme && g2.p == g.p && ci2.val == ci.val;
        extra_agree += same;
        std::printf("extra x=%u: scheme %s, p identical to x=%u: %s (%.1f ms)\n", x2,
                    g2.scheme ? "holds" : "FAILS", x, same ? "yes" : "NO", h2.gpu_ms);
      }
    }

    // JSON
    std::filesystem::create_directories(out.parent_path().empty() ? "." : out.parent_path());
    std::ofstream f(out);
    if (!f) throw std::runtime_error("cannot write " + out.string());
    f << "{\n";
    f << "  \"task\": \"T2.2\",\n";
    f << "  \"description\": \"Intersection numbers p[k][i][j] = #{z : class(x,z)=i, class(z,y)=j} for class(x,y)=k, "
         "Leech minimal vectors, class c <-> inner product classes[c] (sqrt(8) scaling). Computed for x = C[x] and all y.\",\n";
    f << "  \"N\": " << N << ",\n";
    f << "  \"x\": " << x << ",\n";
    f << "  \"x_vector\": [";
    for (int c = 0; c < kiss::DIM; ++c) f << static_cast<int>(L.C[x][static_cast<std::size_t>(c)]) << (c + 1 < kiss::DIM ? ", " : "");
    f << "],\n";
    f << "  \"classes\": [";
    for (int c = 0; c < NCLASS; ++c) f << kClassDot[static_cast<std::size_t>(c)] << (c + 1 < NCLASS ? ", " : "");
    f << "],\n";
    f << "  \"class_labels\": [";
    for (int c = 0; c < NCLASS; ++c) f << "\"" << kClassDot[static_cast<std::size_t>(c)] << "\"" << (c + 1 < NCLASS ? ", " : "");
    f << "],\n";
    f << "  \"identity_class\": " << kIdentity << ",\n";
    f << "  \"conflict_class\": 5,\n";
    f << "  \"valencies\": [";
    for (int c = 0; c < NCLASS; ++c) f << ci.val[static_cast<std::size_t>(c)] << (c + 1 < NCLASS ? ", " : "");
    f << "],\n";
    f << "  \"y_checked\": " << N << ",\n";
    f << "  \"scheme\": " << (g.scheme ? "true" : "false") << ",\n";
    f << "  \"identities_ok\": " << (id.ok() ? "true" : "false") << ",\n";
    f << "  \"distinct_matrices_per_class\": [";
    for (int c = 0; c < NCLASS; ++c) f << g.variants[static_cast<std::size_t>(c)].size() << (c + 1 < NCLASS ? ", " : "");
    f << "],\n";
    f << "  \"example_y_per_class\": [";
    for (int c = 0; c < NCLASS; ++c) f << g.example_y[static_cast<std::size_t>(c)] << (c + 1 < NCLASS ? ", " : "");
    f << "],\n";
    f << "  \"extra_x\": [";
    for (std::size_t t = 0; t < extra_xs.size(); ++t) f << extra_xs[t] << (t + 1 < extra_xs.size() ? ", " : "");
    f << "],\n";
    f << "  \"extra_x_agree\": " << extra_agree << ",\n";
    f << "  \"orbit_checked\": " << (do_orbit ? "true" : "false") << ",\n";
    f << "  \"orbit_size\": " << orbit.orbit_size << ",\n";
    f << "  \"orbit_generators\": " << orbit.n_generators << ",\n";
    f << "  \"xi_orthogonal\": " << (orbit.xi_orthogonal ? "true" : "false") << ",\n";
    f << "  \"vertex_transitive_verified\": " << (do_orbit && orbit.transitive() ? "true" : "false") << ",\n";
    f << "  \"gpu_ms\": " << h.gpu_ms << ",\n";
    f << "  \"p\": [\n";
    for (int k = 0; k < NCLASS; ++k)
      f << "    " << mat_json(g.p[static_cast<std::size_t>(k)], "    ") << (k + 1 < NCLASS ? ",\n" : "\n");
    f << "  ]";
    if (!g.scheme) {
      f << ",\n  \"variants\": [\n";
      bool first = true;
      for (int k = 0; k < NCLASS; ++k)
        for (const auto& [M, cnt] : g.variants[static_cast<std::size_t>(k)]) {
          f << (first ? "" : ",\n") << "    {\"k\": " << k << ", \"count\": " << cnt << ", \"matrix\": "
            << mat_json(M, "    ") << "}";
          first = false;
        }
      f << "\n  ]";
    }
    f << "\n}\n";
    f.close();
    std::printf("wrote %s\n", out.string().c_str());

    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    const bool ok = g.scheme && id.ok() && extra_agree == static_cast<int>(extra_xs.size()) &&
                    (!do_orbit || orbit.transitive());
    std::printf("RESULT ok=%d scheme=%d identities=%d x=%u y_checked=%d extra_x=%zu extra_x_agree=%d "
                "orbit_checked=%d orbit_size=%zu transitive=%d gpu_ms=%.1f total_ms=%.0f out=%s\n",
                ok ? 1 : 0, g.scheme ? 1 : 0, id.ok() ? 1 : 0, x, N, extra_xs.size(), extra_agree,
                do_orbit ? 1 : 0, orbit.orbit_size, (do_orbit && orbit.transitive()) ? 1 : 0,
                h.gpu_ms, total_ms, out.string().c_str());
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "exception: %s\n", ex.what());
    std::printf("RESULT ok=0 exception=1\n");
    return 1;
  }
}

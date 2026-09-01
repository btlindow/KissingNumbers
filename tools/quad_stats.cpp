// quad_stats (B2): quadruple orbit statistics of the Leech minimal-vector
// scheme — the data a four-point (Lasserre level-3) relaxation, or clique
// cuts of size >= 4 in the T2.3 SDP, would be built on.  docs/reports/B2.md.
//
// What it computes, for x = C[0] and one representative y_i per pair class i
// (the same representatives as tools/triple_orbitals):
//
//   level 2 (re-derived): orbits of Stab(x, y_i) on vertices z, as labels of
//     explicit random stabiliser elements (union-find over IndexPerms).  The
//     counts must reproduce T2.3's labels_per_class = 7,25,27,30,27,25,7.
//
//   level 3 (new): for every Stab(x, y_i)-orbit representative z_r,
//     the orbits of Stab(x, y_i, z_r) on vertices w, bracketed from BOTH
//     sides:
//       * LOWER bound (GPU, cuda/quad_stats.cu): the number of distinct
//         five-point histograms H_w[a][b][c][d] = #{v : class(x,v)=a,
//         class(y,v)=b, class(z,v)=c, class(w,v)=d} — H_w is a
//         Stab(x,y,z)-invariant, so distinct H_w means distinct orbits.
//       * UPPER bound (CPU): labels = orbits of explicit random elements of
//         Stab(x, y_i, z_r) (products of Stab(x,y_i) elements pulled back
//         through a Schreier transversal of the z-orbit), which REFINE the
//         true orbits.
//     When the two agree the orbit count is exact (same proof pattern as
//     T2.3's D = 148 = 148).
//
//   dim T(x, y_i) = #orbitals of Stab(x, y_i) on ordered pairs (z, w)
//                 = sum over z-orbit reps r of #orbits of Stab(x,y_i,z_r) on w
//     — the dimension of the centraliser algebra a four-point SDP's PSD
//     blocks live in (the analogue of T2.3's dim T(x) = 148 one level up).
//
// Consistency anchors built in: classes 0 (y = -x) and 6 (y = x) have
// Stab(x,y) = Stab(x), so their dim T(x,y) must be exactly 148 = T2.3's D;
// classes 1/5 and 2/4 are conjugate via y -> -y, so their dims must agree;
// every per-rep orbit-size sum must be N; labels must refine histogram
// classes with zero violations; per pass, random w's are re-histogrammed on
// the CPU against the GPU rows.
//
// Output: data/scheme/quad_orbits.json (or --out): per class the z-orbits and
// for each z-orbit the list of w-orbits [w_rep, size, a, b, c, forbidden,
// distinct4] plus the histogram/label counts, and summary totals.
// Last line: RESULT ok=<0|1> dimT=<7 comma-separated values> ...
//
// Usage: quad_stats [--data DIR] [--group DIR] [--out PATH] [--x IDX]
//                   [--seed S] [--classes 0,3,5] [--patience P] [--chunk NW]
//                   [--spot K]
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kiss/group.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss_cuda.h"
#include "quad_cuda.h"

using kiss::IndexPerm;
using kiss::Leech;
using kiss::N;
using kiss::cuda::NCLASS4;

namespace {

constexpr int NC = 7;
constexpr int kConflict = 5;
constexpr int kIdentity = 6;
const std::array<int, NC> kClassDot{-32, -16, -8, 0, 8, 16, 32};

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
};

// A set of "moves" (permutations and their inverses) with a BFS Schreier
// vector rooted at one or several roots (same as tools/triple_orbitals.cpp).
struct Schreier {
  std::vector<IndexPerm> moves;      // moves[2m] = g_m, moves[2m+1] = g_m^-1
  std::vector<uint32_t> parent;      // parent[v]; parent[root] = root
  std::vector<uint16_t> move;        // v = moves[move[v]](parent[v])
  std::vector<uint8_t> is_root;
  uint32_t max_depth = 0;

  void set_moves(const std::vector<IndexPerm>& gens) {
    moves.clear();
    for (const auto& g : gens) {
      moves.push_back(g);
      moves.push_back(kiss::inverse(g));
    }
  }
  std::size_t bfs(const std::vector<uint32_t>& roots) {
    parent.assign(static_cast<std::size_t>(N), UINT32_MAX);
    move.assign(static_cast<std::size_t>(N), 0);
    is_root.assign(static_cast<std::size_t>(N), 0);
    std::vector<uint32_t> depth(static_cast<std::size_t>(N), 0);
    std::vector<uint32_t> queue;
    queue.reserve(static_cast<std::size_t>(N));
    for (uint32_t r : roots) {
      parent[r] = r;
      is_root[r] = 1;
      queue.push_back(r);
    }
    max_depth = 0;
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const uint32_t v = queue[head];
      for (std::size_t m = 0; m < moves.size(); ++m) {
        const uint32_t w = moves[m][v];
        if (parent[w] == UINT32_MAX) {
          parent[w] = v;
          move[w] = static_cast<uint16_t>(m);
          depth[w] = depth[v] + 1;
          max_depth = std::max(max_depth, depth[w]);
          queue.push_back(w);
        }
      }
    }
    return queue.size();
  }
  // u_v^-1 applied to p, where u_v is the transversal element with u_v(root) = v.
  uint32_t apply_inv(uint32_t v, uint32_t p) const {
    while (!is_root[v]) {
      p = moves[move[v] ^ 1u][p];
      v = parent[v];
    }
    return p;
  }
  // Full permutation u_v^-1 composed with g: out[p] = u_v^-1(g[p]).
  void inv_after(uint32_t v, const IndexPerm& g, IndexPerm& out) const {
    out.resize(static_cast<std::size_t>(N));
#pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p) out[static_cast<std::size_t>(p)] = apply_inv(v, g[static_cast<std::size_t>(p)]);
  }
};

struct UnionFind {
  std::vector<uint32_t> parent;
  explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
  uint32_t find(uint32_t a) {
    while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  }
  bool unite(uint32_t a, uint32_t b) {
    a = find(a); b = find(b);
    if (a == b) return false;
    if (a < b) parent[b] = a; else parent[a] = b;
    return true;
  }
  std::size_t add(const IndexPerm& g) {
    std::size_t merges = 0;
    for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) merges += unite(v, g[v]);
    return merges;
  }
  // Labels 0..count-1 in order of the smallest member; returns count.
  uint32_t labels(std::vector<uint32_t>& lab) {
    lab.assign(static_cast<std::size_t>(N), UINT32_MAX);
    uint32_t count = 0;
    for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
      const uint32_t r = find(v);
      if (lab[r] == UINT32_MAX) lab[r] = count++;
      lab[v] = lab[r];
    }
    return count;
  }
};

int cls_of(const Leech& L, uint32_t a, uint32_t b) {
  const int c = kiss::ip_class(kiss::dot(L.C[a], L.C[b]));
  if (c < 0) throw std::runtime_error("inner product outside the classes");
  return c;
}

uint64_t hash_row(const uint32_t* row, int n) {
  uint64_t h = 0x9e3779b97f4a7c15ull;
  for (int e = 0; e < n; ++e) {
    h ^= row[e];
    h *= 0x100000001b3ull;
    h ^= h >> 29;
  }
  return h;
}

// One w-orbit of Stab(x, y, z): representative, size, the classes of w
// against (x, y, z), and the admissibility flags of the quadruple
// (x, y, z, w_rep).
struct WOrbit {
  uint32_t w = 0;      // smallest member
  uint32_t size = 0;
  int a = 0, b = 0, c = 0;   // class(x,w), class(y,w), class(z,w)
  bool forbidden = false;    // some pairwise class of the quadruple is 5 (dot 16)
  bool distinct4 = false;    // all four points pairwise distinct
};

struct ZOrbitResult {
  uint32_t z = 0;
  int j = 0, k = 0;          // class(x,z), class(y,z)
  uint32_t size = 0;         // |z-orbit| under Stab(x, y)
  uint32_t hist_orbits = 0;  // distinct 5-point histograms (LOWER bound)
  uint32_t label_orbits = 0; // orbits of explicit stabiliser elements (UPPER bound)
  int stab3_elements = 0;
  bool exact = false;
  bool stabilised = true;
  long long refine_violations = 0;   // labels not refining histogram classes (must be 0)
  std::vector<WOrbit> w_orbits;      // from the labels
  double gpu_ms = 0, host_ms = 0, label_ms = 0;
};

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data", group_dir, out = "data/scheme/quad_orbits.json";
  uint32_t x = 0;
  uint64_t seed = 20260829ull;
  int patience = 6, spot = 2;
  uint32_t chunk = 32768;
  std::vector<int> classes{0, 1, 2, 3, 4, 5, 6};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--group") && i + 1 < argc) group_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--x") && i + 1 < argc) x = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--patience") && i + 1 < argc) patience = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = static_cast<uint32_t>(std::max(256, std::atoi(argv[++i])));
    else if (!std::strcmp(argv[i], "--spot") && i + 1 < argc) spot = std::max(0, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--classes") && i + 1 < argc) {
      classes.clear();
      for (const char* p = argv[++i]; *p;) {
        classes.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
      }
    } else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  if (group_dir.empty()) group_dir = data_dir / "group";
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::fprintf(stderr, "quad_stats: no CUDA device (%s)\n", cudaGetErrorString(e));
    std::printf("RESULT ok=0 devices=0\n");
    return 1;
  }
  try {
    Timer total;
    const Leech L = kiss::load_leech(data_dir);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    if (x >= static_cast<uint32_t>(N)) throw std::runtime_error("x out of range");
    for (int i : classes)
      if (i < 0 || i >= NC) throw std::runtime_error("--classes entries must be 0..6");

    std::vector<uint8_t> cls(static_cast<std::size_t>(N));
    std::array<uint32_t, NC> val{};
    std::array<uint32_t, NC> yrep{};
    std::array<bool, NC> have{};
    for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
      const int c = cls_of(L, x, v);
      cls[v] = static_cast<uint8_t>(c);
      ++val[static_cast<std::size_t>(c)];
      if (!have[static_cast<std::size_t>(c)]) { have[static_cast<std::size_t>(c)] = true; yrep[static_cast<std::size_t>(c)] = v; }
    }
    std::printf("x=%u valencies:", x);
    for (int c = 0; c < NC; ++c) std::printf(" %u", val[static_cast<std::size_t>(c)]);
    std::printf("  class representatives y_i:");
    for (int c = 0; c < NC; ++c) std::printf(" %u", yrep[static_cast<std::size_t>(c)]);
    std::printf("\n");

    // Gamma, the orbit of x with a Schreier vector, random Stab(x) elements
    // (steps 1-4 of tools/triple_orbitals.cpp, unchanged).
    Timer tg;
    const std::vector<IndexPerm> gens = kiss::co0_generator_perms(L, group_dir);
    Schreier S0;
    S0.set_moves(gens);
    const std::size_t orb = S0.bfs({x});
    std::printf("Gamma = <%zu generators>: orbit of x = %zu of %d, Schreier depth %u (%.0f ms)\n",
                gens.size(), orb, N, S0.max_depth, tg.ms());
    if (orb != static_cast<std::size_t>(N)) throw std::runtime_error("Gamma is not transitive on C");

    kiss::ProductReplacement pr(gens, 10, seed);
    pr.burn_in(200);
    std::vector<IndexPerm> hs;
    auto random_stab_x = [&](IndexPerm& h) {
      const IndexPerm& g = pr.next();
      S0.inv_after(g[x], g, h);
      if (h[x] != x) throw std::runtime_error("random element does not fix x");
    };
    UnionFind ufx(static_cast<std::size_t>(N));
    std::vector<uint32_t> labx;
    uint32_t nx = 0;
    for (int m = 0; m < 4 || nx != NC; ++m) {
      if (m > 64) throw std::runtime_error("Stab(x) orbits did not reach the 7 classes");
      hs.emplace_back();
      random_stab_x(hs.back());
      ufx.add(hs.back());
      nx = ufx.labels(labx);
    }
    for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v)
      if (labx[v] != labx[yrep[cls[v]]]) throw std::runtime_error("Stab(x)-orbit is not a class");
    std::printf("Stab(x): %zu random elements, orbits on C = %u = the classes\n", hs.size(), nx);

    Schreier S1;
    S1.set_moves(hs);
    {
      std::vector<uint32_t> roots(yrep.begin(), yrep.end());
      if (S1.bfs(roots) != static_cast<std::size_t>(N)) throw std::runtime_error("class BFS incomplete");
    }

    // Device buffer for one w-chunk of five-point histograms.
    uint32_t* dM = nullptr;
    const std::size_t chunk_bytes = static_cast<std::size_t>(chunk) * NCLASS4 * sizeof(uint32_t);
    KISS_CUDA_CHECK(cudaMalloc(&dM, chunk_bytes));
    std::vector<uint32_t> hM(static_cast<std::size_t>(chunk) * NCLASS4);

    std::mt19937_64 sprng(seed ^ 0xa5a5a5a5a5a5a5a5ull);
    long long spot_checked = 0, spot_bad = 0;
    long long refine_bad_total = 0, size_sum_bad = 0;
    bool all_exact = true, all_stable = true;

    struct ClassResult {
      int i = 0;
      uint32_t y = 0;
      int pool = 0;
      uint32_t nlab2 = 0;
      std::vector<ZOrbitResult> zorb;
      uint64_t dimT_hist = 0, dimT_labels = 0;
    };
    std::vector<ClassResult> results;

    for (int i : classes) {
      const uint32_t yi = yrep[static_cast<std::size_t>(i)];
      std::printf("== class i=%d (dot %d), y = %u ==\n", i, kClassDot[static_cast<std::size_t>(i)], yi);

      // level 2: random elements of Stab(x, y_i) -> pool + labels lab2.
      Timer t2;
      std::vector<IndexPerm> pool;
      UnionFind uf2(static_cast<std::size_t>(N));
      std::vector<uint32_t> lab2;
      uint32_t nlab2 = static_cast<uint32_t>(N);
      {
        IndexPerm h, k(static_cast<std::size_t>(N));
        int stable = 0;
        while (stable < patience || static_cast<int>(pool.size()) < 3) {
          random_stab_x(h);
          const uint32_t hy = h[yi];
#pragma omp parallel for schedule(static)
          for (int p = 0; p < N; ++p) k[static_cast<std::size_t>(p)] = S1.apply_inv(hy, h[static_cast<std::size_t>(p)]);
          if (k[x] != x || k[yi] != yi) throw std::runtime_error("random element does not fix (x, y_i)");
          pool.push_back(k);
          uf2.add(k);
          std::vector<uint32_t> tmp;
          const uint32_t c = uf2.labels(tmp);
          if (c == nlab2) ++stable; else { stable = 0; nlab2 = c; }
          lab2 = std::move(tmp);
          if (pool.size() > 200) throw std::runtime_error("level-2 label count did not stabilise");
        }
      }
      std::printf("  Stab(x,y): %zu random elements, orbits on C: %u (%.0f ms)\n", pool.size(), nlab2, t2.ms());

      // z-orbit reps (smallest member of each level-2 orbit) and sizes.
      std::vector<uint32_t> zrep(nlab2, UINT32_MAX), zsize(nlab2, 0);
      for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
        const uint32_t o = lab2[v];
        ++zsize[o];
        if (zrep[o] == UINT32_MAX) zrep[o] = v;
      }

      // Schreier transversals within the z-orbits, and a product-replacement
      // walk on the Stab(x, y_i) pool.
      Schreier S2;
      S2.set_moves(pool);
      {
        std::vector<uint32_t> roots(zrep.begin(), zrep.end());
        if (S2.bfs(roots) != static_cast<std::size_t>(N)) throw std::runtime_error("z-orbit BFS incomplete");
      }
      kiss::ProductReplacement prk(pool, 10, seed ^ (0x1234567ull * static_cast<uint64_t>(i + 1)));
      prk.burn_in(100);

      ClassResult CR;
      CR.i = i; CR.y = yi; CR.pool = static_cast<int>(pool.size()); CR.nlab2 = nlab2;

      for (uint32_t r = 0; r < nlab2; ++r) {
        const uint32_t zr = zrep[r];
        ZOrbitResult Z;
        Z.z = zr;
        Z.j = cls[zr];
        Z.k = cls_of(L, yi, zr);
        Z.size = zsize[r];

        // --- GPU: distinct five-point histograms over all w (lower bound) ---
        Timer tgpu;
        std::vector<uint32_t> histid(static_cast<std::size_t>(N), UINT32_MAX);
        std::vector<std::vector<uint32_t>> hists;
        std::unordered_map<uint64_t, std::vector<uint32_t>> bucket;
        double gpu_ms = 0;
        for (uint32_t w0 = 0; w0 < static_cast<uint32_t>(N); w0 += chunk) {
          const uint32_t nw = std::min(chunk, static_cast<uint32_t>(N) - w0);
          Timer tk;
          kiss::cuda::quad_class_histogram(dl, x, yi, zr, w0, nw, dM, 0);
          gpu_ms += tk.ms();
          KISS_CUDA_CHECK(cudaMemcpy(hM.data(), dM, static_cast<std::size_t>(nw) * NCLASS4 * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost));
          std::vector<uint64_t> hashes(nw);
#pragma omp parallel for schedule(static)
          for (int t = 0; t < static_cast<int>(nw); ++t)
            hashes[static_cast<std::size_t>(t)] = hash_row(hM.data() + static_cast<std::size_t>(t) * NCLASS4, NCLASS4);
          for (uint32_t t = 0; t < nw; ++t) {
            const uint32_t* row = hM.data() + static_cast<std::size_t>(t) * NCLASS4;
            uint32_t id = UINT32_MAX;
            auto& cand = bucket[hashes[t]];
            for (uint32_t cid : cand)
              if (std::memcmp(row, hists[cid].data(), NCLASS4 * sizeof(uint32_t)) == 0) { id = cid; break; }
            if (id == UINT32_MAX) {
              id = static_cast<uint32_t>(hists.size());
              hists.emplace_back(row, row + NCLASS4);
              cand.push_back(id);
            }
            histid[w0 + t] = id;
          }
          // CPU spot check: recompute a few random rows of this chunk.
          for (int s = 0; s < spot; ++s) {
            std::uniform_int_distribution<uint32_t> d(0, nw - 1);
            const uint32_t w = w0 + d(sprng);
            std::array<uint32_t, static_cast<std::size_t>(NCLASS4)> ref{};
            for (uint32_t v = 0; v < static_cast<uint32_t>(N); ++v) {
              const int a = cls[v];
              const int b = cls_of(L, yi, v);
              const int cc = cls_of(L, zr, v);
              const int dd = cls_of(L, w, v);
              ++ref[static_cast<std::size_t>(((a * NC + b) * NC + cc) * NC + dd)];
            }
            ++spot_checked;
            if (std::memcmp(ref.data(), hM.data() + static_cast<std::size_t>(w - w0) * NCLASS4,
                            NCLASS4 * sizeof(uint32_t)) != 0) ++spot_bad;
          }
        }
        Z.hist_orbits = static_cast<uint32_t>(hists.size());
        Z.gpu_ms = gpu_ms;
        Z.host_ms = tgpu.ms() - gpu_ms;

        // --- CPU: labels = orbits of explicit Stab(x, y_i, z_r) elements ---
        Timer tlab;
        UnionFind uf3(static_cast<std::size_t>(N));
        std::vector<uint32_t> lab3;
        uint32_t nlab3 = static_cast<uint32_t>(N);
        {
          IndexPerm m(static_cast<std::size_t>(N));
          int stable = 0, made = 0;
          while (true) {
            const IndexPerm& k = prk.next();
            const uint32_t kz = k[zr];
#pragma omp parallel for schedule(static)
            for (int p = 0; p < N; ++p) m[static_cast<std::size_t>(p)] = S2.apply_inv(kz, k[static_cast<std::size_t>(p)]);
            if (m[x] != x || m[yi] != yi || m[zr] != zr)
              throw std::runtime_error("random element does not fix (x, y, z)");
            ++made;
            uf3.add(m);
            std::vector<uint32_t> tmp;
            const uint32_t c = uf3.labels(tmp);
            if (c == nlab3) ++stable; else { stable = 0; nlab3 = c; }
            lab3 = std::move(tmp);
            if (nlab3 == Z.hist_orbits) break;              // exact: bounds met
            if (stable >= patience) break;
            if (made >= 200) { Z.stabilised = false; break; }
          }
          Z.stab3_elements = made;
        }
        Z.label_orbits = nlab3;
        Z.exact = (nlab3 == Z.hist_orbits);
        Z.label_ms = tlab.ms();

        // labels must refine histogram classes (each label has one histogram).
        {
          std::vector<uint32_t> lab_hist(nlab3, UINT32_MAX);
          for (uint32_t w = 0; w < static_cast<uint32_t>(N); ++w) {
            const uint32_t o = lab3[w];
            if (lab_hist[o] == UINT32_MAX) lab_hist[o] = histid[w];
            else if (lab_hist[o] != histid[w]) ++Z.refine_violations;
          }
          refine_bad_total += Z.refine_violations;
        }

        // w-orbit data from the labels.
        Z.w_orbits.assign(nlab3, WOrbit{});
        {
          std::vector<bool> seen(nlab3, false);
          uint64_t sum = 0;
          for (uint32_t w = 0; w < static_cast<uint32_t>(N); ++w) {
            const uint32_t o = lab3[w];
            WOrbit& W = Z.w_orbits[o];
            ++W.size;
            if (!seen[o]) {
              seen[o] = true;
              W.w = w;                                       // smallest member
              W.a = cls[w];
              W.b = cls_of(L, yi, w);
              W.c = cls_of(L, zr, w);
              const std::array<int, 6> pc{i, Z.j, Z.k, W.a, W.b, W.c};
              W.forbidden = false;
              W.distinct4 = true;
              for (int p : pc) {
                if (p == kConflict) W.forbidden = true;
                if (p == kIdentity) W.distinct4 = false;
              }
            }
          }
          for (const WOrbit& W : Z.w_orbits) sum += W.size;
          if (sum != static_cast<uint64_t>(N)) ++size_sum_bad;
        }

        all_exact = all_exact && Z.exact;
        all_stable = all_stable && Z.stabilised;
        CR.dimT_hist += Z.hist_orbits;
        CR.dimT_labels += Z.label_orbits;
        std::printf("  z-orbit %2u: z=%6u cell (%d,%d,%d) size %6u -> w-orbits: hist %4u, labels %4u %s "
                    "(%d stab elements; gpu %.0f ms, host %.0f ms, labels %.0f ms)\n",
                    r, zr, i, Z.j, Z.k, Z.size, Z.hist_orbits, Z.label_orbits,
                    Z.exact ? "EXACT" : (Z.stabilised ? "bracket" : "UNSTABLE"), Z.stab3_elements,
                    Z.gpu_ms, Z.host_ms, Z.label_ms);
        CR.zorb.push_back(std::move(Z));
      }
      std::printf("  dim T(x, y_%d): hist (lower) %llu, labels (upper) %llu%s\n", i,
                  static_cast<unsigned long long>(CR.dimT_hist),
                  static_cast<unsigned long long>(CR.dimT_labels),
                  CR.dimT_hist == CR.dimT_labels ? " -> EXACT" : "");
      results.push_back(std::move(CR));
    }
    KISS_CUDA_CHECK(cudaFree(dM));

    // ---- summary totals over the computed classes ----
    uint64_t tot_lo = 0, tot_hi = 0, adm = 0, adm_distinct = 0, cliques4 = 0;
    for (const ClassResult& CR : results) {
      tot_lo += CR.dimT_hist;
      tot_hi += CR.dimT_labels;
      for (const ZOrbitResult& Z : CR.zorb)
        for (const WOrbit& W : Z.w_orbits) {
          if (!W.forbidden) {
            ++adm;
            if (W.distinct4) ++adm_distinct;
          }
          if (CR.i == kConflict && Z.j == kConflict && Z.k == kConflict &&
              W.a == kConflict && W.b == kConflict && W.c == kConflict) ++cliques4;
        }
    }
    std::printf("summary over %zu classes: ordered-4-tuple orbits in [%llu, %llu]; admissible (no dot-16 pair) %llu; "
                "admissible with 4 distinct points %llu; all-conflict (4-clique) orbits %llu\n",
                results.size(), static_cast<unsigned long long>(tot_lo), static_cast<unsigned long long>(tot_hi),
                static_cast<unsigned long long>(adm), static_cast<unsigned long long>(adm_distinct),
                static_cast<unsigned long long>(cliques4));
    std::printf("spot checks: %lld GPU rows re-computed on the CPU, %lld mismatches; refine violations %lld; "
                "orbit-size sums != N: %lld\n", spot_checked, spot_bad, refine_bad_total, size_sum_bad);

    // ---- JSON ----
    std::filesystem::create_directories(out.parent_path().empty() ? "." : out.parent_path());
    std::ofstream f(out);
    if (!f) throw std::runtime_error("cannot write " + out.string());
    f << "{\n  \"task\": \"B2\",\n";
    f << "  \"description\": \"Quadruple orbit statistics of the Leech minimal-vector scheme, one level below "
         "T2.3 (data/scheme/orbitals.json). For x = C[x] and one representative y_i per pair class, the orbits of "
         "Stab(x, y_i) on z (level 2, labels of explicit random stabiliser elements) and, per z-orbit "
         "representative z_r, the orbits of Stab(x, y_i, z_r) on w, bracketed by the number of distinct GPU "
         "five-point histograms (lower bound; cuda/quad_stats.cu) and the label count of explicit random "
         "Stab(x,y,z) elements (upper bound); exact where they agree. dim T(x, y_i) = sum over z-orbits = the "
         "number of Stab(x, y_i)-orbitals on ordered pairs (z, w) = the PSD block dimension of a four-point SDP "
         "in the T2.3 style. w_orbits rows are [w_rep, size, class(x,w), class(y,w), class(z,w), forbidden, "
         "distinct4] with forbidden = some pairwise inner product of (x, y_i, z_r, w_rep) equal to 16 and "
         "distinct4 = all four points pairwise distinct.\",\n";
    f << "  \"N\": " << N << ",\n  \"x\": " << x << ",\n  \"seed\": " << seed << ",\n";
    f << "  \"classes_dots\": [-32, -16, -8, 0, 8, 16, 32],\n  \"identity_class\": 6,\n  \"conflict_class\": 5,\n";
    f << "  \"valencies\": [";
    for (int c = 0; c < NC; ++c) f << val[static_cast<std::size_t>(c)] << (c + 1 < NC ? ", " : "");
    f << "],\n  \"representatives\": [";
    for (int c = 0; c < NC; ++c) f << yrep[static_cast<std::size_t>(c)] << (c + 1 < NC ? ", " : "");
    f << "],\n  \"computed_classes\": [";
    for (std::size_t q = 0; q < results.size(); ++q) f << results[q].i << (q + 1 < results.size() ? ", " : "");
    f << "],\n";
    f << "  \"spot_checks\": {\"rows\": " << spot_checked << ", \"mismatches\": " << spot_bad << "},\n";
    f << "  \"checks\": {\"refine_violations\": " << refine_bad_total << ", \"size_sum_bad\": " << size_sum_bad << "},\n";
    f << "  \"totals\": {\"ordered_orbits_lower\": " << tot_lo << ", \"ordered_orbits_upper\": " << tot_hi
      << ", \"admissible\": " << adm << ", \"admissible_distinct4\": " << adm_distinct
      << ", \"clique4_orbits\": " << cliques4 << "},\n";
    f << "  \"classes\": [\n";
    for (std::size_t q = 0; q < results.size(); ++q) {
      const ClassResult& CR = results[q];
      f << "    {\"i\": " << CR.i << ", \"dot\": " << kClassDot[static_cast<std::size_t>(CR.i)]
        << ", \"y\": " << CR.y << ", \"stab_xy_elements\": " << CR.pool
        << ", \"z_orbit_count\": " << CR.nlab2
        << ", \"dimT_hist\": " << CR.dimT_hist << ", \"dimT_labels\": " << CR.dimT_labels << ",\n";
      f << "     \"z_orbits\": [\n";
      for (std::size_t r = 0; r < CR.zorb.size(); ++r) {
        const ZOrbitResult& Z = CR.zorb[r];
        f << "      {\"z\": " << Z.z << ", \"j\": " << Z.j << ", \"k\": " << Z.k << ", \"size\": " << Z.size
          << ", \"hist_orbits\": " << Z.hist_orbits << ", \"label_orbits\": " << Z.label_orbits
          << ", \"exact\": " << (Z.exact ? "true" : "false")
          << ", \"stabilised\": " << (Z.stabilised ? "true" : "false")
          << ", \"stab3_elements\": " << Z.stab3_elements << ",\n       \"w_orbits\": [";
        for (std::size_t o = 0; o < Z.w_orbits.size(); ++o) {
          const WOrbit& W = Z.w_orbits[o];
          f << (o ? ", " : "") << "[" << W.w << ", " << W.size << ", " << W.a << ", " << W.b << ", " << W.c
            << ", " << (W.forbidden ? 1 : 0) << ", " << (W.distinct4 ? 1 : 0) << "]";
        }
        f << "]}" << (r + 1 < CR.zorb.size() ? ",\n" : "\n");
      }
      f << "     ]}" << (q + 1 < results.size() ? ",\n" : "\n");
    }
    f << "  ]\n}\n";
    f.close();
    std::printf("wrote %s\n", out.string().c_str());

    const bool ok = spot_bad == 0 && refine_bad_total == 0 && size_sum_bad == 0 && all_stable;
    std::printf("RESULT ok=%d all_exact=%d classes=%zu dimT_lo=%llu dimT_hi=%llu adm=%llu adm4=%llu cliques4=%llu "
                "spot=%lld/%lld total_ms=%.0f out=%s\n",
                ok ? 1 : 0, all_exact ? 1 : 0, results.size(), static_cast<unsigned long long>(tot_lo),
                static_cast<unsigned long long>(tot_hi), static_cast<unsigned long long>(adm),
                static_cast<unsigned long long>(adm_distinct), static_cast<unsigned long long>(cliques4),
                spot_bad, spot_checked, total.ms(), out.string().c_str());
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "quad_stats: %s\n", ex.what());
    std::printf("RESULT ok=0 error=1\n");
    return 1;
  }
}

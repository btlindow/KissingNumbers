// frames — frame-structured search on the antipodal classes of the Leech
// minimal vectors (docs/design.md T3.3, README §3 W2c). Library: kiss/frames.h.
//
//   frames count   [--class C=0] [--k K=24] [--shape S=-1] [--coords M=24]
//                  [--probes P=200000] [--time-limit SEC=0] [--seed X]
//       Neighbourhood of class C in the orthogonality graph (optionally
//       restricted to classes of shape S and to support inside coordinates
//       < M), Knuth estimate of the number of K-cliques through C, and — if
//       --time-limit > 0 — the exact ordered DFS count (a lower bound when
//       stopped). Total = f_C · 98280 / K by vertex transitivity.
//   frames crosses <S.txt> [--no-8] [--cover] [--cover-time SEC=300]
//       Cross structure of a vector set: orthogonality degrees of its classes,
//       maximal cliques by size, the full frames, the number of 8-cliques, and
//       with --cover the exact cover "one frame + 8-cliques" for each full frame.
//   frames extend  <S.txt> [--tiers 4,6,8] [--tmax T] [--kmin K=2] [--seeds N]
//                  [--time-limit SEC=20] [--min-gain G=-8] [--frames-only] [--write-plateau]
//       Frame-level swaps around an independent set: for every class c ∉ X
//       with conf(c) = t (t running over the tiers), enumerate the orthogonal
//       cliques through c inside {x ∈ X} ∪ {d ∉ X : conf(d) ≤ tmax (default t)}
//       and maximise gain = |Q \ X| − |Conf(Q)|. Positive gain ⇒ the new set is
//       written to runs/found/ and verified.
//   frames greedy  [--restarts R=100] [--seed X=1] [--no-frames-first] [--min-clique m=1]
//       Greedy frame unions from a random frame; size distribution.
//   frames ls      <S.txt> | --greedy SEED  [--time-limit SEC=60] [--tmax T=6]
//                  [--max-drop D=2] [--tenure T=50] [--seed X=1] [--target n=249]
//       Clique-move tabu local search on classes.
//
// Every subcommand ends with a `RESULT ok=1 ...` line.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kiss/frames.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/verify.h"

namespace {

using namespace kiss;

void usage() {
  std::fprintf(stderr,
               "usage: frames count|crosses|extend|greedy|ls ... (see the header of tools/frames.cpp)\n");
}

std::string utc_stamp() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

struct Args {
  std::vector<std::string> pos;
  std::map<std::string, std::string> opt;
  bool has(const std::string& k) const { return opt.count(k) > 0; }
  long geti(const std::string& k, long def) const {
    auto it = opt.find(k);
    return it == opt.end() ? def : std::stol(it->second);
  }
  double getd(const std::string& k, double def) const {
    auto it = opt.find(k);
    return it == opt.end() ? def : std::stod(it->second);
  }
  std::string gets(const std::string& k, const std::string& def) const {
    auto it = opt.find(k);
    return it == opt.end() ? def : it->second;
  }
};

Args parse(int argc, char** argv) {
  Args a;
  static const char* flags[] = {"--no-frames-first", "--frames-only", "--exact", "--write-plateau", "--no-8", "--cover", nullptr};
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      bool flag = false;
      for (const char** f = flags; *f; ++f)
        if (s == *f) flag = true;
      if (flag) {
        a.opt[s] = "1";
      } else if (i + 1 < argc) {
        a.opt[s] = argv[++i];
      } else {
        throw std::runtime_error("missing value for " + s);
      }
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

std::string hist_str(const std::map<int, long>& h) {
  std::string s;
  for (const auto& kv : h) s += (s.empty() ? "" : " ") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
  return s;
}
std::string hist_str_ii(const std::map<int, int>& h) {
  std::string s;
  for (const auto& kv : h) s += (s.empty() ? "" : " ") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
  return s;
}

std::string vec_str(const std::vector<uint32_t>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
  return s;
}

Leech load(const Args& a) {
  const std::string dir = a.gets("--data", "data");
  if (std::filesystem::exists(std::filesystem::path(dir) / "leech_min.i8")) return load_leech(dir);
  return generate_leech();
}

// Write a class set as vectors, verify, report. Returns the file name.
std::string save_and_verify(const Leech& L, const Classes& K, const std::vector<uint32_t>& X,
                            const std::string& tag, const std::string& note) {
  const std::vector<Vec> S = classes_to_vectors(L, K, X);
  std::filesystem::create_directories("runs/found");
  const std::string file =
      "runs/found/S_" + std::to_string(S.size()) + "_" + utc_stamp() + "_" + tag + ".txt";
  write_set(file, S, "T3.3 frames (" + tag + "): " + note);
  const VerifyResult vr = verify_independent(L, S);
  std::printf("WROTE %s (%zu vectors): verify_independent -> ok=%d %s\n", file.c_str(), S.size(), vr.ok ? 1 : 0,
              vr.message.c_str());
  if (S.size() > 496) {
    std::printf("\n**************** |S| = %zu > 496 — NEW RECORD CANDIDATE: %s ****************\n\n", S.size(),
                file.c_str());
    std::printf("verify with: build/t33/tools/verify_s %s ; .venv/bin/python python/verify_S.py %s\n", file.c_str(),
                file.c_str());
  }
  return file;
}

// ---------------------------------------------------------------------------
// count
// ---------------------------------------------------------------------------
int cmd_count(const Args& a) {
  const uint32_t c = static_cast<uint32_t>(a.geti("--class", 0));
  const int k = static_cast<int>(a.geti("--k", FRAME_SIZE));
  const int shape = static_cast<int>(a.geti("--shape", -1));
  const int coords = static_cast<int>(a.geti("--coords", DIM));
  const long probes = a.geti("--probes", 200000);
  const double tl = a.getd("--time-limit", 0);
  const bool exact = a.has("--exact");
  const uint64_t seed = static_cast<uint64_t>(a.geti("--seed", 1));
  const auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

  const Leech L = load(a);
  const Classes K = make_classes(L);
  std::printf("class %u: rep vertex %u = ", c, K.rep[c]);
  for (int i = 0; i < DIM; ++i) std::printf("%d ", L.C[K.rep[c]][static_cast<std::size_t>(i)]);
  std::printf("(shape %d)\n", leech_shape(L.C[K.rep[c]]));

  Bitset allowed(NCLASS);
  allowed.fill();
  if (shape >= 0) allowed.and_with(classes_of_shape(L, K, shape));
  if (coords < DIM) {
    for (int b = 0; b < NCLASS; ++b) {
      const Vec& v = L.C[K.rep[static_cast<std::size_t>(b)]];
      for (int i = coords; i < DIM; ++i)
        if (v[static_cast<std::size_t>(i)] != 0) {
          allowed.reset(b);
          break;
        }
    }
  }
  std::printf("restriction: shape=%d coords<%d -> %d admissible classes\n", shape, coords, allowed.count());
  const Neighbourhood Nb = build_neighbourhood(L, K, c, &allowed);
  long dmin = 1L << 40, dmax = 0, dsum = 0;
  for (int i = 0; i < Nb.size(); ++i) {
    const long d = Nb.adj[static_cast<std::size_t>(i)].count();
    dmin = std::min(dmin, d);
    dmax = std::max(dmax, d);
    dsum += d;
  }
  std::printf("neighbourhood of class %u: %d vertices, degree min/mean/max = %ld / %.1f / %ld, built in %.1f s\n", c,
              Nb.size(), dmin, Nb.size() ? static_cast<double>(dsum) / Nb.size() : 0.0, dmax, now());

  // Knuth estimate
  const TreeEstimate est = estimate_cliques_through(Nb, k, probes, seed);
  std::printf("random-descent estimate (%ld probes): %d-cliques through class %u ≈ %.6g ± %.3g (rel. se %.2f%%); ordered-DFS nodes ≲ %.4g\n",
              probes, k, c, est.leaves, est.leaves_se, est.leaves > 0 ? 100 * est.leaves_se / est.leaves : 0.0,
              est.nodes);
  std::string md;
  for (const auto& kv : est.mean_candidates_by_depth)
    md += (md.empty() ? "" : " ") + std::to_string(kv.first) + ":" + std::to_string(static_cast<long>(kv.second));
  std::printf("  mean |P| by depth (over probes): %s\n", md.c_str());
  std::string cd;
  for (const auto& kv : est.cliques_by_depth) {
    if (kv.first + 1 >= k) continue;   // the k-clique count is the leaves estimate above
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s%d:%.4g", cd.empty() ? "" : " ", kv.first + 1, kv.second);
    cd += buf;
  }
  std::printf("  estimated j-cliques through class %u (j = size incl. the centre): %s\n", c, cd.c_str());
  const bool restricted = shape >= 0 || coords < DIM;
  const double total_est = est.leaves * static_cast<double>(NCLASS) / k;
  const double total_se = est.leaves_se * static_cast<double>(NCLASS) / k;
  if (!restricted)
    std::printf("  => total %d-cliques ≈ %.6g ± %.3g (× %d/%d, vertex transitivity)\n", k, total_est, total_se, NCLASS, k);

  // exact DFS
  FrameCount fc;
  bool ran_exact = false;
  if (exact || tl > 0) {
    ran_exact = true;
    fc = count_cliques_through(Nb, k, exact ? 0 : tl);
    std::printf("exact DFS: %d-cliques through class %u = %ld (%s), nodes %ld, first level %ld/%ld, %.1f s\n", k, c,
                fc.frames, fc.complete ? "complete" : "TIME LIMIT — lower bound", fc.nodes, fc.first_level_done,
                fc.first_level_total, fc.seconds);
    if (fc.complete && !restricted) {
      const long double total = static_cast<long double>(fc.frames) * NCLASS / k;
      std::printf("  => total %d-cliques = %ld × %d / %d = %.0Lf\n", k, fc.frames, NCLASS, k, total);
    }
  }
  // reference figures
  if (k == FRAME_SIZE) {
    const long double m24 = double_factorial_odd(24);
    std::printf("reference: Conway crosses (README) = 8292375; (±4,±4)-only frames = 23!! = %.0Lf "
                "(through one (±4,±4) class: 21!! = %.0Lf)\n",
                m24, double_factorial_odd(22));
  }
  if (shape == 2 && k == coords) {
    std::printf("reference: (±4,±4)-only %d-cliques on %d coordinates through class %u = (%d)!! = %.0Lf; "
                "all such cliques = (%d)!! = %.0Lf\n",
                k, coords, c, coords - 3, double_factorial_odd(coords - 2), coords - 1, double_factorial_odd(coords));
  }
  std::printf("RESULT ok=1 cmd=count class=%u k=%d shape=%d coords=%d nbhd=%d est=%.6g est_se=%.3g est_total=%.6g "
              "exact=%ld complete=%d nodes=%ld seconds=%.1f\n",
              c, k, shape, coords, Nb.size(), est.leaves, est.leaves_se, total_est, ran_exact ? fc.frames : -1L,
              ran_exact ? (fc.complete ? 1 : 0) : -1, fc.nodes, now());
  return 0;
}

// ---------------------------------------------------------------------------
// crosses
// ---------------------------------------------------------------------------
// Exact cover of a class pool by `block`-cliques (T3.3 --cover): repeatedly take
// the lowest uncovered vertex and branch over the block-cliques through it inside
// the uncovered set. Ordered enumeration (all other members exceed that vertex,
// which is the minimum of the available set) so no clique is tried twice.
struct CoverSearch {
  const PoolGraph& G;
  int block;
  double limit;
  long nodes = 0, placements = 0;
  bool stop = false;
  std::vector<std::vector<int>> blocks;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

  double elapsed() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

  bool extend_block(const Bitset& avail, const Bitset& cand, std::vector<int>& cur) {
    if (stop) return false;
    ++nodes;
    if ((nodes & 1023) == 0 && limit > 0 && elapsed() > limit) {
      stop = true;
      return false;
    }
    if (static_cast<int>(cur.size()) == block) {
      Bitset rest = avail;
      for (int v : cur) rest.reset(v);
      blocks.push_back(cur);
      if (solve(rest)) return true;
      blocks.pop_back();
      return false;
    }
    const int need = block - static_cast<int>(cur.size());
    if (cand.count() < need) return false;
    Bitset next(G.size());
    for (int v = cand.first(); v >= 0; v = cand.next(v + 1)) {
      next.intersect_above(cand, G.adj[static_cast<std::size_t>(v)], v);
      cur.push_back(v);
      if (extend_block(avail, next, cur)) return true;
      cur.pop_back();
      if (stop) return false;
    }
    return false;
  }

  // Branch on the uncovered vertex of smallest uncovered degree; prune as soon as
  // some uncovered vertex has fewer than block−1 uncovered neighbours (T1.3-convert §3).
  bool solve(const Bitset& avail) {
    if (stop) return false;
    int u = -1, best = 1 << 30;
    Bitset t(G.size());
    for (int v = avail.first(); v >= 0; v = avail.next(v + 1)) {
      t = G.adj[static_cast<std::size_t>(v)];
      t.and_with(avail);
      const int d = t.count();
      if (d < block - 1) return false;
      if (d < best) {
        best = d;
        u = v;
      }
    }
    if (u < 0) return true;
    Bitset cand = G.adj[static_cast<std::size_t>(u)];
    cand.and_with(avail);
    std::vector<int> cur{u};
    ++placements;
    return extend_block(avail, cand, cur);
  }
};

int cmd_crosses(const Args& a) {
  if (a.pos.empty()) throw std::runtime_error("crosses: need <S.txt>");
  const Leech L = load(a);
  const Classes K = make_classes(L);
  const std::vector<Vec> S = read_set(a.pos[0]);
  const VerifyResult vr = verify_independent(L, S);
  std::printf("%s: %zu vectors, verify_independent ok=%d (%s)\n", a.pos[0].c_str(), S.size(), vr.ok ? 1 : 0,
              vr.message.c_str());
  const CrossReport r = cross_structure(L, K, S, !a.has("--no-8"));
  std::printf("classes %zu, antipodal %d\n", r.classes.size(), r.antipodal ? 1 : 0);
  std::printf("orthogonality degrees: %s\n", hist_str(r.degree_hist).c_str());
  std::printf("maximal cliques: %ld, sizes %s; clique number %d\n", r.maximal_total, hist_str(r.maximal_by_size).c_str(),
              r.clique_number);
  std::printf("full frames (24-cliques): %zu\n", r.frames.size());
  for (std::size_t i = 0; i < r.frames.size(); ++i) {
    std::printf("  frame %zu: classes %s\n", i, vec_str(r.frames[i]).c_str());
  }
  // pairwise disjointness of the frames
  bool disjoint = true;
  for (std::size_t i = 0; i < r.frames.size(); ++i)
    for (std::size_t j = i + 1; j < r.frames.size(); ++j) {
      std::vector<uint32_t> inter;
      std::set_intersection(r.frames[i].begin(), r.frames[i].end(), r.frames[j].begin(), r.frames[j].end(),
                            std::back_inserter(inter));
      if (!inter.empty()) disjoint = false;
    }
  std::printf("frames pairwise disjoint: %d\n", disjoint ? 1 : 0);
  if (r.cliques8 >= 0) std::printf("8-cliques (all): %ld\n", r.cliques8);

  // --cover: reproduce the README §1.4 decomposition "one full frame + 28 octuples"
  // by exact cover of the remaining classes with 8-cliques (T1.3-convert §3.2, in C++).
  int covers_found = 0;
  if (a.has("--cover") && !r.frames.empty()) {
    const double cover_limit = a.getd("--cover-time", 300);
    const PoolGraph G = induced_orthogonality(L, K, r.classes);
    const int n = G.size();
    std::vector<int> idx_of(static_cast<std::size_t>(NCLASS), -1);
    for (int i = 0; i < n; ++i) idx_of[G.pool[static_cast<std::size_t>(i)]] = i;
    for (std::size_t fi = 0; fi < r.frames.size(); ++fi) {
      Bitset avail0(n);
      for (int i = 0; i < n; ++i) avail0.set(i);
      for (uint32_t c : r.frames[fi]) avail0.reset(idx_of[c]);
      CoverSearch cs{G, 8, cover_limit, 0, 0, false, {}, std::chrono::steady_clock::now()};
      const bool ok = cs.solve(avail0);
      std::printf("  cover with frame %zu + %d-cliques: %s (%zu blocks, %ld branch nodes, %ld DFS nodes, %.1f s)\n",
                  fi, cs.block, ok ? "FOUND" : (cs.stop ? "time limit" : "impossible"), cs.blocks.size(),
                  cs.placements, cs.nodes, cs.elapsed());
      if (ok) {
        ++covers_found;
        std::printf("    blocks:");
        for (const std::vector<int>& b : cs.blocks) {
          std::printf(" {");
          for (std::size_t j = 0; j < b.size(); ++j)
            std::printf("%s%u", j ? "," : "", G.pool[static_cast<std::size_t>(b[j])]);
          std::printf("}");
        }
        std::printf("\n");
      }
    }
  }
  std::printf("RESULT ok=1 cmd=crosses size=%zu classes=%zu maximal=%ld clique_number=%d frames=%zu disjoint=%d "
              "cliques8=%ld covers=%d seconds=%.2f\n",
              S.size(), r.classes.size(), r.maximal_total, r.clique_number, r.frames.size(), disjoint ? 1 : 0,
              r.cliques8, covers_found, r.seconds);
  return 0;
}

// ---------------------------------------------------------------------------
// extend
// ---------------------------------------------------------------------------
int cmd_extend(const Args& a) {
  if (a.pos.empty()) throw std::runtime_error("extend: need <S.txt>");
  const auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  const Leech L = load(a);
  const Classes K = make_classes(L);
  const std::vector<Vec> S = read_set(a.pos[0]);
  const VerifyResult vr = verify_independent(L, S);
  if (!vr.ok) throw std::runtime_error("input is not an independent set: " + vr.message);
  const std::vector<uint32_t> X = set_classes(L, K, S);
  std::printf("%s: %zu vectors, %zu classes, antipodal %d\n", a.pos[0].c_str(), S.size(), X.size(),
              is_antipodal(S) ? 1 : 0);
  const std::vector<uint16_t> conf = class_conflicts(L, K, X);
  Bitset inX(NCLASS);
  for (uint32_t x : X) inX.set(static_cast<int>(x));
  std::map<int, long> ch;
  for (int b = 0; b < NCLASS; ++b)
    if (!inX.test(b)) ++ch[conf[static_cast<std::size_t>(b)]];
  std::printf("class conflict histogram (classes outside X): %s\n", hist_str(ch).c_str());

  std::vector<int> tiers;
  {
    std::stringstream ss(a.gets("--tiers", "4,6,8"));
    std::string tok;
    while (std::getline(ss, tok, ',')) tiers.push_back(std::stoi(tok));
  }
  const long tmax_fixed = a.geti("--tmax", -1);
  const long max_seeds = a.geti("--seeds", 1000000);
  ExtendOptions eo;
  eo.kmin = static_cast<int>(a.geti("--kmin", 2));
  eo.min_gain = static_cast<int>(a.geti("--min-gain", -8));
  eo.time_limit_s = a.getd("--time-limit", 20);
  eo.node_limit = a.geti("--node-limit", 0);
  eo.frames_only = a.has("--frames-only");
  const bool write_plateau = a.has("--write-plateau");   // apply the first gain-0 hit with ≥ 24 added classes
  std::string plateau_file;

  int global_best = -1000000;
  ExtendHit global_hit;
  long total_seeds = 0, total_frames = 0, incomplete = 0;
  std::map<int, long> gain_all;
  for (int t : tiers) {
    std::vector<uint32_t> seeds;
    for (int b = 0; b < NCLASS; ++b)
      if (!inX.test(b) && conf[static_cast<std::size_t>(b)] == t) seeds.push_back(static_cast<uint32_t>(b));
    if (static_cast<long>(seeds.size()) > max_seeds) seeds.resize(static_cast<std::size_t>(max_seeds));
    eo.tmax = tmax_fixed >= 0 ? static_cast<int>(tmax_fixed) : t;
    std::printf("\n== tier conf=%d: %zu seed classes, pool cap tmax=%d, kmin=%d, min_gain=%d, %.0f s/seed ==\n", t,
                seeds.size(), eo.tmax, eo.kmin, eo.min_gain, eo.time_limit_s);
    int tier_best = -1000000;
    std::map<int, long> tier_gain;
    std::map<int, long> pool_hist;
    long tier_frames = 0, tier_nodes = 0;
    double tier_t0 = now();
    for (std::size_t si = 0; si < seeds.size(); ++si) {
      const ExtendResult er = extend_from_class(L, K, X, conf, seeds[si], eo);
      ++total_seeds;
      tier_nodes += er.nodes;
      tier_frames += er.frames;
      ++pool_hist[er.pool_outside];
      if (!er.complete) ++incomplete;
      for (const auto& kv : er.gain_hist) {
        tier_gain[kv.first] += kv.second;
        gain_all[kv.first] += kv.second;
      }
      if (er.best_gain > tier_best) tier_best = er.best_gain;
      if (er.best_gain > global_best || (er.best_gain == global_best && er.best.added > global_hit.added)) {
        global_best = er.best_gain;
        global_hit = er.best;
      }
      if (si < 8 || er.best_gain >= 0 || !er.complete) {
        std::printf("  seed %u (conf %d): pool %d (%d outside X), nodes %ld, cliques %ld, best gain %d "
                    "(added %d, removed %d, |Q|=%zu), frames %ld (best frame gain %d), %s, %.2f s\n",
                    er.seed, conf[er.seed], er.pool_size, er.pool_outside, er.nodes, er.cliques, er.best_gain,
                    er.best.added, er.best.removed, er.best.clique.size(), er.frames,
                    er.frames ? er.best_frame_gain : 0, er.complete ? "complete" : "INCOMPLETE", er.seconds);
        if (si < 8) std::printf("    best gain by |Q\\X|: %s\n", hist_str_ii(er.best_gain_by_size).c_str());
      }
      if (write_plateau && plateau_file.empty() && er.best_gain == 0 && er.best.added >= FRAME_SIZE) {
        const std::vector<uint32_t> X2 = apply_extend_hit(L, K, X, er.best);
        const std::vector<Vec> S2 = classes_to_vectors(L, K, X2);
        std::filesystem::create_directories("runs/frames");
        plateau_file = "runs/frames/plateau_" + std::to_string(S2.size()) + "_seed" + std::to_string(er.seed) + ".txt";
        write_set(plateau_file, S2, "T3.3 frames: gain-0 frame swap of " + a.pos[0] + " at seed class " +
                                        std::to_string(er.seed) + " (removed " + std::to_string(er.best.removed) +
                                        " classes, added " + std::to_string(er.best.added) + ")");
        const VerifyResult v2 = verify_independent(L, S2);
        std::printf("  plateau frame swap written to %s (%zu vectors, verify ok=%d); removed classes:", plateau_file.c_str(),
                    S2.size(), v2.ok ? 1 : 0);
        for (uint32_t x : X)
          if (std::find(X2.begin(), X2.end(), x) == X2.end()) std::printf(" %u", x);
        std::printf("\n");
      }
      if (er.best_gain > 0) {
        std::printf("  *** POSITIVE GAIN %d at seed %u: clique %s ***\n", er.best_gain, er.seed,
                    vec_str(er.best.clique).c_str());
        const std::vector<uint32_t> X2 = apply_extend_hit(L, K, X, er.best);
        save_and_verify(L, K, X2, "frames",
                        "extend from " + a.pos[0] + " seed class " + std::to_string(er.seed) + " gain " +
                            std::to_string(er.best_gain));
      }
    }
    total_frames += tier_frames;
    std::printf("tier conf=%d summary: seeds %zu, best gain %d, gain histogram %s\n", t, seeds.size(), tier_best,
                hist_str(tier_gain).c_str());
    std::printf("  outsider pool sizes: %s\n", hist_str(pool_hist).c_str());
    std::printf("  frames through seeds (inside pools): %ld, DFS nodes %ld, %.1f s\n", tier_frames, tier_nodes,
                now() - tier_t0);
  }
  std::printf("\nbest over all tiers: gain %d (seed %u, added %d, removed %d, clique [%s])\n", global_best,
              global_hit.seed, global_hit.added, global_hit.removed, vec_str(global_hit.clique).c_str());
  std::printf("gain histogram (all tiers): %s\n", hist_str(gain_all).c_str());
  std::printf("RESULT ok=1 cmd=extend size=%zu classes=%zu seeds=%ld best_gain=%d best_added=%d best_removed=%d "
              "frames=%ld incomplete=%ld improved=%d seconds=%.1f\n",
              S.size(), X.size(), total_seeds, global_best, global_hit.added, global_hit.removed, total_frames,
              incomplete, global_best > 0 ? 1 : 0, now());
  return 0;
}

// ---------------------------------------------------------------------------
// greedy
// ---------------------------------------------------------------------------
int cmd_greedy(const Args& a) {
  const auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  const Leech L = load(a);
  const Classes K = make_classes(L);
  const long restarts = a.geti("--restarts", 100);
  const uint64_t seed0 = static_cast<uint64_t>(a.geti("--seed", 1));
  GreedyOptions go;
  go.frames_first = !a.has("--no-frames-first");
  go.min_clique = static_cast<int>(a.geti("--min-clique", 1));
  go.max_clique_time_s = a.getd("--clique-time", 5);
  std::map<int, long> sizes;
  std::map<int, long> first_sizes;   // size of the second clique (after the first frame)
  std::map<int, long> free_after;
  std::vector<uint32_t> best;
  int best_size = 0;
  long sum = 0;
  for (long r = 0; r < restarts; ++r) {
    const GreedyResult g = greedy_frame_union(L, K, seed0 + static_cast<uint64_t>(r), go);
    if (!classes_independent(L, K, g.classes)) throw std::runtime_error("greedy: result not independent");
    const int n = static_cast<int>(g.classes.size());
    ++sizes[2 * n];
    sum += n;
    if (g.clique_sizes.size() > 1) ++first_sizes[g.clique_sizes[1]];
    ++free_after[g.free_after_frames];
    std::string cs;
    for (int s : g.clique_sizes) cs += (cs.empty() ? "" : "+") + std::to_string(s);
    if (r < 10 || n > best_size)
      std::printf("restart %ld: %d classes = %d vectors, cliques %s, free after first frame %d, %.2f s\n", r, n, 2 * n,
                  cs.c_str(), g.free_after_frames, g.seconds);
    if (n > best_size) {
      best_size = n;
      best = g.classes;
    }
  }
  std::printf("size distribution (vectors): %s\n", hist_str(sizes).c_str());
  std::printf("second clique size: %s\n", hist_str(first_sizes).c_str());
  std::printf("free classes after the first frame: %s\n", hist_str(free_after).c_str());
  std::printf("mean %.1f vectors, best %d vectors (496 = the record)\n",
              2.0 * static_cast<double>(sum) / static_cast<double>(std::max(1L, restarts)), 2 * best_size);
  std::filesystem::create_directories("runs/frames");
  const std::string bf = "runs/frames/greedy_best_" + std::to_string(2 * best_size) + ".txt";
  write_set(bf, classes_to_vectors(L, K, best), "T3.3 frames greedy: best of " + std::to_string(restarts) + " restarts");
  std::printf("best written to %s\n", bf.c_str());
  if (best_size > 248) save_and_verify(L, K, best, "greedy", "greedy frame union");
  std::printf("RESULT ok=1 cmd=greedy restarts=%ld best=%d mean=%.1f seconds=%.1f\n", restarts, 2 * best_size,
              2.0 * static_cast<double>(sum) / static_cast<double>(std::max(1L, restarts)), now());
  return 0;
}

// ---------------------------------------------------------------------------
// ls
// ---------------------------------------------------------------------------
int cmd_ls(const Args& a) {
  const auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  const Leech L = load(a);
  const Classes K = make_classes(L);
  std::vector<uint32_t> X0;
  std::string source;
  if (a.has("--greedy")) {
    GreedyOptions go;
    const GreedyResult g = greedy_frame_union(L, K, static_cast<uint64_t>(a.geti("--greedy", 1)), go);
    X0 = g.classes;
    source = "greedy seed " + a.gets("--greedy", "1");
  } else {
    if (a.pos.empty()) throw std::runtime_error("ls: need <S.txt> or --greedy SEED");
    const std::vector<Vec> S = read_set(a.pos[0]);
    const VerifyResult vr = verify_independent(L, S);
    if (!vr.ok) throw std::runtime_error("input is not an independent set: " + vr.message);
    X0 = set_classes(L, K, S);
    source = a.pos[0];
  }
  std::printf("start: %s, %zu classes = %zu vectors\n", source.c_str(), X0.size(), 2 * X0.size());
  LocalSearchOptions lo;
  lo.time_limit_s = a.getd("--time-limit", 60);
  lo.tmax = static_cast<int>(a.geti("--tmax", 6));
  lo.max_drop = static_cast<int>(a.geti("--max-drop", 2));
  lo.tabu_tenure = static_cast<int>(a.geti("--tenure", 50));
  lo.node_limit = a.geti("--node-limit", 200000);
  lo.move_time_s = a.getd("--move-time", 2);
  lo.seed_tmax = static_cast<int>(a.geti("--seed-tmax", 8));
  lo.rng_seed = static_cast<uint64_t>(a.geti("--seed", 1));
  lo.target = static_cast<int>(a.geti("--target", 249));
  const LocalSearchResult r = clique_local_search(L, K, X0, lo, [](const std::string& s) { std::printf("  %s\n", s.c_str()); });
  std::printf("moves %ld (improving %ld, plateau %ld, worsening %ld, rejected %ld), %.1f s\n", r.moves, r.improving,
              r.plateau, r.worsening, r.rejected, r.seconds);
  std::printf("size visits (classes): %s\n", hist_str(r.size_visits).c_str());
  std::printf("best %zu classes = %zu vectors (start %d classes)\n", r.best.size(), 2 * r.best.size(), r.start_size);
  std::filesystem::create_directories("runs/frames");
  const std::string bf = "runs/frames/ls_best_" + std::to_string(2 * r.best.size()) + "_" + utc_stamp() + ".txt";
  write_set(bf, classes_to_vectors(L, K, r.best), "T3.3 frames ls: from " + source);
  std::printf("best written to %s\n", bf.c_str());
  if (r.best.size() > 248) save_and_verify(L, K, r.best, "ls", "clique local search from " + source);
  std::printf("RESULT ok=1 cmd=ls start=%d best=%zu moves=%ld improving=%ld plateau=%ld worsening=%ld rejected=%ld "
              "seconds=%.1f\n",
              2 * r.start_size, 2 * r.best.size(), r.moves, r.improving, r.plateau, r.worsening, r.rejected, now());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  try {
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv);
    if (a.has("--threads")) omp_set_num_threads(static_cast<int>(a.geti("--threads", 1)));
    if (cmd == "count") return cmd_count(a);
    if (cmd == "crosses") return cmd_crosses(a);
    if (cmd == "extend") return cmd_extend(a);
    if (cmd == "greedy") return cmd_greedy(a);
    if (cmd == "ls") return cmd_ls(a);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "frames: error: %s\n", e.what());
    std::printf("RESULT ok=0 reason=\"%s\"\n", e.what());
    return 1;
  }
}

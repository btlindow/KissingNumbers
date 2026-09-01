// plateau_walk — best-first exploration of the plateau graph of an independent
// set (T3.1b, extends T3.1: docs/reports/T3.1b.md).
//
//   plateau_walk <S.txt> [--kmax K=12] [--node-time SEC=60] [--max-nodes N=200] [--wall SEC=7200]
//                [--out DIR] [--data DIR=data] [--max-sets N=10000000] [--combos M=4]
//                [--verify-bin PATH=build/t31b/tools/verify_s] [--python PATH=.venv/bin/python] [--no-verify]
//
// Nodes = independent sets of the current size, edges = (k,k)-plateau moves
// with k ≤ K (kiss::plateau_walk, include/kiss/plateau.h). Every expanded node
// gets the exhaustive (k,k+1)-swap search of T3.1; an improving swap is applied
// at once, the improved set is written to runs/found/S_<size>_<utc>_plateau.txt,
// re-verified with tools/verify_s and python/verify_S.py (both run as
// subprocesses), and the walk continues from it.
//
// Per node one progress line; with --out DIR also DIR/nodes.tsv, DIR/summary.txt
// (fingerprint classes) and DIR/sets/node_<id>.txt for the first node of every
// fingerprint class and for every improved set.
//
// Isomorphism invariants (Co_0-invariant): tightness histogram, Gram histogram of
// S, Gram histogram among the minimum-tightness vertices; after expansion also the
// per-k plateau-move counts. Shape counts (octad / 3,1^23 / 4,4) are only
// monomial-invariant and are reported separately ("fp_shapes").
//
// Last line: RESULT ok=1 nodes=<expanded> fingerprints=<f> best_size=<b> improving_found=<0|1> ...
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/plateau.h"
#include "kiss/swaps.h"
#include "kiss/types.h"
#include "kiss/verify.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: plateau_walk <S.txt> [--kmax K=12] [--node-time SEC=60] [--max-nodes N=200] [--wall SEC=7200]\n"
               "       [--out DIR] [--data DIR=data] [--max-sets N=10000000] [--combos M=4]\n"
               "       [--verify-bin PATH=build/t31b/tools/verify_s] [--python PATH=.venv/bin/python] [--no-verify]\n");
}

std::string utc_stamp() {
  const std::time_t t = std::time(nullptr);
  std::tm tm {};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

std::string vec_str(const std::vector<uint32_t>& v, std::size_t cap = 16) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size() && i < cap; ++i) s += (i ? " " : "") + std::to_string(v[i]);
  if (v.size() > cap) s += " ...";
  return s + "]";
}

std::string counts_str(const std::vector<std::size_t>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (!v[i]) continue;
    if (!s.empty()) s += ',';
    s += std::to_string(i + 1) + ":" + std::to_string(v[i]);
  }
  return s.empty() ? "-" : s;
}

std::string gram_str(const std::array<long, 8>& h) {
  static const int cls[7] = {-32, -16, -8, 0, 8, 16, 32};
  std::string s;
  for (int c = 0; c < 7; ++c) {
    if (!h[static_cast<std::size_t>(c)]) continue;
    if (!s.empty()) s += ',';
    s += std::to_string(cls[c]) + ":" + std::to_string(h[static_cast<std::size_t>(c)]);
  }
  if (h[7]) s += ",other:" + std::to_string(h[7]);
  return s.empty() ? "-" : s;
}

std::string shapes_str(const kiss::Leech& L, const std::vector<uint32_t>& S) {
  int sh[3] = {0, 0, 0};
  for (uint32_t v : S) {
    const int s = kiss::leech_shape(L.C[v]);
    if (s >= 0 && s < 3) ++sh[s];
  }
  return std::to_string(sh[0]) + "/" + std::to_string(sh[1]) + "/" + std::to_string(sh[2]);
}

std::vector<kiss::Vec> vectors_of(const kiss::Leech& L, const std::vector<uint32_t>& S) {
  std::vector<kiss::Vec> V;
  V.reserve(S.size());
  for (uint32_t v : S) V.push_back(L.C[v]);
  return V;
}

int run_cmd(const std::string& cmd) {
  std::printf("$ %s\n", cmd.c_str());
  std::fflush(stdout);
  const int rc = std::system(cmd.c_str());
  std::printf("  -> exit status %d\n", rc);
  std::fflush(stdout);
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::string file, data_dir = "data", out_dir, verify_bin = "build/t31b/tools/verify_s", python = ".venv/bin/python";
  bool do_verify = true;
  kiss::PlateauOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); usage(); std::exit(1); }
      return argv[++i];
    };
    if (a == "--kmax") opt.kmax = std::atoi(need("--kmax"));
    else if (a == "--node-time") opt.node_time_s = std::atof(need("--node-time"));
    else if (a == "--max-nodes") opt.max_nodes = static_cast<std::size_t>(std::atoll(need("--max-nodes")));
    else if (a == "--wall") opt.wall_s = std::atof(need("--wall"));
    else if (a == "--out") out_dir = need("--out");
    else if (a == "--data") data_dir = need("--data");
    else if (a == "--max-sets") opt.max_sets = static_cast<std::size_t>(std::atoll(need("--max-sets")));
    else if (a == "--combos") opt.combo_max_moves = std::atoi(need("--combos"));
    else if (a == "--verify-bin") verify_bin = need("--verify-bin");
    else if (a == "--python") python = need("--python");
    else if (a == "--no-verify") do_verify = false;
    else if (a == "-h" || a == "--help") { usage(); return 1; }
    else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    else if (file.empty()) file = a;
    else { usage(); return 1; }
  }
  if (file.empty()) { usage(); std::printf("RESULT ok=0 nodes=0 reason=\"usage\"\n"); return 1; }
  if (opt.kmax < 1 || opt.kmax > kiss::KSWAP_MAX_K) {
    std::printf("RESULT ok=0 nodes=0 reason=\"kmax must be 1..%d\"\n", kiss::KSWAP_MAX_K);
    return 1;
  }

  try {
    kiss::Leech L;
    try {
      L = kiss::load_leech(data_dir);
      std::printf("leech            : loaded from %s\n", data_dir.c_str());
    } catch (const std::exception& e) {
      std::printf("leech            : load failed (%s); generating\n", e.what());
      L = kiss::generate_leech();
    }
    const std::filesystem::path adj_file = std::filesystem::path(data_dir) / "adj.u32";
    const kiss::Adjacency adj(adj_file);
    std::printf("adjacency        : %s (mmap, %zu bytes)\n", adj_file.c_str(), adj.bytes());
    const kiss::LeechSwapGraph g(L, adj);

    const std::vector<kiss::Vec> SV = kiss::read_set(file);
    const kiss::VerifyResult vr = kiss::verify_independent(L, SV);
    if (!vr.ok) {
      std::printf("input            : %s is NOT a valid independent set: %s\n", file.c_str(), vr.message.c_str());
      std::printf("RESULT ok=0 nodes=0 reason=\"input not independent: %s\"\n", vr.message.c_str());
      return 1;
    }
    std::vector<uint32_t> S0 = kiss::set_indices(L, SV);
    std::sort(S0.begin(), S0.end());
    std::printf("input            : %s  |S|=%zu  antipodal=%d  gram=%s  shapes=%s\n", file.c_str(), S0.size(),
                kiss::is_antipodal(SV) ? 1 : 0, gram_str(kiss::gram_histogram(SV)).c_str(), shapes_str(L, S0).c_str());
    std::printf("walk             : kmax=%d node_time=%.0fs/level max_nodes=%zu wall=%.0fs combos<=%d max_sets=%zu threads=%d\n",
                opt.kmax, opt.node_time_s, opt.max_nodes, opt.wall_s, opt.combo_max_moves, opt.max_sets, omp_get_max_threads());
    std::fflush(stdout);

    if (!out_dir.empty()) {
      std::filesystem::create_directories(std::filesystem::path(out_dir) / "sets");
    }
    FILE* tsv = nullptr;
    if (!out_dir.empty()) {
      tsv = std::fopen((std::filesystem::path(out_dir) / "nodes.tsv").c_str(), "w");
      if (tsv)
        std::fprintf(tsv, "id\tparent\tdepth\tvia\tsize\tmin_tight\tfree\tt1\tt2\tt3\tmoves\tneighbours\tneighbours_new\timproving\tkmax_exh\tunions\tseconds\tfp0_new\tfp_new\tshapes\tplateau\tconnected\thist\thash\n");
    }

    // Co_0-invariant extra fingerprint: Gram histogram of S + Gram histogram among the
    // minimum-tightness vertices (capped at 2000 vertices).
    const kiss::PlateauInvariant extra = [&](const std::vector<uint32_t>& S) -> std::string {
      std::string s = "gram=" + gram_str(kiss::gram_histogram(vectors_of(L, S)));
      const std::vector<uint16_t> tight = kiss::tightness_from_graph(g, S);
      std::vector<uint8_t> inS(static_cast<std::size_t>(kiss::N), 0);
      for (uint32_t v : S) inS[v] = 1;
      uint16_t mt = 0xFFFF;
      for (uint32_t v = 0; v < static_cast<uint32_t>(kiss::N); ++v)
        if (!inS[v] && tight[v] < mt) mt = tight[v];
      std::vector<uint32_t> cls;
      for (uint32_t v = 0; v < static_cast<uint32_t>(kiss::N) && cls.size() < 2000; ++v)
        if (!inS[v] && tight[v] == mt) cls.push_back(v);
      s += "|mingram(" + std::to_string(mt) + ")=" + gram_str(kiss::gram_histogram(vectors_of(L, cls)));
      return s;
    };

    std::map<std::string, std::size_t> fp_shapes;      // fp + shapes -> count (monomial-invariant classes)
    std::map<std::string, std::vector<std::size_t>> fp_nodes;   // fp -> node ids
    std::string found_file;
    std::size_t verify_ok = 0, verify_fail = 0, nodes_t123 = 0;
    int min_tight_seen = -1;

    kiss::PlateauHooks hooks;
    hooks.on_improve = [&](const kiss::PlateauNode& from, const kiss::Swap& sw, const std::vector<uint32_t>& S2) {
      const std::vector<kiss::Vec> V2 = vectors_of(L, S2);
      const kiss::VerifyResult v2 = kiss::verify_independent(L, V2);
      std::printf("=================================================================\n");
      std::printf("!!! IMPROVEMENT at node %zu (depth %zu): |S| %zu -> %zu by removing %zu %s and adding %zu %s; verify_independent: %s\n",
                  from.id, from.depth, from.S.size(), S2.size(), sw.remove.size(), vec_str(sw.remove).c_str(), sw.add.size(),
                  vec_str(sw.add).c_str(), v2.ok ? "OK" : v2.message.c_str());
      std::filesystem::create_directories("runs/found");
      found_file = "runs/found/S_" + std::to_string(S2.size()) + "_" + utc_stamp() + "_plateau.txt";
      kiss::write_set(found_file, V2,
                      "plateau_walk (T3.1b): (" + std::to_string(sw.remove.size()) + "," + std::to_string(sw.add.size()) +
                          ")-swap at plateau node " + std::to_string(from.id) + " (depth " + std::to_string(from.depth) + ") of " + file +
                          "\nsize " + std::to_string(S2.size()) + "; verify with tools/verify_s and python/verify_S.py");
      std::printf("!!! written to %s\n", found_file.c_str());
      std::fflush(stdout);
      if (do_verify) {
        const int r1 = std::filesystem::exists(verify_bin) ? run_cmd(verify_bin + " " + found_file + " --data " + data_dir) : -1;
        const int r2 = std::filesystem::exists(python) ? run_cmd(python + " python/verify_S.py " + found_file) : -1;
        if (r1 == 0 && r2 == 0) ++verify_ok; else ++verify_fail;
        std::printf("!!! verification: verify_s=%s verify_S.py=%s\n", r1 == 0 ? "ok" : (r1 < 0 ? "missing" : "FAILED"),
                    r2 == 0 ? "ok" : (r2 < 0 ? "missing" : "FAILED"));
      }
      std::printf("=================================================================\n");
      std::fflush(stdout);
    };
    hooks.on_node = [&](const kiss::PlateauNode& nd, const kiss::PlateauWalkResult& part) {
      const std::string shapes = shapes_str(L, nd.S);
      ++fp_shapes[nd.fp + "|shapes=" + shapes];
      fp_nodes[nd.fp].push_back(nd.id);
      if (nd.t1 || nd.t2 || nd.t3) ++nodes_t123;
      if (nd.min_tight >= 0 && (min_tight_seen < 0 || nd.min_tight < min_tight_seen)) min_tight_seen = nd.min_tight;
      const double el = std::chrono::duration<double>(clock::now() - t0).count();
      std::printf("node=%zu parent=%zu depth=%zu via=%s size=%zu min_tight=%d free=%zu t123=%zu/%zu/%zu plateau=%s connected=%s moves=%zu nbrs=%zu new=%zu improving=%d kexh=%d unions=%zu fp0=%s fp=%s shapes=%s expanded=%zu discovered=%zu fps=%zu node_s=%.1f elapsed_s=%.0f\n",
                  nd.id, nd.parent, nd.depth, nd.via.c_str(), nd.S.size(), nd.min_tight, nd.free_vertices, nd.t1, nd.t2, nd.t3,
                  counts_str(nd.plateau_per_k).c_str(), counts_str(nd.connected_per_k).c_str(), nd.moves.size(), nd.neighbours,
                  nd.neighbours_new, nd.improving ? 1 : 0, nd.kmax_exhaustive, nd.unions, nd.fp0_new ? "NEW" : "seen",
                  nd.fp_new ? "NEW" : "seen", shapes.c_str(), part.expanded, part.nodes.size(), fp_nodes.size(), nd.seconds, el);
      if (nd.fp_new)
        std::printf("   fingerprint #%zu (node %zu): hist=%s | %s\n", fp_nodes.size(), nd.id, kiss::histogram_string(nd.hist).c_str(),
                    nd.fp.c_str());
      if (nd.fp_new && nd.moves.size() <= 16)
        for (const kiss::Swap& sw : nd.moves)
          std::printf("   move (%zu,%zu): remove %s add %s\n", sw.remove.size(), sw.add.size(), vec_str(sw.remove).c_str(),
                      vec_str(sw.add).c_str());
      std::fflush(stdout);
      if (tsv) {
        std::fprintf(tsv, "%zu\t%zu\t%zu\t%s\t%zu\t%d\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%d\t%d\t%zu\t%.2f\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n",
                     nd.id, nd.parent, nd.depth, nd.via.c_str(), nd.S.size(), nd.min_tight, nd.free_vertices, nd.t1, nd.t2, nd.t3,
                     nd.moves.size(), nd.neighbours, nd.neighbours_new, nd.improving ? 1 : 0, nd.kmax_exhaustive, nd.unions,
                     nd.seconds, nd.fp0_new ? 1 : 0, nd.fp_new ? 1 : 0, shapes.c_str(), counts_str(nd.plateau_per_k).c_str(),
                     counts_str(nd.connected_per_k).c_str(), kiss::histogram_string(nd.hist).c_str(), nd.hash.c_str());
        std::fflush(tsv);
      }
      if (!out_dir.empty() && (nd.fp_new || nd.improving || nd.id == 0)) {
        char name[64];
        std::snprintf(name, sizeof name, "node_%05zu.txt", nd.id);
        kiss::write_set(std::filesystem::path(out_dir) / "sets" / name, vectors_of(L, nd.S),
                        "plateau_walk node " + std::to_string(nd.id) + " depth " + std::to_string(nd.depth) + " via " + nd.via +
                            " from " + file + "\n" + nd.fp);
      }
    };

    const kiss::PlateauWalkResult W = kiss::plateau_walk(g, S0, opt, extra, hooks);
    if (tsv) std::fclose(tsv);

    // ---- summary of the fingerprint classes -------------------------------------
    std::printf("classes          : %zu distinct fingerprints (Co_0-invariant) over %zu expanded nodes; %zu with shapes; %zu distinct fp0 among all %zu discovered\n",
                W.fingerprints, W.expanded, fp_shapes.size(), W.fingerprints0_all, W.nodes.size());
    FILE* sum = out_dir.empty() ? nullptr : std::fopen((std::filesystem::path(out_dir) / "summary.txt").c_str(), "w");
    std::size_t ci = 0;
    for (const auto& kv : fp_nodes) {
      const kiss::PlateauNode& nd = W.nodes[kv.second.front()];
      std::printf("  class %zu: nodes=%zu first=%zu size=%zu min_tight=%d hist=%s plateau=%s connected=%s\n", ++ci, kv.second.size(),
                  nd.id, nd.S.size(), nd.min_tight, kiss::histogram_string(nd.hist).c_str(), counts_str(nd.plateau_per_k).c_str(),
                  counts_str(nd.connected_per_k).c_str());
      if (sum) std::fprintf(sum, "class %zu nodes=%zu first=%zu size=%zu %s\n", ci, kv.second.size(), nd.id, nd.S.size(), nd.fp.c_str());
    }
    if (sum) {
      std::fprintf(sum, "\nwith shapes (monomial-invariant): %zu classes\n", fp_shapes.size());
      for (const auto& kv : fp_shapes) std::fprintf(sum, "  nodes=%zu %s\n", kv.second, kv.first.c_str());
      std::fclose(sum);
    }
    // depth profile of the expanded nodes
    std::map<std::size_t, std::size_t> depth_hist;
    for (const auto& nd : W.nodes)
      if (nd.expanded) ++depth_hist[nd.depth];
    std::string dh;
    for (const auto& kv : depth_hist) dh += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
    std::printf("depth profile    : (depth:expanded) %s; frontier left %zu; wall_hit=%d node_cap_hit=%d\n", dh.c_str(), W.frontier_left,
                W.wall_hit ? 1 : 0, W.node_cap_hit ? 1 : 0);
    if (W.improving_found) {
      std::printf("!!! best set: size %zu (%zu improvement(s) applied) — %s\n", W.best_size, W.improvements, found_file.c_str());
    }

    const double total = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("RESULT ok=1 nodes=%zu discovered=%zu fingerprints=%zu fingerprints0=%zu fp_shapes=%zu best_size=%zu start_size=%zu improving_found=%d improvements=%zu verify_ok=%zu verify_fail=%zu min_tight_seen=%d nodes_t123=%zu frontier_left=%zu wall_hit=%d node_cap_hit=%d kmax=%d found_file=%s total_s=%.1f\n",
                W.expanded, W.nodes.size(), W.fingerprints, W.fingerprints0, fp_shapes.size(), W.best_size, S0.size(),
                W.improving_found ? 1 : 0, W.improvements, verify_ok, verify_fail, min_tight_seen, nodes_t123, W.frontier_left,
                W.wall_hit ? 1 : 0, W.node_cap_hit ? 1 : 0, opt.kmax, found_file.empty() ? "-" : found_file.c_str(), total);
    return 0;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    std::printf("RESULT ok=0 nodes=0 reason=\"%s\"\n", e.what());
    return 1;
  }
}

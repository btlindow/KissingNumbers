// swapsearch — exhaustive small-swap neighbourhoods of an independent set
// (docs/design.md T3.1, README §3 W2a).
//
//   swapsearch <S.txt> [--kmax K] [--time-limit SEC] [--data DIR] [--max-sets N]
//              [--list N] [--struct T] [--no-classical] [--dump-plateau FILE]
//   swapsearch --greedy SEED [same options]      (random greedy maximal set instead of a file)
//
// Pipeline: load C and the mmap'ed adjacency, index S, check it is an
// independent set (norm 32, membership, distinct, Gram ≤ 8 — inline, no
// dependency on kiss/verify.h), compute tight[] from the adjacency rows and
// print its histogram, run the classical (1,2)/(2,3) routines, describe the
// conf-set structure of the least-tight class, then the generalised
// (k, k+1)-swap search for k = 1..K. Any improving swap is applied, the new
// set re-verified (Gram ≤ 8) and written to runs/found/S_<size>_<utc>_swap.txt.
//
// Plateau moves ((k,k)-swaps) are counted per k and split into "connected"
// ones (not a disjoint union of smaller plateau moves) and composite ones.
// --dump-plateau FILE (T3.1b) writes every stored plateau / improving move in
// machine-readable form, one per line:
//   plateau k=<k> connected=<0|1> remove=<i,j,...> add=<u,v,...>
//   improving k=<k> gain=<g> remove=... add=...
// (vertex indices into the canonical C; the stored moves are the connected
// ones, up to --list per level, see kiss::KSwapOptions::keep_moves).
//
// Last line: RESULT ok=1 size=<n> improving=<0|1> kmax_exhaustive=<k> plateau_moves=<m>
//            plateau_connected=<c> ...
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/plateau.h"
#include "kiss/swaps.h"
#include "kiss/types.h"
#include "kiss/platform.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: swapsearch <S.txt> | --greedy SEED  [--kmax K=8] [--time-limit SEC=120]\n"
               "       [--data DIR=data] [--max-sets N=10000000] [--list N=10] [--struct T] [--no-classical]\n"
               "       [--dump-plateau FILE]\n");
}

std::string utc_stamp() {
  const std::time_t t = std::time(nullptr);
  const std::tm tm = kiss::gmtime_utc(t);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

// Inline certificate check (README §2): norm 32, in C, distinct, Gram ≤ 8.
bool check_independent(const kiss::Leech& L, const std::vector<kiss::Vec>& S, std::string& why) {
  for (std::size_t i = 0; i < S.size(); ++i) {
    if (kiss::norm2(S[i]) != 32) { why = "row " + std::to_string(i) + " has norm != 32"; return false; }
    if (L.index_of(S[i]) < 0) { why = "row " + std::to_string(i) + " is not a Leech minimal vector"; return false; }
  }
  bool ok = true;
  std::string w;
#pragma omp parallel for schedule(dynamic, 8)
  for (std::size_t i = 0; i < S.size(); ++i) {
    if (!ok) continue;
    for (std::size_t j = i + 1; j < S.size(); ++j) {
      const int d = kiss::dot(S[i], S[j]);
      if (d > 8) {
#pragma omp critical
        {
          ok = false;
          w = "rows " + std::to_string(i) + "," + std::to_string(j) + " have inner product " + std::to_string(d);
        }
        break;
      }
    }
  }
  if (!ok) why = w;
  return ok;
}

std::string vec_str(const std::vector<uint32_t>& v, std::size_t cap = 16) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size() && i < cap; ++i) s += (i ? " " : "") + std::to_string(v[i]);
  if (v.size() > cap) s += " ...";
  return s + "]";
}

std::string posv_str(const std::vector<uint32_t>& S, const std::vector<uint32_t>& verts) {
  // vertices as S positions
  std::string s = "{";
  for (std::size_t i = 0; i < verts.size(); ++i) {
    auto it = std::lower_bound(S.begin(), S.end(), verts[i]);
    const long p = (it != S.end() && *it == verts[i]) ? static_cast<long>(it - S.begin()) : -1;
    s += (i ? "," : "") + std::to_string(p);
  }
  return s + "}";
}

}  // namespace

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::string file, data_dir = "data", dump_file;
  long greedy_seed = -1;
  kiss::KSwapOptions opt;
  opt.max_sets = 10000000;
  std::size_t list_n = 10;
  int struct_t = -1;
  bool classical = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); usage(); std::exit(1); }
      return argv[++i];
    };
    if (a == "--kmax") opt.kmax = std::atoi(need("--kmax"));
    else if (a == "--time-limit") opt.time_limit_s = std::atof(need("--time-limit"));
    else if (a == "--data") data_dir = need("--data");
    else if (a == "--max-sets") opt.max_sets = static_cast<std::size_t>(std::atoll(need("--max-sets")));
    else if (a == "--list") list_n = static_cast<std::size_t>(std::atol(need("--list")));
    else if (a == "--struct") struct_t = std::atoi(need("--struct"));
    else if (a == "--greedy") greedy_seed = std::atol(need("--greedy"));
    else if (a == "--no-classical") classical = false;
    else if (a == "--dump-plateau") dump_file = need("--dump-plateau");
    else if (a == "-h" || a == "--help") { usage(); return 1; }
    else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    else if (file.empty()) file = a;
    else { usage(); return 1; }
  }
  if (file.empty() && greedy_seed < 0) { usage(); std::printf("RESULT ok=0 size=0 reason=\"usage\"\n"); return 1; }
  if (opt.kmax < 1 || opt.kmax > kiss::KSWAP_MAX_K) {
    std::printf("RESULT ok=0 size=0 reason=\"kmax must be 1..%d\"\n", kiss::KSWAP_MAX_K);
    return 1;
  }
  opt.keep_moves = std::max<std::size_t>(list_n, 1);

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
    std::printf("adjacency        : %s (mmap, %zu bytes)\n", adj_file.string().c_str(), adj.bytes());
    const kiss::LeechSwapGraph g(L, adj);

    // ---- the set -----------------------------------------------------------
    std::vector<uint32_t> S;
    std::string label;
    if (greedy_seed >= 0) {
      S = kiss::greedy_maximal_set(g, static_cast<uint64_t>(greedy_seed));
      label = "greedy(seed=" + std::to_string(greedy_seed) + ")";
    } else {
      const std::vector<kiss::Vec> SV = kiss::read_set(file);
      std::string why;
      if (!check_independent(L, SV, why)) {
        std::printf("input            : %s is NOT a valid independent set: %s\n", file.c_str(), why.c_str());
        std::printf("RESULT ok=0 size=%zu reason=\"input not independent: %s\"\n", SV.size(), why.c_str());
        return 1;
      }
      for (const auto& v : SV) S.push_back(static_cast<uint32_t>(L.index_of(v)));
      std::sort(S.begin(), S.end());
      label = file;
    }
    const std::size_t n = S.size();
    bool antipodal = true;
    for (uint32_t s : S)
      if (!std::binary_search(S.begin(), S.end(), L.neg[s])) { antipodal = false; break; }
    std::printf("input            : %s  |S|=%zu  antipodal=%d  independent(Gram<=8)=%d\n", label.c_str(), n,
                antipodal ? 1 : 0, kiss::is_independent(g, S) ? 1 : 0);

    // ---- tightness ---------------------------------------------------------
    const std::vector<uint16_t> tight = kiss::tightness_from_graph(g, S);
    std::size_t bad_inS = 0;
    for (uint32_t s : S) bad_inS += tight[s] != 0;
    const std::vector<std::size_t> hist = kiss::tightness_histogram(tight, S);
    std::printf("tightness        : histogram over v not in S (value:count): %s\n",
                kiss::histogram_string(hist).c_str());
    std::printf("tightness        : free vertices (tight==0, not in S) = %zu; S members with tight!=0: %zu\n",
                hist.empty() ? 0 : hist[0], bad_inS);
    int min_t = -1;
    for (std::size_t t = 1; t < hist.size(); ++t)
      if (hist[t]) { min_t = static_cast<int>(t); break; }
    std::printf("tightness        : minimum tightness outside S = %d\n", min_t);

    // antipodal conf structure: conf(-v) = -conf(v)  ⇔ tight[neg v] == tight[v] and, for the
    // candidates, the conf sets match under neg.
    if (antipodal) {
      std::size_t tmis = 0;
      for (uint32_t v = 0; v < static_cast<uint32_t>(kiss::N); ++v) tmis += tight[v] != tight[L.neg[v]];
      std::size_t cmis = 0, ccount = 0;
      std::vector<std::vector<uint32_t>> conf(static_cast<std::size_t>(kiss::N));
      for (uint32_t s : S) {
        std::size_t cnt = 0;
        const uint32_t* row = g.neighbours(s, cnt);
        for (std::size_t j = 0; j < cnt; ++j)
          if (tight[row[j]] <= opt.kmax) conf[row[j]].push_back(s);
      }
      for (uint32_t v = 0; v < static_cast<uint32_t>(kiss::N); ++v) {
        if (tight[v] == 0 || tight[v] > opt.kmax) continue;
        ++ccount;
        std::vector<uint32_t> a;
        for (uint32_t s : conf[v]) a.push_back(L.neg[s]);
        std::sort(a.begin(), a.end());
        std::vector<uint32_t> b = conf[L.neg[v]];
        std::sort(b.begin(), b.end());
        cmis += a != b;
      }
      std::printf("antipodal        : tight[-v]!=tight[v] for %zu vertices; conf(-v)!=-conf(v) for %zu of %zu candidates (tight<=%d)\n",
                  tmis, cmis, ccount, opt.kmax);
    }

    // ---- classical (1,2) and (2,3) ------------------------------------------
    kiss::Swap best;
    bool improving = false;
    auto consider = [&](const kiss::Swap& sw, const char* src) {
      if (sw.gain() <= 0) return;
      if (!kiss::swap_is_valid(g, S, sw)) {
        std::printf("!! %s reported an INVALID swap %s -> %s (bug)\n", src, vec_str(sw.remove).c_str(), vec_str(sw.add).c_str());
        return;
      }
      std::printf("!! IMPROVING SWAP from %s: remove %s add %s (gain %d)\n", src, vec_str(sw.remove).c_str(),
                  vec_str(sw.add).c_str(), sw.gain());
      if (!improving || sw.gain() > best.gain()) { best = sw; improving = true; }
    };
    if (classical) {
      const kiss::Swap12Result r12 = kiss::find_swap12(g, S, tight);
      std::printf("(1,2)-swaps      : hits=%zu  |L_x| hist: %s  total=%zu max=%zu  %.3f s\n", r12.hits,
                  kiss::histogram_string(r12.L_hist).c_str(), r12.L_total, r12.L_max, r12.seconds);
      if (r12.found) consider(r12.swap, "(1,2)");
      const kiss::Swap23Result r23 = kiss::find_swap23(g, S, tight, r12);
      std::printf("(2,3)-swaps      : hits=%zu  pairs=%zu nonempty=%zu |P|>=3: %zu  tight2=%zu  |P| hist: %s  max=%zu  %.3f s\n",
                  r23.hits, r23.pairs, r23.pairs_nonempty, r23.pairs_ge3, r23.t2_candidates,
                  kiss::histogram_string(r23.pool_hist).c_str(), r23.pool_max, r23.seconds);
      if (r23.found) consider(r23.swap, "(2,3)");
    }

    // ---- conf-set structure of the least tight class ------------------------
    const int st = struct_t > 0 ? struct_t : min_t;
    if (st > 0 && st <= kiss::KSWAP_MAX_K) {
      const std::vector<kiss::ConfGroup> groups = kiss::group_by_conf(g, S, tight, st);
      std::size_t total = 0;
      int best_mis = 0;
      std::map<std::size_t, std::size_t> size_hist;
      for (const auto& gr : groups) {
        total += gr.members.size();
        best_mis = std::max(best_mis, static_cast<int>(gr.mis.size()));
        ++size_hist[gr.members.size()];
      }
      std::string sh;
      for (const auto& kv : size_hist) sh += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
      std::printf("conf structure   : tightness-%d vertices: %zu in %zu distinct conf sets; group-size hist (size:count): %s; max MIS in a group = %d (need %d for a (%d,%d)-swap)\n",
                  st, total, groups.size(), sh.c_str(), best_mis, st + 1, st, st + 1);
      // Cross-group structure: how the conf sets overlap, which S members they
      // cover, Gram among the class members, antipodal pairing.
      {
        std::map<std::size_t, std::size_t> isect;   // |conf_a ∩ conf_b| over pairs of groups
        std::map<uint32_t, std::size_t> cover;      // S member -> #conf sets containing it
        for (std::size_t a = 0; a < groups.size(); ++a) {
          for (uint32_t s : groups[a].conf) ++cover[s];
          for (std::size_t b = a + 1; b < groups.size(); ++b) {
            std::size_t c = 0;
            for (uint32_t s : groups[a].conf)
              if (std::binary_search(groups[b].conf.begin(), groups[b].conf.end(), s)) ++c;
            ++isect[c];
          }
        }
        std::map<std::size_t, std::size_t> cover_hist;   // multiplicity -> #S members
        for (const auto& kv : cover) ++cover_hist[kv.second];
        std::string is, ch;
        for (const auto& kv : isect) is += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
        for (const auto& kv : cover_hist) ch += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
        std::vector<uint32_t> members;
        for (const auto& gr : groups) members.insert(members.end(), gr.members.begin(), gr.members.end());
        std::sort(members.begin(), members.end());
        std::map<int, std::size_t> mg;
        std::size_t neg_in = 0, neg_conf_disjoint = 0;
        for (std::size_t a = 0; a < members.size(); ++a) {
          for (std::size_t b = a + 1; b < members.size(); ++b) ++mg[kiss::dot(L.C[members[a]], L.C[members[b]])];
          if (std::binary_search(members.begin(), members.end(), L.neg[members[a]])) ++neg_in;
        }
        // conf(v) ∩ conf(−v) = ∅ ?  (conf(−v) = −conf(v) when S is antipodal)
        for (const auto& gr : groups) {
          bool disjoint = true;
          for (uint32_t s : gr.conf)
            if (std::binary_search(gr.conf.begin(), gr.conf.end(), L.neg[s])) disjoint = false;
          if (disjoint) neg_conf_disjoint += gr.members.size();
        }
        std::string mgs;
        for (const auto& kv : mg) mgs += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
        std::printf("conf structure   : |conf_a ∩ conf_b| over %zu pairs of distinct conf sets (size:count): %s\n",
                    groups.size() * (groups.size() - 1) / 2, is.c_str());
        std::printf("conf structure   : S members covered by the conf sets: %zu of %zu; multiplicity hist (#sets:count): %s\n",
                    cover.size(), n, ch.c_str());
        std::printf("conf structure   : Gram among the %zu tightness-%d vertices (ip:count): %s; -v also tightness-%d for %zu of them; conf(v) antipode-free for %zu\n",
                    members.size(), st, mgs.c_str(), st, neg_in, neg_conf_disjoint);
      }
      std::size_t shown = 0;
      for (const auto& gr : groups) {
        if (shown++ >= list_n) { std::printf("conf structure   : ... (%zu groups not listed)\n", groups.size() - list_n); break; }
        std::string gram = "gram(conf)=";
        for (std::size_t a = 0; a < gr.conf.size(); ++a)
          for (std::size_t b = a + 1; b < gr.conf.size(); ++b)
            gram += std::to_string(kiss::dot(L.C[gr.conf[a]], L.C[gr.conf[b]])) + (b + 1 < gr.conf.size() || a + 2 < gr.conf.size() ? "," : "");
        int shapes[3] = {0, 0, 0};
        for (uint32_t v : gr.members) {
          const int s = kiss::leech_shape(L.C[v]);
          if (s >= 0 && s < 3) ++shapes[s];
        }
        // pairwise inner products among the members
        std::map<int, int> mg;
        for (std::size_t a = 0; a < gr.members.size(); ++a)
          for (std::size_t b = a + 1; b < gr.members.size(); ++b) ++mg[kiss::dot(L.C[gr.members[a]], L.C[gr.members[b]])];
        std::string mgs;
        for (const auto& kv : mg) mgs += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
        std::printf("  conf=%s (S pos %s) %s  members=%zu %s shapes(2^8/3,1^23/4^2)=%d/%d/%d  member-gram %s MIS=%zu\n",
                    vec_str(gr.conf).c_str(), posv_str(S, gr.conf).c_str(), gram.c_str(), gr.members.size(),
                    vec_str(gr.members, 8).c_str(), shapes[0], shapes[1], shapes[2], mgs.c_str(), gr.mis.size());
      }
    }

    // ---- generalised (k, k+1) search ---------------------------------------
    std::printf("kswap            : kmax=%d time_limit=%.0fs/level max_sets=%zu threads=%d\n", opt.kmax,
                opt.time_limit_s, opt.max_sets, omp_get_max_threads());
    std::size_t plateau_total = 0, plateau_conn_total = 0;
    auto on_level = [&](const kiss::KSwapLevel& Lv) {
      std::printf("k=%d cand_eq=%zu cand_le=%zu distinct_conf=%zu R_total=%zu R_examined=%zu largest_pool=%zu mean_pool=%.2f best_m=%d plateau=%zu plateau_connected=%zu improving=%zu exhaustive=%d time_s=%.2f\n",
                  Lv.k, Lv.cand_eq_k, Lv.cand_le_k, Lv.distinct_conf_eq_k, Lv.R_total, Lv.R_examined, Lv.largest_pool,
                  Lv.R_examined ? static_cast<double>(Lv.pool_total) / static_cast<double>(Lv.R_examined) : 0.0,
                  Lv.best_m, Lv.plateau_moves, Lv.plateau_connected, Lv.improving_moves, Lv.exhaustive ? 1 : 0, Lv.seconds);
      std::fflush(stdout);
      for (std::size_t i = 0; i < Lv.plateau.size() && i < list_n; ++i) {
        std::vector<uint32_t> tv;
        for (uint32_t a : Lv.plateau[i].add) tv.push_back(tight[a]);
        std::printf("   plateau (%d,%d): remove %s (S pos %s) add %s tight %s\n", Lv.k, Lv.k, vec_str(Lv.plateau[i].remove).c_str(),
                    posv_str(S, Lv.plateau[i].remove).c_str(), vec_str(Lv.plateau[i].add).c_str(), vec_str(tv).c_str());
      }
      if (Lv.plateau_connected > list_n)
        std::printf("   ... %zu more connected plateau moves at k=%d (listed: connected ones only; %zu composite not listed)\n",
                    Lv.plateau_connected - list_n, Lv.k, Lv.plateau_moves - Lv.plateau_connected);
      else if (Lv.plateau_moves > Lv.plateau_connected)
        std::printf("   ... %zu composite plateau moves at k=%d (disjoint unions of smaller ones) not listed\n",
                    Lv.plateau_moves - Lv.plateau_connected, Lv.k);
      for (const auto& sw : Lv.improving) consider(sw, "kswap");
    };
    const kiss::KSwapResult K = kiss::kswap_search(g, S, tight, opt, on_level);
    for (const auto& Lv : K.levels) { plateau_total += Lv.plateau_moves; plateau_conn_total += Lv.plateau_connected; }
    if (K.free_vertices) {
      kiss::Swap sw;
      sw.add = {K.free_list[0]};
      consider(sw, "free-vertex");
    }
    std::printf("kswap            : unions of size<=%d enumerated: %zu  kmax_exhaustive=%d  improving=%d  plateau_total=%zu (connected %zu)  %.2f s\n",
                opt.kmax, K.unions_total, K.kmax_exhaustive, K.improving_found ? 1 : 0, plateau_total, plateau_conn_total, K.seconds);

    // ---- machine-readable dump of the stored moves (T3.1b) -------------------
    if (!dump_file.empty()) {
      FILE* f = std::fopen(dump_file.c_str(), "w");
      if (!f) throw std::runtime_error("cannot write " + dump_file);
      auto join = [](const std::vector<uint32_t>& v) {
        std::string s;
        for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
        return s;
      };
      std::fprintf(f, "# swapsearch --dump-plateau: input=%s size=%zu kmax=%d kmax_exhaustive=%d\n", label.c_str(), n, opt.kmax,
                   K.kmax_exhaustive);
      std::size_t written = 0;
      for (const auto& Lv : K.levels) {
        for (const auto& sw : Lv.plateau) {
          std::fprintf(f, "plateau k=%d connected=%d remove=%s add=%s\n", Lv.k, kiss::plateau_move_is_connected(g, sw) ? 1 : 0,
                       join(sw.remove).c_str(), join(sw.add).c_str());
          ++written;
        }
        for (const auto& sw : Lv.improving) {
          std::fprintf(f, "improving k=%d gain=%d remove=%s add=%s\n", Lv.k, sw.gain(), join(sw.remove).c_str(), join(sw.add).c_str());
          ++written;
        }
      }
      std::fclose(f);
      std::printf("dump             : %zu moves written to %s\n", written, dump_file.c_str());
    }

    // ---- improvement handling -----------------------------------------------
    std::string found_file;
    std::size_t new_size = n;
    if (improving) {
      const std::vector<uint32_t> S2 = kiss::apply_swap(S, best);
      std::vector<kiss::Vec> V2;
      for (uint32_t v : S2) V2.push_back(L.C[v]);
      std::string why;
      const bool ok2 = check_independent(L, V2, why);
      new_size = V2.size();
      std::printf("=================================================================\n");
      std::printf("!!! IMPROVEMENT: |S| %zu -> %zu by removing %zu and adding %zu; re-verified Gram<=8: %s %s\n", n, new_size,
                  best.remove.size(), best.add.size(), ok2 ? "OK" : "FAILED", ok2 ? "" : why.c_str());
      if (ok2) {
        std::filesystem::create_directories("runs/found");
        found_file = "runs/found/S_" + std::to_string(new_size) + "_" + utc_stamp() + "_swap.txt";
        kiss::write_set(found_file, V2,
                        "swapsearch (T3.1): (" + std::to_string(best.remove.size()) + "," + std::to_string(best.add.size()) +
                            ")-swap applied to " + label + "\nsize " + std::to_string(new_size) + "; verify with tools/verify_s and python/verify_S.py");
        std::printf("!!! written to %s — run tools/verify_s and python/verify_S.py on it\n", found_file.c_str());
      }
      std::printf("=================================================================\n");
    }

    const double total = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("RESULT ok=%d size=%zu improving=%d new_size=%zu kmax=%d kmax_exhaustive=%d plateau_moves=%zu plateau_connected=%zu free=%zu min_tight=%d unions=%zu antipodal=%d found_file=%s total_s=%.2f\n",
                1, n, improving ? 1 : 0, new_size, opt.kmax, K.kmax_exhaustive, plateau_total, plateau_conn_total, K.free_vertices, min_t,
                K.unions_total, antipodal ? 1 : 0, found_file.empty() ? "-" : found_file.c_str(), total);
    return 0;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    std::printf("RESULT ok=0 size=0 reason=\"%s\"\n", e.what());
    return 1;
  }
}

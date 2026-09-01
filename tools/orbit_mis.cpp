// T3.4 — symmetry-restricted MIS: H-orbits on C, orbit conflict graph, best
// union of orbits (README §3 W2d, docs/design.md T3.4).
//
//   orbit_mis [common] --group-file F [--name NAME] [--antipodal]
//   orbit_mis [common] --perm-file F [--perm-file F2 ...] --name NAME
//   orbit_mis [common] --aut-file F [--aut-file F2 ...] --name NAME    (den + 24x24 matrices, e.g. xi.txt)
//   orbit_mis [common] --suite DIR [--subgroups-dir DIR2]
//
// common: --data data --adj data/adj.u32 --set data/S496.txt --seed 1
//         --ls-seconds 10 --bb-seconds 30 --max-explicit 12000 --found runs/found
//         --export-graph F   (write the reduced orbit graph for python/tools/orbit_milp.py)
//
// One row per group H:
//   ROW name=.. order=.. orbits=.. self=.. edges=.. explicit=0|1 greedy=.. ls=..
//       bb=.. optimal=0|1 nodes=.. best=.. union496=0|1 seconds=..
// `best` is the largest weight of a union of non-self-conflicting, pairwise
// non-conflicting orbits found (greedy → iterated local search → branch and
// bound when the orbit graph is explicit); optimal=1 means B&B proved it.
// union496=1 iff the reference set is a union of H-orbits (it is then also
// used as a lower bound for B&B). Every best set is converted to vectors and
// checked with verify_independent; a set larger than the reference goes to
// --found/orbit_<name>_<size>.txt.
//
// --suite runs the T3.4 table: the monomial stabiliser of the reference set
// (runs/orbits/stab496/stabiliser_generators.txt, else −I), the sign group
// 2^12, cyclic groups for every M24 element order (pure permutations, with a
// random Golay sign part, and with −I adjoined), 23:11, PSL(2,23), M24, the
// monomial group, every monomial list in --subgroups-dir (plain and with −I),
// dihedral groups ⟨ξ, m⟩ for involutions m, and random Co_0 conjugates of two
// groups as a sanity check. Writes DIR/rows.txt and DIR/table.md.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kiss/adjacency.h"
#include "kiss/golay.h"
#include "kiss/group.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/orbits.h"
#include "kiss/verify.h"

namespace fs = std::filesystem;
using kiss::IndexPerm;
using kiss::Monomial;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Row {
  std::string name, order;
  uint32_t orbits = 0, self = 0;
  uint64_t edges = 0;
  bool explicit_graph = false;
  uint64_t greedy = 0, ls = 0, bb = 0, best = 0, nodes = 0;
  bool optimal = false, union_ref = false;
  double seconds = 0;
  std::string sizes;  // orbit size histogram (largest few)
};

struct Ctx {
  kiss::Leech L;
  std::unique_ptr<kiss::Adjacency> A;
  std::vector<uint32_t> S_idx;  // sorted
  std::string set_name;
  uint64_t seed = 1;
  double ls_seconds = 10, bb_seconds = 30;
  uint32_t max_explicit = 12000;
  fs::path found_dir = "runs/found";
  fs::path export_path;  // --export-graph: reduced orbit graph (single-group mode)
  std::vector<Row> rows;
  FILE* rows_file = nullptr;
  int errors = 0;
};

std::string order_string(const std::vector<Monomial>& gens, uint64_t cap = 3000000) {
  const uint64_t o = kiss::monomial_group_order(gens, cap);
  return o ? std::to_string(o) : (">" + std::to_string(cap));
}

std::string size_histogram(const kiss::Orbits& O) {
  std::map<uint32_t, uint32_t> h;
  for (uint32_t o = 0; o < O.count(); ++o) ++h[O.size(o)];
  std::string s;
  int k = 0;
  for (auto it = h.rbegin(); it != h.rend() && k < 4; ++it, ++k)
    s += (k ? "," : "") + std::to_string(it->first) + "x" + std::to_string(it->second);
  if (static_cast<int>(h.size()) > 4) s += ",...";
  return s;
}

void print_row(const Row& r, FILE* f) {
  std::fprintf(f,
               "ROW name=%s order=%s orbits=%u self=%u edges=%llu explicit=%d greedy=%llu ls=%llu bb=%llu "
               "optimal=%d nodes=%llu best=%llu union496=%d seconds=%.1f sizes=%s\n",
               r.name.c_str(), r.order.c_str(), r.orbits, r.self, static_cast<unsigned long long>(r.edges),
               r.explicit_graph ? 1 : 0, static_cast<unsigned long long>(r.greedy),
               static_cast<unsigned long long>(r.ls), static_cast<unsigned long long>(r.bb), r.optimal ? 1 : 0,
               static_cast<unsigned long long>(r.nodes), static_cast<unsigned long long>(r.best),
               r.union_ref ? 1 : 0, r.seconds, r.sizes.c_str());
  std::fflush(f);
}

// --export-graph: the orbit graph restricted to the non-self-conflicting
// orbits, for an external exact solver (python/tools/orbit_milp.py).
// Format: "ORBITS n_total n_usable", then n_usable lines "id weight", then
// "EDGES m" and m lines "a b" (orbit ids, a < b, both usable), then
// "REF k" and the k orbit ids of the reference set (k = 0 if it is not an
// orbit union).
void export_graph(const Ctx& ctx, const kiss::Orbits& O, const kiss::OrbitGraph& G,
                  const std::vector<uint32_t>& ref_orbits, bool union_ref) {
  FILE* f = std::fopen(ctx.export_path.string().c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + ctx.export_path.string());
  uint32_t usable = 0;
  for (uint32_t o = 0; o < G.n(); ++o) usable += G.self_conflicting(o) ? 0 : 1;
  std::fprintf(f, "ORBITS %u %u\n", G.n(), usable);
  for (uint32_t o = 0; o < G.n(); ++o)
    if (!G.self_conflicting(o)) std::fprintf(f, "%u %u\n", o, O.size(o));
  std::vector<std::pair<uint32_t, uint32_t>> E;
  std::vector<uint32_t> buf;
  for (uint32_t o = 0; o < G.n(); ++o) {
    if (G.self_conflicting(o)) continue;
    uint32_t cnt = 0;
    const uint32_t* nb = G.neighbours(o, cnt, buf);
    for (uint32_t k = 0; k < cnt; ++k)
      if (nb[k] > o && !G.self_conflicting(nb[k])) E.emplace_back(o, nb[k]);
  }
  std::fprintf(f, "EDGES %zu\n", E.size());
  for (const auto& e : E) std::fprintf(f, "%u %u\n", e.first, e.second);
  std::fprintf(f, "REF %zu\n", union_ref ? ref_orbits.size() : static_cast<std::size_t>(0));
  if (union_ref) for (uint32_t o : ref_orbits) std::fprintf(f, "%u\n", o);
  std::fclose(f);
  std::printf("exported reduced orbit graph: %u usable orbits, %zu edges -> %s\n", usable, E.size(),
              ctx.export_path.string().c_str());
}

// The whole pipeline for one group.
Row run_group_impl(Ctx& ctx, const std::string& name, const std::string& order, const std::vector<IndexPerm>& gens);

// In --suite mode a failing row is reported (ROW ... error=...) and the table
// continues; in single-group mode the exception propagates (RESULT ok=0).
Row run_group(Ctx& ctx, const std::string& name, const std::string& order, const std::vector<IndexPerm>& gens) {
  if (!ctx.rows_file) return run_group_impl(ctx, name, order, gens);
  try {
    return run_group_impl(ctx, name, order, gens);
  } catch (const std::exception& e) {
    Row r;
    r.name = name + "[ERROR:" + e.what() + "]";
    r.order = order;
    std::printf("ROW name=%s error=\"%s\"\n", name.c_str(), e.what());
    std::fprintf(ctx.rows_file, "ROW name=%s error=\"%s\"\n", name.c_str(), e.what());
    std::fflush(ctx.rows_file);
    ctx.rows.push_back(r);
    ++ctx.errors;
    return r;
  }
}

Row run_group_impl(Ctx& ctx, const std::string& name, const std::string& order, const std::vector<IndexPerm>& gens) {
  const auto t0 = std::chrono::steady_clock::now();
  Row r;
  r.name = name;
  r.order = order;
  kiss::Subgroup H{gens, name};
  const kiss::Orbits O = kiss::orbits(H);
  for (const IndexPerm& g : gens)
    if (!kiss::orbits_invariant(O, g)) throw std::runtime_error(name + ": orbits not invariant under a generator");
  r.orbits = O.count();
  r.sizes = size_histogram(O);
  kiss::OrbitGraph G(O, *ctx.A, ctx.max_explicit);
  r.self = G.self_conflicting_count();
  r.edges = G.edges();
  r.explicit_graph = G.is_explicit();
  std::vector<uint32_t> ref_orbits;
  r.union_ref = kiss::is_orbit_union(O, ctx.S_idx, ref_orbits);
  kiss::OrbitSet ref;
  if (r.union_ref) {
    ref.orbits = ref_orbits;
    ref.weight = ctx.S_idx.size();
    if (!kiss::orbit_set_valid(G, ref)) throw std::runtime_error(name + ": reference set is an orbit union but not valid?");
  }
  if (!ctx.export_path.empty()) export_graph(ctx, O, G, ref_orbits, r.union_ref);
  const kiss::OrbitSet g = kiss::greedy_orbit_mis(G);
  r.greedy = g.weight;
  kiss::OrbitSet best = kiss::local_search_orbit_mis(G, g, ctx.ls_seconds, ctx.seed);
  r.ls = best.weight;
  kiss::OrbitSet lower = best;
  if (r.union_ref && ref.weight > lower.weight) lower = ref;
  if (G.is_explicit()) {
    kiss::OrbitSet b = kiss::branch_and_bound_orbit_mis(G, lower, ctx.bb_seconds, ctx.max_explicit);
    r.bb = b.weight;
    r.optimal = b.optimal;
    r.nodes = b.nodes;
    if (b.weight >= best.weight) best = b;
  }
  if (r.union_ref && ref.weight > best.weight) best = ref;
  if (!kiss::orbit_set_valid(G, best)) throw std::runtime_error(name + ": best orbit set is not valid");
  const std::vector<uint32_t> verts = kiss::union_of_orbits(O, best.orbits);
  if (verts.size() != best.weight) throw std::runtime_error(name + ": weight mismatch");
  std::vector<kiss::Vec> vecs;
  for (uint32_t v : verts) vecs.push_back(ctx.L.C[v]);
  const kiss::VerifyResult vr = kiss::verify_independent(ctx.L, vecs);
  if (!vr.ok) throw std::runtime_error(name + ": best union is NOT independent: " + vr.message);
  r.best = best.weight;
  r.seconds = seconds_since(t0);
  if (best.weight > ctx.S_idx.size()) {
    fs::create_directories(ctx.found_dir);
    std::string safe = name;
    for (char& c : safe) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) c = '_';
    const fs::path out = ctx.found_dir / ("orbit_" + safe + "_" + std::to_string(best.weight) + ".txt");
    kiss::write_set(out, vecs, "T3.4 orbit union under H=" + name + " (|H|=" + order + "), " +
                                   std::to_string(best.orbits.size()) + " orbits, size " + std::to_string(best.weight));
    std::printf("!!! RECORD CANDIDATE: %llu > %zu under H=%s, written to %s — verify with verify_s and verify_S.py\n",
                static_cast<unsigned long long>(best.weight), ctx.S_idx.size(), name.c_str(), out.string().c_str());
  }
  print_row(r, stdout);
  if (ctx.rows_file) print_row(r, ctx.rows_file);
  ctx.rows.push_back(r);
  return r;
}

std::vector<IndexPerm> perms_of(const kiss::Leech& L, const std::vector<Monomial>& ms) {
  std::vector<IndexPerm> out;
  for (const Monomial& m : ms) out.push_back(kiss::index_permutation(L, m));
  return out;
}

Row run_monomial(Ctx& ctx, const std::string& name, std::vector<Monomial> gens, bool antipodal) {
  if (antipodal) gens.push_back(kiss::sign_flip(0xFFFFFFu));
  return run_group(ctx, name + (antipodal ? "±" : ""), order_string(gens), perms_of(ctx.L, gens));
}

void write_table(const Ctx& ctx, const fs::path& path) {
  FILE* f = std::fopen(path.string().c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + path.string());
  std::fprintf(f, "| H | \\|H\\| | #orbits | self-conf. | edges | greedy | LS | B&B | best | opt | %s union? | s | orbit sizes |\n",
               ctx.set_name.c_str());
  std::fprintf(f, "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
  for (const Row& r : ctx.rows)
    std::fprintf(f, "| %s | %s | %u | %u | %llu | %llu | %llu | %s | **%llu** | %s | %s | %.1f | %s |\n", r.name.c_str(),
                 r.order.c_str(), r.orbits, r.self, static_cast<unsigned long long>(r.edges),
                 static_cast<unsigned long long>(r.greedy), static_cast<unsigned long long>(r.ls),
                 r.explicit_graph ? std::to_string(r.bb).c_str() : "—", static_cast<unsigned long long>(r.best),
                 r.optimal ? "yes" : (r.explicit_graph ? "no" : "—"), r.union_ref ? "yes" : "no", r.seconds,
                 r.sizes.c_str());
  std::fclose(f);
}

// x -> 2x on F_23 (2 = 5^2 is a square, so this lies in PSL(2,23)), ∞ fixed. Order 11.
Monomial multiplier2() {
  std::array<uint8_t, kiss::DIM> p;
  for (int i = 0; i < 23; ++i) p[static_cast<std::size_t>(i)] = static_cast<uint8_t>((2 * i) % 23);
  p[23] = 23;
  if (!kiss::preserves_golay_code(p)) throw std::runtime_error("x->2x does not preserve the code");
  return kiss::coordinate_permutation(p);
}

void run_suite(Ctx& ctx, const fs::path& dir, const fs::path& subgroups_dir, const fs::path& data_dir) {
  fs::create_directories(dir);
  ctx.rows_file = std::fopen((dir / "rows.txt").string().c_str(), "w");
  const fs::path group_dir = data_dir / "group";
  const std::vector<Monomial> m24 = kiss::load_m24_generators(group_dir / "m24_generators.txt");
  const Monomial alpha = m24.at(0), gamma = m24.at(1), delta = m24.at(2);
  const Monomial negI = kiss::sign_flip(0xFFFFFFu);
  const auto octads = kiss::golay_octads();
  const auto codewords = kiss::golay_codewords();
  std::mt19937_64 rng(ctx.seed);

  // 1. the monomial stabiliser of the reference set
  {
    const fs::path stab = fs::path("runs/orbits/stab496/stabiliser_generators.txt");
    std::vector<Monomial> gens;
    std::string nm = "Stab(496)";
    if (fs::exists(stab)) gens = kiss::load_monomial_list(stab);
    else { gens = {negI}; nm += "=<-I>"; std::printf("note: %s missing, using <-I>\n", stab.string().c_str()); }
    for (const Monomial& m : gens) {
      const IndexPerm g = kiss::index_permutation(ctx.L, m);
      std::vector<uint32_t> img = kiss::apply(g, ctx.S_idx);
      std::sort(img.begin(), img.end());
      if (img != ctx.S_idx) throw std::runtime_error("stabiliser generator does not fix the reference set");
    }
    run_monomial(ctx, nm, gens, false);
    // the full Co_0 stabiliser (python/tools/stabilizer_496.py, Gram automorphisms extended to Λ)
    std::vector<fs::path> co0;
    for (int k = 0; k < 64; ++k) {
      const fs::path p = fs::path("runs/orbits/stab496") / ("co0_stab_gen" + std::to_string(k) + ".txt");
      if (fs::exists(p)) co0.push_back(p);
    }
    if (!co0.empty()) {
      std::vector<IndexPerm> gp;
      for (const fs::path& p : co0) {
        gp.push_back(kiss::index_permutation(ctx.L, kiss::load_aut(p)));  // throws unless an automorphism
        std::vector<uint32_t> img = kiss::apply(gp.back(), ctx.S_idx);
        std::sort(img.begin(), img.end());
        if (img != ctx.S_idx) throw std::runtime_error(p.string() + " does not fix the reference set");
      }
      run_group(ctx, "Stab_Co0(496)", "?", gp);
    }
  }
  // 2. the sign group 2^12 (orbits = sign classes); with α; with M24 (= the monomial group)
  {
    std::vector<Monomial> all = kiss::monomial_generators(group_dir, true);
    std::vector<Monomial> signs(all.begin() + 3, all.end());
    run_monomial(ctx, "2^12", signs, false);
    std::vector<Monomial> sa = signs;
    sa.push_back(alpha);
    run_monomial(ctx, "2^12:23", sa, false);
    run_monomial(ctx, "2^12:M24", all, false);
  }
  // 3. cyclic groups, one element per M24 element order
  {
    std::map<int, Monomial> by_order;
    std::uniform_int_distribution<int> LEN(8, 40);
    for (int t = 0; t < 20000 && by_order.size() < 14; ++t) {
      const Monomial m = kiss::random_monomial_word(m24, LEN(rng), rng);
      const int k = kiss::monomial_order(m);
      if (k > 1 && !by_order.count(k)) by_order.emplace(k, m);
    }
    std::string found;
    for (const auto& kv : by_order) found += (found.empty() ? "" : ",") + std::to_string(kv.first);
    std::printf("M24 element orders sampled: %s\n", found.c_str());
    std::uniform_int_distribution<std::size_t> CW(1, codewords.size() - 2);
    for (const auto& kv : by_order) {
      const int k = kv.first;
      const Monomial& pi = kv.second;
      run_monomial(ctx, "C" + std::to_string(k), {pi}, false);
      run_monomial(ctx, "C" + std::to_string(k), {pi}, true);
      // random Golay sign part on the same permutation
      const uint32_t c = codewords[CW(rng)];
      const Monomial ms = kiss::compose(kiss::sign_flip(c), pi);
      run_monomial(ctx, "C" + std::to_string(k) + "s" + std::to_string(kiss::monomial_order(ms)), {ms}, false);
    }
  }
  // 4. 23:11, PSL(2,23), M24
  const Monomial mu2 = multiplier2();
  run_monomial(ctx, "23:11", {alpha, mu2}, false);
  run_monomial(ctx, "23:11", {alpha, mu2}, true);
  run_monomial(ctx, "23:11.s", {kiss::compose(kiss::sign_flip(octads[0]), alpha), mu2}, false);
  run_monomial(ctx, "PSL(2,23)", {alpha, gamma}, false);
  run_monomial(ctx, "PSL(2,23)", {alpha, gamma}, true);
  run_monomial(ctx, "M24", {alpha, gamma, delta}, false);
  // 5. subgroups exported from sympy
  if (!subgroups_dir.empty() && fs::is_directory(subgroups_dir)) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(subgroups_dir))
      if (e.path().extension() == ".txt") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const fs::path& f : files) {
      const std::vector<Monomial> gens = kiss::load_monomial_list(f);
      if (gens.empty()) continue;
      run_monomial(ctx, f.stem().string(), gens, false);
      run_monomial(ctx, f.stem().string(), gens, true);
    }
  }
  // 6. dihedral groups <xi, m> for involutions m
  {
    const IndexPerm xi = kiss::index_permutation(ctx.L, kiss::load_aut(group_dir / "xi.txt"));
    std::vector<std::pair<std::string, Monomial>> invs = {{"gamma", gamma}, {"-I", negI}};
    for (int k = 0; k < 3; ++k)
      invs.emplace_back("flip(oct" + std::to_string(k) + ")", kiss::sign_flip(octads[static_cast<std::size_t>(k)]));
    invs.emplace_back("gamma.flip(oct0)", kiss::compose(gamma, kiss::sign_flip(octads[0])));
    int got = 0;
    std::uniform_int_distribution<int> LEN(8, 40);
    for (int t = 0; t < 20000 && got < 3; ++t) {
      const Monomial m = kiss::random_monomial_word(m24, LEN(rng), rng);
      if (kiss::monomial_order(m) == 2) invs.emplace_back("inv" + std::to_string(got++), m);
    }
    for (const auto& nm : invs) {
      const Monomial& m = nm.second;
      if (kiss::monomial_order(m) != 2) continue;
      const IndexPerm gm = kiss::index_permutation(ctx.L, m);
      const uint64_t n = kiss::index_perm_order(kiss::compose(xi, gm));
      run_group(ctx, "D" + std::to_string(2 * n) + "=<xi," + nm.first + ">", std::to_string(2 * n), {xi, gm});
    }
  }
  // 7. random Co_0 conjugates of C23 and 23:11 (must reproduce their rows)
  {
    std::vector<IndexPerm> auts;
    for (int k = 0; k < 2; ++k) {
      const fs::path p = fs::path("runs/orbits/auts") / ("aut_000" + std::to_string(k) + ".u32");
      if (fs::exists(p)) auts.push_back(kiss::read_index_perm(p));
    }
    if (auts.empty()) {
      kiss::ProductReplacement pr(kiss::co0_generator_perms(ctx.L, group_dir), 10, ctx.seed + 7);
      pr.burn_in(100);
      for (int k = 0; k < 2; ++k) auts.push_back(pr.next());
    }
    const IndexPerm a = kiss::index_permutation(ctx.L, alpha), u = kiss::index_permutation(ctx.L, mu2);
    for (std::size_t k = 0; k < auts.size(); ++k) {
      const IndexPerm gi = kiss::inverse(auts[k]);
      auto conj = [&](const IndexPerm& h) { return kiss::compose(auts[k], kiss::compose(h, gi)); };
      run_group(ctx, "g" + std::to_string(k) + ".C23.g^-1", "23", {conj(a)});
      run_group(ctx, "g" + std::to_string(k) + ".(23:11).g^-1", "253", {conj(a), conj(u)});
    }
  }
  std::fclose(ctx.rows_file);
  ctx.rows_file = nullptr;
  write_table(ctx, dir / "table.md");
}

}  // namespace

int main(int argc, char** argv) {
  std::string data_dir = "data", adj_path, set_path, name, group_file, suite_dir, subgroups_dir;
  std::vector<std::string> perm_files, aut_files;
  bool antipodal = false;
  Ctx ctx;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* opt) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", opt); std::exit(2); }
      return argv[++i];
    };
    if (a == "--data") data_dir = val("--data");
    else if (a == "--adj") adj_path = val("--adj");
    else if (a == "--set") set_path = val("--set");
    else if (a == "--seed") ctx.seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
    else if (a == "--ls-seconds") ctx.ls_seconds = std::atof(val("--ls-seconds").c_str());
    else if (a == "--bb-seconds") ctx.bb_seconds = std::atof(val("--bb-seconds").c_str());
    else if (a == "--max-explicit") ctx.max_explicit = static_cast<uint32_t>(std::atoi(val("--max-explicit").c_str()));
    else if (a == "--found") ctx.found_dir = val("--found");
    else if (a == "--export-graph") ctx.export_path = val("--export-graph");
    else if (a == "--name") name = val("--name");
    else if (a == "--group-file") group_file = val("--group-file");
    else if (a == "--perm-file") perm_files.push_back(val("--perm-file"));
    else if (a == "--aut-file") aut_files.push_back(val("--aut-file"));
    else if (a == "--antipodal") antipodal = true;
    else if (a == "--suite") suite_dir = val("--suite");
    else if (a == "--subgroups-dir") subgroups_dir = val("--subgroups-dir");
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (group_file.empty() && perm_files.empty() && aut_files.empty() && suite_dir.empty()) {
    std::fprintf(stderr, "usage: orbit_mis (--group-file F | --perm-file F.. | --aut-file F.. | --suite DIR) [options]\n");
    return 2;
  }
  const auto t0 = std::chrono::steady_clock::now();
  try {
    if (adj_path.empty()) adj_path = (fs::path(data_dir) / "adj.u32").string();
    if (set_path.empty()) set_path = (fs::path(data_dir) / "S496.txt").string();
    try { ctx.L = kiss::load_leech(data_dir); } catch (const std::exception&) { ctx.L = kiss::generate_leech(); }
    ctx.A = std::make_unique<kiss::Adjacency>(adj_path);
    const std::vector<kiss::Vec> S = kiss::read_set(set_path);
    const kiss::VerifyResult vr = kiss::verify_independent(ctx.L, S);
    if (!vr.ok) throw std::runtime_error("reference set " + set_path + " is not independent: " + vr.message);
    ctx.S_idx = kiss::set_indices(ctx.L, S);
    std::sort(ctx.S_idx.begin(), ctx.S_idx.end());
    ctx.set_name = fs::path(set_path).stem().string();
    std::printf("reference set %s: %zu vectors; adjacency %s; ls=%.0fs bb=%.0fs max_explicit=%u seed=%llu\n",
                set_path.c_str(), ctx.S_idx.size(), adj_path.c_str(), ctx.ls_seconds, ctx.bb_seconds,
                ctx.max_explicit, static_cast<unsigned long long>(ctx.seed));
    if (!suite_dir.empty()) {
      run_suite(ctx, suite_dir, subgroups_dir, data_dir);
    } else if (!group_file.empty()) {
      std::vector<Monomial> gens = kiss::load_monomial_list(group_file);
      if (name.empty()) name = fs::path(group_file).stem().string();
      run_monomial(ctx, name, gens, antipodal);
    } else {
      std::vector<IndexPerm> gens;
      for (const std::string& f : perm_files) gens.push_back(kiss::read_index_perm(f));
      for (const std::string& f : aut_files) gens.push_back(kiss::index_permutation(ctx.L, kiss::load_aut(f)));
      if (antipodal) gens.push_back(kiss::index_permutation(ctx.L, kiss::sign_flip(0xFFFFFFu)));
      if (name.empty()) name = "perms";
      run_group(ctx, name, "?", gens);
    }
    uint64_t best = 0;
    std::string best_name;
    for (const Row& r : ctx.rows)
      if (r.best > best) { best = r.best; best_name = r.name; }
    std::printf("RESULT ok=%d groups=%zu errors=%d best=%llu best_group=%s reference=%zu exceeded=%d seconds=%.1f\n",
                ctx.errors == 0 ? 1 : 0, ctx.rows.size(), ctx.errors, static_cast<unsigned long long>(best),
                best_name.c_str(), ctx.S_idx.size(), best > ctx.S_idx.size() ? 1 : 0, seconds_since(t0));
    return ctx.errors == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("RESULT ok=0 error=\"%s\"\n", e.what());
    return 1;
  }
}

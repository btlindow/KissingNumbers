// T3.2b — acceptance test for the per-chain ILS kernel (cuda/ls_search.cuh,
// cuda/ls_search.cu) on top of T3.2a's state and moves.
//
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) without a CUDA device, without
// data/adj.u32, or with < 5.5 GB of free device memory after retrying for a few
// minutes (the GPU is shared), 1 on failure. Last line: RESULT ok=<0|1> ...
//
// Stages (B = 512 unless --small, which uses B = 8 for the sanitizers)
//   inv      mixed seeds (subsets of the 496 / 488, greedy sets), 20 launches x K=50,
//            stall_limit small so restarts fire; after EVERY launch: lsa_check == 0
//            errors, every S and best_S independent (verify_independent), best
//            monotone, best_hash == hash(best_S), n1 / tabu bitmap == recompute on a
//            rotating subset of chains, restart_req consistent with the restart counter.
//   inv-anti the same in antipodal mode (+ antipodal_bad == 0).
//   refill   460-subsets of the 496, both modes: chains back at 496 after 1/5/20 launches;
//            target = 496 so the output slots are exercised (each reported set verified).
//   climb    random greedy maximal sets (~231): best after 1/5/20 launches.
//   s488     all chains seeded with the 488, both modes: any 489+? best distribution, dips.
//   s496     all chains seeded with the 496, both modes: any 497? different 496s reached?
//   plateau  lsk_apply_swaps with T3.1's four (12,12) moves (validated on the CPU first).
//   bench    B = 512 / 2048 / 4096: iterations/s per chain and aggregate, moves/s.
// Any set >= 497 is written to runs/found/ immediately and verified with
// tools/verify_s and python/verify_S.py.
//
// Usage: test_ls_search [--data DIR] [--seed S] [--small] [--no-bench] [--launches n] [--K k] [--only STAGE]
//   --only runs the stages whose name starts with STAGE (e.g. "climb", "s496") — for long experiments;
//   --strength/--greedy-pct/--tmin/--tmax/--list-tmax/--tenure/--max-drop/--select-scan/--swap/--stall-limit/
//   --restart-del/--wpb override the LSSearchParams of every stage (parameter sweeps).
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "adjacency_cuda.h"
#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/verify.h"
#include "kiss_cuda.h"
#include "ls_search.cuh"
#include "ls_state.cuh"

using kiss::Adjacency;
using kiss::DEG;
using kiss::Leech;
using kiss::N;
using kiss::Vec;
using kiss::cuda::INS_WORDS;
using kiss::cuda::LS_NONE;
using kiss::cuda::LSChainStats;
using kiss::cuda::LSParams;
using kiss::cuda::LSSearch;
using kiss::cuda::LSSearchParams;
using kiss::cuda::LSState;
using kiss::cuda::SMAX;

namespace {

constexpr int kSkip = 77;
int g_failures = 0;
int g_fail_printed = 0;
std::filesystem::path g_exe_dir;
std::filesystem::path g_data_dir = "data";
uint32_t g_record_max = 0;   // largest verified set size seen
int g_records = 0;           // sets >= 497 written to runs/found

void fail(const std::string& msg) {
  ++g_failures;
  if (g_fail_printed++ < 200) std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

std::string mb(std::size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.1f MB", static_cast<double>(bytes) / 1048576.0);
  return buf;
}

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------
std::vector<uint32_t> load_indices(const Leech& L, const std::filesystem::path& p) {
  std::vector<uint32_t> S;
  for (const Vec& r : kiss::read_set(p)) {
    const int32_t idx = L.index_of(r);
    if (idx < 0) throw std::runtime_error(p.string() + ": row is not a minimal vector");
    S.push_back(static_cast<uint32_t>(idx));
  }
  return S;
}

std::vector<Vec> to_vecs(const Leech& L, const std::vector<uint32_t>& S) {
  std::vector<Vec> v;
  v.reserve(S.size());
  for (uint32_t i : S) v.push_back(L.C[i]);
  return v;
}

// Random subset with `size` members of `full`; antipodal: size/2 antipodal pairs.
std::vector<uint32_t> subset_of(const Leech& L, const std::vector<uint32_t>& full, std::mt19937_64& rng, int size,
                                bool antipodal) {
  if (!antipodal) {
    std::vector<uint32_t> s = full;
    std::shuffle(s.begin(), s.end(), rng);
    s.resize(std::min<std::size_t>(s.size(), static_cast<std::size_t>(size)));
    return s;
  }
  std::vector<uint32_t> reps, out;
  for (uint32_t v : full)
    if (v < L.neg[v]) reps.push_back(v);
  std::shuffle(reps.begin(), reps.end(), rng);
  for (int i = 0; i < size / 2 && i < static_cast<int>(reps.size()); ++i) {
    out.push_back(reps[static_cast<std::size_t>(i)]);
    out.push_back(L.neg[reps[static_cast<std::size_t>(i)]]);
  }
  return out;
}

// Greedy random maximal independent set via the adjacency (antipodal: pairs).
std::vector<uint32_t> greedy_set(const Leech& L, const Adjacency& adj, std::mt19937_64& rng, bool antipodal) {
  std::vector<uint32_t> order(static_cast<std::size_t>(N));
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint8_t> blocked(static_cast<std::size_t>(N), 0);
  std::vector<uint32_t> S;
  auto take = [&](uint32_t v) {
    blocked[v] = 1;
    S.push_back(v);
    const uint32_t* row = adj.row(v);
    for (int i = 0; i < DEG; ++i) blocked[row[i]] = 1;
  };
  for (uint32_t v : order) {
    if (blocked[v]) continue;
    if (antipodal) {
      const uint32_t nv = L.neg[v];
      if (blocked[nv]) continue;
      take(v);
      take(nv);
    } else {
      take(v);
    }
  }
  return S;
}

// ---------------------------------------------------------------------------
// record custody: write_set -> verify_s -> verify_S.py
// ---------------------------------------------------------------------------
std::string utc_stamp() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

void handle_candidate(const Leech& L, const std::vector<uint32_t>& S, uint32_t chain, const std::string& stage) {
  const std::vector<Vec> V = to_vecs(L, S);
  const kiss::VerifyResult r = kiss::verify_independent(L, V);
  if (!r.ok) {
    fail(stage + ": reported set of size " + std::to_string(S.size()) + " is NOT independent: " + r.message);
    return;
  }
  g_record_max = std::max<uint32_t>(g_record_max, static_cast<uint32_t>(S.size()));
  if (S.size() < 497) return;
  ++g_records;
  std::filesystem::create_directories("runs/found");
  const std::filesystem::path f = std::filesystem::path("runs/found") /
                                  ("S_" + std::to_string(S.size()) + "_" + utc_stamp() + "_" + std::to_string(chain) + ".txt");
  kiss::write_set(f, V, "T3.2b test_ls_search stage " + stage + " chain " + std::to_string(chain));
  std::printf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  std::printf("!!! CANDIDATE RECORD: |S| = %zu from chain %u (stage %s) written to %s\n", S.size(), chain, stage.c_str(),
              f.c_str());
  std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  std::fflush(stdout);
  const std::filesystem::path tool = g_exe_dir / ".." / "tools" / "verify_s";
  const std::string c1 = tool.string() + " " + f.string() + " --data " + g_data_dir.string();
  const std::string c2 = ".venv/bin/python python/verify_S.py " + f.string();
  std::printf("!!! %s\n", c1.c_str());
  std::fflush(stdout);
  const int r1 = std::system(c1.c_str());
  std::printf("!!! verify_s exit=%d\n", r1);
  std::printf("!!! %s\n", c2.c_str());
  std::fflush(stdout);
  const int r2 = std::system(c2.c_str());
  std::printf("!!! verify_S.py exit=%d\n", r2);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// a stage = one state + one search, `launches` launches of K iterations
// ---------------------------------------------------------------------------
struct StageCfg {
  std::string name;
  LSParams p;
  LSSearchParams sp;
  std::vector<std::vector<uint32_t>> seeds;
  uint32_t target = 497;
  int launches = 20;
  int K = 50;
  uint64_t seed = 1;
  bool invariants = false;      // full checks after every launch
  int report_at[3] = {1, 5, 20};
  std::vector<std::vector<uint32_t>> plateau_rem, plateau_add;   // optional lsk_apply_swaps before launch 0
  uint32_t reference_hash = 0;  // if != 0: count chains whose best_S differs from it
  std::vector<uint32_t> reference;   // the set behind reference_hash (overlap histogram, distinct-set dump)
};

struct StageResult {
  std::vector<uint32_t> best;      // final best per chain
  std::vector<uint32_t> size;      // final size per chain
  std::vector<LSChainStats> stats;
  std::map<uint32_t, int> best_hist;
  uint32_t best_max = 0, best_min = 0, min_dip = 0;
  double mean_best = 0;
  int at_target = 0, differs_from_reference = 0;
  long long iterations = 0, adds = 0, removes = 0, swaps = 0, force_adds = 0, restarts = 0, reverts = 0, plateau = 0;
  double gpu_ms = 0;
  std::vector<int> at_target_after;   // per launch
  std::vector<double> mean_best_after;
  std::vector<uint32_t> max_best_after;
  std::vector<uint32_t> hashes;       // final best_hash per chain
  std::map<uint32_t, int> overlap_hist;   // |best_S ∩ reference| over chains whose best differs
  int distinct_best = 0;                  // distinct best_S (by hash) over all chains
};

uint32_t host_hash(const std::vector<uint32_t>& S) { return kiss::cuda::ls_set_hash_host(S.data(), S.size()); }

// n1 / tabu-bitmap / hash recompute on one downloaded chain.
void check_chain_deep(const Leech& L, const LSState& st, const LSSearch& ss, int b, const std::string& tag) {
  const kiss::cuda::LSHostChain h = kiss::cuda::lsa_download(st, b, true);
  uint32_t n1 = 0;
  for (int v = 0; v < N; ++v)
    if (h.tight[static_cast<std::size_t>(v)] == 1 && !((h.inS[static_cast<std::size_t>(v) >> 5] >> (v & 31)) & 1u)) ++n1;
  uint32_t dn1 = 0;
  KISS_CUDA_CHECK(cudaMemcpy(&dn1, ss.n1 + b, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  if (dn1 != n1) fail(tag + " chain " + std::to_string(b) + ": n1 device " + std::to_string(dn1) + " != " + std::to_string(n1));
  std::vector<uint32_t> tb(static_cast<std::size_t>(INS_WORDS));
  KISS_CUDA_CHECK(cudaMemcpy(tb.data(), ss.tabu_bits + static_cast<std::size_t>(b) * INS_WORDS,
                             tb.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::vector<uint32_t> ref(static_cast<std::size_t>(INS_WORDS), 0u);
  for (uint32_t v : h.tabu_v)
    if (v != LS_NONE && v < static_cast<uint32_t>(N)) ref[v >> 5] |= 1u << (v & 31);
  if (ref != tb) fail(tag + " chain " + std::to_string(b) + ": tabu bitmap != ring contents");
  // candidate-list entries are vertices (< N) outside S at push time; only range is an invariant
  std::vector<uint32_t> cand;
  uint32_t ch = 0;
  KISS_CUDA_CHECK(cudaMemcpy(&ch, ss.cand_head + b, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  if (ch > static_cast<uint32_t>(ss.sp.CL)) fail(tag + ": cand_head > CL");
  cand.resize(std::min<uint32_t>(ch, static_cast<uint32_t>(ss.sp.CL)));
  if (!cand.empty())
    KISS_CUDA_CHECK(cudaMemcpy(cand.data(), ss.cand + static_cast<std::size_t>(b) * ss.sp.CL,
                               cand.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (uint32_t v : cand)
    if (v >= static_cast<uint32_t>(N)) { fail(tag + ": candidate entry out of range"); break; }
  (void)L;
}

const LSSearchParams* g_sp_override = nullptr;

StageResult run_stage(const StageCfg& c0, const Leech& L, const kiss::cuda::DeviceLeech& dl, const uint32_t* d_adj) {
  StageCfg c = c0;
  if (g_sp_override) {
    const int stall = c.sp.stall_limit, drop = c.sp.max_drop;   // stage-specific values survive unless overridden
    c.sp = *g_sp_override;
    if (c.sp.stall_limit == LSSearchParams{}.stall_limit) c.sp.stall_limit = stall;
    if (c.sp.max_drop == LSSearchParams{}.max_drop) c.sp.max_drop = drop;
  }
  StageResult res;
  const int B = c.p.B;
  LSState st = kiss::cuda::lsa_alloc(c.p);
  LSSearch ss;
  uint32_t* d_flag = nullptr;
  uint32_t* d_out = nullptr;
  const int nslots = 64;
  try {
    kiss::cuda::lsa_init_from_sets(st, c.seeds, dl, d_adj, c.seed);
    ss = kiss::cuda::lss_alloc(st, c.sp);
    kiss::cuda::lss_init(ss, st, dl, c.seed * 7919 + 17);
    KISS_CUDA_CHECK(cudaMalloc(&d_flag, sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(uint32_t)));
    const std::size_t ow = kiss::cuda::ls_out_words(nslots);
    KISS_CUDA_CHECK(cudaMalloc(&d_out, ow * sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaMemset(d_out, 0, ow * sizeof(uint32_t)));

    // optional scripted plateau moves
    if (!c.plateau_rem.empty()) {
      int R = 0, A = 0;
      for (const auto& r : c.plateau_rem) R = std::max<int>(R, static_cast<int>(r.size()));
      for (const auto& a : c.plateau_add) A = std::max<int>(A, static_cast<int>(a.size()));
      std::vector<uint32_t> hr(static_cast<std::size_t>(B) * R, LS_NONE), ha(static_cast<std::size_t>(B) * A, LS_NONE);
      for (int b = 0; b < B; ++b) {
        for (std::size_t i = 0; i < c.plateau_rem[static_cast<std::size_t>(b)].size(); ++i)
          hr[static_cast<std::size_t>(b) * R + i] = c.plateau_rem[static_cast<std::size_t>(b)][i];
        for (std::size_t i = 0; i < c.plateau_add[static_cast<std::size_t>(b)].size(); ++i)
          ha[static_cast<std::size_t>(b) * A + i] = c.plateau_add[static_cast<std::size_t>(b)][i];
      }
      uint32_t *dr = nullptr, *da = nullptr, *dret = nullptr;
      KISS_CUDA_CHECK(cudaMalloc(&dr, hr.size() * sizeof(uint32_t)));
      KISS_CUDA_CHECK(cudaMalloc(&da, ha.size() * sizeof(uint32_t)));
      KISS_CUDA_CHECK(cudaMalloc(&dret, static_cast<std::size_t>(B) * 2 * sizeof(uint32_t)));
      KISS_CUDA_CHECK(cudaMemcpy(dr, hr.data(), hr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
      KISS_CUDA_CHECK(cudaMemcpy(da, ha.data(), ha.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
      kiss::cuda::lsk_apply_swaps(st, ss, dr, R, da, A, d_adj, dl, static_cast<uint32_t>(c.sp.tenure), dret);
      KISS_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint32_t> ret(static_cast<std::size_t>(B) * 2);
      KISS_CUDA_CHECK(cudaMemcpy(ret.data(), dret, ret.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      const std::vector<kiss::cuda::LSCheck> chk = kiss::cuda::lsa_check(st, dl);
      int errs = 0;
      for (int b = 0; b < B; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        errs += chk[bb].errors() != 0;
        const uint32_t exp_r = static_cast<uint32_t>(c.plateau_rem[bb].size()) * (c.p.antipodal ? 1u : 1u);
        const uint32_t exp_a = static_cast<uint32_t>(c.plateau_add[bb].size());
        if (ret[bb * 2] != exp_r || ret[bb * 2 + 1] != exp_a)
          fail(c.name + " chain " + std::to_string(b) + ": plateau move applied removed=" + std::to_string(ret[bb * 2]) +
               " added=" + std::to_string(ret[bb * 2 + 1]) + " expected " + std::to_string(exp_r) + "/" + std::to_string(exp_a));
        const kiss::cuda::LSHostChain h = kiss::cuda::lsa_download(st, b);
        const kiss::VerifyResult vr = kiss::verify_independent(L, to_vecs(L, h.S));
        if (!vr.ok) fail(c.name + " chain " + std::to_string(b) + ": after plateau move S not independent: " + vr.message);
        std::printf("%-9s: chain %d plateau move removed=%u added=%u -> |S|=%u independent=%d lsa_check errors=%u\n",
                    c.name.c_str(), b, ret[bb * 2], ret[bb * 2 + 1], h.size, vr.ok ? 1 : 0, chk[bb].errors());
      }
      if (errs) fail(c.name + ": lsa_check errors after plateau moves");
      KISS_CUDA_CHECK(cudaFree(dr));
      KISS_CUDA_CHECK(cudaFree(da));
      KISS_CUDA_CHECK(cudaFree(dret));
    }

    std::vector<uint32_t> prev_best(static_cast<std::size_t>(B), 0u);
    KISS_CUDA_CHECK(cudaMemcpy(prev_best.data(), st.best_size, prev_best.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> prev_restarts(static_cast<std::size_t>(B), 0u);
    cudaEvent_t e0, e1;
    KISS_CUDA_CHECK(cudaEventCreate(&e0));
    KISS_CUDA_CHECK(cudaEventCreate(&e1));
    int deep_rot = 0;
    for (int launch = 0; launch < c.launches; ++launch) {
      KISS_CUDA_CHECK(cudaEventRecord(e0));
      kiss::cuda::lsk_run_chains(st, ss, d_adj, dl, c.K, c.target, d_flag, d_out, nslots);
      KISS_CUDA_CHECK(cudaEventRecord(e1));
      KISS_CUDA_CHECK(cudaEventSynchronize(e1));
      float fms = 0;
      KISS_CUDA_CHECK(cudaEventElapsedTime(&fms, e0, e1));
      res.gpu_ms += fms;

      // output slots (record custody)
      uint32_t flag = 0;
      KISS_CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
      const std::vector<kiss::cuda::LSFound> found = kiss::cuda::lss_download_output(d_out, nslots, d_flag, true);
      if (!found.empty() && !flag) fail(c.name + ": output slots claimed but flag not set");
      for (const kiss::cuda::LSFound& f : found) {
        if (f.size < c.target) fail(c.name + ": reported set below target");
        if (f.S.size() != f.size) fail(c.name + ": reported size mismatch");
        handle_candidate(L, f.S, f.chain, c.name);
      }

      // sizes / best after this launch
      std::vector<uint32_t> best(static_cast<std::size_t>(B)), size(static_cast<std::size_t>(B));
      KISS_CUDA_CHECK(cudaMemcpy(best.data(), st.best_size, best.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      KISS_CUDA_CHECK(cudaMemcpy(size.data(), st.size, size.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      int at = 0;
      double mean = 0;
      uint32_t mx = 0;
      for (int b = 0; b < B; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        if (best[bb] < prev_best[bb]) fail(c.name + " chain " + std::to_string(b) + ": best decreased");
        at += best[bb] >= c.target;
        mean += best[bb];
        mx = std::max(mx, best[bb]);
        if (best[bb] > g_record_max && best[bb] >= 497) {
          // must have been reported through a slot (target <= 497 in every stage) — checked above
        }
      }
      mean /= B;
      res.at_target_after.push_back(at);
      res.mean_best_after.push_back(mean);
      res.max_best_after.push_back(mx);
      prev_best = best;

      const bool check_now = c.invariants || launch + 1 == c.launches;
      if (check_now) {
        const std::vector<kiss::cuda::LSCheck> chk = kiss::cuda::lsa_check(st, dl);
        int errs = 0;
        for (int b = 0; b < B; ++b) {
          const kiss::cuda::LSCheck& k = chk[static_cast<std::size_t>(b)];
          if (k.errors()) {
            ++errs;
            fail(c.name + " launch " + std::to_string(launch) + " chain " + std::to_string(b) + ": lsa_check errors=" +
                 std::to_string(k.errors()) + " (tight " + std::to_string(k.tight_mismatch) + ", list " +
                 std::to_string(k.list_bad) + ", bitmap " + std::to_string(k.bitmap_bad) + ", conflicts " +
                 std::to_string(k.conflict_pairs) + ", antipodal " + std::to_string(k.antipodal_bad) + ", fl " +
                 std::to_string(k.fl_bad) + ")");
          }
        }
        // independence of S and best_S via the CPU verifier; best_hash
        std::vector<kiss::cuda::LSHostChain> hs(static_cast<std::size_t>(B));
        for (int b = 0; b < B; ++b) hs[static_cast<std::size_t>(b)] = kiss::cuda::lsa_download(st, b);
        std::vector<uint32_t> dh(static_cast<std::size_t>(B));
        KISS_CUDA_CHECK(cudaMemcpy(dh.data(), ss.best_hash, dh.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::vector<int> bad_s(static_cast<std::size_t>(B), 0), bad_b(static_cast<std::size_t>(B), 0);
#pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < B; ++b) {
          const std::size_t bb = static_cast<std::size_t>(b);
          const kiss::VerifyResult r1 = kiss::verify_independent(L, to_vecs(L, hs[bb].S));
          const kiss::VerifyResult r2 = kiss::verify_independent(L, to_vecs(L, hs[bb].best_S));
          bad_s[bb] = !r1.ok;
          bad_b[bb] = !r2.ok;
          if (c.p.antipodal) {
            for (uint32_t v : hs[bb].S)
              if (std::find(hs[bb].S.begin(), hs[bb].S.end(), L.neg[v]) == hs[bb].S.end()) { bad_s[bb] = 2; break; }
          }
        }
        for (int b = 0; b < B; ++b) {
          const std::size_t bb = static_cast<std::size_t>(b);
          if (bad_s[bb] == 1) fail(c.name + " chain " + std::to_string(b) + ": S not independent");
          if (bad_s[bb] == 2) fail(c.name + " chain " + std::to_string(b) + ": S not antipodal-closed");
          if (bad_b[bb]) fail(c.name + " chain " + std::to_string(b) + ": best_S not independent");
          if (hs[bb].best_size != best[bb]) fail(c.name + ": best_size mismatch on download");
          if (host_hash(hs[bb].best_S) != dh[bb]) fail(c.name + " chain " + std::to_string(b) + ": best_hash mismatch");
          if (hs[bb].size != size[bb]) fail(c.name + ": size mismatch on download");
          if (hs[bb].best_S.size() < hs[bb].S.size()) fail(c.name + ": best smaller than current");
        }
        // deep checks on a rotating subset (tight download = 393 KB per chain)
        const int ndeep = std::min(B, 16);
        for (int i = 0; i < ndeep; ++i) {
          const int b = (deep_rot + i) % B;
          check_chain_deep(L, st, ss, b, c.name + " launch " + std::to_string(launch));
        }
        deep_rot += ndeep;
        // restart requests vs restart counter
        const std::vector<uint32_t> rr = kiss::cuda::lss_download_restart_req(ss, true);
        const std::vector<LSChainStats> stt = kiss::cuda::lss_download_stats(ss);
        for (int b = 0; b < B; ++b) {
          const std::size_t bb = static_cast<std::size_t>(b);
          const bool restarted = stt[bb].restarts != prev_restarts[bb];
          if (rr[bb] && !restarted) fail(c.name + ": restart_req set without a restart");
          if (!rr[bb] && restarted) fail(c.name + ": restart without restart_req");
          prev_restarts[bb] = stt[bb].restarts;
          if (stt[bb].best_size != best[bb] || stt[bb].size != size[bb]) fail(c.name + ": stats snapshot mismatch");
          if (stt[bb].min_size > size[bb]) fail(c.name + ": min_size > size");
        }
        if (c.invariants)
          std::printf("%-9s: launch %2d ok: lsa_check errors=%d, %d chains S/best_S independent, best monotone; best mean %.1f max %u, at target %d\n",
                      c.name.c_str(), launch, errs, B, mean, mx, at);
      }
      if (!c.invariants && (launch + 1 == c.report_at[0] || launch + 1 == c.report_at[1] || launch + 1 == c.report_at[2] ||
                            launch + 1 == c.launches)) {
        uint32_t mn = 0xffffffffu;
        for (uint32_t v : best) mn = std::min(mn, v);
        std::printf("%-9s: after %2d launches (%d iters): best min %u mean %.1f max %u, at target(%u): %d/%d, gpu %.0f ms\n",
                    c.name.c_str(), launch + 1, (launch + 1) * c.K, mn, mean, mx, c.target, at, B, res.gpu_ms);
        std::fflush(stdout);
      }
    }
    KISS_CUDA_CHECK(cudaEventDestroy(e0));
    KISS_CUDA_CHECK(cudaEventDestroy(e1));

    // final summary
    res.best.resize(static_cast<std::size_t>(B));
    res.size.resize(static_cast<std::size_t>(B));
    KISS_CUDA_CHECK(cudaMemcpy(res.best.data(), st.best_size, res.best.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    KISS_CUDA_CHECK(cudaMemcpy(res.size.data(), st.size, res.size.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    res.stats = kiss::cuda::lss_download_stats(ss);
    res.hashes.resize(static_cast<std::size_t>(B));
    KISS_CUDA_CHECK(cudaMemcpy(res.hashes.data(), ss.best_hash, res.hashes.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    res.best_min = 0xffffffffu;
    res.min_dip = 0xffffffffu;
    for (int b = 0; b < B; ++b) {
      const std::size_t bb = static_cast<std::size_t>(b);
      res.best_hist[res.best[bb]]++;
      res.best_max = std::max(res.best_max, res.best[bb]);
      res.best_min = std::min(res.best_min, res.best[bb]);
      res.mean_best += res.best[bb];
      res.at_target += res.best[bb] >= c.target;
      const LSChainStats& s = res.stats[bb];
      res.min_dip = std::min(res.min_dip, s.min_size);
      res.iterations += s.iterations;
      res.adds += s.adds;
      res.removes += s.removes;
      res.swaps += s.swaps;
      res.force_adds += s.force_adds;
      res.restarts += s.restarts;
      res.reverts += s.reverts;
      res.plateau += s.plateau;
      if (c.reference_hash && res.hashes[bb] != c.reference_hash) ++res.differs_from_reference;
    }
    res.mean_best /= B;
    if (c.reference_hash && !c.reference.empty()) {
      std::vector<uint8_t> inref(static_cast<std::size_t>(N), 0);
      for (uint32_t v : c.reference) inref[v] = 1;
      std::map<uint32_t, std::vector<uint32_t>> distinct;   // hash -> one representative best_S
      for (int b = 0; b < B; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        const kiss::cuda::LSHostChain h = kiss::cuda::lsa_download(st, b);
        if (!distinct.count(res.hashes[bb])) distinct[res.hashes[bb]] = h.best_S;
        if (res.hashes[bb] != c.reference_hash) {
          uint32_t ov = 0;
          for (uint32_t v : h.best_S) ov += inref[v];
          res.overlap_hist[ov]++;
        }
      }
      res.distinct_best = static_cast<int>(distinct.size());
      std::filesystem::create_directories("runs/ls_search");
      const std::filesystem::path f = std::filesystem::path("runs/ls_search") /
                                      (c.name + "_B" + std::to_string(B) + "_it" + std::to_string(c.launches * c.K) + "_bestsets.txt");
      if (FILE* fp = std::fopen(f.c_str(), "w")) {
        std::fprintf(fp, "# %s: distinct best_S per chain (vertex indices, canonical order); reference hash %08x\n",
                     c.name.c_str(), c.reference_hash);
        for (const auto& kv : distinct) {
          std::vector<uint32_t> v = kv.second;
          std::sort(v.begin(), v.end());
          std::fprintf(fp, "%08x %zu", kv.first, v.size());
          for (uint32_t x : v) std::fprintf(fp, " %u", x);
          std::fprintf(fp, "\n");
        }
        std::fclose(fp);
      }
    }
    KISS_CUDA_CHECK(cudaFree(d_flag));
    KISS_CUDA_CHECK(cudaFree(d_out));
  } catch (...) {
    if (d_flag) cudaFree(d_flag);
    if (d_out) cudaFree(d_out);
    kiss::cuda::lss_free(ss);
    kiss::cuda::lsa_free(st);
    throw;
  }
  kiss::cuda::lss_free(ss);
  kiss::cuda::lsa_free(st);
  return res;
}

std::string hist_str(const std::map<uint32_t, int>& h) {
  std::string s;
  for (const auto& kv : h) s += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
  return s;
}

void print_summary(const StageCfg& c, const StageResult& r, double wall_ms) {
  const double it_s = r.iterations / (r.gpu_ms / 1000.0);
  std::printf("%-9s: B=%d K=%d launches=%d antipodal=%d | best min %u mean %.1f max %u | at target(%u) %d/%d | deepest dip %u | "
              "best hist %s\n",
              c.name.c_str(), c.p.B, c.K, c.launches, c.p.antipodal ? 1 : 0, r.best_min, r.mean_best, r.best_max, c.target,
              r.at_target, c.p.B, r.min_dip, hist_str(r.best_hist).c_str());
  std::printf("%-9s: iterations %lld (%.3g/s aggregate, %.1f/s per chain), adds %lld removes %lld (%.3g row-updates/s), "
              "swaps %lld, force_adds %lld, reverts %lld, restarts %lld, plateau copies %lld; gpu %.0f ms wall %.0f ms\n",
              c.name.c_str(), r.iterations, it_s, it_s / c.p.B, r.adds, r.removes,
              (r.adds + r.removes) / (r.gpu_ms / 1000.0), r.swaps, r.force_adds, r.reverts, r.restarts, r.plateau,
              r.gpu_ms, wall_ms);
  if (c.reference_hash)
    std::printf("%-9s: chains whose best_S differs from the reference set: %d/%d; distinct best sets %d; overlap |best ∩ ref| hist: %s\n",
                c.name.c_str(), r.differs_from_reference, c.p.B, r.distinct_best, hist_str(r.overlap_hist).c_str());
  std::fflush(stdout);
}

// T3.1's four (12,12) plateau moves of the 496: S positions (into the ascending
// index list of data/S496.txt) to remove, vertices to add.
struct PlateauMove {
  int rem_pos[12];
  uint32_t add[12];
};
const PlateauMove kMoves496[4] = {
    {{8, 14, 101, 103, 127, 209, 286, 368, 392, 394, 481, 487},
     {4479, 4901, 26476, 27189, 52296, 85215, 111344, 144263, 169370, 170083, 191658, 192080}},
    {{30, 36, 119, 144, 148, 172, 323, 347, 351, 376, 459, 465},
     {13131, 17682, 24333, 39173, 55836, 94360, 102199, 140723, 157386, 172226, 178877, 183428}},
    {{43, 51, 53, 89, 223, 234, 261, 272, 406, 442, 444, 452},
     {7694, 19999, 45656, 61824, 70844, 87859, 108700, 125715, 134735, 150903, 176560, 188865}},
    {{84, 133, 138, 162, 189, 203, 292, 306, 333, 357, 362, 411},
     {33755, 45348, 63855, 70602, 73008, 79976, 116583, 123551, 125957, 132704, 151211, 162804}},
};

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  bool small = false, bench = true;
  int launches = 20, K = 50;
  std::string only;
  LSSearchParams spo;   // overrides applied to every stage
  bool have_spo = false;
  uint64_t seed = 20260826ull;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--launches") && i + 1 < argc) launches = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--K") && i + 1 < argc) K = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    else if (!std::strcmp(argv[i], "--strength") && i + 1 < argc) { spo.strength = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--greedy-pct") && i + 1 < argc) { spo.greedy_pct = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--tmin") && i + 1 < argc) { spo.tmin = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--tmax") && i + 1 < argc) { spo.tmax = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--list-tmax") && i + 1 < argc) { spo.list_tmax = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--tenure") && i + 1 < argc) { spo.tenure = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--max-drop") && i + 1 < argc) { spo.max_drop = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--select-scan") && i + 1 < argc) { spo.select_scan = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--swap") && i + 1 < argc) { spo.swap_enable = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--stall-limit") && i + 1 < argc) { spo.stall_limit = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--restart-del") && i + 1 < argc) { spo.restart_del = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--wpb") && i + 1 < argc) { spo.warps_per_block = std::atoi(argv[++i]); have_spo = true; }
    else if (!std::strcmp(argv[i], "--small")) small = true;
    else if (!std::strcmp(argv[i], "--no-bench")) bench = false;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  g_exe_dir = std::filesystem::absolute(std::filesystem::path(argv[0])).parent_path();
  if (have_spo) {
    g_sp_override = &spo;
    std::printf("params    : overrides strength=%d greedy_pct=%d tmin=%d tmax=%d list_tmax=%d tenure=%d max_drop=%d select_scan=%d swap=%d stall_limit=%d restart_del=%d wpb=%d\n",
                spo.strength, spo.greedy_pct, spo.tmin, spo.tmax, spo.list_tmax, spo.tenure, spo.max_drop, spo.select_scan,
                spo.swap_enable, spo.stall_limit, spo.restart_del, spo.warps_per_block);
  }
  g_data_dir = data_dir;
  if (small) { launches = std::min(launches, 3); K = std::min(K, 5); bench = false; }
  const int B = small ? 8 : 512;

  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::printf("test_ls_search: no CUDA device (%s) — skipping\n", cudaGetErrorString(e));
    std::printf("RESULT ok=1 skipped=1 devices=0\n");
    return kSkip;
  }
  const std::filesystem::path adj_path = data_dir / "adj.u32";
  if (!std::filesystem::exists(adj_path)) {
    std::printf("test_ls_search: %s missing (run tools/build_adj) — skipping\n", adj_path.c_str());
    std::printf("RESULT ok=1 skipped=1 reason=no_adjacency\n");
    return kSkip;
  }
  {
    const std::size_t need = static_cast<std::size_t>(5.5 * 1024 * 1024 * 1024);
    std::size_t fr = 0, to = 0;
    for (int attempt = 0; attempt < 12; ++attempt) {
      kiss::cuda::device_mem_info(&fr, &to);
      if (fr >= need) break;
      std::printf("test_ls_search: only %s of %s device memory free (need 5.5 GB) — retry %d/12 in 15 s\n", mb(fr).c_str(),
                  mb(to).c_str(), attempt + 1);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(15));
    }
    if (fr < need) {
      std::printf("test_ls_search: %s free of %s — skipping (another process holds the GPU)\n", mb(fr).c_str(), mb(to).c_str());
      std::printf("RESULT ok=1 skipped=1 reason=vram free=%zu\n", fr);
      return kSkip;
    }
  }

  int stages = 0;
  double t_total = 0;
  std::map<std::string, std::string> summary;
  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const Adjacency adj(adj_path);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    const std::vector<uint32_t> S496 = load_indices(L, data_dir / "S496.txt");
    const std::vector<uint32_t> S488 = load_indices(L, data_dir / "S488.txt");
    std::printf("fixture   : S496 |S|=%zu, S488 |S|=%zu, adjacency %s, omp threads %d\n", S496.size(), S488.size(),
                adj_path.c_str(), omp_get_max_threads());
    std::size_t free0 = 0, total0 = 0;
    kiss::cuda::device_mem_info(&free0, &total0);
    const auto ta = std::chrono::steady_clock::now();
    uint32_t* d_adj = kiss::cuda::adjacency_to_device(adj);
    std::size_t free1 = 0;
    kiss::cuda::device_mem_info(&free1, nullptr);
    std::printf("adjacency : uploaded in %.0f ms; device free %s -> %s\n", ms_since(ta), mb(free0).c_str(), mb(free1).c_str());
    {
      LSSearchParams sp;
      std::printf("memory    : search state %zu bytes per chain (+ %zu of T3.2a state)\n", kiss::cuda::lss_bytes_per_chain(sp),
                  kiss::cuda::lsa_report_memory(LSParams{}).per_chain);
    }
    std::mt19937_64 rng(seed);
    const uint32_t hash496 = host_hash(S496);
    const uint32_t hash488 = host_hash(S488);

    auto mixed_seeds = [&](bool antipodal) {
      std::vector<std::vector<uint32_t>> s;
      for (int b = 0; b < B; ++b) {
        const int kind = b % 3;
        if (kind == 0) s.push_back(subset_of(L, S496, rng, 400 + static_cast<int>(rng() % 96), antipodal));
        else if (kind == 1) s.push_back(subset_of(L, S488, rng, 400 + static_cast<int>(rng() % 88), antipodal));
        else s.push_back(greedy_set(L, adj, rng, antipodal));
      }
      return s;
    };
    auto same_seed = [&](const std::vector<uint32_t>& S) { return std::vector<std::vector<uint32_t>>(static_cast<std::size_t>(B), S); };
    auto want = [&](const char* name) { return only.empty() || std::string(name).rfind(only, 0) == 0; };

    // ---- inv / inv-anti -------------------------------------------------------------
    for (int mode = 0; mode < 2 && want("inv"); ++mode) {
      StageCfg c;
      c.name = mode ? "inv-anti" : "inv";
      c.p.B = B;
      c.p.antipodal = mode == 1;
      c.sp.stall_limit = 40;    // so restarts fire during the test
      c.sp.max_drop = 16;
      c.seeds = mixed_seeds(c.p.antipodal);
      c.target = 497;
      c.launches = mode ? std::max(1, launches / 2) : launches;
      c.K = K;
      c.seed = seed + 1 + static_cast<uint64_t>(mode);
      c.invariants = true;
      const auto t0 = std::chrono::steady_clock::now();
      const StageResult r = run_stage(c, L, dl, d_adj);
      print_summary(c, r, ms_since(t0));
      summary[c.name] = "restarts=" + std::to_string(r.restarts) + " swaps=" + std::to_string(r.swaps);
      ++stages;
    }

    // ---- refill from 460-subsets ---------------------------------------------------
    for (int mode = 0; mode < 2 && want("refill"); ++mode) {
      StageCfg c;
      c.name = mode ? "refill-a" : "refill";
      c.p.B = B;
      c.p.antipodal = mode == 1;
      for (int b = 0; b < B; ++b) c.seeds.push_back(subset_of(L, S496, rng, 460, c.p.antipodal));
      c.target = 496;
      c.launches = launches;
      c.K = K;
      c.seed = seed + 10 + static_cast<uint64_t>(mode);
      c.reference_hash = hash496;
      c.reference = S496;
      const auto t0 = std::chrono::steady_clock::now();
      const StageResult r = run_stage(c, L, dl, d_adj);
      print_summary(c, r, ms_since(t0));
      summary[c.name] = std::to_string(r.at_target) + "/" + std::to_string(B);
      ++stages;
    }

    // ---- climb from greedy sets ----------------------------------------------------
    if (want("climb")) {
      StageCfg c;
      c.name = "climb";
      c.p.B = B;
      for (int b = 0; b < B; ++b) c.seeds.push_back(greedy_set(L, adj, rng, false));
      uint32_t smin = 0xffffffffu, smax = 0;
      double smean = 0;
      for (const auto& s : c.seeds) {
        smin = std::min<uint32_t>(smin, static_cast<uint32_t>(s.size()));
        smax = std::max<uint32_t>(smax, static_cast<uint32_t>(s.size()));
        smean += s.size();
      }
      std::printf("climb    : greedy seeds %u..%u mean %.1f\n", smin, smax, smean / B);
      c.target = 497;
      c.launches = launches;
      c.K = K;
      c.seed = seed + 20;
      c.report_at[0] = 1; c.report_at[1] = std::max(1, launches / 4); c.report_at[2] = std::max(1, launches / 2);
      const auto t0 = std::chrono::steady_clock::now();
      const StageResult r = run_stage(c, L, dl, d_adj);
      print_summary(c, r, ms_since(t0));
      summary["climb"] = "mean_best=" + std::to_string(r.mean_best) + " max=" + std::to_string(r.best_max);
      ++stages;
    }

    // ---- seeds = the 488 -------------------------------------------------------------
    for (int mode = 0; mode < 2 && want("s488"); ++mode) {
      StageCfg c;
      c.name = mode ? "s488-a" : "s488";
      c.p.B = B;
      c.p.antipodal = mode == 1;
      c.seeds = same_seed(S488);
      c.target = 489;
      c.launches = launches;
      c.K = K;
      c.seed = seed + 30 + static_cast<uint64_t>(mode);
      c.reference_hash = hash488;
      c.reference = S488;
      const auto t0 = std::chrono::steady_clock::now();
      const StageResult r = run_stage(c, L, dl, d_adj);
      print_summary(c, r, ms_since(t0));
      summary[c.name] = "max=" + std::to_string(r.best_max) + " diff=" + std::to_string(r.differs_from_reference);
      ++stages;
    }

    // ---- seeds = the 496 -------------------------------------------------------------
    for (int mode = 0; mode < 2 && want("s496"); ++mode) {
      StageCfg c;
      c.name = mode ? "s496-a" : "s496";
      c.p.B = B;
      c.p.antipodal = mode == 1;
      c.seeds = same_seed(S496);
      c.target = 497;
      c.launches = launches;
      c.K = K;
      c.seed = seed + 40 + static_cast<uint64_t>(mode);
      c.reference_hash = hash496;
      c.reference = S496;
      const auto t0 = std::chrono::steady_clock::now();
      const StageResult r = run_stage(c, L, dl, d_adj);
      print_summary(c, r, ms_since(t0));
      summary[c.name] = "max=" + std::to_string(r.best_max) + " diff=" + std::to_string(r.differs_from_reference);
      ++stages;
    }

    // ---- plateau primitive: T3.1's (12,12) moves --------------------------------------
    if (want("plateau")) {
      std::vector<uint32_t> sorted = S496;
      std::sort(sorted.begin(), sorted.end());
      std::vector<std::vector<uint32_t>> rem(4), add(4);
      bool moves_ok = true;
      for (int m = 0; m < 4; ++m) {
        for (int i = 0; i < 12; ++i) {
          rem[static_cast<std::size_t>(m)].push_back(sorted[static_cast<std::size_t>(kMoves496[m].rem_pos[i])]);
          add[static_cast<std::size_t>(m)].push_back(kMoves496[m].add[i]);
        }
        // CPU validation: every added vertex conflicts with members of the removed set only,
        // the added vertices are mutually non-adjacent, none is in S
        for (uint32_t a : add[static_cast<std::size_t>(m)]) {
          if (std::find(S496.begin(), S496.end(), a) != S496.end()) moves_ok = false;
          for (uint32_t s : S496)
            if (kiss::dot(L.C[a], L.C[s]) == 16 &&
                std::find(rem[static_cast<std::size_t>(m)].begin(), rem[static_cast<std::size_t>(m)].end(), s) ==
                    rem[static_cast<std::size_t>(m)].end())
              moves_ok = false;
          for (uint32_t a2 : add[static_cast<std::size_t>(m)])
            if (a2 != a && kiss::dot(L.C[a], L.C[a2]) == 16) moves_ok = false;
        }
      }
      std::printf("plateau  : T3.1's four (12,12) moves validated on the CPU: %s\n", moves_ok ? "ok" : "INVALID");
      if (!moves_ok) fail("plateau: T3.1 moves do not validate against data/S496.txt (position convention?)");
      else {
        StageCfg c;
        c.name = "plateau";
        c.p.B = 8;
        c.p.antipodal = false;
        c.seeds = std::vector<std::vector<uint32_t>>(8, S496);
        c.plateau_rem.assign(8, {});
        c.plateau_add.assign(8, {});
        for (int m = 0; m < 4; ++m) {
          c.plateau_rem[static_cast<std::size_t>(m)] = rem[static_cast<std::size_t>(m)];
          c.plateau_add[static_cast<std::size_t>(m)] = add[static_cast<std::size_t>(m)];
        }
        for (int m = 0; m < 2; ++m) {   // chain 4: moves 0+1
          c.plateau_rem[4].insert(c.plateau_rem[4].end(), rem[static_cast<std::size_t>(m)].begin(), rem[static_cast<std::size_t>(m)].end());
          c.plateau_add[4].insert(c.plateau_add[4].end(), add[static_cast<std::size_t>(m)].begin(), add[static_cast<std::size_t>(m)].end());
        }
        for (int m = 0; m < 4; ++m) {   // chain 5: all four
          c.plateau_rem[5].insert(c.plateau_rem[5].end(), rem[static_cast<std::size_t>(m)].begin(), rem[static_cast<std::size_t>(m)].end());
          c.plateau_add[5].insert(c.plateau_add[5].end(), add[static_cast<std::size_t>(m)].begin(), add[static_cast<std::size_t>(m)].end());
        }
        c.target = 497;
        c.launches = std::max(1, std::min(launches, 5));
        c.K = K;
        c.seed = seed + 50;
        c.invariants = true;
        c.reference_hash = hash496;
        c.reference = S496;
        const auto t0 = std::chrono::steady_clock::now();
        const StageResult r = run_stage(c, L, dl, d_adj);
        print_summary(c, r, ms_since(t0));
        for (int b = 0; b < 8; ++b) {
          const std::size_t bb = static_cast<std::size_t>(b);
          if (r.best[bb] < 496) fail("plateau chain " + std::to_string(b) + ": best < 496 after a plateau move");
          if (b < 6 && r.hashes[bb] == hash496 && r.best[bb] == 496)
            std::printf("plateau  : note: chain %d walked back to the original 496\n", b);
        }
        summary["plateau"] = "diff=" + std::to_string(r.differs_from_reference) + "/8";
        ++stages;
      }
    }

    // ---- throughput ------------------------------------------------------------------
    std::string bench_str;
    if (bench && want("bench")) {
      for (int Bb : {512, 2048, 4096}) {
        std::size_t frb = 0;
        kiss::cuda::device_mem_info(&frb, nullptr);
        const std::size_t need = (kiss::cuda::lsa_report_memory(LSParams{}).per_chain + kiss::cuda::lss_bytes_per_chain(LSSearchParams{})) *
                                     static_cast<std::size_t>(Bb) + (std::size_t(512) << 20);
        if (frb < need) {
          std::printf("bench    : B=%d skipped (%s free, need %s)\n", Bb, mb(frb).c_str(), mb(need).c_str());
          continue;
        }
        StageCfg c;
        c.name = "bench" + std::to_string(Bb);
        c.p.B = Bb;
        std::mt19937_64 r2(seed + 60);
        for (int b = 0; b < Bb; ++b) c.seeds.push_back(subset_of(L, S496, r2, 440 + static_cast<int>(r2() % 50), false));
        c.target = 497;
        c.launches = 4;   // first launch = warm-up, all counted in gpu time though
        c.K = 50;
        c.seed = seed + 61;
        c.report_at[0] = c.report_at[1] = c.report_at[2] = -1;
        const auto t0 = std::chrono::steady_clock::now();
        const StageResult r = run_stage(c, L, dl, d_adj);
        print_summary(c, r, ms_since(t0));
        char buf[128];
        std::snprintf(buf, sizeof buf, "it_s_B%d=%.4g ", Bb, r.iterations / (r.gpu_ms / 1000.0));
        bench_str += buf;
      }
    }

    kiss::cuda::adjacency_device_free(d_adj);
    t_total = ms_since(t_start) / 1000.0;
    std::printf("RESULT ok=%d stages=%d failures=%d record_max=%u records=%d inv{%s} inv_anti{%s} refill=%s refill_a=%s "
                "climb{%s} s488{%s} s488_a{%s} s496{%s} s496_a{%s} plateau{%s} %stotal_s=%.0f\n",
                g_failures == 0 ? 1 : 0, stages, g_failures, g_record_max, g_records, summary["inv"].c_str(),
                summary["inv-anti"].c_str(), summary["refill"].c_str(), summary["refill-a"].c_str(), summary["climb"].c_str(),
                summary["s488"].c_str(), summary["s488-a"].c_str(), summary["s496"].c_str(), summary["s496-a"].c_str(),
                summary["plateau"].c_str(), bench_str.c_str(), t_total);
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "test_ls_search: exception: %s\n", ex.what());
    std::printf("RESULT ok=0 exception=1\n");
    return 1;
  }
}

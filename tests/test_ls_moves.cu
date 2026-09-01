// T3.2a — acceptance test for the chain state and the incremental move kernels
// (cuda/ls_state.cuh, cuda/ls_moves.cuh, cuda/ls_moves.cu).
//
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) without a CUDA device, without
// data/adj.u32, or with < 5.5 GB of free device memory (after retrying for a
// while: the GPU is shared), 1 on failure. Last line: RESULT ok=<0|1> ...
//
// Stages
//   1. init: B=64 chains (32 random subsets of the 496 with 300..496 members,
//      32 random-greedy maximal sets) in a non-antipodal and an antipodal state;
//      lsa_check must report 0 errors and the downloaded state must equal a CPU
//      model built from the mmap adjacency.
//   2. fuzz: `rounds` x `moves` scripted moves per chain (add a free vertex,
//      remove a member, force-add a vertex of tightness <= 8 — occasionally up
//      to 40 to exercise the second register slot / the overflow fallback —,
//      pop-add — every force-add is followed by a burst of 2..9 pop-adds, the
//      perturb-then-drain pattern of T3.2b, which keeps sizes near the seeds —,
//      tick, best, is_tabu) applied by lsk_apply_moves and by a CPU
//      model that mirrors the device semantics exactly (list order, free-list
//      push order/overflow, tabu ring). After every round tight (all N), the S
//      list, inS, the free-list, the tabu ring, iter/best and every move's
//      return value are compared; the free-list is also checked against the true
//      free set (stale entries counted, never a non-free entry that was pushed
//      while free). Three states: non-antipodal (force_cap 5, so the fallback
//      path runs constantly), antipodal (force_cap 64), and a small-FL state
//      (FL = 64) that overflows and is rescanned every round.
//   3. rescan + drain: 64 chains from random 400-subsets of the 496; report how
//      many refill to 496.
//   4. throughput: B=2048 chains x 1000 add/remove moves (and 100 force_adds),
//      moves/s and achieved adjacency bandwidth.
//   5. (run by hand / the report) compute-sanitizer memcheck + racecheck with
//      --small.
//
// Usage: test_ls_moves [--data DIR] [--rounds R] [--moves M] [--bench-B B]
//                      [--no-bench] [--small] [--seed S]
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "kiss_cuda.h"
#include "ls_state.cuh"

using kiss::Adjacency;
using kiss::DEG;
using kiss::Leech;
using kiss::N;
using kiss::Vec;
using kiss::cuda::INS_WORDS;
using kiss::cuda::LS_NONE;
using kiss::cuda::LSMove;
using kiss::cuda::LSParams;
using kiss::cuda::LSState;
using kiss::cuda::SMAX;

namespace {

constexpr int kSkip = 77;
int g_failures = 0;
int g_fail_printed = 0;

void fail(const std::string& msg) {
  ++g_failures;
  if (g_fail_printed++ < 200) std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

// ---------------------------------------------------------------------------
// CPU model of one chain. Mirrors ls_moves.cuh exactly (see the header's
// "Semantics" block): list order, free-list push order, overflow, tabu ring.
// ---------------------------------------------------------------------------
struct RefChain {
  const Leech* L = nullptr;
  const Adjacency* adj = nullptr;
  int FL = 4096, TABU = 32, force_cap = 64;
  bool antipodal = false;

  std::vector<uint16_t> tight;
  std::vector<uint32_t> inS;   // INS_WORDS
  std::vector<uint32_t> S;     // ordered like the device list
  std::vector<uint32_t> fl;    // head entries
  bool fl_overflow = false;
  std::vector<uint32_t> tabu_v, tabu_exp;
  uint32_t tabu_head = 0, iter = 0, best_size = 0;
  std::vector<uint32_t> best_S;

  void init(const Leech& l, const Adjacency& a, const LSParams& p, const std::vector<uint32_t>& set) {
    L = &l;
    adj = &a;
    FL = p.FL;
    TABU = p.TABU;
    force_cap = p.force_cap;
    antipodal = p.antipodal;
    tight.assign(static_cast<std::size_t>(N), 0);
    inS.assign(static_cast<std::size_t>(INS_WORDS), 0u);
    S.clear();
    for (uint32_t v : set) {
      inS[v >> 5] |= 1u << (v & 31);
      S.push_back(v);
      const uint32_t* row = adj->row(v);
      for (int i = 0; i < DEG; ++i) ++tight[row[i]];
    }
    tabu_v.assign(static_cast<std::size_t>(TABU), LS_NONE);
    tabu_exp.assign(static_cast<std::size_t>(TABU), 0u);
    tabu_head = 0;
    iter = 0;
    best_size = static_cast<uint32_t>(S.size());
    best_S = S;
    rescan();
  }
  bool in_S(uint32_t v) const { return (inS[v >> 5] >> (v & 31)) & 1u; }
  bool is_free(uint32_t v) const { return tight[v] == 0 && !in_S(v); }
  bool free_pair(uint32_t v) const { return is_free(v) && (!antipodal || is_free(L->neg[v])); }
  bool adjacent(uint32_t u, uint32_t w) const { return kiss::dot(L->C[u], L->C[w]) == 16; }

  void push1(uint32_t v) {
    if (fl.size() < static_cast<std::size_t>(FL)) fl.push_back(v);
    else fl_overflow = true;
  }
  void tabu_push(uint32_t v, uint32_t expiry) {
    const uint32_t slot = tabu_head % static_cast<uint32_t>(TABU);
    tabu_v[slot] = v;
    tabu_exp[slot] = expiry;
    ++tabu_head;
  }
  int add1(uint32_t v) {
    if (in_S(v)) return 0;
    if (S.size() >= static_cast<std::size_t>(SMAX)) return 0;
    inS[v >> 5] |= 1u << (v & 31);
    S.push_back(v);
    const uint32_t* row = adj->row(v);
    for (int i = 0; i < DEG; ++i) ++tight[row[i]];
    return 1;
  }
  int remove1(uint32_t v, uint32_t tenure) {
    if (!in_S(v)) return 0;
    const auto it = std::find(S.begin(), S.end(), v);
    if (it == S.end()) return 0;
    inS[v >> 5] &= ~(1u << (v & 31));
    *it = S.back();
    S.pop_back();
    const uint32_t* row = adj->row(v);
    std::size_t head = fl.size();
    for (int i = 0; i < DEG; ++i) {
      const uint32_t n = row[i];
      const uint16_t nt = static_cast<uint16_t>(tight[n] - 1);
      tight[n] = nt;
      if (nt == 0 && !in_S(n)) {
        if (head < static_cast<std::size_t>(FL)) fl.push_back(n);
        ++head;
      }
    }
    if (head > static_cast<std::size_t>(FL)) fl_overflow = true;
    if (tight[v] == 0) push1(v);
    tabu_push(v, iter + tenure);
    return 1;
  }
  int add(uint32_t v) {
    int n = add1(v);
    if (antipodal) n += add1(L->neg[v]);
    return n;
  }
  int remove(uint32_t v, uint32_t tenure) {
    int n = remove1(v, tenure);
    if (antipodal) n += remove1(L->neg[v], tenure);
    return n;
  }
  bool conflicts(uint32_t v, uint32_t nv, uint32_t s) const {
    return adjacent(v, s) || (antipodal && adjacent(nv, s));
  }
  int force_add(uint32_t v, uint32_t tenure) {
    if (in_S(v)) return -1;
    const uint32_t nv = antipodal ? L->neg[v] : v;
    std::vector<uint32_t> list;
    for (uint32_t s : S)
      if (conflicts(v, nv, s)) list.push_back(s);
    const int cnt = static_cast<int>(list.size());
    const int cap = std::min(force_cap, 64);
    const int listed = std::min(cnt, cap);
    int removed = 0;
    for (int k = 0; k < listed; ++k) removed += remove(list[static_cast<std::size_t>(k)], tenure);
    if (cnt > cap) {
      for (;;) {
        uint32_t found = LS_NONE;
        for (uint32_t s : S)
          if (conflicts(v, nv, s)) { found = s; break; }
        if (found == LS_NONE) break;
        removed += remove(found, tenure);
      }
    }
    add(v);
    return removed;
  }
  uint32_t pop() {
    uint32_t out = LS_NONE;
    while (!fl.empty()) {
      const uint32_t v = fl.back();
      fl.pop_back();
      if (free_pair(v)) { out = v; break; }
    }
    return out;
  }
  bool is_tabu(uint32_t v, uint32_t it) const {
    const uint32_t nv = antipodal ? L->neg[v] : v;
    for (int i = 0; i < TABU; ++i)
      if ((tabu_v[static_cast<std::size_t>(i)] == v || tabu_v[static_cast<std::size_t>(i)] == nv) &&
          tabu_exp[static_cast<std::size_t>(i)] > it)
        return true;
    return false;
  }
  bool update_best() {
    if (S.size() <= best_size) return false;
    best_S = S;
    best_size = static_cast<uint32_t>(S.size());
    return true;
  }
  void rescan() {
    fl.clear();
    std::size_t cnt = 0;
    for (int v = 0; v < N; ++v)
      if (free_pair(static_cast<uint32_t>(v))) {
        if (cnt < static_cast<std::size_t>(FL)) fl.push_back(static_cast<uint32_t>(v));
        ++cnt;
      }
    fl_overflow = cnt > static_cast<std::size_t>(FL);
  }
  int32_t apply(const LSMove& m, uint32_t tenure) {
    switch (m.op) {
      case kiss::cuda::LS_OP_ADD: return add(m.v);
      case kiss::cuda::LS_OP_REMOVE: return remove(m.v, tenure);
      case kiss::cuda::LS_OP_FORCE_ADD: return force_add(m.v, tenure);
      case kiss::cuda::LS_OP_POP_ADD: {
        const uint32_t v = pop();
        if (v != LS_NONE) add(v);
        return static_cast<int32_t>(v);
      }
      case kiss::cuda::LS_OP_TICK: iter += m.v; return 0;
      case kiss::cuda::LS_OP_BEST: return update_best() ? 1 : 0;
      case kiss::cuda::LS_OP_IS_TABU: return is_tabu(m.v, iter) ? 1 : 0;
      default: return 0;
    }
  }
};

// ---------------------------------------------------------------------------
// bulk download of a whole state (few large copies instead of lsa_download's
// per-chain ones; lsa_download itself is exercised in stage 1)
// ---------------------------------------------------------------------------
struct HostState {
  int B = 0, FL = 0, TABU = 0;
  std::vector<uint16_t> tight;
  std::vector<uint32_t> inS, S, size, fl, fl_head, tabu_v, tabu_exp, tabu_head, iter, best_size, best_S;
  std::vector<uint8_t> fl_overflow;
};
template <class T>
void d2h(std::vector<T>& out, const T* src, std::size_t count) {
  out.resize(count);
  if (count) KISS_CUDA_CHECK(cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost));
}
HostState download_all(const LSState& st) {
  HostState h;
  const std::size_t B = static_cast<std::size_t>(st.p.B);
  h.B = st.p.B;
  h.FL = st.p.FL;
  h.TABU = st.p.TABU;
  d2h(h.tight, st.tight, B * N);
  d2h(h.inS, st.inS, B * INS_WORDS);
  d2h(h.S, st.S, B * SMAX);
  d2h(h.size, st.size, B);
  d2h(h.fl, st.fl, B * static_cast<std::size_t>(st.p.FL));
  d2h(h.fl_head, st.fl_head, B);
  d2h(h.fl_overflow, st.fl_overflow, B);
  d2h(h.tabu_v, st.tabu_v, B * static_cast<std::size_t>(st.p.TABU));
  d2h(h.tabu_exp, st.tabu_exp, B * static_cast<std::size_t>(st.p.TABU));
  d2h(h.tabu_head, st.tabu_head, B);
  d2h(h.iter, st.iter, B);
  d2h(h.best_size, st.best_size, B);
  d2h(h.best_S, st.best_S, B * SMAX);
  return h;
}

struct CompareStats {
  long long tight_mismatch = 0, list_mismatch = 0, bitmap_mismatch = 0, fl_mismatch = 0, tabu_mismatch = 0,
            misc_mismatch = 0, fl_stale = 0, fl_entries = 0, fl_nonfree_bad = 0;
};

// Compare chain b of the downloaded state with its model. Returns #mismatching fields.
long long compare_chain(const HostState& h, int b, const RefChain& r, const std::string& tag, CompareStats& cs) {
  const std::size_t bb = static_cast<std::size_t>(b);
  long long bad = 0;
  // tight
  {
    long long m = 0;
    int first = -1;
    const uint16_t* t = h.tight.data() + bb * N;
    for (int v = 0; v < N; ++v)
      if (t[v] != r.tight[static_cast<std::size_t>(v)]) { if (first < 0) first = v; ++m; }
    if (m) {
      fail(tag + " chain " + std::to_string(b) + ": tight mismatches=" + std::to_string(m) + " first v=" +
           std::to_string(first) + " gpu=" + std::to_string(t[first]) + " cpu=" +
           std::to_string(r.tight[static_cast<std::size_t>(first)]));
      cs.tight_mismatch += m;
      ++bad;
    }
  }
  // S list (exact order)
  {
    const uint32_t sz = h.size[bb];
    bool ok = sz == r.S.size();
    if (ok)
      for (uint32_t i = 0; i < sz; ++i)
        if (h.S[bb * SMAX + i] != r.S[i]) { ok = false; break; }
    if (!ok) {
      fail(tag + " chain " + std::to_string(b) + ": S list differs (gpu size " + std::to_string(sz) + ", cpu " +
           std::to_string(r.S.size()) + ")");
      ++cs.list_mismatch;
      ++bad;
    }
  }
  // bitmap
  if (std::memcmp(h.inS.data() + bb * INS_WORDS, r.inS.data(), INS_WORDS * sizeof(uint32_t)) != 0) {
    fail(tag + " chain " + std::to_string(b) + ": inS bitmap differs");
    ++cs.bitmap_mismatch;
    ++bad;
  }
  // free-list: exact vs model, and vs the true free set
  {
    const uint32_t head = h.fl_head[bb];
    bool ok = head == r.fl.size() && (h.fl_overflow[bb] != 0) == r.fl_overflow;
    if (ok)
      for (uint32_t i = 0; i < head; ++i)
        if (h.fl[bb * static_cast<std::size_t>(h.FL) + i] != r.fl[i]) { ok = false; break; }
    if (!ok) {
      fail(tag + " chain " + std::to_string(b) + ": free-list differs (gpu head " + std::to_string(head) +
           " overflow " + std::to_string(h.fl_overflow[bb]) + ", cpu " + std::to_string(r.fl.size()) + " overflow " +
           std::to_string(r.fl_overflow) + ")");
      ++cs.fl_mismatch;
      ++bad;
    }
    for (uint32_t i = 0; i < std::min<uint32_t>(head, static_cast<uint32_t>(h.FL)); ++i) {
      const uint32_t v = h.fl[bb * static_cast<std::size_t>(h.FL) + i];
      ++cs.fl_entries;
      if (v >= static_cast<uint32_t>(N)) { ++cs.fl_nonfree_bad; continue; }
      // Every entry was free when pushed (the exact match with the model above proves it); entries that
      // are no longer free are "stale" (validate-on-pop discards them) and only counted here.
      if (!r.is_free(v)) ++cs.fl_stale;
    }
  }
  // tabu ring
  {
    bool ok = h.tabu_head[bb] == r.tabu_head;
    for (int i = 0; ok && i < h.TABU; ++i) {
      const std::size_t k = bb * static_cast<std::size_t>(h.TABU) + static_cast<std::size_t>(i);
      if (h.tabu_v[k] != r.tabu_v[static_cast<std::size_t>(i)] || h.tabu_exp[k] != r.tabu_exp[static_cast<std::size_t>(i)])
        ok = false;
    }
    if (!ok) {
      fail(tag + " chain " + std::to_string(b) + ": tabu ring differs");
      ++cs.tabu_mismatch;
      ++bad;
    }
  }
  // iter, best
  {
    bool ok = h.iter[bb] == r.iter && h.best_size[bb] == r.best_size;
    for (uint32_t i = 0; ok && i < r.best_size; ++i)
      if (h.best_S[bb * SMAX + i] != r.best_S[i]) ok = false;
    if (!ok) {
      fail(tag + " chain " + std::to_string(b) + ": iter/best differ");
      ++cs.misc_mismatch;
      ++bad;
    }
  }
  return bad;
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------
std::vector<uint32_t> load_s496(const Leech& L, const std::filesystem::path& p) {
  std::vector<uint32_t> S;
  for (const Vec& r : kiss::read_set(p)) {
    const int32_t idx = L.index_of(r);
    if (idx < 0) throw std::runtime_error(p.string() + ": row is not a minimal vector");
    S.push_back(static_cast<uint32_t>(idx));
  }
  return S;
}

// Random subset of the 496 with `size` members; antipodal: `size`/2 antipodal pairs.
std::vector<uint32_t> subset_496(const Leech& L, const std::vector<uint32_t>& S496, std::mt19937_64& rng, int size,
                                 bool antipodal) {
  std::vector<uint32_t> out;
  if (!antipodal) {
    std::vector<uint32_t> s = S496;
    std::shuffle(s.begin(), s.end(), rng);
    s.resize(static_cast<std::size_t>(size));
    return s;
  }
  std::vector<uint32_t> reps;
  for (uint32_t v : S496)
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
// fuzz script generation from the model state (per chain, own RNG)
// ---------------------------------------------------------------------------
struct ScriptGen {
  std::mt19937_64 rng;
  int pending_pops = 0;   // drain burst after a force_add (the T3.2b perturb-then-drain pattern)
  explicit ScriptGen(uint64_t seed) : rng(seed) {}
  uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng() % n); }

  // pick a free vertex (pair-free in antipodal mode); LS_NONE if none found cheaply
  uint32_t pick_free(const RefChain& r) {
    if (!r.fl.empty())
      for (int t = 0; t < 30; ++t) {
        const uint32_t v = r.fl[rnd(static_cast<uint32_t>(r.fl.size()))];
        if (r.free_pair(v)) return v;
      }
    for (int t = 0; t < 200; ++t) {
      const uint32_t v = rnd(static_cast<uint32_t>(N));
      if (r.free_pair(v)) return v;
    }
    return LS_NONE;
  }
  uint32_t pick_low_tight(const RefChain& r, int max_t) {
    for (int t = 0; t < 2000; ++t) {
      const uint32_t v = rnd(static_cast<uint32_t>(N));
      if (!r.in_S(v) && r.tight[v] <= max_t && (!r.antipodal || !r.in_S(r.L->neg[v]))) return v;
    }
    return LS_NONE;
  }
  // Generates one move consistent with the model, applies it to the model, returns it + the model's return value.
  LSMove next(RefChain& r, uint32_t tenure, int32_t* ret) {
    LSMove m{kiss::cuda::LS_OP_NOP, 0};
    if (pending_pops > 0) {
      --pending_pops;
      m = {kiss::cuda::LS_OP_POP_ADD, 0};
      *ret = r.apply(m, tenure);
      return m;
    }
    const uint32_t k = rnd(32);
    if (k < 9) {                       // add a free vertex
      const uint32_t v = pick_free(r);
      if (v != LS_NONE) m = {kiss::cuda::LS_OP_ADD, v};
      else m = {kiss::cuda::LS_OP_POP_ADD, 0};
    } else if (k < 18) {               // remove a member
      if (!r.S.empty()) m = {kiss::cuda::LS_OP_REMOVE, r.S[rnd(static_cast<uint32_t>(r.S.size()))]};
    } else if (k < 26) {               // force-add a low-tightness vertex
      const int max_t = (rnd(16) == 0) ? 40 : 8;
      const uint32_t v = pick_low_tight(r, max_t);
      if (v != LS_NONE) { m = {kiss::cuda::LS_OP_FORCE_ADD, v}; pending_pops = 2 + static_cast<int>(rnd(8)); }
      else if (!r.S.empty()) m = {kiss::cuda::LS_OP_REMOVE, r.S[rnd(static_cast<uint32_t>(r.S.size()))]};
    } else if (k < 28) {
      m = {kiss::cuda::LS_OP_POP_ADD, 0};
    } else if (k < 30) {
      m = {kiss::cuda::LS_OP_TICK, 1u + rnd(3)};
    } else if (k < 31) {
      m = {kiss::cuda::LS_OP_BEST, 0};
    } else {                           // is_tabu: a ring entry half of the time (true hits), else random
      uint32_t v = r.tabu_v[rnd(static_cast<uint32_t>(r.TABU))];
      if (v == LS_NONE || rnd(2) == 0) v = rnd(static_cast<uint32_t>(N));
      if (r.antipodal && rnd(2) == 0) v = r.L->neg[v];
      m = {kiss::cuda::LS_OP_IS_TABU, v};
    }
    *ret = r.apply(m, tenure);
    return m;
  }
};

struct FuzzConfig {
  std::string name;
  LSParams p;
  int rounds = 0, moves = 0;
  bool rescan_every_round = false;
};

struct FuzzResult {
  long long moves = 0, bad = 0;
  CompareStats cs;
  std::map<uint32_t, long long> op_hist;
  long long force_removed = 0, force_calls = 0, force_over_cap = 0, pop_hits = 0, tabu_hits = 0, tabu_queries = 0;
  double gpu_ms = 0, cpu_ms = 0;
  uint32_t min_size = 0xffffffffu, max_size = 0;
  int overflow_events = 0;
};

FuzzResult run_fuzz(const FuzzConfig& cfg, const Leech& L, const Adjacency& adj, const kiss::cuda::DeviceLeech& dl,
                    const uint32_t* d_adj, const std::vector<std::vector<uint32_t>>& sets, uint64_t seed,
                    int check_every) {
  const int B = cfg.p.B, M = cfg.moves;
  FuzzResult fr;
  LSState st = kiss::cuda::lsa_alloc(cfg.p);
  std::vector<RefChain> ref(static_cast<std::size_t>(B));
  std::vector<ScriptGen> gen;
  for (int b = 0; b < B; ++b) gen.emplace_back(seed * 1000003ull + static_cast<uint64_t>(b) * 7919ull + 1);
  try {
    kiss::cuda::lsa_init_from_sets(st, sets, dl, d_adj, seed);
    for (int b = 0; b < B; ++b) ref[static_cast<std::size_t>(b)].init(L, adj, cfg.p, sets[static_cast<std::size_t>(b)]);
    {
      const HostState h = download_all(st);
      for (int b = 0; b < B; ++b) fr.bad += compare_chain(h, b, ref[static_cast<std::size_t>(b)], cfg.name + " init", fr.cs);
      const auto chk = kiss::cuda::lsa_check(st, dl);
      for (int b = 0; b < B; ++b)
        if (chk[static_cast<std::size_t>(b)].errors()) {
          fail(cfg.name + " init chain " + std::to_string(b) + ": lsa_check errors=" +
               std::to_string(chk[static_cast<std::size_t>(b)].errors()));
          ++fr.bad;
        }
    }
    LSMove* d_moves = nullptr;
    int32_t* d_ret = nullptr;
    KISS_CUDA_CHECK(cudaMalloc(&d_moves, static_cast<std::size_t>(B) * M * sizeof(LSMove)));
    KISS_CUDA_CHECK(cudaMalloc(&d_ret, static_cast<std::size_t>(B) * M * sizeof(int32_t)));
    std::vector<LSMove> hm(static_cast<std::size_t>(B) * M);
    std::vector<int32_t> href(static_cast<std::size_t>(B) * M), hret(static_cast<std::size_t>(B) * M);
    cudaEvent_t e0, e1;
    KISS_CUDA_CHECK(cudaEventCreate(&e0));
    KISS_CUDA_CHECK(cudaEventCreate(&e1));
    const uint32_t tenure = 7;
    for (int round = 0; round < cfg.rounds; ++round) {
      const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic)
      for (int b = 0; b < B; ++b) {
        RefChain& r = ref[static_cast<std::size_t>(b)];
        for (int i = 0; i < M; ++i) {
          int32_t rv = 0;
          const LSMove m = gen[static_cast<std::size_t>(b)].next(r, tenure, &rv);
          hm[static_cast<std::size_t>(b) * M + i] = m;
          href[static_cast<std::size_t>(b) * M + i] = rv;
        }
      }
      fr.cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      KISS_CUDA_CHECK(cudaMemcpy(d_moves, hm.data(), hm.size() * sizeof(LSMove), cudaMemcpyHostToDevice));
      KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
      kiss::cuda::lsk_apply_moves(st, d_moves, M, d_adj, dl, tenure, d_ret, 0);
      KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
      KISS_CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0;
      KISS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      fr.gpu_ms += ms;
      KISS_CUDA_CHECK(cudaMemcpy(hret.data(), d_ret, hret.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
      for (int b = 0; b < B; ++b) {
        RefChain& r = ref[static_cast<std::size_t>(b)];
        fr.min_size = std::min<uint32_t>(fr.min_size, static_cast<uint32_t>(r.S.size()));
        fr.max_size = std::max<uint32_t>(fr.max_size, static_cast<uint32_t>(r.S.size()));
        for (int i = 0; i < M; ++i) {
          const std::size_t k = static_cast<std::size_t>(b) * M + i;
          const LSMove& m = hm[k];
          ++fr.op_hist[m.op];
          ++fr.moves;
          if (m.op == kiss::cuda::LS_OP_FORCE_ADD) {
            ++fr.force_calls;
            fr.force_removed += href[k] > 0 ? href[k] : 0;
            if (href[k] > cfg.p.force_cap) ++fr.force_over_cap;
          }
          if (m.op == kiss::cuda::LS_OP_POP_ADD && href[k] >= 0) ++fr.pop_hits;
          if (m.op == kiss::cuda::LS_OP_IS_TABU) { ++fr.tabu_queries; fr.tabu_hits += href[k]; }
          if (hret[k] != href[k]) {
            fail(cfg.name + " round " + std::to_string(round) + " chain " + std::to_string(b) + " move " +
                 std::to_string(i) + " op " + std::to_string(m.op) + " v " + std::to_string(m.v) + ": ret gpu=" +
                 std::to_string(hret[k]) + " cpu=" + std::to_string(href[k]));
            ++fr.bad;
          }
        }
      }
      // rescan (all chains, both models) when configured or when any chain overflowed
      bool do_rescan = cfg.rescan_every_round || (round % 50 == 49);
      if (!do_rescan) {
        std::vector<uint8_t> of;
        d2h(of, st.fl_overflow, static_cast<std::size_t>(B));
        for (int b = 0; b < B; ++b) if (of[static_cast<std::size_t>(b)]) { do_rescan = true; break; }
      }
      if (do_rescan) {
        ++fr.overflow_events;
        kiss::cuda::lsk_rescan_free(st, dl);
        KISS_CUDA_CHECK(cudaDeviceSynchronize());
#pragma omp parallel for schedule(static)
        for (int b = 0; b < B; ++b) ref[static_cast<std::size_t>(b)].rescan();
      }
      if ((round + 1) % check_every == 0 || round + 1 == cfg.rounds) {
        const HostState h = download_all(st);
        for (int b = 0; b < B; ++b)
          fr.bad += compare_chain(h, b, ref[static_cast<std::size_t>(b)], cfg.name + " round " + std::to_string(round), fr.cs);
      }
      if ((round + 1) % 100 == 0 || round + 1 == cfg.rounds) {
        const auto chk = kiss::cuda::lsa_check(st, dl);
        for (int b = 0; b < B; ++b)
          if (chk[static_cast<std::size_t>(b)].errors()) {
            fail(cfg.name + " round " + std::to_string(round) + " chain " + std::to_string(b) +
                 ": lsa_check errors=" + std::to_string(chk[static_cast<std::size_t>(b)].errors()) + " (tight " +
                 std::to_string(chk[static_cast<std::size_t>(b)].tight_mismatch) + ", conflicts " +
                 std::to_string(chk[static_cast<std::size_t>(b)].conflict_pairs) + ", antipodal " +
                 std::to_string(chk[static_cast<std::size_t>(b)].antipodal_bad) + ")");
            ++fr.bad;
          }
      }
      if (g_failures > 50) break;   // no point in continuing
    }
    KISS_CUDA_CHECK(cudaEventDestroy(e0));
    KISS_CUDA_CHECK(cudaEventDestroy(e1));
    KISS_CUDA_CHECK(cudaFree(d_moves));
    KISS_CUDA_CHECK(cudaFree(d_ret));
  } catch (...) {
    kiss::cuda::lsa_free(st);
    throw;
  }
  kiss::cuda::lsa_free(st);
  return fr;
}

std::string mb(std::size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.1f MB", static_cast<double>(bytes) / 1048576.0);
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "data";
  int rounds = 1000, moves = 100, bench_B = 2048, check_every = 1;
  bool bench = true, small = false;
  uint64_t seed = 20260826ull;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--moves") && i + 1 < argc) moves = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bench-B") && i + 1 < argc) bench_B = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--check-every") && i + 1 < argc) check_every = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--no-bench")) bench = false;
    else if (!std::strcmp(argv[i], "--small")) small = true;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  if (small) { rounds = std::min(rounds, 5); moves = std::min(moves, 20); bench = false; }
  const int B = small ? 4 : 64;

  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::printf("test_ls_moves: no CUDA device (%s) — skipping\n", cudaGetErrorString(e));
    std::printf("RESULT ok=1 skipped=1 devices=0\n");
    return kSkip;
  }
  const std::filesystem::path adj_path = data_dir / "adj.u32";
  if (!std::filesystem::exists(adj_path)) {
    std::printf("test_ls_moves: %s missing (run tools/build_adj) — skipping\n", adj_path.c_str());
    std::printf("RESULT ok=1 skipped=1 reason=no_adjacency\n");
    return kSkip;
  }
  {
    // the GPU is shared: the adjacency needs 3.62 GB; wait a while for others' short jobs
    const std::size_t need = static_cast<std::size_t>(5.5 * 1024 * 1024 * 1024);
    std::size_t fr = 0, to = 0;
    for (int attempt = 0; attempt < 8; ++attempt) {
      kiss::cuda::device_mem_info(&fr, &to);
      if (fr >= need) break;
      std::printf("test_ls_moves: only %s of %s device memory free (need 5.5 GB) — retry %d/8 in 15 s\n",
                  mb(fr).c_str(), mb(to).c_str(), attempt + 1);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(15));
    }
    if (fr < need) {
      std::printf("test_ls_moves: %s free of %s — skipping (another process holds the GPU)\n", mb(fr).c_str(),
                  mb(to).c_str());
      std::printf("RESULT ok=1 skipped=1 reason=vram free=%zu\n", fr);
      return kSkip;
    }
  }

  try {
    const auto t_start = std::chrono::steady_clock::now();
    const Leech L = kiss::load_leech(data_dir);
    const Adjacency adj(adj_path);
    const kiss::cuda::ScopedDeviceLeech dl(L);
    const std::vector<uint32_t> S496 = load_s496(L, data_dir / "S496.txt");
    std::printf("fixture         : S496 |S|=%zu, adjacency %s\n", S496.size(), adj_path.c_str());

    std::size_t free0 = 0, total0 = 0;
    kiss::cuda::device_mem_info(&free0, &total0);
    const auto ta = std::chrono::steady_clock::now();
    uint32_t* d_adj = kiss::cuda::adjacency_to_device(adj);
    const double adj_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
    std::size_t free1 = 0;
    kiss::cuda::device_mem_info(&free1, nullptr);
    std::printf("adjacency       : uploaded in %.0f ms; device free %s -> %s\n", adj_ms, mb(free0).c_str(), mb(free1).c_str());

    // ---- memory report -------------------------------------------------------
    LSParams pm;
    pm.B = 1;
    const kiss::cuda::LSMemory mem = kiss::cuda::lsa_report_memory(pm);
    const int maxB = mem.max_B(free1, std::size_t(256) << 20);
    std::printf("memory          : %zu bytes per chain (%.1f KB); free after adjacency %s; max B (256 MB margin) = %d; "
                "B=2048 -> %s, B=4096 -> %s\n",
                mem.per_chain, mem.per_chain / 1024.0, mb(free1).c_str(), maxB, mb(mem.per_chain * 2048).c_str(),
                mb(mem.per_chain * 4096).c_str());

    std::mt19937_64 rng(seed);

    // ---- stage 1 + 2: fuzz on three states ------------------------------------
    std::vector<FuzzConfig> cfgs;
    {
      FuzzConfig c;
      c.name = "plain";
      c.p.B = B; c.p.antipodal = false; c.p.force_cap = 5;
      c.rounds = rounds; c.moves = moves;
      cfgs.push_back(c);
      c.name = "antipodal";
      c.p.antipodal = true; c.p.force_cap = 64;
      cfgs.push_back(c);
      c.name = "smallFL";
      c.p.antipodal = false; c.p.force_cap = 64; c.p.FL = 64;
      c.rounds = std::max(1, rounds / 10); c.rescan_every_round = true;
      cfgs.push_back(c);
    }
    long long fuzz_bad = 0, fuzz_moves = 0, fuzz_stale = 0, fuzz_entries = 0;
    double fuzz_gpu_ms = 0, fuzz_cpu_ms = 0;
    for (const FuzzConfig& c : cfgs) {
      // seeds: half subsets of the 496 (300..496), half greedy sets
      std::vector<std::vector<uint32_t>> sets;
      for (int b = 0; b < c.p.B; ++b) {
        if (b < c.p.B / 2 || c.p.B == 1) {
          const int size = 300 + static_cast<int>(rng() % 197);
          sets.push_back(subset_496(L, S496, rng, size, c.p.antipodal));
        } else {
          sets.push_back(greedy_set(L, adj, rng, c.p.antipodal));
        }
      }
      uint32_t smin = 0xffffffffu, smax = 0;
      for (const auto& s : sets) { smin = std::min<uint32_t>(smin, static_cast<uint32_t>(s.size())); smax = std::max<uint32_t>(smax, static_cast<uint32_t>(s.size())); }
      std::printf("fuzz %-9s : B=%d rounds=%d moves=%d FL=%d force_cap=%d antipodal=%d seed sizes %u..%u\n",
                  c.name.c_str(), c.p.B, c.rounds, c.moves, c.p.FL, c.p.force_cap, c.p.antipodal ? 1 : 0, smin, smax);
      std::fflush(stdout);
      const FuzzResult fr = run_fuzz(c, L, adj, dl, d_adj, sets, seed + static_cast<uint64_t>(cfgs.size()), check_every);
      std::printf("fuzz %-9s : moves=%lld mismatching_fields=%lld ops{add=%lld rem=%lld force=%lld pop=%lld tick=%lld "
                  "best=%lld tabu=%lld} force: calls=%lld removed=%lld over_cap=%lld; pop hits=%lld; tabu hits=%lld/%lld; "
                  "sizes %u..%u; fl entries=%lld stale=%lld; rescans=%d; gpu %.0f ms cpu %.0f ms\n",
                  c.name.c_str(), fr.moves, fr.bad, fr.op_hist.count(1) ? fr.op_hist.at(1) : 0,
                  fr.op_hist.count(2) ? fr.op_hist.at(2) : 0, fr.op_hist.count(3) ? fr.op_hist.at(3) : 0,
                  fr.op_hist.count(4) ? fr.op_hist.at(4) : 0, fr.op_hist.count(5) ? fr.op_hist.at(5) : 0,
                  fr.op_hist.count(6) ? fr.op_hist.at(6) : 0, fr.op_hist.count(7) ? fr.op_hist.at(7) : 0,
                  fr.force_calls, fr.force_removed, fr.force_over_cap, fr.pop_hits, fr.tabu_hits, fr.tabu_queries,
                  fr.min_size, fr.max_size, fr.cs.fl_entries, fr.cs.fl_stale, fr.overflow_events, fr.gpu_ms, fr.cpu_ms);
      std::fflush(stdout);
      fuzz_bad += fr.bad;
      fuzz_moves += fr.moves;
      fuzz_stale += fr.cs.fl_stale;
      fuzz_entries += fr.cs.fl_entries;
      fuzz_gpu_ms += fr.gpu_ms;
      fuzz_cpu_ms += fr.cpu_ms;
      if (fr.cs.fl_nonfree_bad) fail(c.name + ": free-list entries >= N");
    }

    // ---- stage 1b: lsa_download on one chain (API exercised) ----------------------
    {
      LSParams p;
      p.B = 2;
      LSState st = kiss::cuda::lsa_alloc(p);
      std::vector<std::vector<uint32_t>> sets = {S496, subset_496(L, S496, rng, 400, false)};
      kiss::cuda::lsa_init_from_sets(st, sets, dl, d_adj, seed);
      const kiss::cuda::LSHostChain h0 = kiss::cuda::lsa_download(st, 0, true);
      const kiss::cuda::LSHostChain h1 = kiss::cuda::lsa_download(st, 1, false);
      std::map<int, long long> hist;
      for (int v = 0; v < N; ++v)
        if (!((h0.inS[static_cast<std::size_t>(v >> 5)] >> (v & 31)) & 1u)) ++hist[h0.tight[static_cast<std::size_t>(v)]];
      std::printf("download        : chain0 |S|=%u fl_head=%u best=%u; chain1 |S|=%u fl_head=%u; 496 tightness histogram (outside S):",
                  h0.size, h0.fl_head, h0.best_size, h1.size, h1.fl_head);
      for (const auto& kv : hist) std::printf(" %d:%lld", kv.first, kv.second);
      std::printf("\n");
      if (h0.size != 496 || h0.fl_head != 0 || h0.best_size != 496) fail("download: chain 0 state wrong");
      if (h1.size != 400 || h1.fl_head < 96) fail("download: chain 1 state wrong");
      if (hist.count(1) || hist.count(2) || hist.count(3) || hist.count(0)) fail("496 tightness histogram has 0..3");
      if (!hist.count(4) || hist.at(4) != 80) fail("496 tightness-4 count != 80");
      kiss::cuda::lsa_free(st);
    }

    // ---- stage 3: rescan + drain from 400-subsets ---------------------------------
    int refill_plain = 0, refill_anti = 0, drain_B = 0;
    {
      for (int mode = 0; mode < 2; ++mode) {
        LSParams p;
        p.B = B;
        p.antipodal = mode == 1;
        drain_B = p.B;
        std::vector<std::vector<uint32_t>> sets;
        for (int b = 0; b < p.B; ++b) sets.push_back(subset_496(L, S496, rng, 400, p.antipodal));
        LSState st = kiss::cuda::lsa_alloc(p);
        uint32_t* d_adds = nullptr;
        KISS_CUDA_CHECK(cudaMalloc(&d_adds, static_cast<std::size_t>(p.B) * sizeof(uint32_t)));
        kiss::cuda::lsa_init_from_sets(st, sets, dl, d_adj, seed);
        std::vector<uint32_t> fl_before;
        d2h(fl_before, st.fl_head, static_cast<std::size_t>(p.B));
        kiss::cuda::lsk_drain_free(st, d_adj, dl, 1024, d_adds);
        KISS_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> sizes, adds, fl_after;
        d2h(sizes, st.size, static_cast<std::size_t>(p.B));
        d2h(adds, d_adds, static_cast<std::size_t>(p.B));
        d2h(fl_after, st.fl_head, static_cast<std::size_t>(p.B));
        const auto chk = kiss::cuda::lsa_check(st, dl);
        // maximality: rescan and check the free-list is empty
        kiss::cuda::lsk_rescan_free(st, dl);
        KISS_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> fl_rescan;
        d2h(fl_rescan, st.fl_head, static_cast<std::size_t>(p.B));
        int refill = 0, maximal = 0, errors = 0;
        std::map<uint32_t, int> size_hist;
        uint32_t fl_min = 0xffffffffu, fl_max = 0;
        for (int b = 0; b < p.B; ++b) {
          const std::size_t bb = static_cast<std::size_t>(b);
          if (sizes[bb] == 496) ++refill;
          if (fl_rescan[bb] == 0) ++maximal;
          if (chk[bb].errors()) ++errors;
          ++size_hist[sizes[bb]];
          fl_min = std::min(fl_min, fl_before[bb]);
          fl_max = std::max(fl_max, fl_before[bb]);
          if (sizes[bb] != 400 + adds[bb] * (p.antipodal ? 2u : 1u))
            fail("drain: size accounting wrong on chain " + std::to_string(b));
        }
        std::printf("drain %-9s : B=%d from 400-subsets: free-list %u..%u before, refilled to 496: %d/%d, maximal: %d/%d, "
                    "lsa_check errors: %d; sizes:", p.antipodal ? "antipodal" : "plain", p.B, fl_min, fl_max, refill, p.B,
                    maximal, p.B, errors);
        for (const auto& kv : size_hist) std::printf(" %u:%d", kv.first, kv.second);
        std::printf("\n");
        if (errors) fail("drain: lsa_check errors");
        if (maximal != p.B) fail("drain: some chains are not maximal after drain");
        if (refill == 0) fail("drain: no chain refilled to 496");
        (mode == 0 ? refill_plain : refill_anti) = refill;
        KISS_CUDA_CHECK(cudaFree(d_adds));
        kiss::cuda::lsa_free(st);
      }
    }

    // ---- stage 4: throughput ---------------------------------------------------------
    double moves_per_s = 0, adj_gbps = 0, total_gbps = 0, force_per_s = 0, drain_adds_per_s = 0;
    if (bench) {
      LSParams p;
      p.B = bench_B;
      const kiss::cuda::LSMemory m2 = kiss::cuda::lsa_report_memory(p);
      std::size_t frn = 0;
      kiss::cuda::device_mem_info(&frn, nullptr);
      if (m2.total + (std::size_t(256) << 20) > frn) {
        p.B = m2.max_B(frn, std::size_t(256) << 20);
        std::printf("bench           : B=%d does not fit (%s free), using B=%d\n", bench_B, mb(frn).c_str(), p.B);
      }
      const int BB = p.B, MM = 1000;
      std::vector<std::vector<uint32_t>> sets;
      for (int b = 0; b < BB; ++b) sets.push_back(subset_496(L, S496, rng, 400, false));
      LSState st = kiss::cuda::lsa_alloc(p);
      const auto ti = std::chrono::steady_clock::now();
      kiss::cuda::lsa_init_from_sets(st, sets, dl, d_adj, seed);
      const double init_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
      // script: remove a random member, re-add it (both stream one adjacency row; re-add is legal: v is free again)
      std::vector<LSMove> hm(static_cast<std::size_t>(BB) * MM);
      for (int b = 0; b < BB; ++b) {
        std::mt19937_64 r2(seed + static_cast<uint64_t>(b));
        const auto& s = sets[static_cast<std::size_t>(b)];
        for (int i = 0; i < MM; i += 2) {
          const uint32_t v = s[r2() % s.size()];
          hm[static_cast<std::size_t>(b) * MM + i] = {kiss::cuda::LS_OP_REMOVE, v};
          hm[static_cast<std::size_t>(b) * MM + i + 1] = {kiss::cuda::LS_OP_ADD, v};
        }
      }
      LSMove* d_moves = nullptr;
      KISS_CUDA_CHECK(cudaMalloc(&d_moves, hm.size() * sizeof(LSMove)));
      KISS_CUDA_CHECK(cudaMemcpy(d_moves, hm.data(), hm.size() * sizeof(LSMove), cudaMemcpyHostToDevice));
      cudaEvent_t e0, e1;
      KISS_CUDA_CHECK(cudaEventCreate(&e0));
      KISS_CUDA_CHECK(cudaEventCreate(&e1));
      float best_ms = 1e30f;
      for (int rep = 0; rep < 4; ++rep) {
        KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
        kiss::cuda::lsk_apply_moves(st, d_moves, MM, d_adj, dl, 7, nullptr, 0);
        KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
        KISS_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        KISS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (rep > 0) best_ms = std::min(best_ms, ms);
      }
      const double nm = static_cast<double>(BB) * MM;
      moves_per_s = nm / (best_ms * 1e-3);
      const double row_bytes = static_cast<double>(DEG) * 4;
      adj_gbps = moves_per_s * row_bytes / 1e9;
      total_gbps = moves_per_s * (row_bytes + 2.0 * DEG * 2) / 1e9;   // + tight read + write
      std::printf("bench add/remove: B=%d x %d moves: init %.0f ms; best %.2f ms -> %.3e moves/s; adjacency %.1f GB/s, "
                  "adjacency+tight r/w %.1f GB/s\n",
                  BB, MM, init_ms, best_ms, moves_per_s, adj_gbps, total_gbps);
      {
        const auto chk = kiss::cuda::lsa_check(st, dl);
        int errors = 0;
        for (int b = 0; b < BB; ++b) errors += chk[static_cast<std::size_t>(b)].errors() ? 1 : 0;
        if (errors) fail("bench: lsa_check errors after add/remove");
        std::printf("bench check     : lsa_check chains with errors = %d/%d\n", errors, BB);
      }
      // force_add: 100 vertices of tightness 4..8 w.r.t. the full 496 (<= 8 for any subset), then drain
      {
        std::vector<uint16_t> t496(static_cast<std::size_t>(N), 0);
        for (uint32_t s : S496) { const uint32_t* row = adj.row(s); for (int i = 0; i < DEG; ++i) ++t496[row[i]]; }
        std::vector<uint32_t> low;
        std::vector<uint8_t> in496(static_cast<std::size_t>(N), 0);
        for (uint32_t s : S496) in496[s] = 1;
        for (int v = 0; v < N; ++v) if (!in496[static_cast<std::size_t>(v)] && t496[static_cast<std::size_t>(v)] <= 8) low.push_back(static_cast<uint32_t>(v));
        const int MF = 100;
        std::vector<LSMove> hf(static_cast<std::size_t>(BB) * MF);
        for (int b = 0; b < BB; ++b) {
          std::mt19937_64 r2(seed * 3 + static_cast<uint64_t>(b));
          for (int i = 0; i < MF; ++i) hf[static_cast<std::size_t>(b) * MF + i] = {kiss::cuda::LS_OP_FORCE_ADD, low[r2() % low.size()]};
        }
        LSMove* d_f = nullptr;
        int32_t* d_ret = nullptr;
        KISS_CUDA_CHECK(cudaMalloc(&d_f, hf.size() * sizeof(LSMove)));
        KISS_CUDA_CHECK(cudaMalloc(&d_ret, hf.size() * sizeof(int32_t)));
        KISS_CUDA_CHECK(cudaMemcpy(d_f, hf.data(), hf.size() * sizeof(LSMove), cudaMemcpyHostToDevice));
        KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
        kiss::cuda::lsk_apply_moves(st, d_f, MF, d_adj, dl, 7, d_ret, 0);
        KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
        KISS_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        KISS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<int32_t> ret;
        d2h(ret, d_ret, hf.size());
        long long removed = 0, calls = 0;
        for (int32_t r : ret) if (r >= 0) { removed += r; ++calls; }
        force_per_s = static_cast<double>(BB) * MF / (ms * 1e-3);
        std::printf("bench force_add : B=%d x %d: %.2f ms -> %.3e force_adds/s, mean removed %.2f per call (%lld calls), "
                    "%.3e row-updates/s\n",
                    BB, MF, ms, force_per_s, calls ? static_cast<double>(removed) / calls : 0.0, calls,
                    (static_cast<double>(removed) + calls) / (ms * 1e-3));
        uint32_t* d_adds = nullptr;
        KISS_CUDA_CHECK(cudaMalloc(&d_adds, static_cast<std::size_t>(BB) * sizeof(uint32_t)));
        KISS_CUDA_CHECK(cudaEventRecord(e0, 0));
        kiss::cuda::lsk_drain_free(st, d_adj, dl, 1024, d_adds);
        KISS_CUDA_CHECK(cudaEventRecord(e1, 0));
        KISS_CUDA_CHECK(cudaEventSynchronize(e1));
        KISS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::vector<uint32_t> adds;
        d2h(adds, d_adds, static_cast<std::size_t>(BB));
        long long tot = 0;
        for (uint32_t a : adds) tot += a;
        drain_adds_per_s = static_cast<double>(tot) / (ms * 1e-3);
        std::printf("bench drain     : B=%d: %.2f ms, %lld adds (%.1f per chain) -> %.3e adds/s\n", BB, ms, tot,
                    static_cast<double>(tot) / BB, drain_adds_per_s);
        const auto chk = kiss::cuda::lsa_check(st, dl);
        int errors = 0;
        for (int b = 0; b < BB; ++b) errors += chk[static_cast<std::size_t>(b)].errors() ? 1 : 0;
        if (errors) fail("bench: lsa_check errors after force_add/drain");
        std::printf("bench check     : lsa_check chains with errors = %d/%d after force_add + drain\n", errors, BB);
        KISS_CUDA_CHECK(cudaFree(d_adds));
        KISS_CUDA_CHECK(cudaFree(d_f));
        KISS_CUDA_CHECK(cudaFree(d_ret));
      }
      KISS_CUDA_CHECK(cudaEventDestroy(e0));
      KISS_CUDA_CHECK(cudaEventDestroy(e1));
      KISS_CUDA_CHECK(cudaFree(d_moves));
      kiss::cuda::lsa_free(st);
    }

    kiss::cuda::adjacency_device_free(d_adj);
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    const bool ok = g_failures == 0 && fuzz_bad == 0;
    std::printf("RESULT ok=%d fuzz_moves=%lld fuzz_mismatches=%lld fl_entries=%lld fl_stale=%lld refill_plain=%d/%d "
                "refill_antipodal=%d/%d moves_per_s=%.3e adj_GBps=%.1f total_GBps=%.1f force_adds_per_s=%.3e "
                "drain_adds_per_s=%.3e bytes_per_chain=%zu max_B=%d fuzz_gpu_ms=%.0f fuzz_cpu_ms=%.0f omp_threads=%d "
                "total_s=%.0f failures=%d\n",
                ok ? 1 : 0, fuzz_moves, fuzz_bad, fuzz_entries, fuzz_stale, refill_plain, drain_B, refill_anti, drain_B,
                moves_per_s, adj_gbps, total_gbps, force_per_s, drain_adds_per_s, mem.per_chain, maxB, fuzz_gpu_ms,
                fuzz_cpu_ms, omp_get_max_threads(), total_s, g_failures);
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "test_ls_moves: exception: %s\n", ex.what());
    std::printf("RESULT ok=0 exception=1\n");
    return 1;
  }
}

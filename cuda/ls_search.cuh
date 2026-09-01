// T3.2b — per-chain iterated local search on top of T3.2a's state and moves
// (docs/design.md §5.4, adapted to the fact that near the 496 there is no
// vertex of tightness 1..3, so the perturbation must force-add vertices of
// tightness 4..8 and the classical (1,2)-swap only matters far from the record).
//
// The search adds, in its OWN arrays (LSSearch), what the loop needs beyond
// LSState and never modifies T3.2a's files:
//   cand, cand_head   uint32 [B][CL], [B]   "low-tightness candidate list": vertices
//                                           seen with 1 <= tight <= list_tmax; rebuilt
//                                           as the CL lowest-tightness non-members by a
//                                           warp-level two-pass rescan every
//                                           rescan_every iterations, extended by pushes
//                                           during removes (dropped when full);
//                                           validate on pop (tight window, !inS, !tabu)
//   tabu_bits         uint32 [B][INS_WORDS]  bitmap of the vertices present in the tabu
//                                           ring (any expiry) -> O(1) per-lane negative
//                                           test; ring consulted only when the bit is set
//   n1                uint32 [B]            exact #{v not in S : tight[v] == 1}, kept by
//                                           the search's own add/remove (gates the swap)
//   best_hash         uint32 [B]            order-independent hash of best_S (plateau
//                                           detection: equal size, different set)
//   stall, reported, restart_req, rng[B][32], stats[B]
//
// Everything that mutates S goes through lss_add / lss_remove / lss_force_add
// below (T3.2a's semantics plus the bookkeeping above). After ANY T3.2a batched
// kernel touched the state (lsk_apply_moves, lsk_drain_free, lsa_init_from_sets)
// call lss_init(...) again to resynchronise n1 / tabu_bits / best_hash.
//
// Calling convention of the device functions: identical to ls_moves.cuh — one
// warp per chain, all 32 lanes call convergently with uniform arguments, no
// __syncthreads anywhere (blocks hold several chains).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kiss/types.h"
#include "kiss_cuda.h"
#include "ls_state.cuh"

namespace kiss::cuda {

// ---- parameters ---------------------------------------------------------------------
enum LSSelectStrategy : unsigned {
  LSS_CAND = 1u,      // (b) the low-tightness candidate list (primary)
  LSS_NEIGH = 2u,     // (c) random neighbour of a random S member
  LSS_UNIFORM = 4u,   // (a) uniform random vertex indices
};

struct LSSearchParams {
  int CL = 4096;               // candidate-list capacity per chain
  int tmin = 1, tmax = 8;      // perturbation window: force-add v with tmin <= tight[v] <= tmax
  int list_tmax = 8;           // push / rescan threshold of the candidate list (1..16)
  int tenure = 8;              // tabu tenure in ILS iterations for removed vertices ...
  int tenure_jitter = 4;       // ... plus a uniform random 0..tenure_jitter
  int strength = 1;            // force-adds per perturbation
  int select_scan = 1;         // 1: full candidate-list argmin of tightness (random tie-break);
                               // 0: sampled tournament (tournament_rounds x 32 samples)
  int tournament_rounds = 4;
  int greedy_pct = 100;        // probability (%) that the selection is greedy (min tightness)
                               // instead of uniform among the valid samples
  unsigned strategies = LSS_CAND | LSS_NEIGH | LSS_UNIFORM;
  int neigh_rounds = 2;        // rounds of 32 neighbour samples when the list yields nothing
  int uniform_rounds = 8;      // rounds of 32 uniform probes after that
  int swap_enable = 1;         // ARW (1,2)-swap when n1 > 0
  int max_x_tries = 4;         // (1,2)-swap: candidate x per iteration
  int tabu_drain = 1;          // drain skips tabu free vertices (pushed back afterwards)
  int max_drop = 24;           // leash: revert to best_S when size < best_size - max_drop
  int stall_limit = 1000000;   // iterations without strict improvement -> revert + restart_req
  int restart_del = 8;         // random (tabu) deletions after a revert
  int rescan_every = 8;        // candidate-list rebuild period (iterations)
  int accept_equal = 1;        // plateau: an equal-size, different set replaces best_S
  int warps_per_block = 8;     // 4 or 8
};

// ---- per-chain statistics (device counters, cumulative since lss_init) ---------------
struct LSChainStats {
  uint32_t iterations = 0;
  uint32_t adds = 0;            // successful single-vertex adds (drain + swap + revert)
  uint32_t removes = 0;         // single-vertex removes (force-add, swap, deletions, revert)
  uint32_t swaps = 0;           // (1,2)-swaps applied
  uint32_t swap_tries = 0;      // iterations in which the swap search ran
  uint32_t force_adds = 0;
  uint32_t force_removed = 0;   // members removed by force-adds
  uint32_t select_fail = 0;     // iterations where no candidate was found
  uint32_t reverts = 0;         // leash reverts (size < best - max_drop)
  uint32_t restarts = 0;        // stall reverts (restart_req set for the host)
  uint32_t cand_rescans = 0;
  uint32_t fl_rescans = 0;
  uint32_t cand_pushes = 0;
  uint32_t plateau = 0;         // best_S replaced by a different equal-size set
  uint32_t improvements = 0;    // strict improvements of best_size
  uint32_t min_size = 0;        // deepest dip of |S| ever visited
  uint32_t best_size = 0;       // snapshot at the end of the last launch
  uint32_t size = 0;            // snapshot
  uint32_t stall = 0;           // snapshot
  uint32_t n1 = 0;              // snapshot
  uint32_t cand_head = 0;       // snapshot
  uint32_t sel_cand = 0, sel_neigh = 0, sel_uniform = 0;   // where the perturbation vertex came from
  uint32_t sel_tight_sum = 0;   // sum of tight[v] over force-added v (mean = / force_adds)
};

// ---- device-side search state ------------------------------------------------------------
struct LSSearch {
  LSSearchParams sp;
  int B = 0;
  uint32_t* cand = nullptr;         // [B][CL]
  uint32_t* cand_head = nullptr;    // [B]
  uint32_t* tabu_bits = nullptr;    // [B][INS_WORDS]
  uint32_t* n1 = nullptr;           // [B]
  uint32_t* best_hash = nullptr;    // [B]
  uint32_t* stall = nullptr;        // [B]
  uint32_t* reported = nullptr;     // [B] largest size already written to an output slot
  uint32_t* restart_req = nullptr;  // [B] 1 when the stall limit fired (host may reseed)
  uint64_t* rng = nullptr;          // [B][32]
  LSChainStats* stats = nullptr;    // [B]
};

// Output slots: d_out_slots = uint32[1 + nslots * LS_OUT_STRIDE]; [0] = number of
// claimed slots (may exceed nslots: only the first nslots were written); slot i at
// 1 + i * LS_OUT_STRIDE holds {size, chain, members[size]}.
constexpr int LS_OUT_STRIDE = SMAX + 2;
inline std::size_t ls_out_words(int nslots) { return 1 + static_cast<std::size_t>(nslots) * LS_OUT_STRIDE; }

// ---- host API (cuda/ls_search.cu) -------------------------------------------------------------
std::size_t lss_bytes_per_chain(const LSSearchParams& sp);
LSSearch lss_alloc(const LSState& st, const LSSearchParams& sp);   // throws on error / no fit
void lss_free(LSSearch& ss);

// Resynchronise the search state with st (call after lsa_init_from_sets or any
// T3.2a kernel): tabu bitmap from the ring, n1 from tight/inS, best_hash from
// best_S, cand_head = 0 (rebuilt at the first iteration), reported = 0,
// restart_req = 0, stall = 0, per-lane RNG seeded from `seed`; reset_stats
// zeroes the counters (min_size/best_size/size = current). Synchronous.
void lss_init(LSSearch& ss, const LSState& st, const DeviceLeech& L, uint64_t seed, bool reset_stats = true,
              cudaStream_t stream = 0);

// The loop kernel: every chain runs K ILS iterations (warp per chain,
// sp.warps_per_block warps per block). Sets *d_flag = 1 and claims an output
// slot whenever a chain's |S| >= target and larger than what it reported before.
// The host must verify anything reported. Asynchronous on `stream`.
void lsk_run_chains(const LSState& st, const LSSearch& ss, const uint32_t* d_adj, const DeviceLeech& L, int K,
                    uint32_t target, uint32_t* d_flag, uint32_t* d_out_slots, int nslots, cudaStream_t stream = 0);

std::vector<LSChainStats> lss_download_stats(const LSSearch& ss);   // synchronous
std::vector<uint32_t> lss_download_restart_req(const LSSearch& ss, bool clear = true);

struct LSFound {
  uint32_t chain = 0, size = 0;
  std::vector<uint32_t> S;
};
// Reads the claimed slots (at most nslots), then resets the counter and the flag
// if `reset`. Synchronous.
std::vector<LSFound> lss_download_output(uint32_t* d_out_slots, int nslots, uint32_t* d_flag, bool reset = true);

// Plateau / scripted swap primitive (e.g. T3.1's four (12,12) moves of the 496):
// chain b removes d_rem[b][0..R) (each existing member, with tabu `tenure`) and
// then adds d_add[b][0..A) (each only if it is free — pair-free in antipodal
// mode — so independence is never broken; LS_NONE entries are skipped). Goes
// through the search's own moves, so n1 / candidate list / tabu bitmap stay in
// sync. d_ret (optional, uint32[B][2]) receives {removed, added} per chain.
// best_S is not touched (the next run_chains iteration tracks it). Asynchronous.
void lsk_apply_swaps(const LSState& st, const LSSearch& ss, const uint32_t* d_rem, int R, const uint32_t* d_add, int A,
                     const uint32_t* d_adj, const DeviceLeech& L, uint32_t tenure, uint32_t* d_ret = nullptr,
                     cudaStream_t stream = 0);

// Order-independent hash used for best_hash (host copy for tests).
inline uint32_t ls_mix32(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7FEB352Du;
  v ^= v >> 15;
  v *= 0x846CA68Bu;
  v ^= v >> 16;
  return v;
}
inline uint32_t ls_set_hash_host(const uint32_t* S, std::size_t n) {
  uint32_t h = 0;
  for (std::size_t i = 0; i < n; ++i) h += ls_mix32(S[i]);
  return h;
}

// =====================================================================================
// Device primitives (nvcc only)
// =====================================================================================
#ifdef __CUDACC__
}  // namespace kiss::cuda
#include "ls_moves.cuh"
#include "ls_rng.cuh"
namespace kiss::cuda {

constexpr int LSS_LX_CAP = 256;   // shared-memory capacity of L_x per warp

// ---- accessors ------------------------------------------------------------------------
__device__ __forceinline__ uint32_t* lss_cand(const LSSearch& ss, int b) {
  return ss.cand + static_cast<std::size_t>(b) * ss.sp.CL;
}
__device__ __forceinline__ uint32_t* lss_tbits(const LSSearch& ss, int b) {
  return ss.tabu_bits + static_cast<std::size_t>(b) * INS_WORDS;
}
__device__ __forceinline__ bool lss_bit(const uint32_t* bits, uint32_t v) { return (bits[v >> 5] >> (v & 31)) & 1u; }

__device__ __forceinline__ uint32_t ls_mix32_dev(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7FEB352Du;
  v ^= v >> 15;
  v *= 0x846CA68Bu;
  v ^= v >> 16;
  return v;
}

// Warp reductions (full mask, result in every lane).
__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(LS_FULL, v, o);
  return v;
}
__device__ __forceinline__ uint32_t warp_min_u32(uint32_t v) {
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(LS_FULL, v, o));
  return v;
}
// Exclusive prefix over the lanes; total in every lane.
__device__ __forceinline__ uint32_t warp_exscan_u32(uint32_t v, uint32_t& total) {
  const int lane = ls_lane();
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(LS_FULL, x, o);
    if (lane >= o) x += y;
  }
  total = __shfl_sync(LS_FULL, x, 31);
  return x - v;
}

__device__ __forceinline__ uint32_t lss_hash_S(const uint32_t* S, uint32_t sz) {
  uint32_t h = 0;
  for (uint32_t i = static_cast<uint32_t>(ls_lane()); i < sz; i += 32) h += ls_mix32_dev(S[i]);
  return warp_sum_u32(h);
}

// ---- tabu ring with bitmap ---------------------------------------------------------------
// Warp-uniform push (all lanes, uniform v): evicts the oldest slot, keeps the
// bitmap exact (bit set <=> vertex present in some slot).
__device__ __forceinline__ void lss_tabu_push(const LSState& st, const LSSearch& ss, int b, uint32_t v,
                                              uint32_t expiry) {
  const int TABU = st.p.TABU;
  uint32_t* tv = st.tabu_v + static_cast<std::size_t>(b) * TABU;
  uint32_t* te = st.tabu_exp + static_cast<std::size_t>(b) * TABU;
  uint32_t* tb = lss_tbits(ss, b);
  const uint32_t h = st.tabu_head[b];
  const uint32_t slot = h % static_cast<uint32_t>(TABU);
  const uint32_t old = tv[slot];
  bool other = false;
  for (int base = 0; base < TABU; base += 32) {
    const int i = base + ls_lane();
    other |= (static_cast<uint32_t>(i) != slot) && tv[i] == old;
  }
  const bool keep_old = __ballot_sync(LS_FULL, other) != 0;
  __syncwarp();
  if (ls_lane() == 0) {
    if (old != LS_NONE && old != v && !keep_old) tb[old >> 5] &= ~(1u << (old & 31));
    tv[slot] = v;
    te[slot] = expiry;
    st.tabu_head[b] = h + 1;
    tb[v >> 5] |= 1u << (v & 31);
  }
  __syncwarp();
}

// Per-lane (divergent, no collectives) tabu test; v may differ per lane.
__device__ __forceinline__ bool lss_lane_tabu(const LSState& st, const LSSearch& ss, int b, uint32_t v, uint32_t it,
                                              const uint32_t* __restrict__ neg) {
  const uint32_t* tb = lss_tbits(ss, b);
  uint32_t nv = v;
  bool maybe = lss_bit(tb, v);
  if (st.p.antipodal) {
    nv = neg[v];
    maybe |= lss_bit(tb, nv);
  }
  if (!maybe) return false;
  const int TABU = st.p.TABU;
  const uint32_t* tv = st.tabu_v + static_cast<std::size_t>(b) * TABU;
  const uint32_t* te = st.tabu_exp + static_cast<std::size_t>(b) * TABU;
  for (int i = 0; i < TABU; ++i) {
    const uint32_t x = tv[i];
    if ((x == v || x == nv) && te[i] > it) return true;
  }
  return false;
}

// ---- candidate list -----------------------------------------------------------------------
__device__ __forceinline__ void lss_cand_push1(const LSSearch& ss, int b, uint32_t v, LSChainStats& cs) {
  const uint32_t h = ss.cand_head[b];
  if (h < static_cast<uint32_t>(ss.sp.CL)) {
    if (ls_lane() == 0) {
      lss_cand(ss, b)[h] = v;
      ss.cand_head[b] = h + 1;
    }
    ++cs.cand_pushes;
  }
  __syncwarp();
}

// ---- single-vertex moves with bookkeeping -------------------------------------------------------
__device__ __forceinline__ int lss_add1(const LSState& st, const LSSearch& ss, int b, uint32_t v,
                                        const uint32_t* __restrict__ d_adj, LSChainStats& cs) {
  if (in_S(st, b, v)) return 0;
  const uint32_t sz = st.size[b];
  if (sz >= static_cast<uint32_t>(st.p.SMAX)) return 0;
  uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  int d = (t[v] == 1) ? -1 : 0;   // v leaves the "outside S" population
  if (ls_lane() == 0) {
    ls_bits(st, b)[v >> 5] |= 1u << (v & 31);
    ls_S(st, b)[sz] = v;
    st.size[b] = sz + 1;
  }
  __syncwarp();
  const uint32_t* row = d_adj + static_cast<std::size_t>(v) * kiss::DEG;
  for (int base = 0; base < kiss::DEG; base += 32) {
    const int i = base + ls_lane();
    bool up = false, down = false;
    if (i < kiss::DEG) {
      const uint32_t n = row[i];
      const uint16_t old = t[n];
      t[n] = static_cast<uint16_t>(old + 1);
      const bool ins = lss_bit(bits, n);
      up = !ins && old == 0;
      down = !ins && old == 1;
    }
    d += __popc(__ballot_sync(LS_FULL, up)) - __popc(__ballot_sync(LS_FULL, down));
  }
  __syncwarp();
  if (ls_lane() == 0) ss.n1[b] = static_cast<uint32_t>(static_cast<int>(ss.n1[b]) + d);
  __syncwarp();
  ++cs.adds;
  return 1;
}

__device__ __forceinline__ int lss_remove1(const LSState& st, const LSSearch& ss, int b, uint32_t v,
                                           const uint32_t* __restrict__ d_adj, uint32_t tenure, bool tabu,
                                           LSChainStats& cs) {
  if (!in_S(st, b, v)) return 0;
  uint32_t* S = ls_S(st, b);
  const uint32_t sz = st.size[b];
  int pos = -1;
  for (uint32_t base = 0; base < sz; base += 32) {
    const uint32_t i = base + static_cast<uint32_t>(ls_lane());
    const bool hit = (i < sz) && (S[i] == v);
    const unsigned m = __ballot_sync(LS_FULL, hit);
    if (m) { pos = static_cast<int>(base) + (__ffs(static_cast<int>(m)) - 1); break; }
  }
  if (pos < 0) return 0;
  if (ls_lane() == 0) {
    ls_bits(st, b)[v >> 5] &= ~(1u << (v & 31));
    S[pos] = S[sz - 1];
    st.size[b] = sz - 1;
  }
  __syncwarp();
  uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  uint32_t* fl = st.fl + static_cast<std::size_t>(b) * st.p.FL;
  uint32_t* cand = lss_cand(ss, b);
  const uint32_t FL = static_cast<uint32_t>(st.p.FL);
  const uint32_t CL = static_cast<uint32_t>(ss.sp.CL);
  const uint32_t LT = static_cast<uint32_t>(ss.sp.list_tmax);
  uint32_t head = st.fl_head[b];      // uniform
  uint32_t chead = ss.cand_head[b];   // uniform
  int d = 0;
  const uint32_t* row = d_adj + static_cast<std::size_t>(v) * kiss::DEG;
  for (int base = 0; base < kiss::DEG; base += 32) {
    const int i = base + ls_lane();
    bool pushf = false, pushc = false, up = false, down = false;
    uint32_t n = 0;
    if (i < kiss::DEG) {
      n = row[i];
      const uint16_t nt = static_cast<uint16_t>(t[n] - 1);
      t[n] = nt;
      const bool ins = lss_bit(bits, n);
      pushf = (nt == 0) && !ins;
      pushc = !ins && nt >= 1 && nt <= LT;
      up = !ins && nt == 1;
      down = pushf;   // was 1, now 0
    }
    const unsigned mf = __ballot_sync(LS_FULL, pushf);
    if (pushf) {
      const uint32_t p = head + static_cast<uint32_t>(__popc(mf & ls_lanemask_lt()));
      if (p < FL) fl[p] = n;
    }
    head += static_cast<uint32_t>(__popc(mf));
    const unsigned mc = __ballot_sync(LS_FULL, pushc);
    if (pushc) {
      const uint32_t p = chead + static_cast<uint32_t>(__popc(mc & ls_lanemask_lt()));
      if (p < CL) cand[p] = n;
    }
    chead += static_cast<uint32_t>(__popc(mc));
    d += __popc(__ballot_sync(LS_FULL, up)) - __popc(__ballot_sync(LS_FULL, down));
  }
  const uint32_t tvv = t[v];   // unchanged by the loop (no self-adjacency)
  if (tvv == 1) ++d;           // v is outside S now
  if (ls_lane() == 0) {
    if (head > FL) { st.fl_overflow[b] = 1; head = FL; }
    st.fl_head[b] = head;
    const uint32_t pushed = chead > CL ? CL : chead;
    cs.cand_pushes += pushed - ss.cand_head[b];
    ss.cand_head[b] = pushed;
    ss.n1[b] = static_cast<uint32_t>(static_cast<int>(ss.n1[b]) + d);
  }
  cs.cand_pushes = __shfl_sync(LS_FULL, cs.cand_pushes, 0);
  __syncwarp();
  if (tvv == 0) warp_free_push1(st, b, v);
  else if (tvv <= LT) lss_cand_push1(ss, b, v, cs);
  if (tabu) lss_tabu_push(st, ss, b, v, st.iter[b] + tenure);
  ++cs.removes;
  return 1;
}

// ---- antipodal-aware pair moves -------------------------------------------------------------------
__device__ __forceinline__ int lss_add(const LSState& st, const LSSearch& ss, int b, uint32_t v,
                                       const uint32_t* __restrict__ d_adj, const uint32_t* __restrict__ neg,
                                       LSChainStats& cs) {
  int n = lss_add1(st, ss, b, v, d_adj, cs);
  if (st.p.antipodal) n += lss_add1(st, ss, b, neg[v], d_adj, cs);
  return n;
}
__device__ __forceinline__ int lss_remove(const LSState& st, const LSSearch& ss, int b, uint32_t v,
                                          const uint32_t* __restrict__ d_adj, const uint32_t* __restrict__ neg,
                                          uint32_t tenure, bool tabu, LSChainStats& cs) {
  int n = lss_remove1(st, ss, b, v, d_adj, tenure, tabu, cs);
  if (st.p.antipodal) n += lss_remove1(st, ss, b, neg[v], d_adj, tenure, tabu, cs);
  return n;
}

// Random tenure: tenure + U{0..jitter}, warp-uniform.
__device__ __forceinline__ uint32_t lss_draw_tenure(const LSSearch& ss, uint64_t& rng) {
  const uint32_t j = static_cast<uint32_t>(ss.sp.tenure_jitter);
  return static_cast<uint32_t>(ss.sp.tenure) + (j ? ls_rng_below_warp(rng, j + 1) : 0u);
}

// T3.2a's warp_force_add with the search's remove/add. Returns members removed, -1 if v in S.
__device__ __forceinline__ int lss_force_add(const LSState& st, const LSSearch& ss, int b, uint32_t v,
                                             const uint32_t* __restrict__ d_adj, const DeviceLeech& L, uint64_t& rng,
                                             LSChainStats& cs) {
  if (in_S(st, b, v)) return -1;
  const uint32_t* S = ls_S(st, b);
  const uint32_t nv = st.p.antipodal ? L.neg[v] : v;
  int wv[PACKED_WORDS], wn[PACKED_WORDS];
  load_packed(L.packed, v, wv);
  load_packed(L.packed, nv, wn);
  const uint32_t sz = st.size[b];
  uint32_t slot0 = LS_NONE, slot1 = LS_NONE;
  int cnt = 0;
  for (uint32_t base = 0; base < sz; base += 32) {
    const uint32_t i = base + static_cast<uint32_t>(ls_lane());
    uint32_t s = LS_NONE;
    bool hit = false;
    if (i < sz) {
      s = S[i];
      hit = adjacent_pre(L.packed, wv, s) || (st.p.antipodal && adjacent_pre(L.packed, wn, s));
    }
    unsigned m = __ballot_sync(LS_FULL, hit);
    while (m) {
      const int j = __ffs(static_cast<int>(m)) - 1;
      m &= m - 1u;
      const uint32_t sv = __shfl_sync(LS_FULL, s, j);
      if (cnt < LS_FORCE_CAP && ls_lane() == (cnt & 31)) {
        if (cnt < 32) slot0 = sv; else slot1 = sv;
      }
      ++cnt;
    }
  }
  int removed = 0;
  const int cap = st.p.force_cap < LS_FORCE_CAP ? st.p.force_cap : LS_FORCE_CAP;
  const int listed = cnt < cap ? cnt : cap;
  for (int k = 0; k < listed; ++k) {
    const uint32_t s = __shfl_sync(LS_FULL, k < 32 ? slot0 : slot1, k & 31);
    removed += lss_remove(st, ss, b, s, d_adj, L.neg, lss_draw_tenure(ss, rng), true, cs);
  }
  if (cnt > cap) {
    for (;;) {
      const uint32_t csz = st.size[b];
      uint32_t found = LS_NONE;
      for (uint32_t base = 0; base < csz && found == LS_NONE; base += 32) {
        const uint32_t i = base + static_cast<uint32_t>(ls_lane());
        uint32_t s = LS_NONE;
        bool hit = false;
        if (i < csz) {
          s = S[i];
          hit = adjacent_pre(L.packed, wv, s) || (st.p.antipodal && adjacent_pre(L.packed, wn, s));
        }
        const unsigned m = __ballot_sync(LS_FULL, hit);
        if (m) found = __shfl_sync(LS_FULL, s, __ffs(static_cast<int>(m)) - 1);
      }
      if (found == LS_NONE) break;
      removed += lss_remove(st, ss, b, found, d_adj, L.neg, lss_draw_tenure(ss, rng), true, cs);
    }
  }
  lss_add(st, ss, b, v, d_adj, L.neg, cs);
  ++cs.force_adds;
  cs.force_removed += static_cast<uint32_t>(removed);
  return removed;
}

// ---- drain: pop-validate-add until empty; tabu vertices are set aside and pushed back ----------------
__device__ __forceinline__ int lss_drain(const LSState& st, const LSSearch& ss, int b, const uint32_t* __restrict__ d_adj,
                                         const DeviceLeech& L, uint32_t it, LSChainStats& cs) {
  uint32_t stash0 = LS_NONE, stash1 = LS_NONE;
  int ns = 0;
  bool over = false;
  int adds = 0;
  for (;;) {
    const uint32_t v = warp_free_pop(st, b, L.neg);
    if (v == LS_NONE) break;
    if (ss.sp.tabu_drain && is_tabu(st, b, v, it, L.neg)) {
      if (ns < 64) {
        if (ls_lane() == (ns & 31)) { if (ns < 32) stash0 = v; else stash1 = v; }
      } else {
        over = true;
      }
      ++ns;
      continue;
    }
    adds += lss_add(st, ss, b, v, d_adj, L.neg, cs);
  }
  const int nst = ns < 64 ? ns : 64;
  for (int k = 0; k < nst; ++k) {
    const uint32_t v = __shfl_sync(LS_FULL, k < 32 ? stash0 : stash1, k & 31);
    warp_free_push1(st, b, v);
  }
  if (over) {   // some free tabu vertices were dropped: rebuild the free-list at the next iteration
    if (ls_lane() == 0) st.fl_overflow[b] = 1;
    __syncwarp();
  }
  return adds;
}

// ---- warp-level rescans -------------------------------------------------------------------------------
// Free-list: ascending compaction of {v : tight == 0 && !inS} (antipodal: neg[v] too).
__device__ __forceinline__ void lss_rescan_free_warp(const LSState& st, int b, const uint32_t* __restrict__ neg,
                                                     LSChainStats& cs) {
  const uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  uint32_t* fl = st.fl + static_cast<std::size_t>(b) * st.p.FL;
  const uint32_t FL = static_cast<uint32_t>(st.p.FL);
  uint32_t head = 0;
  for (int base = 0; base < kiss::N; base += 32) {
    const int v = base + ls_lane();
    bool fr = false;
    if (v < kiss::N) {
      fr = (t[v] == 0) && !lss_bit(bits, static_cast<uint32_t>(v));
      if (fr && st.p.antipodal) {
        const uint32_t nv = neg[v];
        fr = (t[nv] == 0) && !lss_bit(bits, nv);
      }
    }
    const unsigned m = __ballot_sync(LS_FULL, fr);
    if (fr) {
      const uint32_t p = head + static_cast<uint32_t>(__popc(m & ls_lanemask_lt()));
      if (p < FL) fl[p] = static_cast<uint32_t>(v);
    }
    head += static_cast<uint32_t>(__popc(m));
  }
  if (ls_lane() == 0) {
    st.fl_overflow[b] = head > FL ? 1 : 0;
    st.fl_head[b] = head > FL ? FL : head;
  }
  __syncwarp();
  ++cs.fl_rescans;
}

// Candidate list: the CL lowest-tightness non-members with 1 <= tight <= list_tmax.
// Pass 1 histogram of tightness 1..16 (uint4 loads, 8 tightness values per lane),
// threshold t* = largest t with cum(t) <= CL, pass 2 compaction of tight <= t* plus
// the first CL - cum(t*) vertices of tightness t* + 1 (index order).
__device__ __forceinline__ void lss_rescan_cand_warp(const LSState& st, const LSSearch& ss, int b,
                                                     const uint32_t* __restrict__ neg, LSChainStats& cs) {
  const uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  uint32_t* cand = lss_cand(ss, b);
  const uint32_t CL = static_cast<uint32_t>(ss.sp.CL);
  const int LT = ss.sp.list_tmax < 16 ? ss.sp.list_tmax : 16;
  constexpr int CHUNKS = kiss::N / 8;   // 24570 uint4 chunks of 8 tightness values (N % 8 == 0)
  uint32_t c[17];
#pragma unroll
  for (int k = 0; k < 17; ++k) c[k] = 0;
  const uint4* t4 = reinterpret_cast<const uint4*>(t);
  for (int chunk = ls_lane(); chunk < CHUNKS; chunk += 32) {
    const uint4 q = t4[chunk];
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
    const uint32_t v0 = static_cast<uint32_t>(chunk) * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t tv = (w[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu;
      const uint32_t v = v0 + static_cast<uint32_t>(j);
      if (tv >= 1 && tv <= 16 && !lss_bit(bits, v)) {
#pragma unroll
        for (int k = 1; k <= 16; ++k) c[k] += (tv == static_cast<uint32_t>(k));
      }
    }
  }
  uint32_t cum = 0;
  int tstar = 0;
  uint32_t cum_star = 0;
#pragma unroll
  for (int k = 1; k <= 16; ++k) {
    const uint32_t ck = warp_sum_u32(c[k]);
    if (k <= LT) {
      cum += ck;
      if (cum <= CL) { tstar = k; cum_star = cum; }
    }
  }
  const uint32_t fill = CL - cum_star;   // slots for tightness tstar + 1
  const uint32_t tnext = static_cast<uint32_t>(tstar + 1);
  uint32_t hmain = 0, hextra = 0;
  for (int chunk = ls_lane() * 0; chunk < CHUNKS; chunk += 32) {   // uniform loop bound
    const int my = chunk + ls_lane();
    uint32_t cm = 0, ce = 0;
    uint32_t mmask = 0, emask = 0;   // bit j: value j of my chunk qualifies (main / extra)
    if (my < CHUNKS) {
      const uint4 q = t4[my];
      const uint32_t w[4] = {q.x, q.y, q.z, q.w};
      const uint32_t v0 = static_cast<uint32_t>(my) * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t tv = (w[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu;
        const uint32_t v = v0 + static_cast<uint32_t>(j);
        bool ok = tv >= 1 && tv <= tnext && tnext <= static_cast<uint32_t>(LT) + 1 && !lss_bit(bits, v);
        if (ok && tv == tnext && tstar >= LT) ok = false;   // nothing beyond LT
        if (ok && st.p.antipodal) ok = !lss_bit(bits, neg[v]);
        if (ok) {
          if (tv <= static_cast<uint32_t>(tstar)) { mmask |= 1u << j; ++cm; }
          else { emask |= 1u << j; ++ce; }
        }
      }
    }
    uint32_t tm = 0, te = 0;
    const uint32_t pm = warp_exscan_u32(cm, tm);
    const uint32_t pe = warp_exscan_u32(ce, te);
    if (my < CHUNKS) {
      const uint32_t v0 = static_cast<uint32_t>(my) * 8;
      uint32_t km = 0, ke = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if ((mmask >> j) & 1u) {
          const uint32_t p = hmain + pm + km++;
          if (p < cum_star) cand[p] = v0 + static_cast<uint32_t>(j);
        }
        if ((emask >> j) & 1u) {
          const uint32_t p = hextra + pe + ke++;
          if (p < fill) cand[cum_star + p] = v0 + static_cast<uint32_t>(j);
        }
      }
    }
    hmain += tm;
    hextra += te;
  }
  const uint32_t nmain = hmain < cum_star ? hmain : cum_star;
  const uint32_t nextra = hextra < fill ? hextra : fill;
  // main entries occupy [0, nmain) only if hmain == cum_star (it is: same predicate); guard anyway
  if (ls_lane() == 0) ss.cand_head[b] = (nmain == cum_star) ? cum_star + nextra : nmain;
  __syncwarp();
  ++cs.cand_rescans;
}

// ---- perturbation vertex selection --------------------------------------------------------------------
// Per-lane validation of a candidate v: not in S (nor neg[v] in antipodal mode),
// tmin <= tight <= tmax, not tabu. Returns the tightness in tv.
__device__ __forceinline__ bool lss_valid_pert(const LSState& st, const LSSearch& ss, int b, uint32_t v, uint32_t it,
                                               const uint32_t* __restrict__ neg, uint32_t& tv) {
  if (v >= static_cast<uint32_t>(kiss::N)) return false;
  const uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  if (lss_bit(bits, v)) return false;
  tv = t[v];
  if (tv < static_cast<uint32_t>(ss.sp.tmin) || tv > static_cast<uint32_t>(ss.sp.tmax)) return false;
  if (st.p.antipodal) {
    const uint32_t nv = neg[v];
    if (lss_bit(bits, nv)) return false;
    const uint32_t tn = t[nv];
    if (tn > tv) tv = tn;
    if (tv > static_cast<uint32_t>(ss.sp.tmax)) return false;
  }
  return !lss_lane_tabu(st, ss, b, v, it, neg);
}

// Reduce (valid, tv, v) over the lanes: greedy -> minimum tv with random tie-break,
// else uniform among valid lanes. Returns LS_NONE if no lane is valid.
__device__ __forceinline__ uint32_t lss_pick_lane(bool valid, uint32_t tv, uint32_t v, bool greedy, uint64_t& rng,
                                                  uint32_t& out_t) {
  const uint32_t key = ls_rng_next(rng);
  uint32_t val = 0xFFFFFFFFu;
  if (valid) val = greedy ? ((tv & 0xFFu) << 24) | (key >> 8) : (key >> 1);
  const uint32_t mn = warp_min_u32(val);
  if (mn == 0xFFFFFFFFu) return LS_NONE;
  const unsigned m = __ballot_sync(LS_FULL, val == mn);
  const int src = __ffs(static_cast<int>(m)) - 1;
  out_t = __shfl_sync(LS_FULL, tv, src);
  return __shfl_sync(LS_FULL, v, src);
}

// Returns the vertex to force-add (LS_NONE if none found); `src` = 1 cand, 2 neigh, 3 uniform.
__device__ __forceinline__ uint32_t lss_select(const LSState& st, const LSSearch& ss, int b, const DeviceLeech& L,
                                               const uint32_t* __restrict__ d_adj, uint32_t it, uint64_t& rng,
                                               uint32_t& out_t, int& src) {
  const bool greedy = (ls_rng_below_warp(rng, 100) < static_cast<uint32_t>(ss.sp.greedy_pct));
  src = 0;
  const uint32_t ch = ss.cand_head[b];
  if ((ss.sp.strategies & LSS_CAND) && ch > 0) {
    const uint32_t* cand = lss_cand(ss, b);
    if (ss.sp.select_scan) {
      // full-list argmin: every lane keeps its best entry, then one reduction
      uint32_t bt = 0xFFFFFFFFu, bv = LS_NONE, bkey = 0;
      bool any = false;
      for (uint32_t i = static_cast<uint32_t>(ls_lane()); i < ch; i += 32) {
        const uint32_t v = cand[i];
        uint32_t tv = 0;
        if (lss_valid_pert(st, ss, b, v, it, L.neg, tv)) {
          const uint32_t key = ls_rng_next(rng);
          const bool better = !any || (greedy ? (tv < bt || (tv == bt && key > bkey)) : key > bkey);
          if (better) { bt = tv; bv = v; bkey = key; any = true; }
        }
      }
      const uint32_t v = lss_pick_lane(any, bt, bv, greedy, rng, out_t);
      if (v != LS_NONE) { src = 1; return v; }
    } else {
      for (int r = 0; r < ss.sp.tournament_rounds; ++r) {
        const uint32_t v = cand[ls_rng_below(rng, ch)];
        uint32_t tv = 0;
        const bool ok = lss_valid_pert(st, ss, b, v, it, L.neg, tv);
        const uint32_t w = lss_pick_lane(ok, tv, v, greedy, rng, out_t);
        if (w != LS_NONE) { src = 1; return w; }
      }
    }
  }
  const uint32_t sz = st.size[b];
  if ((ss.sp.strategies & LSS_NEIGH) && sz > 0) {
    const uint32_t* S = ls_S(st, b);
    for (int r = 0; r < ss.sp.neigh_rounds; ++r) {
      const uint32_t s = S[ls_rng_below(rng, sz)];
      const uint32_t v = d_adj[static_cast<std::size_t>(s) * kiss::DEG + ls_rng_below(rng, kiss::DEG)];
      uint32_t tv = 0;
      const bool ok = lss_valid_pert(st, ss, b, v, it, L.neg, tv);
      const uint32_t w = lss_pick_lane(ok, tv, v, greedy, rng, out_t);
      if (w != LS_NONE) { src = 2; return w; }
    }
  }
  if (ss.sp.strategies & LSS_UNIFORM) {
    for (int r = 0; r < ss.sp.uniform_rounds; ++r) {
      const uint32_t v = ls_rng_below(rng, kiss::N);
      uint32_t tv = 0;
      const bool ok = lss_valid_pert(st, ss, b, v, it, L.neg, tv);
      const uint32_t w = lss_pick_lane(ok, tv, v, greedy, rng, out_t);
      if (w != LS_NONE) { src = 3; return w; }
    }
  }
  return LS_NONE;
}

// ---- ARW (1,2)-swap -----------------------------------------------------------------------------------------
// A tightness-1 non-member u conflicts with exactly one member x; L_x = the
// tightness-1, non-tabu non-members of row(x); any non-adjacent pair {a, c} in L_x
// (antipodal mode: additionally a not adjacent to neg[c]) gives remove x, add a, add c.
// u is found in the candidate list (tightness-1 vertices are pushed there on 2 -> 1
// transitions and included by every rescan). sLx: 256 uint32 of shared memory.
__device__ __forceinline__ bool lss_try_swap12(const LSState& st, const LSSearch& ss, int b,
                                               const uint32_t* __restrict__ d_adj, const DeviceLeech& L, uint32_t it,
                                               uint64_t& rng, uint32_t* sLx, LSChainStats& cs) {
  const uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  const uint32_t* S = ls_S(st, b);
  const uint32_t* cand = lss_cand(ss, b);
  const uint32_t ch = ss.cand_head[b];
  if (ch == 0) return false;
  uint32_t tried = LS_NONE;   // lane k holds the k-th tried x
  int ntried = 0;
  ++cs.swap_tries;
  const uint32_t start = ls_rng_below_warp(rng, ch);
  uint32_t scan_pos = 0;   // uniform scan position over the rotated list
  for (int tr = 0; tr < ss.sp.max_x_tries; ++tr) {
    // ---- find u: tightness 1, not in S, not tabu
    uint32_t u = LS_NONE;
    while (scan_pos < ch && u == LS_NONE) {
      const uint32_t i = scan_pos + static_cast<uint32_t>(ls_lane());
      uint32_t v = LS_NONE;
      bool hit = false;
      if (i < ch) {
        v = cand[(start + i) % ch];
        hit = t[v] == 1 && !lss_bit(bits, v) && (!st.p.antipodal || !lss_bit(bits, L.neg[v])) &&
              !lss_lane_tabu(st, ss, b, v, it, L.neg);
      }
      const unsigned m = __ballot_sync(LS_FULL, hit);
      scan_pos += 32;
      if (m) {
        // random hit
        const int nb = __popc(m);
        const uint32_t pick = ls_rng_below_warp(rng, static_cast<uint32_t>(nb));
        unsigned mm = m;
        for (uint32_t k = 0; k < pick; ++k) mm &= mm - 1u;
        u = __shfl_sync(LS_FULL, v, __ffs(static_cast<int>(mm)) - 1);
      }
    }
    if (u == LS_NONE) break;
    // ---- x = the unique conflict of u in S
    int wu[PACKED_WORDS];
    load_packed(L.packed, u, wu);
    uint32_t x = LS_NONE;
    const uint32_t sz = st.size[b];
    for (uint32_t base = 0; base < sz && x == LS_NONE; base += 32) {
      const uint32_t i = base + static_cast<uint32_t>(ls_lane());
      uint32_t s = LS_NONE;
      bool hit = false;
      if (i < sz) { s = S[i]; hit = adjacent_pre(L.packed, wu, s); }
      const unsigned m = __ballot_sync(LS_FULL, hit);
      if (m) x = __shfl_sync(LS_FULL, s, __ffs(static_cast<int>(m)) - 1);
    }
    if (x == LS_NONE) continue;   // stale (tight[u] was 1 but no conflict found: cannot happen with exact tight)
    if (__ballot_sync(LS_FULL, tried == x)) continue;
    if (ls_lane() == (ntried & 31)) tried = x;
    ++ntried;
    // ---- gather L_x into shared memory
    const uint32_t* row = d_adj + static_cast<std::size_t>(x) * kiss::DEG;
    uint32_t m_cnt = 0;
    for (int base = 0; base < kiss::DEG && m_cnt < LSS_LX_CAP; base += 32) {
      const int i = base + ls_lane();
      uint32_t w = LS_NONE;
      bool ok = false;
      if (i < kiss::DEG) {
        w = row[i];
        ok = t[w] == 1 && !lss_bit(bits, w) && (!st.p.antipodal || !lss_bit(bits, L.neg[w])) &&
             !lss_lane_tabu(st, ss, b, w, it, L.neg);
      }
      const unsigned m = __ballot_sync(LS_FULL, ok);
      if (ok) {
        const uint32_t p = m_cnt + static_cast<uint32_t>(__popc(m & ls_lanemask_lt()));
        if (p < LSS_LX_CAP) sLx[p] = w;
      }
      m_cnt += static_cast<uint32_t>(__popc(m));
    }
    if (m_cnt > LSS_LX_CAP) m_cnt = LSS_LX_CAP;
    __syncwarp();
    // ---- all-pairs test: a non-adjacent pair {a, c}
    uint32_t a = LS_NONE, cpair = LS_NONE;
    for (uint32_t i = 0; i + 1 < m_cnt && a == LS_NONE; ++i) {
      const uint32_t ai = sLx[i];
      int wa[PACKED_WORDS];
      load_packed(L.packed, ai, wa);
      for (uint32_t j = i + 1 + static_cast<uint32_t>(ls_lane()); ; j += 32) {
        const bool inr = j < m_cnt;
        uint32_t cj = LS_NONE;
        bool ok = false;
        if (inr) {
          cj = sLx[j];
          ok = !adjacent_pre(L.packed, wa, cj);
          if (ok && st.p.antipodal) ok = !adjacent_pre(L.packed, wa, L.neg[cj]);
        }
        const unsigned m = __ballot_sync(LS_FULL, ok);
        if (m) {
          a = ai;
          cpair = __shfl_sync(LS_FULL, cj, __ffs(static_cast<int>(m)) - 1);
          break;
        }
        if (!__ballot_sync(LS_FULL, inr)) break;
      }
    }
    __syncwarp();
    if (a != LS_NONE) {
      lss_remove(st, ss, b, x, d_adj, L.neg, lss_draw_tenure(ss, rng), true, cs);
      lss_add(st, ss, b, a, d_adj, L.neg, cs);
      lss_add(st, ss, b, cpair, d_adj, L.neg, cs);
      ++cs.swaps;
      return true;
    }
  }
  return false;
}

// ---- best tracking / revert -------------------------------------------------------------------------------------
__device__ __forceinline__ void lss_copy_best(const LSState& st, const LSSearch& ss, int b, uint32_t hash) {
  const uint32_t sz = st.size[b];
  const uint32_t* S = ls_S(st, b);
  uint32_t* bs = st.best_S + static_cast<std::size_t>(b) * st.p.SMAX;
  for (uint32_t i = static_cast<uint32_t>(ls_lane()); i < sz; i += 32) bs[i] = S[i];
  __syncwarp();
  if (ls_lane() == 0) {
    st.best_size[b] = sz;
    ss.best_hash[b] = hash;
  }
  __syncwarp();
}

__device__ __forceinline__ bool lss_in_list(const uint32_t* list, uint32_t n, uint32_t v) {
  for (uint32_t base = 0; base < n; base += 32) {
    const uint32_t i = base + static_cast<uint32_t>(ls_lane());
    if (__ballot_sync(LS_FULL, i < n && list[i] == v)) return true;
  }
  return false;
}

// S := best_S by the symmetric difference (no tabu pushes).
__device__ __forceinline__ void lss_revert_to_best(const LSState& st, const LSSearch& ss, int b,
                                                   const uint32_t* __restrict__ d_adj, const DeviceLeech& L,
                                                   LSChainStats& cs) {
  const uint32_t* bs = st.best_S + static_cast<std::size_t>(b) * st.p.SMAX;
  const uint32_t bsz = st.best_size[b];
  const uint32_t* S = ls_S(st, b);
  uint32_t i = 0;
  while (i < st.size[b]) {
    const uint32_t v = S[i];
    if (!lss_in_list(bs, bsz, v)) lss_remove(st, ss, b, v, d_adj, L.neg, 0u, false, cs);
    else ++i;
  }
  for (uint32_t j = 0; j < bsz; ++j) {
    const uint32_t v = bs[j];
    if (!in_S(st, b, v)) lss_add(st, ss, b, v, d_adj, L.neg, cs);
  }
}

}  // namespace kiss::cuda
#else
}  // namespace kiss::cuda  (host translation units: close the namespace opened above)
#endif  // __CUDACC__

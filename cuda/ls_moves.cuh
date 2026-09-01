// T3.2a — warp-cooperative incremental moves on one chain (docs/design.md §5.3).
//
// Calling convention (all functions below):
//   * one warp == one chain; ALL 32 lanes call the function convergently with
//     identical (b, v, ...) arguments. Internally every collective uses the full
//     mask (LS_FULL); no __syncthreads, so blocks may hold several chains.
//   * only the owning warp ever touches chain b's arrays, so no atomics are
//     needed: a row of the adjacency table has no duplicates and row(v) and
//     row(neg[v]) are disjoint (dot(n,v)=16 and dot(n,-v)=16 is impossible), so
//     within one move every tight[b][n] is read-modify-written by exactly one lane.
//   * the row loops read d_adj[v*DEG ...] with lanes striding by 32 (coalesced).
//   * vertices are uint32 < N; LS_NONE (0xFFFFFFFF) means "none".
//
// Semantics (non-antipodal):
//   warp_add(v)        v not in S: set bit, append to S, tight[n]++ over row(v).
//                      Returns 1 (0 if v was already in S or |S| == SMAX: no-op).
//                      Does NOT check tight[v]==0: adding a conflicting vertex is
//                      allowed (tight stays exact); force_add uses this.
//   warp_remove(v,ten) v in S: clear bit, swap-with-last in the S list, tight[n]--
//                      over row(v); every n whose tightness reaches 0 and is not in
//                      S is pushed to the free-list (ballot + prefix, one head update
//                      per 32-entry window; overflow flag if dropped). v itself is
//                      pushed too if tight[v]==0 after the removal (it is then free).
//                      v is pushed to the tabu ring with expiry iter[b] + ten.
//                      Returns 1 (0 if v was not in S: no-op).
//   warp_force_add(v)  removes every s in S adjacent to v (in S-list order), then
//                      adds v. Returns the number of vertices removed; -1 if v was
//                      already in S (no-op).
//   warp_free_pop()    pops from the free-list until an entry validates
//                      (tight==0 && !inS); returns it or LS_NONE when empty.
//   is_tabu(v, it)     any tabu slot with vertex v (or neg[v] in antipodal mode) and
//                      expiry > it.
// Antipodal mode (st.p.antipodal): add/remove apply to v and neg[v] (each only if
// its own precondition holds), force_add gathers the conflicts of both v and
// neg[v], pop validates both v and neg[v]. Everything stays exact even if S is
// not closed under negation; lsa_check reports closure violations.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kiss/types.h"
#include "kiss_cuda.h"
#include "ls_state.cuh"

namespace kiss::cuda {

constexpr unsigned LS_FULL = 0xFFFFFFFFu;

// Lane id and "lanes below me" mask.
__device__ __forceinline__ int ls_lane() { return static_cast<int>(threadIdx.x) & 31; }
__device__ __forceinline__ unsigned ls_lanemask_lt() { return (1u << ls_lane()) - 1u; }

// 6 dp4a on the packed SoA words: true iff <C[u],C[w]> == 16.
__device__ __forceinline__ bool adjacent(const uint32_t* __restrict__ packed, uint32_t u, uint32_t w) {
  int acc = 0;
#pragma unroll
  for (int k = 0; k < PACKED_WORDS; ++k)
    acc = __dp4a(static_cast<int>(packed[static_cast<std::size_t>(k) * kiss::N + u]),
                 static_cast<int>(packed[static_cast<std::size_t>(k) * kiss::N + w]), acc);
  return acc == 16;
}

// Same with the 6 words of u preloaded in registers (for scans).
__device__ __forceinline__ void load_packed(const uint32_t* __restrict__ packed, uint32_t u, int (&w)[PACKED_WORDS]) {
#pragma unroll
  for (int k = 0; k < PACKED_WORDS; ++k)
    w[k] = static_cast<int>(packed[static_cast<std::size_t>(k) * kiss::N + u]);
}
__device__ __forceinline__ bool adjacent_pre(const uint32_t* __restrict__ packed, const int (&wu)[PACKED_WORDS],
                                             uint32_t w) {
  int acc = 0;
#pragma unroll
  for (int k = 0; k < PACKED_WORDS; ++k)
    acc = __dp4a(wu[k], static_cast<int>(packed[static_cast<std::size_t>(k) * kiss::N + w]), acc);
  return acc == 16;
}

// ---- per-chain accessors -----------------------------------------------------------
__device__ __forceinline__ uint16_t* ls_tight(const LSState& st, int b) {
  return st.tight + static_cast<std::size_t>(b) * kiss::N;
}
__device__ __forceinline__ uint32_t* ls_bits(const LSState& st, int b) {
  return st.inS + static_cast<std::size_t>(b) * INS_WORDS;
}
__device__ __forceinline__ uint32_t* ls_S(const LSState& st, int b) {
  return st.S + static_cast<std::size_t>(b) * st.p.SMAX;
}
__device__ __forceinline__ bool in_S(const LSState& st, int b, uint32_t v) {
  return (ls_bits(st, b)[v >> 5] >> (v & 31)) & 1u;
}
__device__ __forceinline__ bool is_free(const LSState& st, int b, uint32_t v) {
  return ls_tight(st, b)[v] == 0 && !in_S(st, b, v);
}

// ---- tabu ring ---------------------------------------------------------------------
// Lane 0 pushes; call from all lanes (uniform), no sync needed afterwards for
// is_tabu because is_tabu is always separated by a __syncwarp in the callers.
__device__ __forceinline__ void warp_tabu_push(const LSState& st, int b, uint32_t v, uint32_t expiry) {
  if (ls_lane() == 0) {
    const uint32_t h = st.tabu_head[b];
    const uint32_t slot = h % static_cast<uint32_t>(st.p.TABU);
    st.tabu_v[static_cast<std::size_t>(b) * st.p.TABU + slot] = v;
    st.tabu_exp[static_cast<std::size_t>(b) * st.p.TABU + slot] = expiry;
    st.tabu_head[b] = h + 1;
  }
  __syncwarp();
}

// True (uniformly) iff v — or neg[v] in antipodal mode — sits in the ring with expiry > it.
__device__ __forceinline__ bool is_tabu(const LSState& st, int b, uint32_t v, uint32_t it,
                                        const uint32_t* __restrict__ d_neg = nullptr) {
  const uint32_t nv = (st.p.antipodal && d_neg) ? d_neg[v] : v;
  const uint32_t* tv = st.tabu_v + static_cast<std::size_t>(b) * st.p.TABU;
  const uint32_t* te = st.tabu_exp + static_cast<std::size_t>(b) * st.p.TABU;
  bool hit = false;
  for (int base = 0; base < st.p.TABU; base += 32) {
    const int i = base + ls_lane();
    const uint32_t x = tv[i];
    hit |= (x == v || x == nv) && te[i] > it;
  }
  return __ballot_sync(LS_FULL, hit) != 0;
}

// ---- free-list ------------------------------------------------------------------------
// Single push (lane 0), e.g. for v itself after a remove.
__device__ __forceinline__ void warp_free_push1(const LSState& st, int b, uint32_t v) {
  if (ls_lane() == 0) {
    const uint32_t h = st.fl_head[b];
    if (h < static_cast<uint32_t>(st.p.FL)) {
      st.fl[static_cast<std::size_t>(b) * st.p.FL + h] = v;
      st.fl_head[b] = h + 1;
    } else {
      st.fl_overflow[b] = 1;
    }
  }
  __syncwarp();
}

// Pop with validate-on-pop; antipodal mode requires neg[v] free too.
__device__ __forceinline__ uint32_t warp_free_pop(const LSState& st, int b, const uint32_t* __restrict__ d_neg) {
  uint32_t* fl = st.fl + static_cast<std::size_t>(b) * st.p.FL;
  uint32_t h = st.fl_head[b];
  uint32_t out = LS_NONE;
  while (h > 0) {
    --h;
    const uint32_t v = fl[h];
    bool ok = is_free(st, b, v);
    if (st.p.antipodal && ok) ok = is_free(st, b, d_neg[v]);
    if (ok) { out = v; break; }
  }
  __syncwarp();
  if (ls_lane() == 0) st.fl_head[b] = h;
  __syncwarp();
  return out;
}

// ---- single-vertex primitives (no antipodal handling) ----------------------------------
__device__ __forceinline__ int warp_add1(const LSState& st, int b, uint32_t v, const uint32_t* __restrict__ d_adj) {
  if (in_S(st, b, v)) return 0;
  const uint32_t sz = st.size[b];
  if (sz >= static_cast<uint32_t>(st.p.SMAX)) return 0;
  if (ls_lane() == 0) {
    ls_bits(st, b)[v >> 5] |= 1u << (v & 31);
    ls_S(st, b)[sz] = v;
    st.size[b] = sz + 1;
  }
  __syncwarp();
  uint16_t* t = ls_tight(st, b);
  const uint32_t* row = d_adj + static_cast<std::size_t>(v) * kiss::DEG;
  for (int i = ls_lane(); i < kiss::DEG; i += 32) {
    const uint32_t n = row[i];
    t[n] = static_cast<uint16_t>(t[n] + 1);
  }
  __syncwarp();
  return 1;
}

__device__ __forceinline__ int warp_remove1(const LSState& st, int b, uint32_t v, const uint32_t* __restrict__ d_adj,
                                            uint32_t tenure) {
  if (!in_S(st, b, v)) return 0;
  uint32_t* S = ls_S(st, b);
  const uint32_t sz = st.size[b];
  // locate v (warp scan of <= SMAX entries)
  int pos = -1;
  for (uint32_t base = 0; base < sz; base += 32) {
    const uint32_t i = base + static_cast<uint32_t>(ls_lane());
    const bool hit = (i < sz) && (S[i] == v);
    const unsigned m = __ballot_sync(LS_FULL, hit);
    if (m) { pos = static_cast<int>(base) + (__ffs(static_cast<int>(m)) - 1); break; }
  }
  // pos >= 0 always (bit set <=> in list); guard anyway
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
  const uint32_t FL = static_cast<uint32_t>(st.p.FL);
  uint32_t head = st.fl_head[b];   // identical in every lane
  const uint32_t* row = d_adj + static_cast<std::size_t>(v) * kiss::DEG;
  for (int base = 0; base < kiss::DEG; base += 32) {
    const int i = base + ls_lane();
    bool push = false;
    uint32_t n = 0;
    if (i < kiss::DEG) {
      n = row[i];
      const uint16_t nt = static_cast<uint16_t>(t[n] - 1);
      t[n] = nt;
      push = (nt == 0) && !((bits[n >> 5] >> (n & 31)) & 1u);
    }
    const unsigned m = __ballot_sync(LS_FULL, push);
    if (push) {
      const uint32_t p = head + static_cast<uint32_t>(__popc(m & ls_lanemask_lt()));
      if (p < FL) fl[p] = n;
    }
    head += static_cast<uint32_t>(__popc(m));
  }
  if (ls_lane() == 0) {
    if (head > FL) { st.fl_overflow[b] = 1; head = FL; }
    st.fl_head[b] = head;
  }
  __syncwarp();
  if (t[v] == 0) warp_free_push1(st, b, v);   // v itself is free now (if S was independent)
  warp_tabu_push(st, b, v, st.iter[b] + tenure);
  return 1;
}

// ---- public moves (antipodal-aware) ------------------------------------------------------
__device__ __forceinline__ int warp_add(const LSState& st, int b, uint32_t v, const uint32_t* __restrict__ d_adj,
                                        const uint32_t* __restrict__ d_neg) {
  int n = warp_add1(st, b, v, d_adj);
  if (st.p.antipodal) n += warp_add1(st, b, d_neg[v], d_adj);
  return n;
}

__device__ __forceinline__ int warp_remove(const LSState& st, int b, uint32_t v, const uint32_t* __restrict__ d_adj,
                                           const uint32_t* __restrict__ d_neg, uint32_t tenure) {
  int n = warp_remove1(st, b, v, d_adj, tenure);
  if (st.p.antipodal) n += warp_remove1(st, b, d_neg[v], d_adj, tenure);
  return n;
}

// Conflicts of v (and neg[v] in antipodal mode) in S, in S-list order, gathered
// into per-lane registers (LS_FORCE_CAP entries; beyond that a rescan loop
// removes the rest one at a time), then removed, then v added.
constexpr int LS_FORCE_CAP = 64;   // 2 registers per lane

__device__ __forceinline__ int warp_force_add(const LSState& st, int b, uint32_t v, const uint32_t* __restrict__ d_adj,
                                              const DeviceLeech& L, uint32_t tenure) {
  if (in_S(st, b, v)) return -1;
  const uint32_t* S = ls_S(st, b);
  const uint32_t nv = st.p.antipodal ? L.neg[v] : v;
  int wv[PACKED_WORDS], wn[PACKED_WORDS];
  load_packed(L.packed, v, wv);
  load_packed(L.packed, nv, wn);
  const uint32_t sz = st.size[b];
  uint32_t slot0 = LS_NONE, slot1 = LS_NONE;
  int cnt = 0;   // uniform
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
    removed += warp_remove(st, b, s, d_adj, L.neg, tenure);   // no-op if already gone (antipodal partner)
  }
  if (cnt > cap) {
    // Overflow of the register list: remove the first remaining conflict, rescan, repeat.
    for (;;) {
      const uint32_t cs = st.size[b];
      uint32_t found = LS_NONE;
      for (uint32_t base = 0; base < cs && found == LS_NONE; base += 32) {
        const uint32_t i = base + static_cast<uint32_t>(ls_lane());
        uint32_t s = LS_NONE;
        bool hit = false;
        if (i < cs) {
          s = S[i];
          hit = adjacent_pre(L.packed, wv, s) || (st.p.antipodal && adjacent_pre(L.packed, wn, s));
        }
        const unsigned m = __ballot_sync(LS_FULL, hit);
        if (m) found = __shfl_sync(LS_FULL, s, __ffs(static_cast<int>(m)) - 1);
      }
      if (found == LS_NONE) break;
      removed += warp_remove(st, b, found, d_adj, L.neg, tenure);
    }
  }
  warp_add(st, b, v, d_adj, L.neg);
  return removed;
}

// best-ever tracking: if size > best_size copy S -> best_S (lanes stride).
__device__ __forceinline__ bool warp_update_best(const LSState& st, int b) {
  const uint32_t sz = st.size[b];
  if (sz <= st.best_size[b]) return false;
  const uint32_t* S = ls_S(st, b);
  uint32_t* bs = st.best_S + static_cast<std::size_t>(b) * st.p.SMAX;
  for (uint32_t i = static_cast<uint32_t>(ls_lane()); i < sz; i += 32) bs[i] = S[i];
  __syncwarp();
  if (ls_lane() == 0) st.best_size[b] = sz;
  __syncwarp();
  return true;
}

}  // namespace kiss::cuda

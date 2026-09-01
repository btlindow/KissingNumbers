// T3.2a — chain state of the GPU local-search engine (docs/design.md §5.2, §5.3).
//
// Structure-of-arrays over B chains; one warp owns one chain. This header is
// plain C++ apart from the RNG type: it can be included from .cpp files (it only
// needs <cuda_runtime.h>, like cuda/kiss_cuda.h); the typed Philox state is only
// visible under nvcc (LSRng is an opaque struct elsewhere).
//
// Layout (all device pointers, row-major, chain b at offset b*stride):
//   tight       uint16 [B][N]          #S-members adjacent (dot == 16) to v
//   inS         uint32 [B][INS_WORDS]  membership bitmap, bit (v&31) of word (v>>5)
//                                      (T1.5 layout, so kiss::cuda::count_free works)
//   S, size     uint32 [B][SMAX], [B]  member list (unordered; remove = swap-with-last)
//   fl, fl_head uint32 [B][FL], [B]    lazy free-list (§5.3): vertices pushed when
//   fl_overflow uint8  [B]             their tightness dropped to 0 outside S; entries
//                                      may be stale -> validate on pop; overflow flag
//                                      set when a push was dropped (host rescans)
//   tabu_v      uint32 [B][TABU]       ring of recently removed vertices (LS_NONE = empty)
//   tabu_exp    uint32 [B][TABU]       expiry: v is tabu at iteration it iff exp > it
//   tabu_head   uint32 [B]             next slot (mod TABU)
//   iter        uint32 [B]             chain iteration counter (advanced by the caller)
//   best_size   uint32 [B], best_S uint32 [B][SMAX]   best-ever set per chain
//   rng         Philox4x32-10 [B]      subsequence = chain id (see lsa_seed_rng)
//
// Memory per chain (SMAX=1024, FL=4096, TABU=32): 442,753 bytes (see
// lsa_report_memory); the shared adjacency table (3.62 GB) is NOT part of the
// state — the caller uploads it once with kiss::cuda::adjacency_to_device and
// passes the pointer to every kernel.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kiss/types.h"
#include "kiss_cuda.h"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace kiss::cuda {

#ifdef __CUDACC__
using LSRng = curandStatePhilox4_32_10_t;
#else
struct LSRngOpaque;
using LSRng = LSRngOpaque;
#endif

constexpr uint32_t LS_NONE = 0xFFFFFFFFu;   // "no vertex" (empty pop, empty tabu slot)

struct LSParams {
  int B = 0;            // number of chains
  int N = kiss::N;      // must equal kiss::N (the kernels use the compile-time constant)
  int SMAX = kiss::cuda::SMAX;   // must equal kiss::cuda::SMAX (tightness_full's stride)
  int FL = 4096;        // free-list capacity per chain
  int TABU = 32;        // tabu ring length; multiple of 32 (one ballot per 32 slots)
  int force_cap = 64;   // conflicts gathered per force_add before falling back to
                        // rescan-and-remove-first (1..LS_FORCE_CAP=64); tests use small values
  bool antipodal = false;   // moves apply to {v, neg[v]} together
};

struct LSState {
  LSParams p;
  uint16_t* tight = nullptr;       // [B][N]
  uint32_t* inS = nullptr;         // [B][INS_WORDS]
  uint32_t* S = nullptr;           // [B][SMAX]
  uint32_t* size = nullptr;        // [B]
  uint32_t* fl = nullptr;          // [B][FL]
  uint32_t* fl_head = nullptr;     // [B]
  uint8_t* fl_overflow = nullptr;  // [B]
  uint32_t* tabu_v = nullptr;      // [B][TABU]
  uint32_t* tabu_exp = nullptr;    // [B][TABU]
  uint32_t* tabu_head = nullptr;   // [B]
  uint32_t* iter = nullptr;        // [B]
  uint32_t* best_size = nullptr;   // [B]
  uint32_t* best_S = nullptr;      // [B][SMAX]
  LSRng* rng = nullptr;            // [B]
};

// ---- memory accounting -----------------------------------------------------
struct LSMemory {
  std::size_t per_chain = 0;   // bytes of chain state per chain
  std::size_t fixed = 0;       // bytes independent of B (0 for now; kept for completeness)
  std::size_t total = 0;       // per_chain * B + fixed
  std::size_t device_free = 0, device_total = 0;   // cudaMemGetInfo at call time (0 if no device)
  // Largest B that fits in `avail` bytes (e.g. device_free after the adjacency
  // upload) with `margin` bytes left over.
  int max_B(std::size_t avail, std::size_t margin) const {
    if (avail <= margin + fixed) return 0;
    return static_cast<int>((avail - margin - fixed) / per_chain);
  }
};
LSMemory lsa_report_memory(const LSParams& p);

// ---- allocation --------------------------------------------------------------
// cudaMalloc of every array; verifies p (N, SMAX, TABU, FL > 0, B > 0) and that
// total + margin <= cudaMemGetInfo().free, otherwise throws std::runtime_error
// (loudly, docs/design.md §6.2). All arrays are zero-initialised except tabu_v (LS_NONE)
// and rng (uninitialised until lsa_seed_rng / lsa_init_from_sets).
LSState lsa_alloc(const LSParams& p, std::size_t margin_bytes = std::size_t(256) << 20);
void lsa_free(LSState& st);   // cudaFree everything, pointers reset; safe on empty state

// ---- initialisation ------------------------------------------------------------
// Fills S/size/inS from `sets` (sets.size() must be p.B, each <= SMAX entries of
// vertices < N, no duplicates), computes tight with T1.5's tightness_full,
// rebuilds the free-lists with lsk_rescan_free, resets tabu/iter/overflow, sets
// best = current, and seeds the RNG (Philox seed `seed`, subsequence = chain).
// In antipodal mode the sets are used as given (closure under neg is the
// caller's job; lsa_check reports violations). Synchronous; throws on error.
void lsa_init_from_sets(LSState& st, const std::vector<std::vector<uint32_t>>& sets,
                        const DeviceLeech& L, const uint32_t* d_adj, uint64_t seed = 0,
                        cudaStream_t stream = 0);

// curand_init(seed, subsequence = b, offset, &rng[b]) for all chains. Synchronous.
void lsa_seed_rng(LSState& st, uint64_t seed, uint64_t offset = 0, cudaStream_t stream = 0);

// ---- copy-back ------------------------------------------------------------------
struct LSHostChain {
  uint32_t size = 0, iter = 0, best_size = 0, fl_head = 0, tabu_head = 0;
  bool fl_overflow = false;
  std::vector<uint32_t> S;          // size entries
  std::vector<uint32_t> best_S;     // best_size entries
  std::vector<uint32_t> fl;         // fl_head entries (may contain stale vertices)
  std::vector<uint32_t> tabu_v, tabu_exp;   // TABU entries each
  std::vector<uint16_t> tight;      // N entries if requested, else empty
  std::vector<uint32_t> inS;        // INS_WORDS entries if requested, else empty
};
// Synchronous copy of chain b (0 <= b < B). with_tight also copies tight+inS.
LSHostChain lsa_download(const LSState& st, int b, bool with_tight = false);

// ---- self-check -------------------------------------------------------------------
// Recomputes tightness with tightness_full (in chunks of <= 64 chains into a
// scratch buffer) and compares with st.tight; also audits the list/bitmap
// consistency, independence, antipodal closure (antipodal mode only) and the
// free-list. Synchronous; returns one entry per chain.
struct LSCheck {
  uint32_t tight_mismatch = 0;   // #v with st.tight[b][v] != recompute
  uint32_t list_bad = 0;         // S entries >= N, without their inS bit, or size > SMAX
  uint32_t bitmap_bad = 0;       // |popcount(inS) - size| (duplicates / stray bits)
  uint32_t conflict_pairs = 0;   // #{ {s,t} in S : adjacent } = (sum_{s in S} tight[s]) / 2
  uint32_t antipodal_bad = 0;    // antipodal mode: #{ s in S : neg[s] not in S }
  uint32_t fl_bad = 0;           // free-list entries >= N or fl_head > FL
  uint32_t fl_stale = 0;         // free-list entries that are not free any more (informational)
  uint32_t fl_missing = 0;       // free vertices absent from the free-list (informational; 0 unless overflow / never pushed)
  uint32_t errors() const {
    return tight_mismatch + list_bad + bitmap_bad + conflict_pairs + antipodal_bad + fl_bad;
  }
};
std::vector<LSCheck> lsa_check(const LSState& st, const DeviceLeech& L, cudaStream_t stream = 0);

// ---- batched host-callable kernels (implemented in cuda/ls_moves.cu) -------------
// Scripted move lists (tests, seeding): moves[b*M + i] is executed in order by
// chain b's warp. ret (optional, [B][M] int32) receives each move's return value.
enum LSOp : uint32_t {
  LS_OP_NOP = 0,
  LS_OP_ADD = 1,         // warp_add(v)
  LS_OP_REMOVE = 2,      // warp_remove(v, tenure)
  LS_OP_FORCE_ADD = 3,   // warp_force_add(v, tenure)
  LS_OP_POP_ADD = 4,     // v = warp_free_pop(); if v != LS_NONE: warp_add(v); returns v (as int32)
  LS_OP_TICK = 5,        // iter[b] += v
  LS_OP_BEST = 6,        // warp_update_best()
  LS_OP_IS_TABU = 7,     // returns is_tabu(v, iter[b]) (0/1); no state change
};
struct LSMove {
  uint32_t op;
  uint32_t v;
};
void lsk_apply_moves(const LSState& st, const LSMove* d_moves, int M, const uint32_t* d_adj, const DeviceLeech& L,
                     uint32_t tenure, int32_t* d_ret = nullptr, cudaStream_t stream = 0);

// Block per chain: rebuilds the free-list as the ascending compaction of
// { v : tight == 0 && !inS } (antipodal mode: additionally neg[v] free), clears
// the overflow flag (sets it again if the free set exceeds FL). L.neg is used
// in antipodal mode only.
void lsk_rescan_free(const LSState& st, const DeviceLeech& L, cudaStream_t stream = 0);

// Warp per chain: pop-validate-add until the free-list is empty or max_adds
// adds were made (greedy completion). d_adds (optional, [B]) receives the count.
void lsk_drain_free(const LSState& st, const uint32_t* d_adj, const DeviceLeech& L, uint32_t max_adds,
                    uint32_t* d_adds = nullptr, cudaStream_t stream = 0);

}  // namespace kiss::cuda

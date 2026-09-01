// T3.2b — the per-chain ILS loop kernel (run_chains), the plateau-move
// primitive, and the host API declared in ls_search.cuh.
//
// One ILS iteration of chain b (warp per chain, docs/design.md §5.4, adapted to the fact that near the 496 the minimum tightness is 4):
//   0. maintenance: free-list rescan if fl_overflow; candidate-list rebuild every
//      rescan_every iterations (or when it is empty);
//   1. drain the free-list (greedy completion, tabu vertices set aside);
//   2. ARW (1,2)-swap if the chain has tightness-1 vertices (n1 > 0) and swaps are on;
//      on success drain again and skip the perturbation (ARW: "continue");
//   3. perturbation: `strength` force-adds of non-tabu vertices with
//      tmin <= tight <= tmax (candidate list first, then random neighbours of random
//      members, then uniform probes); the removed members become tabu with
//      tenure + U{0..jitter}; then drain;
//   4. best tracking (strict improvement -> copy, stall = 0; equal size with a
//      different set -> plateau copy), output slot if |S| >= target and larger than
//      previously reported; leash (|S| < best - max_drop -> revert to best_S);
//      stall > stall_limit -> revert + restart_del random tabu deletions +
//      restart_req[b] = 1 (the host may reseed instead);
//   5. iter[b] += 1 (the tabu clock).
#include <algorithm>
#include <stdexcept>
#include <string>

#include "ls_search.cuh"

namespace kiss::cuda {

namespace {

// ---------------------------------------------------------------------------
// init: block per chain
// ---------------------------------------------------------------------------
__global__ void lss_init_kernel(LSState st, LSSearch ss, uint64_t seed, int reset_stats) {
  __shared__ uint32_t sh[32];
  const int b = static_cast<int>(blockIdx.x);
  uint32_t* tb = ss.tabu_bits + static_cast<std::size_t>(b) * INS_WORDS;
  for (int i = static_cast<int>(threadIdx.x); i < INS_WORDS; i += static_cast<int>(blockDim.x)) tb[i] = 0u;
  __syncthreads();
  // tabu bitmap: every vertex present in some ring slot (any expiry)
  for (int i = static_cast<int>(threadIdx.x); i < st.p.TABU; i += static_cast<int>(blockDim.x)) {
    const uint32_t v = st.tabu_v[static_cast<std::size_t>(b) * st.p.TABU + i];
    if (v != LS_NONE && v < static_cast<uint32_t>(kiss::N)) atomicOr(&tb[v >> 5], 1u << (v & 31));
  }
  // n1 = #{v not in S : tight[v] == 1}; best_hash
  const uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  uint32_t n1 = 0;
  for (int v = static_cast<int>(threadIdx.x); v < kiss::N; v += static_cast<int>(blockDim.x))
    n1 += (t[v] == 1) && !lss_bit(bits, static_cast<uint32_t>(v));
  const uint32_t* bs = st.best_S + static_cast<std::size_t>(b) * st.p.SMAX;
  const uint32_t bsz = min(st.best_size[b], static_cast<uint32_t>(st.p.SMAX));
  uint32_t h = 0;
  for (uint32_t i = threadIdx.x; i < bsz; i += blockDim.x) h += ls_mix32_dev(bs[i]);
  n1 = warp_sum_u32(n1);
  h = warp_sum_u32(h);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    sh[threadIdx.x >> 5] = n1;
  }
  __syncthreads();
  uint32_t n1_tot = 0;
  for (unsigned w = 0; w < (blockDim.x + 31) / 32; ++w) n1_tot += sh[w];
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = h;
  __syncthreads();
  uint32_t h_tot = 0;
  for (unsigned w = 0; w < (blockDim.x + 31) / 32; ++w) h_tot += sh[w];
  // per-lane RNG streams
  if (threadIdx.x < 32)
    ss.rng[static_cast<std::size_t>(b) * 32 + threadIdx.x] =
        ls_rng_seed(seed, static_cast<uint64_t>(b) * 32ull + threadIdx.x);
  if (threadIdx.x == 0) {
    ss.n1[b] = n1_tot;
    ss.best_hash[b] = h_tot;
    ss.cand_head[b] = 0;
    ss.stall[b] = 0;
    ss.reported[b] = 0;
    ss.restart_req[b] = 0;
    LSChainStats cs = ss.stats[b];
    if (reset_stats) cs = LSChainStats{};
    cs.min_size = reset_stats ? st.size[b] : min(cs.min_size, st.size[b]);
    cs.best_size = st.best_size[b];
    cs.size = st.size[b];
    cs.stall = 0;
    cs.n1 = n1_tot;
    cs.cand_head = 0;
    ss.stats[b] = cs;
  }
}

// ---------------------------------------------------------------------------
// loop kernel helpers (warp-level, uniform arguments)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void lss_report(const LSState& st, const LSSearch& ss, int b, uint32_t* out, int nslots,
                                           uint32_t* flag) {
  const uint32_t sz = st.size[b];
  uint32_t slot = 0;
  if (ls_lane() == 0) slot = atomicAdd(out, 1u);
  slot = __shfl_sync(LS_FULL, slot, 0);
  if (slot < static_cast<uint32_t>(nslots)) {
    uint32_t* dst = out + 1 + static_cast<std::size_t>(slot) * LS_OUT_STRIDE;
    const uint32_t* S = ls_S(st, b);
    for (uint32_t i = static_cast<uint32_t>(ls_lane()); i < sz; i += 32) dst[2 + i] = S[i];
    if (ls_lane() == 0) {
      dst[0] = sz;
      dst[1] = static_cast<uint32_t>(b);
    }
  }
  __threadfence();
  if (ls_lane() == 0) {
    atomicExch(flag, 1u);
    ss.reported[b] = sz;
  }
  __syncwarp();
}

// Best tracking after any change of S. Returns true on a strict improvement.
__device__ __forceinline__ bool lss_track_best(const LSState& st, const LSSearch& ss, int b, uint32_t target,
                                               uint32_t* out, int nslots, uint32_t* flag, uint32_t& stall,
                                               LSChainStats& cs) {
  const uint32_t sz = st.size[b];
  const uint32_t bsz = st.best_size[b];
  bool improved = false;
  if (sz > bsz) {
    const uint32_t h = lss_hash_S(ls_S(st, b), sz);
    lss_copy_best(st, ss, b, h);
    stall = 0;
    ++cs.improvements;
    improved = true;
  } else if (sz == bsz && ss.sp.accept_equal) {
    const uint32_t h = lss_hash_S(ls_S(st, b), sz);
    if (h != ss.best_hash[b]) {
      lss_copy_best(st, ss, b, h);
      ++cs.plateau;
    }
  }
  if (sz >= target && sz > ss.reported[b]) lss_report(st, ss, b, out, nslots, flag);
  return improved;
}

// restart_del random members removed with tabu (antipodal: pairs).
__device__ __forceinline__ void lss_random_deletions(const LSState& st, const LSSearch& ss, int b,
                                                     const uint32_t* __restrict__ d_adj, const DeviceLeech& L,
                                                     uint64_t& rng, LSChainStats& cs) {
  for (int k = 0; k < ss.sp.restart_del; ++k) {
    const uint32_t sz = st.size[b];
    if (sz == 0) break;
    const uint32_t v = ls_S(st, b)[ls_rng_below_warp(rng, sz)];
    lss_remove(st, ss, b, v, d_adj, L.neg, lss_draw_tenure(ss, rng), true, cs);
  }
}

// ---------------------------------------------------------------------------
// the loop kernel
// ---------------------------------------------------------------------------
__global__ void run_chains_kernel(LSState st, LSSearch ss, const uint32_t* __restrict__ d_adj, DeviceLeech L, int K,
                                  uint32_t target, uint32_t* flag, uint32_t* out, int nslots) {
  extern __shared__ uint32_t smem[];
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int b = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x >> 5) + warp;
  if (b >= st.p.B) return;
  uint32_t* sLx = smem + static_cast<std::size_t>(warp) * LSS_LX_CAP;
  const int lane = ls_lane();

  LSChainStats cs = ss.stats[b];   // uniform copy in every lane
  uint64_t rng = ss.rng[static_cast<std::size_t>(b) * 32 + lane];
  uint32_t stall = ss.stall[b];
  const uint32_t rescan_every = static_cast<uint32_t>(ss.sp.rescan_every > 0 ? ss.sp.rescan_every : 1);
  const uint32_t max_drop = static_cast<uint32_t>(ss.sp.max_drop);

  for (int k = 0; k < K; ++k) {
    const uint32_t it = st.iter[b];

    // 0. maintenance
    if (st.fl_overflow[b]) lss_rescan_free_warp(st, b, L.neg, cs);
    if (it % rescan_every == 0 || ss.cand_head[b] == 0) lss_rescan_cand_warp(st, ss, b, L.neg, cs);

    // 1. greedy completion
    lss_drain(st, ss, b, d_adj, L, it, cs);
    lss_track_best(st, ss, b, target, out, nslots, flag, stall, cs);

    // 2. ARW (1,2)-swap (only meaningful far from the record: needs tightness-1 vertices)
    bool did_swap = false;
    if (ss.sp.swap_enable && ss.n1[b] > 0) {
      did_swap = lss_try_swap12(st, ss, b, d_adj, L, it, rng, sLx, cs);
      if (did_swap) {
        lss_drain(st, ss, b, d_adj, L, it, cs);
        lss_track_best(st, ss, b, target, out, nslots, flag, stall, cs);
      }
    }

    // 3. perturbation: force-add low-tightness vertices, then drain
    if (!did_swap) {
      for (int s = 0; s < ss.sp.strength; ++s) {
        uint32_t tv = 0;
        int src = 0;
        const uint32_t v = lss_select(st, ss, b, L, d_adj, it, rng, tv, src);
        if (v == LS_NONE) {
          ++cs.select_fail;
          break;
        }
        lss_force_add(st, ss, b, v, d_adj, L, rng, cs);
        cs.sel_tight_sum += tv;
        if (src == 1) ++cs.sel_cand;
        else if (src == 2) ++cs.sel_neigh;
        else ++cs.sel_uniform;
      }
      cs.min_size = min(cs.min_size, st.size[b]);
      lss_drain(st, ss, b, d_adj, L, it, cs);
      lss_track_best(st, ss, b, target, out, nslots, flag, stall, cs);
    }

    // 4. leash and stall handling
    if (st.size[b] + max_drop < st.best_size[b]) {
      lss_revert_to_best(st, ss, b, d_adj, L, cs);
      ++cs.reverts;
    }
    ++stall;
    if (stall > static_cast<uint32_t>(ss.sp.stall_limit)) {
      lss_revert_to_best(st, ss, b, d_adj, L, cs);
      lss_random_deletions(st, ss, b, d_adj, L, rng, cs);
      if (lane == 0) ss.restart_req[b] = 1u;
      ++cs.restarts;
      stall = 0;
    }

    // 5. tick
    if (lane == 0) st.iter[b] = it + 1;
    ++cs.iterations;
    __syncwarp();
  }

  cs.best_size = st.best_size[b];
  cs.size = st.size[b];
  cs.stall = stall;
  cs.n1 = ss.n1[b];
  cs.cand_head = ss.cand_head[b];
  ss.rng[static_cast<std::size_t>(b) * 32 + lane] = rng;
  if (lane == 0) {
    ss.stall[b] = stall;
    ss.stats[b] = cs;
  }
}

// ---------------------------------------------------------------------------
// plateau / scripted swap: remove rem[b][0..R) (tabu), then add add[b][0..A)
// (each only if free; pair-free in antipodal mode). ret[b] = {removed, added}.
// ---------------------------------------------------------------------------
__global__ void apply_swaps_kernel(LSState st, LSSearch ss, const uint32_t* __restrict__ rem, int R,
                                   const uint32_t* __restrict__ add, int A, const uint32_t* __restrict__ d_adj,
                                   DeviceLeech L, uint32_t tenure, uint32_t* ret) {
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int b = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x >> 5) + warp;
  if (b >= st.p.B) return;
  LSChainStats cs = ss.stats[b];
  uint32_t removed = 0, added = 0;
  for (int i = 0; i < R; ++i) {
    const uint32_t v = rem[static_cast<std::size_t>(b) * R + i];
    if (v == LS_NONE || v >= static_cast<uint32_t>(kiss::N)) continue;
    removed += static_cast<uint32_t>(lss_remove(st, ss, b, v, d_adj, L.neg, tenure, true, cs));
  }
  for (int j = 0; j < A; ++j) {
    const uint32_t v = add[static_cast<std::size_t>(b) * A + j];
    if (v == LS_NONE || v >= static_cast<uint32_t>(kiss::N)) continue;
    bool ok = is_free(st, b, v);
    if (ok && st.p.antipodal) ok = is_free(st, b, L.neg[v]);
    if (ok) added += static_cast<uint32_t>(lss_add(st, ss, b, v, d_adj, L.neg, cs));
  }
  cs.size = st.size[b];
  cs.n1 = ss.n1[b];
  cs.cand_head = ss.cand_head[b];
  cs.min_size = min(cs.min_size, st.size[b]);
  if (ls_lane() == 0) {
    ss.stats[b] = cs;
    if (ret) {
      ret[static_cast<std::size_t>(b) * 2] = removed;
      ret[static_cast<std::size_t>(b) * 2 + 1] = added;
    }
  }
}

template <class T>
T* dmalloc_zero(std::size_t count) {
  T* p = nullptr;
  KISS_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
  KISS_CUDA_CHECK(cudaMemset(p, 0, count * sizeof(T)));
  return p;
}

void validate_params(const LSSearchParams& sp) {
  auto bad = [](const std::string& m) { throw std::runtime_error("lss_alloc: " + m); };
  if (sp.CL <= 0) bad("CL must be > 0");
  if (sp.tmin < 1 || sp.tmax < sp.tmin) bad("need 1 <= tmin <= tmax");
  if (sp.list_tmax < 1 || sp.list_tmax > 16) bad("list_tmax must be in 1..16");
  if (sp.tenure < 0 || sp.tenure_jitter < 0) bad("tenure/jitter must be >= 0");
  if (sp.strength < 1) bad("strength must be >= 1");
  if (sp.greedy_pct < 0 || sp.greedy_pct > 100) bad("greedy_pct must be in 0..100");
  if (sp.tournament_rounds < 1 || sp.neigh_rounds < 0 || sp.uniform_rounds < 0) bad("bad sampling rounds");
  if (sp.max_x_tries < 1 || sp.max_x_tries > 32) bad("max_x_tries must be in 1..32");
  if (sp.max_drop < 0 || sp.stall_limit < 1 || sp.restart_del < 0) bad("bad max_drop/stall_limit/restart_del");
  if (sp.rescan_every < 1) bad("rescan_every must be >= 1");
  if (sp.warps_per_block != 1 && sp.warps_per_block != 2 && sp.warps_per_block != 4 && sp.warps_per_block != 8 &&
      sp.warps_per_block != 16)
    bad("warps_per_block must be 1, 2, 4, 8 or 16");
}

}  // namespace

// ---------------------------------------------------------------------------
// host API
// ---------------------------------------------------------------------------
std::size_t lss_bytes_per_chain(const LSSearchParams& sp) {
  std::size_t pc = 0;
  pc += static_cast<std::size_t>(sp.CL) * sizeof(uint32_t);   // cand
  pc += sizeof(uint32_t);                                     // cand_head
  pc += static_cast<std::size_t>(INS_WORDS) * sizeof(uint32_t);   // tabu_bits
  pc += 5 * sizeof(uint32_t);                                 // n1, best_hash, stall, reported, restart_req
  pc += 32 * sizeof(uint64_t);                                // rng
  pc += sizeof(LSChainStats);
  return pc;
}

LSSearch lss_alloc(const LSState& st, const LSSearchParams& sp) {
  validate_params(sp);
  if (st.p.B <= 0) throw std::runtime_error("lss_alloc: state has B == 0");
  if (st.p.TABU % 32 != 0) throw std::runtime_error("lss_alloc: TABU must be a multiple of 32");
  const std::size_t B = static_cast<std::size_t>(st.p.B);
  const std::size_t need = lss_bytes_per_chain(sp) * B;
  std::size_t fr = 0, to = 0;
  KISS_CUDA_CHECK(cudaMemGetInfo(&fr, &to));
  const std::size_t margin = std::size_t(128) << 20;
  if (need + margin > fr)
    throw std::runtime_error("lss_alloc: search state needs " + std::to_string(need) + " bytes + margin " +
                             std::to_string(margin) + " but only " + std::to_string(fr) + " of " + std::to_string(to) +
                             " bytes are free");
  LSSearch ss;
  ss.sp = sp;
  ss.B = st.p.B;
  try {
    ss.cand = dmalloc_zero<uint32_t>(B * static_cast<std::size_t>(sp.CL));
    ss.cand_head = dmalloc_zero<uint32_t>(B);
    ss.tabu_bits = dmalloc_zero<uint32_t>(B * INS_WORDS);
    ss.n1 = dmalloc_zero<uint32_t>(B);
    ss.best_hash = dmalloc_zero<uint32_t>(B);
    ss.stall = dmalloc_zero<uint32_t>(B);
    ss.reported = dmalloc_zero<uint32_t>(B);
    ss.restart_req = dmalloc_zero<uint32_t>(B);
    ss.rng = dmalloc_zero<uint64_t>(B * 32);
    ss.stats = dmalloc_zero<LSChainStats>(B);
  } catch (...) {
    lss_free(ss);
    throw;
  }
  return ss;
}

void lss_free(LSSearch& ss) {
  auto f = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
  f(ss.cand); f(ss.cand_head); f(ss.tabu_bits); f(ss.n1); f(ss.best_hash); f(ss.stall); f(ss.reported);
  f(ss.restart_req); f(ss.rng); f(ss.stats);
  ss.B = 0;
}

void lss_init(LSSearch& ss, const LSState& st, const DeviceLeech& /*L: reserved*/, uint64_t seed, bool reset_stats,
              cudaStream_t stream) {
  if (ss.B != st.p.B) throw std::runtime_error("lss_init: search state allocated for a different B");
  lss_init_kernel<<<st.p.B, 256, 0, stream>>>(st, ss, seed, reset_stats ? 1 : 0);
  KISS_CUDA_CHECK(cudaGetLastError());
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lsk_run_chains(const LSState& st, const LSSearch& ss, const uint32_t* d_adj, const DeviceLeech& L, int K,
                    uint32_t target, uint32_t* d_flag, uint32_t* d_out_slots, int nslots, cudaStream_t stream) {
  if (K <= 0) return;
  if (ss.B != st.p.B) throw std::runtime_error("lsk_run_chains: search state allocated for a different B");
  const int wpb = ss.sp.warps_per_block;
  const int blocks = (st.p.B + wpb - 1) / wpb;
  const std::size_t smem = static_cast<std::size_t>(wpb) * LSS_LX_CAP * sizeof(uint32_t);
  run_chains_kernel<<<blocks, 32 * wpb, smem, stream>>>(st, ss, d_adj, L, K, target, d_flag, d_out_slots, nslots);
  KISS_CUDA_CHECK(cudaGetLastError());
}

void lsk_apply_swaps(const LSState& st, const LSSearch& ss, const uint32_t* d_rem, int R, const uint32_t* d_add, int A,
                     const uint32_t* d_adj, const DeviceLeech& L, uint32_t tenure, uint32_t* d_ret,
                     cudaStream_t stream) {
  if (ss.B != st.p.B) throw std::runtime_error("lsk_apply_swaps: search state allocated for a different B");
  if (R < 0 || A < 0) throw std::runtime_error("lsk_apply_swaps: negative list length");
  const int wpb = 4;
  const int blocks = (st.p.B + wpb - 1) / wpb;
  apply_swaps_kernel<<<blocks, 32 * wpb, 0, stream>>>(st, ss, d_rem, R, d_add, A, d_adj, L, tenure, d_ret);
  KISS_CUDA_CHECK(cudaGetLastError());
}

std::vector<LSChainStats> lss_download_stats(const LSSearch& ss) {
  std::vector<LSChainStats> h(static_cast<std::size_t>(ss.B));
  if (ss.B > 0)
    KISS_CUDA_CHECK(cudaMemcpy(h.data(), ss.stats, h.size() * sizeof(LSChainStats), cudaMemcpyDeviceToHost));
  return h;
}

std::vector<uint32_t> lss_download_restart_req(const LSSearch& ss, bool clear) {
  std::vector<uint32_t> h(static_cast<std::size_t>(ss.B));
  if (ss.B > 0) {
    KISS_CUDA_CHECK(cudaMemcpy(h.data(), ss.restart_req, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (clear) KISS_CUDA_CHECK(cudaMemset(ss.restart_req, 0, h.size() * sizeof(uint32_t)));
  }
  return h;
}

std::vector<LSFound> lss_download_output(uint32_t* d_out_slots, int nslots, uint32_t* d_flag, bool reset) {
  uint32_t claimed = 0;
  KISS_CUDA_CHECK(cudaMemcpy(&claimed, d_out_slots, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  const uint32_t n = std::min<uint32_t>(claimed, static_cast<uint32_t>(nslots));
  std::vector<LSFound> out;
  if (n > 0) {
    std::vector<uint32_t> buf(static_cast<std::size_t>(n) * LS_OUT_STRIDE);
    KISS_CUDA_CHECK(cudaMemcpy(buf.data(), d_out_slots + 1, buf.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t* s = buf.data() + static_cast<std::size_t>(i) * LS_OUT_STRIDE;
      LSFound f;
      f.size = std::min<uint32_t>(s[0], static_cast<uint32_t>(SMAX));
      f.chain = s[1];
      f.S.assign(s + 2, s + 2 + f.size);
      out.push_back(std::move(f));
    }
  }
  if (reset) {
    KISS_CUDA_CHECK(cudaMemset(d_out_slots, 0, sizeof(uint32_t)));
    if (d_flag) KISS_CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(uint32_t)));
  }
  return out;
}

}  // namespace kiss::cuda

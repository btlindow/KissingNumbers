// T3.2a — host-callable batched kernels wrapping the warp-cooperative moves of
// ls_moves.cuh: scripted move lists, full free-list rescan, greedy completion.
#include <stdexcept>

#include "ls_moves.cuh"
#include "ls_state.cuh"

namespace kiss::cuda {

namespace {

constexpr int WARPS_PER_BLOCK = 4;   // 128 threads; no __syncthreads inside chain logic

__device__ __forceinline__ int chain_of_warp() {
  return static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + static_cast<int>(threadIdx.x >> 5);
}

__global__ void apply_moves_kernel(LSState st, const LSMove* __restrict__ moves, int M,
                                   const uint32_t* __restrict__ d_adj, DeviceLeech L, uint32_t tenure,
                                   int32_t* ret) {
  const int b = chain_of_warp();
  if (b >= st.p.B) return;
  const LSMove* mv = moves + static_cast<std::size_t>(b) * M;
  for (int i = 0; i < M; ++i) {
    const LSMove m = mv[i];   // same address in every lane -> broadcast load
    int32_t r = 0;
    switch (m.op) {
      case LS_OP_ADD: r = warp_add(st, b, m.v, d_adj, L.neg); break;
      case LS_OP_REMOVE: r = warp_remove(st, b, m.v, d_adj, L.neg, tenure); break;
      case LS_OP_FORCE_ADD: r = warp_force_add(st, b, m.v, d_adj, L, tenure); break;
      case LS_OP_POP_ADD: {
        const uint32_t v = warp_free_pop(st, b, L.neg);
        if (v != LS_NONE) warp_add(st, b, v, d_adj, L.neg);
        r = static_cast<int32_t>(v);
        break;
      }
      case LS_OP_TICK:
        if (ls_lane() == 0) st.iter[b] += m.v;
        __syncwarp();
        break;
      case LS_OP_BEST: r = warp_update_best(st, b); break;
      case LS_OP_IS_TABU: r = is_tabu(st, b, m.v, st.iter[b], L.neg) ? 1 : 0; break;
      default: break;
    }
    if (ret && ls_lane() == 0) ret[static_cast<std::size_t>(b) * M + i] = r;
  }
}

// Block per chain, 1024 threads = 32 warps; tiles of 1024 vertices; deterministic
// ascending compaction via ballot + per-warp prefix in shared memory.
__global__ void rescan_free_kernel(LSState st, const uint32_t* __restrict__ neg) {
  __shared__ uint32_t wcount[32];
  const int b = static_cast<int>(blockIdx.x);
  const uint16_t* t = ls_tight(st, b);
  const uint32_t* bits = ls_bits(st, b);
  uint32_t* fl = st.fl + static_cast<std::size_t>(b) * st.p.FL;
  const uint32_t FL = static_cast<uint32_t>(st.p.FL);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = ls_lane();
  uint32_t base = 0;   // uniform across the block
  for (int tile = 0; tile < kiss::N; tile += 1024) {
    const int v = tile + static_cast<int>(threadIdx.x);
    bool fr = false;
    if (v < kiss::N) {
      fr = (t[v] == 0) && !((bits[v >> 5] >> (v & 31)) & 1u);
      if (fr && st.p.antipodal) {
        const uint32_t nv = neg[v];
        fr = (t[nv] == 0) && !((bits[nv >> 5] >> (nv & 31)) & 1u);
      }
    }
    const unsigned m = __ballot_sync(LS_FULL, fr);
    if (lane == 0) wcount[warp] = static_cast<uint32_t>(__popc(m));
    __syncthreads();
    uint32_t pre = 0, tot = 0;
    for (int w = 0; w < 32; ++w) {
      const uint32_t c = wcount[w];
      if (w < warp) pre += c;
      tot += c;
    }
    if (fr) {
      const uint32_t p = base + pre + static_cast<uint32_t>(__popc(m & ls_lanemask_lt()));
      if (p < FL) fl[p] = static_cast<uint32_t>(v);
    }
    base += tot;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    st.fl_overflow[b] = base > FL ? 1 : 0;
    st.fl_head[b] = base > FL ? FL : base;
  }
}

__global__ void drain_free_kernel(LSState st, const uint32_t* __restrict__ d_adj, DeviceLeech L, uint32_t max_adds,
                                  uint32_t* adds) {
  const int b = chain_of_warp();
  if (b >= st.p.B) return;
  uint32_t n = 0;
  while (n < max_adds) {
    const uint32_t v = warp_free_pop(st, b, L.neg);
    if (v == LS_NONE) break;
    warp_add(st, b, v, d_adj, L.neg);
    ++n;
  }
  if (adds && ls_lane() == 0) adds[b] = n;
}

}  // namespace

void lsk_apply_moves(const LSState& st, const LSMove* d_moves, int M, const uint32_t* d_adj, const DeviceLeech& L,
                     uint32_t tenure, int32_t* d_ret, cudaStream_t stream) {
  if (M <= 0) return;
  const int blocks = (st.p.B + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  apply_moves_kernel<<<blocks, 32 * WARPS_PER_BLOCK, 0, stream>>>(st, d_moves, M, d_adj, L, tenure, d_ret);
  KISS_CUDA_CHECK(cudaGetLastError());
}

void lsk_rescan_free(const LSState& st, const DeviceLeech& L, cudaStream_t stream) {
  rescan_free_kernel<<<st.p.B, 1024, 0, stream>>>(st, L.neg);
  KISS_CUDA_CHECK(cudaGetLastError());
}

void lsk_drain_free(const LSState& st, const uint32_t* d_adj, const DeviceLeech& L, uint32_t max_adds,
                    uint32_t* d_adds, cudaStream_t stream) {
  const int blocks = (st.p.B + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  drain_free_kernel<<<blocks, 32 * WARPS_PER_BLOCK, 0, stream>>>(st, d_adj, L, max_adds, d_adds);
  KISS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kiss::cuda

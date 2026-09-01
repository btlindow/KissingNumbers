// T3.2a — allocation / initialisation / copy-back / self-check of the chain state
// (docs/design.md §5.2). Kernels here are block-per-chain "bulk" kernels used at init
// and for checks; the incremental moves live in ls_moves.cuh / ls_moves.cu.
#include <curand_kernel.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "ls_moves.cuh"
#include "ls_state.cuh"

namespace kiss::cuda {

// ---------------------------------------------------------------------------
// memory accounting
// ---------------------------------------------------------------------------
LSMemory lsa_report_memory(const LSParams& p) {
  LSMemory m;
  std::size_t pc = 0;
  pc += static_cast<std::size_t>(kiss::N) * sizeof(uint16_t);      // tight
  pc += static_cast<std::size_t>(INS_WORDS) * sizeof(uint32_t);    // inS
  pc += static_cast<std::size_t>(p.SMAX) * sizeof(uint32_t) + 4;   // S + size
  pc += static_cast<std::size_t>(p.FL) * sizeof(uint32_t) + 4 + 1; // fl + head + overflow
  pc += static_cast<std::size_t>(p.TABU) * 2 * sizeof(uint32_t) + 4;   // tabu_v, tabu_exp, head
  pc += 4 + 4;                                                     // iter, best_size
  pc += static_cast<std::size_t>(p.SMAX) * sizeof(uint32_t);       // best_S
  pc += sizeof(curandStatePhilox4_32_10_t);                        // rng
  m.per_chain = pc;
  m.fixed = 0;
  m.total = pc * static_cast<std::size_t>(p.B > 0 ? p.B : 0);
  std::size_t fr = 0, to = 0;
  if (cudaMemGetInfo(&fr, &to) != cudaSuccess) { cudaGetLastError(); fr = to = 0; }
  m.device_free = fr;
  m.device_total = to;
  return m;
}

// ---------------------------------------------------------------------------
// allocation
// ---------------------------------------------------------------------------
namespace {
template <class T>
T* dmalloc(std::size_t count, int fill = 0) {
  T* p = nullptr;
  const std::size_t bytes = count * sizeof(T);
  KISS_CUDA_CHECK(cudaMalloc(&p, bytes));
  KISS_CUDA_CHECK(cudaMemset(p, fill, bytes));
  return p;
}
}  // namespace

LSState lsa_alloc(const LSParams& p, std::size_t margin_bytes) {
  if (p.B <= 0) throw std::runtime_error("lsa_alloc: B must be > 0");
  if (p.N != kiss::N) throw std::runtime_error("lsa_alloc: N must equal kiss::N");
  if (p.SMAX != SMAX) throw std::runtime_error("lsa_alloc: SMAX must equal kiss::cuda::SMAX");
  if (p.FL <= 0) throw std::runtime_error("lsa_alloc: FL must be > 0");
  if (p.TABU <= 0 || p.TABU % 32 != 0) throw std::runtime_error("lsa_alloc: TABU must be a positive multiple of 32");
  if (p.force_cap <= 0 || p.force_cap > LS_FORCE_CAP) throw std::runtime_error("lsa_alloc: force_cap must be in 1..64");
  const LSMemory m = lsa_report_memory(p);
  if (m.total + margin_bytes > m.device_free) {
    throw std::runtime_error("lsa_alloc: chain state needs " + std::to_string(m.total) + " bytes (B=" +
                             std::to_string(p.B) + " x " + std::to_string(m.per_chain) + ") + margin " +
                             std::to_string(margin_bytes) + " but only " + std::to_string(m.device_free) +
                             " of " + std::to_string(m.device_total) + " bytes are free; max B = " +
                             std::to_string(m.max_B(m.device_free, margin_bytes)));
  }
  LSState st;
  st.p = p;
  const std::size_t B = static_cast<std::size_t>(p.B);
  try {
    st.tight = dmalloc<uint16_t>(B * kiss::N);
    st.inS = dmalloc<uint32_t>(B * INS_WORDS);
    st.S = dmalloc<uint32_t>(B * static_cast<std::size_t>(p.SMAX));
    st.size = dmalloc<uint32_t>(B);
    st.fl = dmalloc<uint32_t>(B * static_cast<std::size_t>(p.FL));
    st.fl_head = dmalloc<uint32_t>(B);
    st.fl_overflow = dmalloc<uint8_t>(B);
    st.tabu_v = dmalloc<uint32_t>(B * static_cast<std::size_t>(p.TABU), 0xff);   // LS_NONE
    st.tabu_exp = dmalloc<uint32_t>(B * static_cast<std::size_t>(p.TABU));
    st.tabu_head = dmalloc<uint32_t>(B);
    st.iter = dmalloc<uint32_t>(B);
    st.best_size = dmalloc<uint32_t>(B);
    st.best_S = dmalloc<uint32_t>(B * static_cast<std::size_t>(p.SMAX));
    st.rng = dmalloc<curandStatePhilox4_32_10_t>(B);
  } catch (...) {
    lsa_free(st);
    throw;
  }
  return st;
}

void lsa_free(LSState& st) {
  auto f = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
  f(st.tight); f(st.inS); f(st.S); f(st.size); f(st.fl); f(st.fl_head); f(st.fl_overflow);
  f(st.tabu_v); f(st.tabu_exp); f(st.tabu_head); f(st.iter); f(st.best_size); f(st.best_S); f(st.rng);
  st.p.B = 0;
}

// ---------------------------------------------------------------------------
// init kernels
// ---------------------------------------------------------------------------
namespace {

// block per chain: bitmap from S list; reset the bookkeeping; best = current.
__global__ void init_chain_kernel(LSState st) {
  const int b = static_cast<int>(blockIdx.x);
  uint32_t* bits = st.inS + static_cast<std::size_t>(b) * INS_WORDS;
  for (int i = static_cast<int>(threadIdx.x); i < INS_WORDS; i += static_cast<int>(blockDim.x)) bits[i] = 0u;
  __syncthreads();
  const uint32_t sz = st.size[b];
  const uint32_t* S = st.S + static_cast<std::size_t>(b) * st.p.SMAX;
  uint32_t* bS = st.best_S + static_cast<std::size_t>(b) * st.p.SMAX;
  for (uint32_t i = threadIdx.x; i < sz; i += blockDim.x) {
    const uint32_t v = S[i];
    atomicOr(&bits[v >> 5], 1u << (v & 31));
    bS[i] = v;
  }
  for (int i = static_cast<int>(threadIdx.x); i < st.p.TABU; i += static_cast<int>(blockDim.x)) {
    st.tabu_v[static_cast<std::size_t>(b) * st.p.TABU + i] = LS_NONE;
    st.tabu_exp[static_cast<std::size_t>(b) * st.p.TABU + i] = 0u;
  }
  if (threadIdx.x == 0) {
    st.tabu_head[b] = 0;
    st.iter[b] = 0;
    st.fl_head[b] = 0;
    st.fl_overflow[b] = 0;
    st.best_size[b] = sz;
  }
}

__global__ void seed_rng_kernel(LSState st, uint64_t seed, uint64_t offset) {
  const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (b < st.p.B) curand_init(seed, static_cast<uint64_t>(b), offset, &st.rng[b]);
}

}  // namespace

void lsa_seed_rng(LSState& st, uint64_t seed, uint64_t offset, cudaStream_t stream) {
  const int threads = 128;
  seed_rng_kernel<<<(st.p.B + threads - 1) / threads, threads, 0, stream>>>(st, seed, offset);
  KISS_CUDA_CHECK(cudaGetLastError());
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lsa_init_from_sets(LSState& st, const std::vector<std::vector<uint32_t>>& sets, const DeviceLeech& L,
                        const uint32_t* /*d_adj: unused — tight comes from tightness_full*/, uint64_t seed,
                        cudaStream_t stream) {
  const int B = st.p.B;
  if (static_cast<int>(sets.size()) != B) throw std::runtime_error("lsa_init_from_sets: sets.size() != B");
  std::vector<uint32_t> hS(static_cast<std::size_t>(B) * st.p.SMAX, LS_NONE);
  std::vector<uint32_t> hSize(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    const auto& s = sets[static_cast<std::size_t>(b)];
    if (s.size() > static_cast<std::size_t>(st.p.SMAX))
      throw std::runtime_error("lsa_init_from_sets: chain " + std::to_string(b) + " has > SMAX entries");
    for (uint32_t v : s)
      if (v >= static_cast<uint32_t>(kiss::N))
        throw std::runtime_error("lsa_init_from_sets: chain " + std::to_string(b) + " has a vertex >= N");
    std::copy(s.begin(), s.end(), hS.begin() + static_cast<std::ptrdiff_t>(b) * st.p.SMAX);
    hSize[static_cast<std::size_t>(b)] = static_cast<uint32_t>(s.size());
  }
  KISS_CUDA_CHECK(cudaMemcpyAsync(st.S, hS.data(), hS.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
  KISS_CUDA_CHECK(cudaMemcpyAsync(st.size, hSize.data(), hSize.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                  stream));
  init_chain_kernel<<<B, 256, 0, stream>>>(st);
  KISS_CUDA_CHECK(cudaGetLastError());
  tightness_full(L, st.S, st.size, B, st.tight, stream);
  lsk_rescan_free(st, L, stream);
  seed_rng_kernel<<<(B + 127) / 128, 128, 0, stream>>>(st, seed, 0);
  KISS_CUDA_CHECK(cudaGetLastError());
  KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// copy-back
// ---------------------------------------------------------------------------
namespace {
template <class T>
void d2h(std::vector<T>& out, const T* src, std::size_t count) {
  out.resize(count);
  if (count) KISS_CUDA_CHECK(cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost));
}
template <class T>
T d2h1(const T* src) {
  T v{};
  KISS_CUDA_CHECK(cudaMemcpy(&v, src, sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
}  // namespace

LSHostChain lsa_download(const LSState& st, int b, bool with_tight) {
  if (b < 0 || b >= st.p.B) throw std::runtime_error("lsa_download: chain index out of range");
  LSHostChain h;
  const std::size_t bb = static_cast<std::size_t>(b);
  h.size = d2h1(st.size + b);
  h.iter = d2h1(st.iter + b);
  h.best_size = d2h1(st.best_size + b);
  h.fl_head = d2h1(st.fl_head + b);
  h.tabu_head = d2h1(st.tabu_head + b);
  h.fl_overflow = d2h1(st.fl_overflow + b) != 0;
  d2h(h.S, st.S + bb * st.p.SMAX, std::min<std::size_t>(h.size, static_cast<std::size_t>(st.p.SMAX)));
  d2h(h.best_S, st.best_S + bb * st.p.SMAX, std::min<std::size_t>(h.best_size, static_cast<std::size_t>(st.p.SMAX)));
  d2h(h.fl, st.fl + bb * st.p.FL, std::min<std::size_t>(h.fl_head, static_cast<std::size_t>(st.p.FL)));
  d2h(h.tabu_v, st.tabu_v + bb * st.p.TABU, static_cast<std::size_t>(st.p.TABU));
  d2h(h.tabu_exp, st.tabu_exp + bb * st.p.TABU, static_cast<std::size_t>(st.p.TABU));
  if (with_tight) {
    d2h(h.tight, st.tight + bb * kiss::N, static_cast<std::size_t>(kiss::N));
    d2h(h.inS, st.inS + bb * INS_WORDS, static_cast<std::size_t>(INS_WORDS));
  }
  return h;
}

// ---------------------------------------------------------------------------
// self-check
// ---------------------------------------------------------------------------
namespace {

struct LSCheckDev {
  uint32_t tight_mismatch, list_bad, bitmap_bad, conflict_pairs, antipodal_bad, fl_bad, fl_stale, fl_missing;
};

__device__ __forceinline__ uint32_t block_sum(uint32_t v, uint32_t* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(LS_FULL, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  uint32_t r = 0;
  for (unsigned i = 0; i < (blockDim.x + 31) / 32; ++i) r += sh[i];
  __syncthreads();
  return r;
}

// block per chain c (chain b = first + c); ref = recomputed tightness of that chain
__global__ void check_kernel(LSState st, int first, const uint16_t* __restrict__ ref, const uint32_t* __restrict__ neg,
                             LSCheckDev* out) {
  __shared__ uint32_t sh[32];
  __shared__ uint32_t sh_word[INS_WORDS];   // 24.6 KB: bitmap of the free-list contents (dedup)
  const int c = static_cast<int>(blockIdx.x);
  const int b = first + c;
  const uint16_t* t = st.tight + static_cast<std::size_t>(b) * kiss::N;
  const uint16_t* r = ref + static_cast<std::size_t>(c) * kiss::N;
  const uint32_t* bits = st.inS + static_cast<std::size_t>(b) * INS_WORDS;
  const uint32_t* S = st.S + static_cast<std::size_t>(b) * st.p.SMAX;
  const uint32_t sz = st.size[b];
  const uint32_t* fl = st.fl + static_cast<std::size_t>(b) * st.p.FL;
  const uint32_t head = st.fl_head[b];
  const uint32_t headc = head < static_cast<uint32_t>(st.p.FL) ? head : static_cast<uint32_t>(st.p.FL);

  for (int i = static_cast<int>(threadIdx.x); i < INS_WORDS; i += static_cast<int>(blockDim.x)) sh_word[i] = 0u;
  __syncthreads();

  uint32_t mism = 0;
  for (int v = static_cast<int>(threadIdx.x); v < kiss::N; v += static_cast<int>(blockDim.x)) mism += (t[v] != r[v]);

  uint32_t pop = 0;
  for (int i = static_cast<int>(threadIdx.x); i < INS_WORDS; i += static_cast<int>(blockDim.x)) pop += __popc(bits[i]);

  uint32_t lbad = 0, tsum = 0, abad = 0;
  const uint32_t szc = sz < static_cast<uint32_t>(st.p.SMAX) ? sz : static_cast<uint32_t>(st.p.SMAX);
  for (uint32_t i = threadIdx.x; i < szc; i += blockDim.x) {
    const uint32_t v = S[i];
    if (v >= static_cast<uint32_t>(kiss::N)) { ++lbad; continue; }
    if (!((bits[v >> 5] >> (v & 31)) & 1u)) ++lbad;
    tsum += r[v];
    if (st.p.antipodal) {
      const uint32_t nv = neg[v];
      if (!((bits[nv >> 5] >> (nv & 31)) & 1u)) ++abad;
    }
  }
  if (threadIdx.x == 0 && sz > static_cast<uint32_t>(st.p.SMAX)) ++lbad;

  uint32_t fbad = 0, fstale = 0;
  for (uint32_t i = threadIdx.x; i < headc; i += blockDim.x) {
    const uint32_t v = fl[i];
    if (v >= static_cast<uint32_t>(kiss::N)) { ++fbad; continue; }
    const bool fr = (r[v] == 0) && !((bits[v >> 5] >> (v & 31)) & 1u);
    if (!fr) ++fstale;
    atomicOr(&sh_word[v >> 5], 1u << (v & 31));
  }
  if (threadIdx.x == 0 && head > static_cast<uint32_t>(st.p.FL)) ++fbad;
  __syncthreads();
  uint32_t fmiss = 0;
  for (int v = static_cast<int>(threadIdx.x); v < kiss::N; v += static_cast<int>(blockDim.x)) {
    const bool fr = (r[v] == 0) && !((bits[v >> 5] >> (v & 31)) & 1u);
    if (fr && !((sh_word[v >> 5] >> (v & 31)) & 1u)) ++fmiss;
  }

  mism = block_sum(mism, sh);
  pop = block_sum(pop, sh);
  lbad = block_sum(lbad, sh);
  tsum = block_sum(tsum, sh);
  abad = block_sum(abad, sh);
  fbad = block_sum(fbad, sh);
  fstale = block_sum(fstale, sh);
  fmiss = block_sum(fmiss, sh);
  if (threadIdx.x == 0) {
    LSCheckDev o;
    o.tight_mismatch = mism;
    o.list_bad = lbad;
    o.bitmap_bad = pop > sz ? pop - sz : sz - pop;
    o.conflict_pairs = tsum / 2;
    o.antipodal_bad = abad;
    o.fl_bad = fbad;
    o.fl_stale = fstale;
    o.fl_missing = fmiss;
    out[c] = o;
  }
}

}  // namespace

std::vector<LSCheck> lsa_check(const LSState& st, const DeviceLeech& L, cudaStream_t stream) {
  const int B = st.p.B;
  const int CH = 64;
  uint16_t* scratch = nullptr;
  LSCheckDev* d_out = nullptr;
  KISS_CUDA_CHECK(cudaMalloc(&scratch, static_cast<std::size_t>(CH) * kiss::N * sizeof(uint16_t)));
  KISS_CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(CH) * sizeof(LSCheckDev)));
  std::vector<LSCheck> res(static_cast<std::size_t>(B));
  std::vector<LSCheckDev> h(static_cast<std::size_t>(CH));
  try {
    for (int first = 0; first < B; first += CH) {
      const int n = std::min(CH, B - first);
      tightness_full(L, st.S + static_cast<std::size_t>(first) * st.p.SMAX, st.size + first, n, scratch, stream);
      check_kernel<<<n, 1024, 0, stream>>>(st, first, scratch, L.neg, d_out);
      KISS_CUDA_CHECK(cudaGetLastError());
      KISS_CUDA_CHECK(cudaMemcpyAsync(h.data(), d_out, static_cast<std::size_t>(n) * sizeof(LSCheckDev),
                                      cudaMemcpyDeviceToHost, stream));
      KISS_CUDA_CHECK(cudaStreamSynchronize(stream));
      for (int c = 0; c < n; ++c) {
        LSCheck& o = res[static_cast<std::size_t>(first + c)];
        const LSCheckDev& d = h[static_cast<std::size_t>(c)];
        o.tight_mismatch = d.tight_mismatch;
        o.list_bad = d.list_bad;
        o.bitmap_bad = d.bitmap_bad;
        o.conflict_pairs = d.conflict_pairs;
        o.antipodal_bad = d.antipodal_bad;
        o.fl_bad = d.fl_bad;
        o.fl_stale = d.fl_stale;
        o.fl_missing = d.fl_missing;
      }
    }
  } catch (...) {
    cudaFree(scratch);
    cudaFree(d_out);
    throw;
  }
  KISS_CUDA_CHECK(cudaFree(scratch));
  KISS_CUDA_CHECK(cudaFree(d_out));
  return res;
}

}  // namespace kiss::cuda

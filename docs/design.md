# Design notes: data formats, APIs and the GPU engine

Reference documentation for the code in this repository: the machine the work was done on,
the byte-exact data formats and API conventions the tools share, and the design of the GPU
local-search engine. Section numbers are preserved from the original design document, because
source comments throughout the tree cite them (`§2.2`, `§2.4`, `§5.3`, ...).

For *what was found*, see [`docs/reports/`](reports/) and the summary paper
[`docs/handover/findings.pdf`](handover/findings.pdf). For *how to re-run it*, see
[`REPRODUCE.md`](../REPRODUCE.md).

## 1. Machine envelope (surveyed 2026-08-25)

| Item | Value | Consequence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop (GA104M, Ampere), **sm_86**, 40 SMs, 8 GB (7840 MiB usable), 80 W cap | Target `-arch=sm_86` only. ~7.4 GB of VRAM is the real budget. Laptop thermals: expect clocks to sag on multi-hour runs; monitor with `nvidia-smi`. |
| Driver | 580.119.02 (CUDA 13.0 capable) | Either toolkit below runs. |
| CUDA toolkits | 12.8 at `/usr/local/cuda-12.8` (**this `nvcc` is on PATH**) and 13.0 at `/usr/local/cuda-13.0` (`/usr/local/cuda` symlinks here) | **Pin 12.8 explicitly in CMake** (`CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`) so PATH vs. symlink never disagree. Both were tested compiling+running an sm_86 kernel. |
| CUDA libs available | cuBLAS/cuBLASLt, cuRAND, cuSPARSE, CUB, Thrust; `compute-sanitizer`, `nsys`, `ncu` | No need for cuBLAS for the core (see §5: fused int8 `dp4a` kernels are simpler and exact); cuRAND device API (Philox) for per-chain RNG; CUB for block scans/compaction. |
| Host compiler | gcc 11.4 (no clang) | C++17. |
| CPU / RAM | Ryzen 9 5900HS, 16 threads, 15 GB RAM (only ~7 GB free at survey time) | Never hold the 3.6 GB adjacency table plus copies in host RAM; build it on the GPU and stream to disk. CPU work (LPs, swap search, verifiers) uses OpenMP/numpy. |
| Disk | 629 GB free on NVMe | Adjacency file (3.6 GB) and run logs are fine. |
| Python | 3.10.12, numpy 1.21.5, scipy 1.8.0 (has `linprog(method="highs")`); **no** cvxpy/torch/cupy/numba; `python3 -m venv` lacks ensurepip on this box — `scripts/setup_venv.sh` falls back to `virtualenv` (works), pip 25.3 | Independent verifiers and LPs in Python are fine. Create `.venv` and pip-install `sympy` (exact rationals, permutation groups) and, only if T2.3 triggers, an SDP solver. No Python-side GPU: all GPU code is CUDA C++. |
| Build tools | cmake 3.25, ninja 1.10, make, git 2.34 | CMake + Ninja, ctest for tests. |
| Network | github.com reachable; PackingStar and Kallal–Kan–Wang repos resolve | T1.3 can fetch fixtures. |

**Design decisions forced by the envelope**

1. **Adjacency lives on the GPU as `uint32[196560][4600]` = 3.62 GB.** It fits in 8 GB with ~3.8 GB left for chain state. Fallback if VRAM gets tight: 3×21-bit packing in `uint64` (2.41 GB). Never materialise it on the host in full.
2. **Hot loop is incremental (neighbor-list tightness updates), not GEMM.** A full recompute of tightness for one chain is ≈ 196560 × 500 × 6 ≈ 6×10^8 `dp4a`; an incremental add/remove is 4600 reads + 4600 read-modify-writes. The fused GEMM-style kernel is still built (T1.5) for initialisation, self-checks, W2a, and the scheme computation.
3. **Chain count ≈ 2k–8k, one warp per chain.** 40 SMs × 48 resident warps = 1920 warps; a few thousand chains saturate the card while keeping per-chain state (≈ 0.4–0.6 MB) within budget.
4. **All decisions are integer.** int8 coordinates, int32 dot products via `__dp4a`. Floating point appears only in LP solvers (whose certificates are then re-verified in exact rational arithmetic) and in the float "sanity" half of the dimension verifiers.

---

## 2. Contracts (read before starting any task)

### 2.1 Repository layout

```
kissing/
  CMakeLists.txt            # T0.1
  PLAN.md  README.md
  cmake/                    # toolchain/pin helpers
  include/kiss/             # public headers (API in §2.3) — frozen after T1.2 lands
  src/                      # CPU C++ library: golay, leech, io, verify, scheme, frames, group
  cuda/                     # CUDA library: packed vectors, conflict kernels, adjacency, localsearch
  tools/                    # one main() per CLI executable (gen_leech, verify_s, build_adj, swapsearch, gpu_mis, ...)
  python/                   # INDEPENDENT implementations (verifiers, LPs) — must not import from src/
  tests/                    # ctest targets + pytest
  data/                     # generated canonical data + fixtures (large generated files gitignored)
  docs/reports/             # one report per task: Tx.y.md
  runs/                     # logs, checkpoints, found sets (gitignored)
```

File ownership: a task may create/modify only the files its card lists as deliverables plus tests for them. Shared headers in `include/kiss/` are owned by T1.2; other tasks code against §2.3 and may propose header changes in their report, not apply them silently.

### 2.2 Data formats (canonical, byte-exact)

| Name | Path | Format |
|---|---|---|
| Minimal vectors (canonical order) | `data/leech_min.i8` | raw int8, 196560 rows × 24 columns, row-major. Rows sorted **lexicographically ascending as signed integers, coordinate 0 most significant**. This order defines the vertex index `0..196559` used everywhere. |
| Same, human-readable | `data/leech_min.txt` | one row per line, 24 space-separated ints, same order. |
| Negation table | `data/neg.u32` | uint32[196560], `neg[i]` = index of `−C[i]`. |
| Packed vectors for dp4a | `data/leech_packed.u32` | uint32[6][196560] (structure-of-arrays). Word `w` of vector `i` holds coordinates `4w..4w+3` as four int8 bytes, little-endian (byte 0 = coordinate `4w`). Enables `__dp4a` in 6 instructions per inner product. |
| Adjacency (conflict graph) | `data/adj.u32` | raw uint32, 196560 rows × 4600 columns; row `i` = indices `j` with `⟨C[i],C[j]⟩ = 16`, **sorted ascending**. 3,616,704,000 bytes. Gitignored, regenerated by `build_adj`. |
| Adjacency checksum | `data/adj.sha256` | committed, so a regenerated file can be checked. |
| A vector set S | `*.txt` | one vector per line, 24 ints; lines starting `#` are comments; blank lines ignored. Vectors must have squared norm 32. Order is free. (This is the §5 certificate format from the README.) |
| A family S_1..S_k | directory with `S_01.txt … S_k.txt` + `family.json` | json: `{"dim": n, "sets": [...], "T": {...}, "extra": [...]}` — precise schema fixed by T4.3. |
| Found/record sets | `runs/found/S_<size>_<utc-timestamp>_<chain>.txt` | written immediately by the search engine, before anything else. |

Inner product classes (integer scaling): `{-32,-16,-8,0,8,16,32}`; class index `0..6` in that order. Conflict ⇔ dot == 16. Independent set ⇔ all off-diagonal Gram entries ≤ 8.

### 2.3 C++ API (namespace `kiss`, header-only declarations in `include/kiss/`)

```cpp
// kiss/types.h
constexpr int N = 196560, DIM = 24, DEG = 4600;
using Vec = std::array<int8_t, DIM>;
inline int dot(const Vec& a, const Vec& b);                     // exact int32

// kiss/golay.h
std::vector<uint32_t> golay_codewords();                        // 4096 words as 24-bit masks (bit i = coordinate i)
std::array<int, 25> golay_weight_distribution(const std::vector<uint32_t>&);

// kiss/leech.h
struct Leech {
  std::vector<Vec> C;              // canonical order (§2.2)
  std::vector<uint32_t> neg;       // neg[i] = index of -C[i]
  int32_t index_of(const Vec&) const;   // binary search; -1 if absent
  bool is_lattice_vector(const std::array<int,DIM>&) const;    // membership test from README §1.2 (any norm)
};
Leech generate_leech();                                          // deterministic; builds from golay_codewords()
Leech load_leech(const std::filesystem::path& data_dir);         // reads leech_min.i8 + neg.u32, validates size

// kiss/io.h
std::vector<Vec> read_set(const std::filesystem::path&);         // S text format
void write_set(const std::filesystem::path&, const std::vector<Vec>&, const std::string& header_comment = "");

// kiss/verify.h
struct VerifyResult { bool ok; std::size_t size; std::string message; };
VerifyResult verify_independent(const Leech&, const std::vector<Vec>&);   // norms, membership, distinct, Gram ≤ 8
std::vector<uint16_t> tightness_cpu(const Leech&, const std::vector<uint32_t>& S_idx);  // reference, O(N·|S|)

// kiss/adjacency.h
class Adjacency {                 // memory-mapped data/adj.u32 on host, or device pointer on GPU
 public:
  const uint32_t* row(uint32_t v) const;   // DEG sorted entries
  bool adjacent(uint32_t u, uint32_t v) const;  // binary search in row(u)
};
```

CUDA-side contract (`cuda/kiss_cuda.h`, C-linkage-free C++):

```cpp
namespace kiss::cuda {
struct DeviceLeech { const uint32_t* packed /*[6][N]*/; const uint32_t* neg; };
DeviceLeech upload_leech(const Leech&);
// T1.5: tightness for B chains, each with S_b (≤ SMAX entries). Out: tight[B][N] uint16.
void tightness_full(const DeviceLeech&, const uint32_t* d_S, const uint32_t* d_Ssize, int B, uint16_t* d_tight, cudaStream_t);
// T1.6
void build_adjacency(const DeviceLeech&, uint32_t* d_adj_rows_chunk, uint32_t first_row, uint32_t nrows, cudaStream_t);
}
```

### 2.4 Conventions

- C++17, `-O3 -Wall -Wextra -Werror` for host; `nvcc -O3 -arch=sm_86 -lineinfo --expt-relaxed-constexpr`. Debug config adds `-G` and runs under `compute-sanitizer`.
- Every CLI tool prints a one-line machine-readable summary at exit (`RESULT key=value ...`) so scripts can parse it.
- Every generated artefact is deterministic given the seed; seeds and git hash are logged in every run directory.
- **No record claim from GPU state alone.** The chain of custody is: device flag → host copies S → `write_set` to `runs/found/` → `tools/verify_s` → `python/verify_S.py` (independent) → only then log "RECORD". Both verifiers must print the same size.
- Python code in `python/` is an **independent re-implementation** — it must not read the C++ generator's output as ground truth except when explicitly cross-checking. It regenerates the Golay code and minimal vectors itself.
- Tests: `ctest` (C++/CUDA) and `pytest python/tests` must pass at the end of every task. GPU tests are tagged `gpu` and skipped gracefully when no device is present.
- Each task ends with `docs/reports/Tx.y.md`: what was built, how it was tested (with pasted outputs of the acceptance tests), timing numbers, and anything that contradicts README/PLAN.

## 5. GPU engine design notes

### 5.1 Fused tightness kernel (T1.5)

- Inputs: packed C (`uint32[6][N]`, SoA so a warp's 32 candidates read 6 coalesced 128-byte lines), the chain's S as packed words (`uint32[6][SMAX]`, ≤ 24 KB) copied to shared memory once per block.
- Thread ↔ candidate `v`; loop `s` over S: `acc = __dp4a(w0,s0,0); acc = __dp4a(w1,s1,acc); … ; cnt += (acc == 16)`. Six int8 MACs per pair, exact in int32 (|dot| ≤ 32). Unroll by 4 over `s`. Output `tight[b][v]` (uint16; max possible = |S| ≤ 1024).
- Grid: `(N/256 tiles) × B`. Per chain ≈ N·|S|·6 dp4a ≈ 6×10^8 for |S| = 500 — ~1 ms-class per chain, i.e. **fine for initialisation and periodic self-checks, too slow to be the inner loop for thousands of chains.**
- Variant for T2.2: instead of `cnt += (acc==16)`, index a 7-bin histogram by `(acc+32)/8` combined with the precomputed class of `(x,z)`; one launch per `y`, or batch 64 `y`'s per block.

### 5.2 Chain state (T3.2a) — structure-of-arrays, B chains

| Array | Type / shape | Size for B = 4096 |
|---|---|---|
| `tight` | uint16 [B][N] | 1.61 GB |
| `inS` bitmap | uint32 [B][N/32] | 100 MB |
| `S` list | uint32 [B][SMAX=1024] + `size` | 16 MB |
| free-list (lazy) | uint32 [B][FL=4096] + head/overflow flag | 67 MB |
| tabu ring | uint32 [B][32] vertices + uint32 [B][32] expiry | 1 MB |
| RNG | Philox4x32-10 state [B] | small |
| best-ever | uint32 [B][SMAX] + size | 16 MB |
| adjacency (shared) | uint32 [N][4600] | 3.62 GB |
| packed C, neg | | 5.5 MB |
| **Total** | | **≈ 5.4 GB** (of ~7.4 usable) |

B = 8192 pushes `tight` to 3.2 GB → ≈ 7.1 GB total: possible but tight; T3.2e decides. If needed: uint8 tightness with saturation at 255 (a vertex with ≥ 255 conflicts is never a candidate anyway, but then decrements must be guarded — simplest is to keep uint16 and cap B).

### 5.3 Incremental updates and the lazy free-list

- `add(v)`: warp lanes stride `row(v)` (4600 entries, 144 iterations of 32 coalesced loads); `tight[b][n]++` — no atomics: only one warp ever touches chain b's arrays, and within a warp the 4600 targets are distinct (rows have no duplicates).
- `remove(v)`: `tight[b][n]--`; lanes where the new value is 0 and `!inS[n]` push `n` to the free-list via warp ballot + one atomicAdd on the head per iteration (or a warp-prefix and a single `head` increment by lane 0). On overflow set `needs_rescan`.
- Free-list entries may go stale (a later `add` raised the vertex's tightness) → **validate on pop** (`tight==0 && !inS`), discard otherwise. Amortised cost of finding a free vertex becomes O(1) instead of an O(N) scan.
- Full rescan (`block` per chain, CUB block-compaction over the 196560 tightness values) every R iterations or on overflow. 192–384 KB per chain per rescan.
- Adjacency query `adjacent(u,w)` for the (1,2)-swap pair test: 6 dp4a on packed C — cheaper than binary-searching a 4600-row.

### 5.4 Per-chain iteration (T3.2b), ARW-flavoured ILS

```
repeat K times:
  while free-list non-empty: v = pop(); if valid(v): add(v)          // greedy completion
  if try_12_swap(): continue                                          // ARW 2-improvement: remove x, add u,w ⊂ L_x with u≁w
  perturb(): choose v with tight[v] ∈ {1,2} (not tabu), remove its conflicts (→ tabu), add v; drain free-list
  accept/track: if |S| ≥ best: record; stall counter; if stall > limit → flag for host restart
  if |S| ≥ target: set global flag, copy S to out slot (host verifies)
```

- `try_12_swap`: iterate candidates `x ∈ S` starting at a random offset; for each, lanes stride `row(x)` and ballot-compact the vertices with `tight==1` and not tabu into shared memory (cap 256 per warp); then all-pairs test with dp4a (≤ 32k pairs, ~10 µs worst case). Because a tightness-1 neighbour of `x` conflicts **only** with `x`, any non-adjacent pair `{u,w} ⊂ L_x` gives a valid swap. Give up after `max_x_tries` per iteration (tunable; ARW checks all, we cap for lockstep fairness).
- Antipodal mode: every add/remove is applied to both `v` and `neg[v]` (they are never adjacent). `L_x` then consists of pairs, and the pair test uses the class representatives.
- Divergence: chains diverge from each other freely (warp = chain). Within a warp, the swap search and neighbour loops are naturally lane-parallel. Use `__syncwarp()` around shared-memory phases; never `__syncthreads()` inside chain logic if blocks hold multiple chains — or run 1 warp per block (occupancy still fine on Ampere with small blocks; T3.2e measures).
- Expected cost: dominated by adjacency streaming ≈ 18.4 KB per add/remove. At ~400 GB/s that is ~2×10^7 moves/s aggregate, i.e. thousands of ILS iterations per second per chain across 4k chains.

### 5.5 Seeding, restarts, elite pool (T3.2c)

- Seeds: (a) the 496 minus a random 5–15% (forces exploration near the record), (b) the 488 likewise, (c) random-greedy maximal sets (~238 — far from the frontier; keep a minority for diversity), (d) best-ever sets from previous runs, (e) images `g·496` under random automorphisms once T4.1 exists (equivalent under symmetry but different in coordinates — useful for antipodal-off runs).
- Restart a chain when `stall > limit`: re-seed from the elite pool with random deletions; keep per-chain RNG streams independent (Philox subsequence = chain id, offset = restart count).
- Log every set ≥ 490 (README asks for it) with its `tight` histogram; they characterise the local structure near the record.

---


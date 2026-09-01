# Lattice construction, data artifacts, and verification

This document covers the foundation layer of the project: the extended binary Golay code
`G24`, the construction and canonical ordering of the 196560 minimal vectors of the Leech
lattice `Λ24`, the data files those produce, the 3.6 GB adjacency (60-degree neighbour) table,
the fused GPU tightness kernel, the independent verifiers, and the external record data
(PackingStar and Kallal–Kan–Wang) together with the coordinate permutation that maps it into
our conventions.

Throughout, minimal vectors are in the **√8-integer scaling**: integer coordinate vectors of
squared norm 32, so that two of them subtend 60° exactly when their integer inner product is
16, and are "independent" (no closer than 60°) exactly when their inner product is at most 8.
The kissing configuration in dimension 24 is the whole set of 196560; the objects of interest
in higher dimensions are *independent* subsets `S ⊂ C`, i.e. sets whose pairwise inner products
never exceed 8. See `README.md` for the document index and `REPRODUCE.md` for how to re-run
everything.

Related documents: `02-upper-bounds.md` (LP/SDP upper bounds), `03-search-for-497.md`
(searches for a 497-element independent set), `04-structure-of-the-496.md`,
`05-lifting-template-and-families.md`, `06-record-and-priority.md`.

---

## 1. Build system, environment pin, and conventions

### 1.1 Hardware and toolchain

All timings in this document were measured on a single machine:

| item | value |
|---|---|
| CPU | AMD Ryzen 9 5900HS, 8 cores / 16 threads |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, `sm_86` (GA104), 40 SMs |
| GPU memory | 7.66 GiB = 8,220,901,376 bytes as reported by cudart (nvidia-smi reports the 8192 MiB physical size; cudart reports slightly less. The "~7.8 GB" figure sometimes quoted is the decimal-GB form of the same number: 8.22e9 B = 8.22 GB = 7.66 GiB.) |
| driver / runtime | 13000 / 12080 (driver 580) |
| CUDA | 12.8, pinned to `/usr/local/cuda-12.8/bin/nvcc`, identified as NVIDIA 12.8.93 |
| host compiler | gcc 11.4.0, reached through ccache as `/usr/lib/ccache/c++` (this is simply what CMake finds on `PATH`; set `CMAKE_CXX_COMPILER=/usr/bin/g++` to bypass ccache) |
| Python | 3.10.12 in `.venv`, numpy 2.2.6, scipy 1.15.3, sympy 1.14.0, pytest 9.1.1 |

The system Python outside the venv has numpy 1.21.5 / scipy 1.8.0 / pytest 8.3.3; the pure-Python
parts of the reference implementation run on either. `linprog(method="highs")` is available in
both scipy versions.

### 1.2 CMake layout

| File | Purpose |
|---|---|
| `CMakeLists.txt` | CMake ≥ 3.25, languages CXX + CUDA, C++17 / CUDA 17. Targets: static lib `kiss` (`src/*.cpp`, headers in `include/kiss/`, links OpenMP), static lib `kiss_cuda` (`cuda/*.cu`, links `kiss`, `CUDA::cudart`, `CUDA::curand`), one executable per `tools/*.cpp` (output in `build/<preset>/tools/`), tests via `tests/CMakeLists.txt`. All source lists use `file(GLOB ... CONFIGURE_DEPENDS)`, so new files are picked up without editing CMake. |
| `CMakePresets.json` | Presets `release`, `debug`, `sanitize` (configure/build/test). Ninja generator, `binaryDir=build/<preset>`, cache vars `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`, `CMAKE_CUDA_ARCHITECTURES=86`, `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. |
| `cmake/KissOptions.cmake` | Options `KISS_ENABLE_GPU_TESTS` (ON), `KISS_WERROR` (ON), `KISS_SANITIZE` (OFF; set by the sanitize preset). Warns if nvcc is not the pinned 12.8 or the arch list lacks 86. `kiss_git_hash()` helper. |
| `cmake/KissFlags.cmake` | `kiss_apply_flags(target)`: host `-Wall -Wextra -Werror`, release `-O3`, debug `-O0 -g`; CUDA `--expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra,-Werror -Werror=all-warnings`, release `-O3 -lineinfo`, debug `-O0 -g -G`; sanitize adds host ASan+UBSan, also through nvcc's host pass. |
| `cmake/version_info.h.in` | Generates `kiss/version_info.h` with version, git hash, build type, CUDA compiler/version/arch list, for run-directory logging. |
| `include/kiss/version.h`, `src/version.cpp`, `cuda/version.cu` | `kiss::version()`, `kiss::git_hash()`, `kiss::cuda::device_count()`, `kiss::cuda::device_name()`. |
| `tools/kiss_info.cpp` | Prints version + device name; registered as a CPU ctest (`kiss_info`) to prove the tools → `kiss` + `kiss_cuda` link path. |
| `tests/smoke_gpu.cu` | GPU smoke test: device 0 name, compute capability, total memory, SM count; runs `y = 3x + y` on 2^20 ints and verifies on the host; asserts `sm_86`. ctest name `smoke_gpu`, label `gpu`, `SKIP_RETURN_CODE 77` when no device, pass regex `RESULT .*ok=1`. |
| `.gitignore` | `build/ runs/ data/adj.u32 data/*.i8 data/*.u32 data/external/ .venv/ __pycache__/` plus pytest cache and editor noise. |
| `python/requirements.txt` | numpy, scipy, sympy, pytest. |
| `scripts/setup_venv.sh` | Idempotent `.venv` creation + `pip install -r requirements.txt`; prints a `RESULT` line. |

Later tasks register their tests through per-task files under `tests/tasks/` (e.g.
`tests/tasks/T1_4.cmake`, `T1_5.cmake`, `T1_6.cmake`) using a `kiss_add_test` helper, so
`tests/CMakeLists.txt` itself is not edited repeatedly. The two earliest tests
(`test_golay`, `test_leech`) predate that helper and are registered directly in
`tests/CMakeLists.txt` — that file does not glob its sources, so registering a ctest target
requires editing it.

### 1.3 The `RESULT` line convention

Every tool and every test ends with a single machine-readable line of the form

```
RESULT key=value key=value ...
```

with `ok=1` on success. ctest targets use `RESULT .*ok=1` as their pass regex. This convention
is used uniformly across the C++ tools, the CUDA tests and the Python scripts, and is what the
pasted acceptance outputs throughout this document quote.

### 1.4 Commands

```
scripts/setup_venv.sh && source .venv/bin/activate
cmake --preset release && cmake --build build/release && ctest --test-dir build/release --output-on-failure
cmake --preset debug    && cmake --build build/debug    && ctest --test-dir build/debug
cmake --preset sanitize && cmake --build build/sanitize && ctest --test-dir build/sanitize
ctest --test-dir build/release -L gpu -V        # only GPU-labelled tests, verbose
.venv/bin/python -m pytest python/tests
```

Equivalent preset forms: `cmake --build --preset release`, `ctest --preset release`.

### 1.5 Environment notes a reproducer needs

1. **`python3 -m venv` does not work on this machine.** The system Python 3.10.12 has no
   `ensurepip` (`python3-venv` is not installed): a venv *is* created, but without pip.
   `scripts/setup_venv.sh` therefore implements a fallback chain — first `python3 -m venv`,
   then `python3 -m virtualenv` (virtualenv 21.6.1 from the user site), and as a last resort
   `venv --without-pip` followed by `pip --python .venv/bin/python install pip` using the
   user-site pip 25.3. The venv actually used here was created by virtualenv.
2. **nvcc's `-Xcompiler` splits on commas**, so sanitizer flags must be passed as separate
   `-Xcompiler=` options; `-fsanitize=address,undefined` breaks the build. Documented in
   `cmake/KissFlags.cmake`.
3. **`-Werror` is applied to CUDA as well** (`-Werror=all-warnings` plus `-Xcompiler=-Werror`).
   Note that `-Werror=all-warnings` flags *unused kernels* (nvcc diagnostic #177-D). Disable
   per-configure with `-DKISS_WERROR=OFF` if a header warning cannot be fixed.
4. Sanitize-preset tests run with `ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0`, because
   cudart maps a huge virtual range at initialisation that collides with ASan's protected
   shadow gap.
5. `data/external/` is gitignored (large clones); the small fixtures are copied out of it
   into `data/`.
6. GPU-labelled tests exit 77 (ctest "skipped") when no CUDA device is visible, so the suite
   is runnable on a CPU-only machine.

### 1.6 Baseline acceptance output

```
$ cmake --preset release
-- The CUDA compiler identification is NVIDIA 12.8.93
-- CUDA compiler : /usr/local/cuda-12.8/bin/nvcc (12.8.93)
-- CUDA archs    : 86
-- C++ compiler  : /usr/lib/ccache/c++ (11.4.0)
-- Build type    : Release
-- GPU tests     : ON   sanitize: OFF   werror: ON
-- Found CUDAToolkit: /usr/local/cuda-12.8/include (found suitable version "12.8.93", minimum required is "12.8")
-- Found OpenMP_CXX: -fopenmp (found version "4.5")

$ ctest --test-dir build/release -L gpu -V | grep -E "^[0-9]+: "
1: device 0        : NVIDIA GeForce RTX 3070 Laptop GPU
1: compute cap     : sm_86
1: total memory    : 7.66 GiB (8220901376 bytes)
1: SMs             : 40
1: driver/runtime  : 13000 / 12080
1: kernel check    : 1048576 elements, 0 mismatches
1: RESULT ok=1 device="NVIDIA GeForce RTX 3070 Laptop GPU" cc=sm_86 mem_gb=7.66 sms=40 n=1048576 mismatches=0 driver=13000 runtime=12080

$ grep -o "nvcc.*smoke_gpu.cu.o$" build/release/build.ninja
  FLAGS = -O3 -DNDEBUG -fPIC -Wall -Wextra -Werror -O3 -fopenmp -std=c++17
  FLAGS = -O3 -DNDEBUG --generate-code=arch=compute_86,code=[compute_86,sm_86] -Xcompiler=-fPIE --expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra,-Werror -O3 -lineinfo -Werror=all-warnings -std=c++17
```

Debug CUDA flags are `-O0 -g -G`. The no-device skip path:

```
$ CUDA_VISIBLE_DEVICES="" ctest --test-dir build/debug -L gpu -V
1: smoke_gpu: no CUDA device (no CUDA-capable device is detected) — skipping
1: RESULT ok=1 skipped=1 devices=0
1/1 Test #1: smoke_gpu ........................***Skipped   0.02 sec
```

venv setup:

```
$ scripts/setup_venv.sh
setup_venv: <repo>/.venv already exists
python 3.10.12 numpy 2.2.6 scipy 1.15.3 sympy 1.14.0 pytest 9.1.1
RESULT ok=1 venv=<repo>/.venv
```

Timing: clean release configure ≈ 3 s, build ≈ 10 s, ctest 0.44 s for the two baseline tests.
`setup_venv.sh` first run ≈ 40 s (downloads), subsequent runs ≈ 2 s.

---

## 2. The extended binary Golay code G24

The Golay code was implemented **twice, independently**: a C++ implementation
(`include/kiss/golay.h`, `src/golay.cpp`) and a pure-Python reference
(`python/kiss_ref/golay.py`). The two were written without reading each other's source; the
only information shared was the specification and a SHA-256 fingerprint of the sorted codeword
list, which the C++ test compares against. This is the pattern used throughout the project:
one fast implementation, one independent reference, and an equality check between them.

### 2.1 Construction and coordinate convention

The generator polynomial is

```
g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1      (bit mask 0xC75)
```

exported in C++ as `kiss::GOLAY_GENERATOR` and in Python as `GENERATOR_POLY`. For each of the
2^12 message polynomials m(x) of degree ≤ 11, the codeword is the carry-less GF(2) product
m(x)·g(x). Because that product has degree ≤ 22 it already fits in 23 coordinates and needs no
reduction mod x^23 − 1. Bit 23 is then set to the overall parity of bits 0..22.

Coordinate convention: **bit i = coordinate i = coefficient of x^i; bit 23 is the parity
coordinate.** The list is sorted ascending and cached.

A sanity check on the generator is included in the Python tests: g(x) divides x^23 + 1 over
GF(2).

### 2.2 API

```cpp
// include/kiss/golay.h, namespace kiss
constexpr uint32_t GOLAY_GENERATOR = 0xC75u;                    // x^11+x^10+x^6+x^5+x^4+x^2+1
std::vector<uint32_t> golay_codewords();                        // 4096 masks, sorted ascending, cached
std::array<int,25> golay_weight_distribution(const std::vector<uint32_t>&);  // index = weight
bool golay_is_codeword(uint32_t mask);                          // O(1), 2^24-bit table (2 MiB, lazy)
std::vector<uint32_t> golay_octads();                           // 759 weight-8 words, sorted
```

The Python package `python/kiss_ref/golay.py` exposes `GENERATOR_POLY`,
`carry_less_multiply(a, b)`, `golay_codewords()`, `weight_distribution(words)`,
`is_codeword(mask)` and `octads()`, with the same conventions. It imports nothing from `src/`
and reads no generated data; numpy is not required (integer bit masks are faster at this size),
though it remains available for the other `kiss_ref` modules.

There is no `python/tests/__init__.py`: each test file adds `python/` to `sys.path` itself, so
`pytest python/tests` works from the repository root without a `pytest.ini` or a `pythonpath`
setting.

The C++ codeword list is cached in a function-local static (thread-safe initialisation) and
`golay_codewords()` returns a copy. `golay_is_codeword` builds a 2^24-bit bitset from the cached
list on first use (2 MiB) and rejects any mask with bits ≥ 24 set.

### 2.3 What is verified

The C++ test `tests/test_golay.cpp` (ctest name `test_golay`) is exhaustive throughout:

1. 4096 words, sorted, distinct, all < 2^24.
2. Weight distribution exactly `{0:1, 8:759, 12:2576, 16:759, 24:1}`.
3. Bit 23 equals the parity of bits 0..22 for every word.
4. Linearity two ways: (a) `a ^ b` is a codeword for all 4096² ordered pairs; (b) the GF(2)-span
   of the 12 basis words x^k·g(x), k = 0..11 (parity-extended, computed with a separate
   carry-less multiply written inside the test) has 4096 distinct elements and equals the
   codeword list.
5. Zero word and all-ones word present.
6. Minimum distance 8: minimum nonzero weight, plus a direct check of all C(4096,2) pairwise
   Hamming distances.
7. Octads: 759, each of weight 8, sorted, each a codeword whose complement is also a codeword;
   all 759² ordered pairs intersect in 0, 2, 4 or 8 coordinates, and the histogram equals the
   S(5,8,24) counts (per octad: 30 disjoint, 448 meeting in 2, 280 meeting in 4, 1 itself).
8. `golay_is_codeword` agrees with the sorted list on every one of the 2^24 masks; rejects
   `1<<24` and `0xffffffff`.
9. Fingerprint: the sorted masks rendered as 4096 lines of `%06x\n` are SHA-256'd with a small
   FIPS 180-4 implementation embedded in the test (known-answer-tested on `""` and `"abc"`) and
   compared against the Python value.
10. A second `golay_codewords()` call returns the identical list.

The Python suite `python/tests/test_golay.py` (12 tests) covers the same ground, sampling where
the C++ version is exhaustive: g(x) | x^23+1; 4096 distinct sorted words in [0, 2^24); the exact
weight distribution; the parity bit and even weight of every word; zero and all-ones present;
XOR closure on all C(300,2) = 44,850 pairs from a seeded random sample of 300 words; full
closure via the 12-word GF(2)-span; minimum distance 8 (min nonzero weight plus a direct
pairwise check on 200 sampled words); octad properties on 120 sampled octads; `is_codeword`
against set membership on 2000 random 24-bit masks and against every single-bit flip of 50
sampled codewords; `weight_distribution` on small hand inputs; and a write/read round trip of
the sorted masks through a pytest `tmp_path` file outside the repo.

### 2.4 Observed values

- Weight distribution `{0:1, 8:759, 12:2576, 16:759, 24:1}`.
- 4096 distinct words, XOR-closed, span of 12 independent basis words; all-ones (`0xffffff`)
  present; minimum distance 8; 759 octads.
- Octad pair intersections only in {0, 2, 4, 8}, ordered-pair counts
  `0:22770 2:340032 4:212520 8:759`; per octad 30 / 448 / 280 / 1.
- First masks (hex, sorted): `000000 00149f 00293e 003da1 0046e3 00527c 006fdd 007b42`;
  last: `ffc25e ffd6c1 ffeb60 ffffff`.
- SHA-256 of the sorted list, written as one lowercase 6-hex-digit mask per line with `\n`
  terminators (4096 lines):
  **`bf7ccc59243c42c70adbc220c5c49d4b55ec66c94e155ad4a9369ed12dfea610`**, identical for the two
  implementations, so the two codeword sets are identical *and identically ordered*.

```
$ build/release/tests/test_golay
weight distribution : 0:1 8:759 12:2576 16:759 24:1
octad intersections : 0:22770 2:340032 4:212520 8:759 (ordered pairs)
first masks         : 000000 00149f 00293e 003da1
last masks          : ffeb60 ffffff
sha256(sorted list) : bf7ccc59243c42c70adbc220c5c49d4b55ec66c94e155ad4a9369ed12dfea610 (matches python)
RESULT ok=1 words=4096 octads=759 w8=759 w12=2576 w16=759 min_d=8 sha_match=1 gen_ms=0.162 total_ms=87 failures=0
```

The fingerprint was also confirmed with the system tool, using a throwaway dumper linked
against `src/golay.cpp` (written to a scratch directory, not the repo):

```
$ ./dump > golay_masks.txt && wc -l golay_masks.txt && sha256sum golay_masks.txt
4096 golay_masks.txt
bf7ccc59243c42c70adbc220c5c49d4b55ec66c94e155ad4a9369ed12dfea610  golay_masks.txt
```

`test_golay` passes on the release, debug and sanitize presets (0.08 s / 0.62 s / 0.77 s;
ASan + UBSan clean).

### 2.5 Timing

Generating and sorting the 4096 words takes 0.16 ms in C++ release and ~6 ms in pure Python.
The whole C++ test — including the 2^24-mask membership sweep, 16.7M XOR lookups and 8.4M
pairwise distances — is 87 ms release, ~0.6 s debug, ~0.8 s under ASan/UBSan.

---

## 3. The Leech lattice minimal vectors

### 3.1 Constants and types

```cpp
// include/kiss/types.h
N = 196560;  DIM = 24;  DEG = 4600;
using Vec = std::array<int8_t, 24>;
int dot(const Vec&, const Vec&);
inline int norm2(const Vec&);          // <a,a>
inline Vec negate(const Vec&);         // -a
inline int ip_class(int ip);           // {-32,-16,-8,0,8,16,32} -> 0..6, else -1
```

`DEG = 4600` is the number of minimal vectors at 60° to a given one — the degree of the
"conflict graph" whose independent sets are the objects of interest.

### 3.2 The three shapes

Integer √8 scaling, three shapes:

| shape | generated as | count |
|---|---|---|
| (a) (±2^8, 0^16) | for each of the 759 octads: signs of the first 7 octad coordinates free (128 patterns), the 8th sign fixed so the number of minus signs is even | 97152 |
| (b) (∓3, ±1^23) | for each coordinate i and each of the 4096 codewords c: start from (−3 at i, +1 elsewhere), negate the coordinates lying in c | 98304 |
| (c) (±4, ±4, 0^22) | every pair i < j, all four sign combinations | 1104 |

Total 97152 + 98304 + 1104 = **196560**.

**Shape (b) sign convention — a question that was open and is now settled.** It was unclear
whether the construction should start from (−3, +1^23) or from (+3, −1^23). The answer: the
(−3, +1^23) convention passes the membership test as written, and the other convention is
merely its negation and generates the *same* set. Before negation all 24 entries are ≡ 1
(mod 4), so the "≡ 3 (mod 4)" class is the empty set = the zero codeword, and Σ = 20 ≡ 4
(mod 8). Negating the coordinates in c turns exactly c into the ≡ 3 class, and changes the sum
by −2·(|c| − 4) if i ∈ c, or by −2|c| if i ∉ c — both ≡ 0 (mod 8), since |c| ∈ {0, 8, 12, 16,
24}. The alternative (+3, −1^23) start has all 24 entries ≡ 3 (mod 4), i.e. class = the
all-ones word, with Σ = −20 ≡ 4; because the all-ones word is in the code, it produces the same
set with c ↔ complement(c). `test_leech` builds both sets and asserts they are equal (98304
rows) and entirely contained in the lattice. No fix-up was needed, and the code uses
(−3, +1^23).

### 3.3 Canonical order

The canonical order is `std::sort` on `Vec`: `std::array<int8_t,24>::operator<` is exactly the
signed lexicographic comparison with coordinate 0 most significant. The first row is
`-4 -4 0 … 0` and the last is `4 4 0 … 0`. The negation permutation is
`neg[i] = index_of(−C[i])`, computed by binary search.

This ordering is the index convention used by every downstream artifact in the project: the
adjacency table, the certificate files, the GPU kernels, and every index list in
`03-search-for-497.md` and `05-lifting-template-and-families.md` refer to rows of
`data/leech_min.txt` in this order.

### 3.4 Packed layout

`uint32[6][N]` structure-of-arrays: word `w` of vector `i` sits at `packed[w*N + i]` and holds
coordinates `4w .. 4w+3` as `uint8(int8)` bytes with coordinate `4w` in the low byte —
equivalently, `int8[4]` reinterpreted as a little-endian `uint32`. This is the layout the
`__dp4a` kernels consume: the SoA arrangement makes the 32 lanes of a warp read six 128-byte
cache lines. The test checks the layout byte-by-byte against the int8 rows, checks
`unpack(pack(C)) == C`, and checks that the file `leech_packed.u32` decodes back to
`leech_min.i8` byte-exactly.

### 3.5 Membership test

`Leech::is_lattice_vector` accepts an arbitrary `std::array<int,24>` and applies the standard
criterion: common parity via `x & 1`; residue class via `x & 3` (correct for negative integers
in two's complement); Golay membership of that class via `golay_is_codeword`; then
`(Σx) & 7 == 4m`. Only one of the two complementary residue classes (≡ 0 resp. ≡ 1 mod 4) is
checked in the function, because the presence of the all-ones word in `G24` makes the other
automatic; the test verifies this on all 196560 rows.

Rejected in the tests: `(1^24)` (Σ = 24), mixed parity, `(2^7, 0^17)`, `(2^9, 0^15)`, an octad
with one coordinate moved off the octad, an octad with one minus sign (Σ ≡ 4), `(4, 0^23)`
(Σ = 4), `(4^3, 0^21)`, `(5, −3^23)` (Σ ≡ 0), and 20000 random vectors in [−4,4]^24.

Accepted: `0`, `(1, −3^23)`, `(4, −4, 0^22)`, 2000 random sums of two minimal vectors, and the
**non-minimal** lattice vectors `(8, 0^23)` and `(4^4, 0^20)` — both have all coordinates ≡ 0
(mod 4), so their class is the empty set = zero codeword and their sum is ≡ 0 (mod 8). They lie
in `Λ24` with norm 64, and `index_of` correctly returns −1 for both.

### 3.6 API

```cpp
// include/kiss/leech.h
struct Leech { /* C, neg, index_of, is_lattice_vector */ };
Leech generate_leech();
Leech load_leech(const std::filesystem::path& data_dir);
int leech_shape(const Vec&);                                       // 0 octad / 1 (∓3,±1^23) / 2 (±4,±4) / -1
std::vector<uint32_t> pack_vectors(const std::vector<Vec>&);       // SoA dp4a layout
std::vector<Vec> unpack_vectors(const std::vector<uint32_t>&);
void save_leech(const Leech&, const std::filesystem::path& data_dir);

// include/kiss/io.h
std::vector<Vec> read_set(const std::filesystem::path&);
void write_set(const std::filesystem::path&, const std::vector<Vec>&, header);
std::vector<uint8_t> read_binary_file(const std::filesystem::path&);
void write_binary_file(const std::filesystem::path&, const void*, std::size_t);
std::string sha256_hex(const void*, std::size_t);
std::string sha256_file(const std::filesystem::path&);
```

`load_leech` validates both file sizes, strict ascending row order, and that
`C[neg[i]] == −C[i]` with `neg[i] < N` for every i — all O(N), about 70 ms. Truncated,
unsorted and wrong-`neg` inputs are rejected (tested).

The Python side is `python/kiss_ref/leech.py`, an independent numpy generator importing only
`kiss_ref.golay`.

### 3.7 The S text format

`read_set` accepts 24 integers per line; lines whose first non-blank character is `#` are
comments; blank lines are ignored; a trailing `# …` after the 24 values is stripped; tabs and
CR are accepted. It raises errors (prefixed `file:line:`) on wrong arity, non-integer tokens,
and values outside [−128, 127]. `write_set` writes each line of a header comment as `# <line>`,
then the rows as space-separated integers. `read_set` deliberately does **not** check norms —
that is the verifier's job (§7).

`data/leech_min.txt` is written with **no header**, so that it is byte-deterministic and loads
directly with `np.loadtxt`; it is also a valid S-format file and `read_set` reads it back equal
to `C`.

### 3.8 Data files

`tools/gen_leech.cpp` (`gen_leech [data_dir]`) writes the four data files plus
`checksums.sha256`, reloads them and compares.

```
$ build/release/tools/gen_leech data/
aea59406d9129ec635ab03c6221e7ac7c308bf860205c57f1d931aa8ddb1c111  leech_min.i8
5a9cb899e141ecf1f894c11412aa9356f2a84b44839d70e4e11d0990aa5eed57  neg.u32
24cadb80ff2e7ea6062cea611fd4981595a8e601a251e98013ee95fbb83f045c  leech_packed.u32
683a6828bcbe1d6ff54f38083646b33044d1b0ac4ded66d2bb00be594c62b6da  leech_min.txt
RESULT ok=1 n=196560 shape_octad=97152 shape_31=98304 shape_44=1104 roundtrip=1 sha256_i8=aea59406d9129ec635ab03c6221e7ac7c308bf860205c57f1d931aa8ddb1c111 dir=data/ gen_ms=64.2 write_ms=204.0 verify_ms=73.7

$ (cd data && sha256sum -c checksums.sha256)        # system tool agrees with the embedded SHA-256
leech_min.i8: OK
neg.u32: OK
leech_packed.u32: OK
leech_min.txt: OK

$ ls -l data/
     316 checksums.sha256
 4717440 leech_min.i8
11004240 leech_min.txt
 4717440 leech_packed.u32
  786240 neg.u32

$ head -2 data/leech_min.txt; tail -1 data/leech_min.txt; wc -l data/leech_min.txt
-4 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
-4 0 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
196560 data/leech_min.txt

$ build/release/tools/gen_leech <scratch dir> && diff data/checksums.sha256 <scratch>/checksums.sha256
DETERMINISTIC: checksums identical
```

`data/leech_min.txt` and `data/checksums.sha256` are committed; `leech_min.i8`, `neg.u32` and
`leech_packed.u32` are generated and gitignored.

### 3.9 Verification

`tests/test_leech.cpp` (all checks exhaustive over the 196560 rows unless noted):

1. Shape counts 97152 / 98304 / 1104, total 196560; rows distinct; strictly increasing; every
   row has norm 32, passes `is_lattice_vector`, and its complementary residue class is a
   codeword.
2. `neg[neg[i]] == i`, `C[neg[i]] == −C[i]`, `index_of(C[i]) == i` for all i; `index_of`
   returns −1 for 0 and for `(8, 0^23)`, and agrees with the octad test for `(2^8, 0^16)` on
   coordinates 0..7.
3. The shape (b) sign convention (§3.2).
4. The membership accept/reject cases listed in §3.5.
5. Inner-product histogram (OpenMP, 16 threads) from the first row of each shape (indices 2094,
   46, 0) and 64 random rows (`mt19937(20260825)`): all 67 histograms equal
   `{32:1, 16:4600, 8:47104, 0:93150, −8:47104, −16:4600, −32:1}`, confirming `DEG == 4600`.
6. Packed layout byte-exact against the int8 rows; `unpack(pack(C)) == C`.
7. Files: SHA-256 known answers (`""`, `"abc"`, `"a"×10^6`); `save_leech` produces sizes
   4717440 / 786240 / 4717440 bytes; `load_leech` equals `generate_leech` in both `C` and
   `neg`; the packed file decodes to the int8 file byte-exactly; in-memory SHA-256 equals file
   SHA-256; `leech_min.txt` read back through `read_set` equals `C`; truncated / unsorted /
   bad-`neg` / missing inputs throw; `read_set`/`write_set` round trip, header lines, tolerant
   parsing (tabs, CR, trailing comment, blank and comment lines), and errors for 3 values, 25
   values, 128, −129, `x`, `1.5`, and a missing file; a comment-only file yields an empty set.

`python/tests/test_leech.py` (13 tests) is the independent counterpart: shape counts and dtypes
and octad supports equal to the 759 octads with even sign parity; 196560 distinct rows,
strictly increasing and equal to Python's own tuple sort; norms 32; negation closure both as a
set and as an involutive permutation; vectorised membership on all rows plus scalar checks on
samples and the complementary class; the same negative and non-minimal cases as the C++ test;
the histogram from a base of each shape and 16 random bases (int64 matmul, batched); inner
products of random pairs falling only in the seven classes; and a **cross-check** (skipped if
`data/leech_min.txt` is absent) that the file set equals the Python set as sets of tuples and
that the file order equals `sorted()` order, hence the two agree row for row. That cross-check
ran for real, not skipped.

```
$ build/release/tests/test_leech
shapes              : octad=97152 three_one=98304 four_four=1104 total=196560
shape (b) convention: (-3 at i, +1 elsewhere) then negate c: members=98304/98304, (+3,-1^23) variant gives the same set=1
histogram shape 0  : base=2094 -32:1 -16:4600 -8:47104 0:93150 8:47104 16:4600 32:1
histogram shape 1  : base=46 -32:1 -16:4600 -8:47104 0:93150 8:47104 16:4600 32:1
histogram shape 2  : base=0 -32:1 -16:4600 -8:47104 0:93150 8:47104 16:4600 32:1
RESULT ok=1 n=196560 octad=97152 three_one=98304 four_four=1104 hist_ok=67/67 threads=16 gen_ms=68.1 hist_ms=15.6 total_ms=604 failures=0

$ build/sanitize/tests/test_leech | tail -1
RESULT ok=1 n=196560 octad=97152 three_one=98304 four_four=1104 hist_ok=67/67 threads=16 gen_ms=1336.4 hist_ms=824.8 total_ms=6311 failures=0
```

`test_leech` passes on release (0.83 s), debug (3.18 s) and sanitize (6.29 s). On the Python
side, `pytest python/tests/test_leech.py -q` gives 13 passed in 3.32 s and `pytest python/tests
-q` gives 68 passed in 6.04 s at this stage (the suite reaches 94 tests once the verifier tests
land, §6).

### 3.10 Timing (release)

| step | time |
|---|---|
| `generate_leech()` (three shapes, sort, `neg` via 196560 binary searches) | 64–68 ms |
| `save_leech` (4 files, 21 MB) | ~205 ms |
| `load_leech` + validation | ~74 ms |
| 67 inner-product histograms × 196560 dots, OpenMP 16 threads | 16 ms |
| whole `test_leech` | 0.6 s release, 3.2 s debug, 6.3 s ASan/UBSan |
| `gen_leech` end-to-end | 0.35 s |
| Python: generate + sort | ~0.5 s; full `test_leech.py` 3.3 s |

---

## 4. The adjacency table

`data/adj.u32` is the full 60-degree neighbour list of the conflict graph: for every one of the
196560 minimal vectors, the sorted list of the 4600 minimal vectors at inner product 16.

**File:** 196560 rows × 4600 `uint32` = **3,616,704,000 bytes**, SHA-256
`83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1`, recorded in the committed
`data/adj.sha256` as a `sha256sum -c`-compatible line. The file itself is gitignored and is
rebuilt in **1.2–1.9 s** on the RTX 3070 Laptop, deterministically: the same SHA-256 for chunk
sizes 1024 … 16384 and across repeated runs.

### 4.1 Files

| File | Purpose |
|---|---|
| `cuda/adjacency_cuda.h` | CUDA-header-free public API (includable from plain `.cpp`): `build_adjacency_rows_device()` (raw device pointers), `class AdjacencyBuilder` (context object; packed words uploaded once; `build_rows()` into host memory; `build_file()` streaming), `build_adjacency_file(L, path, chunk_rows)` (used by both the tool and the test), `adjacency_to_device()` / `adjacency_device_free()`, `cuda_available()`, `device_mem_info()`. |
| `cuda/build_adjacency.cu` | The kernel and the above. |
| `include/kiss/adjacency.h`, `src/adjacency.cpp` | `class Adjacency`: mmap (`PROT_READ`, `MAP_PRIVATE`) of `adj.u32` with the size validated in the constructor; `row(v)`, `adjacent(u,v)` (binary search), `rows()`, `cols()`, `bytes()`, `data()`, `path()`, `advise_sequential()`. Move-only. |
| `tools/build_adj.cpp` | `build_adj [data_dir] [chunk_rows]` → writes `adj.u32` and `adj.sha256`. |
| `tests/test_adjacency.cpp`, `tests/tasks/T1_6.cmake` | Acceptance test `test_adjacency` (label `gpu`, `SKIP_RETURN_CODE 77`, `TIMEOUT 900`, run from the source root with argument `data`). |
| `data/adj.sha256` | Committed. |

### 4.2 Kernel design

One block of 512 threads handles **2 consecutive rows**:

- The block's two row vectors (6 packed words each, from `packed[w*N + i]`) are held in
  registers.
- Threads stride over `j ∈ [0, N)`: 6 coalesced loads of the SoA packed words of `j`, then per
  row `acc = __dp4a(a_w, b_w, acc)` for w = 0..5 (signed overload; the `uint32` words are cast
  to `int`), i.e. 12 dp4a per 6 loads. Handling two rows per block halves the L2 traffic over
  the 4.7 MB packed table, which is slightly larger than GA104's 4 MB L2.
- A hit (`acc == 16`) is appended to the row's shared-memory buffer via `atomicAdd` on a shared
  counter, with a bounded store (`idx < 4608`). The hit rate is 2.3 %, so divergence is
  negligible.
- After the sweep, each row's ≤ 4608 hits are loaded 9 per thread and sorted with
  `cub::BlockRadixSort<uint32_t, 512, 9>` on the low **18 bits** (N < 2^18, so 5 radix passes
  instead of 8), written back to shared memory and copied out coalesced. The sort's temporary
  storage **aliases the row's own hit buffer** (the keys are in registers by that point), so
  static shared memory is 2 × 18.5 KB = 37 KB, `static_assert`ed ≤ 48 KB. Padding keys are
  `0xFFFFFFFF`, so only the first 4600 sorted entries are stored.
- The raw hit count of every row goes to `d_counts[]`, and **the host checks that it is exactly
  4600 for every row**, throwing with the offending row named otherwise; nothing is written to
  the final file in that case.

If N were ever raised to ≥ 2^18 the `static_assert` on `KEY_BITS` would fire.

### 4.3 Streaming and memory

`AdjacencyBuilder::build_file` streams chunks of `chunk_rows` (default **4096** rows = 75.4 MB)
through two device buffers, two pinned host buffers, two streams and two events: the kernel and
device-to-host copy of chunk *k+1* run while chunk *k* is validated and `fwrite`-n in row order.
The file is written as `adj.u32.tmp` and renamed on success, so a partial file can never be
mistaken for a valid one. Peak host RSS of `build_adj` is **259 MB** (`/usr/bin/time -v`
reports `Maximum resident set size (kbytes): 259176`); the full table never exists in host RAM.

**Recommended chunk size: 4096 rows** (`ADJ_DEFAULT_CHUNK_ROWS`). It sits on the flat part of
the curve (1.36 s versus 1.29 s at 16384) while keeping the device footprint at 2 × 75 MB +
4.7 MB and pinned host memory at 151 MB. Larger chunks only pay in pinned-allocation time;
1024-row chunks lose ~0.4 s to per-launch overhead and reduced kernel/`fwrite` overlap. For
consumers that want rows on the device without going through a file,
`build_adjacency_rows_device` with 4096-row chunks costs ~19 ms per chunk.

**Design note.** `Adjacency::to_device` is provided as the free function
`kiss::cuda::adjacency_to_device(const Adjacency&)` (declared in `cuda/adjacency_cuda.h`)
rather than as a member, so that `libkiss` stays a pure CPU library (`kiss_cuda` links `kiss`,
not the other way round). It `cudaMalloc`s 3,616,704,000 bytes and copies from the mmap in
64 MB pieces; on `cudaMalloc` failure it throws with the CUDA error string and the device's
free/total memory ("Is another process using the GPU?"). Free with `adjacency_device_free`.
`include/kiss/adjacency.h` documents this.

### 4.4 Verification

`tests/test_adjacency.cpp`. If `data/adj.u32` is absent or has the wrong size, the test first
builds it with the same `build_adjacency_file()` that the tool uses, so ctest is
self-contained; with no CUDA device *and* no file it exits 77 (skip). All checks are CPU except
step 6; seed `mt19937_64(20260825)`.

1. Full pass over the 3.6 GB mmap (OpenMP, 16 threads): every row has 4600 entries, strictly
   increasing, all `< N`, none equal to the row index.
2. Symmetry: 10^6 random `(i, k)` → `j = row(i)[k]`, check `adjacent(j, i)`; plus, for 200
   random full rows, `i ∈ row(j)` for every `j ∈ row(i)`.
3. Those same 200 rows equal a CPU brute-force row (`kiss::dot == 16` over all N)
   element-for-element.
4. For 10^5 random edges and for every edge of the 200 rows: `C[i] − C[j] ∈ C`
   (`index_of ≥ 0`), `neg[j] ∉ row(i)`, `dot == 16`; and on 1000 random vertex pairs
   `adjacent(i,j) == (dot == 16)`, `adjacent(i, neg[i]) == false`, `adjacent(i,i) == false`.
5. File size == 3,616,704,000; `sha256_file(adj.u32)` equals the first token of
   `data/adj.sha256`.
6. `adjacency_to_device` succeeds (device free memory printed before / during / after) and is
   freed.

### 4.5 Outputs

Build of the committed file, with the GPU idle (`nvidia-smi` showed 15 MiB used, 0 % util, only
Xorg):

```
$ build/release/tools/build_adj data
leech           : loaded from data
device memory   : free 7.50 / total 7.66 GiB before build
build           : chunk_rows=4096 upload_s=0.082 build_s=1.897 (kernel+copy+fwrite)
83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1  adj.u32   (sha256 in 12.5 s)
RESULT ok=1 rows=196560 deg=4600 bytes=3616704000 sha256=83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1 build_s=1.897 chunk_rows=4096 sha_s=12.5 total_s=14.7 file=data/adj.u32

$ (cd data && sha256sum -c adj.sha256)         # system tool agrees with the embedded SHA-256
adj.u32: OK
```

Acceptance test on that file (file hot in the page cache):

```
$ build/release/tests/test_adjacency data
leech             : 196560 vectors
file size         : 3616704000 bytes (expected 3616704000)
full pass         : 196560 rows, 0 malformed (strictly increasing, < N, != self), 0.18 s, 16 threads
symmetry (pairs)  : 1000000 random (i, j∈row(i)), 0 with i∉row(j), 0.11 s
full rows         : 200 random rows: brute-force mismatches=0, asymmetric edges=0, neg[j]∈row(i)=0, C[i]-C[j]∉C=0, adjacent(i,j) false=0, 0.07 s
random edges      : 100000 edges: C[i]-C[j]∉C=0, neg[j]∈row(i)=0, dot!=16: 0, 0.02 s
sha256            : 83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1  (data/adj.sha256: match) 13.4 s
to_device         : ok, 0.70 s (5.14 GB/s); device free 7.50 -> 4.13 -> 7.50 GiB (total 7.66)
RESULT ok=1 rows=196560 deg=4600 bytes=3616704000 built=0 build_s=0.000 bad_rows=0 asym_pairs=0 brute_rows=200 brute_mismatch=0 neg_in_row=0 diff_not_in_C=0 sha256=83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1 sha_checked=1 to_device=1 upload_s=0.70 pass_s=0.18 total_s=14.7 failures=0
```

Self-build path, from a scratch data directory containing only `leech_min.i8` and `neg.u32`.
This run overlapped with the tightness-kernel benchmark, a compute-sanitizer run, and a second
3.6 GB file competing for the page cache — hence the slow `build_s` and the disk-bound
`to_device`:

```
$ test_adjacency <scratch>/d1
adjacency         : <scratch>/d1/adj.u32 absent/invalid, building on the GPU ...
build             : chunk_rows=4096 upload_s=2.091 build_s=7.020 total_s=9.110
sha256            : 83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1  (no <scratch>/d1/adj.sha256 to compare against — NOT checked)
to_device         : ok, 12.17 s (0.30 GB/s); device free 7.50 -> 4.13 -> 7.50 GiB (total 7.66)
RESULT ok=1 rows=196560 deg=4600 bytes=3616704000 built=1 build_s=7.020 ... sha_checked=0 to_device=1 upload_s=12.17 pass_s=2.19 total_s=154.7 failures=0
```

Determinism and chunk-size sweep (GPU otherwise idle; output to a scratch directory, same
SHA-256 every time):

```
chunk_rows=1024   upload_s=0.025 build_s=1.790   sha256=83dd4d9b…
chunk_rows=2048   upload_s=0.046 build_s=1.449   sha256=83dd4d9b…
chunk_rows=4096   upload_s=0.086 build_s=1.358   sha256=83dd4d9b…
chunk_rows=8192   upload_s=0.152 build_s=1.562   sha256=83dd4d9b…
chunk_rows=16384  upload_s=0.301 build_s=1.289   sha256=83dd4d9b…
```

Kernel-only timing (`AdjacencyBuilder::build_rows` into host memory, starting at row 100000,
best of 3) and sanitizer runs:

```
rows=4096  best_s=0.0196 -> 4.8 us/row, 40.99 Gdot/s, projected full table 0.94 s, bad=0
rows=16384 best_s=0.0767 -> 4.7 us/row, 41.97 Gdot/s, projected full table 0.92 s, bad=0
compute-sanitizer --tool memcheck   (64 rows): ERROR SUMMARY: 0 errors
compute-sanitizer --tool racecheck  (8 rows):  RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

Official ctest run in the shared build tree. These runs were **contended** — `nvidia-smi`
showed 169 MiB already in use when ctest started (another GPU test), and three other Python
processes were at 60–85 % CPU each — which is why the SHA-256 took 35 s instead of 13 s, the
full pass 2.6 s instead of 0.2 s, and `build_s` 8.5 s instead of 1.2–1.9 s (the `fwrite` of
3.6 GB competes for page cache). Correctness is unaffected: the SHA-256 is identical to the
idle-GPU build.

```
$ cmake --preset release && cmake --build build/release      # rc=0
-- kiss tools: build_adj;gen_leech;kiss_info;random_aut;verify_s

$ ctest --test-dir build/release --output-on-failure -R adjacency
1/1 Test #7: test_adjacency ...................   Passed   41.82 sec

full pass         : 196560 rows, 0 malformed (strictly increasing, < N, != self), 2.57 s, 16 threads
symmetry (pairs)  : 1000000 random (i, j∈row(i)), 0 with i∉row(j), 0.38 s
full rows         : 200 random rows: brute-force mismatches=0, asymmetric edges=0, neg[j]∈row(i)=0, C[i]-C[j]∉C=0, adjacent(i,j) false=0, 0.60 s
random edges      : 100000 edges: C[i]-C[j]∉C=0, neg[j]∈row(i)=0, dot!=16: 0, 0.08 s
sha256            : 83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1  (data/adj.sha256: match) 34.9 s
to_device         : ok, 2.20 s (1.64 GB/s); device free 7.50 -> 4.13 -> 7.50 GiB (total 7.66)
RESULT ok=1 rows=196560 deg=4600 bytes=3616704000 built=0 build_s=0.000 bad_rows=0 asym_pairs=0 brute_rows=200 brute_mismatch=0 neg_in_row=0 diff_not_in_C=0 sha256=83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1 sha_checked=1 to_device=1 upload_s=2.20 pass_s=2.57 total_s=41.2 failures=0

$ build/release/tools/build_adj data                          # official binary, regenerates the same file
build           : chunk_rows=4096 upload_s=0.137 build_s=8.487 (kernel+copy+fwrite)
83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1  adj.u32   (sha256 in 18.3 s)
RESULT ok=1 rows=196560 deg=4600 bytes=3616704000 sha256=83dd4d9bc42c9373a9e15fc71ace1b4a52807d1707cf5917a9b8340a66bc14e1 build_s=8.487 chunk_rows=4096 sha_s=18.3 total_s=27.4 file=data/adj.u32
```

### 4.6 Timing (uncontended figures are the ones to plan with)

| step | time |
|---|---|
| pack + upload packed words, allocate buffers | 0.03–0.3 s (grows with chunk size: pinned allocations) |
| kernel + D2H, whole table | 0.92 s (≈ 42 × 10^9 dots/s, ≈ 2.5 × 10^11 dp4a/s) |
| kernel + D2H + `fwrite` of 3.62 GB (`build_s`) | **1.2–1.9 s** (write into the page cache; NVMe flush proceeds in the background) |
| `sha256_file` of 3.62 GB (the in-tree software SHA-256, ~290 MB/s) | 11–13 s — the dominant cost of `build_adj` and of the test |
| full validation pass over the mmap (16 threads, page cache hot) | 0.18 s (2.2 s when the cache was under pressure) |
| `adjacency_to_device` (cudaMalloc 3.62 GB + copy from mmap) | 0.70 s (5.1 GB/s) with the file hot; 12 s when the mmap had to be re-read from disk |
| `test_adjacency` end-to-end (file present) | 14.7 s |
| `test_adjacency` including the build | ~25 s uncontended |

---

## 5. The fused GPU tightness kernel

For a batch of independent sets ("chains") this kernel computes, for every vertex of the
conflict graph, how many members of that chain are at 60° to it. A vertex with count 0 that is
not itself in the chain is *free* — it could be added. This is the inner loop of the local
searches in `03-search-for-497.md`.

### 5.1 API

```cpp
namespace kiss::cuda {
#define KISS_CUDA_CHECK(expr)   // throws std::runtime_error "file:line: expr failed: <cudaGetErrorString> (<name>)"
constexpr int SMAX = 1024;               // max |S| per chain
constexpr int INS_WORDS = (N + 31) / 32; // = 6143, words per chain in an inS bitmap
constexpr int PACKED_WORDS = 6;

struct DeviceLeech { const uint32_t* packed = nullptr /*[6][N] SoA*/; const uint32_t* neg = nullptr /*[N]*/; };
DeviceLeech upload_leech(const Leech&);          // synchronous; throws
void free_leech(DeviceLeech&);                   // cudaFree both, nulls pointers
class ScopedDeviceLeech;                         // RAII: .get(), implicit conversion to const DeviceLeech&

void tightness_full(const DeviceLeech&, const uint32_t* d_S /*[B][SMAX]*/, const uint32_t* d_Ssize /*[B]*/,
                    int B, uint16_t* d_tight /*[B][N]*/, cudaStream_t stream = 0);
void count_free(const uint16_t* d_tight, const uint32_t* d_inS_bits /*[B][INS_WORDS] or nullptr*/,
                int B, uint32_t* d_out /*[B]*/, cudaStream_t stream = 0);
}
```

`cuda/kiss_cuda.h` is a plain C++ header (it needs only `<cuda_runtime.h>`), so it is usable
from `.cpp` files. `DeviceLeech` members are default-initialised to `nullptr`; an aggregate with
default member initialisers is still an aggregate in C++17, so `DeviceLeech{p, q}` works.

Semantics of `tightness_full`:
`tight[b*N + v] = #{ i < Ssize[b] : <C[v], C[S[b][i]]> == 16 }`. Entries of `d_S` beyond
`Ssize[b]` are never read; duplicates in S count twice; `Ssize[b] > SMAX` is clamped to SMAX
(documented, not an error). Vertex indices are not range-checked — the precondition is `< N`.

`count_free`: `out[b] = #{ v : tight[b][v] == 0 && !(inS[b][v>>5] >> (v&31) & 1) }`; with
`d_inS_bits == nullptr` it counts `tight == 0` only.

`cuda/device_leech.cu` implements `upload_leech` (packs with `kiss::pack_vectors`, then
2 × `cudaMalloc` + `cudaMemcpy`, 5.5 MB total, exception-safe) and `free_leech`.

**Known gap in the API surface.** A `build_adjacency(const DeviceLeech&, uint32_t* d_out,
uint32_t first_row, uint32_t nrows, cudaStream_t)` wrapper is *not* declared in
`cuda/kiss_cuda.h`. The adjacency builder's raw entry point,

```cpp
void kiss::cuda::build_adjacency_rows_device(const uint32_t* d_packed, uint32_t first_row, uint32_t nrows,
                                             uint32_t* d_out, uint32_t* d_counts, void* stream /*cudaStream_t*/);
```

takes an extra per-row counts buffer (`stream` is `void*` only to keep that header free of CUDA
includes), so a faithful wrapper needs a policy for it: allocating and discarding the counts
would lose the "exactly 4600 per row" check. The suggested resolution is to add the wrapper in
`cuda/device_leech.cu` with a `cudaMallocAsync`'d counts buffer and a device-side assert, or to
change the declared signature to take `d_counts`.

### 5.2 Kernel design (`cuda/tightness_full.cu`)

1. **Gather pass.** `gather_S_kernel<<<Bchunk, 256>>>` writes `Sp[b][i][w] = packed[w*N + S[b][i]]`
   (AoS, 6 words per member, ≤ 24 KB per chain) into a stream-ordered scratch buffer
   (`cudaMallocAsync`/`cudaFreeAsync` on the caller's stream, `min(B,256) × 24 KB ≤ 6 MB`).
   Chains are processed in chunks of 256, which also keeps `gridDim.y` small.
2. **Main kernel** `tightness_kernel<VPT=4>`, grid `(ceil(N/1024), Bchunk)`, block 256, 24 KB
   static shared memory. Each block copies its chain's 6·|S| words into shared memory with
   coalesced loads. Each thread owns 4 candidates `v, v+256, v+512, v+768` (24 packed words in
   registers; the SoA layout makes the 32 lanes of a warp read six 128-byte lines). The S loop
   is unrolled by 4 members: the 24 words of 4 consecutive members are fetched as 6 `uint4`
   shared loads (broadcast, `LDS.128`), then for each (candidate, member) pair
   `acc = __dp4a(w0,s0,0); … acc = __dp4a(w5,s5,acc); cnt += (acc == 16)` using the signed
   `__dp4a(int,int,int)` overload on the int8 bytes. The tail of `|S| mod 4` members is handled
   scalar. Output `tight[b][v]` is `uint16` (max 1024 = SMAX). The N × |S| matrix is never
   formed. ptxas reports 54 registers, 0 spills, 24576 B shared → 4 blocks (32 warps) per SM,
   shared-memory-limited.
3. `count_free_kernel<<<B, 1024>>>`: grid-stride over N, warp shuffles plus one shared
   reduction.

Why the gather pass and VPT = 4 (best-of-10 timings, B = 64, |S| = 500, on a box also running
other GPU/CPU jobs — hence the ranges):

| variant | best ms | pairs/s |
|---|---|---|
| VPT=1 (one candidate per thread, the literal design) | 15.7 | 4.0e11 |
| VPT=2 | 12.7 – 14.4 | 4.4 – 4.9e11 |
| **VPT=4 (chosen)** | 12.0 – 13.7 | 4.6 – 5.3e11 |
| VPT=2, no gather pass (each block gathers its S straight from `packed`) | 15.1 | 4.2e11 |

The in-block gather costs 6·|S| scattered 4-byte loads per block (each a 32-byte L2 sector);
with 384 blocks per chain that is ~12 % of the run time at |S| = 500, and the pre-gather
removes it. Larger VPT amortises the shared staging and the loop overhead over more candidates;
VPT = 4 also halves the block count for the B = 1 case (192 blocks on 40 SMs) without hurting
it (0.33–0.46 ms versus 0.51–0.54 ms for |S| = 231).

### 5.3 Throughput and the arithmetic bound

Measured throughput is **4.4–5.3 × 10^11 pairs/s** (2.6–3.2 × 10^12 dp4a/s) at B = 64,
|S| = 500.

*Correction to an earlier expectation.* The design target quoted a figure of "≳ 10^12 pairs/s".
That is not reachable on this card, and the earlier estimate was wrong rather than the kernel
being slow. One pair costs 6 `IDP4A` + 1 compare + 1 add = 8 integer instructions. GA104 has 64
INT32 lanes per SM; at the ~1.5–1.7 GHz that the 80 W laptop part sustains, 40 SMs deliver
≈ 3.8–4.4 × 10^12 integer ops/s, i.e. an upper bound of ≈ 4.8–5.4 × 10^11 pairs/s. The measured
rate is *at* that bound: the kernel is issue-bound on the dp4a pipe, not memory-bound. Getting
past it would need either a 2× faster part or a formulation with fewer than 3 instructions per
pair (e.g. an IMMA tensor-core int8 GEMM), which was deliberately rejected: the fused kernel is
exact and tiny, and this recompute is not the hot loop of the local search anyway. Per chain at
|S| = 500, 196560·500 = 9.8e7 pairs ≈ 0.2 ms amortised inside a batch (B = 64: 12–14 ms), or
0.33–0.5 ms alone at B = 1 (launch and tail effects) — well inside the "~1 ms-class per chain"
budget.

### 5.4 Verification

`tests/test_tightness_gpu.cu` (ctest name `test_tightness_gpu`, label `gpu`, skip code 77;
registered by `tests/tasks/T1_5.cmake`, which also adds `-Xcompiler=-fopenmp` to the CUDA host
pass — the OpenMP flag from `OpenMP::OpenMP_CXX` is `$<COMPILE_LANGUAGE:CXX>`-guarded and would
otherwise leave `#pragma omp` as an unknown-pragma error under `-Wall -Werror` in nvcc's host
pass). The CPU reference is a plain double loop over `kiss::dot` (OpenMP over v); it
deliberately does not use `kiss/verify.h`. Exact equality on all N entries is required for
every chain, and `count_free` is checked against a host count on every chain.

1. Warm-up call on a B = 1, |S| = 0 chain (verified too; exercises the empty-S path and pays
   the one-off lazy-module-load and `cudaMallocAsync` pool cost, 5–330 ms depending on load).
2. Fixture: `data/S496.txt` if present (`read_set` → `index_of`; every row must be a minimal
   vector), else a greedy random maximal independent set built on the CPU (seed 20260825). The
   fixture is checked to be independent (all off-diagonal Gram ≤ 8, no duplicates), then B = 1
   launch, CPU comparison, full tightness histogram, and the `tight==0 && !inS` count, which
   must be 0 (maximality).
3. Random batches B = 1, 3, 64 (68 chains, one S each); sizes cycle through
   {1, 2, 7, 64, 255, 256, 500, 1000, 1024}. Three special chains in the B = 64 batch: |S| = 0;
   an S with every vertex twice (250 × 2); and S = 1024 of the 4600 neighbours of one vertex u,
   so that `tight[u] = 1024 = SMAX`, exercising the uint16 range (asserted).
4. Throughput: B = 64, |S| = 500 random, `cudaEvent` around `tightness_full` only (upload
   excluded), 1 warm-up + `--bench-iters` (default 5) iterations, best and mean; chain 63 of
   the benchmark is also verified.

Flags: `--data DIR`, `--wait-s496 SEC`, `--bench-iters N`, `--no-bench`. Exit 77 without a
device.

### 5.5 Outputs

On the 496 (`data/S496.txt`):

```
warm-up         : first call 6.257 ms (B=1, |S|=0)
fixture         : S496.txt |S|=496 conflicts=0 duplicates=0
fixture GPU time: 0.778 ms (B=1, |S|=496)
fixture tightness histogram (value:count): 0:496 4:80 6:640 7:256 8:2704 9:8064 10:31424 11:52672 12:49552 13:31360 14:13440 15:2560 16:1480 17:256 18:704 19:64 20:528 22:128 24:152
fixture free    : tight==0 && !inS = 0 (expect 0 for a maximal set)
special chain   : u=20142 with 1024 neighbours in S
batch B=1       : 2.438 ms GPU, 1.966e+05 pairs, 8.062e+07 pairs/s, mismatches so far 0
batch B=3       : 3.284 ms GPU, 1.435e+07 pairs, 4.369e+09 pairs/s, mismatches so far 0
batch B=64      : 9.184 ms GPU, 4.563e+09 pairs, 4.969e+11 pairs/s, mismatches so far 0
max tightness seen in random batches: 1024 (expect 1024 from the special chain)
bench B=64 |S|=500: best 12.017 ms, mean 13.570 ms over 10 iters → 5.234e+11 pairs/s (3.141e+12 dp4a/s)
RESULT ok=1 fixture=S496.txt fixture_size=496 fixture_free=0 chains=69 mismatches=0 max_tight=1024 bench_best_ms=12.017 bench_pairs_per_s=5.234e+11 cpu_ref_ms=3943 omp_threads=16 total_ms=239392
```

All 496 rows resolve to minimal vectors via `index_of`; the set is independent (0 pairs with
Gram > 8, no duplicates); GPU tightness equals the CPU reference on all 196560 entries; and
**free vertices (`tight==0 && !inS`) = 0**, i.e. the 496 is maximal.

Tightness multiset over all N vertices (value:count):

```
0:496 4:80 6:640 7:256 8:2704 9:8064 10:31424 11:52672 12:49552 13:31360 14:13440 15:2560 16:1480 17:256 18:704 19:64 20:528 22:128 24:152
```

The counts sum to 196560; Σ tight = 496·4600 = 2,281,600 (every member has all 4600 of its
neighbours outside S, as it must for an independent set); the 496 members themselves have
tightness 0; every non-member vertex is blocked by at least 4 members (minimum non-zero
tightness 4, mode 11 at 52672 vertices, max 24). Odd values occur (7, 9, …) even though the 496
is antipodally closed, because the antipode of a conflict is not a conflict, so nothing forces
even counts.

The official ctest run under release:

```
$ ctest --test-dir build/release --output-on-failure -R tightness -V
6: warm-up         : first call 25.167 ms (B=1, |S|=0)
6: fixture         : S496.txt |S|=496 conflicts=0 duplicates=0
6: fixture GPU time: 0.470 ms (B=1, |S|=496)
6: fixture tightness histogram (value:count): 0:496 4:80 6:640 7:256 8:2704 9:8064 10:31424 11:52672 12:49552 13:31360 14:13440 15:2560 16:1480 17:256 18:704 19:64 20:528 22:128 24:152
6: fixture free    : tight==0 && !inS = 0 (expect 0 for a maximal set)
6: batch B=64      : 9.179 ms GPU, 4.563e+09 pairs, 4.971e+11 pairs/s, mismatches so far 0
6: bench B=64 |S|=500: best 13.691 ms, mean 13.700 ms over 5 iters → 4.594e+11 pairs/s (2.757e+12 dp4a/s)
6: RESULT ok=1 fixture=S496.txt fixture_size=496 fixture_free=0 chains=69 mismatches=0 max_tight=1024 bench_best_ms=13.691 bench_pairs_per_s=4.594e+11 cpu_ref_ms=4417 omp_threads=16 total_ms=5020 failures=0
1/1 Test #6: test_tightness_gpu ...............   Passed    5.33 sec
```

An earlier run of the same test, before the converted 496 fixture existed, used the greedy
fallback:

```
6: fixture         : greedy_mis |S|=231 conflicts=0 duplicates=0
6: fixture GPU time: 0.366 ms (B=1, |S|=231)
6: fixture tightness histogram (value:count): 0:231 1:315 2:3065 3:13461 4:34338 5:52581 6:49614 7:29091 8:10985 9:2514 10:329 11:34 12:2
6: fixture free    : tight==0 && !inS = 0 (expect 0 for a maximal set)
6: RESULT ok=1 fixture=greedy_mis fixture_size=231 fixture_free=0 chains=69 mismatches=0 max_tight=1024 bench_best_ms=13.671 bench_pairs_per_s=4.601e+11 cpu_ref_ms=5281 omp_threads=16 total_ms=5863 failures=0
```

**A greedy random maximal independent set has only 231 vertices** (seed 20260825; only that one
seed was tried for this construction). That is far below 496 and is a useful calibration point:
random greedy starts reach roughly 50 % of the record, so seeding a local search from them is
not competitive — see `03-search-for-497.md`.

compute-sanitizer on the final VPT = 4 binary (memcheck slows the kernel about 35×; the greedy
fallback was in use because the 496 fixture was absent at that moment):

```
$ compute-sanitizer --tool memcheck --leak-check full <build>/tests/test_tightness_gpu --bench-iters 1
========= COMPUTE-SANITIZER
fixture         : greedy_mis |S|=231 conflicts=0 duplicates=0
batch B=64      : 405.797 ms GPU, 4.563e+09 pairs, 1.125e+10 pairs/s, mismatches so far 0
bench B=64 |S|=500: best 492.455 ms, mean 492.455 ms over 1 iters → 1.277e+10 pairs/s (7.664e+10 dp4a/s)
RESULT ok=1 fixture=greedy_mis fixture_size=231 fixture_free=0 chains=69 mismatches=0 max_tight=1024 bench_best_ms=492.455 bench_pairs_per_s=1.277e+10 cpu_ref_ms=4411 omp_threads=16 total_ms=6699 failures=0
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
```

### 5.6 Timing summary (shared machine)

| item | time |
|---|---|
| first `tightness_full` call (lazy module load + `cudaMallocAsync` pool) | 5–330 ms one-off, load-dependent |
| `tightness_full`, B = 1, \|S\| = 231 | 0.33–0.5 ms |
| `tightness_full`, B = 64, \|S\| = 500 (best of 10) | 12.0–13.7 ms → 4.6–5.3 × 10^11 pairs/s |
| `tightness_full`, B = 64 mixed sizes (4.56 × 10^9 pairs) | 8.7–9.2 ms |
| under memcheck, B = 64, \|S\| = 500 | 490 ms |
| CPU reference, 69 chains (≈ 2.2 × 10^10 dots, OpenMP 16 threads) | 2.8 s quiet, 5–16 s under load |
| whole `test_tightness_gpu` | 3–6 s quiet, 11–18 s loaded |

### 5.7 Extension to a pair-class histogram

A pair-class histogram kernel (counting all seven inner-product classes rather than just the
16s) was not built, since nothing needed it before the association-scheme computation of
`02-upper-bounds.md` was designed. The derivation from `tightness_kernel` is mechanical: keep
the staging and the register-resident candidate words, replace `cnt += (acc == 16)` by
`hist[(acc + 32) >> 3]++` with `int hist[7]` (for two minimal vectors `acc` is always one of
{−32, −16, −8, 0, 8, 16, 32}, so the index is 0..6 — but keep the `acc == 32` self-pair separate
if v ∈ S), and write `uint16[7]` per (chain, v): 14 bytes per candidate instead of 2, i.e. a
`[B][7][N]` SoA output to keep the stores coalesced. For the association-scheme intersection
numbers p^k_{ij}, one launch per fixed pair (x, z) with S = {y} restricted to the class-j
neighbours of x is the "one launch per y, or batch 64 y's per block" shape; the per-block shared
budget (24 KB) then holds 64 y-lists of ≤ 16 members, or one list of ≤ 1024. Register cost is
+6 counters per candidate, so use VPT = 2.

---

## 6. Verifiers and certificate tooling

There is a three-link custody chain for any claimed independent set: a fast C++ verifier, an
independent Python verifier that shares no code with it, and a dimension-25 verifier that
checks the full lifted configuration.

### 6.1 Files

| File | Purpose |
|---|---|
| `include/kiss/verify.h`, `src/verify.cpp` | `VerifyResult`, `verify_independent`, `tightness_cpu`; plus `set_indices`, `is_antipodal`, `gram_histogram`, `set_line_numbers`. |
| `tools/verify_s.cpp` | `verify_s <S.txt> [--data DIR] [--generate]`: loads C (`load_leech`, falling back to `generate_leech` with a note), verifies, reports antipodality, Gram histogram and tightness profile; prints `RESULT ok=… size=… antipodal=…`; exit 0/1. |
| `tests/test_verify.cpp`, `tests/tasks/T1_4.cmake` | Acceptance test (working directory = source root, timeout 600 s). |
| `python/verify_S.py` | Independent verifier (uses only `kiss_ref`; regenerates C in numpy; reads nothing produced by the C++ side). `RESULT ok=… size=… antipodal=…`; exit 0/1. |
| `python/verify_dim25.py` | Builds the R^25 configuration (d = 1, T = {±1}); exact integer casework plus float64 explicit coordinates; prints both counts. |
| `python/tests/test_verify.py` | 15 pytest tests (fixture-based ones skip if `data/S49x.txt` or `build/release/tools/verify_s` are absent). |

Additional API:

```cpp
VerifyResult verify_independent(const Leech&, const std::vector<Vec>&, const std::vector<long>& lines); // "line L (row k)" labels
std::vector<uint32_t> set_indices(const Leech&, const std::vector<Vec>&);   // throws on a non-member
bool is_antipodal(const std::vector<Vec>&);
std::array<long, 8> gram_histogram(const std::vector<Vec>&);               // classes 0..6, [7] = other
std::vector<long> set_line_numbers(const std::filesystem::path&);          // read_set row -> 1-based file line
```

### 6.2 What `verify_independent` checks

Exact integer arithmetic, in this order, stopping at the first failing check; the message names
the first offender and how many offenders that check has:

1. every row has squared norm 32 —
   `norm: row k has squared norm 64 != 32: (…) [n offending row(s)]`;
2. every row is in C via `Leech::index_of` —
   `membership: row k is not a Leech minimal vector: (…)`;
3. all rows distinct, via the canonical indices —
   `distinct: row a and row b are the same vector: (…)`;
4. every off-diagonal Gram entry ≤ 8, a serial double loop `i < j` in row order —
   `gram: row i and row j have inner product 16 > 8 (60 degrees): (…) . (…) [n offending pair(s)]`.

Rows are named "row k" (0-based index into the vector list). Because `read_set` drops blank and
comment lines, the tool re-scans the file with `set_line_numbers` (same grammar as `read_set`)
and passes the 1-based file lines to an overload, which then prints `line L (row k)`.
`verify_S.py` reads the file itself and prints the **identical wording** — the two verifiers'
failure messages are byte-identical on every corrupted fixture, which makes diffing them
trivial.

Python membership is a dictionary lookup in the regenerated C **and** the arithmetic membership
test (`is_lattice_vector_rows` ∧ norm 32); the two must agree or the verifier fails with
`internal:`.

`tightness_cpu(L, S_idx)` computes `out[v] = #{s ∈ S : ⟨C[v], C[s]⟩ = 16}` for all v: it
gathers the members into a contiguous array, bounds-checks them, then runs a plain
`#pragma omp parallel for` over v with an inner loop over S calling `dot`. It throws on an index
≥ N or |S| > 65535. About 60 ms for |S| = 496 on 16 threads (N·|S| = 9.7·10^7 dots).

### 6.3 The dimension-25 verifier

`python/verify_dim25.py` works in norm-4 scaling (integer coordinates divided by √8). Equatorial
vectors are `(x, 0)` for x ∈ C \ S; lifted vectors are `(x·√(2/3), ±√(4/3))` for x ∈ S. With
`ip` the integer inner product of two minimal vectors:

| pair | condition | reduces to |
|---|---|---|
| eq–eq | ip/8 ≤ 2 | ip ≤ 16 (distinct minimal vectors) |
| eq–lifted | √(2/3)·ip/8 ≤ 2 ⟺ ip ≤ 8√6 ≈ 19.6 | ip ≤ 16 suffices; **requires** equatorial = C \ S (an overlap gives ip = 32 and fails) |
| lifted–lifted, x ≠ x' | ip + 16·y·y' ≤ 24 | same sign: ip ≤ 8 (S independent); opposite sign: ip ≤ 40, always true |
| lifted–lifted, same x | (2/3)·4 − 4/3 = 4/3 ≤ 2 | always true |

**(a) Exact pass.** Structural checks (S indices distinct, `C[S_idx] == S`, the equatorial index
set is exactly the complement, `n_eq + |S| = N`, no S index in the equatorial set);
lifted–lifted via the int64 Gram of S (max off-diagonal reported, must be ≤ 8); eq–lifted via
int64 products `S × C[eq]ᵀ` (max reported, must be ≤ 16 — an overlap would show up as 32 here
too); and eq–eq by a **full pass over all C(196560,2) pairs** with max ≤ 16, plus the Leech
minimal-norm argument (|x−x'|² = 64 − 2·ip ≥ 32) stated for the record. `--skip-full-equatorial`
uses only the argument and says so.

The full pass uses float32 GEMM, which is **exact** here: the inputs are integers with
|value| ≤ 4, products ≤ 16, and partial sums ≤ 384 ≪ 2^24, so every intermediate is an exactly
representable integer in any association order. The diagonal is checked to be exactly 32 before
masking, and the first row of every tile is checked to lie in {−32, −16, −8, 0, 8, 16} (the run
reports "values seen [-32, -16, -8, 0, 8, 16]", a class check on 37249 sampled rows). Pair
counts per class are summed and checked against C(196560 + |S|, 2).

**(b) Float pass.** Explicit float64 coordinates of all 196560 + |S| vectors (norms checked
= 4 ± 1e-9), Gram by blocked matmul, max off-diagonal ≤ 2 + 1e-9, together with the pair
attaining the maximum and its class (eq-eq / eq-lift / lift-lift-same / lift-lift-opp /
lift-lift-same-x).

**Blocking design note (performance).** The original plan called for ~8192-row blocks. Row
blocks against the full column range are 1.6 GB (1024 rows) to 12.9 GB (8192 rows) of float64
per block, never cache-resident, and — the actual bottleneck — OpenBLAS does not parallelise a
K = 25 GEMM at all (measured 20 GFLOPS at 1 thread, 23 at 16, for every M ∈ {2048 … 16384}).
The first version (1024-row blocks, BLAS-threaded) took 111 s for the float pass and 62 s for
the exact pass. Both passes now use upper-triangular 512×1024 tiles (a pair (i, j) with j < i is
covered by the row block containing j) with one `ThreadPoolExecutor` task per row block running
single-threaded BLAS; `OPENBLAS_NUM_THREADS=1` is set before numpy is imported, and `--tile` /
`--workers` override the defaults. The default worker count is `cpu_count/2` = 8 physical
cores, which measured faster than 16. Result: exact pass 5.9–8.1 s, float pass 22–23 s, whole
script 31 s against a 2-minute target.

### 6.4 Test contents

`tests/test_verify.cpp` (1.7 s release; 9.3 s under ASan/UBSan):

1. Greedy random maximal independent set from `generate_leech()` (`mt19937` seed 20260825,
   random vertex order, add unless at 60° to a member): size **228**; it verifies,
   `set_indices` round-trips, and the empty set is accepted with size 0.
2. Corruptions, each rejected with the offender named: row 1 replaced by the first
   60°-neighbour of row 0 (message contains `row 0`, `row 1`, `inner product 16`; with the
   line-number overload, `line 10 (row 0)` / `line 12 (row 1)`); a duplicate row (`row 3` /
   `row 5`); `(8, 0^23)` (norm 64); an octad vector with one ±2 moved off the octad (norm 32,
   `index_of` = −1, fails `is_lattice_vector`, and `set_indices` throws); and check order (a
   norm error is reported before a Gram error).
3. `tightness_cpu`: greedy set → 0 free vertices, members have tightness 0, Σ tight = |S|·4600,
   histogram printed; a random 100-subset equals a serial brute-force double loop with the
   opposite loop order (0 mismatches, Σ = 460000, max 12); a single member yields exactly its
   4600 neighbours; empty S yields all zeros; index N throws `out_of_range`.
4. `set_line_numbers` on a file with header, blank, mid-file comment and trailing comment lines
   (`{3,4,7,8,9}`), consistent with `read_set`; a missing file throws.
5. Fixtures (skipped with a `SKIP` line if absent): `S496.txt` / `S488.txt` verify with sizes
   496 / 488, antipodal, Gram histogram printed, tightness free = 0 (both maximal), members
   tight 0, Σ = |S|·4600, tightness histogram printed; a 60°-corruption is rejected naming
   `row 0` / `row 1`.

`python/tests/test_verify.py` (15 tests, 41 s including the dimension-25 run): a greedy set
(numpy, seed 20260825, size **230**) verifies and is maximal; the empty set; the four
corruptions with the same message assertions as C++, plus a value outside int8 and an octad
with odd sign parity; `read_set` grammar and errors; `verify_S.py` end-to-end via subprocess
(good → `ok=1`, corrupted → exit 1 naming `line 3 (row 0)` / `line 4 (row 1)`, missing file →
exit 1); the exact casework in-process (count, max inner products, pair-count identity, a
60°-corruption failing "same sign", a wrong index list failing structurally); the float
coordinates (norms 4, same-x lifted pairs = 4/3, pair classes); the fixtures (Python `ok=1`
with size 496/488 and antipodal); **C++ versus Python** on each fixture — identical `size`,
`antipodal` and `gram` fields, with the 60°-corrupted file rejected by both naming the same
pair; and `verify_dim25.py data/S496.txt` → `count_exact = count_float = 197056`.

Note that the three greedy maximal independent sets quoted in this document — 228 (C++
verifier test), 230 (Python verifier test), 231 (GPU tightness test) — come from three
different greedy constructions with the same seed, not from one construction giving three
answers.

### 6.5 Outputs

```
$ build/release/tests/test_verify data
greedy MIS          : size=228 antipodal=0 gram=-32:1,-16:618,-8:6260,0:11917,8:7082 verify=ok
corrupt 60-degree   : gram: row 0 and row 1 have inner product 16 > 8 (60 degrees): (0 -2 0 0 0 0 0 2 -2 0 0 0 0 0 0 -2 0 2 0 -2 0 2 2 0) . (-3 -1 -1 -1 -1 -1 1 1 -1 -1 1 1 1 1 -1 -1 1 1 1 -1 -1 1 1 -1) [7 offending pair(s)]
corrupt duplicate   : distinct: row 3 and row 5 are the same vector: (-2 2 0 0 2 0 0 0 0 0 0 0 -2 0 0 -2 -2 0 2 0 2 0 0 0) [1 duplicate row(s)]
corrupt norm        : norm: row 2 has squared norm 64 != 32: (8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0) [1 offending row(s)]
corrupt not-in-C    : membership: row 4 is not a Leech minimal vector: (0 -2 -2 -2 -2 -2 0 -2 0 0 -2 0 -2 0 0 0 0 0 0 0 0 0 0 0) [1 offending row(s)]
tightness greedy    : free=0 hist(outside S)=1:334,2:3219,3:14214,4:36005,5:54264,6:48802,7:27477,8:9622,9:2053,10:323,11:17,12:1,13:1 sum=1048800=228*4600 [18.0 ms]
tightness random100 : mismatches=0 sum=460000 max=12
fixture S496.txt    : ok=1 size=496 antipodal=1 gram=-32:248,-8:37504,0:47504,8:37504
fixture S496.txt    : tightness free=0 hist(outside S)=4:80,6:640,7:256,8:2704,9:8064,10:31424,11:52672,12:49552,13:31360,14:13440,15:2560,16:1480,17:256,18:704,19:64,20:528,22:128,24:152 [56.6 ms]
fixture S496.txt    : corrupted -> gram: line 7 (row 0) and line 8 (row 1) have inner product 16 > 8 (60 degrees): (1 1 -1 1 3 -1 1 1 -1 1 1 1 1 1 -1 1 1 1 -1 1 1 1 -1 -1) . (-3 1 -1 1 -1 -1 1 1 -1 1 1 1 1 1 -1 1 1 1 -1 1 1 1 -1 -1) [11 offending pair(s)]
fixture S488.txt    : ok=1 size=488 antipodal=1 gram=-32:244,-8:36544,0:45496,8:36544
fixture S488.txt    : tightness free=0 hist(outside S)=2:48,4:208,5:256,6:608,7:576,8:3850,9:16376,10:35712,11:47136,12:43636,13:26648,14:13408,15:3216,16:2698,17:736,18:340,19:32,20:348,22:12,24:228 [57.1 ms]
fixture S488.txt    : corrupted -> gram: line 7 (row 0) and line 8 (row 1) have inner product 16 > 8 (60 degrees): (-1 1 1 1 1 -1 -1 1 -1 1 -1 1 1 1 1 -1 -1 1 1 1 -1 -3 1 1) . (-4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -4 0 0) [13 offending pair(s)]
RESULT ok=1 greedy_size=228 s496=496 s488=488 fixtures=present threads=16 tight_ms=27.7 total_ms=1733 failures=0

$ ASAN_OPTIONS=detect_leaks=0 ./test_verify_san data | tail -1
RESULT ok=1 greedy_size=228 s496=496 s488=488 fixtures=present threads=16 tight_ms=830.1 total_ms=9259 failures=0

$ build/release/tools/verify_s data/S496.txt
file      : data/S496.txt
C source  : load_leech(data) (10 ms)
rows      : 496
antipodal : 1
gram      : -32:248,-8:37504,0:47504,8:37504 (unordered pairs by inner product)
tightness : free=0 (vertices outside S with no 60-degree neighbour in S) histogram(outside S)=4:80,6:640,7:256,8:2704,9:8064,10:31424,11:52672,12:49552,13:31360,14:13440,15:2560,16:1480,17:256,18:704,19:64,20:528,22:128,24:152  [63.8 ms, 16 threads]
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 free=0 maximal=1 tight_hist=4:80,6:640,7:256,8:2704,9:8064,10:31424,11:52672,12:49552,13:31360,14:13440,15:2560,16:1480,17:256,18:704,19:64,20:528,22:128,24:152 file="data/S496.txt" ms=80

$ build/release/tools/verify_s data/S488.txt | tail -1
RESULT ok=1 size=488 antipodal=1 max_offdiag=8 gram=-32:244,-8:36544,0:45496,8:36544 free=0 maximal=1 tight_hist=2:48,4:208,5:256,6:608,7:576,8:3850,9:16376,10:35712,11:47136,12:43636,13:26648,14:13408,15:3216,16:2698,17:736,18:340,19:32,20:348,22:12,24:228 file="data/S488.txt" ms=75

$ .venv/bin/python python/verify_S.py data/S496.txt
file      : data/S496.txt
C source  : kiss_ref.leech.leech_min_vectors() (196560 vectors, 0.11 s)
rows      : 496
antipodal : 1
gram      : -32:248,-8:37504,0:47504,8:37504 (unordered pairs by inner product)
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 file="data/S496.txt" s=0.12

$ .venv/bin/python python/verify_S.py data/S488.txt | tail -1
RESULT ok=1 size=488 antipodal=1 max_offdiag=8 gram=-32:244,-8:36544,0:45496,8:36544 file="data/S488.txt" s=0.12

$ .venv/bin/python python/verify_dim25.py data/S496.txt
file        : data/S496.txt
S check     : ok=1 size=496 ok
config      : equatorial=196064 (C \ S)  lifted=992 (S x {+1,-1})  total=197056  cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
exact       : eq-eq: full exact pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (5.9 s)
exact       : lifted-lifted max ip = 8 (same sign needs <= 8); equatorial-lifted max ip = 16 (needs <= 16 < 8 sqrt 6)
exact       : pairs lift-lift-same=245520 lift-lift-opp=245520 lift-lift-same-x=496 eq-lift=194495488 eq-eq=19220448016 total=19415435040=C(197056,2)
exact       : count = 197056  (7.5 s)
float       : max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (46, 132876) class=eq-eq
float       : count = 197056  (23.4 s, 37249 tiles, min off-diagonal -4.000000)
RESULT ok=1 size=496 count_exact=197056 count_float=197056 max_offdiag_float=2.000000000000 max_class=eq-eq exact_s=7.5 float_s=23.4 s=31.1
real 0m31.142s   user 3m50.147s   sys 0m0.865s

$ .venv/bin/python -m pytest python/tests -q
94 passed in 40.95s
```

Corrupted scratch fixtures (a greedy 230-set from `data/leech_min.txt` with one row replaced,
duplicated, given a wrong norm, or with an octad coordinate moved) produce byte-identical
`FAIL` lines from C++ and Python, e.g. `gram: line 2 (row 0) and line 3 (row 1) have inner
product 16 > 8 (60 degrees): (…) . (…) [7 offending pair(s)]`, `distinct: line 5 (row 3) and
line 7 (row 5) are the same vector …`, `norm: line 4 (row 2) has squared norm 64 != 32 …`,
`membership: line 6 (row 4) is not a Leech minimal vector …`; exit code 1 in all four cases,
0 for the good set.

### 6.6 What the verifier runs establish

- The 496 is **antipodal and maximal**: 0 free vertices. Its tightness histogram outside S
  ranges over {4, 6..20, 22, 24} with mode 11 (52672 vertices). Because no vertex outside S has
  fewer than 4 members at 60°, no single-vertex addition and no (1,1)-swap can improve it.
  Larger swaps are the subject of `03-search-for-497.md`.
- The 488 is also antipodal and maximal, with minimum tightness 2 (48 vertices).
- Gram histograms over **unordered** pairs: 496 → {−32: 248, −8: 37504, 0: 47504, 8: 37504};
  488 → {−32: 244, −8: 36544, 0: 45496, 8: 36544}. There are no −16 pairs at all in either set.
  (The ordered-pair histograms measured on the upstream files — {−32: 496, −8: 75008, 0: 95008,
  8: 75008} for the 496 — are exactly twice these; the two are the same measurement, not a
  discrepancy.)
- The float pass's maximum off-diagonal inner product is exactly 2.0, attained by eq–eq pairs at
  60°, so the 1e-9 tolerance is genuinely needed only for rounding of the lifted coordinates
  √(2/3) and √(4/3). The lifted classes stay strictly below 2 in exact arithmetic — same-sign
  lifted pairs reach (2/3)·1 + 4/3 = 2 only at ip = 8, and those also attain 2.0 up to rounding,
  which is why the reported argmax pair may be of either class.
- `verify_S.py` implements the stated verification protocol literally: regenerate C, check
  membership, check distinctness, check the integer Gram ≤ 8, print |S|.

### 6.7 Timing

| step | time |
|---|---|
| `verify_independent` on the 496 (including 122760 dots) | < 1 ms |
| `tightness_cpu`, \|S\| = 496, 16 threads | 57–64 ms (ASan: 1.7 s) |
| `verify_s data/S496.txt` end-to-end (`load_leech` 10 ms) | 80 ms |
| `verify_S.py data/S496.txt` end-to-end (regenerating C, 0.11 s) | 0.12 s |
| `verify_dim25.py` exact pass incl. the full 1.93·10^10-pair sweep | 7.5 s |
| `verify_dim25.py` float pass, 197056²/2 entries, 37249 tiles | 23.4 s |
| `verify_dim25.py` total | 31 s |
| `test_verify` | 1.7 s release, 9.3 s ASan/UBSan |

---

## 7. External record data

Two public repositories were cloned (shallow) into `data/external/`, which is gitignored;
provenance, hashes and formats are recorded in `data/external/SOURCES.md`, and
`python/tools/inspect_external.py` performs read-only inspection (≈13 s on the system
python3/numpy 1.21).

| clone | commit | date |
|---|---|---|
| `data/external/PackingStar` | `50ea645a9805d4f29b96180550186d26a166c3be` | 2026-06-08 |
| `data/external/Kissing-Numbers` | `548a09282ea1037077a435be4157e7f464b1b146` | 2016-08-01 |

### 7.1 Current records

From Cohn's table (https://cohn.mit.edu/kissing-numbers/, fetched 2026-08-25):

| n | lower | upper | lower-bound source | upper-bound source |
|---|---|---|---|---|
| 24 | 196560 | 196560 | Leech 1967 | Levenšteĭn 1979, Odlyzko–Sloane 1979 |
| 25 | 197056 | 265006 | Ma et al. 2025, arXiv:2511.13391 | de Laat–Leijenhorst 2024 |
| 26 | 198550 | 367775 | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 27 | 200044 | 522212 | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 28 | 204520 | 752292 | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 29 | 209496 | 1075991 | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 30 | 220440 | 1537707 | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 31 | 238350 | 2213487 | Ma et al. 2025 | de Laat–Leijenhorst 2024 |

**Nothing has moved past 496 in dimensions 24–31**; the target for a new record remains an
independent set of 497 minimal vectors. (The `06-record-and-priority.md` document reports what
happened subsequently in dimension 27.)

The page carries **no "last updated" date** — none in the text, none in HTML metadata. An
internal note had described it as "last updated June 2026"; that date is unverifiable. The page
cites arXiv:2607.20359 (July 2026) and arXiv:2606.18984, so it was edited in or after July
2026. This does not affect the bounds. Outside our range, for completeness: the n = 12 lower
bound is now 841 (Takhanov–Assylbekov–Yun 2026) and n = 19 cites Ho 2026. The data archive
linked from the page is https://hdl.handle.net/1721.1/153312 (not fetched).

### 7.2 PackingStar repository

There is exactly **one explicit S file**: `25D-31D/Si_configurations/24D_496_Si.npy` — a
(496, 24) int16 array in √8-integer scaling (entries −4..4, squared norm 32). There are **no
per-dimension S_i files**; for n = 26..31 the S_i are implicit in the seven unit-vector
configurations `25D-31D/25D-31D_new_bounds_configurations/<n>D_<count>_coordinates.npy`
(float64, (count, n)). Those decompose exactly into the lifting template — each row is
`(x,0)/√32`, or `(x/(4√3), y/√3)`, or `(0,y')` — with an integer residual ≤ 8.9e−16 after
rescaling:

| n | file rows | equatorial | lifted | extra spheres | #S_i (all of size 496) | T_i partition | count check |
|---|---|---|---|---|---|---|---|
| 25 | 197056 | 196064 | 992 | 0 | 1 | 1 pair | 196560 + 496 |
| 26 | 198550 | 195568 | 2976 | 6 = K(2) | 2 | 2 triangles | 6 + 196560 + 4·496 |
| 27 | 200044 | 194080 | 5952 | 12 = K(3) | 5 | 2 triangles + 3 pairs | 12 + 196560 + 7·496 |
| 28 | 204520 | 192592 | 11904 | 24 = K(4) | 8 | 8 triangles | 24 + 196560 + 16·496 |
| 29 | 209496 | 189616 | 19840 | 40 = K(5) | 14 | 12 triangles + 2 pairs | 40 + 196560 + 26·496 |
| 30 | 220440 | 184656 | 35712 | 72 = K(6) | 24 | 24 triangles | 72 + 196560 + 48·496 |
| 31 | 238350 | 175728 | 62496 | 126 = K(7) | 42 | 42 triangles | 126 + 196560 + 84·496 |

For every file: the S_i are pairwise disjoint; each S_i has norm 32 with off-diagonal Gram
entries in {−32, −8, 0, 8} and the same Gram histogram as the 496 ({−32: 496, −8: 75008,
0: 95008, 8: 75008} as ordered pairs); the equatorial set is exactly C \ ∪S_i; the y-vectors
within a T_i have inner product −1/2 (−1 for pairs) and at most 1/2 across different T_i; and
the extra spheres are pairwise at cosine ≤ 1/2 and at cosine ≤ √3/2 to every T-vector.

Total S_i extracted: 96; **distinct as sets: 52**. They are nested across dimensions: the 25D S
is S_1 of 27D/29D/30D/31D; 27D's S_2..S_5 are 29D/30D/31D's S_2..S_5; 29D's S_6..S_14 are
30D/31D's S_6..S_14; 30D's S_15..S_24 are 31D's S_15..S_24. The 26D pair and the 28D eight are
not reused. So the **31D family of 42 disjoint 496-sets is the superset that matters** for
constructing disjoint families (see `05-lifting-template-and-families.md`). All 52 are
antipodally closed.

Auxiliary files: `partitioned_D5.npy` (int64 (40,5), D5 roots, norm 2) and `partitioned_E7.npy`
(float64 (126,7), unit E7 roots) give the R^5 and R^7 kissing configurations in partition order.

**Correction to a natural assumption: `24D_496_Si.npy` is *not* the set lifted in the 25D
file.** It has the same Gram histogram but different vectors; it is in fact S_8 of the
29D/30D/31D configurations. Any statement about "the 496" should say *which* one — all 52 are
Gram-identical but they are not all the same set.

Furthermore, the 52 distinct S_i are **not** all related by coordinate permutations and sign
changes. Monomial-invariant shape counts (#(±2^8), #(∓3,±1^23), #(±4,±4)) are (258, 232, 6) for
`24D_496_Si.npy` but take 18 different values across the 52 sets. The disjoint copies were
therefore made with genuine, non-monomial Leech automorphisms — which is what
`05-lifting-template-and-families.md` has to reproduce.

### 7.3 Kallal–Kan–Wang repository

`S_1.txt … S_59.txt` (one vector per line, 24 integers, norm 32, the same √8 scaling),
`minvects.txt` (196560 × 24 integers), and `Vbasis.txt` (24 × 24, a basis of minimal vectors
with det Gram = 8^24 = 4.72237e+21).

**The 488-set is `S_1.txt`.** Sizes across the 59 sets: 488 ×24, 486, 484 ×4, 482 ×4, 480 ×4,
478 ×3, 476, 474 ×5, 472, 468 ×6, 466 ×2, 464 ×2, 462, 460 — sum 28324. All 59 are pairwise
disjoint, all independent, all antipodally closed. `S_1`'s ordered-pair Gram histogram is
{−32: 488, −8: 73088, 0: 90992, 8: 73088}, max off-diagonal 8.

**They provide 59 sets, not 51** as had been assumed internally. Their 2018 template applied to
these sizes gives 198512 / 199976 / 204368 / 208272 / 219984 / 232878 for n = 26..31. The
PackingStar paper's "previous work" column prints 232874 for n = 31 — a typo on their side,
immaterial to anything here.

### 7.4 Formats and scaling

| set | file | container | dtype | scaling | diag | off-diagonal multiset (ordered pairs) |
|---|---|---|---|---|---|---|
| 496 | `PackingStar/25D-31D/Si_configurations/24D_496_Si.npy` | `.npy` (496,24) | int16 | √8-integer, norm 32 | 32 | {−32: 496, −8: 75008, 0: 95008, 8: 75008} |
| 488 | `Kissing-Numbers/S_1.txt` | text, space-separated | int | √8-integer, norm 32 | 32 | {−32: 488, −8: 73088, 0: 90992, 8: 73088} |
| C (both) | `24D_196560_coordinates.npy` (float64, integral) / `minvects.txt` | — | — | √8-integer, norm 32 | 32 | {−32:1, −16:4600, −8:47104, 0:93150, 8:47104, 16:4600, 32:1} |

Neither set is in the norm-4 scaling; both are already integer √8-scaled. Both repositories use
**the same 196560-vector set** (set equality verified; the row orders differ), so a single
coordinate map serves both.

### 7.5 The 238078-versus-238350 question, resolved

- arXiv v1 (2025-11-17) reported n = 31 → **238078★**, with the footnote "under our new
  construction form, the theoretical lower bound in 31 dimensions should be 238350. The present
  version reports 238078 only due to computational and time limitations".
- The repository's `25D-31D_new_bounds_configurations.zip` was deleted and re-uploaded on
  2026-01-15 (commits `bb57cf509703`, `1e9851a5e095`); the `31D_238350_coordinates.npy` inside
  is dated 2026-01-14, the other six files 2025-11-04..17. arXiv v2 followed on 2026-01-21;
  v4 (2026-06-02) prints "31: 238350 = 196560 + 84·496 + 126" and no longer mentions 238078.
- The file we hold has 238350 rows and decomposes into 42 disjoint 496-sets, 42 triangles and
  126 extra spheres (§7.2).

**The 238350 data exist; 238078 is v1-only history.** Cohn's table's 238350 entry is backed by
data. What was checked here is the template decomposition and the per-S_i Gram properties; a
full pairwise-cosine verification of the seven big configurations is a separate matter, and the
one dimension where it was carried out end-to-end is dimension 25 (§6.3).

### 7.6 The coordinate map

**Both repositories use a different labelling of the 24 coordinates from our cyclic Golay
code.** Relative to *their own* Golay code they follow the same conventions we do (checked on
the PackingStar list): every (±2^8, 0^16) vector has an even number of minus signs, every
(∓3, ±1^23) vector has its ≡ 3 (mod 4) class in their code with Σ ≡ 4 (mod 8), and their 759
octad supports span a 12-dimensional code. But under identity coordinates their 759 octads and
the 759 octads of our code from g(x) = x^11+x^10+x^6+x^5+x^4+x^2+1 have **zero octads in
common**.

A permutation π ∈ S_24 with π(their octads) = our octads therefore suffices: π preserves
parities, sums and residue classes, so it maps the set of norm-32 vectors passing the
membership test with respect to their code onto the same set with respect to ours, and both
sets have 196560 elements. **No sign pattern is needed.** (Had one been needed, only Golay
codeword sign patterns could have worked: negating a coordinate set D preserves the
octad-shape vectors iff |D ∩ octad| is even for every octad, i.e. D ∈ (octad span)^⊥ = G24,
since the code is self-dual. The tool tries those 4096 patterns as a fallback.)

This settles an open question: the earlier expectation was that mapping the external
coordinates might require a general Leech automorphism. It does not — a pure coordinate
permutation, with no signs at all, suffices for both repositories.

**How π was found.** `python/tools/convert_external.py find-map` backtracks over the 24
coordinates. Coordinates are assigned in a greedy order (each next point chosen to lie on as
many of their octads with ≥ 4 already-assigned points as possible, so constraints bite early);
after each assignment, for every one of their octads through the new point that has ≥ 5
assigned points, the unique our-octad through the images of the first five (using the S(5,8,24)
property via a `dict` from the 42504 five-subsets to their octad) must contain the images of
the rest. The first complete assignment is accepted: **102 nodes, 5 ms.**

```
perm = [0, 1, 2, 3, 4, 5, 15, 20, 6, 19, 16, 12, 8, 18, 22, 23, 11, 9, 13, 10, 14, 7, 21, 17]
signs = all +1
convention: ours[perm[i]] = signs[i] * theirs[i]     (i = their coordinate)
inverse:  [0, 1, 2, 3, 4, 5, 8, 21, 12, 17, 19, 16, 11, 18, 20, 6, 10, 23, 13, 9, 7, 22, 14, 15]
```

Their coordinates 0–5 are fixed; the map permutes the other 18. Any π∘m with m ∈ M24 acting on
their side is equally valid — there are 244,823,040 such maps — and this is the deterministic
first one. It is recorded in `data/external/coordinate_map.json` together with how it was found,
source hashes, and verification counts.

**Verification of the map:**

| check | result |
|---|---|
| π maps their 759 octads onto our 759 (as sets of masks) | yes |
| π(PackingStar `24D_196560_coordinates.npy`) == Python-reference C (sets of tuples) | yes, 196560 distinct images |
| all 196560 images pass `kiss_ref.leech.is_lattice_vector_rows` | 196560/196560, norms {32} |
| π(KKW `minvects.txt`) == C++ `data/leech_min.txt` as a set (independent second source, second reference) | yes |
| `find-map --ref data/external/Kissing-Numbers/minvects.txt` (text source instead of the npy) | returns the identical π, 102 nodes |
| `data/S496.txt`, `data/S488.txt` ⊂ `data/leech_min.txt` | yes |

Hence π is a bijection of their C onto ours and preserves every inner product, so all the
Gram-level facts measured in the repositories' own coordinates transfer verbatim. After
applying π, the Kallal–Kan–Wang list *is* our C coordinate-by-coordinate as a set — a stronger
statement than the "compare via membership test and histogram, not coordinate-by-coordinate"
that had been anticipated.

### 7.7 The converted fixtures

| file | source | sha256 |
|---|---|---|
| `data/S496.txt` | PackingStar `24D_496_Si.npy` | `139c1a3080ed0c4e789625468798eaaf7bd3ec29f2f33b073df659ab61a87a5f` |
| `data/S488.txt` | Kallal–Kan–Wang `S_1.txt` | `216e3c68ab8d25c289a6daf4990eac3f94cb27dd5b351d305c5289f5593b4799` |

Both are in the S text format with six header comment lines giving the label, source path and
its sha256, the repository URL and commit, the map, the conversion date, and the row order
(source order, so row k of the file is row k of the upstream file, mapped). They were written by
`convert_external.py convert`, then re-read and compared row-for-row against the in-memory
conversion.

```
$ .venv/bin/python python/tools/convert_external.py verify data/S496.txt data/S488.txt
data/S496.txt: size=496 distinct=True norm32=True in_C=496/496 membership_test=496/496 max_offdiag=8 gram_hist={-32: 496, -8: 75008, 0: 95008, 8: 75008} antipodal_closed=True
RESULT ok=1 size=496 file=data/S496.txt
data/S488.txt: size=488 distinct=True norm32=True in_C=488/488 membership_test=488/488 max_offdiag=8 gram_hist={-32: 488, -8: 73088, 0: 90992, 8: 73088} antipodal_closed=True
RESULT ok=1 size=488 file=data/S488.txt
```

Every row is a minimal vector, both by set membership in the independently generated Python C
and by the arithmetic membership test; rows are distinct; squared norm 32; Gram off-diagonals
take only the values {−32, −8, 0, 8} (no 16 and no −16); sizes 496 and 488; both antipodally
closed. The C++ `verify_s` and `verify_S.py` outputs for these two files are in §6.5 and agree.

**Converting the rest.** The tool takes any input path, so each of the 59 KKW sets is one
command:

```
.venv/bin/python python/tools/convert_external.py convert --txt data/external/Kissing-Numbers/S_7.txt --out runs/families/kkw/S_07.txt
.venv/bin/python python/tools/convert_external.py convert --npy some_S_i.npy --out ... [--sort] [--label "..."]
.venv/bin/python python/tools/convert_external.py verify runs/families/kkw/S_*.txt
```

PackingStar has no per-dimension S_i files, so the 52 distinct S_i must first be extracted from
the seven `<n>D_<count>_coordinates.npy` files with `decompose_config` in
`python/tools/inspect_external.py` (lifted rows × 4√3 → integers, grouped by their T-vector),
saved as `.npy` or text in the repositories' coordinates, and then converted with the same map
— both repositories share the coordinate system. The 31D file's 42 sets are the superset that
matters.

### 7.8 Cross structure of the 496

The description in circulation is that the 496 decomposes into "28 cross structures of 16
vectors plus one 24-dimensional cross of 48". This was examined with
`convert_external.py crosses FILE`, which builds the 248 antipodal classes, forms the graph
with an edge iff the class representatives are orthogonal (⟨x,y⟩ = 0 is sign-invariant), runs
Bron–Kerbosch with pivoting on integer bitmasks for all maximal cliques, counts k-cliques, and
runs an exact-cover search for a partition of the classes into 8-cliques plus a chosen set of
24-cliques (branching on the uncovered vertex of smallest uncovered degree; enumerating the
7-cliques in its uncovered neighbourhood; pruning when any uncovered vertex has fewer than 7
uncovered neighbours). Each cover search gets a time budget, default 120 s.

| quantity | value |
|---|---|
| antipodal classes | 248 |
| orthogonality degree distribution | {79: 16, 87: 88, 95: 4, 99: 32, 103: 72, 107: 32, 111: 4} |
| relation "orthogonal" transitive on classes | **no** |
| maximal cliques | 2718: sizes {8: 1280, 10: 800, 12: 448, 16: 171, 18: 4, 20: 8, 22: 4, 24: 3} |
| clique number | 24 |
| 24-cliques = full frames (48 vectors, 24 orthogonal pairs) | **3**, pairwise disjoint |
| 8-cliques (all, not only maximal) | 7,069,615 |
| partition into 1 frame + 28 octuples | **exists for each of the three frames** (frame #2: 4268 nodes / 0.2 s; frame #0: 508,579 nodes / 31 s; frame #1: 7,648,135 nodes / 453 s in a separate run with the same search) |
| partition into all 3 frames + 22 octuples | unresolved (see below) |
| partition into 31 octuples, no frame | unresolved in 120 s (3.2 M nodes) — not needed for the claim |

So the "28 crosses of 16 plus one cross of 48" description is **true as an existence
statement**: the 248 antipodal classes can be partitioned into 28 mutually-orthogonal octuples
and one 24-clique. The explicit partitions were verified inside the tool (cliques checked pair
by pair, disjointness, exact cover of the 224 remaining classes).

**But the decomposition is not intrinsic.** Two earlier claims need correcting:

1. It had been noted that the 28×16 + 48 structure was "not confirmed" because the relation
   ⟨x,y⟩ ∈ {0, −32} is not transitive on the 496 (each vector is orthogonal or antipodal to
   159–223 others), so the "crosses" are not equivalence classes. That observation stands —
   and the resolution is that the well-defined objects are maximal cliques and exact covers,
   not equivalence classes. Phrased that way, the structure does exist.
2. But the 496 contains **three** disjoint full frames, not one, and any of the three can play
   the role of "the 24-dimensional cross". With 7.07 M candidate octuples the exact-cover
   search finds a different partition on every restart (three distinct partitions for frame #2
   within 0.3 s), so the 28 octuples are not determined by the set.

No obvious invariant singles out a canonical family of octuples. For the partition found with
frame #2, the number of minimal vectors in the 8-dimensional span of an octuple is
{16: 3, 32: 4, 48: 2, 64: 7, 96: 2, 128: 10}; with frame #0 it is {32: 13, 64: 3, 96: 1,
128: 9, 240: 2}, where 240 marks a √2·E8 section — those two octuples are genuine E8 crosses
and the others are not. A random sample of 3000 8-cliques of the graph gives
{16: 1632, 32: 1221, 48: 17, 64: 77, 96: 39, 112: 7, 128: 7}. Moreover the upstream row order of
`24D_496_Si.npy` carries no block structure (no 16 or 48 consecutive rows are mutually
orthogonal or antipodal; antipodes are not adjacent rows), so the file does not record the
authors' decomposition either.

Consequence for a frame-structured search (`03-search-for-497.md`): "28 subframes + 1 frame"
should be treated as one of many admissible decompositions. The natural rigid objects in the
496 are its **three full frames** and its 2718 maximal cliques, and a frame-level search should
be seeded with all three frames, not one.

**The 488, for comparison:** 244 classes; degrees
{83: 8, 85: 16, 87: 48, 89: 72, 91: 40, 93: 8, 107: 8, 109: 8, 111: 20, 113: 16}; 1683 maximal
cliques of sizes {8: 472, 10: 523, 12: 486, 14: 37, 16: 139, 18: 15, 20: 6, 22: 2, 24: 3};
clique number 24; **three full frames, but not disjoint** (frames #0 and #1 share 8 classes);
6,362,969 8-cliques. Since 244 ≡ 4 (mod 8), no partition into 8- and 24-cliques exists at all —
the 488 is not "crosses" in this sense, although it too contains full frames.

**Open item.** Whether the 176 classes outside the three frames of the 496 can be partitioned
into 22 octuples was **not decided**: the 120 s budget in the main run exhausted at 3.98 M nodes
without an answer, and a dedicated 1500 s run was set up but its result was never recorded in
the working notes. This is genuinely open, not negative.

### 7.9 Pasted output: `convert_external.py all --budget 120`

```
their octads: 759, ours: 759, common under identity: 0
permutation found: perm=[0, 1, 2, 3, 4, 5, 15, 20, 6, 19, 16, 12, 8, 18, 22, 23, 11, 9, 13, 10, 14, 7, 21, 17] (backtracking nodes=102)
perm alone maps their 196560-set onto ours: True
mapped rows: distinct=196560, in C=196560, pass membership test=196560, norms=[32]
wrote data/external/coordinate_map.json
RESULT ok=1 perm=0,1,2,3,4,5,15,20,6,19,16,12,8,18,22,23,11,9,13,10,14,7,21,17 negated=0 nodes=102 time_s=1.9
wrote data/S496.txt (496 rows)
data/S496.txt: size=496 distinct=True norm32=True in_C=496/496 membership_test=496/496 max_offdiag=8 gram_hist={-32: 496, -8: 75008, 0: 95008, 8: 75008} antipodal_closed=True
RESULT ok=1 size=496 file=data/S496.txt
wrote data/S488.txt (488 rows)
data/S488.txt: size=488 distinct=True norm32=True in_C=488/488 membership_test=488/488 max_offdiag=8 gram_hist={-32: 488, -8: 73088, 0: 90992, 8: 73088} antipodal_closed=True
RESULT ok=1 size=488 file=data/S488.txt
data/S496.txt: 496 vectors, 248 antipodal classes; orthogonality graph degree distribution {79: 16, 87: 88, 95: 4, 99: 32, 103: 72, 107: 32, 111: 4}
  relation 'orthogonal' transitive on classes: False
  maximal cliques: 2718, size distribution {8: 1280, 10: 800, 12: 448, 16: 171, 18: 4, 20: 8, 22: 4, 24: 3}, clique number 24
  24-cliques (full frames): 3, pairwise disjoint: True
    frame #0: classes [115, 125, 128, 129, 134, 142, 146, 161, 165, 171, 175, 177, 178, 179, 180, 181, 184, 191, 200, 208, 215, 220, 235, 240]
    frame #1: classes [114, 120, 122, 124, 143, 150, 162, 163, 167, 172, 182, 193, 197, 202, 205, 206, 210, 217, 225, 232, 233, 234, 243, 245]
    frame #2: classes [112, 118, 123, 132, 139, 140, 141, 147, 149, 153, 158, 166, 170, 173, 176, 187, 190, 203, 213, 218, 224, 230, 236, 246]
  8-cliques (all, not only maximal): 7069615
  partition frame #0 + 28 octuples: FOUND (nodes 508579, 31.3s); |C ∩ span| per octuple: {32: 13, 64: 3, 96: 1, 128: 9, 240: 2}
  partition frame #1 + 28 octuples: unresolved (timeout) (nodes 2463153, 120.0s)
  partition frame #2 + 28 octuples: FOUND (nodes 4268, 0.2s); |C ∩ span| per octuple: {16: 3, 32: 4, 48: 2, 64: 7, 96: 2, 128: 10}
  partition all 3 frames + 22 octuples: unresolved (timeout) (nodes 3982913, 120.0s)
  partition 31 octuples, no frame: unresolved (timeout) (nodes 3211618, 120.0s)
  |C ∩ span| for frames: [196560, 196560, 196560] (expect 196560)
RESULT file=data/S496.txt classes=248 maximal_cliques=2718 omega=24 frames=3 eight_cliques=7069615 partition_1frame_28oct=yes time_s=403.0
data/S488.txt: 488 vectors, 244 antipodal classes; orthogonality graph degree distribution {83: 8, 85: 16, 87: 48, 89: 72, 91: 40, 93: 8, 107: 8, 109: 8, 111: 20, 113: 16}
  relation 'orthogonal' transitive on classes: False
  maximal cliques: 1683, size distribution {8: 472, 10: 523, 12: 486, 14: 37, 16: 139, 18: 15, 20: 6, 22: 2, 24: 3}, clique number 24
  24-cliques (full frames): 3, pairwise disjoint: False
    frame #0: classes [2, 5, 10, 11, 29, 55, 57, 86, 88, 94, 103, 104, 106, 108, 116, 121, 129, 158, 165, 181, 194, 206, 208, 239]
    frame #1: classes [2, 11, 17, 44, 46, 57, 79, 89, 96, 130, 147, 152, 157, 158, 162, 165, 169, 181, 194, 206, 223, 224, 235, 242]
    frame #2: classes [4, 13, 21, 35, 38, 53, 61, 67, 81, 97, 99, 109, 135, 141, 145, 149, 156, 159, 168, 195, 201, 205, 241, 243]
  8-cliques (all, not only maximal): 6362969
  244 classes is not a multiple of 8: no partition into 8- and 24-cliques exists
  |C ∩ span| for frames: [196560, 196560, 196560] (expect 196560)
RESULT file=data/S488.txt classes=244 maximal_cliques=1683 omega=24 frames=3 eight_cliques=6362969 partition_1frame_28oct=no time_s=6.6
```

Frame #1 + 28 octuples (same search, 900 s budget, separate run): `SOLUTION` after 7,648,135
nodes, 452.8 s. The octuples, as class indices numbered by `antipodal_classes` on
`data/S496.txt` (i.e. by first occurrence in source order):

```
[80,83,85,86,100,101,103,128] [88,90,92,93,95,96,98,129] [104,121,157,171,179,180,192,194] [89,91,99,106,108,109,110,175]
[81,82,84,87,94,97,102,146] [115,116,133,161,164,183,191,200] [107,112,118,123,139,147,213,236] [31,33,34,35,36,39,41,44]
[53,54,60,72,73,74,77,79] [113,135,137,149,170,185,187,246] [2,4,7,9,12,15,18,40] [22,43,49,56,64,69,70,75]
[0,1,6,11,42,45,48,158] [3,8,16,19,46,51,57,190] [14,17,47,52,131,156,199,212] [5,10,65,66,144,151,201,231]
[13,23,59,68,140,155,173,176] [20,21,63,67,153,203,224,230] [25,26,29,30,55,58,216,229] [37,38,127,130,138,145,159,188]
[134,142,152,169,198,223,237,241] [24,27,71,78,111,174,211,222] [61,148,189,209,214,219,228,247] [50,177,178,181,208,215,235,240]
[125,132,141,165,166,184,218,220] [105,117,154,168,186,204,207,238] [119,126,136,196,227,239,242,244] [28,32,62,76,160,195,221,226]
```

Cross-checks run outside the tool: π(KKW `minvects.txt`) == `data/leech_min.txt` as a set
(196560 elements); `S496.txt` and `S488.txt` ⊂ `data/leech_min.txt`; the upstream row orders
carry no cross blocks.

### 7.10 Pasted output: `python/tools/inspect_external.py`

```
== PackingStar: Si_configurations/24D_496_Si.npy
  24D_496_Si: n=496 dim=24 dtype=int16 coords=[-4, -3, -2, -1, 0, 1, 2, 3, 4]
      diag=[32] off-diag multiset={-32: 496, -8: 75008, 0: 95008, 8: 75008} max_off=8 distinct=496 antipodal=True independent(max<=8)=True
  24D_496_Si: |{y: <x,y> in {0,-32}}| distribution = {159: 32, 175: 176, 191: 8, 199: 64, 207: 144, 215: 64, 223: 8}; relation transitive = False
== PackingStar: Si_configurations/24D_196560_coordinates.npy
  shape=(196560, 24) dtype=float64 integral=True norms=[32.0]
  distinct=196560; ip histogram vs row 0: {-32: 1, -16: 4600, -8: 47104, 0: 93150, 8: 47104, 16: 4600, 32: 1}
  496 subset of PackingStar 196560: True
== Kissing-Numbers: minvects.txt, Vbasis.txt
  minvects shape=(196560, 24) distinct=196560 norms=[32] ip histogram vs row 0: {-32: 1, -16: 4600, -8: 47104, 0: 93150, 8: 47104, 16: 4600, 32: 1}
  shape counts: {'(±2^8,0^16)': 97152, '(∓3,±1^23)': 98304, '(±4,±4,0^22)': 1104}
  PackingStar 196560 set == KKW minvects set: True; same row order: False
  496 subset of KKW minvects: True
  Vbasis shape=(24, 24) det(Gram)=4.72237e+21 (expect 8^24=4.72237e+21 for sqrt8-scaled Leech); diag=[32]
  Vbasis rows all in minvects: True
== Octads vs cyclic Golay code (identity coordinates)
  octads in minvects: 759; cyclic-Golay octads: 759; common: 0
  GF(2)-span of minvects octads has dimension 12 (expect 12)
== Kissing-Numbers: S_1..S_59
  S_1: n=488 dim=24 dtype=int64 coords=[-4, -3, -2, -1, 0, 1, 2, 3, 4]
      diag=[32] off-diag multiset={-32: 488, -8: 73088, 0: 90992, 8: 73088} max_off=8 distinct=488 antipodal=True independent(max<=8)=True
  S_2: n=488 ... {-32: 488, -8: 73088, 0: 90992, 8: 73088} max_off=8 distinct=488 antipodal=True independent(max<=8)=True
  S_24: n=488 ... {-32: 488, -8: 73088, 0: 90992, 8: 73088} max_off=8 distinct=488 antipodal=True independent(max<=8)=True
  S_25: n=486 ... {-32: 486, -8: 72472, 0: 90280, 8: 72472} max_off=8 distinct=486 antipodal=True independent(max<=8)=True
  S_59: n=460 ... {-32: 460, -8: 65000, 0: 80680, 8: 65000} max_off=8 distinct=460 antipodal=True independent(max<=8)=True
  sizes S_1..S_59: [488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 488, 486, 484, 484, 484, 484, 482, 482, 482, 482, 480, 480, 480, 480, 478, 478, 478, 476, 474, 474, 474, 474, 474, 472, 468, 468, 468, 468, 468, 468, 466, 466, 464, 464, 462, 460]
  sets failing norm-32 / off-diag<=8: []
  overlapping pairs: []; all subsets of minvects: True; sum=28324
  S_1 (488): |{y: <x,y> in {0,-32}}| distribution = {167: 16, 171: 32, 175: 96, 179: 144, 183: 80, 187: 16, 215: 16, 219: 16, 223: 40, 227: 32}; relation transitive = False
  KKW 2018 template values from these sizes: {26: 198512, 27: 199976, 28: 204368, 29: 208272, 30: 219984, 31: 232878}
== PackingStar 25D-31D configurations: implicit S_i / T_i decomposition
  25D_197056_coordinates.npy: n=197056 dim=25 | equatorial=196064 lifted=992 extra=0 (K(1)=2)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 0.00e+00
      distinct T-vectors=2; groups (S_i,T_i)=1; |T_i| distribution={2: 1}; |S_i| distribution={496: 1}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 0 + 196560 + 496 = 197056  (file count 197056, match=True; K(1)=2)
      S_i pairwise disjoint=True; |union S_i|=496; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 1/1
      T inner products within T_i={-1.0: 1}; across T_i max=None
      S_i equal (as sets) to 24D_496_Si.npy: []
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
  26D_198550_coordinates.npy: n=198550 dim=26 | equatorial=195568 lifted=2976 extra=6 (K(2)=6)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 4.44e-16
      distinct T-vectors=6; groups (S_i,T_i)=2; |T_i| distribution={3: 2}; |S_i| distribution={496: 2}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 6 + 196560 + 1984 = 198550  (file count 198550, match=True; K(2)=6)
      S_i pairwise disjoint=True; |union S_i|=992; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 2/2
      T inner products within T_i={-0.5: 4, -0.499999: 2}; across T_i max=0.5
      extra spheres: 6; pairwise cos max=0.5; max cos(extra, T)=0.866025 (need <= sqrt3/2=0.866025)
      S_i equal (as sets) to 24D_496_Si.npy: []
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
  27D_200044_coordinates.npy: n=200044 dim=27 | equatorial=194080 lifted=5952 extra=12 (K(3)=12)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 4.44e-16
      distinct T-vectors=12; groups (S_i,T_i)=5; |T_i| distribution={2: 3, 3: 2}; |S_i| distribution={496: 5}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 12 + 196560 + 3472 = 200044  (file count 200044, match=True; K(3)=12)
      S_i pairwise disjoint=True; |union S_i|=2480; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 5/5
      T inner products within T_i={-1.000001: 2, -1.0: 1, -0.5: 4, -0.499999: 2}; across T_i max=0.500001
      extra spheres: 12; pairwise cos max=0.5; max cos(extra, T)=0.853554 (need <= sqrt3/2=0.866025)
      S_i equal (as sets) to 24D_496_Si.npy: []
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
  28D_204520_coordinates.npy: n=204520 dim=28 | equatorial=192592 lifted=11904 extra=24 (K(4)=24)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 8.88e-16
      distinct T-vectors=24; groups (S_i,T_i)=8; |T_i| distribution={3: 8}; |S_i| distribution={496: 8}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 24 + 196560 + 7936 = 204520  (file count 204520, match=True; K(4)=24)
      S_i pairwise disjoint=True; |union S_i|=3968; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 8/8
      T inner products within T_i={-0.500001: 2, -0.5: 20, -0.499999: 2}; across T_i max=0.500001
      extra spheres: 24; pairwise cos max=0.5; max cos(extra, T)=0.866026 (need <= sqrt3/2=0.866025)
      S_i equal (as sets) to 24D_496_Si.npy: []
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
  29D_209496_coordinates.npy: n=209496 dim=29 | equatorial=189616 lifted=19840 extra=40 (K(5)=40)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 0.00e+00
      distinct T-vectors=40; groups (S_i,T_i)=14; |T_i| distribution={2: 2, 3: 12}; |S_i| distribution={496: 14}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 40 + 196560 + 12896 = 209496  (file count 209496, match=True; K(5)=40)
      S_i pairwise disjoint=True; |union S_i|=6944; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 14/14
      T inner products within T_i={-0.5: 38}; across T_i max=0.5
      extra spheres: 40; pairwise cos max=0.5; max cos(extra, T)=0.853554 (need <= sqrt3/2=0.866025)
      S_i equal (as sets) to 24D_496_Si.npy: [8]
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
  30D_220440_coordinates.npy: n=220440 dim=30 | equatorial=184656 lifted=35712 extra=72 (K(6)=72)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 4.44e-16
      distinct T-vectors=72; groups (S_i,T_i)=24; |T_i| distribution={3: 24}; |S_i| distribution={496: 24}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 72 + 196560 + 23808 = 220440  (file count 220440, match=True; K(6)=72)
      S_i pairwise disjoint=True; |union S_i|=11904; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 24/24
      T inner products within T_i={-0.5: 66, -0.499999: 6}; across T_i max=0.5
      extra spheres: 72; pairwise cos max=0.5; max cos(extra, T)=0.866025 (need <= sqrt3/2=0.866025)
      S_i equal (as sets) to 24D_496_Si.npy: [8]
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
  31D_238350_coordinates.npy: n=238350 dim=31 | equatorial=175728 lifted=62496 extra=126 (K(7)=126)
      lifted head-norm^2 values=[0.666667] (expect 2/3=0.666667), tail-norm^2=[0.333333] (expect 1/3)
      integrality residual: equatorial*sqrt32 0.00e+00, lifted*4sqrt3 0.00e+00
      distinct T-vectors=126; groups (S_i,T_i)=42; |T_i| distribution={3: 42}; |S_i| distribution={496: 42}
      formula #extra + 196560 + sum(|T_i|-1)|S_i| = 126 + 196560 + 41664 = 238350  (file count 238350, match=True; K(7)=126)
      S_i pairwise disjoint=True; |union S_i|=20832; equatorial∩union=0; |eq|+|union|=196560
      every S_i independent (norm 32, off-diag<=8)=True; S_i with Gram histogram identical to the 496 = 42/42
      T inner products within T_i={-0.5: 104, -0.499999: 22}; across T_i max=0.5
      extra spheres: 126; pairwise cos max=0.5; max cos(extra, T)=0.853554 (need <= sqrt3/2=0.866025)
      S_i equal (as sets) to 24D_496_Si.npy: [8]
      union of S_i subset of KKW minvects: True; equatorial subset: True; equatorial == C \ union: True
== Identity of S_i sets across dimensions
  total S_i extracted: 96; distinct as sets: 52; 24D_496_Si.npy among them: True
  sets appearing in more than one place: {((25, 1), (27, 1), (29, 1), (30, 1), (31, 1)): 5, ((27, 2), (29, 2), (30, 2), (31, 2)): 4, ((27, 3), (29, 3), (30, 3), (31, 3)): 4, ((27, 4), (29, 4), (30, 4), (31, 4)): 4, ((27, 5), (29, 5), (30, 5), (31, 5)): 4, ((29, 6), (30, 6), (31, 6)): 3, ((29, 7), (30, 7), (31, 7)): 3, ((29, 8), (30, 8), (31, 8)): 3, ((29, 9), (30, 9), (31, 9)): 3, ((29, 10), (30, 10), (31, 10)): 3, ((29, 11), (30, 11), (31, 11)): 3, ((29, 12), (30, 12), (31, 12)): 3, ((29, 13), (30, 13), (31, 13)): 3, ((29, 14), (30, 14), (31, 14)): 3, ((30, 15), (31, 15)): 2, ((30, 16), (31, 16)): 2, ((30, 17), (31, 17)): 2, ((30, 18), (31, 18)): 2, ((30, 19), (31, 19)): 2, ((30, 20), (31, 20)): 2, ((30, 21), (31, 21)): 2, ((30, 22), (31, 22)): 2, ((30, 23), (31, 23)): 2, ((30, 24), (31, 24)): 2}
  antipodal-closed: 52/52
== Monomial-invariant shape counts (#(±2^8), #(∓3,±1^23), #(±4,±4)) per set
  24D_496_Si.npy: (258, 232, 6); KKW S_1: (248, 236, 4); distinct S_i of 25D-31D: {(218, 272, 6): 1, (238, 256, 2): 3, (246, 248, 2): 4, (228, 264, 4): 4, (272, 224, 0): 2, (222, 272, 2): 7, (244, 248, 4): 4, (254, 240, 2): 4, (236, 256, 4): 6, (258, 232, 6): 3, (268, 224, 4): 1, (230, 264, 2): 2, (252, 240, 4): 3, (270, 224, 2): 1, (234, 256, 6): 2, (264, 232, 0): 1, (242, 248, 6): 2, (220, 272, 4): 2}
  KKW S_1..S_59 shape counts: Counter({(248, 236, 4): 3, (258, 228, 2): 3, (218, 268, 2): 2, (244, 220, 4): 2, (240, 244, 4): 1, (226, 260, 2): 1, (220, 264, 4): 1, (264, 224, 0): 1, (260, 228, 0): 1, (222, 260, 6): 1, (266, 220, 2): 1, (246, 240, 2): 1, (252, 232, 4): 1, (210, 276, 2): 1, (208, 276, 4): 1, (216, 268, 4): 1, (244, 240, 4): 1, (228, 256, 4): 1, (252, 236, 0): 1, (224, 260, 4): 1, (256, 228, 2): 1, (222, 258, 4): 1, (262, 218, 4): 1, (234, 246, 4): 1, (240, 242, 2): 1, (240, 238, 4): 1, (254, 224, 4): 1, (252, 226, 4): 1, (266, 216, 0): 1, (246, 232, 2): 1, (240, 236, 4): 1, (228, 250, 2): 1, (210, 268, 2): 1, (206, 272, 0): 1, (216, 258, 4): 1, (244, 230, 4): 1, (224, 250, 2): 1, (242, 230, 2): 1, (232, 238, 4): 1, (254, 216, 4): 1, (254, 220, 0): 1, (224, 248, 2): 1, (274, 196, 2): 1, (228, 236, 4): 1, (240, 226, 2): 1, (234, 232, 2): 1, (214, 250, 4): 1, (222, 240, 4): 1, (246, 218, 2): 1, (240, 224, 0): 1, (230, 230, 4): 1, (262, 198, 2): 1, (210, 246, 4): 1})
```

### 7.11 Timing of the external tooling

| step | time |
|---|---|
| `find-map`: load the 196560×24 npy, octads, backtracking (102 nodes), full-set check, membership test, JSON | 1.9 s |
| `convert` + `verify` for both sets | < 1 s each (C generation ≈ 0.5 s, cached once) |
| `crosses` on the 496: maximal cliques 0.14 s, 8-clique count 6 s, covers 31 s + 120 s + 0.2 s + 120 s + 120 s | 403 s |
| `crosses` on the 488 (no cover search possible) | 6.6 s |
| `convert_external.py all` end to end | 6 min 52 s (≈ 10 s if the `crosses` stage is skipped) |
| `inspect_external.py` | ≈ 13 s |

Note that dimension 25 adds no extra sphere for d = 1: K(1) = 2 would require cosine ≤ √3/2
against y = ±1, which is impossible. That is why the dimension-25 count is exactly
K(24) + |S_1| = 196560 + 496 = 197056.

---

## 8. Corrections and clarifications

Collected here for visibility; each is also stated in context above.

1. **GPU throughput.** The design estimate of "≳ 10^12 pairs/s" for the fused tightness kernel
   was wrong. The correct figure on this hardware is 4.4–5.3 × 10^11 pairs/s, which is the
   integer-issue bound (8 integer instructions per pair, 64 INT32 lanes/SM, 40 SMs, ≤ 1.7 GHz).
   The kernel is at that bound. (§5.3)
2. **`24D_496_Si.npy` is not the 25D record's own S.** It has the same Gram histogram but
   different vectors; it is S_8 of the 29D/30D/31D configurations. An earlier reading assumed
   the repository's one explicit 496 file was the set lifted into 25 dimensions. Any statement
   about "the 496" must say which of the 52 distinct S_i is meant. (§7.2)
3. **The S_i are not monomially equivalent.** Their shape counts (#(±2^8), #(∓3,±1^23),
   #(±4,±4)) take 18 different values across the 52 distinct sets, so the disjoint copies were
   made with genuine non-monomial Leech automorphisms, not coordinate permutations and signs.
   (§7.2)
4. **n = 31 is 238350, not 238078.** arXiv v1 reported 238078★ with a footnote saying the true
   value under their construction should be 238350; the 238350 data were uploaded on
   2026-01-15 and v4 prints 238350. 238078 is v1-only history. (§7.5)
5. **Kallal–Kan–Wang provide 59 disjoint S_i, not 51.** Sizes 488 down to 460, summing to
   28324. (§7.3)
6. **A typo in the PackingStar paper's comparison column**: it prints 232874 for Kallal–Kan–Wang
   at n = 31; their own set sizes give 232878. Immaterial, but noted so nobody re-derives it and
   thinks they have found an error of their own. (§7.3)
7. **No Leech automorphism is needed to import the external coordinates.** The expectation was
   that a general automorphism might be required; in fact a pure coordinate permutation with no
   sign changes maps both repositories' data onto our canonical C. (§7.6)
8. **After the permutation, the external minimal-vector list is our C coordinate-by-coordinate
   as a set** — stronger than the anticipated "compare only via membership test and histogram".
   (§7.6)
9. **"Partition into maximal groups of mutually orthogonal-or-antipodal vectors" is not
   well defined**, because that relation is not transitive on the 496 (each vector is orthogonal
   or antipodal to between 159 and 223 others). The well-defined objects are maximal cliques of
   the orthogonality graph on antipodal classes, and exact covers by them. (§7.8)
10. **The "28 crosses of 16 plus one cross of 48" decomposition exists but is not canonical.**
    The 496 contains **three** pairwise disjoint full frames, not one; each admits a completion
    to a 1-frame + 28-octuple partition; and with 7.07 M candidate octuples the exact-cover
    search returns a different partition on every restart. A frame-structured search should be
    seeded with all three frames. (§7.8)
11. **The Golay shape-(b) sign convention question is settled.** Starting from (−3, +1^23) and
    starting from (+3, −1^23) generate the same 98304 vectors, related by c ↔ complement(c),
    because the all-ones word lies in G24. No fix-up was needed. (§3.2)
12. **Ordered versus unordered Gram histograms.** The histograms {−32: 496, −8: 75008, 0: 95008,
    8: 75008} (measured on the upstream files) and {−32: 248, −8: 37504, 0: 47504, 8: 37504}
    (reported by the verifiers) are the same measurement; the former counts ordered pairs, the
    latter unordered. (§6.6)
13. **`verify_dim25.py` does not use 8192-row blocks** as originally planned. Large row blocks
    are 1.6–12.9 GB of float64 and never cache-resident, and OpenBLAS does not parallelise a
    K = 25 GEMM at all (20 GFLOPS at 1 thread, 23 at 16). Upper-triangular 512×1024 tiles with a
    thread pool over row blocks and single-threaded BLAS cut the exact pass from 62 s to 5.9–8.1 s
    and the float pass from 111 s to 22–23 s. (§6.3)
14. **The "last updated June 2026" attribution for Cohn's table is unverifiable**: the page
    carries no date anywhere, and its citations place it in or after July 2026. The bounds
    themselves are unchanged. (§7.1)
15. **`python3 -m venv` does not work on this machine** — the system Python has no `ensurepip`,
    contrary to an earlier environment note. `scripts/setup_venv.sh` falls back to `virtualenv`
    and then to `venv --without-pip` plus an explicit pip install. (§1.5)
16. **The GPU memory figure.** cudart reports 7.66 GiB = 8,220,901,376 bytes; nvidia-smi reports
    the 8192 MiB physical size. The "~7.8 GB" occasionally quoted is the decimal-GB form of the
    same number. (§1.1)

---

## 9. Open items in this layer

1. **Whether the 176 antipodal classes outside the three full frames of the 496 can be
   partitioned into 22 octuples is undecided.** The 120 s exact-cover budget exhausted at
   3.98 M nodes; a 1500 s run was set up but no result was recorded. This is an open question,
   not a negative result. (§7.8)
2. **Whether the 248 classes admit a partition into 31 octuples with no frame is undecided**
   (unresolved after 3.21 M nodes in 120 s). Not needed for anything claimed here. (§7.8)
3. **The `build_adjacency(const DeviceLeech&, …)` convenience wrapper is not implemented**; the
   raw `build_adjacency_rows_device` entry point provides the functionality but requires the
   caller to supply and check a per-row counts buffer. (§5.1)
4. **A pair-class histogram kernel was not built**; the derivation from the tightness kernel is
   sketched in §5.7.
5. **The seven high-dimensional PackingStar configurations have not been verified by a full
   pairwise-cosine sweep.** What was checked is the template decomposition, the per-S_i Gram
   properties, disjointness, the equatorial complement identity, the T-vector inner products and
   the extra-sphere cosines. The one dimension verified end to end, twice and by two independent
   methods, is dimension 25 (§6.3).

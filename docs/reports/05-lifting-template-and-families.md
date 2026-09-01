# The lifting template and disjoint families

This document collects the mathematics and the verification of the *lifting template*: the
construction that takes the Leech lattice's 196560 minimal vectors, a family of pairwise disjoint
60°-free subsets of them, and a small kissing configuration in `R^d`, and assembles a spherical
code in `R^(24+d)`. It is the mechanism behind every published kissing-number lower bound in
dimensions 25–31, and it is the mechanism behind this repository's improvement in dimension 27.

Contents:

* §1 the template itself and the count formula;
* §2 the Leech automorphism machinery (M24, the monomial group `2^12:M24`, Conway's ξ, product
  replacement) that supplies the automorphisms;
* §3 the construction of disjoint families `S_1..S_k` for dimensions 26–31, including the search
  methods that fail and the one that works;
* §4 the dimension-N verifiers and their acceptance runs;
* §5 optimality of the `R^d` half — triangle partitions, extra spheres, and the dimension-27
  partition-weight improvement `K(27) >= 200540`;
* §6 the sweep over `n >= 32` showing the template cannot win there, with the structural reason;
* §7 corrections to earlier analyses in this repository;
* §8 files, commands, and the consolidated `RESULT` lines.

The narrative of the dimension-27 record, the priority check and the independent co-discovery live
in `06-record-and-priority.md`, as does the verified `K(25) >= 197058` lowered-lift observation.
The Golay/Leech construction, the canonical order on the 196560 minimal vectors, the coordinate map
onto the external record data and the base verifiers are in `01-lattice-and-verification.md`.
Upper bounds on the size of a 60°-free subset are in `02-upper-bounds.md`; the search for a 497-set
is in `03-search-for-497.md`; the internal structure of the 496-sets is in
`04-structure-of-the-496.md`.

---

## 1. The template

Let `C` be the set of 196560 minimal vectors of the Leech lattice, scaled so that every vector has
squared norm 32 and all inner products are integers (two vectors are at 60° exactly when their
inner product is 16). A subset `S ⊂ C` is *independent* (60°-free) when its Gram matrix has all
off-diagonal entries `<= 8`. The largest such set known has 496 elements.

Fix a dimension `d >= 1` and a kissing configuration `T` in `R^d`: `K(d)` unit vectors with pairwise
`cos <= 1/2`. Partition (part of) `T` into groups `T_1, ..., T_k` with pairwise `cos <= -1/2`
*inside* each group. Because four unit vectors with pairwise inner product `<= -1/2` would have
`|Σ|^2 <= 4 - 6 < 0`, every group has `|T_i| <= 3`; a group of size 3 is an equilateral triangle
(all pairwise `cos = -1/2`) and a group of size 2 is either an antipodal pair or two vertices of a
triangle. Define the *weight* of group `i` to be `|T_i| - 1`.

Choose pairwise disjoint independent sets `S_1, ..., S_k ⊂ C`, one per group. Optionally choose a
set `E` of *extra spheres*: unit vectors in `R^d` with pairwise `cos <= 1/2` and `cos <= √3/2`
(i.e. at least 30°) to every vector of `T`.

The configuration in `R^(24+d)`, in the scaling where every row has squared norm 4, is

```
equatorial :  (x/√8, 0)                                   for every x ∈ C
lifted     :  (√(2/3)·x/√8, √(4/3)·u(y))                  for x ∈ S_i, y ∈ T_i
extra      :  (0, 2·u(y'))                                for y' ∈ E
```

where `u(·)` is the unit vector in the given direction. Every equatorial vector appears once; a
vector of `S_i` appears once for each of the `|T_i|` group members and *not* equatorially, which is
where the `|T_i| - 1` weight comes from. The size is

```
count = |E| + 196560 + Σ_i (|T_i| − 1)·|S_i|
```

and the resulting maximum off-diagonal inner product is exactly 2, i.e. the configuration is a
kissing configuration in `R^(24+d)`.

The records for `n = 24 + d`, `d = 2..7`, all instantiate this with `|S_i| = 496` and `|E| = K(d)`.
With those two values fixed the count is determined entirely by the *partition weight*
`Σ_i (|T_i| − 1)`, which is what §5 optimises.

---

## 2. Leech automorphisms

The template needs many pairwise disjoint copies of a 496-set. Automorphisms of the Leech lattice
are the natural source: `g·S` is again independent of size 496 for every `g ∈ Co_0`, and the
stabiliser of the 496-set in `Co_0` has order 8 (`2^3`; see `03-search-for-497.md`), so essentially
every group element gives a genuinely different set.

This section describes the group machinery: `include/kiss/group.h`, `src/group.cpp`,
`python/group/m24.py`, and the data files under `data/group/`.

### 2.1 M24 from the Golay code, without octad backtracking

The Golay code used throughout (`kiss_ref.golay`, `kiss/golay.h`) is the cyclic `(23,12,7)` code with
generator polynomial `g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1`, extended by a parity bit in
coordinate 23. A cyclic Golay code *is* a quadratic-residue code on `Z/23`, so the obvious labelling
— coordinate `k` ↔ field element `k` for `k = 0..22`, coordinate 23 ↔ `∞` — was tried first,
together with the 21 other multiplier labellings `k ↔ a·k`. Computationally (`preserves_code` over
all 4096 codewords):

| map on P¹(F₂₃) | identity labelling preserves the code? |
|---|---|
| α: x → x+1 | yes (it is the cyclic shift; it works for every labelling) |
| γ: x → −1/x (0 ↔ ∞) | yes (for every labelling `a` — `−1/x` commutes with the multipliers up to a Q/N swap) |
| δ: x → x³/9 for x ∈ Q ∪ {0}, x → 9x³ for x ∈ N, ∞ fixed | **yes** for the identity labelling and for `a ∈ Q`; the swapped rule (`x³/9` on N, `9x³` on Q) works exactly for `a ∈ N` — the two variants distinguish this code from its "N-code" twin. The swapped variant is asserted to *fail* in `python/tests/test_m24.py`. |

So no octad backtracking was needed: the code is already in QR form, with
`Q = {1,2,3,4,6,8,9,12,13,16,18}`. The three generators, as images of `0..23`:

```
alpha = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 0 23   (order 23)
gamma = 23 22 11 15 17 9 19 13 20 5 16 2 21 7 18 3 10 4 14 6 8 12 1 0   (order 2)
delta = 0 18 6 3 2 21 1 5 16 12 7 19 8 9 17 15 13 11 4 22 10 20 14 23   (order 5)
```

Each of the three permutes the 4096 codewords (checked in Python by `preserves_code` and in C++ by
`preserves_golay_code`, both over the full codeword list). Schreier–Sims via sympy gives
`⟨α,γ⟩` of order **6072 = |PSL(2,23)|** and `⟨α,γ,δ⟩` of order **244 823 040 = |M24|** (0.6 s). The
transposition `(0 1)` and a random permutation do not preserve the code (negative controls).

```
$ .venv/bin/python -m pytest python/tests/test_m24.py -q
...........                                                              [100%]
11 passed in 2.50s
$ .venv/bin/python python/group/m24.py
RESULT ok=1 m24_order=244823040 psl_order=6072 out=.../data/group
```

Running `python python/group/m24.py` regenerates `data/group/m24_generators.txt` (α, γ, δ as
coordinate permutations), `data/group/sextet.txt` and `data/group/xi.txt`.

### 2.2 The monomial group 2^12:M24

`struct Monomial { std::array<uint8_t,24> perm; uint32_t signs; }` with the convention
`(m·v)[perm[i]] = s_{perm[i]} · v[i]`, where `s_j = −1` iff bit `j` of `signs` (which must be a Golay
codeword) is set — i.e. `m = D_signs · P_perm`. Composition is `compose(a,b) = a∘b` (b first) with
`signs = a.signs ^ P_a(b.signs)`; there are `inverse` and `apply`; and `index_permutation(L, m)` maps
every `C[i]` and throws naming the first index whose image is not a minimal vector. That throw *is*
the automorphism test, and the tests rely on it:

* α, γ, δ each induce a bijection of `C`, with 20 000 random inner products preserved apiece.
* The transposition `(0 1)`, a sign flip on the non-codeword `0b111`, and a sign flip on a weight-8
  non-octad are all rejected.
* `−I` (sign flip on the all-ones word) induces exactly `Leech::neg`.
* Consistency over the pool {α, γ, δ, octad flip}:
  `index_permutation(compose(a,b)) == compose(idx(a), idx(b))`, `idx(inverse(a)) == inverse(idx(a))`,
  `apply(a, apply(b, v)) == apply(a∘b, v)`, and `aut_from_monomial` agrees.
* `monomial_generators(dir)` returns {α, γ, δ, sign flip of the first octad} — 4 elements. Since M24
  is transitive on octads and the octads span the code, this generates `2^12:M24`.
  `monomial_generators(dir, true)` instead adds sign flips on a GF(2)-basis (12 words, checked to
  span all 4096), giving the 15-element generating set; 4 generators suffice and keep the product
  replacement slot count meaningful.

### 2.3 Conway's ξ (the non-monomial generator)

`struct Aut { std::array<int32_t,576> num; int32_t den; }`, row-major, with
`(A v)[r] = Σ_c num[r·24+c]·v[c] / den`. `apply` throws on a non-divisible or out-of-`int8` image
(`try_apply` reports instead); `compose` reduces by the gcd; `transpose` is available.

The sextet through `{0,1,2,3}` (`find_sextet`, from `S(5,8,24)`: `T ∪ {p}` lies in one octad; C++
and Python agree; file `data/group/sextet.txt`):

```
{0,1,2,3} {4,7,10,12} {5,14,17,23} {6,9,11,22} {8,15,16,19} {13,18,20,21}
```

ξ acts on the tetrad `T` with sign `s_T` by `x_i → s_T·((Σ_{j∈T} x_j)/2 − x_i)`, i.e. by the block
`s_T·(J − 2I)/2`. All 64 sign patterns were tested by applying the matrix to all 196560 minimal
vectors (numpy first, then the C++ `index_permutation`):

* **uniform** signs `(+,+,+,+,+,+)`: only **49 104** of the 196 560 images are minimal vectors — so
  the uniform-sign matrix is **not** an automorphism;
* every pattern with an **odd** number of negated tetrads: all 196 560 images are minimal vectors and
  distinct, giving a bijection of `C` — 32 such patterns, i.e. 16 up to global sign. Every **even**
  pattern fails, with the same 49 104 hits.

The canonical choice is `XI_TETRAD_SIGNS = {−1,+1,+1,+1,+1,+1}` (block negated on `{0,1,2,3}`); it is
what `data/group/xi.txt` and `make_xi(sextet)` produce. Verified properties: bijection of `C`; `10^5`
random inner products preserved; `ξ² = 1` both as an index permutation and as a matrix; `ξ = ξᵀ`;
every row has 4 non-zeros (so ξ is not monomial); and it mixes shapes — sampled transition counts
octad → (octad, 31, 44) = 481/509/10, 31 → 507/506/0, 44 → 11/0/3. `apply(ξ, e₀)` throws, the tetrad
sum being odd.

### 2.4 Product replacement, with an accumulator

The generating set for `Co_0` (`co0_generator_perms`) is {α, γ, δ, octad sign flip, ξ} as index
permutations (5 × 786 KB). `ProductReplacement(gens, slots=10, seed)` initialises the slots by
cycling the generators; a step picks `i ≠ j` and replaces `x_i` by one of
`x_i x_j^{±1}`, `x_j^{±1} x_i`; **and additionally maintains an accumulator** `a ← a·x_i` whose value
is the output. This is the "rattle" variant (Leedham-Green–Murray), the one GAP's `PseudoRandom`
uses.

The accumulator is not cosmetic. With plain product replacement the first run produced only **980
distinct elements in 1000 outputs** — 20 exact repeats at near-consecutive steps, e.g. `(21,29)`,
`(38,61)`, from an involution slot being multiplied in twice, or `x_j` followed by `x_j^{-1}` — and
23 collisions among the images of vertex 0, against a uniform expectation of about 2.5. With the
accumulator: 1000/1000 distinct elements, vertex-image collisions 2–4 (consistent with uniform), and
fixed-point counts of mean 0.96 (a random permutation would give 1; the counts are even because
fixed points come in ± pairs).

`tools/random_aut.cpp` exposes this:

```
random_aut --seed S --count K --out dir/ [--slots 10] [--burnin 100] [--data data] \
           [--group data/group] [--check 1]      →  dir/aut_%04d.u32
```

`random_aut --seed 1 --count 1000 --out dir/` takes 8.7 s: about 0.4 s to load and build the 5
generator index permutations, then about 8 ms per element, dominated by the 1000-pair check and the
786 KB write. It is deterministic for a fixed seed (byte-identical files).

### 2.5 How random images of the 496 behave

The group acceptance test (`tests/test_group.cpp`, ctest target `test_group`, `TIMEOUT 900`) was run
on `data/S496.txt`, the converted external record 496-set:

```
leech loaded: 71.0 ms
monomial section: 5194.3 ms            (first run; ≈1.2 s when the machine is idle)
sextet: {0,1,2,3} {4,7,10,12} {5,14,17,23} {6,9,11,22} {8,15,16,19} {13,18,20,21}
xi shape transitions (sampled, rows=from octad/31/44): | 481 509 10 | 507 506 0 | 11 0 3
xi sign patterns: 32 automorphisms, 32 not; all automorphisms have an odd number of negated tetrads: 1
product replacement: 1000 elements, ip_fail=0 bij_fail=0 distinct images of vertex 0: 998
S = data/S496.txt, |S| = 496
overlap |gS ∩ S| over 200 random elements: mean=1.300 expected(random)=1.252 histogram: 0:110 2:63 4:16 6:9 8:2
S shape counts (octad, 31, 44): 258 232 6
generator invariance of S: alpha:|gS∩S|=0 gamma:|gS∩S|=0 delta:|gS∩S|=2 octad_flip:|gS∩S|=8 neg_I:|gS∩S|=496(INVARIANT) xi:|gS∩S|=8
random monomial elements (2000): fixing S as a set: 0; overlap mean=1.275 histogram: 0:1080 2:630 4:236 6:44 8:9 10:1
RESULT ok=1 m24_gens=3 xi_bijective=1 xi_sign_patterns_ok=32 pr_elements=1000 ip_fail=0 bij_fail=0 applied=200 image_fail=0 set=data/S496.txt size=496 overlap_mean=1.300 ms=10145.7
```

```
$ ctest --test-dir build/t41 -R group
1/1 Test #8: test_group .......................   Passed   18.06 sec
100% tests passed, 0 tests failed out of 1
```

Reading of the run:

* **1000 random elements**: all pass the `10^4`-sample inner-product check and are bijections of `C`.
* **200 images of the 496**: every one is an independent set of 496 distinct minimal vectors (norm
  32, Gram `<= 8`, membership in `C`, distinctness — all checked inside the test itself, not by
  `kiss/verify.h`).
* **Overlap histogram** `|gS ∩ S|` over the 200 images: 0 → 110, 2 → 63, 4 → 16, 6 → 9, 8 → 2; mean
  **1.30** against `496²/196560 = 1.25` for a uniformly random image. All overlaps are **even**: `S`
  is antipodal (`−I·S = S`) and every automorphism commutes with `−I`, so `gS ∩ S` is closed under
  negation. The right null model is therefore 248 antipodal pairs hitting 248 of 98280 pairs, which
  is Poisson-ish with mean 0.63 pairs, `P(0) ≈ 0.53`; observed `110/200 = 0.55`.
* The same run on `data/S488.txt` (`--set data/S488.txt --seed 2`): mean 1.18 (expected 1.21),
  histogram 0:112 2:64 4:18 6:6; shape counts (248, 236, 4).

### 2.6 Observations on the 496 (observations, not proofs)

* Shape counts of the 496: 258 of type `(±2^8, 0^16)`, 232 of type `(∓3, ±1^23)`, 6 of type
  `(±4, ±4, 0^22)`. The "24-dimensional cross" is therefore **not** the coordinate frame — the frame
  would contribute 48 vectors of type `(±4, ±4)`, and only three coordinate pairs actually
  contribute. (See `04-structure-of-the-496.md` for what the structure turns out to be.)
* None of α, γ, δ, the octad sign flip or ξ maps `S` to itself: `|gS ∩ S| = 0, 0, 2, 8, 8`. `−I`
  does, `S` being antipodal.
* 2000 random *monomial* elements (product replacement on {α, γ, δ, octad flip} only): **none**
  fixes `S`; overlap mean 1.275, histogram 0:1080 2:630 4:236 6:44 8:9 10:1. The monomial group
  alone already scatters `S` like a random automorphism does. This is **not** an estimate of the
  stabiliser — a stabiliser of order `10^6` would still fix `S` with probability `10^{-6}` per
  sample. The actual orbit/stabiliser computation is in `03-search-for-497.md`.
* Because images of `S` meet `S` in an even number of vectors with `P(0) ≈ 0.55`, and two random
  images meet each other the same way, a pool of a few thousand random images should contain many
  pairwise-disjoint 42-cliques — a prediction that turns out to be **wrong**, and §3.3 says why.

### 2.7 Group-machinery timings

| step | time |
|---|---|
| `index_permutation` of one element (196560 binary searches, OpenMP, 16 threads) | ≈ 25 ms |
| all 64 ξ sign patterns | 1.7 s |
| 1000 product-replacement steps incl. `10^4`-sample checks + bijection checks | 1.6 s |
| 200 images of `S` + Gram checks | < 0.5 s for `|S| ≈ 500` |
| the whole `test_group` | 4.5 s standalone, 23 s under ctest with other builds running |

Build used: `cmake -S . -B build/t41 -G Ninja -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build
build/t41 && ctest --test-dir build/t41 -R group`.

---

## 3. Disjoint families S_1..S_k for dimensions 26–31

Two independent sources of families are documented here: the published record configurations,
decomposed exactly out of their float coordinate files, and families constructed from scratch in
this repository. Both reach the same counts in every dimension 26–31.

### 3.1 The published record families, extracted exactly

The external record data publishes one explicit `S` file (`24D_496_Si.npy`) and seven float
configurations; the `S_i`, `T_i` and extra spheres are implicit. `python/tools/extract_packingstar_families.py`
recovers them exactly:

* rows split into equatorial `(x/√32, 0)`, lifted `(x/(4√3), y/√3)` and extra `(0, y')` with integer
  residual `<= 1e-15`; lifted rows grouped by `y` give the `S_i`, and the `y`'s grouped by `S_i` give
  the `T_i`;
* the float `R^d` configuration is mapped onto the exact integer model of `data/families/SCHEMA.md`
  §2 (A2; A3 = D3; D4; D5; E6; E7 roots) by an isometry determined on a basis (backtracking on the
  Gram matrix) and checked on every row; the extras go through the same isometry and are rationalised
  as `(a + b√M)/D`;
* everything in SCHEMA.md §3 is then re-verified in integer arithmetic: norms, `cos <= 1/2` within
  `T`, within-group `cos <= −1/2`, and for the extras `cos <= 1/2` among themselves and `cos <= √3/2`
  to every `T` vector;
* each `S_i` is converted with `data/external/coordinate_map.json` (a pure coordinate permutation)
  and verified with `python/kiss_ref` — membership in `C`, norm 32, distinctness, Gram `<= 8` — and
  the family's pairwise disjointness is checked.

The result is committed as `data/families/dim26/` … `data/families/dim31/` in the schema of
`data/families/SCHEMA.md`.

| n | d | T config | K(d) | #S_i | T_i partition | count = K(d) + 196560 + Σ(\|T_i\|−1)·496 | C++ `--verify` | `verify_dimN.py` |
|---|---|---|---|---|---|---|---|---|
| 26 | 2 | A2 | 6 | 2 | 2 triangles | 6 + 196560 + 4·496 = **198550** | ok | ok |
| 27 | 3 | A3 | 12 | 5 | 2 triangles + 3 pairs | 12 + 196560 + 7·496 = **200044** | ok | ok (exact + float, 67 s) |
| 28 | 4 | D4 | 24 | 8 | 8 triangles | 24 + 196560 + 16·496 = **204520** | ok | ok (83 s) |
| 29 | 5 | D5 | 40 | 14 | 12 triangles + 2 pairs | 40 + 196560 + 26·496 = **209496** | ok | ok (non-strict; see §4.7 on the two non-antipodal pairs, which SCHEMA §3.3 allows) |
| 30 | 6 | E6 | 72 | 24 | 24 triangles | 72 + 196560 + 48·496 = **220440** | ok | ok (91 s) |
| 31 | 7 | E7 | 126 | 42 | 42 triangles | 126 + 196560 + 84·496 = **238350** | ok | ok |

Note that the `n = 27` row uses partition weight 7 where weight 8 is achievable — that is the
subject of §5.3.

The 96 extracted sets are only **52 distinct** ones, nested exactly as the external data was found to
nest (`25 ⊂ 27 ⊂ 29 ⊂ 30 ⊂ 31`, with 26 and 28 separate). They are committed once and referenced:
`dim31/S_01..42`, `dim26/S_01..02`, `dim28/S_01..08`, with dim27/29/30 referencing `../dim31/`
(SCHEMA §4).

```
$ .venv/bin/python python/tools/extract_packingstar_families.py --dry-run   # 141 s; the committed files were written by the same tool without --dry-run
our C: 196560 vectors; map perm=[0, 1, 2, 3, 4, 5, 15, 20, 6, 19, 16, 12, 8, 18, 22, 23, 11, 9, 13, 10, 14, 7, 21, 17]
dim 31 (d=7): 31D_238350_coordinates.npy rows=238350 sets=42 sizes=[(496, 42)] pairs={} groups={3: 42} T=E7 ambient=8 norm2=8 extra=126 sqrt=2 den=4 count=238350 touching(T-T,E-E,E-T)=2016,2016,0
dim 26 (d=2): 26D_198550_coordinates.npy rows=198550 sets=2 sizes=[(496, 2)] pairs={} groups={3: 2} T=A2 ambient=3 norm2=2 extra=6 sqrt=3 den=3 count=198550 touching(T-T,E-E,E-T)=6,6,12
dim 28 (d=4): 28D_204520_coordinates.npy rows=204520 sets=8 sizes=[(496, 8)] pairs={} groups={3: 8} T=D4 ambient=4 norm2=2 extra=24 sqrt=3 den=3 count=204520 touching(T-T,E-E,E-T)=96,96,48
dim 27 (d=3): 27D_200044_coordinates.npy rows=200044 sets=5 sizes=[(496, 5)] pairs={'antipodal': 3} groups={3: 2, 2: 3} T=A3 ambient=3 norm2=2 extra=12 sqrt=2 den=2 count=200044 touching(T-T,E-E,E-T)=24,24,0
dim 29 (d=5): 29D_209496_coordinates.npy rows=209496 sets=14 sizes=[(496, 14)] pairs={'cos=-1/2': 2} groups={3: 12, 2: 2} T=D5 ambient=5 norm2=2 extra=40 sqrt=2 den=2 count=209496 touching(T-T,E-E,E-T)=240,240,0
dim 30 (d=6): 30D_220440_coordinates.npy rows=220440 sets=24 sizes=[(496, 24)] pairs={} groups={3: 24} T=E6 ambient=8 norm2=8 extra=72 sqrt=3 den=3 count=220440 touching(T-T,E-E,E-T)=720,720,144
dim 25 (d=1): 25D_197056_coordinates.npy rows=197056 sets=1 sizes=[(496, 1)] pairs={'antipodal': 1} groups={2: 1} T=A1 ambient=1 norm2=1 extra=0 sqrt=1 den=1 count=197056 touching(T-T,E-E,E-T)=0,0,0
distinct 496-sets written: 52
RESULT ok=1 dims=31,26,28,27,29,30,25 counts=238350,198550,204520,200044,209496,220440,197056 sets=42,2,8,5,14,24,1 distinct=52 time_s=141.0
```

```
$ for n in 26 27 28 29 30 31; do build/t42/tools/disjoint_family --verify data/families/dim$n | tail -1; done
RESULT ok=1 dir=data/families/dim26 dim=26 sets=2 total=992 count=198550 seconds=0.0
RESULT ok=1 dir=data/families/dim27 dim=27 sets=5 total=2480 count=200044 seconds=0.1
RESULT ok=1 dir=data/families/dim28 dim=28 sets=8 total=3968 count=204520 seconds=0.1
RESULT ok=1 dir=data/families/dim29 dim=29 sets=14 total=6944 count=209496 seconds=0.1
RESULT ok=1 dir=data/families/dim30 dim=30 sets=24 total=11904 count=220440 seconds=0.1
RESULT ok=1 dir=data/families/dim31 dim=31 sets=42 total=20832 count=238350 seconds=0.3
```

### 3.2 What a random image looks like — the number everything follows from

For a random `g ∈ Co_0`, `|g·S ∩ S| ∈ {0, 2, 4, 6, 8}` with

```
P(g·S ∩ S = ∅) = p = 0.544
```

(§2.5 measured 0.55 on 200 elements; the disjointness graphs below have density 0.5438–0.5439 over
`10^7`–`10^8` pairs). The overlap is always even because `S` is antipodal, and the distribution is far
from Poisson — a Poisson with mean 1.25 would give `e^{−1.25} = 0.29`, not 0.54.

The probability that a random image avoids the union of `j` already-chosen disjoint images is well
described by `(1 − 248·j/98280)^248` (`sweep_success_probability`): 0.54, 0.29, 0.15, …, down to
`1.7·10^{-12}` for `j = 41`. The acceptance counts of the sweep below track this within a factor 2
at every step.

### 3.3 Random pool → disjointness graph → clique (tops out at 21)

`disjoint_family --method greedy|ls|bb --pool P` builds `P` random images by product replacement,
forms the graph with an edge iff two images are disjoint (via an inverted vertex → images index; 1 s
for `P = 20000`), then runs greedy growth from random starts, swap-based local search with tabu, or
exact branch and bound.

| P | graph density p | E[#42-cliques in G(P,p)] | pool for E = 1 | greedy | local search | time |
|---|---|---|---|---|---|---|
| 5000 | 0.5439 | 10⁻¹²⁴ | 4.4·10⁶ | 17 | 20 (152 s) | 7 s / 172 s |
| 20000 | 0.5438 | 10⁻⁹⁸ | 4.4·10⁶ | 19 | 21 (303 s) | 45 s / 457 s |

For the small requirements the exact solver is instant: `k = 8` (`n = 28`) and `k = 14` (`n = 29`)
from a pool of 5000 in 9 and 16 branch-and-bound nodes (`runs/families/pool_bb8_5000`,
`runs/families/pool_bb14_5000`, both verified).

The clique number of `G(n, 0.544)` is about `2·log_{1/p} n` = 28 for `n = 5000` and 32.5 for
`n = 20000`; a 42-clique needs `n ≈ 4·10^6` images — 100 GB of 196560-bit masks and `10^13` pair
tests. **The random-pool pipeline cannot reach the record in `n = 31`, and this is a property of the
random images, not of the solver.** This corrects the optimistic expectation recorded in §2.6.

### 3.4 Sweep: a greedy chain over an implicit pool (reaches 39)

`disjoint_family --method sweep` takes `E` random elements `x_e` and `I` random images `h_i(S)`; the
pool `{x_m x_l x_j h_i(S)}` of `E³·I` elements is never materialised. For each `(m, l, j)` the
current union `U` is pulled back to `(x_m x_l x_j)^{-1}U` as a 98280-bit set of antipodal pairs, and
each image's 248 pair ids are scanned with early exit (about 2 lookups per rejected sample; the first
16 ids of every image live in a compact array). A sample rejected against `U` stays rejected when `U`
grows, so a single monotone pass accepts a maximal chain of pairwise disjoint images. On 16 threads
this runs at about `2·10^7` samples/s.

| n | k | E, I | samples | time | result |
|---|---|---|---|---|---|
| 26 | 2 | 64, 4096 | 4 | 4.7 s (setup) | 2 ✓ `runs/families/dim26` |
| 27 | 5 | 64, 4096 | 50 | 28 s | 5 ✓ |
| 28 | 8 | 64, 4096 | 282 | 40 s | 8 ✓ |
| 29 | 14 | 64, 4096 | 1.3·10⁴ | 63 s | 14 ✓ |
| 30 | 24 | 64, 4096 | 4.3·10⁶ | 67 s | 24 ✓ |
| 31 | 42 | 640, 16384 | 7.3·10⁹ | 467 s → 39, then cut off | **39** (`runs/families/dim31`, checkpoint) |

Expected samples to the 42nd set are about `1/p_41 = 5.8·10^11`, roughly 10 hours at this rate. The
sweep is a good random baseline and the wrong tool for `k = 42`.

### 3.5 Chain: the orbit of a single element (reaches 42)

The observation that changes the complexity: `g^a S ∩ g^b S = g^a (S ∩ g^{b−a} S)`, so
`S, gS, …, g^{k−1}S` are pairwise disjoint **iff** `g^i S ∩ S = ∅` for `i = 1..k−1`. That is `k − 1`
conditions instead of `C(k, 2)`. For antipodal `S` the conditions `i` and `m − i` coincide when `g`
has order `m` on antipodal pairs, so an element of order 42 needs only **21** independent-looking
conditions: `P ≈ 0.544^21 ≈ 3·10^{-6}`, times the roughly `1/42` of `Co_1` that has order 42 (class
42A, centraliser order 42), giving about `10^{-7}` per random element.

Define `L(g) = min{i >= 1 : g^i S ∩ S ≠ ∅}`. The search samples elements and computes `L(g)` with
early exit (about 2 steps on average, 496 lookups per step), stopping at `L(g) >= 42`.

Sampling: a pool of `E = 1024` product-replacement elements, candidates `g = x_m x_l x_j`
(`E³ = 10^9` of them; one 786 KB composition per `(m, l)`, then two lookups per vector and step),
running at `6·10^4` candidates/s on 16 shared cores. Drawing a fresh product-replacement element per
candidate manages only `1.2·10^3`/s — the 196560-point composition dominates. Elements with
`L(g) >= 12` also get their order on antipodal pairs (the lcm of the cycle lengths), to see which
classes contribute:

```
$ build/t42/tools/disjoint_family --set data/S496.txt --k 42 --method chain --elements 1024 --seed 1 --out runs/families/dim31_chain
chain: pool of 1024 elements (805.1 MB) in 8.8 s; candidates g = x_m x_l x_j, 1.07e+09 triples
chain: 2590263 candidates on 16 threads in 40.2 s (64378/s); L(g) histogram: 1:1184998 2:645936 3:353834 4:190836 5:103939 6:54717 7:27642 8:13599 9:8041 10:4571 11:3382 12:3169 13:678 14:465 15:1557 16:33 17:17 18:10 19:8 20:101 21:367 22:76 23:242 26:25 28:15 30:10 33:9 35:1 42:1
chain: candidates with L(g) >= 12 by order on antipodal pairs (order: L=count ...):
   12: 12=2374
   13: 13=418
   14: 14=315
   15: 15=1469
   16: 16=1
   20: 20=101
   21: 21=367
   22: 22=76
   23: 23=242
   24: 12=209
   26: 12=27 13=8 26=25
   28: 12=108 13=58 14=22 28=15
   30: 12=139 13=78 14=23 15=29 30=10
   33: 12=96 13=49 14=22 15=9 16=6 33=9
   35: 12=40 13=16 14=13 15=5 16=2 17=1 35=1
   36: 12=26 13=10 14=11 15=3 16=3 17=1 18=2
   39: 12=77 14=33 15=22 16=15 17=12 18=6 19=8
   40: 12=35 13=18 14=14 15=6 16=3
   42: 12=11 13=12 15=11 16=1 17=1 18=2 42=1
   60: 12=27 13=11 14=12 15=3 16=2 17=2
chain: found g with L(g) = 42 (order 42 on antipodal pairs) after 2590263 candidates, 40.2 s
```

The `L` histogram is geometric with ratio 0.54 per step (`L = 1`: 46 %, `L = 2`: 25 %, …) with spikes
at `L = 12, 15, 20, 21, 22, 23` — whole orbits of elements of that order. An element of order 21
whose full orbit is disjoint occurs with probability about `0.54^10` among order-21 elements, in line
with 367 out of 2.6 M. The winner has order 42 on antipodal pairs and `L(g) = 42` exactly
(`g^42 S = S`), i.e. the family is the *full* orbit of `⟨g⟩` — there is no 43rd disjoint image inside
it.

```
RESULT ok=1 k=42 found=42 pool=0 method=chain seed=1 dim=31 total=20832 target_met=1 candidates=2590263 elements=1024 chain_length=42 order=42 timed_out=0 exhausted=0 seconds=40.9
```

A second seed confirms the rate:

```
$ build/t42/tools/disjoint_family --set data/S496.txt --k 42 --method chain --elements 1024 --seed 2 --out runs/families/dim31_chain_seed2
chain: 15158058 candidates on 16 threads in 674.9 s (22459/s); L(g) histogram: 1:6921479 2:3775592 3:2060443 4:1112888 5:608581 6:318883 7:160327 8:77821 9:47194 10:26566 11:19502 12:18541 13:4048 14:2794 15:9329 16:179 17:72 18:35 19:11 20:512 21:2011 22:498 23:1523 26:128 28:123 30:79 33:38 35:12 39:14 42:1
chain: found g with L(g) = 42 (order 42 on antipodal pairs) after 15158058 candidates, 674.9 s
written runs/families/dim31_chain_seed2: 42 sets, total 20832 vectors, verify: ok
RESULT ok=1 k=42 found=42 pool=0 method=chain seed=2 dim=31 total=20832 target_met=1 candidates=15158058 elements=1024 chain_length=42 order=42 timed_out=0 exhausted=0 seconds=677.3
```

Seed 2 needed 15.2 M candidates, 5.9× seed 1 — the number of candidates to a hit is geometric, and
the mean of the two, about 9 M, is `1.1·10^{-7}` per candidate, exactly the estimate above. The
`2.2·10^4` candidates/s here reflects the machine being at load 60–70 during that run. Its winner also
has order 42 and `L(g) = 42`; its `L(g) >= 12` table shows 1 success in 222 order-42 elements with
`L >= 12`, and full-orbit spikes at `L = 12, 15, 20, 21, 22, 23, 26, 28, 30, 33, 35, 39` for all
orders `<= 39` that occur.

The smaller `k` are essentially free:

```
$ for k in 2 5 8 14 24: disjoint_family --set data/S496.txt --k $k --method chain --elements 256 --seed 1 --out runs/families/dim<n>_chain
RESULT ok=1 k=2 found=2 pool=0 method=chain seed=1 dim=26 total=992 target_met=1 candidates=1 elements=256 chain_length=3 order=33 timed_out=0 exhausted=0 seconds=14.2
RESULT ok=1 k=5 found=5 pool=0 method=chain seed=1 dim=27 total=2480 target_met=1 candidates=20 elements=256 chain_length=5 order=18 timed_out=0 exhausted=0 seconds=8.6
RESULT ok=1 k=8 found=8 pool=0 method=chain seed=1 dim=28 total=3968 target_met=1 candidates=19 elements=256 chain_length=10 order=33 timed_out=0 exhausted=0 seconds=17.4
RESULT ok=1 k=14 found=14 pool=0 method=chain seed=1 dim=29 total=6944 target_met=1 candidates=41 elements=256 chain_length=19 order=39 timed_out=0 exhausted=0 seconds=16.8
RESULT ok=1 k=24 found=24 pool=0 method=chain seed=1 dim=30 total=11904 target_met=1 candidates=62985 elements=256 chain_length=30 order=30 timed_out=0 exhausted=0 seconds=9.1
```

Candidates to the first hit: 1, 20, 19, 41, 62985. The wall times are dominated by the 256-element
pool build on the loaded machine plus verification and writing. The `k = 24` winner has order 30 with
`L(g) = 30`; the `k = 14` one has order 39 with `L(g) = 19`.

### 3.6 The committed families

`data/families/ours/dim31` is the seed-1 chain (42 × 496 vectors, 1.2 MB) together with `g.txt`: the
exact 24×24 rational matrix of `g` with denominator 8, recovered from the images of the
`(±4, ±4, 0^22)` vectors via `8·A e_i = A f_ij + A f_ik − A f_jk`. Its induced index permutation
reproduces `g`; since `index_permutation` throws unless the matrix preserves the lattice, the file is
simultaneously the proof that `g ∈ Co_0`. `ours/dim26..30` reference `../dim31/S_01..S_k.txt` with the
record's `T`/extra blocks, written by `python/tools/nest_family.py`. `family.json` is unchanged and
`schema_version` stays 1.

| n | k | record (`data/families/dim<n>`) | ours (`data/families/ours/dim<n>`) | method / time |
|---|---|---|---|---|
| 26 | 2 | 198550 ✓ | **198550** ✓ | chain S_1..2 (also: sweep 5 s; greedy pool 300, 2 s) |
| 27 | 5 | 200044 ✓ | **200044** ✓ | chain S_1..5 (also: sweep 28 s) |
| 28 | 8 | 204520 ✓ | **204520** ✓ | chain S_1..8 (also: sweep 40 s; exact clique in pool 5000, 59 s) |
| 29 | 14 | 209496 ✓ | **209496** ✓ | chain S_1..14 (also: sweep 63 s; exact clique in pool 5000, 54 s) |
| 30 | 24 | 220440 ✓ | **220440** ✓ | chain S_1..24 (also: sweep 67 s) |
| 31 | 42 | 238350 ✓ | **238350** ✓ | chain, 41 s (seed 2: 677 s) (sweep: 39 in 467 s; pools ≤ 20000: ≤ 21) |

```
$ for n in 26 27 28 29 30 31; do build/t42/tools/disjoint_family --verify data/families/ours/dim$n | tail -1; done
RESULT ok=1 dir=data/families/ours/dim26 dim=26 sets=2 total=992 count=198550 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim27 dim=27 sets=5 total=2480 count=200044 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim28 dim=28 sets=8 total=3968 count=204520 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim29 dim=29 sets=14 total=6944 count=209496 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim30 dim=30 sets=24 total=11904 count=220440 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim31 dim=31 sets=42 total=20832 count=238350 seconds=0.3
```

And the full independent verifier on the chain family (identical sets to `data/families/ours/dim31`):

```
$ .venv/bin/python python/verify_dimN.py runs/families/dim31_chain
family      : runs/families/dim31_chain  dim=31 d=7 sets=42 T=126 groups=42 extra=126 claimed_count=238350
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=42 sizes=[496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496] union=20832 pairwise disjoint, each independent; equatorial = C \ U = 175728  (2.1 s)
T           : 126 vectors in Z^8-ambient field, norms ['8'], pairwise cos <= 1/2 (max 0.500000); groups: 42 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (2.9 s)
extra       : 126 vectors (K(7) = 126), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.853553)  (20.8 s)
rank        : span(T u extra) has exact rank 7 <= d = 7
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 126 + 196560 + 41664 = 238350  (rows: equatorial 175728 + lifted 62496 + extra 126)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (61.3 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=15440077128 eq-lift=10982297088 lift-lift-same-x=62496 lift-lift-same-set=46403280 lift-lift-cross-set=1906377984 extra-eq=22141728 extra-lift=7874496 extra-extra=7875 total=28405242075=C(238350,2)
exact       : count = 238350  (89.2 s)
float       : 238350 vectors in R^31; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (175790, 210593) class=lift-lift-cross-set; min -4.000000  (214.9 s, 54522 tiles)
RESULT ok=1 dim=31 d=7 sets=42 count=238350 count_exact=238350 count_float=238350 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=126 K=126 exact_s=89.2 float_s=214.9 s=331.0
```

### 3.7 Family-search tests

`tests/test_family.cpp` (ctest target `test_family`) checks: masks against brute force; that the two
disjointness-graph builders agree on a pool of 400 images; that greedy, local search and exact
branch-and-bound all recover a planted 14-clique in `G(600, 0.3)` and that exact search proves
non-existence on a small instance; a family-directory round trip with `T`/extra copied from
`data/families/dim27`; rejection of corrupted variants (overlapping sets, a non-independent set, a
size mismatch); that `data/families/dim26` verifies with total 992; that a short sweep finds 6
disjoint images; and that the chain search finds 6 with `pair_order`, `chain_length` and
`aut_from_index_perm` cross-checked (`S_{i+1} = g S_i`, and the matrix reproduces `g`).

```
$ ctest --test-dir build/t42 -R family
Test project <repo>/build/t42
    Start 17: test_family
1/1 Test #17: test_family ......................   Passed   16.79 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =  16.80 sec
```

### 3.8 Why there is nothing beyond 42 in dimension 31

With `|S_i| = 496` for all `i` the count formula gives exactly the record values. Exceeding them
requires either `|S| > 496` — no such set is known; see `03-search-for-497.md`, and
`02-upper-bounds.md` for the upper bounds (the two-point LP gives 850, the three-point Terwilliger
SDP 837) — or a better `R^d` template (§5), not more disjoint copies. `E7` has 42 triangles, so
`k = 42` is the template's maximum for `n = 31`.

This also explains the *structure* of the published record data: those families nest across
dimensions (`25 ⊂ 27 ⊂ 29 ⊂ 30 ⊂ 31`) like a growing chain, and 42, the number of triangles of E7,
is exactly the maximum the template can use. "42 disjoint 496s" is the natural endpoint for anyone
with an automorphism source. Whether the published 42 are a single orbit was **not** checked — that
would mean running the stabiliser machinery of `03-search-for-497.md` 41 times to find the element
mapping one 496 to another, and it is not needed for anything here.

---

## 4. The dimension-N verifiers

`python/verify_dimN.py` is an independent checker for the whole `R^(24+d)` template on a family
directory in the schema of `data/families/SCHEMA.md`. It regenerates `C` from scratch, checks every
condition exactly, then assembles the explicit configuration and runs a floating-point pass over it.
It passes on all twelve families under `data/families/` — the six converted record families and the
six chain families — with exact and float counts agreeing in every dimension.

```
verify_dimN.py <family_dir> [--strict] [--skip-full-equatorial] [--skip-float] \
               [--tile R C] [--workers W]
```

It prints `RESULT ok=1 dim=… count=… count_exact=… count_float=… max_offdiag=…` and exits 0/1.

Supporting modules:

| File | Purpose |
|---|---|
| `python/verify_dimN.py` | the verifier itself (schema v1, tolerant superset reader) |
| `python/kiss_ref/small_kissing.py` | exact integer models of the `K(d)` kissing configurations (A2, A3 = D3, D4, D5, E6, E7), integer angle predicates, all triangles, exact DFS triangle partition / maximum partition, extra-sphere search (three strategies, all results exact), schema `T`/`extra` block writers, `check_extra_ab`, isometry finder onto the model |
| `python/kiss_ref/exact.py` | exact arithmetic in `Q(√p_1, √p_2, …)`: class `Q` (dict of square-free radical → Fraction; exact sign, inverse via Galois conjugates), entry parser (ints, Fractions, dyadic floats, strings like `"(2+sqrt(2))/4"`, `[a,b]` pairs with a block radical, `{"1":a,"3":b}` dicts), `dot`/`norm2`/`rank`, and the predicates `cos_le_half`, `cos_le_neg_half`, `cos_le_sqrt3_half`, `is_antipodal`, `is_same_direction` |
| `python/tests/test_small_kissing.py` | 46 tests |
| `python/tests/test_verify_dimN.py` | 24 tests |

### 4.1 The R^d models

| d | K(d) | model | ambient m | norm² | triangles (cos = −1/2 triples) | touching pairs (60°) |
|---|---|---|---|---|---|---|
| 2 | 6 | A2: permutations of (1,−1,0), sum 0 | 3 | 2 | 2 | 6 |
| 3 | 12 | D3 roots (±1,±1,0) | 3 | 2 | 8 | 24 |
| 4 | 24 | D4 roots | 4 | 2 | 32 | 96 |
| 5 | 40 | D5 roots | 5 | 2 | 80 | 240 |
| 6 | 72 | E8 roots ⊥ (2,−2,0⁶), (0,2,−2,0⁵) | 8 | 8 | 240 | 720 |
| 7 | 126 | E8 roots ⊥ (2,−2,0⁶) | 8 | 8 | 672 | 2016 |

(E8 roots as `(±2,±2,0^6)` and `(±1)^8` with an even number of minus signs, norm 8.) Every
configuration is antipodal, and each vector has `2·touching/K` neighbours at 60°: 2, 4, 8, 12, 20, 32.
The ambient dimension exceeding `d` is harmless for every angle condition — they involve only inner
products — and is undone for the float pass by `orthonormal_embedding` (thin SVD of the rows, Gram
equality verified to 1e-9), after the exact rank of `span(T ∪ extra)` has been checked to be `<= d`
in the field.

Angle predicates on integer (or field) vectors `a, b` with `p = ⟨a,b⟩`, `Na = |a|²`, `Nb = |b|²`:

```
cos <= 1/2         <=>  p <= 0  or  4 p^2 <= Na Nb
cos <= -1/2        <=>  p <  0  and 4 p^2 >= Na Nb
cos <= sqrt(3)/2   <=>  p <= 0  or  4 p^2 <= 3 Na Nb
cos == -1          <=>  p <  0  and   p^2 == Na Nb
cos == +1          <=>  p >  0  and   p^2 == Na Nb
```

`_ints` promotes to Python integers when `4p²` or `3·Na·Nb` could overflow int64. For the schema's
`Q(√M)` extras `y' = (a + b√M)/D` the same predicates become sign tests of `A + B√M` (SCHEMA.md
§3.4; `cmp_sqrt` / `_sign_ab` for the vectorised path, `Q.sign` in `exact.py` for the general one).

### 4.2 What the verifier checks, in order (first failure is reported)

**A. Sets.** Each `S_i` file parses; every row has norm 32 and lies in `C` (dictionary lookup in the
regenerated `C` *and* the arithmetic membership test, via `verify_S.check_set`); rows are distinct;
the Gram off-diagonal is `<= 8`; `size` matches the json, and `sha256` matches (a warning normally, a
failure under `--strict`). Pairwise disjointness is checked via the canonical indices, and the message
names the first shared vector and the number shared. The equatorial set is `C \ ∪S_i`, and
`n_eq + |∪S_i| = 196560` is asserted.

**B. T.** Non-zero rows, pairwise distinct directions, pairwise `cos <= 1/2`; `len(groups) == k`;
every group of size 2 or 3 with indices in range and no `T` vector in two groups; pairwise
`cos <= −1/2` inside each group, so a size-3 group is an equilateral triangle and a size-2 group is
either an antipodal pair or two vertices of a triangle. The latter is what the record 29D family uses
(SCHEMA.md §3.3); it is reported as a note, never a failure. Unused `T` vectors are allowed with a
warning.

**C. Extras.** Non-zero, pairwise distinct directions, pairwise `cos <= 1/2`, and `cos <= √3/2` to
*every* `T` vector, grouped or not. Under `--strict`, count and norms must be as claimed.

**D. Rank.** The exact rank of `span(T ∪ extra)` over the field is `<= d`.

**E. Count.** `count = #extra + 196560 + Σ_i (|T_i|−1)|S_i|`, compared against `count` and
`count_terms` in the json.

**F. Exact casework** on the assembled configuration (below); the pair counts of the eight classes
are summed and must equal `C(count, 2)`.

**G. Float.** Explicit float64 rows in `R^(24+d)`, norms `4 ± 1e-9`, a tiled upper-triangular GEMM
(`verify_dim25.pairwise_offdiag_max`, 512 × 1024 tiles, one single-threaded BLAS per Python thread),
max off-diagonal `<= 2 + 1e-9`, the class of the attaining pair, and row count equal to `count`.

### 4.3 The exact inequalities

Norm-4 scaling; `ip = ⟨x,x'⟩` is the `√8`-integer Leech inner product, in `{−32,…,32}`. Rows are
equatorial `(x/√8, 0)`, lifted `(√(2/3)·x/√8, √(4/3)·u(y))` for `x ∈ S_i, y ∈ T_i`, and extra
`(0, 2u(y'))`. With `p = ⟨y,y'⟩`, `Na = |y|²`, `Nb = |y'|²` in the `R^d` model (so
`cos = p/√(Na·Nb)`):

| pair class | inner product | condition | reduces to | how it is checked |
|---|---|---|---|---|
| eq–eq | ip/8 | ≤ 2 | ip ≤ 16 | full float32-GEMM pass over all C(196560,2) pairs (exact: integer inputs ≤ 4, partial sums ≤ 384), max = 16, values ⊂ {−32,−16,−8,0,8,16}; or the minimal-norm argument (\|x−x'\|² = 64 − 2·ip ≥ 32) under `--skip-full-equatorial` |
| eq–lift | √(2/3)·ip/8 | ≤ 2 ⟺ ip ≤ 8√6 ≈ 19.6 | ip ≤ 16 | max of (∪S_i) × (C \ ∪S_i) by exact float32 GEMM (`cross_max`); 32 would mean a lifted x is also equatorial |
| lift–lift, same x, y ≠ y' (same T_i) | (2/3)·4 + (4/3)·cos | ≤ 2 | cos ≤ −1/2 ⟺ p < 0 ∧ 4p² ≥ Na·Nb | check B, exactly in the field |
| lift–lift, same S_i, x ≠ x', y = y' | ip/12 + 4/3 | ≤ 2 | ip ≤ 8 | S_i independent (check A) and the max over same-set pairs of the union Gram (`union_gram_maxima`) |
| lift–lift, same S_i, x ≠ x', y ≠ y' | ip/12 + (4/3)·cos, cos ≤ −1/2 | ≤ 2 ⇐ ip ≤ 32 | always | — |
| lift–lift, S_i ≠ S_j (x ≠ x' by disjointness) | ip/12 + (4/3)·cos, cos ≤ 1/2 | ≤ 2 ⇐ ip ≤ 16 | ip ≤ 16 | max over cross-set pairs of the union Gram = 16; **tight**: ip = 16 with cos = 1/2 gives exactly 2 |
| extra–eq | 0 | ≤ 2 | always | — |
| extra–lift | 2·√(4/3)·cos = (4/√3)·cos | ≤ 2 | cos ≤ √3/2 ⟺ p ≤ 0 ∨ 4p² ≤ 3·Na·Nb | check C, exactly in the field |
| extra–extra | 4·cos | ≤ 2 | cos ≤ 1/2 ⟺ p ≤ 0 ∨ 4p² ≤ Na·Nb | check C, exactly in the field |

There are exactly two tight cases. The float maximum `2.000000000000` is attained on
`lift-lift-same-x` pairs (`8/3 − 2/3 = 2`) for `d <= 5`, and on `lift-lift-cross-set` pairs (`ip = 16`
with `cos(y,y') = 1/2`: `16/12 + 2/3 = 2`) for `d = 6, 7`. Which one `argmax` reports first depends
only on tile order. The minimum off-diagonal is `−4` everywhere (antipodal equatorial pairs).

### 4.4 Verifier tests

`test_small_kissing.py` (46 tests): counts, norms, distinctness, antipodality and touching pairs of
every configuration; `config()` returns a copy; `d = 8` rejected; float embeddings reproduce the
Gram; embedding into too small a `d` raises; predicate ties (`3/√12 = √3/2` exactly; `cmp_sqrt`);
predicates against float on 300 × 300 random integer pairs; the int64-overflow guard; triangle counts
and their sum-zero property; that the partitions (2, 4, 8, 12 + 2 pairs, 24, 42) are exact covers;
that `d = 5` with 13 triangles is infeasible and exhaustive; `max_triangle_partition`; that lattice
extras reach `K` for `d = 2, 4` (max cos to `T` exactly `√3/2` and `1/√2`); that the `d = 3` lattice
search is partial (8–12); that `check_extra` rejects a `T` vector and a 45° pair; that exact rotation
extras reach `K` for `d = 3, 5` over `Q(√2)` and agree with the general field arithmetic; `_sign_ab`;
that schema blocks round-trip through `exact.Q`; the isometry finder for `d = 2, 3, 4`; field
arithmetic (`√8 = 2√2`, inverses, the sign of `√3 − √2`); parser forms; and exact rank against numpy.

`test_verify_dimN.py` (24 tests): the fixture is `data/S496.txt` together with its image under the
first M24 generator whose image is disjoint from it (α: `x → x+1`, for which §2.5 found
`|gS ∩ S| = 0`), written as a schema-v1 dim26 directory with `schema_T_block(2)` and the six
`(2,−1,−1)`-type extras (`schema_extra_block`: sqrt 3, den 3). For the good family: in-process exact
checks (198550, max ips 16/8/16, pair sums, no warnings), the CLI with full pass and float
(`count_float=198550`, max 2 on `lift-lift-same-x`), and the assembled coordinates (shape, norms,
pair classes, extras orthogonal to the Leech part). Reader tolerance: plain file names, `T` as groups
of string vectors, extras as `[0,b]` pairs with `sqrt`/`scale`, per-set groups, and missing extras
(count 198544 plus a warning). Rejections: `S_2 := S_1` (496 shared), one shared vector, a dependent
set (ip 16), a 60° pair in a group, a `T` vector used twice, an extra at 19.1°, two extras at < 60°,
a count claim of 198551, a missing group, rank 3 > `d` = 2, bad json / missing file, and a bad sha256
(warning normally, failure under `--strict`). Finally, the exact part of every present
`data/families/dim*` under `--strict`: the expected counts, max ips 16/8/16, and number of extras
equal to `K(d)`.

```
$ .venv/bin/python -m pytest python/tests/test_small_kissing.py python/tests/test_verify_dimN.py -q --durations=6
......................................................................   [100%]
============================= slowest 6 durations ==============================
135.39s call     python/tests/test_verify_dimN.py::test_cli_full_pass_and_float
42.80s call     python/tests/test_verify_dimN.py::test_packingstar_family_exact[31]
16.94s call     python/tests/test_verify_dimN.py::test_packingstar_family_exact[30]
9.34s call     python/tests/test_verify_dimN.py::test_packingstar_family_exact[29]
5.25s call     python/tests/test_verify_dimN.py::test_packingstar_family_exact[28]
3.28s call     python/tests/test_verify_dimN.py::test_packingstar_family_exact[27]
70 passed in 228.23s (0:03:48)

real	3m49.023s
user	11m29.857s
sys	3m22.472s
```

Corrupted inputs — overlapping `S_i`, a 60° pair in a `T` group, a `T` vector used twice, an extra
sphere at 19°, extras at < 60° from each other, a wrong count, a missing group, rank > `d`, a bad
sha256 in strict mode — are all rejected with a message naming the first offender.

### 4.5 Acceptance runs on the record families

All six with `--strict`, full eq–eq pass and float pass.

`.venv/bin/python python/verify_dimN.py data/families/dim26 --strict`

```
family      : data/families/dim26  dim=26 d=2 sets=2 T=6 groups=2 extra=6 claimed_count=198550
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=2 sizes=[496, 496] union=992 pairwise disjoint, each independent; equatorial = C \ U = 195568  (0.0 s)
T           : 6 vectors in Z^3-ambient field, norms ['2'], pairwise cos <= 1/2 (max 0.500000); groups: 2 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (0.0 s)
extra       : 6 vectors (K(2) = 6), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.866025)  (0.0 s)
rank        : span(T u extra) has exact rank 2 <= d = 2
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 6 + 196560 + 1984 = 198550  (rows: equatorial 195568 + lifted 2976 + extra 6)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (5.3 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=19123323528 eq-lift=582010368 lift-lift-same-x=2976 lift-lift-same-set=2209680 lift-lift-cross-set=2214144 extra-eq=1173408 extra-lift=17856 extra-extra=15 total=19710951975=C(198550,2)
exact       : count = 198550  (5.7 s)
float       : 198550 vectors in R^26; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (196112, 196608) class=lift-lift-same-x; min -4.000000  (21.2 s, 37830 tiles)
RESULT ok=1 dim=26 d=2 sets=2 count=198550 count_exact=198550 count_float=198550 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=6 K=6 exact_s=5.7 float_s=21.2 s=27.0
```

`.venv/bin/python python/verify_dimN.py data/families/dim27 --strict`

```
family      : data/families/dim27  dim=27 d=3 sets=5 T=12 groups=5 extra=12 claimed_count=200044
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=5 sizes=[496, 496, 496, 496, 496] union=2480 pairwise disjoint, each independent; equatorial = C \ U = 194080  (0.1 s)
T           : 12 vectors in Z^3-ambient field, norms ['2'], pairwise cos <= 1/2 (max 0.500000); groups: 2 triangles + 3 pairs, disjoint, within-group cos <= -1/2  (0.0 s)
extra       : 12 vectors (K(3) = 12), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.853553)  (0.0 s)
rank        : span(T u extra) has exact rank 3 <= d = 3
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 12 + 196560 + 3472 = 200044  (rows: equatorial 194080 + lifted 5952 + extra 12)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (20.6 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=18833426160 eq-lift=1155164160 lift-lift-same-x=4464 lift-lift-same-set=3682800 lift-lift-cross-set=14022912 extra-eq=2328960 extra-lift=71424 extra-extra=66 total=20008700946=C(200044,2)
exact       : count = 200044  (21.5 s)
float       : 200044 vectors in R^27; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (194127, 195119) class=lift-lift-same-x; min -4.000000  (45.4 s, 38416 tiles)
RESULT ok=1 dim=27 d=3 sets=5 count=200044 count_exact=200044 count_float=200044 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=21.5 float_s=45.4 s=67.2
```

`.venv/bin/python python/verify_dimN.py data/families/dim28 --strict`

```
family      : data/families/dim28  dim=28 d=4 sets=8 T=24 groups=8 extra=24 claimed_count=204520
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=8 sizes=[496, 496, 496, 496, 496, 496, 496, 496] union=3968 pairwise disjoint, each independent; equatorial = C \ U = 192592  (0.2 s)
T           : 24 vectors in Z^4-ambient field, norms ['2'], pairwise cos <= 1/2 (max 0.500000); groups: 8 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (0.0 s)
extra       : 24 vectors (K(4) = 24), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.866025)  (0.0 s)
rank        : span(T u extra) has exact rank 4 <= d = 4
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 24 + 196560 + 7936 = 204520  (rows: equatorial 192592 + lifted 11904 + extra 24)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (26.5 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=18545742936 eq-lift=2292615168 lift-lift-same-x=11904 lift-lift-same-set=8838720 lift-lift-cross-set=61996032 extra-eq=4622208 extra-lift=285696 extra-extra=276 total=20914112940=C(204520,2)
exact       : count = 204520  (27.9 s)
float       : 204520 vectors in R^28; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (192645, 193141) class=lift-lift-same-x; min -4.000000  (55.0 s, 40200 tiles)
RESULT ok=1 dim=28 d=4 sets=8 count=204520 count_exact=204520 count_float=204520 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=24 K=24 exact_s=27.9 float_s=55.0 s=83.3
```

`.venv/bin/python python/verify_dimN.py data/families/dim29 --strict`

```
family      : data/families/dim29  dim=29 d=5 sets=14 T=40 groups=14 extra=40 claimed_count=209496
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=14 sizes=[496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496] union=6944 pairwise disjoint, each independent; equatorial = C \ U = 189616  (0.2 s)
T           : 40 vectors in Z^5-ambient field, norms ['2'], pairwise cos <= 1/2 (max 0.500000); groups: 12 triangles + 2 pairs, disjoint, within-group cos <= -1/2  (0.0 s)
extra       : 40 vectors (K(5) = 40), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.853553)  (0.1 s)
rank        : span(T u extra) has exact rank 5 <= d = 5
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 40 + 196560 + 12896 = 209496  (rows: equatorial 189616 + lifted 19840 + extra 40)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (100.7 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=17977018920 eq-lift=3761981440 lift-lift-same-x=18848 lift-lift-same-set=14240160 lift-lift-cross-set=182543872 extra-eq=7584640 extra-lift=793600 extra-extra=780 total=21944182260=C(209496,2)
exact       : count = 209496  (103.6 s)
float       : 209496 vectors in R^29; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (192000, 192496) class=lift-lift-same-x; min -4.000000  (93.3 s, 42230 tiles)
warning     : group 13: pair [27, 32] is not antipodal (cos = -0.500000, 120.00 deg; two vertices of a triangle is allowed by the template and SCHEMA.md 3.3)
warning     : group 14: pair [21, 3] is not antipodal (cos = -0.500000, 120.00 deg; two vertices of a triangle is allowed by the template and SCHEMA.md 3.3)
RESULT ok=1 dim=29 d=5 sets=14 count=209496 count_exact=209496 count_float=209496 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=40 K=40 exact_s=103.6 float_s=93.3 s=197.5
```

`.venv/bin/python python/verify_dimN.py data/families/dim30 --strict`

```
family      : data/families/dim30  dim=30 d=6 sets=24 T=72 groups=24 extra=72 claimed_count=220440
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=24 sizes=[496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496] union=11904 pairwise disjoint, each independent; equatorial = C \ U = 184656  (0.4 s)
T           : 72 vectors in Z^8-ambient field, norms ['8'], pairwise cos <= 1/2 (max 0.500000); groups: 24 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (0.2 s)
extra       : 72 vectors (K(6) = 72), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.866025)  (0.8 s)
rank        : span(T u extra) has exact rank 6 <= d = 6
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 72 + 196560 + 23808 = 220440  (rows: equatorial 184656 + lifted 35712 + extra 72)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (20.5 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=17048826840 eq-lift=6594435072 lift-lift-same-x=35712 lift-lift-same-set=26516160 lift-lift-cross-set=611103744 extra-eq=13295232 extra-lift=2571264 extra-extra=2556 total=24296786580=C(220440,2)
exact       : count = 220440  (26.1 s)
float       : 220440 vectors in R^30; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (190464, 219426) class=lift-lift-cross-set; min -4.000000  (62.8 s, 46656 tiles)
RESULT ok=1 dim=30 d=6 sets=24 count=220440 count_exact=220440 count_float=220440 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=72 K=72 exact_s=26.1 float_s=62.8 s=90.6
```

`.venv/bin/python python/verify_dimN.py data/families/dim31 --strict`

```
family      : data/families/dim31  dim=31 d=7 sets=42 T=126 groups=42 extra=126 claimed_count=238350
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=42 sizes=[496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496, 496] union=20832 pairwise disjoint, each independent; equatorial = C \ U = 175728  (0.7 s)
T           : 126 vectors in Z^8-ambient field, norms ['8'], pairwise cos <= 1/2 (max 0.500000); groups: 42 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (0.8 s)
extra       : 126 vectors (K(7) = 126), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.853553)  (4.7 s)
rank        : span(T u extra) has exact rank 7 <= d = 7
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 126 + 196560 + 41664 = 238350  (rows: equatorial 175728 + lifted 62496 + extra 126)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (26.0 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=15440077128 eq-lift=10982297088 lift-lift-same-x=62496 lift-lift-same-set=46403280 lift-lift-cross-set=1906377984 extra-eq=22141728 extra-lift=7874496 extra-extra=7875 total=28405242075=C(238350,2)
exact       : count = 238350  (35.7 s)
float       : 238350 vectors in R^31; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (176170, 210886) class=lift-lift-cross-set; min -4.000000  (68.7 s, 54522 tiles)
RESULT ok=1 dim=31 d=7 sets=42 count=238350 count_exact=238350 count_float=238350 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=126 K=126 exact_s=35.7 float_s=68.7 s=110.9
```

Summary — published record total against verified total:

| n | d | config | sets | groups | extras | count = K(d) + 196560 + Σ(\|T_i\|−1)\|S_i\| | published | exact | float | max off-diag (class) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 26 | 2 | A2 | 2 | 2 triangles | 6 | 6 + 196560 + 4·496 = **198550** | 198550 | 198550 | 198550 | 2.000000000000 (lift-lift-same-x) | 27 s |
| 27 | 3 | A3 | 5 | 2 triangles + 3 pairs | 12 | 12 + 196560 + 7·496 = **200044** | 200044 | 200044 | 200044 | 2.000000000000 (lift-lift-same-x) | 67 s |
| 28 | 4 | D4 | 8 | 8 triangles | 24 | 24 + 196560 + 16·496 = **204520** | 204520 | 204520 | 204520 | 2.000000000000 (lift-lift-same-x) | 83 s |
| 29 | 5 | D5 | 14 | 12 triangles + 2 pairs (cos −1/2) | 40 | 40 + 196560 + 26·496 = **209496** | 209496 | 209496 | 209496 | 2.000000000000 (lift-lift-same-x) | 198 s* |
| 30 | 6 | E6 | 24 | 24 triangles | 72 | 72 + 196560 + 48·496 = **220440** | 220440 | 220440 | 220440 | 2.000000000000 (lift-lift-cross-set) | 91 s |
| 31 | 7 | E7 | 42 | 42 triangles | 126 | 126 + 196560 + 84·496 = **238350** | 238350 | 238350 | 238350 | 2.000000000000 (lift-lift-cross-set) | 111 s |

All 52 distinct 496-sets of the six families (42 + 2 + 8, with dim27/29/30 referencing
`../dim31/S_xx.txt`) are independent, of size 496, and have the claimed sha256; every family's sets
are pairwise disjoint; every `T` block is exactly the integer model of `small_kissing.config(d)`; and
every `extra` block has exactly `K(d)` rows in `Q(√2)` (`d` odd) or `Q(√3)` (`d` even).

### 4.6 Acceptance runs on the chain families

The chain families `data/families/ours/dim26..31` — `S_i = g^{i−1}·S496`, `i = 1..42`, for the
order-42 element `g` of §3.5, nested into dims 26–30 by `python/tools/nest_family.py` with the same
`T`/extra blocks — all six verify with the same counts (`--strict`, full eq–eq pass and float pass).
The wall times below are inflated by other jobs on the same machine.

```
=== data/families/ours/dim26
RESULT ok=1 dim=26 d=2 sets=2 count=198550 count_exact=198550 count_float=198550 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=6 K=6 exact_s=62.6 float_s=102.1 s=165.4
=== data/families/ours/dim27
RESULT ok=1 dim=27 d=3 sets=5 count=200044 count_exact=200044 count_float=200044 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=47.0 float_s=108.2 s=156.1
=== data/families/ours/dim28
RESULT ok=1 dim=28 d=4 sets=8 count=204520 count_exact=204520 count_float=204520 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=24 K=24 exact_s=49.6 float_s=50.4 s=100.8
=== data/families/ours/dim29
RESULT ok=1 dim=29 d=5 sets=14 count=209496 count_exact=209496 count_float=209496 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=40 K=40 exact_s=39.5 float_s=62.1 s=102.4
=== data/families/ours/dim30
RESULT ok=1 dim=30 d=6 sets=24 count=220440 count_exact=220440 count_float=220440 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=72 K=72 exact_s=32.0 float_s=60.8 s=96.4
=== data/families/ours/dim31
RESULT ok=1 dim=31 d=7 sets=42 count=238350 count_exact=238350 count_float=238350 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=126 K=126 exact_s=41.3 float_s=75.8 s=129.8
```

| n | ours count | published | verified |
|---|---|---|---|
| 26 | 198550 | 198550 | exact = float |
| 27 | 200044 | 200044 | exact = float |
| 28 | 204520 | 204520 | exact = float |
| 29 | 209496 | 209496 | exact = float (groups 13, 14 are cos −1/2 pairs, noted) |
| 30 | 220440 | 220440 | exact = float |
| 31 | 238350 | 238350 | exact = float |

So there are **two independent sources** of 42 pairwise disjoint 496-sets — the published family and a
single `Co_0` element of order 42 acting on one 496 — and both reach 238350. With `|S_i| = 496` the
template cannot give more in that dimension.

### 4.7 Verifier timing, and notes

Hardware: Ryzen 9 5900HS (8C/16T, 16 GB), 8 worker threads, single-threaded OpenBLAS per thread,
512 × 1024 tiles. The machine was shared with other jobs throughout (including an `orbit_milp.py` at
> 100 % CPU and several builds), so wall times are upper bounds; the dim26 run at 27 s was the only
one on an otherwise idle box.

| n | count | exact part (incl. full eq–eq pass) | float pass (tiles) | wall |
|---|---|---|---|---|
| 26 | 198550 | 5.7 s (pass 5.3 s) | 21.2 s (37830) | 27.0 s |
| 27 | 200044 | 21.5 s (pass 20.6 s) | 45.4 s (38416) | 67.2 s |
| 28 | 204520 | 27.9 s (pass 26.5 s) | 55.0 s (40200) | 83.3 s |
| 29 | 209496 | 103.6 s (pass 100.7 s)* | 93.3 s (42230)* | 197.5 s* |
| 30 | 220440 | 26.1 s (pass 20.5 s) | 62.8 s (46656) | 90.6 s |
| 31 | 238350 | 35.7 s (pass 26.0 s) | 68.7 s (54522) | 110.9 s |

\* dim29 was re-run after the strict-mode fix (§7) while another MILP job was saturating the cores;
its eq–eq pass is the same work as the others' (5–26 s).

The exact part outside the full pass is 0.4–10 s, dominated for dim31 by the 126 × 126 + 126 × 126
field-arithmetic extra checks (4.7 s) and the 20832 × 175728 eq–lift GEMM. The float pass scales as
`count²` (dim31: `2.84·10^10` pairs, 54522 tiles, about `4·10^8` pairs/s).
`--skip-full-equatorial --skip-float` — the exact casework relying on the Leech minimal norm for
eq–eq, which is what the pytest suite does — takes 2–43 s for dim27–31. The pytest wall is 228 s for
the 70 tests, 135 s of which is the CLI run of the synthetic dim26 family with full pass and float,
under contention.

Further notes:

* The eq–eq full pass is the same computation in every dimension (all of `C`, `1.93·10^10` pairs) and
  is what `verify_dim25.py` does. It is kept because the verifier is meant to be independent of the
  Leech minimal-norm argument; `--skip-full-equatorial` substitutes that argument and says so.
* `max_cos_ET` is exactly `√3/2` for `d = 2, 4, 6` (the `Q(√3)` rotations touch the 30° bound) and
  `(2+√2)/4 = 0.85355` for `d = 3, 5, 7` (`Q(√2)` rotations, 31.4°). The published extras therefore
  leave no angular slack for even `d` and 1.4° for odd `d`. The extra spheres are the only part of
  the template where more than `K(d)` points could conceivably fit — see §5.5.
* A size-2 group that is two vertices of a triangle rather than an antipodal pair is what the record
  29D family uses in its groups 13 and 14 (pairs `[27, 32]` and `[21, 3]`, `cos = −1/2`). The
  template needs only `cos <= −1/2`, and SCHEMA.md §3 item 3 explicitly allows it; the verifier
  reports it as a note in both modes. See §7 for the earlier `--strict` behaviour, which was wrong.
* Verification scope: the two verifier test files were re-run, not the whole suite, at the time these
  numbers were taken. The new modules import nothing new and touch no other module. The
  `data/families/` tree was re-checked at the end (mtimes unchanged), so the files verified are the
  files on disk.

---

## 5. Optimality of the R^d half

With `|S_i| = 496` fixed, the only levers inside the template are the **partition weight**
`Σ_i (|T_i| − 1)` and the **number of extra spheres** `|E|`. Both were pushed to their exact optima
for `d = 2..7`. One of them was not already at its optimum in the published data, and that is the
dimension-27 improvement.

Tooling:

| File | Content |
|---|---|
| `python/template/triangles.py` | all triangles / admissible pairs; exact ILPs (HiGHS via `scipy.optimize.milp`) for the maximum disjoint-triangle packing and the maximum-weight group partition, with LP bound, MIP dual bound, gap and node count; `weight_upper_bound`, `zero_sum_bound` (solver-free certificates); record comparison (`record_partition`, `analyse`); `continuous_triangle_search` (penalty descent + SLSQP for `n` points in `R^d` with `k` prescribed triangles) |
| `python/template/extra_spheres.py` | the `K(d)` upper-bound statement; candidate pools (integer grid, record extras in `Q(√M)`, plane rotations of `T`); exact-tolerance conflict graph + ILP maximum independent set with clique cuts; exact field re-check (`check_exact`); continuous `K(d)+1` penalty search |
| `python/template/assign.py` | `family_count`, `assign_sets_to_groups` (largest `S_i` to the heaviest groups, by the rearrangement inequality), `optimal_groups(d)`, `rewrite_family` (builds a family directory with the optimal partition; has a CLI) |
| `python/tests/test_template.py` | 47 tests |
| `runs/template/dim27_record/`, `runs/template/dim27_ours/` | the improved 27-dimensional families (`family.json` + `S_01..S_04.txt`) and their `verify_dim27_*.log` |
| `runs/template/{triangles,extra_spheres,continuous_triangles,pytest_all}.log` | the outputs quoted below |

Only `scipy.optimize.milp` (HiGHS) was needed; `highspy` was not installed.

### 5.1 Triangle partitions: exact optima and a solver-free certificate

Two integer programs per configuration, both proven optimal by HiGHS with gap 0, and both matching a
hand certificate:

* **Packing.** A binary `x_t` per triangle, `Σ_{t ∋ p} x_t <= 1`, maximise `Σ x_t`. Certificate:
  `⌊K/3⌋`. For D5 (`K = 40 ≡ 1 mod 3`) there is a sharper leftover-point argument: all six models are
  antipodal so `Σ_T v = 0`, and every triangle sums to zero (`|x+y+z|² = 3N − 3N = 0`), hence the
  uncovered points sum to zero and a *single* uncovered point is impossible — so at most 12. (The LP
  relaxation alone gives only 13.33.)
* **Weight.** Variables for triangles (weight 2) and for admissible pairs (`cos <= −1/2`: antipodal
  pairs and triangle edges, weight 1), required disjoint, maximise total weight. Certificate:
  `2⌊K/3⌋ + [K mod 3 = 2]`, since the groups are disjoint subsets of size `<= 3`. **This bound is
  valid for every `K`-point configuration whatsoever**, lattice or not.

```
$ .venv/bin/python python/template/triangles.py
d=2 K=6 triangles=2 admissible_pairs=9
  packing : max disjoint triangles = 2  lp_bound=2.0000 dual=2.0000 gap=0.0 nodes=0 leftover=0  zero_sum_bound=2 (floor(K/3))  PROVEN  0.01s
  weight  : max = 4 = 2*2 + 0  counting bound=4 gap=0.0 nodes=1  PROVEN  0.02s
  record  : 2 triangles + 0 pairs, weight 4, count 198550  -> OPTIMAL
d=3 K=12 triangles=8 admissible_pairs=30
  packing : max disjoint triangles = 4  lp_bound=4.0000 dual=4.0000 gap=0.0 nodes=1 leftover=0  zero_sum_bound=4 (floor(K/3))  PROVEN  0.01s
  weight  : max = 8 = 2*4 + 0  counting bound=8 gap=0.0 nodes=1  PROVEN  0.01s
  record  : 2 triangles + 3 pairs, weight 7, count 200044  -> SUBOPTIMAL by 1 weight = 496 vectors
  optimal groups: [[0, 6, 10], [1, 7, 8], [2, 4, 11], [3, 5, 9]]
d=4 K=24 triangles=32 admissible_pairs=108
  packing : max disjoint triangles = 8  lp_bound=8.0000 dual=8.0000 gap=0.0 nodes=1 leftover=0  zero_sum_bound=8 (floor(K/3))  PROVEN  0.01s
  weight  : max = 16 = 2*8 + 0  counting bound=16 gap=0.0 nodes=1  PROVEN  0.02s
  record  : 8 triangles + 0 pairs, weight 16, count 204520  -> OPTIMAL
d=5 K=40 triangles=80 admissible_pairs=260
  packing : max disjoint triangles = 12  lp_bound=13.3333 dual=12.0000 gap=0.0 nodes=1 leftover=4  zero_sum_bound=12 (floor(K/3) - 1: configuration sums to zero, triangles sum to zero, one leftover point impossible)  PROVEN  0.08s
  weight  : max = 26 = 2*12 + 2  counting bound=26 gap=0.0 nodes=1  PROVEN  0.24s
  record  : 12 triangles + 2 pairs, weight 26, count 209496  -> OPTIMAL
d=6 K=72 triangles=240 admissible_pairs=756
  packing : max disjoint triangles = 24  lp_bound=24.0000 dual=24.0000 gap=0.0 nodes=5 leftover=0  zero_sum_bound=24 (floor(K/3))  PROVEN  1.71s
  weight  : max = 48 = 2*24 + 0  counting bound=48 gap=0.0 nodes=1  PROVEN  2.75s
  record  : 24 triangles + 0 pairs, weight 48, count 220440  -> OPTIMAL
d=7 K=126 triangles=672 admissible_pairs=2079
  packing : max disjoint triangles = 42  lp_bound=42.0000 dual=42.0000 gap=0.0 nodes=1 leftover=0  zero_sum_bound=42 (floor(K/3))  PROVEN  10.58s
  weight  : max = 84 = 2*42 + 0  counting bound=84 gap=0.0 nodes=1  PROVEN  1.13s
  record  : 42 triangles + 0 pairs, weight 84, count 238350  -> OPTIMAL
```

| d | K | triangles | max disjoint triangles (ILP, gap 0) | hand bound | max weight (ILP, gap 0) | counting bound | record partition | record weight | optimal? |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 2 | 2 | 2 | 4 | 4 | 2 tri | 4 | yes |
| 3 | 12 | 8 | **4** | 4 | **8** | 8 | 2 tri + 3 pairs | 7 | **no: +1 weight = +496** |
| 4 | 24 | 32 | 8 | 8 | 16 | 16 | 8 tri | 16 | yes |
| 5 | 40 | 80 | 12 | 12 (zero-sum) | 26 | 26 | 12 tri + 2 pairs | 26 | yes |
| 6 | 72 | 240 | 24 | 24 | 48 | 48 | 24 tri | 48 | yes |
| 7 | 126 | 672 | 42 | 42 | 84 | 84 | 42 tri | 84 | yes |

An independent exact DFS (`max_triangle_partition` in `python/kiss_ref/small_kissing.py`) computes the
same numbers. It maximises the number of pairwise disjoint triangles subject to every uncovered point
being matched to its antipode, so that every `S_i` gets a group of weight at least 1:

| d | K(d) | ⌊K/3⌋ | max disjoint triangles | leftover pairs | weight Σ(\|T_i\|−1) | exhaustive | DFS nodes | time |
|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 2 | **2** | 0 | 4 | yes | 3 | 0.00 s |
| 3 | 12 | 4 | **4** | 0 | 8 | yes | 6 | 0.00 s |
| 4 | 24 | 8 | **8** | 0 | 16 | yes | 9 | 0.00 s |
| 5 | 40 | 13 | **12** | 2 | 26 | yes (13 infeasible, 18 nodes) | 18 | 0.09 s |
| 6 | 72 | 24 | **24** | 0 | 48 | yes | 1920 | 0.01 s |
| 7 | 126 | 42 | **42** | 0 | 84 | yes | 123324 | 0.53 s |

One caveat on that DFS table: its `triangle_partition(5, 13)` run with `require_pairs=True` is **not**
an exhaustive proof that D5 has no 13 disjoint triangles — a single leftover point is rejected by the
pairing rule before any branching happens. The ILP (gap 0, dual bound 12) and the zero-sum argument
are the actual proofs. See §7.

The optimal D3 partition, as indices into `config(3)` (the lex-sorted `(±1,±1,0)` roots):

```
[[0,6,10], [1,7,8], [2,4,11], [3,5,9]]
```

= `{(−1,−1,0),(0,1,−1),(1,0,1)}`, `{(−1,0,−1),(0,1,1),(1,−1,0)}`, `{(−1,0,1),(0,−1,−1),(1,1,0)}`,
`{(−1,1,0),(0,−1,1),(1,0,−1)}`; each sums to zero. These are the four oriented 3-cycles
`e_i − e_j, e_j − e_k, e_k − e_i` on four indices.

The consequence for every other dimension: within the template with these configurations, the counts
for dims 26–31 are linear in the `|S_i|` only, and each dimension's record moves by exactly its weight
— 4, 7 (now 8), 16, 26, 48, 84 — per unit of `|S|` beyond 496.

### 5.2 The dimension-27 improvement: K(27) >= 200540

**The record 27-dimensional partition is suboptimal.** The cuboctahedron (`D3 = A3` roots,
`K(3) = 12`) splits into **4** disjoint triangles — the four oriented 3-cycles above — for weight 8,
while the published 27D family, and the corresponding form in the Kallal–Kan–Wang table, uses
2 triangles + 3 antipodal pairs, weight 7. With the same 496-sets and the same 12 extra spheres this
gives

```
K(27) >= 12 + 196560 + 8·496 = 200540      (previous record 200044; +496)
```

The improvement uses **four** `S_i` rather than five, because four triangles need four disjoint
496-sets, not five.

This is verified end to end by `python/verify_dimN.py --strict` — exact casework over all
`C(200540, 2)` pairs plus the float GEMM — and independently by the C++ `disjoint_family --verify`,
on **two different families**: one built from the published 496-sets and one from the chain sets of
§3.5.

```
$ .venv/bin/python python/verify_dimN.py runs/template/dim27_record --strict
family      : runs/template/dim27_record  dim=27 d=3 sets=4 T=12 groups=4 extra=12 claimed_count=200540
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=4 sizes=[496, 496, 496, 496] union=1984 pairwise disjoint, each independent; equatorial = C \ U = 194576  (0.1 s)
T           : 12 vectors in Z^3-ambient field, norms ['2'], pairwise cos <= 1/2 (max 0.500000); groups: 4 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (0.0 s)
extra       : 12 vectors (K(3) = 12), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.853553)  (0.0 s)
rank        : span(T u extra) has exact rank 3 <= d = 3
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 12 + 196560 + 3968 = 200540  (rows: equatorial 194576 + lifted 5952 + extra 12)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (43.5 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=18929812600 eq-lift=1158116352 lift-lift-same-x=5952 lift-lift-same-set=4419360 lift-lift-cross-set=13284864 extra-eq=2334912 extra-lift=71424 extra-extra=66 total=20108045530=C(200540,2)
exact       : count = 200540  (44.7 s)
float       : 200540 vectors in R^27; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (194623, 195615) class=lift-lift-same-x; min -4.000000  (102.3 s, 38612 tiles)
RESULT ok=1 dim=27 d=3 sets=4 count=200540 count_exact=200540 count_float=200540 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=44.7 float_s=102.3 s=147.3
```

```
$ .venv/bin/python python/verify_dimN.py runs/template/dim27_ours --strict
family      : runs/template/dim27_ours  dim=27 d=3 sets=4 T=12 groups=4 extra=12 claimed_count=200540
config      : cpus=16 workers=8 tile=(512, 1024) OPENBLAS_NUM_THREADS=1
sets        : k=4 sizes=[496, 496, 496, 496] union=1984 pairwise disjoint, each independent; equatorial = C \ U = 194576  (0.1 s)
T           : 12 vectors in Z^3-ambient field, norms ['2'], pairwise cos <= 1/2 (max 0.500000); groups: 4 triangles + 0 pairs, disjoint, within-group cos <= -1/2  (0.0 s)
extra       : 12 vectors (K(3) = 12), pairwise cos <= 1/2 (max 0.500000), cos to every T vector <= sqrt3/2 (max 0.853553)  (0.0 s)
rank        : span(T u extra) has exact rank 3 <= d = 3
count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = 12 + 196560 + 3968 = 200540  (rows: equatorial 194576 + lifted 5952 + extra 12)
exact       : eq-lift max ip = 16 (needs <= 16 < 8 sqrt6); lift-lift max ip same set = 8 (needs <= 8), cross set = 16 (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)
exact       : eq-eq full pass over all 19317818520 pairs of C: max off-diagonal ip = 16, min = -32, values seen [-32, -16, -8, 0, 8, 16] (46.8 s)
exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); extra-extra: cos <= 1/2 (C); extra-eq: 0
exact       : pairs eq-eq=18929812600 eq-lift=1158116352 lift-lift-same-x=5952 lift-lift-same-set=4419360 lift-lift-cross-set=13284864 extra-eq=2334912 extra-lift=71424 extra-extra=66 total=20108045530=C(200540,2)
exact       : count = 200540  (47.9 s)
float       : 200540 vectors in R^27; max off-diagonal inner product = 2.000000000000 (tol 2+1e-9) at pair (194630, 195622) class=lift-lift-same-x; min -4.000000  (101.7 s, 38612 tiles)
RESULT ok=1 dim=27 d=3 sets=4 count=200540 count_exact=200540 count_float=200540 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=47.9 float_s=101.7 s=150.0
```

The shipped copies for external readers:

* `data/families/dim27_improved/` — `family.json` + `S_01..S_04.txt`, the improved family in the
  repository's family schema, checkable with `python/verify_dimN.py`;
* `docs/note/data/` — a standalone, self-contained certificate directory (`verify.py`, `S_01..S_04.txt`,
  `T.txt`, `extra.txt`, `family.json`, `README.txt`) that needs only Python 3 and NumPy, rebuilds the
  Golay code and all 196560 Leech minimal vectors from scratch, and prints
  `RESULT ok=1 dim=27 sets=4 weight=8 count=200540` in about two seconds. It also has `--full`
  (brute-force maximum inner product over all 19317818520 Leech pairs, about 90 s) and `--float`
  (floating-point pass over explicit `R^27` coordinates) modes.

The priority check and the independent co-discovery of this improvement are in
`06-record-and-priority.md`.

### 5.3 No other R^d configuration can add weight

No continuous search is needed for the conclusion. The counting bound `2⌊K/3⌋ + [K mod 3 = 2]` holds
for **every** `K`-point set, and the lattice configurations attain it for every `d` once the `d = 3`
partition uses 4 triangles — giving weights 4, 8, 16, 26, 48, 84. For `d = 5` a hypothetical
non-lattice 40-point configuration with 13 triangles would leave one point unusable, so weight 26
again.

Therefore the only way any configuration in `R^d` can add weight is by having **more than `K(d)`
points**, i.e. by a new kissing-number record in `R^5`, `R^6` or `R^7`
(`K(5) ∈ [40,44]`, `K(6) ∈ [72,77]`, `K(7) ∈ [126,134]`, per Cohn's table as of 2026-08-26) — out of
scope here. Nothing in this argument relies on the CJKT rigidity results (arXiv:1102.5060); they are
not needed.

The continuous search was still run as a sanity check of the tooling: penalty descent on
`Σ max(0, cos − 1/2)² + Σ_tri (cos + 1/2)²`, then SLSQP on the minimax value with the triangle
equalities as constraints (`runs/template/continuous_triangles.log`):

```
d=3 n=12 k=3 restarts=30: best max non-triangle cos = 0.500000000 (angle 60.000000 deg), triangle err 1.1e-16, kissing(<=1/2+1e-9)=True; feasible restarts 28, best values [0.5, 0.5, 0.5, 0.5, 0.5]  37.1s
d=3 n=12 k=4 restarts=30: best max non-triangle cos = 0.500000000 (angle 60.000000 deg), triangle err 4.0e-14, kissing(<=1/2+1e-9)=True; feasible restarts 30, best values [0.5, 0.5, 0.5, 0.5, 0.5]  20.3s
d=3 n=13 k=4 restarts=20: best max non-triangle cos = 0.612372435 (angle 52.238756 deg), triangle err 1.7e-16, kissing(<=1/2+1e-9)=False; feasible restarts 20, best values [0.612372, 0.612372, 0.612372, 0.612372, 0.612372]  8.4s
d=3 n=13 k=0 restarts=20: best max non-triangle cos = 0.542636487 (angle 57.136703 deg), triangle err 0.0e+00, kissing(<=1/2+1e-9)=False; feasible restarts 20, best values [0.542636, 0.545344, 0.546297, 0.546297, 0.546297]  10.8s
```

The 12-point runs recover a genuine kissing configuration with 3 or 4 prescribed triangles; the
13-point runs never get the maximum cosine down to `1/2`, consistent with `K(3) = 12`. (The driver
script for that log then raised a `TypeError` on a `None` result after the four reported rows; the
rows themselves are complete and the numbers above are what it produced. This is a sanity check, not
a load-bearing result — the counting bound is.)

### 5.4 Extra spheres are capped at K(d)

The extra spheres form a spherical code with pairwise `cos <= 1/2` in `R^d` — that is, a kissing
configuration in `R^d` in its own right. Hence `|E| <= K(d)`, unconditionally. Since `K(d)` is known
exactly for `d <= 4`, the published `K(d)` extras are **optimal** there; for `d = 5, 6, 7` they can be
beaten only by a new kissing-number record in that dimension.

Independently of that argument, three exact search strategies were run
(`python/kiss_ref/small_kissing.py`), each returning only exactly verified configurations (integer or
`Q(√M)`):

1. `lattice_extra_spheres` — primitive integer vectors of the span with entries in `[−2, 2]` at
   `>= 30°` from every `T` vector, giving a boolean compatibility graph, then a greedy clique with
   `(1,2)`-swap local search;
2. `rotated_extra_spheres_exact` — the configuration itself rotated by products of at most 3 plane
   rotations by 45° (`Q(√2)`) or 30°/60° (`Q(√3)`) inside its span, accepted when every image is at
   least 30° from `T` (exact, via `check_extra_ab`); this produces the schema's `(a + b√M)/D` form
   directly;
3. `rotated_extra_spheres` — random rational Cayley rotations (sympy), with exact integer output.

Results (`seed=0`; lattice with `max_entry` 2 and `max_candidates` 12000; exact rotations at most 3
planes and at most 3000 tries; Cayley 200 tries):

| d | K(d) | lattice (entries ≤ 2) | exact rotation, field | rotation found (planes (a, b), angle) | max cos to T | Cayley |
|---|---|---|---|---|---|---|
| 2 | 6 | **6** (6 candidates; (2,−1,−1)-type) | 6, Q(√3), D = 3 | ((1,−1,0),(1,1,−2)) +30° | √3/2 (tight) | 0 |
| 3 | 12 | 10 (38 cand.; 10 also with entries ≤ 3, 86 cand.) | **12**, Q(√2), D = 2 | ((1,1,0),(1,−1,0)) +45° | (2+√2)/4 = 0.8536 (31.4°) | 0 |
| 4 | 24 | **24** (280 cand.) | 24, Q(√3), D = 2 | (e₁,e₄) −30°, ((0,1,1,0),(0,1,−1,0)) +60° | 1/√2 (45°) | 24 (D = 5) |
| 5 | 40 | 34 (2042 cand.; 34 with entries ≤ 3, 11002 cand.) | **40**, Q(√2), D = 2 | (e₁,e₃) +45°, ((0,1,0,0,1),(0,1,0,0,−1)) +45° | (2+√2)/4 | 0 |
| 6 | 72 | 50 (7080 cand.) | **72**, Q(√3), D = 6 | (e₇+e₈, e₇−e₈) +30°, (e₄+e₅, e₄−e₅) +60°, (e₁+e₂+e₃, e₆) +30° (831 tries, 43 s) | √3/2 | 0 |
| 7 | 126 | 104 (12000 of the candidates) | **126**, Q(√2), D = 2 | (e₁+e₂, e₃) +45°, (e₅, e₇) +45°, (e₆+e₈, e₆−e₈) −45° (240 tries, 59 s) | (2+√2)/4 | 0 |

Every `d` reaches `K(d)` extra spheres exactly, in the same fields the published data uses — its
27D/29D/31D extras are `Q(√2)` rotations with denominators 2/2/4, its 26D/28D/30D extras `Q(√3)`
rotations with denominator 3. All results are verified by `check_extra_ab` and by the general field
arithmetic. The integer grid never exceeded `K(d)` and the clique search on entries `<= 3` found
nothing beyond it — but that on its own is weak evidence; the `|E| <= K(d)` argument above is the
real statement.

An ILP over an explicit candidate pool confirms the same maxima with proof
(`runs/template/extra_spheres.log`), for the dimensions where it terminated:

```
d=2 K=6 bound=6 tight=True: extras have pairwise cos <= 1/2, i.e. form a kissing configuration in R^d; K(d) is known exactly, so K(d) extras is optimal
  pool: raw=258 after-30deg=84 dedup=6 sources={'grid': 6}
  ILP max independent set = 6  lp_bound=6.000 dual=6.000 gap=0.0 nodes=0 edges=0 cliques=0 0.0s  chosen sources={'grid': 6} exact-known=6/6
  record extras exact check: {'ok': True, 'n': 6}
  continuous n=K+1=7: best max violation 1.720e-01 (penalty 1.671e-01) hit=False
d=3 K=12 bound=12 tight=True: extras have pairwise cos <= 1/2, i.e. form a kissing configuration in R^d; K(d) is known exactly, so K(d) extras is optimal
  pool: raw=686 after-30deg=312 dedup=94 sources={'grid': 38, 'rotation': 48, 'record': 8}
  ILP max independent set = 12  lp_bound=12.000 dual=12.000 gap=0.0 nodes=1 edges=934 cliques=149 0.0s  chosen sources={'grid': 8, 'rotation': 4} exact-known=8/12
  record extras exact check: {'ok': True, 'n': 12}
  continuous n=K+1=13: best max violation 4.901e-02 (penalty 4.311e-02) hit=False
d=4 K=24 bound=24 tight=True: extras have pairwise cos <= 1/2, i.e. form a kissing configuration in R^d; K(d) is known exactly, so K(d) extras is optimal
  pool: raw=1576 after-30deg=712 dedup=504 sources={'grid': 280, 'rotation': 224}
  ILP max independent set = 24  lp_bound=25.448 dual=24.000 gap=0.0 nodes=1 edges=22348 cliques=1501 10.4s  chosen sources={'grid': 24} exact-known=24/24
  record extras exact check: {'ok': True, 'n': 24}
  continuous n=K+1=25: best max violation 7.667e-02 (penalty 1.044e-01) hit=False
```

The `d = 5` ILP hit the HiGHS time limit and the driver stopped there; that is a cap on the run, not a
negative result, and it does not affect anything, since the `|E| <= K(d)` argument already settles the
question for every `d`. The continuous `K(d)+1` searches never came close to feasibility (best
violations `1.7e-1`, `4.9e-2`, `7.7e-2` for `d = 2, 3, 4`), as expected.

### 5.5 Template test suite

```
$ .venv/bin/python -m pytest python/tests -q          # runs/template/pytest_all.log
........................................................................ [ 33%]
........................................................................ [ 67%]
......................................................................   [100%]
214 passed in 189.14s (0:03:09)
exit=0
........................                                                 [100%]
24 passed in 147.75s (0:02:27)
exit_dimN=0
```

---

## 6. Above dimension 31: the template cannot win

The question is whether the same template beats the recorded lower bound in any dimension `n >= 32`.
It does not — and with the disjoint families that actually exist it is not close, losing by between
93 000 and 52 000 000. Nothing new was constructed here and nothing new is claimed for `n >= 32`.

### 6.1 The structural reason: the Leech half, not the R^d half

The bottleneck is **not** the `R^d` configuration. It is the Leech half. Every group `T_i` needs its
own **pairwise disjoint** 60°-free subset `S_i ⊂ C`, so

```
lifted term = Σ_i (|T_i| − 1)·|S_i| ≤ 2·Σ_i |S_i| ≤ 2·196560 = 393120,
```

and the whole template can therefore never exceed

```
K(d) + 196560 + 393120 = K(d) + 589680.
```

Records pass that ceiling at `n = 39` (755988 > 592244) and never come back. Below `n = 39` the
ceiling is only reachable if one can construct roughly 300–400 pairwise disjoint 496-sets, i.e. a
near-perfect partition of the 196560 Leech minimal vectors into 60°-free classes. The largest such
family known anywhere is **59 sets totalling 28324 vectors** (Kallal–Kan–Wang 2018, re-verified here,
§6.6); the chain construction of §3.5 gives **42 × 496 = 20832**. Both are about 7× short of what
would be needed even in the two dimensions (37, 38) where the ceiling would win.

Concretely:

* The template loses at every `n >= 32` with **42 × 496** sets — by 107 968 at `n = 32` to 51 981 216
  at `n = 48`.
* It loses at every `n >= 32` with the **59 Kallal–Kan–Wang sets** (28324 vectors, the largest
  constructed family) — by 92 984 to 51 966 232.
* At the **unreachable ceiling** it would win only at **`n = 37` (+82 906)** and **`n = 38`
  (+24 960)**.
* **Crossover: `n = 39`.** From `n = 39` upward the template loses even at its absolute ceiling,
  permanently.
* At `n = 32..36` and `n >= 39` the template is dead *even at the ceiling* — no amount of further work
  on disjoint families can rescue it there. `n = 36` is the near miss: `−9407` at the ceiling.

### 6.2 The three caps

1. **`R^d` weight.** Groups are disjoint subsets of a `K(d)`-point 60°-code with pairwise
   `cos <= −1/2`, so `|T_i| <= 3` and the total weight is at most
   `weight_upper_bound(K) = 2⌊K/3⌋ + [K ≡ 2 mod 3]` (§5.1; valid for *any* `K`-point set). This is
   generous — a pure counting bound that ignores whether a perfect triangle partition exists.
2. **Extra spheres.** `E` is itself a 60°-code in `R^d`, so `|E| <= K(d)`. `|E| = K(d)` is used
   throughout (again generous; verified attainable for `d = 8` in §6.5).
3. **The Leech half.** The `S_i` are pairwise disjoint 60°-free subsets of the 196560 minimal
   vectors, each of size at most 496 (the best known independent set). Hence

```
Σ_i |S_i| ≤ 196560,   lifted ≤ 2·196560 = 393120,   at most ⌊196560/496⌋ = 396 groups get a 496-set.
```

In practice the binding number is not 396 but 42 or 59, pinning the lifted term at 41664 or 56648
respectively — about 7× below the ceiling.

That third cap is exactly what changes at `n = 32`. For `n <= 31` the `R^d` configuration is small
(`K(7) = 126` gives 42 triangles), so the set count and the group count coincide and the template is
tight. From `d = 8` upward the `R^d` side has far more triangles than there are sets — E8 alone gives
80 — and the whole question collapses to "how many pairwise disjoint 496-sets can you build?".

### 6.3 What actually holds dimensions 32–37

Cohn's table (<https://cohn.mit.edu/kissing-numbers/>, fetched 2026-08-27; the page still carries no
"last updated" date), lower and upper bounds and the attributed source of the lower bound:

| n | lower | upper | lower-bound source |
|---|---|---|---|
| 24 | 196560 | 196560 | Leech 1967 |
| 25 | 197056 | 265006 | Ma et al. 2025 (PackingStar) |
| 26 | 198550 | 367775 | Ma et al. 2025 |
| 27 | 200044 | 522212 | Ma et al. 2025 — **superseded by 200540, §5.2** |
| 28 | 204520 | 752292 | Ma et al. 2025 |
| 29 | 209496 | 1075991 | Ma et al. 2025 |
| 30 | 220440 | 1537707 | Ma et al. 2025 |
| 31 | 238350 | 2213487 | Ma et al. 2025 |
| 32 | 345408 | 3162316 | Brouwer (n.d.) |
| 33 | 360640 | 4494570 | Brouwer |
| 34 | 380868 | 6422593 | Brouwer |
| 35 | 409548 | 9162403 | Brouwer |
| 36 | 484568 | 13017098 | Brouwer |
| 37 | 494312 | 18498316 | Brouwer |
| 38 | 566652 | 26496684 | Brouwer |
| 39 | 755988 | 37826766 | Brouwer |
| 40 | 1064368 | 53589200 | Brouwer |
| 41 | 1170384 | 76287040 | Brouwer |
| 42 | 1250676 | 108404055 | Brouwer |
| 43 | 2060399 | 153813582 | Sun & Wang 2026 (arXiv:2607.20359) |
| 44 | 2948552 | 220788272 | Edel, Rains & Sloane 1998 |
| 45 | 3047160 | 316735249 | Brouwer |
| 46 | 5318060 | 441900184 | Brouwer |
| 47 | 9741412 | 621658419 | Brouwer |
| 48 | 52416000 | 867897072 | Leech & Sloane 1971 |

All upper bounds are de Laat & Leijenhorst 2024.

**The construction behind 32–37 is not the Leech template.** "Brouwer" on that page is the
constant-weight-code table feeding **Edel–Rains–Sloane 1998, *On kissing numbers in dimensions 32 to
128*** (Electron. J. Combin. 5, #R22) — a purely binary-code construction. Its shape for `n >= 32` is

```
τ_n ≥ 2^17 + 2^7 · A(n, 8, 8) + 2n(n − 1)
```

where `2^17 = A(32,8)` from the Cheng–Sloane `[32,17,8]` code and `2n(n−1) = 2^2·C(n,2)`. It
reproduces the table values exactly: `A(32,8,8) >= 1659 → 345408`; `1777 → 360640`; `1934 → 380868`;
`2817 → 494312` (checked in `test_record_table_uses_echols_where_it_improves_cohn`).

**arXiv:2608.13906** (W. Echols, submitted 2026-08-14), *New lower bounds for constant-weight codes via
seeded bit-swap tabu search*, is exactly this: a tabu search for better `A(n,d,w)`, whose improved
`A(n,8,8)` plug straight into the Edel–Rains–Sloane formula. **It does not use the lifting template.**
Its Table 2:

| τ_n | prior | new | gain | from A(n,8,8) |
|---|---|---|---|---|
| τ_32 | 345408 | **346432** | +1024 | 1659 → 1667 |
| τ_33 | 360640 | **362048** | +1408 | 1777 → 1788 |
| τ_34 | 380868 | **381124** | +256 | 1934 → 1936 |
| τ_37 | 494312 | **496232** | +1920 | 2817 → 2832 |

These were not yet reflected on Cohn's page as of the 2026-08-27 fetch. Everything below compares
against the **stronger** of the two — Echols where it applies — i.e. the template is given the harder
target.

Nothing else in 32–37 has moved: the only other recent items in this range are Sun–Wang's antipode
construction (dimensions 39, 43, 45) and Edel–Rains–Sloane's own 44.

### 6.4 The comparison table

`.venv/bin/python python/template/highdim.py`, log at `runs/highdim/highdim.log`. Here `K(d)` is
Cohn's lower bound for the `R^d` kissing number; `tri = ⌊K/3⌋` triangles available; `wmax` the
counting weight bound; `w@42`/`val@42` the weight and count with 42 × 496; `val@kkw59` the count with
the 59 Kallal–Kan–Wang sets (28324 vectors); `w@ceil`/`sets`/`val@ceil` the ceiling (unlimited disjoint
sets, only the 196560 budget and the 496 cap binding); `record` the best published lower bound (Echols
where it improves Cohn).

```
  n   d    K(d)   tri  wmax  w@42    val@42  val@kkw59  w@ceil  sets  val@ceil     record    diff@42   diff@kkw   diff@ceil
---------------------------------------------------------------------------------------------------------------------------
 32   8     240    80   160    84    238464     253448     160    80    276160     346432    -107968     -92984      -70272
 33   9     306   102   204    84    238530     253514     204   102    298050     362048    -123518    -108534      -63998
 34  10     510   170   340    84    238734     253718     340   170    365710     381124    -142390    -127406      -15414
 35  11     604   201   402    84    238828     253812     402   201    396556     409548    -170720    -155736      -12992
 36  12     841   280   560    84    239065     254049     560   280    475161     484568    -245503    -230519       -9407
 37  13    1154   384   769    84    239378     254362     769   385    579138     496232    -256854    -241870      +82906
 38  14    1932   644  1288    84    240156     255140     794   397    591612     566652    -326496    -311512      +24960
 39  15    2564   854  1709    84    240788     255772     794   397    592244     755988    -515200    -500216     -163744
 40  16    4320  1440  2880    84    242544     257528     794   397    594000    1064368    -821824    -806840     -470368
 41  17    5730  1910  3820    84    243954     258938     794   397    595410    1170384    -926430    -911446     -574974
 42  18    7654  2551  5102    84    245878     260862     794   397    597334    1250676   -1004798    -989814     -653342
 43  19   11948  3982  7965    84    250172     265156     794   397    601628    2060399   -1810227   -1795243    -1458771
 44  20   19448  6482 12965    84    257672     272656     794   397    609128    2948552   -2690880   -2675896    -2339424
 45  21   29768  9922 19845    84    267992     282976     794   397    619448    3047160   -2779168   -2764184    -2427712
 46  22   49896 16632 33264    84    288120     303104     794   397    639576    5318060   -5029940   -5014956    -4678484
 47  23   93150 31050 62100    84    331374     346358     794   397    682830    9741412   -9410038   -9395054    -9058582
 48  24  196560 65520 131040    84    434784     449768     794   397    786240   52416000  -51981216  -51966232   -51629760

beats the record at 42 x 496 = 20832 : NONE
beats the record at KKW 59 sets (28324) : NONE
beats the record at ceiling : [37, 38]
crossover (ceiling never wins again from) : n = 39
```

`K(d)` sources (Cohn, same fetch): `d = 8` Korkine–Zolotareff 1873; 9 Leech–Sloane 1971; 10 Ganzhinov
2025 (510); 11 Bianchi et al. 2026 (604); 12 Takhanov et al. 2026 (841); 13 Zinoviev–Ericson 1999
(1154); 14 Ganzhinov 2025 (1932); 15 Leech–Sloane 1971 (2564); 16 Barnes–Wall 1959 (4320); 17–21
Cohn–Li 2024 / Ho 2026; 22–24 Leech 1967.

### 6.5 How many disjoint 496-sets would be needed

```
  n= 32: needs  151 disjoint 496-sets to beat 346432  (triangles available  80, feasible=False)
  n= 33: needs  167                        362048    (triangles available 102, feasible=False)
  n= 34: needs  186                        381124    (triangles available 170, feasible=False)
  n= 35: needs  215                        409548    (triangles available 201, feasible=False)
  n= 36: needs  290                        484568    (triangles available 280, feasible=False)
  n= 37: needs  301                        496232    (triangles available 384, feasible=TRUE)
  n= 38: needs  372                        566652    (triangles available 644, feasible=TRUE)
  n= 39: needs  562                        755988    (triangles available 854, feasible=False)
  ...  n = 48 would need 52443 (ceiling 396)
```

Here "feasible" means that enough weight-2 groups exist *and* the 196560-vector budget allows that
many 496-sets. At `n = 32..36` the `R^d` configuration simply does not contain enough disjoint
triangles (`n = 32` needs 151 triangles; E8 has 80). At `n >= 39` the 196560-vector budget runs out
first. **`n = 37` and `n = 38` are the only two dimensions where the obstruction is purely "we cannot
build that many disjoint sets".**

Sanity check: the same arithmetic reproduces every count already verified below 32
(`test_reproduces_verified_counts`). With 42 × 496 the formula gives 198550 (`n = 26`), **200540**
(`n = 27`, the improvement of §5.2), 204520, 209496, 220440 and 238350 — matching the families
verified in §4. Dimension 25 is the one place where `|E| = K(d)` fails: `R^1` has no direction 30°
from `±1`, so `|E| = 0` and the count is 197056.

### 6.6 The R^d half at d = 8: explicit, exact, and optimal

The only `R^d` configuration that needed building is E8. The `R^d` side is never the bottleneck above
`d = 7`, but this pins down the two "is it attainable?" questions.

**Configuration: the 240 E8 roots**, in integer coordinates of squared norm 8 in the scaling of
`python/kiss_ref/small_kissing.py` — `(±2,±2,0^6)` and `(±1)^8` with an even number of minus signs.
This is `K(8) = 240` exactly, the unique optimal 8-dimensional kissing configuration.

**Eisenstein structure gives a perfect triangle partition.** E8 contains `A2^4`; rotating each A2
plane by 120° gives an isometry `σ` with `σ² + σ + 1 = 0`. That rotation lies in `W(A2)`, which acts
trivially on `A2*/A2`, so the tetracode glue is preserved and `σ ∈ Aut(E8)` — checked directly by
mapping all 240 roots. In ambient coordinates `σ = M/2` with `M` integral, `M Mᵀ = 4I` and
`M² + 2M + 4I = 0`. Every `σ`-orbit on the roots has size 3 and sums to zero, i.e. **is a triangle**:

```
E8 roots K=240, sigma order 3 fixed-point-free, hexagon partition:
  80 disjoint triangles covering 240/240 points, weight 160 = counting bound 160
```

So the "hexagons, halved" recipe does apply, and it gives `2K/3 = 160` exactly, attaining the counting
bound `weight_upper_bound(240) = 160` — **optimal with no solver at all**. Cross-checked anyway with
the HiGHS ILP of §5.1: maximum disjoint triangle packing = 80, gap 0, proven, 3.3 s
(`test_e8_ilp_confirms_80_triangles`). The same argument covers every Eisenstein configuration (K12
with 756, the complex Leech lattice with 196560, and so on).

**Extra spheres: `|E| = K(8) = 240` is attainable, exactly.** With `J = (2σ+1)/√3` (so `J² = −1`, from
`σ² + σ + 1 = 0`), `Rot = cos30°·I + sin30°·J = (σ + 2)/√3` is an isometry, so `Rot(T)` is again a
240-point 60°-code. For roots `v, w` put `a = ⟨σv, w⟩ + 2⟨v, w⟩ ∈ Z`; then `⟨Rot v, w⟩ = a/√3` and the
`>= 30°` condition is `|a| <= 3N/2 = 12`. Measured:

```
extra spheres via R=(sigma+2I)/sqrt3: ok=True |E|=240 max|a|=12 2*limit=24 touching=960
```

`max |a| = 12` exactly: **960 of the 57600 `(E, T)` pairs sit at exactly 30°**, the rest strictly
inside. So `d = 8` behaves like `d = 2..7` — the extras reach `K(d)` — and by the same mechanism, a
30° rotation of the `T` configuration. All integer arithmetic; no floats enter the decision.

For `d = 9..24` no explicit `K(d)`-attaining configuration was built and `|E| = K(d)` was **assumed**.
That assumption only makes the template look better than it is, so it does not affect the negative
conclusion.

### 6.7 The largest disjoint family that exists

The chain construction of §3.5 gives 42 pairwise disjoint 496-sets (`Σ = 20832`) from a single `Co_0`
element of order 42. Checking the other repository already on disk:
**`data/external/Kissing-Numbers/S_1..S_59.txt` (Kallal–Kan–Wang 2018) is a family of 59 pairwise
disjoint sets**, of sizes 488 (×24), 486, 484 (×4), … down to 460, with `Σ = 28324`. It was verified
here from scratch (`kkw_family_sizes`, `test_kkw_family_...`): every vector has squared norm 32 in the
Leech scaling used throughout, every set has Gram off-diagonal `<= 8` (no 60° pair), and all
`C(59,2) = 1711` set intersections are empty.

```
Kallal-Kan-Wang disjoint family: 59 sets, total 28324 vectors, verified ok=True
```

So the best lifting budget anyone has ever constructed is **28324 vectors → lifted 56648**, against
the **196560 → 393120** the ceiling assumes. That is a factor of 6.9, and it is the single binding
constraint in the only two dimensions the template could otherwise win. It is also a genuinely hard
open problem: 300 disjoint 496-sets would cover 148800 of the 196560 minimal vectors (76 %) with
60°-free classes — essentially a near-perfect colouring of a 196560-vertex, 4600-regular graph. For a
vertex-transitive graph the fractional chromatic number is `χ_f = |V|/α = 196560/496 = 396.29`, so even
396 classes cannot be a full colouring; but 300 disjoint classes are not excluded by that, and nothing
is known either way. Filed as a follow-up, not attempted.

### 6.8 What was verified above dimension 31, and what was not

**Verified** (exact integer arithmetic plus tests; `runs/highdim/pytest_highdim.log`, 22 passed):

* the arithmetic reproduces every already-verified count for `n = 25..31`, including 200540;
* the ceiling `2·196560 = 393120` and the 396-set cap;
* E8: it is a kissing configuration; `σ` is a fixed-point-free order-3 automorphism; the 80-triangle
  partition is perfect and attains the counting bound; the ILP independently proves 80; the 240 extra
  spheres satisfy the `>= 30°` condition exactly;
* the Kallal–Kan–Wang 59-set family is disjoint, independent, and totals 28324;
* the Edel–Rains–Sloane formula reproduces both Cohn's and Echols' values at 32, 33, 34 and 37.

**No improvement was found, so nothing was built and `verify_dimN.py` was not run** for `n >= 32` —
there is no new configuration to certify.

**Not checked / assumed:**

* `|E| = K(d)` for `d = 9..24` (assumed; proved only for `d <= 8`). Generous to the template, so
  harmless to the conclusion.
* No explicit `K(d)`-attaining spherical code was constructed for `d = 9..24` (`Λ9..Λ16`, K12, BW16
  and so on). Not needed: the weight bound `2⌊K/3⌋ + [K ≡ 2]` is a counting bound valid for every
  `K`-point set, so using Cohn's record `K(d)` already gives the *maximum possible* template value for
  that dimension. If such a code has no perfect triangle partition, the true value is lower, not
  higher.
* Whether more than 59 disjoint 60°-free subsets of `C` can be constructed, and how large a total
  `Σ|S_i|` is achievable. This is the only open door left — `n = 37` and `n = 38` — and it is wide:
  301 and 372 sets of size about 496 respectively.
* Whether `K(12)` could grow enough to reopen `n = 36` (the `−9407` near miss). At the ceiling,
  `n = 36` needs `K(12) >= 869` (currently 841) **and** a near-perfect disjoint family. Both, not
  either.
* arXiv:2608.13906 was read from the PDF; no attempt was made to re-derive or re-verify its
  constant-weight codes.

The `n >= 32` computation is pure numpy plus one HiGHS ILP cross-check, and runs in 4 s
(`python/template/highdim.py`, 22 tests in `python/tests/test_highdim.py`).

### 6.9 Scope of the negative result

Stated precisely: **for every `n >= 32`, a configuration of the form**

```
count = |E| + 196560 + Σ_i (|T_i| − 1)·|S_i|
```

**with `E` a 60°-code in `R^{n−24}`, the `T_i` disjoint groups of pairwise-`cos <= −1/2` vectors in a
`K(n−24)`-point 60°-code, and the `S_i` pairwise disjoint 60°-free subsets of the 196560 Leech minimal
vectors, cannot exceed the best published lower bound for dimension `n`.** For `n = 32..36` and
`n >= 39` this holds even under the most generous assumptions available (`|E| = K(d)` at the record
`K(d)`, the counting weight bound, `|S_i| = 496`, and an unlimited supply of disjoint sets subject
only to the 196560-vector budget). For `n = 37, 38` it holds for every disjoint family that has been
constructed — the 42 here and the 59 of Kallal–Kan–Wang — and would only be overturned by
constructing roughly 301 and 372 pairwise disjoint sets of size about 496 respectively.

This is a statement about **this template**, not about kissing numbers. It says nothing about what
other constructions can achieve in those dimensions, and indeed the current records for 32–37 come
from a completely different (binary-code) construction. It is also not an upper bound on `K(n)`; the
upper bounds in §6.3 are de Laat–Leijenhorst's and are orders of magnitude larger.

The contrast with the dimension-27 result is worth stating. That one was a mis-set parameter inside a
regime where the template was already the record holder. Here the template is not the record holder
and is not within reach of becoming one; the inherited assumption worth re-examining — that the `S_i`
must be pairwise disjoint — turns out to be a real, load-bearing constraint rather than an oversight.

---

## 7. Corrections

Corrections to earlier analyses in this repository, gathered in one place.

**1. The dimension-27 partition weight (the important one).** The record 27D family, and earlier
analysis here, used 2 triangles + 3 antipodal pairs in the cuboctahedron for weight 7. The exhaustive
DFS had already computed that D3 admits **4** disjoint triangles, weight 8; the earlier verifier
report nonetheless concluded that "the record forms are exactly the optimal assignments of the `K(d)`
configuration", which is false for `d = 3` — it read the `d = 5` and `d = 7` agreements as covering
every dimension and did not compare the `d = 3` optimum against the record partition. The corrected
statement: every dimension's record partition is optimal **except** `n = 27`, where four triangles give
weight 8 and

```
K(27) >= 12 + 196560 + 8·496 = 200540
```

verified end to end on two independent families (§5.2).

**2. The proof that D5 has no 13 disjoint triangles.** The DFS table records `d = 5` as "exhaustive
(13 infeasible, 18 nodes)". That run — `triangle_partition(5, 13)` with `require_pairs=True` — was
**not** an exhaustive proof: a single leftover point is rejected by the pairing rule before any
branching occurs, so the search never explored the space. The actual proofs are the ILP (dual bound
12, gap 0) and the solver-free zero-sum argument: all six models are antipodal so `Σ_T v = 0`, every
triangle sums to zero, hence the uncovered points sum to zero and a single uncovered point is
impossible. The conclusion (12 disjoint triangles, weight 26) is unchanged.

**3. Strict mode against SCHEMA.md §3.3.** The verifier version on disk at one point *failed* the 29D
family under `--strict` with "group 13: pair [27, 32] is not antipodal (allowed by the template, not
by SCHEMA.md)". That was wrong: SCHEMA.md §3.3 explicitly allows a size-2 group to be two vertices of
a triangle (`cos = −1/2`), notes that the record 29D family uses exactly that, and the template needs
only `cos <= −1/2`. The check is now a note — printed as `warning`, with the cosine and angle — in
both modes; the failing condition is `cos > −1/2`. The 29D groups 13 and 14 are such pairs, `[27, 32]`
and `[21, 3]`.

**4. Product replacement without an accumulator.** The first implementation used plain product
replacement and produced only 980 distinct elements in 1000 outputs, with 23 vertex-0 image collisions
against an expectation of 2.5. It was replaced by the "rattle" variant with an accumulator
(Leedham-Green–Murray), which gives 1000/1000 distinct elements and collision counts consistent with
uniform. See §2.4.

**5. The "24-dimensional cross".** The 496-set is *not* the coordinate frame, contrary to how the
repository README once described it: the frame would contribute 48 vectors of shape `(±4, ±4)`, while
the 496 contains only 6, from three coordinate pairs. Shape counts are 258/232/6 (see §2.6 and
`04-structure-of-the-496.md`).

**6. Random pools do not reach 42.** The group work predicted that "a pool of a few thousand random
images should contain many pairwise-disjoint 42-cliques". That prediction is wrong. With
`p = 0.544` the expected number of 42-cliques in a pool of 20000 is `10^{-98}`; a pool of about
`4·10^6` images is needed. Local search on pools up to 20000 reaches 21. The orbit-of-one-element
construction is what reaches 42, because it needs 21 conditions rather than `C(42,2) = 861` (§3.3,
§3.5).

**7. Verifier test fixtures.** Three test fixtures were themselves wrong and were fixed. The
"partially overlapping" fixture replaced one row of `S₂` by `S[0]`, which is at 60° from 12 rows of
`S₂`, so the set was correctly rejected as *dependent* before the overlap check ever ran; the fixture
now keeps only the rows of `S₂` compatible with `S[0]` (at least 400 of them) plus `S[0]`, and the
verifier reports "sets 1 and 2 overlap … [1 shared vector(s)]". A `lambda: list.pop()` returned the
popped group as the json. The rank fixture now bumps `T.K`, since the reader's row-count check fires
first otherwise (also asserted).

**8. Cohn's table is behind in two ways.** Its entry for `n = 27` (200044) is superseded by 200540
(§5.2), and its entries for `n = 32, 33, 34, 37` do not yet reflect Echols' improved constant-weight
codes (arXiv:2608.13906). The `n >= 32` comparison above uses the *stronger* of the two in every
dimension, which is the harder target for the template.

---

## 8. Files, commands, and RESULT lines

### 8.1 Data and certificates

| Path | Content |
|---|---|
| `data/families/SCHEMA.md` | schema v1 of `family.json` + `S_xx.txt` |
| `data/families/dim{26..31}/` | the converted published record families: 52 distinct sets committed once (`dim31/S_01..42`, `dim26/S_01..02`, `dim28/S_01..08`; dim27/29/30 reference `../dim31/`) |
| `data/families/ours/dim{26..31}/` | the chain families: `ours/dim31/S_01..42.txt` + `g.txt`; dims 26..30 reference `../dim31/S_01..S_k.txt` |
| `data/families/dim27_improved/` | the improved weight-8 dimension-27 family, `family.json` + `S_01..S_04.txt`, count 200540 |
| `docs/note/data/` | standalone `K(27) >= 200540` certificate: `verify.py`, `S_01..S_04.txt`, `T.txt`, `extra.txt`, `family.json`, `README.txt`; needs only Python 3 and NumPy |
| `data/group/m24_generators.txt`, `data/group/sextet.txt`, `data/group/xi.txt` | M24 generators, the sextet through {0,1,2,3}, and Conway's ξ |
| `data/external/Kissing-Numbers/S_1..S_59.txt` | the Kallal–Kan–Wang 59-set disjoint family (Σ = 28324) |
| `data/external/coordinate_map.json` | the coordinate permutation onto the published record data |
| `data/S496.txt`, `data/S488.txt` | the converted 496-set and a 488-set |

### 8.2 Code

| Path | Content |
|---|---|
| `python/group/m24.py`, `python/group/__init__.py` | M24 generators, code-preservation check, sympy order, sextet, ξ numerators, numpy verification of an `Aut` on all 196560 vectors, writer for `data/group/` |
| `include/kiss/group.h`, `src/group.cpp` | `Monomial`, `Aut`, `IndexPerm`, `find_sextet`, `make_xi`, loaders, `ProductReplacement`; plus `try_apply`, `transpose`, `aut_from_monomial`, `load_sextet`, `slot()` |
| `tools/random_aut.cpp` | random `Co_0` elements as index permutations |
| `include/kiss/family.h`, `src/family.cpp` | masks, image pools, disjointness graph, greedy / local-search / exact clique, sweep, chain, family directory I/O + verification |
| `tools/disjoint_family.cpp` | `--method chain\|sweep\|greedy\|ls\|bb`, `--verify dir` |
| `python/tools/extract_packingstar_families.py` | exact decomposition of the published float configurations into `S_i`, `T_i`, extras; conversion and verification |
| `python/tools/nest_family.py` | writes `ours/dim26..30/family.json` from `ours/dim31` |
| `python/verify_dimN.py` | the dimension-N verifier |
| `python/kiss_ref/small_kissing.py`, `python/kiss_ref/exact.py` | the `R^d` models, angle predicates, partitions, extra-sphere searches; exact arithmetic in `Q(√p_1, …)` |
| `python/template/triangles.py`, `extra_spheres.py`, `assign.py` | ILP optima, extra-sphere ILP, family rewriting with the optimal partition |
| `python/template/highdim.py` | Cohn's table, the caps, the `n >= 32` comparison, E8 (σ, hexagon partition, 240 extras), `kkw_family_sizes` |
| `tests/test_group.cpp`, `tests/test_family.cpp` | ctest targets `test_group`, `test_family` |
| `python/tests/test_m24.py`, `test_small_kissing.py`, `test_verify_dimN.py`, `test_template.py`, `test_highdim.py` | 11 / 46 / 24 / 47 / 22 tests |

Search-run outputs (not committed) live under `runs/families/` (`pool_*`, `dim26..31` sweeps,
`dim31_chain*`, `dim{26..30}_chain`, logs), `runs/template/` and `runs/highdim/`. See the repository
root `REPRODUCE.md` for how to re-run everything.

### 8.3 Consolidated RESULT lines

Group machinery:

```
RESULT ok=1 m24_order=244823040 psl_order=6072 out=.../data/group
RESULT ok=1 m24_gens=3 xi_bijective=1 xi_sign_patterns_ok=32 pr_elements=1000 ip_fail=0 bij_fail=0 applied=200 image_fail=0 set=data/S496.txt size=496 overlap_mean=1.300 ms=10145.7
```

Families — extraction, verification of both family sets, and every search run:

```
$ ctest --test-dir build/t42 -R family
1/1 Test #17: test_family ......................   Passed   16.79 sec
RESULT ok=1 dims=31,26,28,27,29,30,25 counts=238350,198550,204520,200044,209496,220440,197056 sets=42,2,8,5,14,24,1 distinct=52 time_s=141.0   # extract_packingstar_families.py --dry-run
RESULT ok=1 dir=data/families/dim26 dim=26 sets=2 total=992 count=198550 seconds=0.0
RESULT ok=1 dir=data/families/dim27 dim=27 sets=5 total=2480 count=200044 seconds=0.1
RESULT ok=1 dir=data/families/dim28 dim=28 sets=8 total=3968 count=204520 seconds=0.1
RESULT ok=1 dir=data/families/dim29 dim=29 sets=14 total=6944 count=209496 seconds=0.1
RESULT ok=1 dir=data/families/dim30 dim=30 sets=24 total=11904 count=220440 seconds=0.1
RESULT ok=1 dir=data/families/dim31 dim=31 sets=42 total=20832 count=238350 seconds=0.3
RESULT ok=1 dir=data/families/ours/dim26 dim=26 sets=2 total=992 count=198550 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim27 dim=27 sets=5 total=2480 count=200044 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim28 dim=28 sets=8 total=3968 count=204520 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim29 dim=29 sets=14 total=6944 count=209496 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim30 dim=30 sets=24 total=11904 count=220440 seconds=0.1
RESULT ok=1 dir=data/families/ours/dim31 dim=31 sets=42 total=20832 count=238350 seconds=0.3
RESULT ok=1 k=42 found=42 pool=0 method=chain seed=1 dim=31 total=20832 target_met=1 candidates=2590263 elements=1024 chain_length=42 order=42 timed_out=0 exhausted=0 seconds=40.9
RESULT ok=1 k=42 found=42 pool=0 method=chain seed=2 dim=31 total=20832 target_met=1 candidates=15158058 elements=1024 chain_length=42 order=42 timed_out=0 exhausted=0 seconds=677.3
RESULT ok=1 k=2 found=2 pool=0 method=chain seed=1 dim=26 total=992 target_met=1 candidates=1 elements=256 chain_length=3 order=33 timed_out=0 exhausted=0 seconds=14.2
RESULT ok=1 k=5 found=5 pool=0 method=chain seed=1 dim=27 total=2480 target_met=1 candidates=20 elements=256 chain_length=5 order=18 timed_out=0 exhausted=0 seconds=8.6
RESULT ok=1 k=8 found=8 pool=0 method=chain seed=1 dim=28 total=3968 target_met=1 candidates=19 elements=256 chain_length=10 order=33 timed_out=0 exhausted=0 seconds=17.4
RESULT ok=1 k=14 found=14 pool=0 method=chain seed=1 dim=29 total=6944 target_met=1 candidates=41 elements=256 chain_length=19 order=39 timed_out=0 exhausted=0 seconds=16.8
RESULT ok=1 k=24 found=24 pool=0 method=chain seed=1 dim=30 total=11904 target_met=1 candidates=62985 elements=256 chain_length=30 order=30 timed_out=0 exhausted=0 seconds=9.1
RESULT ok=1 k=42 found=39 pool=262144000 method=sweep seed=1 dim=31 ... (cut off at 467 s; runs/families/dim31 checkpoint, verify: ok)
RESULT ok=1 k=42 found=21 pool=20000 method=ls seed=1 dim=31 total=10416 target_met=0 density=0.5438 log10_expected=-98.3 pool_needed=4.38e+06 nodes=0 seconds=456.9
RESULT ok=1 dim=31 d=7 sets=42 count=238350 count_exact=238350 count_float=238350 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=126 K=126 exact_s=89.2 float_s=214.9 s=331.0   # verify_dimN.py on the chain family
```

Verifier, record families (`--strict`):

```
RESULT ok=1 dim=26 d=2 sets=2 count=198550 count_exact=198550 count_float=198550 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=6 K=6 exact_s=5.7 float_s=21.2 s=27.0
RESULT ok=1 dim=27 d=3 sets=5 count=200044 count_exact=200044 count_float=200044 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=21.5 float_s=45.4 s=67.2
RESULT ok=1 dim=28 d=4 sets=8 count=204520 count_exact=204520 count_float=204520 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=24 K=24 exact_s=27.9 float_s=55.0 s=83.3
RESULT ok=1 dim=29 d=5 sets=14 count=209496 count_exact=209496 count_float=209496 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=40 K=40 exact_s=103.6 float_s=93.3 s=197.5
RESULT ok=1 dim=30 d=6 sets=24 count=220440 count_exact=220440 count_float=220440 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=72 K=72 exact_s=26.1 float_s=62.8 s=90.6
RESULT ok=1 dim=31 d=7 sets=42 count=238350 count_exact=238350 count_float=238350 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=126 K=126 exact_s=35.7 float_s=68.7 s=110.9
```

Verifier, chain families (`--strict`):

```
RESULT ok=1 dim=26 d=2 sets=2 count=198550 count_exact=198550 count_float=198550 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=6 K=6 exact_s=62.6 float_s=102.1 s=165.4
RESULT ok=1 dim=27 d=3 sets=5 count=200044 count_exact=200044 count_float=200044 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=47.0 float_s=108.2 s=156.1
RESULT ok=1 dim=28 d=4 sets=8 count=204520 count_exact=204520 count_float=204520 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=24 K=24 exact_s=49.6 float_s=50.4 s=100.8
RESULT ok=1 dim=29 d=5 sets=14 count=209496 count_exact=209496 count_float=209496 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=40 K=40 exact_s=39.5 float_s=62.1 s=102.4
RESULT ok=1 dim=30 d=6 sets=24 count=220440 count_exact=220440 count_float=220440 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=72 K=72 exact_s=32.0 float_s=60.8 s=96.4
RESULT ok=1 dim=31 d=7 sets=42 count=238350 count_exact=238350 count_float=238350 max_offdiag=2.000000000000 max_class=lift-lift-cross-set extra=126 K=126 exact_s=41.3 float_s=75.8 s=129.8
```

The dimension-27 improvement (`--strict`, two independent families):

```
RESULT ok=1 dim=27 d=3 sets=4 count=200540 count_exact=200540 count_float=200540 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=44.7 float_s=102.3 s=147.3   # runs/template/dim27_record
RESULT ok=1 dim=27 d=3 sets=4 count=200540 count_exact=200540 count_float=200540 max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12 exact_s=47.9 float_s=101.7 s=150.0   # runs/template/dim27_ours
RESULT ok=1 dim=27 sets=4 weight=8 count=200540                                                                                                                                          # docs/note/data/verify.py
```

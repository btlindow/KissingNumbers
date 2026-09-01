# Upper bounds

This document collects everything the project knows about **upper** bounds on `|S|`, where `S` is a
subset of the 196560 minimal vectors of the Leech lattice containing no pair at 60°. The
construction side — the record `|S| = 496` and the search that produced it — is in
`03-search-for-497.md`; the lattice, its canonical vector order and the adjacency data used
throughout are in `01-lattice-and-verification.md`.

## Contents

1. [Setting and notation](#1-setting-and-notation)
2. [The ladder](#2-the-ladder)
3. [The lattice-free Delsarte LP](#3-the-lattice-free-delsarte-lp)
4. [The association scheme of the 196560 minimal vectors](#4-the-association-scheme-of-the-196560-minimal-vectors)
5. [Why every two-point method stops at 9360/11](#5-why-every-two-point-method-stops-at-936011)
6. [The three-point (Terwilliger/Schrijver) SDP: |S| ≤ 837](#6-the-three-point-terwilligerschrijver-sdp-s--837)
7. [The clique number of the conflict graph is 24, and clique cuts are inert](#7-the-clique-number-of-the-conflict-graph-is-24-and-clique-cuts-are-inert)
8. [Slack structure at the 837 optimum](#8-slack-structure-at-the-837-optimum)
9. [Scoping a four-point (Lasserre level-3) bound](#9-scoping-a-four-point-lasserre-level-3-bound)
10. [Corrections](#10-corrections)

---

## 1. Setting and notation

`C` is the set of `N = 196560` minimal vectors of the Leech lattice, in the integer scaling where
every vector has norm 32 (i.e. `√8` scaling of the standard normalisation), stored in canonical
order in `data/leech_min.i8`. Pairwise inner products take exactly the seven values

```
⟨x,y⟩ ∈ {−32, −16, −8, 0, 8, 16, 32},      cos∠(x,y) = ⟨x,y⟩/32 ∈ {−1, −½, −¼, 0, ¼, ½, 1}.
```

The **conflict graph** `G` has vertex set `C` and an edge whenever `⟨x,y⟩ = 16`, i.e. whenever the
angle is exactly 60°. `S ⊆ C` is admissible iff it is an independent set of `G`; the quantity to be
bounded is `α(G)`. The classes are indexed `c = 0..6` in the order of the seven inner products
above (the order of `kiss::ip_class`); class 6 is the identity relation, class 0 the antipode, class
5 the conflict relation, of valency `k = 4600`.

> **Class-index convention (a fixed bug).** The class index is **not** `(dot+32)/8`. The dots are
> not equally spaced — `(dot+32)/8 ∈ {0, 2, 3, 4, 5, 6, 8}` — and an early version of the GPU kernel
> and its CPU reference used that formula literally: it maps dot 32 to 8 and dot −16 to 2, so the
> kernel threw "inner product out of range" and the reference indexed `M[i*7+j]` out of bounds. The
> canonical mapping, used everywhere now, is
> ```
> t = (d + 32) >> 3;   c = t − [t > 1] − [t > 7].
> ```

---

## 2. The ladder

| Bound | Value | Status |
|---|---|---|
| Best construction (lower bound) | **496** | explicit, verified — see `03-search-for-497.md` |
| Delsarte LP, lattice-free, restricted inner-product set | 9360/11 = 850.909… → **850** | exact rational certificate |
| Association-scheme LP (Delsarte on the 7-class scheme) | 9360/11 → **850** | exact rational certificate |
| Schrijver θ′ of `G` (symmetry-reduced) | 9360/11 → **850** | exact |
| Lovász θ (symmetry-reduced) | 9360/11 → **850** | exact |
| Hoffman ratio bound `N(−λ_min)/(k − λ_min)` | 9360/11 → **850** | exact |
| **Three-point Terwilliger SDP** | 837.535866161… → **837** | exact rational certificate, `data/scheme/sdp3_certificate.json` |
| … + triangle and ω-clique cuts | **837**, unchanged | exact certificate, `data/scheme/sdp3_cliques_certificate.json` |
| Four-point (Lasserre level 3) | not computed | scoped only; see §9 |

**Current state of the ladder: `496 ≤ |S| ≤ 837`,** a gap of 341. Every entry above 496 in that
table is a proved upper bound: the two-point entries come with exact rational Delsarte certificates,
and the three-point entry with an exactly rational, exactly-PSD dual certificate whose final
inequality is evaluated entirely in `fractions.Fraction`. Nothing in the ladder rests on a
floating-point solver output.

Two sanity anchors recur at every level and are reproduced by all three methods:

* forbidding **nothing** returns `N = 196560` (or a value `≥ N` for the SDP, which cannot exclude
  `S = C`);
* forbidding cos = ½ **and** cos = ¼ (i.e. requiring `cos ≤ 0`) returns **48 = 2·24**, Rankin's
  orthoplex bound.

The 837 does not close the gap. What it does establish is that a three-point method genuinely sees
something a two-point method cannot — but only 13 of the 354 that separated 850 from 496.
Consequently the template is **not** proved closed in dimension 25 (see
`05-lifting-template-and-families.md`), and the search of `03-search-for-497.md` runs with the upper
bound as its stopping ceiling — first 850, then 837.

---

## 3. The lattice-free Delsarte LP

This is the weakest and cheapest bound in the ladder, and it is completely independent of the Leech
lattice: it bounds *any* spherical code in `R^24` whose pairwise inner products lie in a prescribed
finite set `A ⊂ [−1, 1)`. Python only, no GPU.

**Theorem used (Delsarte–Goethals–Seidel 1977).** Let `X ⊂ S^{n−1}` be finite with all pairwise
inner products in `A`. If `f = Σ_k f_k G_k` with `f_0 > 0`, `f_k ≥ 0`, and `f ≤ 0` on `A`, then
`|X| ≤ f(1)/f_0`. (`Σ_{x,y} f(⟨x,y⟩) ≤ |X| f(1)` from the sign condition, and `≥ f_0 |X|²` from
positive-definiteness of the `G_k`.)

### 3.1 Results

| Inner-product set `A` | exact LP value | floor | note |
|---|---|---|---|
| **`A = {−1, −1/2, −1/4, 0, 1/4}`** (all Leech cosines except cos 60° = ½) | **9360/11 = 850.909…** | **850** | the bound on `\|S\|`; > 496 |
| `A' = {−1, −1/2, −1/4, 0, 1/4, 1/2}` (all Leech inner products) | 196560 | 196560 | recovers Odlyzko–Sloane/Levenshtein exactly |
| `A'' = {−1, −1/2, −1/4, 0}` (cos ≤ 0) | 48 | 48 | `= 2n`, the Rankin orthoplex bound |
| grid `{−1, −1+1/2000, …, 1/2}` (classical, 3001 points), `d = 20/30/40` | 196560.0000000008 (float) | — | sanity check, §3.5 |

**Comparison with 496.** The pure Delsarte bound with the restricted set `A` is 850, leaving a gap
of 354 above the record 496. This is expected: the LP ignores the lattice entirely (a five-distance
set with those inner products need not embed in the Leech minimal vectors). The bound is attained by
a degree-4 polynomial and is already optimal at degree 10 — raising the degree to 60 changes
nothing. The LP is degenerate: at `d = 60` HiGHS returns a different optimal vertex, support
`{1,2,3,4,9}`, with the same value, and both certification routes still land on 9360/11.

```
$ .venv/bin/python python/bounds/lp_delsarte.py
== restricted: A = {-1, -1/2, -1/4, 0, 1/4}, n = 24
   d       HiGHS float     exact (limit_denominator)                  exact vertex  floor
  10        850.909091                    850.909091                    850.909091  850
  20        850.909091                    850.909091                    850.909091  850
  30        850.909091                    850.909091                    850.909091  850
  40        850.909091                    850.909091                    850.909091  850
  60        850.909091                    850.909091                    850.909091  850
```

Exact value at every degree: `9360/11`. The two independent certification routes (§3.3) agree
exactly; `limit_denominator` hits the true rationals because the denominators are tiny.

### 3.2 Gegenbauer machinery (`python/bounds/gegenbauer.py`)

Convention: `G_k = C_k^{(α)}/C_k^{(α)}(1)` with `α = (n−2)/2 = 11` for `n = 24`, so `G_k(1) = 1` and
`G_k(−1) = (−1)^k`. The recurrence is run exactly in `Fraction`:

```
G_0 = 1,  G_1 = t,  (k + n − 2) G_{k+1}(t) = (2k + n − 2) t G_k(t) − k G_{k−1}(t).
```

(For `n = 3` this is Legendre's recurrence; tested.) `gegenbauer_coeffs(k, n)` gives monomial
coefficients; `gegenbauer_eval(k, t, n)` / `gegenbauer_values(d, t, n)` evaluate exactly at rational
`t` by running the recurrence at the point; `to_gegenbauer_basis` / `from_gegenbauer_basis` change
basis exactly (back-substitution on leading coefficients). First few for `n = 24`:

```
G_2 = (24 t² − 1)/23,   G_3 = (26 t³ − 3 t)/23,   G_4 = (728 t⁴ − 156 t² + 3)/575.
```

### 3.3 The LP and its two exact certification routes (`python/bounds/lp_delsarte.py`)

With `f_0 = 1` and variables `f_1..f_d ≥ 0`, minimise `1 + Σ f_k` subject to
`Σ_{k≥1} f_k G_k(t) ≤ −1` for each `t ∈ A`, via `scipy.optimize.linprog(method="highs")`; the
coefficient matrix `G_k(t)` is computed exactly and then cast to float. **The float solution is
never the result.** Two independent routes take it to an exact certificate:

1. `rationalise` — for each `D ∈ {10³, 10⁴, 10⁶, 10⁸, 10¹⁰, 10¹², 10¹⁵, 10¹⁸}`, set
   `f_k := Fraction(x).limit_denominator(D)` (with `|x| < 1e−12` snapped to 0), then **recompute
   `f_0` exactly** as `f_0 := −max_{t∈A} Σ_{k≥1} f_k G_k(t)`. This "tiny exact slack" perturbation
   makes `f(t) ≤ 0` on `A` hold by construction and needs only `f_0 > 0`. The exact bound is
   `f(1)/f_0 = (f_0 + Σ f_k)/f_0`; the smallest over `D` is returned.
2. `exact_vertex` — the optimum is a vertex, so the support `K` of `f` and the active constraint set
   `T` satisfy `|K| ≤ |T|`; when `|K| = |T|`, the square system `Σ_{k∈K} f_k G_k(t) = −1` (`t ∈ T`)
   is solved by `Fraction` Gauss–Jordan. This yields the exact LP optimum, not merely a rounding of
   it, whenever the returned basis is clean — which it was in every run here.

Both routes end in `verify_certificate(f, A, n)`, which independently re-checks every hypothesis of
the DGS theorem in exact arithmetic (`f_0 > 0`, `f_k ≥ 0`, `A ⊂ [−1,1)`, `f(t) ≤ 0` for all `t ∈ A`)
and returns `f(1)/f_0`. The reported bound is the floor of that.

### 3.4 The exact certificate (best degree 10; the polynomial has degree 4)

```
n = 24,  A = {−1, −1/2, −1/4, 0, 1/4}
f_0 = 1,  f_1 = 24,  f_2 = 5083/77,  f_3 = 4416/11,  f_4 = 27600/77,  f_k = 0 for k ≥ 5
f(1)/f_0 = 1 + 24 + 5083/77 + 4416/11 + 27600/77 = 9360/11 = 850.909…   ⇒   |S| ≤ 850
```

In monomial form `f(t) = (4992/11)(t⁴ + t³) − (312/11)(t² + t) = (4992/11)·(t+1)·t·(t+1/4)·(t−1/4)`,
i.e. the natural polynomial vanishing on `A \ {−1/2}` and negative at `−1/2` (`f(−1/2) = −234/11`).
Hand re-check, with the `G_k` values on `A`:

```
t     : −1    −1/2     −1/4     0       1/4
G_1   : −1    −1/2     −1/4     0       1/4
G_2   :  1    5/23     1/46    −1/23    1/46
G_3   : −1    −7/92    11/736   0      −11/736
G_4   :  1    19/1150 −5/736    3/575  −5/736
```

`f(t) = 1 + 24 G_1 + (5083/77) G_2 + (4416/11) G_3 + (27600/77) G_4` evaluates to
`0, −234/11, 0, 0, 0` respectively; all `f_k > 0` and `f_0 = 1 > 0`. The test
`test_degree4_polynomial_gives_restricted_bound_exactly` derives the same certificate from the other
direction: it expands `(t+1)t(t+1/4)(t−1/4)` in the Gegenbauer basis exactly and checks that the
coefficients are nonnegative with `f(1)/f_0 = 9360/11`.

### 3.5 The two auxiliary LPs and the grid sanity check

**`A'` = all Leech inner products.** 196560 at every degree, with the exact vertex certificate at
`d = 10`:

```
f_0 = 1, f_1 = 24, f_3 = 2576, f_5 = 95680, f_6 = 1733464/513, f_8 = 779723/27, f_10 = 3763214/57
f(t) = 0 at every t ∈ A';  f(1)/f_0 = 196560.
```

This is exactly the Levenshtein/Odlyzko–Sloane polynomial
`(t+1)(t+1/2)²(t+1/4)²t²(t−1/4)²(t−1/2)`; the test `test_levenshtein_polynomial_gives_196560_exactly`
expands it exactly and gets 196560. The value *must* be exactly 196560: `A' ⊂ [−1, 1/2]` so
`bound(A') ≤` the classical LP `= 196560`, while the Leech minimal vectors are a feasible code so
`bound(A') ≥ 196560`.

**`A'' = {−1, −1/2, −1/4, 0}` (cos ≤ 0).** 48 = 2n at every degree; certificate `f_0 = 1`,
`f_1 = 928/77`, `f_2 = 23`, `f_5 = 920/77`, with `f(t) = 0` at `−1, −1/2, 0` and `−267/176` at
`−1/4`. This matches Rankin's theorem (at most `2n` unit vectors with pairwise nonpositive inner
products), so the "more orthogonal `S`" variant of the problem is capped at `|S| ≤ 48` by the LP
alone — a strong hint that that variant cannot beat 496 unless the lift's scale advantage is
enormous.

**Sanity checks.**

* `A = {−1}`: LP value 2 (float and exact) at `d = 1, 5, 10`; certificate `f = 1 + t`.
* **Classical grid.** `A_grid = {−1, −1+1/2000, …, 1/2}` (3001 points, contains `A'` and lies in
  `[−1, 1/2]`). Sandwich: `196560 = bound(A') ≤ bound(A_grid) ≤ bound([−1,1/2]) = 196560`, so for
  `d ≥ 10` the grid LP must return **exactly** 196560 — there is no "grid slack" in either direction
  for this particular problem, because the grid happens to contain all six zeros of the optimal
  polynomial. In general the direction is one-way: a grid is a *subset* of the interval, so it
  weakens the constraints and can only *lower* the bound relative to the continuous LP; here the
  lower bound from `A'` pins it. Observed: 196560.0000000008 at `d = 20, 30, 40` (the test asserts
  relative error `< 1e−6`).

```
$ .venv/bin/python python/bounds/lp_delsarte.py --grid --degrees 10
...
== classical Delsarte kissing LP on grid, |A| = 3001
  20     196560.000000  status=0
  30     196560.000000  status=0
  40     196560.000000  status=0

RESULT bound_restricted=850 bound_all_leech=196560 bound_nonpos=48 grid_classical=196560.00000000081
```

### 3.6 Files, tests, timing

| File | Purpose |
|---|---|
| `python/bounds/__init__.py` | package init, re-exports |
| `python/bounds/gegenbauer.py` | exact rational Gegenbauer polynomials `G_k^{(n)}` normalised `G_k(1) = 1`, exact evaluation, exact monomial ↔ Gegenbauer basis change |
| `python/bounds/lp_delsarte.py` | HiGHS LP, rationalisation, exact verification, exact vertex re-solve, CLI |
| `python/tests/test_lp_delsarte.py` | 43 pytest tests |

Tests (43): Gegenbauer coefficients equal sympy `gegenbauer(k, 11, t)/gegenbauer(k, 11, 1)` for
`k ≤ 12` (and for `n = 3, 4, 8`, `k ≤ 8`; `n = 3` equals `legendre`); table/coefficient consistency;
exact point evaluation agrees with Horner on the coefficient lists for `k ≤ 40`; `G_k(±1)`; numerical
orthogonality with weight `(1−t²)^{10.5}` via `scipy.integrate.quad` for 7 pairs — using the
*relative* criterion `⟨G_j,G_k⟩/(‖G_j‖‖G_k‖) < 1e−8`, because the normalised `G_k` have `L²` norms
as small as `2e−8` by `k = 10` and an absolute threshold would be meaningless; basis-change round
trip. Certificates: Levenshtein polynomial → 196560 exactly; degree-4 polynomial → 9360/11 exactly
and equal to the LP's certificate. LP: antipodal → 2; restricted bound at `d ∈ {10,20,30,40,60}`
(float, `limit_denominator` certificate re-verified, exact vertex = 9360/11, floor 850); headline
850 > 496; `A'` → 196560 exactly at `d = 10, 20, 40`; `A''` → 48; grid → 196560 within `1e−6`;
`verify_certificate` rejects bad inputs; CLI `RESULT` line.

```
$ .venv/bin/python -m pytest python/tests/test_lp_delsarte.py -q
43 passed in 2.40s
```

Timing: the CLI without `--grid` (3 point sets × 5 degrees, exact Gegenbauer values to degree 60,
HiGHS, rationalisation) takes 0.55 s; with `--grid` (3001-point exact Gegenbauer matrix to degree 40)
2.4 s; the test suite 2.4 s. Nothing here needs the GPU.

---

## 4. The association scheme of the 196560 minimal vectors

The next bound uses the lattice: the seven inner-product relations on `C` form a 7-class association
scheme, and the Delsarte LP of *that* scheme is at least as strong as §3. Hardware for this section:
RTX 3070 Laptop (sm_86), CUDA 12.8; Python 3 / numpy 2.2 / scipy 1.15 (HiGHS) / sympy 1.14.

### 4.1 Results

| Quantity | exact value | floor |
|---|---|---|
| **Scheme LP bound on `\|S\|`** (`S ⊂ C` independent in the conflict graph) | **9360/11 = 850.909…** | **850** |
| … with `S` antipodal (`a₋₃₂ = 1`) | 9360/11 | 850 |
| Schrijver θ′ of the conflict graph (symmetry-reduced) | 9360/11 | 850 |
| Lovász θ (symmetry-reduced, sign constraints dropped) | 9360/11 | 850 |
| Hoffman ratio bound `N(−λ_min)/(k − λ_min) = 196560·20/4620` | 9360/11 | 850 |
| Lattice-free Delsarte LP of §3 (`A = {−1,−½,−¼,0,¼}`) | 9360/11 | 850 |
| "cos ≤ 0" variant (classes 16 and 8 forbidden) | 48 | 48 |
| nothing forbidden | 196560 | (sanity: `= N`) |
| only `±32` allowed | 2 | (sanity) |

**The association scheme adds nothing at the two-point level.** The scheme LP, θ′, θ, the Hoffman
bound and the lattice-free polynomial of §3 all give `9360/11`. The reason is structural and is
proved in §5.

**The scheme property holds** — for `x = C[0]`, all 196560 `y` give one matrix per class, and four
more random `x` agree exactly — and all six intersection-number identities hold. Moreover
**vertex-transitivity is verified computationally, not cited**: the orbit of `C[0]` under the five
explicit Co₀ generators (`data/group`; the M24 generators α, γ, δ, one octad sign flip, and Conway's
ξ — see `05-lifting-template-and-families.md`) is all 196560 vectors, and each generator is an
orthogonal map of `C` onto `C` (`ξᵀξ = 4·I` checked exactly; monomials by construction;
`index_permutation` throws if any image leaves `C`). Hence the intersection numbers computed at
`x = C[0]` are the intersection numbers at every `x`, and **no assumption from the literature remains
in the chain "GPU numbers → scheme → P, Q → LP → 850"**. The only remaining literature input is the
*interpretation* of the reduced programs as Lovász's θ (the orbital identification), which does not
affect the validity of any bound.

### 4.2 Definitions and conventions

`p^k_{ij} = #{z : (x,z) ∈ R_i, (z,y) ∈ R_j}` for `(x,y) ∈ R_k`. The intersection matrices are
`(L_i)_{jk} = p^k_{ij}`. The first eigenmatrix `P` has rows indexed by eigenspaces `E_r` and
`P_{ri} =` the eigenvalue of `A_i` on `E_r` (row 0 trivial, `P_{0i} = v_i`; column 6 all ones); rows
are ordered by decreasing conflict eigenvalue. Multiplicities `m_r = N / Σ_i P_{ri}²/v_i`, and
`Q = N·P⁻¹`, equivalently `Q_{ir} = m_r P_{ri}/v_i`. For `S ⊂ C` with inner distribution
`a_i = #{(x,y) ∈ S² : ⟨x,y⟩ ∈ class i}/|S|`: `Σ a_i = |S|`, `a_6 = 1`, `a ≥ 0`, and
`(aQ)_r = (N/|S|)·1_Sᵀ E_r 1_S ≥ 0`, where `E_r = N⁻¹ Σ_i Q_{ir} A_i` is a projector.

### 4.3 The GPU kernel (`cuda/pair_class_histogram.cu`)

Derived from the tightness kernel of `01-lattice-and-verification.md`. For fixed `x` and
`y ∈ [y0, y0+ny)` it computes `M_y[i][j] = #{z : class(x,z)=i, class(z,y)=j}` as `uint32[ny][49]`:

1. `class_kernel`: `cls[z] = class(x,z)` via 6 `dp4a`.
2. Host counting sort of `z` by class → `perm`, `offs[8]`.
3. `gather_sorted_kernel`: packed words in class order (SoA, coalesced).
4. `hist_kernel<YB=8>`: one 256-thread block per 8 consecutive `y` (48 `y`-words held in registers);
   for each class `g` of `x` the threads stride over that class's contiguous `z`-range, compute
   `dot(z,y)` with 6 `dp4a` for each of the 8 `y`, and add `1 << 9·class(z,y)` into a packed
   `uint64` (seven 9-bit fields). A thread sees at most `⌈93150/256⌉ = 364 < 512` vectors `z` of one
   class, so no field can overflow — this is enforced on the host. Per class the fields are
   `__reduce_add_sync`-reduced and atomically added into shared `s_hist[8][49]`, written to
   `M[y][49]` at the end. Scratch is about 5.3 MB, freed inside.

Measured: a full `N×N` pass (`3.86×10¹⁰` dots) takes **171–214 ms ≈ 1.8–2.3×10¹¹ pairs/s** — half
the rate of the plain tightness kernel, because the histogram update costs about 5 integer ops per
pair on top of the 6 `IDP4A`, and the packed-field reduction runs `7×8×7` times per block. The
complete `scheme_numbers` run, including load, orbit BFS and 4 extra `x`'s, takes 3.1 s; an early
estimate of "~10 s" for the GPU pass alone was pessimistic by a factor of ~50.

### 4.4 The scheme numbers

```
$ build/t22/tools/scheme_numbers --extra-x 4
x = 0 : C[x] = -4 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
valencies v_c (c = 0..6 <-> dot -32,-16,-8,0,8,16,32): 1 4600 47104 93150 47104 4600 1
GPU pass over all 196560 y: 213.7 ms (1.808e+11 pairs/s)
distinct M_y per class k = class(x,y): 1 1 1 1 1 1 1  -> scheme property HOLDS
p^0 (dot -32, v=    1, example y=196559): [0 0 0 0 0 0 1][0 0 0 0 0 4600 0][0 0 0 0 47104 0 0][0 0 0 93150 0 0 0][0 0 47104 0 0 0 0][0 4600 0 0 0 0 0][1 0 0 0 0 0 0]
p^1 (dot -16, v= 4600, example y=131126): [0 0 0 0 0 1 0][0 1 0 891 2816 891 1][0 0 2816 20736 20736 2816 0][0 891 20736 49896 20736 891 0][0 2816 20736 20736 2816 0 0][1 891 2816 891 0 1 0][0 1 0 0 0 0 0]
p^2 (dot  -8, v=47104, example y=64366): [0 0 0 0 1 0 0][0 0 275 2025 2025 275 0][0 275 7128 22275 15400 2025 1][0 2025 22275 44550 22275 2025 0][1 2025 15400 22275 7128 275 0][0 275 2025 2025 275 0 0][0 0 1 0 0 0 0]
p^3 (dot   0, v=93150, example y=45): [0 0 0 1 0 0 0][0 44 1024 2464 1024 44 0][0 1024 11264 22528 11264 1024 0][1 2464 22528 43164 22528 2464 1][0 1024 11264 22528 11264 1024 0][0 44 1024 2464 1024 44 0][0 0 0 1 0 0 0]
p^4 (dot   8, v=47104, example y=1070): [0 0 1 0 0 0 0][0 275 2025 2025 275 0 0][1 2025 15400 22275 7128 275 0][0 2025 22275 44550 22275 2025 0][0 275 7128 22275 15400 2025 1][0 0 275 2025 2025 275 0][0 0 0 0 1 0 0]
p^5 (dot  16, v= 4600, example y=1): [0 1 0 0 0 0 0][1 891 2816 891 0 1 0][0 2816 20736 20736 2816 0 0][0 891 20736 49896 20736 891 0][0 0 2816 20736 20736 2816 0][0 1 0 891 2816 891 1][0 0 0 0 0 1 0]
p^6 (dot  32, v=    1, example y=0): [1 0 0 0 0 0 0][0 4600 0 0 0 0 0][0 0 47104 0 0 0 0][0 0 0 93150 0 0 0][0 0 0 0 47104 0 0][0 0 0 0 0 4600 0][0 0 0 0 0 0 1]
identities: (a) row sums = v_i: 0 bad, (b) v_k p^k_ij = v_i p^i_kj: 0 bad, (c) p^k_ij = p^k_ji: 0 bad, (d) p^6_ij = delta v_i: 0 bad, (e) column sums = v_j: 0 bad, (f) p^k_i6 = delta_ik: 0 bad -> OK
orbit of x=0 under 5 Co_0 generators (data/group): 196560 of 196560 vectors -> transitive: YES; xi orthogonal (exact): yes (279 ms)
extra x=28348: scheme holds, p identical to x=0: yes (198.3 ms)
extra x=9913: scheme holds, p identical to x=0: yes (201.5 ms)
extra x=47627: scheme holds, p identical to x=0: yes (202.4 ms)
extra x=119185: scheme holds, p identical to x=0: yes (199.2 ms)
wrote data/scheme/intersection_numbers.json
RESULT ok=1 scheme=1 identities=1 x=0 y_checked=196560 extra_x=4 extra_x_agree=4 orbit_checked=1 orbit_size=196560 transitive=1 gpu_ms=213.7 total_ms=3098 out=data/scheme/intersection_numbers.json
```

Rows of each `p^k` are `i`, columns `j`. Readable facts: the conflict graph has `λ = p⁵₅₅ = 891`
common neighbours per edge; two non-adjacent vertices have 44 (`⟨x,y⟩ = 0`), 275 (`⟨x,y⟩ = 8`), 0
(`⟨x,y⟩ = −8`), 1 (`⟨x,y⟩ = −16`) and 0 (antipodes) common neighbours — so `G` is **not** strongly
regular. It is a 6-class scheme in which the classes `±c` are swapped by the antipode
(`p⁰_{ij} = v_i·[j = 6−i]`).

### 4.5 Eigenmatrices (`python/bounds/scheme.py`, exact)

Common eigenvectors of the seven commuting `L_i` are found from a generic integer combination (which
has simple spectrum); each vector is checked to be an eigenvector of every `L_i`, and all eigenvalues
are integers.

```
first eigenmatrix P (rows: eigenspaces E_r with multiplicity; columns: classes by inner product):
                A(-32)      A(-16)       A(-8)        A(0)        A(8)       A(16)       A(32)
    E0 m=1           1        4600       47104       93150       47104        4600           1
   E1 m=24          -1       -2300      -11776           0       11776        2300           1
  E2 m=299           1        1000        1024       -4050        1024        1000           1
 E3 m=2576          -1        -350         704           0        -704         350           1
E4 m=17250           1          76        -320         486        -320          76           1
E5 m=95680          -1          10         -16           0          16         -10           1
E6 m=80730           1         -20          64         -90          64         -20           1

second eigenmatrix Q = N P^{-1} (rows: classes; columns: eigenspaces), Q_ir = m_r P_ri / v_i:
                    E0          E1          E2          E3          E4          E5          E6
    a(-32)           1         -24         299       -2576       17250      -95680       80730
    a(-16)           1         -12          65        -196         285         208        -351
     a(-8)           1          -6        13/2        77/2    -1875/16       -65/2     1755/16
      a(0)           1           0         -13           0          90           0         -78
      a(8)           1           6        13/2       -77/2    -1875/16        65/2     1755/16
     a(16)           1          12          65         196         285        -208        -351
     a(32)           1          24         299        2576       17250       95680       80730

multiplicities m = (1, 24, 299, 2576, 17250, 95680, 80730), sum = 196560 = N: True
P Q = N I: True
conflict graph (class 16, valency 4600) spectrum (eigenvalue^multiplicity): 4600^1 2300^24 1000^299 350^2576 76^17250 -10^95680 -20^80730
Hoffman bound N(-lmin)/(k-lmin) = 9360/11 = 850.909091
RESULT ok=1 scheme=1 N=196560 PQ_eq_NI=1 multiplicities=1,24,299,2576,17250,95680,80730 conflict_eigenvalues=4600,2300,1000,350,76,-10,-20 hoffman=9360/11
```

**Conflict-graph eigenvalues with multiplicities: `4600¹, 2300²⁴, 1000²⁹⁹, 350²⁵⁷⁶, 76¹⁷²⁵⁰,
(−10)⁹⁵⁶⁸⁰, (−20)⁸⁰⁷³⁰`.**

Two external consistency checks, both asserted in the tests:

* the multiplicities 24, 299, 2576, 17250, 95680, 80730 are degrees of irreducible representations of
  Co₀ (the permutation character on `C` decomposes into seven distinct irreducibles — consistent with
  Co₂ having rank 7 on `C`);
* `m_r = dim Harm_r(R²⁴)` with `Q_{ir} = m_r·G_r(cos_i)` (the Gegenbauer polynomials of §3) for
  `r = 1..5`, because the minimal vectors form a spherical 11-design. `E_6` is what is left over
  (`80730 = N − 1 − 24 − 299 − 2576 − 17250 − 95680`).

This second fact is exactly why the scheme LP **contains** the lattice-free LP of §3: the first six
Delsarte inequalities of the scheme are the Gegenbauer inequalities of degree 0–5 evaluated at the
five allowed cosines, and the only genuinely new constraint is `(aQ)₆ ≥ 0`.

### 4.6 The scheme LP (`python/bounds/lp_scheme.py`)

Maximise `Σ_i a_i` subject to `a₃₂ = 1`, `a₁₆ = 0`, `a_i ≥ 0`, `(aQ)_r ≥ 0` (`r = 0..6`). Two
solvers are run: HiGHS (float) and exact enumeration of all basic solutions in `Fraction` for the
primal (5 free variables, 12 constraints → 792 square systems, 20 feasible vertices) *and* for the
dual. The exact primal and dual optima must agree (strong duality), and `verify_certificate`
re-checks every hypothesis of the dual from scratch.

```
== main: S independent in the conflict graph: a_16 = 0
  fixed: a(16) = 0, a(32) = 1
  free classes: a(-32), a(-16), a(-8), a(0), a(8)
  HiGHS (float): 850.909090909
  exact optimum: 9360/11 = 850.909090909   floor 850   [20 feasible vertices, 1 optimal]
  a* (inner distribution attaining it): a(-32) = 1, a(-16) = 0, a(-8) = 2944/11, a(0) = 3450/11, a(8) = 2944/11, a(16) = 0, a(32) = 1
  (a* Q)_r = 9360/11, 0, 0, 0, 0, 0, 2152800/11   (all >= 0; zeros = active Delsarte inequalities)
  dual certificate beta_r (r = eigenspace E_r, multiplicity m_r):
      r    m_r        beta_r          beta_r * q_r
      0       1                0                0
      1      24          116/231           928/77
      2     299            17/77          5083/77
      3    2576           37/462          6808/33
      4   17250            8/385         27600/77
      5   95680            1/462        47840/231
      6   80730                0                0
  q_r = sum_(fixed i) Q_ir a_i = 1, 24, 299, 2576, 17250, 95680, 80730
  bound = sum_(fixed) a_i + sum_r beta_r q_r = 1 + 9349/11 = 9360/11
  check on every free class i:  F(i) := sum_r beta_r Q_ir  <= -1
      class a(-32):  F =         -1   ok
      class a(-16):  F =         -1   ok
      class a( -8):  F =         -1   ok
      class a(  0):  F =         -1   ok
      class a(  8):  F =         -1   ok
  rationalised HiGHS marginals also verify: beta = (0, 116/231, 17/77, 37/462, 8/385, 1/462, 0) -> identical to the exact dual
```

#### The certificate, for checking by hand

**Theorem used (Delsarte).** If `β₀..β₆ ≥ 0` and `Σ_r β_r Q_{ir} ≤ −1` for every allowed
non-identity class `i`, then every `S ⊂ C` whose pairs avoid the forbidden classes satisfies
`|S| ≤ 1 + Σ_r β_r m_r`.
*Proof.* `|S| − 1 = Σ_{i allowed, i≠6} a_i ≤ −Σ_{i≠6, allowed} a_i Σ_r β_r Q_{ir}
= −Σ_r β_r [(aQ)_r − a₆ Q_{6r}] = −Σ_r β_r (aQ)_r + Σ_r β_r m_r ≤ Σ_r β_r m_r`, using `a_i ≥ 0`,
`a₆ = 1`, `Q_{6r} = m_r`, `(aQ)_r ≥ 0`, `β_r ≥ 0`. ∎

**Certificate.** `β = (0, 116/231, 17/77, 37/462, 8/385, 1/462, 0)`, i.e. **`β_r = (λ_r + 20)/4620`**
for `r = 1..6`, where `λ_r ∈ {2300, 1000, 350, 76, −10, −20}` are the conflict-graph eigenvalues and
`4620 = k − λ_min = 4600 + 20`. With the `Q` table of §4.5 (columns `E₁..E₅` only, since
`β₀ = β₆ = 0`):

| class `i` | `Σ_r β_r Q_{ir}` |
|---|---|
| `a(−32)` | `−24·116/231 + 299·17/77 − 2576·37/462 + 17250·8/385 − 95680/462 = −928/77 + 5083/77 − 6808/33 + 27600/77 − 47840/231 =` **−1** |
| `a(−16)` | `−12·116/231 + 65·17/77 − 196·37/462 + 285·8/385 + 208/462 =` **−1** |
| `a(−8)` | `−6·116/231 + (13/2)(17/77) + (77/2)(37/462) − (1875/16)(8/385) − (65/2)/462 =` **−1** |
| `a(0)` | `0 − 13·17/77 + 0 + 90·8/385 + 0 = −221/77 + 144/77 =` **−1** |
| `a(8)` | `6·116/231 + (13/2)(17/77) − (77/2)(37/462) − (1875/16)(8/385) + (65/2)/462 =` **−1** |

All `≤ −1` (in fact `= −1`), all `β_r ≥ 0`, so
`|S| ≤ 1 + 24·116/231 + 299·17/77 + 2576·37/462 + 17250·8/385 + 95680/462
= 1 + 928/77 + 5083/77 + 6808/33 + 27600/77 + 47840/231 = 1 + 9349/11 = **9360/11** < 851`, hence
**`|S| ≤ 850`**.

The primal `a* = (1, 0, 2944/11, 3450/11, 2944/11, 0, 1)` attains `9360/11`
(`Σ = 2 + 5888/11 + 3450/11 = 9360/11`) with `(a*Q)₁..₅ = 0` and `(a*Q)₆ = 2152800/11 > 0`: the `E₆`
inequality — the only one not already present in the lattice-free LP of §3 — is *slack*. The optimal
primal vertex is unique; the dual is degenerate (five tight classes, five nonzero `β`, but several
optimal duals), and §3's degree-4 polynomial is another optimal dual:
`β = (0, 1, 17/77, 12/77, 8/385, 0, 0)`, i.e. `f_r/m_r` with
`f = 1 + 24G₁ + (5083/77)G₂ + (4416/11)G₃ + (27600/77)G₄`
(test `test_T21_delsarte_polynomial_is_also_a_certificate`). Note that `a*` is non-integral
(`a₋₈ = 2944/11`), so **nothing attains the bound** — it is not a near-miss configuration.

#### Variants

```
== antipodal: additionally S = -S: a_-32 = 1
  exact optimum: 9360/11   [4 feasible vertices, 1 optimal]   a* as above
  beta = (0, 1/2, 17/77, 6/77, 8/385, 0, 0),  q_r = (2, 0, 598, 0, 34500, 0, 161460),  bound = 2 + 9338/11 = 9360/11
== forbid_16_8: cos <= 0 only: a_16 = a_8 = 0
  exact optimum: 48   a* = (1, 0, 0, 46, 0, 0, 1)  (a cross: x, -x and 46 orthogonal vectors)
  beta = (0, 116/231, 1/13, 0, 0, 1/8008, 0),  bound = 1 + 47 = 48;  F = -1, -1, -443/176, -1 on a(-32), a(-16), a(-8), a(0)
== none: nothing forbidden
  exact optimum: 196560   a* = valencies,  beta = (0, 1, 1, 1, 1, 1, 1)
== only_pm32: only +-32 allowed
  exact optimum: 2   beta = (0,0,0,0,0,1/95680,0)  (HiGHS's dual (0,1/24,0,0,0,0,0) is another optimal dual)
RESULT bound_exact=9360/11 bound_floor=850 highs=850.909091 antipodal=9360/11 antipodal_floor=850 forbid_16_8=48 none=196560 only_pm32=2
```

The antipodal constraint is **free**: `a*` already has `a₋₃₂ = 1`, so restricting to antipodal `S`
costs nothing at this level. The "cos ≤ 0" variant gives exactly Rankin's `2n = 48` again — the
lattice does not help there either. The two sanity variants return `N` and 2 as they must.

### 4.7 θ′, θ and Hoffman (`python/bounds/theta_prime.py`)

```
conflict graph G: N = 196560  valency k = 4600
spectrum (eigenvalue^multiplicity): 4600^1 2300^24 1000^299 350^2576 76^17250 -10^95680 -20^80730
Hoffman ratio bound N(-lmin)/(k-lmin) = 196560*20/(4600+20) = 9360/11 = 850.909090909
== theta' (Schrijver) = scheme LP with a >= 0            exact optimum: 9360/11   [20 feasible vertices, 1 optimal]
== theta (Lovasz, symmetry-reduced) = scheme LP WITHOUT a_i >= 0 on the free classes
                                                          exact optimum: 9360/11   [10 feasible vertices, 1 optimal]
== Hoffman's certificate as a dual solution of the LP: beta_r = (lambda_r - lambda_min)/(k - lambda_min), r >= 1
   beta = 0, 116/231, 17/77, 37/462, 8/385, 1/462, 0
   proves |S| <= 9360/11   (equals the LP optimum: True; equals the exact dual found by enumeration: True)
Ordering: alpha(G) <= theta' <= theta_sym <= Hoffman  ->  9360/11 <= 9360/11 <= 9360/11
RESULT hoffman=9360/11 theta_sym=9360/11 theta_prime=9360/11 lp=9360/11 hoffman_cert_ok=1 all_equal=1
```

**What each one is.** `θ(G) = max{1ᵀB1 : B ⪰ 0, tr B = 1, B_xy = 0 on edges}`. Averaging a feasible
`B` over Co₀ (vertex-transitive, verified above; the seven classes are its orbitals given that
`Co₂ = Stab(x)` is transitive on each class — this is the one place where a group-theoretic fact
enters, and it is needed only for the *name* θ, not for the validity of the bound) puts `B` in the
Bose–Mesner algebra, `B = N⁻¹ Σ a_i A_i` with `a₆ = 1`, `a₅ = 0`, and `B ⪰ 0 ⇔ aQ ≥ 0`. So the
symmetry-reduced θ is the scheme LP *without* `a_i ≥ 0`, and Schrijver's θ′ (`B ≥ 0` entrywise) is
the scheme LP *with* `a_i ≥ 0` — the Delsarte bound. Whether or not the orbital identification is
right, both reduced programs are valid upper bounds on `α(G)`, because the inner distribution of an
independent set is feasible for them; and Schrijver (1979) gives `θ′ ≤` Delsarte-LP in general, with
equality for Schurian schemes like this one.

### 4.8 Files, tests, timing

| File | Purpose |
|---|---|
| `cuda/scheme_cuda.h` | `pair_class_histogram(L, x, y0, ny, d_M)` and `_all` |
| `cuda/pair_class_histogram.cu` | the kernel (§4.3) |
| `tools/scheme_numbers.cpp` | runs it for all `y`, checks the scheme property + identities, orbit/transitivity check, `--extra-x K`, writes the json |
| `data/scheme/intersection_numbers.json` | `p[k][i][j]`, valencies, class labels, verification record (committed) |
| `python/bounds/scheme.py` | loader with independent identity re-check, intersection matrices `L_i`, Bose–Mesner check, exact `P`, `Q`, multiplicities, Hoffman |
| `python/bounds/lp_scheme.py` | the LP: HiGHS float + exact vertex enumeration (primal and dual), exact certificate verification, 5 variants |
| `python/bounds/theta_prime.py` | θ′, symmetry-reduced θ, Hoffman, Hoffman's certificate as an LP dual |
| `python/tests/test_scheme.py` | 27 pytest cases |
| `tests/test_scheme_gpu.cu`, `tests/tasks/T2_2.cmake` | ctest `test_scheme_gpu` (label `gpu`, skip 77) |

**ctest `test_scheme_gpu`.** Checks the valencies of `x`; a full pass with all 196560 row sums `= v_i`
and column sums `= v_j`; 73 `y`'s (64 random plus `x`, `−x`, and the first `y` of each class) against
a CPU double loop with a scalar inner product and an explicit class lookup — 0 mismatches; a
sub-range entry point (`y0 = 123456`, `ny = 1001`) equal to the full pass; and the distinct-`M_y`
count per class.

```
$ ctest --test-dir build/t22 -R test_scheme_gpu
1/1 Test #8: test_scheme_gpu ..................   Passed    1.08 sec
$ build/t22/tests/test_scheme_gpu
valencies of x=0 : 1 4600 47104 93150 47104 4600 1
warm-up         : 1.775 ms (ny=8)
full pass       : 170.4 ms for 3.864e+10 pairs -> 2.267e+11 pairs/s
row/col sums    : bad rows 0, bad cols 0 (over all 196560 y)
cpu reference   : 73 y's checked (64 random + specials), 0 mismatches, 188 ms CPU
M_y for y = x   : [1 0 0 0 0 0 0][0 4600 0 0 0 0 0][0 0 47104 0 0 0 0][0 0 0 93150 0 0 0][0 0 0 0 47104 0 0][0 0 0 0 0 4600 0][0 0 0 0 0 0 1]
M_y for y = -x  : [0 0 0 0 0 0 1][0 0 0 0 0 4600 0][0 0 0 0 47104 0 0][0 0 0 93150 0 0 0][0 0 47104 0 0 0 0][0 4600 0 0 0 0 0][1 0 0 0 0 0 0]
sub-range       : y0=123456 ny=1001 1.751 ms, 0 rows differ from the full pass
distinct M_y per class(x,y): 1 1 1 1 1 1 1  (scheme property <=> all 1)
RESULT ok=1 x=0 full_ms=170.4 pairs_per_s=2.267e+11 samples=73 mismatches=0 scheme=1 distinct_total=7 cpu_ref_ms=188 total_ms=521 failures=0
```

`compute-sanitizer --tool memcheck --leak-check full build/t22/tests/test_scheme_gpu --samples 2`
gives `ERROR SUMMARY: 0 errors` and `LEAK SUMMARY: 0 bytes leaked in 0 allocations` (`RESULT ok=1`,
11 `y`'s, 0 mismatches; 15.4 s under the sanitizer).

**pytest `python/tests/test_scheme.py` (27 cases).** The json verification record (all `y`, orbit
`= N`, ξ orthogonal); valencies; spot intersection numbers (`λ = 891`; `μ = 44/275/0/1/0` by class;
the antipode permutation); the identity checker rejecting a perturbed `p`; the Bose–Mesner algebra
and commutativity; `P` equal to the table above; `m`; `P·Q = Q·P = N·I`; `Q_{ir} = m_r P_{ri}/v_i`;
the orthogonality relations `Σ_i P_{ri}P_{si}/v_i = N δ_{rs}/m_r`; the conflict spectrum with
`tr A₅ = 0` and `tr A₅² = N·k`; `m_r = dim Harm_r` and `Q_{ir} = m_r G_r(cos_i)` for `r ≤ 5` (via
`gegenbauer_eval`); Hoffman `= 9360/11`; the exact solver on a toy LP and `solve_square`; the main
bound exact `= 9360/11`, floor 850, HiGHS within `1e−6`, unique optimal vertex, `a*`, active set; the
certificate equalling Hoffman's, verifying, and being tight on all free classes; the certificate
re-checked "by hand" from copied numbers; the degree-4 polynomial as an alternative dual; bad
certificates and a bad primal rejected; all variants (values, `a*`, certificates, float agreement);
the antipodal constraint being free; `θ = θ′ =` Hoffman; bound `> 496`; the three CLIs' `RESULT`
lines.

```
$ .venv/bin/python -m pytest python/tests/test_scheme.py -q
27 passed in 13.7s
$ .venv/bin/python -m pytest python/tests -q
121 passed in 129.30s (0:02:09)
```

Timing: `scheme_numbers --extra-x 4` 3.1 s total (load 0.3 s, GPU pass 0.17–0.21 s per `x`, orbit BFS
including construction of the five index permutations 0.28 s); `scheme.py` 0.25 s; `lp_scheme.py`
≈ 6 s CPU (five variants, exact primal and dual enumeration each); `theta_prime.py` 4.5 s; the pytest
file 14 s. This work was built and run in `build/t22`; `tests/tasks/T2_2.cmake` registers only
`test_scheme_gpu`, and `tools/scheme_numbers.cpp` is picked up by the tools glob. Nothing was
pip-installed for it; `requirements.txt` unchanged.

---

## 5. Why every two-point method stops at 9360/11

That the scheme LP, θ′, θ, Hoffman and the lattice-free polynomial all return the same number is not
a coincidence and not a bug. The argument:

* **Hoffman's bound follows from `aQ ≥ 0` alone.** With `w = aQ`, `w₀ = |S|`, `Σ_r w_r = N a₆ = N`
  and `Σ_r w_r λ_r = N a₅ = 0`, so `|S|·k = −Σ_{r≥1} w_r λ_r ≤ −λ_min (N − |S|)`. Hence
  `θ_sym ≤ Hoffman` without any sign constraint, and `θ′ ≤ θ_sym`.
* **The LP's optimal dual *is* Hoffman's certificate.** The dual found by exact enumeration is
  precisely the multiplier vector of that argument, `β_r = (λ_r − λ_min)/(k − λ_min)`, with `β₆ = 0`
  (the `λ_min` eigenspace itself carries no multiplier).
* **It is tight on every allowed class** (`F(i) = −1` for all five free classes). So the sign
  constraints `a_i ≥ 0` are never active — dropping them gains nothing, `θ = θ′` — and **no
  two-point inequality is left unused**.
* **The equality with the lattice-free LP is the 11-design fact**: the scheme LP's constraints
  `E₀..E₅` *are* the Gegenbauer constraints of degree `≤ 5`, the lattice-free optimum is already
  attained at degree 4, and the extra `E₆` constraint is slack at `a*`.

The practical consequence, which sets up §6: **a three-point bound is the first relaxation that can
see the lattice beyond its 11-design property.** No two-point method — not the LP, not θ, not θ′, not
any spectral ratio bound — can do better than 850 on this graph, and the LP gives no information
whatsoever about how large the true gap to 496 is.

---

## 6. The three-point (Terwilliger/Schrijver) SDP: `|S| ≤ 837`

**Headline: `|S| ≤ 837`, rigorously** — strictly better than the 850 that every two-point method
gives, but only 13 of the 354 that separate 850 from the record 496. The construction gap (496 … 837)
stays wide open.

The bound is rigorous in the strong sense: the numerical dual is rounded to an exactly rational,
exactly PSD certificate, and every step of the final inequality is evaluated in
`fractions.Fraction`. The certificate is stored in `data/scheme/sdp3_certificate.json` and re-checked
from scratch by `bounds.sdp3_scheme.verify_certificate`, which rebuilds the coefficient matrices from
`data/scheme/orbitals.json` and uses **nothing** from the solver.

| | value |
|---|---|
| Two-point bound (§3 = §4) | 9360/11 = 850.909… → **850** |
| Three-point relaxation optimum (numerical) | 837.533238 |
| Three-point bound after exact rounding | 837.535866161… → **`\|S\| ≤ 837`** |
| Best construction | 496 |
| Sanity: nothing forbidden | 196565 ≥ N = 196560 |
| Sanity: forbid inner products 16 **and** 8 (cos ≤ 0) | **48** = orthoplex bound 2·24 |

### 6.1 Co₀ is *not* transitive on ordered triples with given pairwise classes

`tools/triple_orbitals` computes the orbitals of `H = Stab_Γ(x)` (`x = C[0]`, `Γ = ⟨`the five
explicit Co₀ generators`⟩`) on `C × C`, i.e. the Γ-orbits of *ordered triples* `(x, y, z)`, with
every identification witnessed by an explicit group element (Schreier vectors over the generators),
so the labels provably **refine** the true orbitals. `tools/triple_stats` computes on the GPU, for
each class-cell `(i,j,k)`, the distinct four-point histograms
`H_z(a,b,c) = #{w : class(x,w)=a, class(y,w)=b, class(z,w)=c}` — a `Stab(x,y)`-invariant, so the
number of distinct histograms is a **lower** bound on the number of orbitals. The two numbers agree,
so the labels *are* the orbitals:

```
D = 148 labels  ==  148 distinct four-point histograms   →  D = 148 orbitals
```

* Orbitals per class `i = class(x,y)`: `0:7, 1:25, 2:27, 3:30, 4:27, 5:25, 6:7` (sum 148).
* Nonempty **class-cells** `(i,j,k)`: **147**.
* Therefore **Co₀ is not transitive on ordered triples with prescribed pairwise classes.** Exactly
  one cell splits, and it is the most symmetric one: `(i,j,k) = (3,3,3)` — three mutually
  **orthogonal** minimal vectors — splits into two orbits, of sizes (number of `z` for a fixed
  orthogonal pair `x, y`)

  ```
  p^3_{33} = 43164 = 42240 + 924
  ```

  (orbital 74: 42240 vectors; orbital 73: 924 vectors, `|orbit| = 93150·924 = 86 070 600`). This is
  exactly the lattice fact that no two-point method can see: **an orthogonal triple in the Leech
  minimal vectors comes in two Co₀-inequivalent flavours.**
* Γ-orbits of **unordered** triples (`S₃`-orbits of the orbitals): **43**. These are the variables of
  the SDP.

**Dimension of the Terwilliger algebra.** `T(x) = ⟨E_0*, …, E_6*, A_0, …, A_6⟩` (dual idempotents
`E_i*` = the diagonal of the class-`i` sphere around `x`, `A_k` = the class-`k` adjacency matrix).
Closing the generated subspace under left and right multiplication by the generators, using the
structure constants `c[u,s,t]` of the orbital algebra, over `GF(2^61 − 1)`:

```
dim T(x) = 148 = D          (rank over GF(p) ≤ rank over Q ≤ D, so this is exact)
class-cells (i,j,k)  = 147
```

So `T(x)` is the **whole 148-dimensional centraliser algebra of `Stab(x)`** — the Schurian coherent
configuration of the point stabiliser — and it is **strictly larger** than the span of the 147
triple-class matrices `E_i* A_k E_j*`; that 147-dimensional span is *not* closed under multiplication
(a product of two triple-class matrices separates the two orbits of orthogonal triples). **The
Terwilliger algebra is not spanned by the triple-class matrices.** The SDP below therefore lives in
the full 148-dimensional algebra, which is both the correct and the strongest choice.

### 6.2 The relaxation

Everything is proved in the module docstring of `python/bounds/sdp3_scheme.py`; the short version.

For `S ⊆ C` let `λ_u = #{(a,b,c) ∈ S³ : (a,b,c) ∈ O_u}` (`O_u` = the Γ-orbit of ordered triples
belonging to orbital `u`) and set

```
x_u  =  λ_u / (|S| · size_u),        size_u = #{(y,z) : (x₀,y,z) ∈ O_u},  |O_u| = N·size_u.   (1)
```

Then, with `B_u` the 0/1 matrix of the orbital and `diag(k)` the orbital of `(y_k, y_k)`:

* **(P1)** `M₁ = Σ_u x_u B_u ⪰ 0`, because `(|S|/N)·M₁ = |Γ|⁻¹ Σ_{g∈Γ} [x₀ ∈ gS]·χ_{gS} χ_{gS}ᵀ`, a
  nonnegative combination of rank-one PSD matrices whose `(y,z)` entry is `λ_u/|O_u|`.
* **(P2)** `M₂ = Σ_u (x_{diag(k(u))} − x_u) B_u ⪰ 0`, the same average with `[x₀ ∉ gS]`.
* **(P3)** `0 ≤ x_u ≤ 1` (for fixed `a ∈ S`, at most `size_u` pairs `(b,c)` complete a triple in
  `O_u`, so `λ_u ≤ |S|·size_u`), `x_{diag(32)} = 1`, and `Σ_k v_k x_{diag(k)} = |S|`.
* `x_u = 0` whenever one of the three pairwise classes of `u` is a forbidden class (here: the class
  of inner product 16).
* `x` is `S₃`-symmetric (`x_u = x_{uᵀ} = x_{swap(u)}`), which cuts 148 variables to **43**.

**Relaxation.** Maximise `Σ_k v_k x_{diag(k)}` over that feasible set. The normalisation is fixed by
`x_{diag(32)} = x_e = 1`, and is exactly the one for which the objective *is* `|S|`; the checks in
§6.4 pin it down.

**Making "`⪰ 0`" a 148×148 condition.** `A = span{B_u}` is a semisimple matrix \*-algebra (the
orbitals are closed under transposition) and `C^V` is a faithful `A`-module containing every
irreducible `A`-module (double centraliser). Hence for `M ∈ A`

```
M ⪰ 0 as an N×N matrix  ⇔  M ⪰ 0 in every irreducible representation of A
                        ⇔  M ⪰ 0 in the LEFT REGULAR representation of A.
```

In the basis `{B_u}` the left regular representation is `L(x)[w,t] = Σ_u x_u c[w,u,t]`, and it is
self-adjoint for the trace form `⟨B_s,B_t⟩ = δ_{st}·size_s`. With `Δ = diag(size)`,

```
G(x) := Δ·L(x),      G(x)[w,t] = size_w · Σ_u x_u c[w,u,t]                                    (2)
```

is **symmetric with integer coefficient matrices** whenever `x_u = x_{uᵀ}`, and

```
M(x) ⪰ 0  (196560 × 196560)   ⇔   G(x) ⪰ 0  (148 × 148).                                      (3)
```

No numerical block-diagonalisation of the algebra is needed, and no irrational number enters the
exact side: `G(x) = Δ^{1/2} Z(x) Δ^{1/2}` with `Z(x) = Δ^{1/2} L(x) Δ^{−1/2}` the well-scaled
representation handed to the solver (`Z(1)` has spectrum `{196560, 0}`), and the congruence by the
positive diagonal `Δ^{1/2}` preserves positive semidefiniteness.

So the SDP is: **43 variables, two 148×148 PSD blocks, 148 + 43 + 43 linear constraints.** Note
`M₁ + M₂ = Σ_k x_{diag(k)} A_k`, so the two-point (Delsarte/scheme LP) constraint of §4 is implied —
the three-point bound can only improve on 850, and it does.

### 6.3 The exact certificate

For rational `Y₁, Y₂ ⪰ 0`, rational `μ ≥ 0` (multiplier of `y ≥ 0`) and `ρ ≥ 0` (multiplier of
`x ≤ 1`), put

```
h_q = obj_q + ⟨Y₁, G_q⟩ + ⟨Y₂, Gy_q⟩ + (μᵀ Cy)_q − ρ_q .                                       (5)
```

Then for every admissible `S`, using `x ≥ 0`, `x_e = 1`, `x_q = 0` on forbidden orbits and
`x_q ≤ 1`,

```
|S| = objᵀx ≤ objᵀx + ⟨Y₁,G(x)⟩ + ⟨Y₂,G(y)⟩ + μᵀ(Cy·x) + Σ_{q free} ρ_q(1−x_q)
            = Σ_q h_q x_q + Σ_{q free} ρ_q
            ≤ h_e + Σ_{q free} ρ_q + Σ_{q free, q≠e} max(0, h_q).                              (6)
```

Every term of (6) is an exact rational. **Validity does not depend on the numerics at all**: an
inaccurate dual makes (6) larger — never wrong. `Y₁, Y₂` are PSD *by construction*: the solver's
eigendecomposition `Y = Σ_i λ_i q_i q_iᵀ` is mapped into the `G` basis as `Ŷ = Σ_i λ̂_i p̂_i p̂_iᵀ`
with `p̂_i` an integer vector over a power of two and `λ̂_i ≥ 0` — a nonnegative combination of
rank-one terms, so no PSD verification of a rounded matrix is ever required.

For the headline run:

```
h_e            = 836.167091253
rounding slack = Σ_{q free, q≠e} max(0,h_q) + Σ ρ_q  = 1.369
exact bound    = 837.535866161…   (a rational with a 2^-… denominator, stored in the JSON)
               < 838              →   |S| ≤ 837
```

The relaxation's own numerical optimum is 837.533238, so the rounding costs 0.0026 — far less than
the 0.47 of headroom to the next integer. **The relaxation optimum is bracketed from both sides:**
the solver's primal point is feasible to `min eig = −6e−9` against a block scale of 5.97 (relative
`−1e−9`) with objective **837.533238**, and the exact dual proves **`≤ 837.535866`**. Both floor to
837.

### 6.4 Validation of the formulation (the decisive tests)

A wrong normalisation would show up immediately as an *infeasible* known set. Both known sets and the
whole scheme are checked **in exact arithmetic**, PSD included (symmetric Gaussian elimination over
`Fraction` with diagonal pivoting). The sets `data/S496.txt` and `data/S488.txt` come from the search
described in `03-search-for-497.md`.

```
feasibility of S488.txt (|S| = 488): objective = 488, x_e = 1, x >= 0 True, x <= 1 True,
    forbidden zero True, y >= 0 True, PSD1 True, PSD2 True -> FEASIBLE True
feasibility of S496.txt (|S| = 496): objective = 496, x_e = 1, x >= 0 True, x <= 1 True,
    forbidden zero True, y >= 0 True, PSD1 True, PSD2 True -> FEASIBLE True
feasibility of S = C (x == 1, nothing forbidden): objective = 196560 (= N: True),
    PSD1 True, PSD2 True
```

with block ranks 46 / 71 for the 496 and **7 / 0** for `S = C`: with `S = C` the complement is empty,
so the second block is *identically zero* — exactly as the derivation predicts. The objective
evaluates to 496, 488 and 196560 on the nose, which is the check that pins the normalisation. A
negative control in `python/tests/test_sdp3.py` inflates one coordinate of the 496's `x` and confirms
that the exact PSD test then fails.

**Sanity table** (all three from the same code path):

```
forbid nothing: relaxation optimum 196531.811986 (status optimal_inaccurate, 3.4 s)
                -> RIGOROUS |S| <= 196565  [exact 196565.300002, ranks (148,148), slack 1.385e+04]
forbid {16}   : relaxation optimum 837.533238 (status optimal, 4.4 s)
                -> RIGOROUS |S| <= 837     [exact 837.535866, ranks (138,132), slack 1.369e+00]
forbid {16,8} : relaxation optimum 47.999926 (status optimal, 2.6 s)
                -> RIGOROUS |S| <= 48      [exact 48.000060, ranks (141,118), slack 8.523e-02]
```

* nothing forbidden → 196565 ≥ N = 196560 (the relaxation cannot exclude `S = C`; the interior-point
  solver under-converges on this degenerate instance, which only *weakens* the bound);
* forbid 16 → **837**, inside `[496, 850]` and strictly below 850;
* forbid 16 and 8 (`cos ≤ 0`) → **48**, the orthoplex bound `2·24` — the three-point bound reproduces
  it exactly, which it must.

### 6.5 Reproduce, files

```
.venv/bin/pip install -r python/requirements.txt          # adds cvxpy, clarabel, scs
PYTHONPATH=python .venv/bin/python -m bounds.orbitals     # the 148 orbitals + all exact checks
PYTHONPATH=python .venv/bin/python -m bounds.sdp3_scheme \
    --save-certificate data/scheme/sdp3_certificate.json  # ~2 min, prints the table above
.venv/bin/python -m pytest python/tests/test_sdp3.py -q   # 15 passed in 66 s
ctest --test-dir build/t23 -R test_triple_gpu             # the GPU four-point statistics
```

Re-check only the certificate (nothing from the solver; rebuilds `G_q` from `orbitals.json`):

```
PYTHONPATH=python .venv/bin/python -c \
 "from bounds.sdp3_scheme import verify_certificate as v; print(v('data/scheme/sdp3_certificate.json'))"
→ {'bound': Fraction(1002749984536635160184220973136084178873152199902677
   8139, 11972621413014756705924586149611790497021399392059392),
   'bound_floor': 837, 'D': 148, 'vars': 43, 'forbidden_dots': [16],
   'matches_stored': True}                                        (10.1 s)
```

```
$ .venv/bin/python -W ignore -m pytest python/tests/test_sdp3.py -q
15 passed in 66.39s (0:01:06)

$ .venv/bin/python -W ignore -m pytest python/tests -q          # the whole suite
253 passed in 403.90s (0:06:43)

$ ctest --test-dir build/t23 -R test_triple_gpu
1/1 Test #9: test_triple_gpu ..................   Passed    2.82 sec
100% tests passed, 0 tests failed out of 1

$ cmake --build build/t23                                       # -Werror, host + CUDA
[54/54] Linking CXX executable tests/test_ls_search              (clean, no warnings)
```

| File | Role |
|---|---|
| `cuda/triple_cuda.h`, `cuda/triple_stats.cu` | GPU four-point histogram kernel |
| `tools/triple_stats.cpp` | writes `data/scheme/triples.json` (the invariant lower bound on the number of orbitals) |
| `tools/triple_orbitals.cpp` | writes `data/scheme/orbitals.json` (orbitals, structure constants, `S₃` action, triple counts of the 496 and the 488) |
| `tests/test_triple_gpu.cu`, `tests/tasks/T2_3.cmake` | GPU kernel acceptance test |
| `python/bounds/orbitals.py` | loads and exactly re-verifies the orbital algebra |
| **`python/bounds/sdp3_scheme.py`** | the relaxation, the solver driver, the exact certificate, `terwilliger_dimension`, `verify_certificate` |
| **`python/tests/test_sdp3.py`** | 15 acceptance tests |
| **`data/scheme/sdp3_certificate.json`** | the exact dual certificate of `\|S\| ≤ 837` (288 KB) |
| `python/requirements.txt` | `+ cvxpy, clarabel, scs` |

Solver versions: `cvxpy 1.7.5`, `clarabel 0.11.1`, `scs 3.2.11`, now recorded in
`python/requirements.txt`. The headline used **Clarabel** (status `optimal`).

### 6.6 What is *not* proved

1. **The one unverified group-theoretic assumption**, inherited from §4: `Γ = ⟨`the five
   generators`⟩` is taken to be Co₀. Its transitivity on `C` and its 7 orbitals on ordered pairs *are*
   verified computationally, and that is all the relaxation actually uses — a *subgroup* of Co₀ with
   the same orbitals gives the same, valid, relaxation. **Nothing in the bound depends on `Γ = Co₀`.**
2. **The 837 is an upper bound for the relaxation's optimum, not the relaxation's optimum.** The
   relaxation optimum lies in `[837.53324, 837.53587]`; the certificate proves the right-hand end.
   Both ends floor to 837. SCS was tried as a second solver and did not converge in reasonable time
   on this instance; the primal/dual bracket above is the cross-check that was actually used.
3. **Not proved: 837 is the true three-point optimum for this problem.** Adding further valid
   inequalities (four-point terms, the antipodal structure of the record, clique constraints) could
   lower it. Clique constraints were subsequently tried and give exactly nothing (§7); four-point
   terms are scoped in §9.
4. On the exact side, floating point is never used, and the exact contraction `⟨Ŷ, G_q⟩` is a single
   `object`-dtype elementwise product per orbit (about 1 s per block for all 43 orbits), so numpy's
   stacked-matmul slowness is irrelevant to this code path.

---

## 7. The clique number of the conflict graph is 24, and clique cuts are inert

Two questions, one negative answer. First, the exact clique number `ω(G)` of the 60°-conflict graph.
Second, whether valid *lifted* clique inequalities can improve the three-point relaxation of §6. They
cannot, and the reason is structural rather than a sampling accident.

| | value |
|---|---|
| **`ω(G)`, exact** | **24** (max clique of vectors pairwise at 60°) |
| upper-bound argument | Gram of `k` pairwise-16 vectors `= 16(I_k+J_k) ≻ 0`, rank `k` ⇒ `k ≤ dim = 24` |
| lower bound | explicit 24-clique, verified in exact integer arithmetic (§7.3) |
| triangles through an edge | `891 = p⁵₅₅` (scheme number, re-counted from the adjacency) |
| three-point bound, no cuts (reproduced) | 837.533238 float → exact 837.535866 → **837** |
| + triangle cuts | 837.533197 float → exact 837.534915 → **837** |
| + ω-clique (24-clique) cuts | 837.533405 float → exact 837.534727 → **837** |
| + both | 837.532980 float → exact 837.536877 → **837** |
| entrywise nonnegativity (θ⁺ route) | **already in the baseline** (§7.6) — nothing to add |
| **bound after clique cuts** | **`\|S\| ≤ 837`, unchanged** — every clique cut is *slack* at the optimum (§7.8) |

> **A distinction worth restating.** The "clique number 24 with exactly 3 full frames" appearing in
> the structural analysis of the record set (see `04-structure-of-the-496.md`) is about the
> **orthogonality** graph on the 248 antipodal classes of the 496, i.e. inner product **0**. The
> graph `G` of this document has edges at inner product **16**, and its clique number was not
> previously known in the project. That both numbers equal 24 is a coincidence with two entirely
> different proofs.

### 7.1 Common neighbourhoods, from the scheme

For an edge `(x, y)` (`⟨x,y⟩ = 16`) the common neighbours `{z : ⟨x,z⟩ = ⟨y,z⟩ = 16}` number
**`p⁵₅₅ = 891`** (§4.4, row 5 column 5 of `p⁵`). `tools/clique_g` lists them explicitly for the base
edge `(0, 1)` by dot products and cross-checks the list against the mmap'ed `data/adj.u32` row
intersection — identical, 891 entries. The induced graph `G[N(x,y)]` is **336-regular on 891
vertices** (149 688 edges). Triangle counts: 891 per edge, `4600·891/2 = 2 049 300` per vertex, and
`N·4600·891/6 = 134 270 136 000` in `G`.

Cliques grow inside neighbourhoods that shrink fast (`4600 → 891 → 336` for edge → triangle), so
exact branch-and-bound on one edge's common neighbourhood is cheap. **One edge suffices by
arc-transitivity**: `Γ` is transitive on `V` (§4.1) and `Stab(x)` is transitive on every class
(`stab_x_orbits = 7`, recorded in `data/scheme/orbitals.json` and re-verified at every load by
`bounds/orbitals.py`), so `Γ` is transitive on ordered 16-pairs. Two extra random edges were searched
anyway as a computational sanity check, and give the same numbers.

### 7.2 Upper bound `ω ≤ 24` (rank argument, no computation)

`k` vectors of norm 32 with pairwise inner product 16 have Gram matrix `16(I_k + J_k)`, with
eigenvalues `16(k+1)` (once) and `16` (`k−1` times) — positive definite of rank `k`. Vectors with a
nonsingular Gram are linearly independent, so `k ≤ dim R²⁴ = 24`. No sphere-packing input is needed
at all. (The sum `s` of a 24-clique has `|s|² = 16·24·25 = 9600` and every member has
`⟨v, s⟩ = 400`.)

Note the contrast with a natural first guess: the "coordinate-pair" family `{4(e₁+e_i) : i = 2..24}`
is a 23-clique of `(±4,±4)`-shape vectors, and it is **maximal** — the only real vector `w` with
`⟨w, 4(e₁+e_i)⟩ = 16` for all `i` is `w = (3,1²³)`, whose coordinate sum 26 `≢ 4 (mod 8)` puts it
outside the Leech lattice. Maximum cliques are *not* of this shape (§7.4): **maximal ≠ maximum here**,
which is why the branch-and-bound was needed.

### 7.3 Lower bound: the explicit 24-clique (certificate)

Branch-and-bound (greedy-colouring bound, Tomita-style, 891-bit bitsets) on `G[N(0,1)]` gives **max
clique = 22**, so with the two endpoints `ω(G) = 24`. The maximum clique found, as canonical vertex
indices of `data/leech_min.i8` (also stored with full vectors in `data/scheme/clique_g.json`):

```
K24 = [0, 1, 2386, 2398, 2399, 2403, 2404, 2406, 2407, 2410, 2411, 2414,
       2417, 2418, 2419, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 65390]
```

Checker snippet (exact integer arithmetic, run from the repo root):

```python
import numpy as np
C = np.fromfile('data/leech_min.i8', dtype=np.int8).reshape(-1, 24).astype(np.int64)
K = [0,1,2386,2398,2399,2403,2404,2406,2407,2410,2411,2414,2417,2418,2419,
     2422,2423,2424,2425,2426,2427,2428,2429,65390]
G = C[K] @ C[K].T
assert np.array_equal(G, 16*(np.eye(24, dtype=np.int64) + 1))   # Gram = 16(I+J)
```

Composition: 3 vectors of shape `(±4²,0²²)` — `4(e₁+e₂)`, `4(e₁+e₃)`, `4(e₂+e₃)` with signs `−−`,
i.e. the triangle on coordinates `{1,2,3}` — and 21 octad-shape `(±2⁸,0¹⁶)` vectors. Its Gram is
`16(I+J)`, so the 24 vectors are a **basis of `R²⁴`**: the clique is as large as the dimension
allows, meeting the rank bound of §7.2 with equality. The same is verified by `tools/clique_g` and by
ctest `test_clique_g` (exact dots), and independently by `verify_k24` /
`test_k24_gram_is_16_I_plus_J` in Python.

### 7.4 Census of maximal cliques

Bron–Kerbosch with pivoting on `G[N(0,1)]` (maximal cliques of `G[N(x,y)]` of size `s` ↔ maximal
cliques of `G` of size `s+2` containing the edge).

The **triangle** census (`G[N(0,1,2)]`, 336 vertices) is complete — 80 833 616 maximal cliques,
`2.3·10⁸` BK calls (74 s serial; 4 s with the OpenMP root-branch parallel driver, identical counts):

| size in `G[N(x,y,z)]` | count | = maximal clique of `G` through the triangle, size |
|---|---|---|
| 5 | 336 | 8 |
| 9 | 31 539 200 | 12 |
| 12 | 4 076 800 | 15 |
| 14 | 1 175 040 | 17 |
| 20 | 44 029 440 | 23 |
| 21 | 12 800 | 24 |

Since `Γ` is transitive on triangles — the `(5,5,5)` cell is a **single orbital**, id 118, osize 891,
in `orbitals.json`, so `Γ` is transitive even on *ordered* triangles — every triangle carries this
census; and every maximal clique of `G` contains a triangle (edges have `891 > 0` common neighbours,
so no maximal clique has size 2). Hence **the maximal cliques of `G` have sizes exactly
`{8, 12, 15, 17, 23, 24}`** — in particular the hand-built 23-clique of §7.2 sits in a genuine
maximal-size class, and nothing is maximal between 17 and 23. Double counting (each size-`s` maximal
clique through edge `(x,y)` contains `s−2` of the 891 common neighbours) predicts the edge census
exactly, `n_s(edge) = 891·n_s(triangle)/(s−2)`:

```
size  8:      49 896        size 17:     69 797 376
size 12: 2 810 142 720      size 23:  1 868 106 240
size 15:   279 417 600      size 24:        518 400        total 5 028 032 232
```

The **edge census itself also ran to completion** (16 threads, 490 s, `1.64·10¹⁰` BK calls):
**5 028 032 232 maximal cliques of `G[N(0,1)]`**, with per-size counts **exactly equal** to the
predictions above — two independent computations (direct enumeration on 891 vertices versus the
complete triangle census combined with triangle-transitivity) agreeing on six 10-digit numbers. An
earlier serial run capped at a 900 s budget had reached `1.17·10⁹` of them; that partial count had
the same size spectrum.

Global counts, by vertex- and edge-transitivity: 518 400 24-cliques through every edge, hence
`518 400·|E|/C(24,2) =` **849 139 200 000 24-cliques in `G`** (103 680 000 through each vertex).

Sanity anchors: triangles through an edge `= 891 = p⁵₅₅` (an independent count); the common
neighbourhood is 336-regular; max clique through a triangle `= 21 + 3 = 24`; and all the divisions
`n_s(edge) = 891·n_s(triangle)/(s−2)` come out integral.

### 7.5 Lifted clique inequalities: the derivation that is not vacuous

With `a₁₆ = 0` already forced, the pair-level clique inequality `Σ_{v∈K} 1_S(v) ≤ 1` summed over a
Γ-orbit of cliques only gives `|S|·(3|O_T|/N) ≤ |O_T|`, i.e. `|S| ≤ N/3` — useless. The value must
come from cliques *conditioned on points of `S`*. Fix any pair `(a, b) ∈ V²` with
`i = class(a,b) ≠ 5` and any clique `K` of `G`. For every `g ∈ Γ` the set `gS` is independent, so
`|K ∩ gS| ≤ 1` and

```
[a ∈ gS][b ∈ gS] · Σ_{v∈K} [v ∈ gS]  ≤  [a ∈ gS][b ∈ gS].
```

Averaging over `g ∈ Γ` (exactly the (P1) computation of §6.2, term by term) and dividing by `|S|/N`:

```
(C1)   Σ_{v∈K} x_{u(a,b,v)}  ≤  x_diag(i) ,          u(a,b,v) = orbital of the triple (a,b,v),
```

and the same average of `[a∈gS](1−[b∈gS])·Σ_v[v∈gS] ≤ [a∈gS](1−[b∈gS])` gives the
complement-conditioned version

```
(C2)   Σ_{v∈K} ( x_diag(class(a,v)) − x_{u(a,b,v)} )  ≤  1 − x_diag(i) .
```

Both are exact linear inequalities in the variables of §6, valid for **every** admissible `S` — the
derivation never touches the relaxation. `a = b` (`i =` identity) is allowed and turns (C1) into
`Σ_{v∈K} x_diag(class(a,v)) ≤ 1`, the diagonal clique inequality of the `M₁` block. Terms whose
triple has a forbidden pairwise class sit on variables that are exactly 0. With `|K| = 1`, (C1) is
the existing `y ≥ 0` constraint; with `|K| = ω = 24` it is 24 orbit variables against one diagonal —
a priori much stronger. The doubly-complemented version (`(1−[a])(1−[b])·…`) has `N/|S|` on the right
and does not linearise usefully: it is vacuous after replacing `N/|S| ≥ N/850`.

**Identifying `u(a,b,v)`.** The class triple `(i,j,k)` determines the orbital except in the single
split cell `(3,3,3)` (§6.1: `43164 = 924 + 42240`). There the GPU-verified four-point histograms
differ at `H[5,5,5] = #{w : w` a common `G`-neighbour of `a, b, v}` = **2** (orbital 73, the small
one) versus **0** (orbital 74); `w` must be one of the `p³₅₅ = 44` common `G`-neighbours of `(a,b)`,
so the test is 44 dot products. `CliqueCutter.selfcheck_classifier` re-derives the exact split
`924 + 42240` from scratch for a random orthogonal pair on every run, and in the tests.

### 7.6 The mandatory validity gate, and why θ⁺ adds nothing

Every generated cut is checked against the recorded 496 **and** 488 in exact `Fraction` arithmetic
(their triple distributions from `orbitals.json`, `S₃`-consistency re-verified by `set_point`):

```
exact feasibility of S488.txt under all 662 cuts: OK (80 tight, 582 strict)
exact feasibility of S496.txt under all 662 cuts: OK (80 tight, 582 strict)
```

No cut is ever violated; a deliberately corrupted cut *is* caught by the 496
(`test_corrupted_cut_is_caught_by_the_496`). The exactly-tight cuts are the structurally forced ones:
`class(a,b) = −16` under (C1) — the 496 has `a₋₁₆ = 0`, so both sides vanish — and
`class(a,b) = −32` under (C2), with `x_diag(0) = 1` for the antipodal 496.

**Entrywise nonnegativity (Schrijver's θ⁺ route) is already imposed by the baseline**: the orbital
matrices `B_u` are 0/1 with disjoint supports, so `M₁ ≥ 0` entrywise `⇔ x ≥ 0` (present in (P3)) and
`M₂ ≥ 0` entrywise `⇔ y ≥ 0` (also present). In the regular-representation coordinates,
`G(x)[w,t] = size_w Σ_u x_u c[w,u,t] ≥ 0` is then automatic (`c ≥ 0`). Nothing to add: **the
baseline of §6 already *is* the θ⁺-strength three-point bound.**

### 7.7 Solves and rigorous rounding

`python -m bounds.sdp3_cliques` runs Clarabel, then the exact-rational rounding of §6.3 extended with
cut multipliers `ν ≥ 0`:
`h_q = obj_q + ⟨Y1,G_q⟩ + ⟨Y2,Gy_q⟩ + (μᵀCy)_q − (νᵀA)_q − ρ_q`,
`bound = h_e + Σρ + νᵀt + Σ max(0,h_q)`.

```
baseline (no cuts)                 cuts    0  float     837.533238  exact     837.535866  floor    837  status optimal  slack 1.369e+00
+ triangle cuts                    cuts  437  float     837.533197  exact     837.534915  floor    837  status optimal  slack 1.028e+00
+ omega-clique cuts                cuts  225  float     837.533405  exact     837.534727  floor    837  status optimal  slack 1.087e+00
+ triangle + omega cuts            cuts  662  float     837.532980  exact     837.536877  floor    837  status optimal_inaccurate  slack 8.577e-01
violation scan at 'baseline' optimum over 1430 cuts: max -2.520e-14  (#within 1e-7 of tight: 123)
violation scan at '+ triangle + omega cuts' optimum over 1430 cuts: max -2.820e-13
sanity: forbid nothing (no cuts: clique cuts need independence) -> 196565 >= N = 196560: True;
        forbid {16,8} with omega cuts -> 48 (orthoplex 48)
RESULT ok=1 omega=24 baseline=837 with_cuts=837 none=196565 forbid168=48
```

The baseline reproduces §6 exactly (float 837.533238, exact 837.535866… → 837). All cut-augmented
solves stay at the same optimum; the tiny float differences are solver noise — adding constraints
cannot increase the value, and the exact certificates all floor to 837. The 662-cut run returns
`optimal_inaccurate`, and the certificate machinery is indifferent to that: an inaccurate dual only
*weakens* the exact bound, which still floors to 837. The extended certificate of the best run is at
`data/scheme/sdp3_cliques_certificate.json` (369 KB) and is re-checked from scratch by
`verify_cuts_certificate`, which **re-derives every cut row from its recorded instance
(kind, `a`, `b`, `K`)** — re-verifying that `K` is a clique and re-classifying every triple — and
recomputes `h` and the bound in exact rational arithmetic, using nothing from the solver or from the
generating run.

### 7.8 Why clique cuts cannot bite here (the structural reason)

The violation scan says no sampled cut separates the optimum — neither the 1430 instances of the CLI
run (both kinds, `|K| ∈ {3, 24}`) nor a wider scan of **6361 instances with
`|K| ∈ {3, 6, 12, 18, 24}`**, both kinds, all six allowed base classes, random positions (max
violation `−2.5·10⁻¹⁴` overall, `−1.9·10⁻⁶` among the non-degenerate cuts; the 496/488 exact gate
passes on all 6361 too, 790 tight each). The instances "tight to `−3·10⁻⁹`" are exactly the
class-`(−16)` ones, where `x_diag(−16) ≈ 0` makes both sides vanish. The *informative* cuts are slack
by an order of magnitude — least-slack examples at the baseline optimum:

```
(C1, class -8, |K|=24):  LHS_max = 6.5e-04  vs  rhs x_diag = 5.587e-03    (12% used)
(C1, class  0, |K|=24):  LHS_max = 3.7e-04  vs  rhs x_diag = 3.319e-03    (11% used)
(C1, class +8, |K|=24):  LHS_max = 6.8e-04  vs  rhs x_diag = 5.587e-03    (12% used)
(C2, all classes):       slack 0.88+ of rhs ≈ 1
```

The reason is a counting fact. The optimum behaves like a near-product pseudo-density with one-point
density `σ = 837.5/N ≈ 1/235`: conditioned on `(a,b) ∈ S`, the third-point density at any single
vertex `v` is `O(σ)`, so `Σ_{v∈K} x_u ≈ |K|·σ·x_diag(i)`, and the cut `Σ ≤ x_diag(i)` binds only when
`|K| ≳ 1/σ ≈ 235`. But `ω(G) = 24 ≪ 235` — an order of magnitude short. (At the two-point level the
same gap is the classical fact `α·ω ≤ N` for vertex-transitive graphs: clique structure alone could
never push below `N/ω = 8190`, far above 837.) Clique cuts would only start to matter if the
relaxation value ever dropped below `≈ N/ω = 8190` **and** the optimum concentrated its conditional
mass — neither is close to happening for this graph. **Whatever is left at three points has to come
from constraints that see *more* of the lattice** (four-point terms, or the split `(3,3,3)` cell
exploited directly), not from cliques.

### 7.9 Reproduce, files, dead ends

```
cmake -S . -B build/b1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/b1 --target clique_g test_clique_g
build/b1/tools/clique_g --edges 2 --census-seconds 3000 --triangle-census
                                       # full run incl. both complete censuses, ~9 min on 16 threads;
                                       # writes data/scheme/clique_g.json
ctest --test-dir build/b1 -R test_clique_g                 # fast acceptance subset (0.2 s)

PYTHONPATH=python .venv/bin/python -m bounds.sdp3_cliques \
    --save-certificate data/scheme/sdp3_cliques_certificate.json --big-pool 20
PYTHONPATH=python .venv/bin/python -c \
 "from bounds.sdp3_cliques import verify_cuts_certificate as v; \
  print(v('data/scheme/sdp3_cliques_certificate.json'))"
.venv/bin/python -m pytest python/tests/test_sdp3_cliques.py -q      # 13 passed
```

```
$ .venv/bin/python -W ignore -m pytest python/tests/test_sdp3_cliques.py -q
13 passed in 26.09s

$ .venv/bin/python -W ignore -m pytest python/tests/test_sdp3.py \
      python/tests/test_sdp3_cliques.py python/tests/test_scheme.py -q
55 passed in 87.11s (0:01:27)          # the scheme and three-point suites unaffected

$ ctest --test-dir build/b1 -R test_clique_g
1/1 Test #5: test_clique_g ....................   Passed    0.19 sec
100% tests passed, 0 tests failed out of 1

$ build/b1/tools/clique_g --edges 2 --census-seconds 3000 --triangle-census   # full run, abridged
edge (0,1): common neighbours (= triangles through the edge) = 891 (scheme: p^5_55 = 891)
adjacency cross-check (data/adj.u32): row intersection 891, identical to dot-product list: yes
G[N(x,y)]: 891 vertices, 149688 edges, degree min 336 max 336 (336-regular expected)
edge (0,1): max clique in G[N(x,y)] = 22  (omega(G) = 24), 16773283 B&B nodes, 9.1 s
random edge (134465,181548): common 891, max clique 22 -> omega through it 24, 15683114 nodes, 8.8 s
random edge (141870,134493): common 891, max clique 22 -> omega through it 24, 16143874 nodes, 9.1 s
explicit 24-clique: pairwise inner products all 16, norms all 32: yes
census complete: 5028032232 maximal cliques, 16441863075 calls          (16 threads, 490 s)
triangle census complete: 80833616 maximal cliques, 234356922 calls
RESULT ok=1 omega=24 common=891 lambda=891 total_s=515.7
```

| File | Role |
|---|---|
| `tools/clique_g.cpp` | common neighbourhoods, exact max-clique B&B, BK census, JSON certificate |
| `tests/tasks/B1.cmake` | registers ctest `test_clique_g` (= the tool with `--test`, CPU-only, fast) |
| `data/scheme/clique_g.json` | `ω` certificate: the 24-clique (indices + vectors), per-edge results, census |
| **`python/bounds/sdp3_cliques.py`** | (C1)/(C2) cut derivation + classifier, `Sdp3Cuts` (solve + extended exact certificate), `verify_cuts_certificate`, CLI |
| `python/tests/test_sdp3_cliques.py` | 13 acceptance tests, including the mandatory 496/488 gate and a corrupted-cut canary |
| `data/scheme/sdp3_cliques_certificate.json` | exact dual certificate of `\|S\| ≤ 837` with 225 ω-clique cuts (369 KB) |

**Dead ends recorded.**

1. The hand-built 23-clique `{4(e₁+e_i)}` is maximal but not maximum (§7.2) — see the correction in
   §10.
2. Naive orbit-averaged (unconditioned) triangle inequalities reduce to `|S| ≤ N/3`: vacuous. Only
   the pair-conditioned liftings (C1)/(C2) carry information.
3. The doubly-complemented cut `((1−[a])(1−[b]))` has a nonlinear `N/|S|` right-hand side and is
   vacuous after linearising with `|S| ≤ 850`.
4. Special positions do not help: pairs built from negatives of clique members either have a
   forbidden base class or push all clique members into the `−16` class, whose diagonal variable is
   already `≈ 0`; the cut then compares 0 to 0.
5. The max-clique and census tool is C++ (`tools/clique_g.cpp`), which avoids a roughly 40× slower
   Python B&B and census.

---

## 8. Slack structure at the 837 optimum

`bounds.sdp4_scoping.tight_constraints` re-solves the three-point SDP (status `optimal`, 837.533238)
and reports where the bound is pinned:

```
variables: 28 free (+15 forced to 0);  x at 0: 11 orbits;  x at 1: none;  16 interior
y = Cy x >= 0:   65 of 148 tight
block1 (M1):     120 of 148 zero eigenvalues  (rank 28)
block2 (M2):      80 of 148 zero eigenvalues  (rank 68)
x_diag: class 0: 1.000  1: 0.000  2: 0.005587  3: 0.003319  4: 0.005587  5: 0  6: 1
   496: class 0: 1.000  1: 0.000  2: 0.003210  3: 0.002056  4: 0.003210        (ratio ~1.6-1.7)
```

The optimum is **massively degenerate**: both PSD blocks are rank-deficient by 120 and 80, 65 of the
`y`-inequalities are tight, and 11 of the 27 free non-identity orbits sit at zero. The optimal
pseudo-set is antipodal, has *the 496's support* — no `±16` and no `−16` pairs, i.e. the relaxation
discovers on its own that good sets avoid dot `−16` — and its pair-class densities are about
1.6–1.7× the 496's, so it looks like "the 496 inflated by `837/496 = 1.69`". The bound is pinned by a
large active set of genuinely three-point constraints, not by a single slack inequality: **there is
no cheap cut left at this level**, which is exactly the independent finding of §7 (clique cuts give
zero movement; `ω = 24` against a threshold of `≈ 235`).

---

## 9. Scoping a four-point (Lasserre level-3) bound

**Verdict: a four-point SDP in the same style is *borderline* tractable on this machine and clearly
tractable on a workstation — but it is a multi-week project.** No four-point SDP was formulated or
solved; the reduced problem is not tiny (largest block 4107 against a "solve it if blocks ≤ 500"
threshold, a factor of 8 over). The complete orbit data any such bound needs **is** computed and
committed, in `data/scheme/quad_orbits.json`. Hardware: RTX 3070 Laptop (sm_86), CUDA 12.8, 16 GB
RAM / 16 cores.

| quantity | value |
|---|---|
| `dim T(x,y)` per pair class (dot −32,−16,−8,0,8,16,32) | **148, 1893, [2971, 2972], [4097, 4107], [2971, 2972], 1893, 148** |
| Co₀-orbits of ordered 4-tuples (all repetition patterns) | in **[14121, 14133]** |
| … admissible (no dot-16 pair among the six) | 6183 (5721 with four distinct points) |
| SDP variables after `S₄` symmetrisation (estimate) | ≈ 240–300 genuinely-4-point + the 43 of §6 (bracket `[258, 6183]`) |
| PSD blocks (same pattern: 3 membership blocks × 5 base-pair classes + 2 diagonal) | 17 blocks, largest **4107 × 4107**; `Σd² = 1.1×10⁸`, `Σd³ = 3.9×10¹¹` |
| solver estimate (Clarabel) | ~7 GB iterates + SDP data; **hours–days** per solve; RAM on this box (16 GB) is the binding constraint |
| exact rational certification (the left-regular-representation trick) | **possible in principle** (algebra dims stay ≤ 4107) but the `Fraction`-matrix approach must become sparse-integer; structure-constant tensors ~`10⁶–10⁷` nonzeros/class |
| expected movement of the bound | genuine but probably modest, §9.4 |

Bounds ladder unchanged: **496 ≤ |S| ≤ 837**.

### 9.1 The stabiliser chain, one level down

The three-point objects were `Stab(x)`-orbitals on pairs `(y,z)` — `D = 148` — and `dim T(x) = 148`,
the block size of the three-point SDP. One level down, for `x = C[0]` and one representative `y_i`
per pair class (the same `y_i` as `tools/triple_orbitals`):

* **Level 2, re-derived.** Orbits of `Stab(x, y_i)` on vertices `z`, via labels of explicit random
  stabiliser elements (union-find over `IndexPerm`s, Schreier transversals — the same machinery as
  §6.1). Counts per class: **7, 25, 27, 30, 27, 25, 7**, exactly the earlier `labels_per_class`; the
  `z`-orbit `(j,k,size)` multisets reproduce the 148 orbitals verbatim, and the per-cell sizes sum to
  the intersection numbers `p^i_{jk}` (pytest `test_b2.py`). Cells versus orbits: only the `(3,3,3)`
  cell splits (`924 + 42240`), as before. `O_c = (7,25,27,30,27,25,7)`.
* **Level 3, new.** For every `z`-orbit representative `z_r` (148 of them across the 7 classes), the
  orbits of `Stab(x, y_i, z_r)` on `w`, bracketed from both sides:
  * **Lower bound — GPU** (`cuda/quad_stats.cu`, extending the triple kernel from 2 fixed points / 49
    groups to 3 fixed points / 343 groups): for every `w` the five-point histogram
    `H_w[a][b][c][d] = #{v : class(x,v)=a, class(y,v)=b, class(z,v)=c, class(w,v)=d}` — a
    `Stab(x,y,z)`-invariant, so `#distinct H_w ≤ #orbits`. One full pass is `196560²` dots ≈ 0.35 s;
    148 passes ≈ 150 s wall including host-side dedup.
  * **Upper bound — CPU**: orbits of explicit random elements of `Stab(x, y_i, z_r)` (products of the
    `Stab(x,y_i)` pool pulled back through a Schreier transversal of the `z`-orbit). Labels refine
    orbits, so `#labels ≥ #orbits`.
  * Where the two agree the count is **exact** — the same proof pattern that established
    `D = 148 = 148` at three points. They agree for **143 of the 148** `z`-orbits.

Then

```
dim T(x, y_i) = #Stab(x,y_i)-orbitals on ordered pairs (z,w)
             = Σ over z-orbit reps r of #orbits of Stab(x, y_i, z_r) on w
```

is the dimension of the centraliser algebra of `Stab(x, y_i)`, where a four-point SDP's PSD blocks
would live via the same left-regular-representation trick (§6.2: no numerical block-diagonalisation,
integer coefficient matrices from the structure constants).

| class `i` | dot | `O_c` = `z`-orbits | `w`-orbit counts per `z`-rep | `dim T(x, y_i)` |
|---|---|---|---|---|
| 0 | −32 | 7  | 7 … 30   | **148** (exact; anchor: `Stab(x,−x) = Stab(x)` ⇒ must be `D`) |
| 1 | −16 | 25 | 25 … 153 | **1893** (exact) |
| 2 | −8  | 27 | 27 … 269 | **[2971, 2972]** |
| 3 | 0   | 30 | 30 … 282 | **[4097, 4107]** |
| 4 | +8  | 27 | 27 … 269 | **[2971, 2972]** |
| 5 | +16 | 25 | 25 … 153 | **1893** (exact) |
| 6 | +32 | 7  | 7 … 30   | **148** (exact; anchor: `Stab(x,x) = Stab(x)`) |

Validation, all clean: **1776 GPU histogram rows re-computed on the CPU, 0 mismatches**; labels
refine histogram classes with **0 violations**; every per-rep orbit-size sum `= N`; classes 0 and 6
give exactly 148 with per-rep counts `(7,25,27,30,27,25,7)`; conjugate classes 1↔5 and 2↔4
(`Stab(x,y) = Stab(x,−y)`) agree on every deterministic quantity.

**The 5 unresolved cells (the caveat, stated).** Histogram `= 269` versus labels `= 270` on the four
cells of `(·,3,3)`/`(·,3,4)` type in classes 2/3/4, and `282` versus `288` on the **42240-flavour of
the orthogonal `(3,3,3)` cell** (the 924-flavour is exact at 131). A rerun of class 3 with an
independent seed and 3× the stabiliser elements (26–28 per rep,
`runs/b2/quad_c3_retry_seed777.log`) reproduces the label counts exactly, so the likely truth is that
the **labels are the orbit counts and the five-point histogram is too coarse there** — i.e. genuinely
distinct `Stab(x,y,z)`-orbits with identical five-point histograms, the level-3 echo of how the
two-point invariant missed the `(3,3,3)` split until the four-point histogram caught it. **This is
evidence, not proof**; a six-point invariant would settle it. Either endpoint of the bracket changes
no conclusion below. Note that the unresolved cells all sit on the orthogonality-adjacent geometry —
the same corner of the lattice identified in §6.1 as the first thing two-point methods cannot see.

**Free structural facts, of independent use.**

* **Co₀ is transitive on ordered triangles and on ordered 4-cliques of the conflict graph**: there is
  exactly one `z`-orbit with cell `(5,5,5)` (size `891 = p⁵₅₅`) and exactly one all-conflict
  `w`-orbit — representative 4-clique **`(C[0], C[1], C[2], C[3])`**, with 336 vectors completing a
  given triangle to a 4-clique, all equivalent under the stabiliser of the triangle. This is
  consistent with the arc-transitivity and `ω = 24` findings of §7, and the lifted cuts of §7.5 can
  use this orbit data directly.
* The split `(3,3,3)` cell's two flavours stay very different one level down: 131 versus 282–288
  `w`-orbits — the 924-flavour orthogonal triple is far more symmetric.

### 9.2 Size of the four-point relaxation

**Variables.** Each Co₀-orbit of ordered 4-tuples determines `class(x,y) = i` and one
`Stab(x,y_i)`-orbit of `(z,w)`, so `Σ_i dim T(x,y_i) ∈ [14121, 14133]` counts ordered-4-tuple orbits
exactly, repetition patterns included. Admissible (no dot-16 pair): 6183; of those 5721 have four
distinct points and 462 are degenerate (= lower-level configurations). After `S₄` symmetrisation,
generic orbits collapse ≈ 24:1, so ≈ `5721/24 ≈` **240–300 genuinely four-point variables** plus the
43 of §6. Variable count is a **non-issue**.

**Blocks.** One level up from the three-point pattern: per base-pair class, membership conditioning
(both in `S` / one in / both out) gives ~3 blocks of size `dim T(x,y_i)`; base-pair classes with
`y ≠ ±x` and pairs allowed in `S` are `{−16, −8, 0, +8}`, plus the `±32` diagonals (the two 148-blocks
of §6). Accounting conservatively for all five off-diagonal geometries: **17 blocks, sizes 148…4107,
`Σd² = 1.1×10⁸`, `Σd³ = 3.9×10¹¹`.** Block size is **the** issue: 4107 against 148 — a factor 28 in
dimension, ~`2×10⁴` in flops.

**Solver (Clarabel).** Cone scalars `Σd(d+1)/2 = 5.7×10⁷`; iterate storage plus NT scaling work
matrices ≈ 7 GB before the SDP data and KKT factors — **on this 16 GB laptop memory is the binding
constraint**. Runtime ≈ `10·Σd³` flops per iteration × ~60 iterations ≈ hours at LAPACK rates, plus
Schur-complement formation over ~300 variables (sparse `G_q` keep this subdominant). On a 64–128 GB
workstation this is a legitimate but heavy solve. A first pass could drop the two largest classes'
"both-out" blocks (a valid relaxation, weaker) to roughly halve everything.

**Exact certification.** The left-regular-representation trick needs the structure constants
`c[u,s,t]` of each centraliser algebra: computable exactly by the same `O(N)` pass per orbital as
`triple_orbitals` step 7 (≈ 14k orbitals × `N` lookups ≈ minutes), with storage at the three-point
fill rate (`11382/148² = 0.52` nnz/`D²`) ≈ `1.9–8.8×10⁶` nonzeros per class — fine as sparse data.
The algebra dimensions stay ≤ 4107, so the trick itself survives. What does *not* survive unchanged
is the dense `Fraction`-matrix arithmetic: `4107²` exact rational entries per contraction is `~10⁸`
Fractions, so the certificate side must be rewritten as sparse integer contractions —
straightforward, but real work. The 496's quadruple distribution (the decisive feasibility anchor,
from `data/S496.txt`) costs `496⁴ ≈ 6×10¹⁰` orbit-label lookups done naively — it needs the same GPU
treatment as the orbit data, or symmetry reduction; budget a day of work.

### 9.3 What movement to expect

For binary codes the analogous step — Schrijver's triple SDP (2005) → the
Gijswijt–Mittelmann–Schrijver quadruple-distance bound (2012) — is the best available prior: GMS
improved many table entries by a few percent and in the best case closed a gap outright
(`A(20,8) = 256`), at the cost of SDPs that were then at the limit of feasibility. **(That comparison
is cited from memory — the magnitudes are right, but individual entries should be re-checked before
being quoted anywhere.)** Transferring the pattern here:

* Two-point → three-point moved 850.9 → 837.5 (−1.6%). A comparable relative step would put the
  four-point bound around **820–830**; a GMS-like best case, where the extremal structure is highly
  rigid (and §8 shows the optimum already mimics the 496's two-point profile), could be larger, but
  nothing in the slack structure suggests a collapse toward 496.
* What the four-point level genuinely adds: the 5721 distinct-4-tuple orbits carry lattice
  information — e.g. the two orthogonal-triple flavours *interact* with a fourth point in 282–288
  inequivalent ways — that no three-point functional can express. After the clique-cut result of §7,
  four-point terms are the **only** known remaining lever in this family.

**Recommendation.** Do not attempt the full solve on this machine. If the ladder must be pushed:
(1) workstation-class RAM; (2) start with the both-in blocks only for classes `{0,2,3,4}` plus the
three-point baseline (largest block 4107, ~5 blocks — a valid intermediate bound); (3) rewrite the
certificate path sparse-integer *first*, since an unverifiable four-point number would be worthless
by this project's standards. Budget: 2–4 weeks. The alternative — accept 837 and spend the effort on
constructions — remains reasonable: the gap is 341 and even an optimistic four-point step removes
only a few percent of it.

### 9.4 Reproduce, files, notes

```
cmake -S . -B build/b2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/b2                                       # -Werror clean, 66/66
./build/b2/tools/quad_stats --out data/scheme/quad_orbits.json    # 151 s; RESULT ok=1
PYTHONPATH=python .venv/bin/python -m bounds.sdp4_scoping         # scoping + 837 slack analysis
.venv/bin/python -m pytest python/tests/test_b2.py -q             # 7 passed
```

Full tool logs are in `runs/b2/` (`quad_full.log`, plus the independent-seed class-3 rerun). Tool
`RESULT` lines:

```
RESULT ok=1 all_exact=0 classes=7 dimT_lo=14121 dimT_hi=14133 adm=6183 adm4=5721 cliques4=1 spot=0/1776 total_ms=151065 out=data/scheme/quad_orbits.json
RESULT ok=1 dimT=148,1893,2972,4107,2972,1893,148 ordered=14133 admissible=6183 vars_lo=258 max_block=4107 clique4=1
```

| File | Role |
|---|---|
| `cuda/quad_cuda.h`, `cuda/quad_stats.cu` | GPU five-point histogram kernel (3 fixed points, 343 groups; extends `triple_stats.cu`) |
| `tools/quad_stats.cpp` | driver: level-2/3 stabiliser chain (Schreier + union-find), GPU lower bounds, label upper bounds, CPU spot checks, writes the JSON |
| **`data/scheme/quad_orbits.json`** (428 KB, committed) | all 14133 `w`-orbit rows: `[w_rep, size, a, b, c, forbidden, distinct4]` per (class, `z`-orbit) |
| `python/bounds/sdp4_scoping.py` | anchor checks against `orbitals.json`, size/cost scoping, 837 tight-constraint analysis |
| `python/tests/test_b2.py` | 7 acceptance tests (anchors, conjugacy, split cell, clique orbits, intersection numbers) |
| `runs/b2/` | full run log + independent-seed class-3 rerun (bracket reproduced) |

Notes and dead ends:

* The level-3 label loop stops early when labels `=` histograms (exactness certified), which is why
  most representatives needed only 2 stabiliser elements.
* No four-point SDP was formulated: the "solve it if tiny" threshold (blocks ≤ 500) fails by 8×.
  Nothing was attempted and abandoned; the 151 s data run was first-try clean after the class-6 smoke
  test reproduced 148.
* The `S₄`-symmetrised variable count is **estimated, not computed** — computing it exactly needs the
  `S₄` action on orbit labels (mapping permuted representative quadruples back through the label
  machinery), which belongs to a future four-point effort.
* `quad_stats` re-sorts the `v`-array per `w`-chunk (6× per pass) rather than caching — deliberate
  simplicity; the whole run is 151 s.

---

## 10. Corrections

Errors found and fixed along the way, kept here because they are part of the audit trail.

1. **Class index formula.** An early GPU kernel and its CPU reference used `(dot+32)/8` as the
   association-scheme class index. That is wrong: the seven inner products are not equally spaced, so
   the formula maps dot 32 to 8 and dot −16 to 2. The kernel raised "inner product out of range" and
   the CPU reference indexed `M[i*7+j]` out of bounds. The canonical non-uniform mapping
   `t = (d+32)>>3; c = t − [t>1] − [t>7]` (that of `kiss::ip_class`) is used everywhere now. See the
   note in §1.

2. **`ω(G) = 24`, not 23.** The first guess was that the maximum clique of the conflict graph is the
   "coordinate-pair" family `{4(e₁+e_i) : i = 2..24}`, of size 23. That family *is* maximal — the
   only real vector completing it is `(3,1²³)`, whose coordinate sum 26 `≢ 4 (mod 8)` puts it outside
   the Leech lattice — but it is **not maximum**: branch-and-bound finds a 24-clique built from three
   `(±4²,0²²)` vectors and 21 octad-shape `(±2⁸,0¹⁶)` vectors (§7.3). The recorded lesson: *maximal ≠
   maximum here*, which is why the exact search was needed.

3. **The clique number 24 of the conflict graph is not the clique number 24 of the record set's
   orthogonality graph.** Elsewhere in the project a clique number of 24, "with exactly 3 full
   frames", is stated for the **orthogonality** graph (inner product 0) on the 248 antipodal classes
   of the 496 — see `04-structure-of-the-496.md`. This document's `G` has edges at inner product 16.
   The two facts have entirely different proofs and their agreement is a coincidence; conflating them
   would be a mislabelling.

4. **Vertex-transitivity is verified, not assumed.** It was originally to be recorded as the single
   unverified assumption behind the scheme bound. It is instead a computed fact
   (`vertex_transitive_verified: true` in `data/scheme/intersection_numbers.json`), resting on the
   five explicit generators whose membership in `Aut(C)` `index_permutation` proves at construction
   (`--no-orbit` disables the check). The only literature input that remains is the *interpretation*
   of the reduced programs as Lovász's θ, which does not affect any bound.

5. **The grid sanity check is exact, not approximate.** The classical-grid Delsarte LP was expected
   to return "≈ 196560"; the sandwich argument of §3.5 shows it must return *exactly* 196560, which
   is a stronger statement than the approximate one it replaced.

6. **The GPU cost estimate was pessimistic by ~50×.** The full `N×N` pair-class histogram pass was
   estimated at "~10 s"; it measures 0.17–0.21 s. (The estimate of `3.9×10¹⁰` dots for one pass per
   `y` was right.)

7. **A reported `-Werror` failure in `tools/triple_orbitals.cpp` (unused `yi`) does not exist.** `yi`
   is used at lines 291 and 294, and a full `cmake --build build/t23` is clean (54/54, no warnings).
   No change was needed.

8. **The 837 supersedes the 850 as the project's upper bound**, but it does not invalidate it: the
   two-point 850 remains a correct, independently certified bound, and it is the one that is checkable
   entirely by hand (§4.6). The clique-cut work (§7) left 837 unchanged, and the four-point work (§9)
   was scoped but not run, so **the ladder as shipped is `496 ≤ |S| ≤ 837`.**

---

## Reproducing everything

Per-section reproduction commands are given inline (§3.5, §4.8, §6.5, §7.9, §9.4). The repo-root
`REPRODUCE.md` drives the whole pipeline from the lattice construction onwards; the lattice data
these bounds consume (`data/leech_min.i8`, `data/adj.u32`, the canonical vector order) is documented
in `01-lattice-and-verification.md`.

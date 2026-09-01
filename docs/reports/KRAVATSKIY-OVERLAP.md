# Duplication audit against A. Kravatskiy's `KissingNumbers`

**External repository.** <https://github.com/alexlegeartis/KissingNumbers>, cloned read-only at
`--depth 1` into a session scratchpad on 2026-09-01. One squashed commit `0137e87`, dated
2026-08-30 18:37:29 +0300. MIT licence, `Copyright (c) 2026 Alexey Kravatskiy`. `CITATION.cff`:
Alexey Kravatskiy, MIRIAI (Moscow Independent Research Institute of Artificial Intelligence),
`kravatskii.a@miriai.org`, *"Kissing numbers: new lower bounds in dimensions 25 through 96, with
verification packages"*, v1.0.0, released 2026-08-30, "If you use these bounds or this verification
code, please cite it."

His own files spell the surname **Kravatskiy**; that spelling is used throughout this document. The
clone was never modified, no script of his was executed, nothing was redistributed and nothing was
pushed anywhere. Every quantitative statement attributed to him below is quoted or paraphrased from
his own files, with the path given.

**Scope of the repository.** 47 dimensions, 25 through 96, all lower bounds on `τ(n)`;
`common/` (shared library), `verifications/{improved,closed,recovered,superseded}/`,
`KNOWLEDGE.md` (411 KB, 107 sections, "written as the work happened"), `RESULTS.md`, `audit.py`,
`run_all.py`, `update.py`.

**One structural asymmetry to keep in mind.** His project is *broad*: 47 dimensions, five distinct
construction families, four lattices. Ours is *narrow and deep*: one problem — the maximum 60°-free
subset of the Leech minimal vectors — and its immediate consequence in dimensions 25–31. The overlap
is exactly where those two shapes meet, and it is smaller than the two repositories' surface
similarity suggests.

---

## Contents

1. [Units](#1-units-the-one-thing-that-had-to-be-settled-first)
2. [(a) Results we both have](#2-a-results-we-both-have)
3. [(b) What we have that he does not](#3-b-what-we-have-that-he-does-not)
4. [(c) What he has that we have not touched](#4-c-what-he-has-that-we-have-not-touched)
5. [(d) What to stop working on](#5-d-what-to-stop-working-on)
6. [Corrections to our own briefing](#6-corrections-to-the-briefing-that-set-this-audit)
7. [Verification and reproduction](#7-verification-and-reproduction)

---

## 1. Units — the one thing that had to be settled first

Settled in full in [`07-lines-vs-vectors.md`](07-lines-vs-vectors.md). Summary:

* He counts the class problem in **lines** (antipodal pairs); we count in **vectors**. His convention
  is stated and restated at least six times and is never silently switched
  (`verifications/closed/class-problem-upper-bound/README.md:11-14`: *"a line is an antipodal pair,
  so mind which unit is meant"*). Kissing numbers themselves are counted in points by both projects.
* His class problem forbids **both** `±16` (`|cos| ≤ 1/4`); ours forbids `+16` only. His is a strict
  restriction of ours, so our bound transfers to his by doubling.
* `RESULT task1 vector_bound=837 line_bound=418 previous_line_bound=425 improvement_lines=7`.

---

## 2. (a) Results we both have

### 2.1 `τ(27) ≥ 200540` — genuine independent co-discovery; **his documentary priority is earlier**

| | his | ours |
|---|---|---|
| value | 200540 | 200540 |
| improvement over Ma et al. | +496 | +496 |
| mechanism | the 12 cap directions are an `A₃` root system (a cuboctahedron); it admits **exactly two** partitions into four triples at pairwise `cos = −1/2`, so four classes suffice where the published configuration used five | the cuboctahedron is the disjoint union of **four** equilateral triangles, weight 8 instead of 7 |
| the two partitions | `P₁ = {ŵ₀ŵ₇ŵ₁₀, ŵ₁ŵ₆ŵ₉, ŵ₂ŵ₃ŵ₁₁, ŵ₄ŵ₅ŵ₈}`, `P₂ = {ŵ₀ŵ₈ŵ₉, ŵ₁ŵ₄ŵ₁₁, ŵ₂ŵ₅ŵ₁₀, ŵ₃ŵ₆ŵ₇}` (`paper/kissing27.tex`) | `[(0,7,10), (1,6,9), (2,3,11), (4,5,8)]` inside Ma et al.'s own directions (`06-record-and-priority.md:255-265`) |
| the optimality proposition | his Theorem 2.3(i)(ii): `m ≤ ⌊2\|W\|/3⌋ ≤ ⌊2K(k)/3⌋` | our **Proposition 5**: `w = Σ(\|T_i\|−1) ≤ ⌊2K/3⌋` |
| verification | `verify_configuration.py` (40 exact checks, no floating point, Golay *derived from the file*), `verify_exhaustive.py` (all 20 108 045 530 pairs), 15 negative controls, SHA-256 manifest, bit-for-bit rebuild from Cohn's published coordinates | `verify_dimN.py --strict` (exact casework, all `C(200540,2)` pairs accounted by class), the C++ `disjoint_family --verify`, and `docs/note/data/verify.py`; two independent families both giving 200540 |

**These are the same theorem, found twice.** His Theorem 2.3(i) and our Proposition 5 have the same
statement and the same three-line proof (`3a + 2b ≤ K`, `w = 2a + b`, so `3w ≤ 2K`), and his
`P₁` is our partition up to relabelling.

**Dates.** His `paper/kissing27.tex` carries `\date{19 August 2026}`; the package's own `CITATION.cff`
gives `date-released: "2026-08-19"`; and `KNOWLEDGE.md:129` — the first line of a working record that
begins *"Started 2026-08-19"* — already refers to *"the finished dim-27 deliverable"*. Our
`06-record-and-priority.md:413` dates the four-triangle partition in this project to **2026-08-26**.
So on the documentary record available to both parties, **his result predates ours by about a
week.** Our `docs/note/README.md` already records the substance of this: the result was
communicated to Henry Cohn, who confirmed it and reported that Kravatskiy had found the same
improvement independently at about the same time; `06-record-and-priority.md:405-420` states the
co-discovery and that neither derives from the other.

> **Action.** Nothing in our repository needs correcting — it already says the right thing. But
> `200540` must never be presented as ours alone, and where a single name is attached, it should be
> his. The honest form is: *independent co-discovery; Kravatskiy's package is dated a week earlier.*

### 2.2 `τ(25) ≥ 197058` — **his**; we defer completely

Both projects reach 197058 = `196064 + 992 + 2` by the same observation, and both describe it the
same way. His `verifications/improved/dim25-cap-level/README.md`:

> "The whole family is forced by the 60° condition, *except for one number*: the cap level `t = α²`.
> … In dimension 25 the second block is ℝ¹. There are only two directions, +1 and −1. No triple
> exists, so nothing forces `t` down to 2/3 … Raising `t` to 3/4 costs nothing and buys the two
> poles."
>
> "The value `t = 3/4` is *forced*, not chosen: `t ≥ 2/3` keeps the 496-vector head class,
> `t ≤ 3/4` keeps antipodal head sharing, and `t ≥ 3/4` is exactly what admits the poles. The three
> constraints meet in a single point."

Ours is `06-record-and-priority.md:435-461`, the "lowered lift height `h = 1`" observation, with the
same arithmetic and the same conclusion that it gives nothing for `d ≥ 2`. Our own text already
calls it *"a framework check to offer a specialist … rather than as a record claim"* and notes it is
*"very likely subsumed by unpublished work"*. It is.

His `KNOWLEDGE.md` §13, which contains it, sits in the undated 2026-08-19/20 block and is already
load-bearing in §41 (2026-08-20).

> **Action.** `197058` is Kravatskiy's. Do not claim it, do not co-claim it. If it is mentioned at
> all it should be as *"observed independently here as a framework check; the result is
> Kravatskiy's."*

### 2.3 Every two-point method gives `9360/11` — same number, different halves of the proof

| | his | ours |
|---|---|---|
| Delsarte LP | `4680/11 = 425.4545` lines, exact `Fraction`, `certificate.py` route 2, degree-4 certificate `f(t) = (4992/11)t²(t²−1/16)` | `9360/11` vectors, exact, `bounds/lp_delsarte.py`, `f_0..f_4 = 1, 24, 5083/77, 4416/11, 27600/77` |
| Hoffman ratio | `98280·20/4620`, exact Faddeev–LeVerrier on the 4×4 intersection matrix, char. poly `(x−4600)(x−1000)(x−76)(x+20)` | `196560·20/4620`, exact, on the 7-class scheme |
| Lovász `θ`, Schrijver `θ′` | both `425.4545`, from `scheme_lp.py` | both `9360/11`, exact dual certificates |
| scheme | 4-class, on the 98280 **lines**, valencies `1/4600/47104/46575` | 7-class, on the 196560 **vectors**, valencies `1/4600/47104/93150/47104/4600/1`, with `P`, `Q`, multiplicities `1,24,299,2576,17250,95680,80730` |

**The numbers agree exactly** (his degree-4 polynomial is ours divided through, and `4680/11` lines
is `9360/11` vectors). Two differences of substance:

* **He warns his own scheme code off.** `class-problem-upper-bound/README.md:79-84`: *"⚠ **Do not
  quote `scheme_lp.py`'s output.** … its eigenvalue extraction is numerically unstable … returns
  garbage … a bound of 6.22 where the true one is 425.45."* His `θ`/`θ′` row therefore rests on that
  script; only `certificate.py`'s two routes are exact. Our `θ` and `θ′` are exact.
* **We have the reason, he has the fact.** Our `02-upper-bounds.md:609-630` proves *why* the LP, `θ`,
  `θ′`, Hoffman and the lattice-free polynomial must all coincide: Hoffman follows from `aQ ≥ 0`
  alone; the LP's optimal dual *is* Hoffman's certificate; it is tight (`F(i) = −1`) on all five free
  classes, so the sign constraints are never active and `θ = θ′`; and the agreement with the
  lattice-free LP is the spherical-11-design fact. His repository records the coincidence and
  concludes correctly from it, but does not derive it.

He adds a novelty note we should carry: *"Apparently the Delsarte LP value 4680/11 had not been
recorded for this specific problem before; if so, that is a small contribution of this package."*
The published form is `⌊9360/11⌋ = 850` vectors, Cohn–Jiao–Kumar–Torquato, *Rigidity of spherical
codes*, Geom. Topol. 15 (2011) §7.

### 2.4 Maximality of the 496 in the Leech shell — same fact, very different evidence

| | his | ours |
|---|---|---|
| statement | *"Their 496 is **maximal in the Leech shell** (no minimal vector can be added), re-verified here"* (`KNOWLEDGE.md:2430-2433`) | none of the **64** plateau 496-sets extends; 0 free vertices, no vertex of tightness 1/2/3, minimum tightness exactly **4** with exactly **80** tightness-4 vertices |
| how far | asserted for the **five published classes**; minimum outside-conflict count 4, attained by 80 vectors; checked over all 95840 vectors with ≤ 11 conflicts (`KNOWLEDGE.md:528-535`) | all 196560 minimal vectors, exact integer arithmetic, **three** independent implementations (`materialize_64.py`, `check_64.py` over regenerated vectors, `verify_S.py` as a subprocess); plus 1001 distinct census sets, all with `free_total = 0` |
| reproducible from the repo? | **no** — `research/class/rigidity.py`, `lns.py` and `class248_*.npy` are not distributed; `run_all.py` runs only `certificate.py` in that package | yes; `RESULT ok=1 sets=64 free_total=0 min_tight_all=4 t123_total=0` |

**The numbers 4 and 80 agree exactly**, from two independent computations. That is a real
cross-validation of both projects and worth saying so.

### 2.5 His `R` **is** our 15 channels — confirmed, not assumed

This is the substantial finding of the audit. It was checked computationally rather than by reading,
because the two descriptions share no vocabulary.

**His statement** (`class-problem-upper-bound/README.md:134-142`, `KNOWLEDGE.md:537-546`, §20):

> "every known 496-vector class consists of 124 orthogonal **pairs** whose differences span a
> 4-dimensional totally isotropic subspace R of Λ/2Λ, so the whole class lies inside
> `X_R = { line l : b(l, R) = 0 }, b(x,y) = x·y mod 2`, which is just **6120** of the 98 280 lines —
> a 16-fold reduction."
>
> "the 124 pair-differences take only **14** distinct values in `Λ/2Λ` and span a 4-dimensional
> totally isotropic subspace `R` whose 15 non-zero elements are all norm-8 cosets"
>
> "they occupy **14 of the 15** non-zero elements of `R` (one element of `R` is unused); the
> multiplicities are **three cosets with 12 pairs and eleven with 8** (`3·12 + 11·8 = 124`); inside
> one coset the pair sums are pairwise orthogonal — they are part of that coset's **frame of 48
> norm-8 vectors (24 orthogonal pairs)**"
>
> "`X_R` splits into **2925 fibres** (cosets of `R`): **2880 of size 2 and 45 of size 8** … The
> published class uses **124 fibres of size 2 and none of size 8**."

**Our statement** (`04-structure-of-the-496.md:52-71`): the record is a union of **124 duad orbits**
`{±x, ±y}` with `x ⊥ y`; the orbit's **channel** is the common `Λ/2Λ` class of `x+y` and `x−y`; the
15 channels *"are precisely the 15 nonzero elements of a 4-dimensional subspace `T ⊂ Λ/2Λ`"*; each
`(pair, channel)` **cell** is an 8×8 **rook grid** whose rows are the 8 `±` pairs of a channel frame
in one block; `3·12 + 9·8 + 2·8 = 124` orbits, `×4 = 496`.

**The check** (`tools/structure/r_subspace_vs_channels.py`, `tools/structure/arena_is_our_sublattice.py`).
Built from scratch: a modular-HNF basis `B` of the integer lattice `L` spanned by the minimal
vectors, `|det B| = 8¹² = 2³⁶` certifying that it *is* a basis, hence honest `Λ/2Λ` coordinates; the
induced form `b(x,y) = (⟨x,y⟩/8) mod 2` (so `b = 1` exactly when `⟨x,y⟩ = ±8`); then, for a recorded
496, `W := {v : b(v, [s]) = 0 for every class [s] of the set}`. Run under **two independent HNF
bases**, since every reported quantity is basis-free.

```
RESULT basis seed=7:        |det|=2^36 -> a genuine basis of L; rank of b = 24/24
RESULT basis seed=20260901: |det|=2^36 -> a genuine basis of L; rank of b = 24/24
RESULT two-verifier: the two coordinate systems agree on every invariant for all 65 sets: True
RESULT across all 65 recorded 496s the invariant takes 1 distinct value:
    65 x  dimW=4 |W|=16 isotropic=True matching=True pairs=124 channels_used=14
          profile={8: 11, 12: 3} arena=6120 rank=20
RESULT channel identity: |x+y|^2 and |x-y|^2 in [64], [x+y]==[x-y]: True,
       equals phi(x)^phi(y): True, minimal vectors in a channel class: [0]
RESULT task2a ok=1 verdict='R (Kravatskiy) == T (our 15 channels)'
```

and, on the arena:

```
RESULT A1 dim R = 4, |X_R| = 12240 vectors = 6120 lines (Kravatskiy: 6120) -> True
RESULT A3 [Lambda : <S>] = 16 (his 16-fold reduction) -> True
RESULT A2 C ∩ <S> has 12240 vectors; equals X_R: True
RESULT A4 X_R splits into 2925 cosets of R, sizes {2: 2880, 8: 45}
          (Kravatskiy: 2925 fibres = 2880 of size 2 + 45 of size 8) -> True
RESULT A5 the record occupies 124 fibres: {2: 124}
          (Kravatskiy: 124 fibres of size 2, none of size 8) -> True
RESULT A6 |x+y|^2 over the size-2 fibres: [64] -> True
RESULT task2a-arena ok=1
```

**Verdict: the same object, stated two ways — and more than one object.** Line by line:

| his | ours | agreement |
|---|---|---|
| `R`, 4-dimensional, totally isotropic | `T`, the 4-dimensional `F₂` subspace whose nonzero elements are the 15 channels | `dim W = 4`, isotropic ✔ (the word *"totally isotropic"* appears nowhere in our docs; the property holds and is now checked) |
| 124 orthogonal pairs | 124 duad orbits `{±x, ±y}`, `x ⊥ y` | the relation "orthogonal **and** difference in `W`" is a **perfect matching** on the 248 lines — the decomposition is forced, not chosen ✔ |
| pair differences, 14 distinct values | 14 of 15 channels used, one empty (`cE`) | ✔ |
| `3 × 12 pairs + 11 × 8 pairs` | `3 × 12 + 9 × 8 + 2 × 8` orbits | **identical**; ours further resolves his "eleven with 8" into 9 channels carrying a full 8-permutation in a single block-pair and 2 carrying `4+4` |
| each nonzero `r` ↔ a frame of 48 norm-8 vectors = 24 orthogonal pairs | each channel ↔ 135 norm-64 frames of `E8`, 8 `±` pairs per block = the rook rows | ✔ (`\|x±y\|² = 64` on every one of the 124 orbits, `[x+y] = [x−y]`) |
| `X_R`, 6120 lines, 16-fold reduction | never named; but `⟨S⟩` has **index 16** in `Λ` and its minimal vectors are the **12240** monads + duads | **`X_R` = `C ∩ ⟨S⟩`, exactly** ✔ |
| 2880 fibres of size 2 | 2880 duad orbits | ✔ |
| 45 fibres of size 8 | the 720 monads (`45 = 3 blocks × 15 channels`, `45 × 8 = 360` monad lines; `12240 − 11520 = 720` monad vectors) | ✔ |
| `α(one coset) = 12`, by CP-SAT, *"not the naive frame ceiling 24"*; whole family caps at `15 × 12 = 180` pairs | independence within one channel across the three block-pairs is a **matching in `K₈,₈,₈`**, hence `≤ 12` orbits; model ceiling `15 × 12 × 4 = 720` vectors | same number; **his is a computation, ours is a proof** |

> **Credit.** This is an independent discovery on both sides, and his is the earlier one on the
> documentary record: his `KNOWLEDGE.md` §9 and §20 sit in the 2026-08-19/20 block, while our
> `04-structure-of-the-496.md` carries no internal dates and our repository is a single commit of
> 2026-08-31. **`R`, `b`, `X_R` and the 6120 should be credited to him.**
>
> **Our account is strictly finer, and this was confirmed rather than assumed.** A grep of his entire
> tree returns **zero** occurrences of *Turyn*, *PG(3,2)*, *channel*, *monad/duad/triad*, *rook*,
> *spread*, *orbital*, *coherent algebra* or *Bose–Mesner* (the only "Terwilliger" hits in his repo
> are inside scraped HTML bibliographies of Brouwer's binary-code tables). He never gives the `F₂⁴`
> labelling a projective-geometry, Steiner or lattice-decomposition reading; he uses it as a
> disjointness device (classes in different fibres are automatically disjoint) and as a coordinate
> system of 15 frame-multiplicities `m_r`. He says of the latter, `KNOWLEDGE.md:1099`: *"**Not yet
> exploited.**"*
>
> **A real gap on his side.** None of the scripts producing this material is distributed
> (`research/class/arena_mis.py`, `fib8.py`, `cliquepart.py`, `percoset.py`, `framelns.py`,
> `rigidity.py`, `lns.py`), nor the data `class248_lines_norm32.npy`. The 6120, the 2925-fibre split,
> `m = (12,12,12,8¹¹,0)` and the 160-rigidity table are **prose-only** in his repository. Everything
> in the table above is now independently reproducible from ours.

---

## 3. (b) What we have that he does not

Each item below was checked against his tree, not merely against his READMEs.

### 3.1 A three-point upper bound strictly below `9360/11` — **ours, and it contradicts a stated conclusion of his**

`|S| ≤ 837` vectors, hence `≤ 418` lines, against `425`. Established in
[`07-lines-vs-vectors.md`](07-lines-vs-vectors.md) and §6 of
[`02-upper-bounds.md`](02-upper-bounds.md).

He has a three-point bound, and it gives nothing:
`verifications/closed/class-problem-upper-bound/bv3pt.py` is the **Bachoc–Vallentin** bound in
prescribed-inner-product form for antipodal codes in `S²³` with inner products `{−1,−¼,0,¼}`, `S₃`-
and antipode-reduced to 49 realisable label triples in 8 orbits (5 free), at truncations
`d_max = 4…12`, with and without subconstituent and doubly-stabilised constraints. It returns
`M ≤ 9360/11 → N ≤ 425.4545` with the `Y_k` positivity never active, and he concludes:

> "**Conclusion: closing the 248–425 gap needs a four-point / Lasserre-3 relaxation, or a
> lattice-structural argument. A three-point bound cannot do it.**"

**That conclusion is correct for the relaxation he ran and wrong as stated in general.** His is a
*lattice-free spherical* relaxation: Gegenbauer polynomials on `S²³`, symmetry-reduced by `S₃` on
edge labels, knowing nothing about the Leech lattice. Ours is the **Schrijver/Terwilliger SDP over
the full 148-dimensional centraliser algebra of `Stab(x₀)` in `Co₀`** — 43 variables after `S₃`
symmetrisation, two 148×148 PSD blocks, exact rational dual certificate. Its content beyond the
spherical bound is precisely the lattice fact that an orthogonal triple of minimal vectors comes in
two `Co₀`-inequivalent flavours, `p³₃₃ = 43164 = 42240 + 924`, which no polynomial three-point bound
on the sphere can see. **It is the "lattice-structural argument" he names, delivered as a three-point
relaxation.** It should be presented to him as a refinement of his own conclusion, not a
contradiction of it.

Confirmed absent from his tree: no Terwilliger algebra, no orbital algebra, no coherent
configuration, no association-scheme SDP of any kind. The only SDP in the entire repository is
`bv3pt.py`.

### 3.2 `ω(G) = 24`, and clique cuts are provably inert

`ω(G) = 24` exactly: upper bound with no computation (`k` vectors pairwise at `+16` have Gram
`16(I_k + J_k)`, positive definite of rank `k`, so `k ≤ 24`), lower bound by an explicit certified
24-clique. Complete maximal-clique census: sizes exactly `{8, 12, 15, 17, 23, 24}`; 5 028 032 232
maximal cliques of `G[N(0,1)]`; 849 139 200 000 24-cliques in `G`.

Two families of lifted, pair-conditioned clique inequalities were derived and fed to the SDP:

```
baseline (no cuts)        cuts   0   exact 837.535866  floor 837
+ triangle cuts           cuts 437   exact 837.534915  floor 837
+ omega-clique cuts       cuts 225   exact 837.534727  floor 837
+ triangle + omega cuts   cuts 662   exact 837.536877  floor 837
```

Zero movement, and the reason is quantified: the optimum behaves like a near-product pseudo-density
with `σ ≈ 1/235`, so a clique cut binds only when `|K| ≳ 235`, and `ω = 24` is an order of magnitude
short. Nothing comparable exists in his repository — his clique work is CP-SAT maximum-independent-set
solving on much smaller derived graphs.

### 3.3 The 64 certified plateau sets, and the plateau as channel rewiring

Six pairwise-disjoint, negation-closed, commuting plateau atoms — four of size `(12,12)` and two of
size `(80,80)` — generating a 6-cube of `2⁶ = 64` distinct 496-element sets, all with identical Gram
histogram, `0` free vertices, no tightness 1/2/3, and minimum tightness exactly `4` with `80`
tightness-4 vertices. The 64 are pairwise `Co₀`-inequivalent and fall into 8 isometry classes of 8
and 5 tightness-fingerprint classes of sizes `16/16/16/8/8`. Census-scale: 1001 distinct sets in 113
families, all in the same 5 fingerprint classes.

Read in the channel model: the four `(12,12)` atoms are alternating 6-cycle rotations inside channel
`cF2`; the two `(80,80)` atoms are complete rewirings of a `σ`-pair of channels
(`σ : c ↦ c ⊕ cF2`, pairs `{cF0, c07}` and `{cF1, c04}`, with `cF0 ⊕ c07 = cF2` exactly). The
deliverable statement:

> *The cell profile `(4,4,4 / one-pair 8-permutations / 4+4)` is an invariant of the entire
> 64-plateau, not merely of `S`: local search never changes how much each cell carries, only which
> matching realises it. **Any >496 duad set must therefore break the profile itself.***

**This is the sharpest structural constraint either project has on a 249th line, and it is only
ours.** He looked for alternative maxima and did not find or count any; his repository's language for
the class problem is "rigid", "stuck", "does not move" — the word *plateau* never appears in that
context.

**And the two rigidity results agree rather than conflict.** His §54.7 proves by large-neighbourhood
search with CP-SAT that *"248 is optimal among all classes sharing at least 88 lines with the
record"*. Our plateau moves change at most `4×12 + 2×80 = 208` vectors `= 104` lines, so every one of
our 64 sets shares at least `144 ≥ 88` lines with the record. **Our whole plateau lies strictly
inside the region he proved rigid** — which is exactly why his search found nothing there, and why
its structure, not its existence, is the interesting part.

### 3.4 A proof where he has a computation

His CP-SAT result `α(one coset) = 12` — *"not the naive frame ceiling 24"* — is our `K₈,₈,₈`
matching bound, which is a two-line argument once the rook-grid model is set up. Similarly his
`α(any two cosets) = 24 = 12 + 12`, *"for all 105 coset pairs, so the subset-LP hierarchy gives
nothing at level 2"*, is explained by our edge census: cross-pair cross-channel conflicts number
**zero**.

### 3.5 The four-point scoping

`02-upper-bounds.md` §9 is an honest negative deliverable: `dim T(x, y_i)` per pair class
(`148, 1893, [2971,2972], [4097,4107], …`), `Co₀`-orbits of ordered 4-tuples in `[14121, 14133]`,
6183 admissible quadruple orbits, **17 PSD blocks with the largest `4107 × 4107`**, `Σd² = 1.1×10⁸`,
≈ 7 GB of solver iterates, hours-to-days per solve, and an expected landing zone of ≈ 820–830. No
bound was obtained and the report says so. He names the four-point/Lasserre-3 level as the frontier
in two places but has not scoped or attempted it.

### 3.6 Complete structural anatomy of the record

The Turyn `√2E8³` decomposition and `Stab_Co₀(S) = 2³`; `Λ ∩ Bᵢ ≅ √2E8` (verified, unconditional);
`Λ ∩ (Bⱼ ⊕ Bₖ) = BW16`; monads/duads/triads with the census `720 / 11520 / 184320` and the
`208/144/144` split of `S`; the three frames as 24-cliques of the orthogonality graph; the octad
incidence refuting Golay; the 80 tightness-4 vertices as the complementary `cF2` matching; and the
list of refuted structural hypotheses. None of this appears in his repository in any form.

---

## 4. (c) What he has that we have not touched

### 4.1 The antipode construction — *and a correction to our briefing*

`verifications/closed/antipode-construction-ceiling/`. The construction is Sun–Wang's and Chen et
al.'s (arXiv:2607.20359): for a lattice `L` of minimum `μ`, a rank-`k` cross-section with Gram `K`,
`M = K⁻¹`, choose finite `S ⊂ M` with maximal pairwise squared distance `δ < μ`, and count
`N_i = Σ_{j : |u_i−u_j|² = δ} fibre(u_j − u_i)`.

His **Theorem** is a proved equality over the entire admissible search space of `S`:

```
max_S max_i N_i  =  max over delta in (0, mu) of MaxWeightClique(V_delta, E_delta, fibre)
```

with `V_δ = {v ∈ M : |v|² = δ, fibre(v) > 0}` and `v ~ v'` iff `⟨v,v'⟩ ≥ δ/2`. Two proved corollaries
follow: all arms lie pairwise within 60°, so the total is bounded by the heaviest 60° cap's fibre
mass — *"it can never reach even half the shell"*; and at minimal `δ` the clique is equilateral with
Gram `(δ/2)(I+J)`, so it has rank `≤ k`, which is exactly Chen et al.'s dual-basis simplex regime.

Measured over the Leech at `k = 1…12` the best star always loses (dims 12–23; e.g. dim 20: best star
7040 against `Λ₂₀ = 17400` and the record 19448). Verdict: *"the antipode is a ~10% lever that only
pays when the bare cross-section is already within about 10% of the target"*.

> **Correction.** The briefing that set this audit says *"we had the antipode construction as a live
> candidate; he has closed it."* **We did not.** A grep of our entire repository finds "antipode
> construction" exactly once, as a bibliographic entry for Sun–Wang's paper in the arXiv sweep of the
> priority check (`06-record-and-priority.md:189`), plus two references to it as third-party work
> outside our range. We never proposed, evaluated or attempted it. There is nothing of ours for his
> closure to close — but his reduction is the right thing to read before anyone starts.

Two caveats on his closure, both his own or found here: the structural half is unconditional, but the
losing table is a **heuristic** search (randomised greedy + 1-swap clique local search, 250 restarts,
greedy randomised cross-section chains, candidate set truncated to the 700 heaviest fibres, Leech
only). And an earlier "30° cap" statement is corrected to 60° in that README, but the stale 30°
survives at `README.md:214`, `KNOWLEDGE.md:3596` and in `antipode3.py`'s docstring. **The antipode is
explicitly NOT closed in dimensions ~44–47**, where Sun–Wang win with it — his own
`superseded/dim44-45-p48-cross-sections/` records their antipode packing at 7 379 838 in dimension 45.

### 4.2 Dimensions 38–96: three construction families entirely outside our work

* **Edel–Rains–Sloane chains.** `improved/dim39-ers-constant-weight/` (`τ₃₉ ≥ 756116`, +128, the
  entire gain being one constant-weight-code table entry `A(39,8,8) = 3324`);
  `improved/dim62-63-ers-chain/` (`71 310 732` and `138 419 844`, factors 1.36 and 2.64, with the
  levels above 0 *built, not cited*); `improved/dim96-ers-takeover/` → `12 886 999 232`, factor 2.07;
  and the `closed/dim32-44-ers-audit/`, which re-derives Cohn's table across 32–44 and finds four
  entries stale (32: 346 432; 33: 362 048; 34: 381 124; 37: 496 232).
* **`P₄₈` caps.** `improved/dim49-63-p48-caps/`, dims 49–61, class threshold `γ = 1/3` forcing
  `t ≥ 3/4` and therefore **pairs, never triples** — 2 points per line. Base class 7069 lines of
  `P₄₈`'s 26 208 000 minimal lines against Caro–Wei's 712.
* **`Γ₇₂` caps and cross-sections.** `improved/dim73-95-gamma72-caps/` (23 dimensions; `γ = 1/4`, so
  triples, 4 points per line; classes of 2106–2118 lines) and
  `improved/dim68-71-gamma72-cross-sections/` (`τ(71) ≥ 2 603 658 750`, factor 7.85, from Venkov's
  11-design theorem and an over-determined moment system; `τ(70) ≥ 1 249 778 250`;
  `τ(69) ≥ 627 822 180`; `τ(68) ≥ 361 275 480`).

### 4.3 The `k`-point moment LP — a whole methodology we do not have

`common/kpoint_lp.py`, `common/exact_vertex.py`, `common/PROOF-kpoint.md`. It is a Delsarte-style LP
over the **cell counts of a `k`-point inner-product distribution inside a lattice's minimal shell**,
constrained by Venkov's 11-design moment identities (including the **mixed** moments — omitting them
was a real bug that produced 6 462 480 instead of 6 687 320), lattice translation constraints, and a
lattice-range constraint strictly stronger than positive semidefiniteness. **Purely linear; no
semidefinite constraint anywhere.** Exact rational dual certificates; `exact_vertex.py` reads the
active set off the numeric dual, solves in `Fraction`s, and verifies `Aᵀy ≤ c` over every column.

Validated against seven independently known values (`E₈ → E₇ = 126`, `Leech → Λ₂₃ = 93150`,
`P₄₈ → 23 766 960`, Leech orthogonal `43164` and 60° `49896`, `E₈` orthogonal `60` and 60° `72`), and
it re-derives **Ozeki's Theorem 6.3** as a by-product. `common/validate_lp.py` is an adversarial suite
of 18 cases whose whole purpose is that an *invalid* constraint would make the bound too large.

### 4.4 `closed/` in general — nine more mechanisms pushed to their ceiling

Dims 9–19 record maximality (no published record admits a free point; margins down to 6.9% at dims 14
and 15); the Cohn–Li mechanism in 17–23 (exact Delsarte/`θ` ceilings 768, 2048, 2048 against 1155,
2049, 2049 needed; dim 19 closed at Ho's value by CP-SAT with girth-5 cuts); the dim-17 layered family
(`τ(17) = 5730` for that family, by a six-rational-number Delsarte dual verified with one
Walsh–Hadamard transform and no floating point); dims 22–23 maximal cross-sections (Hall + König give
exactly 93150 — *"this killed the most promising untried idea in the project"*); the dim-37 cap
shortfall; and the dim-69 three-point retraction, kept as *"the record of how a sufficient condition
for realizability was mistaken for a necessary one."*

### 4.5 Things in the class problem itself that he has and we do not

* **The arena as a search device.** ILS inside `X_R` returns 230–244 lines where the same search on
  the full graph returns 152 and random greedy 113. **We have the subspace; we never used it to
  restrict a search.**
* **The `m_r` coordinate system.** *"a class is exactly a choice, for each `r ∈ R \ {0}`, of `m_r` of
  the 24 orthogonal pairs of the frame `F_r`, subject to the class condition across cosets … **This
  is the right coordinate system in which to search for a class of more than 248 lines**: 15 small
  choices of `m_r ≤ 24` inside frames, rather than 248 lines out of 6120. Not yet exploited."*
  Combined with our cell-profile invariant this is the most promising untried construction lead
  either project has.
* **160-rigidity.** LNS + exact CP-SAT completion over all 98280 lines: deleting up to 160 lines from
  the record and re-optimising still returns 248, proved optimal. **Deeper in reach than our
  `k ≤ 12`-vector swap exhaustion**, though unreproducible from his repository.
* **Disjoint class families.** 46 greedy classes covering the whole arena; the 644-class partition of
  all 98280 lines that makes dimension 38 work. Our largest disjoint family is 42 (the `Co₀` chain
  construction), and it is aimed at a different question.
* **Free cap heads from second-shell vectors** (`KNOWLEDGE.md` §14, §76) — ruled out with an exact
  Delsarte bound of 280 against the 323 that dimension 29 would need, plus a frame reduction giving
  the same 280 from the Hamming-scheme bound `A(24,10) ≤ 280`. The largest such set he builds is 64.

---

## 5. (d) What to stop working on

**Stop — done elsewhere, and better or first:**

| stop | why | his location |
|---|---|---|
| claiming `τ(25) ≥ 197058` | his, and earlier | `improved/dim25-cap-level/` |
| presenting `τ(27) ≥ 200540` as ours alone | independent co-discovery; his package is dated a week earlier | `improved/dim27-triple-partition/` |
| any further two-point method on the class problem | LP = `θ` = `θ′` = Hoffman = `9360/11`, both sides, exactly; we additionally proved they must coincide | `closed/class-problem-upper-bound/certificate.py` |
| the Bachoc–Vallentin (lattice-free) three-point bound | done, at `d_max` 4–12, with and without subconstituent and doubly-stabilised constraints; returns the two-point value | `closed/class-problem-upper-bound/bv3pt.py` |
| subconstituent splits of the class problem | `1 + 267.64 + 352.11 = 620.7`, far worse than 425 | same README |
| the antipode construction in low dimensions | reduced to max-weight clique and shown to lose by 8–36%; also, we never started | `closed/antipode-construction-ceiling/` |
| free-point extension of records in dims 9–23 | none exists; margins measured exactly | `closed/dim09-19-record-maximality/`, `closed/dim22-23-maximal-cross-sections/` |
| unions of Leech level sets in dim 23 | Hall + König: exactly 93150 | `closed/dim22-23-maximal-cross-sections/` |
| the Cohn–Li mechanism, dims 17–23 | exact ceilings, all short | `closed/dim17-23-cohn-li-mechanism/` |
| Edel–Rains–Sloane in dims 32–44 | audited to the current code tables | `closed/dim32-44-ers-audit/` |
| the Leech cap construction at codimension 13 (dim 37) | 475 336 against 496 232; needs a better *class*, not a better cover | `closed/dim37-cap-shortfall/` |
| anything in dimensions 38–96 | five construction families, 46 dimensions, all his | `improved/` |
| re-deriving `R`, `X_R` or the 6120 as new | his, and earlier | `KNOWLEDGE.md` §9, §15, §20 |

**Keep — genuinely ours, or genuinely open:**

* The **four-point / Lasserre-3 relaxation**. Both projects independently name it as the only route
  left to the upper bound, and only we have the lattice-aware three-point machinery and the scoping
  to build on. Expected landing ≈ 820–830.
* The **cell-profile invariant** as the obstruction to a 249th line, now to be combined with his
  `m_r` coordinate system — the one concrete construction lead neither project has exploited.
* **The codimension-4 step of the lifting template** (the 24-cell, `K(4) = 24`). Untouched in his
  repository and in ours; recorded separately.
* The **`|S| ≤ 837` line of work** generally, including the `±16` variant, which is now certified.

---

## 6. Corrections to the briefing that set this audit

Reported as deviations, per standing practice.

1. **"we had the antipode construction as a live candidate"** — we did not; see §4.1. The phrase
   occurs once in our repository, as a citation of Sun–Wang.
2. **"his 124 orthogonal pairs … versus our 15 channels — very likely the same"** — confirmed, and
   the correspondence is richer than expected: his arena `X_R` is also our sublattice `⟨S⟩`, and his
   fibre split `2880 + 45` is our duads and monads. See §2.5.
3. **"our account … appears to be strictly more detailed"** — confirmed by exhaustive grep of his
   tree, not assumed. But he has things in this area we do not (§4.5), so "more detailed" is not
   "superset".
4. **Our own `04-structure-of-the-496.md` nowhere uses the words "totally isotropic".** The property
   holds — `b(u,v) = 0` for all `u, v ∈ T` — and is verified here for the first time. Worth adding to
   that report.
5. **His `dim27-triple-partition/paper/kissing27.tex` and our `docs/note/kissing27.tex` are different
   documents with the same filename, the same result and different authors.** Anyone diffing the two
   repositories will hit this; it is a coincidence of naming, not a shared source.
6. **A count in our own tooling was inflated by a zeroed diagonal** in an exploratory script: the
   orthogonal line pairs inside the record number **11876**, not 12000. Fixed; the shipped tools are
   correct.

---

## 7. Verification and reproduction

Two verifiers for the §2.5 identification, both exact, per the standing rule.

* **Verifier 1 — basis independence.** `tools/structure/r_subspace_vs_channels.py` runs the whole
  computation under two independent modular-HNF bases of `L` (seeds 7 and 20260901), each certified a
  genuine basis by `|det| = 2³⁶`. Every reported quantity is basis-free, so the two runs must agree;
  they do, on all 65 recorded 496s (the record plus the 64 plateau sets).
* **Verifier 2 — a different definition of the same object.** `tools/structure/arena_is_our_sublattice.py`
  reaches `X_R` from the other side, as `C ∩ ⟨S⟩` with `⟨S⟩` built by an independent HNF of the span
  of the record, and confirms `[Λ : ⟨S⟩] = 16` and set equality with the `b`-perp definition.

```
.venv/bin/python tools/structure/r_subspace_vs_channels.py     # ~6 min, 65 sets x 2 bases
.venv/bin/python tools/structure/arena_is_our_sublattice.py    # ~1 min
```

Neither reads `data/leech_min.*` or any C++ output; both regenerate the 196560 minimal vectors from
the Golay code via `kiss_ref`.

**Provenance.** All statements attributed to Kravatskiy are quoted from
`/…/scratchpad/kravatsky` at commit `0137e87`. That clone is read-only working material; it is not
vendored into this repository and no file of his is redistributed here.

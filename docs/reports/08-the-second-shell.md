# The second shell of the Leech lattice

The 196560 minimal vectors are the first shell. This note records what we know about the
**second** shell — the 16 773 120 vectors of norm 6 — and in particular about the minimal
vectors that are *extremal* against it. The results are self-contained facts about `Λ`; they
were computed in the course of other work and are collected here because they are of
independent interest.

## Contents

1. [The shell, generated and verified](#1-the-shell-generated-and-verified)
2. [Extremal minimal vectors, and the number 33](#2-extremal-minimal-vectors-and-the-number-33)
3. [Higher intersections](#3-higher-intersections)
4. [Two flavours of triple](#4-two-flavours-of-triple)
5. [The 24-cell and its zero-sum triples](#5-the-24-cell-and-its-zero-sum-triples)
6. [Files and reproduction](#6-files-and-reproduction)

---

## 1. The shell, generated and verified

In the `√8`-integer scaling (minimal norm 32), a norm-6 vector has squared norm 48. The shell
is generated from the Golay code in four shapes, each candidate then put through the
independent membership test of [`01-lattice-and-verification.md`](01-lattice-and-verification.md):

```
(+-2^12, 0^12)      on a DODECAD, even number of minus signs      2576 * 2048 = 5275648
(+-4, +-2^8, 0^15)  the +-2s on an OCTAD, the +-4 off it           759 * 4096 = 3108864
(-+5, +-1^23)       base 5 at one place, signs on a codeword        24 * 4096 =   98304
(-+3^3, +-1^21)     base -3 on three places, ditto                2024 * 4096 = 8290304
                                                                  ----------------------
                                                                             16773120
```

The total matches the theta series of `Λ` exactly, which is the check that the shape analysis
is complete. Writing `x = 2z` for the even-parity shapes, `|z|² = 12` and `{i : z_i odd}` must
be a codeword, which is what forces the dodecad and octad supports and excludes `2^3` (whose
coordinate sum is never `0 mod 8`).

**The inner-product spectrum.** For `v` in the shell, over the whole shell:

```
<v,w> (sqrt-8) : -48    -32     -24      -16      -8       0       8      16      24     32   48
count          :   1  11178  257600  1536975 3934656 5292300 3934656 1536975  257600  11178   1
standard       :  -6     -4      -3       -2      -1       0       1       2       3     4    6
```

**There is no `±5`.** `|v − w|² = 12 − 2⟨v,w⟩` must be `0` or a Leech norm `≥ 4`, so
`⟨v,w⟩ = 5` would give norm 2 — impossible. The link `L(v) = {w : ⟨v,w⟩ = 3}` has
**257600** members.

## 2. Extremal minimal vectors, and the number 33

For `v` of norm 6 and `u` minimal, `|v − u|² = 10 − 2⟨u,v⟩` lies in `Λ`, so it is `0` or
`≥ 4`; hence `|⟨u,v⟩| ≤ 3`, and 3 is attained. Call

```
A(v) = { u minimal : <u,v> = -3 }
```

the minimal vectors **extremal** against `v`; `|A(v)| = 552`.

> **Theorem.** For every pair `v, w` of norm-6 vectors with `⟨v,w⟩ = 3`,
> `|A(v) ∩ A(w)| = **33**`.

*Proof.* Write `B(u) = {v of norm 6 : ⟨v,u⟩ = −3}` for `u` minimal, `|B(u)| = 47104`, and let
`P` be the set of ordered pairs of norm-6 vectors at inner product 3,
`|P| = 16773120 × 257600`. Exchanging the order of summation twice,

```
E := sum_{(v,w) in P} |A(v) n A(w)|    = sum_u     #{(v,w) in B(u)^2        : <v,w> = 3}
Q := sum_{(v,w) in P} |A(v) n A(w)|^2  = sum_{u,u'} #{(v,w) in (B(u) n B(u'))^2 : <v,w> = 3}
```

`Γ` is transitive on the 196560 minimal vectors, so the inner sum of `E` is a constant `N` and
`E = 196560·N`; and `Γ` has exactly **seven** orbitals on ordered pairs of minimal vectors, so
the inner sum of `Q` depends only on the class of `(u,u')` and
`Q = Σ_c 196560·v_c·M_c`. Both group facts are verified computationally in
[`02-upper-bounds.md`](02-upper-bounds.md) §4, not cited. Hence `E` and `Q` follow from **one**
minimal vector and **seven** representative pairs:

```
N = 725401600
class c        (<u,u'> = -32, -16, -8,  0,     8,       16,        32)
|B(u) n B(u')| =    0,    0,   1,  0,   275,   2816,    47104
M_c            =    0,    0,   0,  0, 30800, 4730880, 725401600

|P| = 4320755712000     E = 142584938496000     Q = 4705302970368000
mean = E/|P| = 33 exactly            variance = Q/|P| - mean^2 = 0 exactly
```

A variance of zero forces the count to be constant, and the constant is the mean. ∎

Independently, eight **complete stars** were computed — for each of eight base vectors `v`, the
value for *every one* of the 257600 `w` in its link, 2 060 800 ordered pairs — and every one is
33.

## 3. Higher intersections

For `v_1 … v_k` of norm 6, pairwise at inner product 3, the Gram is `3(I_k + J_k)` and
`(I_k + J_k)^{-1} = I − J/(k+1)`. So every `u ∈ A(v_1) ∩ … ∩ A(v_k)` has the **same**
projection

```
p = -(v_1 + ... + v_k)/(k+1),   |p|^2 = 3k/(k+1),   |u - p|^2 = 4 - 3k/(k+1),
```

and `⟨u − p, u' − p⟩ = ⟨u,u'⟩ − 3k/(k+1)`. Cauchy–Schwarz pins `⟨u,u'⟩` to a short list, and
`0 ≤ |Σ(u − p)|²` bounds the size:

| `k` | `\|p\|²` | `\|u−p\|²` | admissible `⟨u,u'⟩` | max `⟨u−p,u'−p⟩` | bound | measured max |
|---|---|---|---|---|---|---|
| 2 | 2 | 2 | `{0,1,2}` | 0 | none | 33 |
| 3 | 9/4 | 7/4 | `{1,2}` | −1/4 | **≤ 8** | 6 |
| 4 | 12/5 | 8/5 | `{1,2}` | −2/5 | **≤ 5** | 3 |
| 5 | 5/2 | 3/2 | `{1,2}` | −1/2 | **≤ 4** | 3 |
| 6 | 18/7 | 10/7 | `{2}` | −4/7 | **≤ 3** | — |

The `k = 2` row is vacuous — the maximum cross term is exactly 0 — which is why the pair count
needed the moment argument of §2.

## 4. Two flavours of triple

Over **all** 32571 completions `v_3` of a fixed pair `(v_1, v_2)`, the triple intersection
takes exactly two values, and they are separated cleanly by the size of the triple's common
second-shell link:

```
(|A_1 n A_2 n A_3|, |L(v_1) n L(v_2) n L(v_3)|) = {(4, 7780): 495, (6, 7966): 32076}
```

The rare flavour is 1.52% of completions. Neither the saturation index of `⟨v_1,v_2,v_3⟩` in
`Λ` (always `(1,1,1)`) nor the `Λ/2Λ` class of `v_1+v_2+v_3` distinguishes them.

This is the exact analogue, one shell up, of the fact that drives our three-point bound: the
`Co₀`-orbit split `p³₃₃ = 43164 = 42240 + 924` of *orthogonal* triples of minimal vectors
([`02-upper-bounds.md`](02-upper-bounds.md) §6.1). In both cases a configuration that looks
homogeneous from the two-point data splits into two flavours at three points.

## 5. The 24-cell and its zero-sum triples

[`05-lifting-template-and-families.md`](05-lifting-template-and-families.md) records the
maximum number of disjoint equilateral triangles in the root systems `A₂ … E₇` as
`2, 4, 8, 12, 24, 42`. The `d = 4` entry is worked out in full here.

* Exactly **32** triples of `D₄` roots are pairwise at cosine `−1/2`, and all 32 are
  **zero-sum**; the two conditions coincide, proved both ways and checked exhaustively
  (`|a+b+c|² = 3·2 + 2(−3) = 0`; conversely `a+b+c = 0` forces all three inner products to
  `−1`).
* **No four** roots are pairwise at cosine `≤ −1/2` (`|a+b+c+d|² ≤ 8 − 12 < 0`), so a group has
  at most three members — Proposition 5 of [`docs/note/kissing27.tex`](../note/kissing27.tex).
* The maximum number of disjoint such triples is **8** = `⌊24/3⌋`, with no leftover, and there
  are exactly **40** perfect partitions, counted twice by independent algorithms (DFS exact
  cover, and brute force over all `C(32,8) = 10 518 300` subsets). Under the order-384 signed
  permutation group they form two orbits, of sizes 8 and 32.
* The maximum weight `Σ(|T_i| − 1)` is **16 = ⌊2·24/3⌋**, attained by 8 triangles and 0 pairs.
* Proposition 6 (zero-sum) permits this: it forbids a perfect triangle partition only when
  `|T| ≡ 1 (mod 3)`, and `24 ≡ 0`. Contrast `d = 5`, where `K(5) = 40 ≡ 1` and it does bite,
  capping the packing at 12.

## 5b. The largest regular simplex of edge `\sqrt6`

**The maximum number of norm-6 vectors of `Λ` that are pairwise at inner product 3 is exactly
24, and it is attained.** Their pairwise differences then also have norm 6, so
`{0, v_1, …, v_k}` is a set of `k+1` lattice points at mutual squared distance 6, spanning a
copy of `√3·A_k`.

*Upper bound, with no computation.* The Gram is `3(I_k + J_k)`, with eigenvalues `3(k+1)` once
and `3` with multiplicity `k−1`: positive definite, so the vectors are linearly independent and
`k ≤ dim ℝ²⁴ = 24`. *Lower bound:* an explicit 24-system, stored in
[`data/norm6_simplex24.json`](../../data/norm6_simplex24.json); randomised greedy in the link of
one vector finds one in seconds, so the configuration is common rather than rare.

```
RESULT dim28-system k=24 gram_is_24(I+J)=True all_in_Leech=True rank=24 link_size=257600
RESULT dim28-upper-bound eigenvalues=[24, 600] -> positive definite, rank k, so k <= 24.
       MAXIMUM = 24, ATTAINED: True
```

## 6. Files and reproduction

`second_shell_intersections.py` and `second_shell_stars.py` read the shell, so run
`norm6_shell.py` first; it writes a 384 MiB cache to `.cache/leech_norm48.npy` (override with
`$LEECH_NORM48_CACHE`). The build streams each shape through the membership test into a
memory-mapped array, so the peak working set is ~1.2 GiB rather than the whole shell plus its
`int64` intermediates.

| file | role |
|---|---|
| [`tools/leech/norm6_shell.py`](../../tools/leech/norm6_shell.py) | the second shell, generated from the Golay code and verified |
| [`tools/leech/second_shell_intersections.py`](../../tools/leech/second_shell_intersections.py) | §2 (the theorem and its proof), §3, §4 |
| [`tools/leech/second_shell_stars.py`](../../tools/leech/second_shell_stars.py) | §2, by exhaustion: eight complete stars, 2 060 800 ordered pairs |
| [`tools/leech/d4_triangle_partitions.py`](../../tools/leech/d4_triangle_partitions.py) | §5 |
| [`tools/leech/norm6_simplex.py`](../../tools/leech/norm6_simplex.py) | §5b, and the `A(v)` intersection data |

```
.venv/bin/python tools/leech/norm6_shell.py                    #  9 s, peak 1.2 GiB
.venv/bin/python tools/leech/second_shell_intersections.py     # 74 s, peak 1.2 GiB
.venv/bin/python tools/leech/second_shell_stars.py             # 20 s, peak 1.3 GiB
PYTHONPATH=python .venv/bin/python tools/leech/d4_triangle_partitions.py   #  3 s, peak 1.2 GiB
```

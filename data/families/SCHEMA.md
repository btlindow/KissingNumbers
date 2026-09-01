# `data/families/` — disjoint families S_1..S_k for dimensions 25–31 (schema v1)

Owner: T4.2 (this file, the extractor, the search tool). Consumer: T4.3 (`python/verify_dimN.py`).
Background: README §1.3 (CJKT template + PackingStar's extra spheres), docs/design.md §2.2 (S text format).

## 1. Directory layout

```
data/families/dim<n>/family.json       n = 25..31 — the record families (PackingStar's data, converted; T4.2 baseline)
data/families/dim<n>/S_01.txt ...       one S text file per set (docs/design.md §2.2: 24 ints per line, '#' comments)
data/families/ours/dim<n>/family.json  n = 26..31 — our own families (tools/disjoint_family --method chain), same format
data/families/ours/dim31/g.txt         the automorphism g (kiss/group.h save_aut format: 'den D' + 24 rows) with S_i = g^{i-1}·S496
```

Both trees are complete families in the same schema; a verifier takes any of the directories.

`S_i.txt` files use the √8-integer scaling (squared norm 32) in **our** Golay labelling (the same
coordinates as `data/S496.txt`; external data goes through `data/external/coordinate_map.json`).
Row order inside a file is free. Files referenced by a `family.json` may live in another family
directory (nested families, see §4): paths are resolved relative to the directory containing the
`family.json` that references them.

## 2. `family.json`

All numbers are JSON integers (never floats): the R^d half of the template is stored **exactly**.

```jsonc
{
  "schema_version": 1,
  "dim": 31,                       // n = 24 + d
  "d": 7,
  "coordinates": "leech-sqrt8-integer",   // constant; S_i files: norm 32, our labelling
  "sets": [                        // k entries, i = 1..k in order; S_i ↔ T.groups[i-1]
    {"file": "S_01.txt", "size": 496, "sha256": "<hex of the file bytes>"},   // sha256 optional
    ...
  ],
  "T": {                           // the K(d)-point kissing configuration in R^d, exact
    "config": "E7",                // label only: A1 | A2 | A3 | D4 | D5 | E6 | E7
    "K": 126,                      // number of rows of "vectors" (= K(d))
    "ambient": 8,                  // m: coordinates per vector (m >= d; rows span a d-dim subspace)
    "norm2": 8,                    // every row has this squared norm (integer)
    "vectors": [[2,-2,0,0,0,0,0,0], ...],    // K rows of m integers
    "groups": [[0,1,2], [3,4,5], ..., [7,9]] // groups[i] = T_{i+1}: indices into "vectors", size 2 or 3,
                                             // pairwise cos <= -1/2 inside a group
  },
  "extra": {                       // PackingStar's extra spheres (0, 2y'), same ambient coordinates as T
    "count": 126,                  // rows of "a"/"b"; 0 allowed (then "a","b" are [])
    "sqrt": 2,                     // M >= 1; M = 1 means all coordinates are rational
    "den": 4,                      // D >= 1
    "a": [[...], ...],             // count rows of m integers
    "b": [[...], ...]              // count rows of m integers;   y'_j = (a_j + b_j * sqrt(M)) / D
  },
  "count": 238350,                 // = count_terms.extra + 196560 + count_terms.lifted
  "count_terms": {"extra": 126, "leech": 196560, "lifted": 41664},
  "count_formula": "126 + 196560 + sum_i (|T_i| - 1) * |S_i|",
  "provenance": { ... }            // free-form: source files, sha256s, tool, seed, date
}
```

`count_terms.lifted = Σ_i (|T.groups[i]| − 1) · sets[i].size`. Every `S_i` gets exactly one group;
`len(sets) == len(T.groups)`. Vectors of `T.vectors` not used by any group are allowed (they are still
part of the kissing configuration and must satisfy the cos ≤ 1/2 condition with everything).

Model conventions used by our files (any integer model with the right Gram matrix is legal):
A1 = {±1} in Z^1 (norm 1); A2 = permutations of (1,−1,0) in Z^3 (sum 0, norm 2); A3 = D3 roots
(±1,±1,0) (norm 2); D4, D5 = roots (±1,±1,0..) (norm 2); E7 = E8 roots ⊥ (2,−2,0^6) (norm 8),
E6 = E8 roots ⊥ (2,−2,0^6) and (0,2,−2,0^5), where the E8 roots are (±2,±2,0^6) and (±1)^8 with an
even number of minus signs. These are the models of `python/kiss_ref/small_kissing.py` (T4.3).

## 3. What a verifier must check (all exact; then the float GEMM of the assembled configuration)

Notation: N = `T.norm2`, t, t' rows of `T.vectors`, y' = (a + b√M)/D rows of `extra`.

1. Sets: each `S_i` file parses, every row is a Leech minimal vector (norm 32, membership), rows
   distinct, all off-diagonal Gram entries ≤ 8 (independent); `size` matches; sets pairwise disjoint.
2. T configuration: every row has squared norm N; pairwise distinct directions; for all pairs
   2⟨t,t'⟩ ≤ N (cos ≤ 1/2).
3. Groups: within a group every pair satisfies 2⟨t,t'⟩ ≤ −N (cos ≤ −1/2); a size-3 group is then
   automatically an equilateral triangle (cos = −1/2 exactly); a size-2 group is an antipodal pair
   (cos = −1) or two vertices of a triangle (cos = −1/2) — PackingStar's 29D family uses the latter;
   groups pairwise disjoint; `len(groups) == len(sets)`.
4. Extra spheres (exact arithmetic in Q(√M); integers only):
   * norm: |a|² + M·|b|² = D²·N and ⟨a,b⟩ = 0  (so |y'|² = N exactly);
   * extra–extra, for j ≠ j': with P = ⟨a,a'⟩ + M⟨b,b'⟩ and Q = ⟨a,b'⟩ + ⟨b,a'⟩ (so ⟨y',y''⟩ = (P + Q√M)/D²),
     require cos ≤ 1/2 ⇔ 2(P + Q√M) ≤ N·D² ⇔ sign((2P − N·D²) + 2Q·√M) ≤ 0, and distinct directions
     (not P + Q√M = N·D²);
   * extra–T: with p = ⟨a,t⟩, q = ⟨b,t⟩ (⟨y',t⟩ = (p + q√M)/D), require cos ≤ √3/2 ⇔
     2(p + q√M) ≤ √3·N·D ⇔ [ 2(p+q√M) ≤ 0 ] or [ sign((4p² + 4Mq² − 3N²D²) + 8pq·√M) ≤ 0 ].
   The sign of A + B√M (integers A, B, M ≥ 1) is decided by comparing A² with B²·M with the signs of A
   and B (`small_kissing.cmp_sqrt`). Equality (touching spheres) is allowed everywhere.
5. Count: `count == extra.count + 196560 + Σ_i (|groups[i]| − 1)·size_i`, and it must equal the number
   of rows of the assembled configuration.
6. Assembled configuration (norm-4 scaling, R^{24+d}): let x/√8 be the norm-4 form of a Leech row x,
   let E: span(T.vectors) → R^d be any orthonormal embedding (e.g. thin SVD of `T.vectors`), and
   u(v) = E(v)/√N for v a T row or an extra row (unit vectors). Rows:
   (x/√8, 0) for x ∈ C \ ∪S_i; (√(2/3)·x/√8, √(4/3)·u(t)) for x ∈ S_i, t ∈ T_i; (0, 2·u(y')) for
   extras. All pairwise inner products must be ≤ 2 (cos ≤ 1/2); T1.4-style exact casework per pair
   type is the certificate, the float GEMM the cross-check.

## 4. Nesting and provenance

PackingStar's record families nest (fetch report §2.1): dim25 ⊂ dim27 ⊂ dim29 ⊂ dim30 ⊂ dim31 as
sets of S_i (same S_i, different T assignment per dimension); dim26 and dim28 are separate. We commit
each distinct 496-set once: `dim31/S_01..S_42.txt`, `dim26/S_01..02.txt`, `dim28/S_01..08.txt`; the
`family.json` of dim27/29/30 reference `../dim31/S_xx.txt`. `provenance` records the upstream file,
its sha256, the coordinate map and the tool that wrote the directory (extractor or
`tools/disjoint_family`, with seed/pool/method).

`ours/` is nested the same way: `ours/dim31` holds the 42 sets S_i = g^{i−1}·S (S = `data/S496.txt`,
g the Leech automorphism in `ours/dim31/g.txt`, of order 42 on antipodal pairs, with g^i S ∩ S = ∅ for
i = 1..41 — that is the whole disjointness proof); `ours/dim<n>` for n = 26..30 reference
`../dim31/S_01.txt .. S_<k>.txt` (k = 2, 5, 8, 14, 24) and copy the T/extra blocks of the record family
of that dimension (`python/tools/nest_family.py`). `provenance.tool` distinguishes the writers.

## 5. Where the numbers come from (README §1.3)

count = K(d) + 196560 + Σ_i (|T_i| − 1)|S_i|; with 496-sets: 26: 6+196560+4·496 = 198550;
27: 12+196560+7·496 = 200044; 28: 24+…+16·496 = 204520; 29: 40+…+26·496 = 209496;
30: 72+…+48·496 = 220440; 31: 126+…+84·496 = 238350. (25: 0+196560+496 = 197056, T = A1 pair.)

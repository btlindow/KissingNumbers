# The dimension-27 record and priority

This document states the dimension-27 lower bound obtained here, describes the certificate that
accompanies it, and records in full the literature and web check that was run before any of it was
circulated. The mathematics of the construction — the lifting template, the cuboctahedron partition,
and the optimality analysis for the other dimensions — is not repeated here; it is in
[`05-lifting-template-and-families.md`](05-lifting-template-and-families.md).

---

## 1. The record

**K(27) ≥ 200540.**

The bound comes from the Cohn–Jiao–Kumar–Torquato lifting template (Theorem 7.5 of *Rigidity of
spherical codes*), applied with four pairwise disjoint 496-element 60°-free subsets of the Leech
minimal vectors, the cuboctahedron as the R³ kissing configuration, its partition into **four**
disjoint equilateral triangles (weight 8 rather than the weight 7 used by all prior work), and the
twelve extra spheres of Ma et al.:

```
K(27) ≥ 12 + 196560 + 8·496 = 200540
```

The previous record is **200044**, which is the value listed for n = 27 in Henry Cohn's table of the
highest kissing numbers presently known and is due to Ma et al. (arXiv:2511.13391, "PackingStar").
The improvement is **+496**, i.e. one further copy of a 496-set. Note that the improved partition
uses *fewer* of the 496-sets than the old one — four instead of five.

### Lineage of the dimension-27 value

| source | value | R³ partition | weight | extra spheres |
|---|---|---|---|---|
| Cohn–Jiao–Kumar–Torquato 2011 (\|S\| = 480) | 199912 | 2 triangles + 3 antipodal pairs | 7 | none |
| Kallal–Kan–Wang 2017 (\|S\| = 488) | 199976 | 2 triangles + 3 antipodal pairs | 7 | none |
| Ma et al. 2025–26 (\|S\| = 496) | 200044 | 2 triangles + 3 antipodal pairs | 7 | K(3) = 12 |
| **this work (\|S\| = 496)** | **200540** | **4 triangles** | **8** | K(3) = 12 |

(Ma et al.'s v1 Table 2 records the 200044 as an improvement on the previous 199976; the value has
not moved across their four arXiv versions.)

### What is new and what is reused

New here: the observation that the cuboctahedron is the disjoint union of four equilateral triangles
— a statement about twelve points in R³ — the resulting bound K(27) ≥ 200540, and the optimality
analysis showing that the partitions used in dimensions 25, 26 and 28–31 are already of maximum
weight, so no analogous improvement is available there.

Reused unchanged: the lifting template and all of its inequalities (Cohn–Jiao–Kumar–Torquato 2011,
Theorem 7.5, arXiv:1102.5060v2); the extra spheres in the last d coordinates and the emphasis on
preferring triangles to antipodal pairs (Ma et al., arXiv:2511.13391); the four 496-element subsets,
which are four of the five that Ma et al. themselves use in dimension 27, recovered from their
published configuration and converted to the coordinate labelling used here; and the twelve extra
spheres, which are theirs verbatim (they depend on the direction set T as a set, not on how T is
partitioned).

Not claimed: any improvement on |S| = 496 — 496 remains the record for a 60°-free subset of the
Leech minimal vectors, and Ma et al. conjecture it is optimal — nor any improvement in dimensions
25, 26 or 28–31 (but see §5 for a separate, verified +2 in dimension 25).

### Verification status

| statement | status |
|---|---|
| K(27) ≥ 200540 | **verified** three times: exact casework over all C(200540,2) = 20108045530 pairs, a float64 GEMM over the assembled 200540 rows in R²⁷, and an independently written C++ set-level checker. Verified on two different families of four 496-sets: `runs/template/dim27_record/` (four of Ma et al.'s sets) and `runs/template/dim27_ours/` (an independently generated disjoint family) |
| the previous record 200044 | **independently reproduced** here from Ma et al.'s own published coordinates: exact count 200044, float count 200044, maximum off-diagonal inner product 2.000000000000, attained on lifted–lifted pairs sharing the same equatorial vector |
| weight 8 is optimal for d = 3, and the partitions used in the other dimensions are optimal | proved by exact ILP with zero gap plus a solver-free counting certificate; see [`05-lifting-template-and-families.md`](05-lifting-template-and-families.md) |

Verifier output on the committed family:

```
RESULT ok=1 dim=27 d=3 sets=4 count=200540 count_exact=200540 count_float=200540
       max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12
```

---

## 2. The certificate

The certificate is `docs/note/data/`, a self-contained copy of `data/families/dim27_improved/` plus
two human-readable files and a standalone checker. It is also packaged as
`docs/note/kissing27-certificate.zip` (10 files, 148 KB uncompressed). The accompanying note is
`docs/note/kissing27.tex`, with a compiled `docs/note/kissing27.pdf`.

| file | content |
|---|---|
| `S_01.txt` … `S_04.txt` | the four sets S_i, 496 vectors each, one per line, 24 integers of squared norm 32 (√8-scaled Leech coordinates), so that two vectors are at 60° exactly when their inner product is 16 and independence means every off-diagonal Gram entry is at most 8. Header lines record the upstream source file, its sha256, the coordinate permutation used, and the scaling |
| `T.txt` | the 12 cuboctahedron vectors in the D₃ model (±1,±1,0); the four triangles as index triples and as explicit vectors; the equivalent A₃ = {e_i − e_j} description with the four 3-cycles; and the explicit A₃ → D₃ isometry |
| `extra.txt` | the 12 extra spheres in exact form y′ = (a + b√2)/2, their normalised coordinates, and the exact maxima (2+√2)/4 and 1/2 of the relevant inner products |
| `family.json` | machine-readable form of all of the above (schema v1 of `data/families/SCHEMA.md`), with sha256 digests of the four set files and the count breakdown; each set records its provenance in Ma et al.'s `27D_200044_coordinates.npy` |
| `README.txt` | reader-facing description of the files and of what is checked |
| `verify.py` | the standalone checker — Python 3 and NumPy only, no project imports, reads nothing outside its own directory |

### What `verify.py` checks

It rebuilds the extended binary Golay code and all 196560 minimal vectors of the Leech lattice from
scratch and checks their weight distribution, membership and inner-product histogram. It then checks,
in exact integer arithmetic, that each S_i lies in that set with 496 distinct elements and
off-diagonal Gram entries ≤ 8; that the four sets are pairwise disjoint; that the twelve T-directions
have pairwise cosine ≤ 1/2, with the four groups disjoint and internally at cosine exactly −1/2; that
the twelve extra spheres (coordinates in ½Z[√2]) are at least 60° from one another and at least 30°
from every T-direction, decided by sign tests on integers A + B√2 and never in floating point; and
that the count is 12 + 196560 + 8·496 = 200540.

Finally it checks every one of the eight pair classes of the template, and the class sizes sum to
exactly C(200540, 2) = 20108045530, so no pair goes unexamined. The equatorial–equatorial class is
covered by the Leech minimum argument (distinct x, x′ in the minimal vectors give x − x′ ∈ Λ₂₄ \ {0},
hence |x − x′|² ≥ 32 and ⟨x, x′⟩ ≤ 16 in integer coordinates).

### What an external reader should run

```
python3 docs/note/data/verify.py
# RESULT ok=1 dim=27 sets=4 weight=8 count=200540          (~2 s)
```

Two slower, more exhaustive modes, both of which have been run and both of which pass:

```
python3 docs/note/data/verify.py --full     # ~90 s: brute-force maximum inner product over all
                                            # 19317818520 pairs of Leech minimal vectors (gives 16),
                                            # replacing the Leech minimum argument
python3 docs/note/data/verify.py --float    # ~2 min: assembles the 200540 explicit rows in R^27 and
                                            # computes the maximum off-diagonal inner product by a
                                            # tiled matrix product -> exactly 2.000000000000
```

The float maximum 2 is attained on lifted–lifted pairs sharing the same equatorial vector, where the
bound is tight (8/3 − 2/3 = 2), and also on lifted–lifted pairs across sets with ⟨x, x′⟩ = 2 and
⟨y, y′⟩ = 1/2.

The certificate directory is a valid family directory, so the project's own verifiers accept it
directly (the same commands work on `data/families/dim27_improved/`, which holds identical files):

```
.venv/bin/python python/verify_dimN.py docs/note/data --strict
build/release/tools/disjoint_family --verify docs/note/data
```

The certificate is not tied to Ma et al.'s particular sets: four pairwise disjoint 496-element sets
follow from any single one by taking g ∈ Co₀ = Aut(Λ₂₄) with gⁱS ∩ S = ∅ for i = 1, 2, 3 and setting
S_i = g^{i−1}S. Empirically a random g satisfies gS ∩ S = ∅ about 54 % of the time, so a few trials
suffice; a family obtained this way was checked to give 200540 as well.

---

## 3. The priority check (27 August 2026)

A read-only literature and web check was run on 2026-08-27, before any circulation. Nothing was
published, posted, emailed or committed as part of it.

**Verdict: 200044 was still the recorded lower bound for dimension 27, and no source was found in
which the four-triangle partition of the cuboctahedron is applied to this template.** The weight-7
choice for d = 3 traces to a single explicit sentence in Cohn–Jiao–Kumar–Torquato 2011 and has been
copied verbatim by every subsequent paper.

This is a statement about what was *found*. It is not a proof that no prior claim exists; see §3.7
and §4.

### 3.1 Sources checked

| source | how checked | result |
|---|---|---|
| Cohn's table, <https://cohn.mit.edu/kissing-numbers/> | re-fetched 2026-08-27 | n = 27 reads 200044; every entry byte-identical to the copy recorded here on 2026-08-25, both bounds and attributions |
| DSpace archival copy of that table, <https://dspace.mit.edu/bitstream/handle/1721.1/153312/table.pdf?sequence=8> | fetch attempted | **HTTP 405 — not checked**; its version date could not be read |
| DSpace data archive linked from the table, `hdl.handle.net/1721.1/153312` | not downloaded | **not checked**; if it contains per-dimension configuration files, they were not inspected |
| arXiv API, title query `ti:"kissing"` | 40 most recent | no paper since PackingStar touches dimensions 25–31 |
| arXiv API, abstract query `abs:"kissing number"` | 60 most recent | same |
| arXiv:2511.13391 (PackingStar) | abs-page metadata for all four versions; v1 and v4 full HTML | dimension-27 value unchanged across versions; abstract changed (see §3.2) |
| PackingStar data repository, <https://github.com/CDM1619/PackingStar> | `git ls-remote origin HEAD` = `50ea645a9805d4f29b96180550186d26a166c3be` | identical to the local clone (2026-06-08 "Add files via upload"); no new commits, contents unchanged |
| Kallal–Kan–Wang data repository, <https://github.com/kenzkallal/Kissing-Numbers> | `git ls-remote origin HEAD` = `548a09282ea1037077a435be4157e7f464b1b146` | unchanged since 2016-08-01 |
| Cohn–Jiao–Kumar–Torquato 2011, §7 | published Geometry & Topology text, pp. 2266–2267, extracted from the arXiv v2 PDF | the origin of the weight-7 recipe (§3.4) |
| Kallal–Kan–Wang 2017, Table 3 and surrounding text | read | same weight-7 grouping, no discussion of optimising it |
| SIAM landing page, <https://epubs.siam.org/doi/abs/10.1137/16M1095810> | fetched | bibliographic details confirmed (§3.5) |
| Semantic Scholar citation graph | queried for KKW (arXiv:1608.07270), CJKT, PackingStar | four papers citing KKW; **graph demonstrably incomplete** — the CJKT query returned zero citations, and the PackingStar query was rate-limited, then returned one citing paper on retry |
| general web and arXiv full-text search for `"200540"` + kissing number | searched | **no hits anywhere** |
| "kissing number dimension 27 lower bound 200540" | searched | no hits; search engines return 199976 / 200044 |
| any dimension-27 value above 200044 | searched | **none found** |
| "cuboctahedron partitions into four disjoint equilateral triangles" in this context | searched | no hits |
| MathSciNet, zbMATH, Google Scholar | **no access** | journal-only, conference-only and paywalled work is unchecked |

### 3.2 arXiv, from November 2025 onward

No paper since PackingStar touches dimensions 25–31. The relevant recent items, none in that range:

| arXiv | dates | title | dimensions touched |
|---|---|---|---|
| 2608.13906 | 2026-08-14 | New lower bounds for constant-weight codes via seeded bit-swap tabu search (W. Echols) | kissing 32–37 |
| 2607.20359 v3 | 2026-07-22, upd. 2026-08-18 | Sphere Packings and Kissing Numbers in Dimensions 39, 43, and 45 from the Antipode Construction (Sun, Wang) | 39, 43, 45 |
| 2606.18984 v2 | 2026-06-17 | Structure of kissing arrangements in R^12 and a place for the 841st sphere (Takhanov, Assylbekov, Yun) | 12 |
| 2606.03299 | 2026-06-02 | Classification of independent sets in signed Johnson graphs and applications to kissing arrangements | 12-ish |
| 2603.10425 v2 | 2026-03-11 | A new lower bound for the kissing number in 19 dimensions (B. S. Ho) | 19 |
| 2602.01638 | 2026-02-02 | Noncommutative Spherical Codes (K. M. Krishna) | none (cites Kallal–Kan–Wang) |
| 2601.03183 | 2026-01-06 | Flat simplices and kissing polytopes | n/a |
| 2411.04916 v2 | upd. 2026-03-21 | Improved kissing numbers in 17–21 dimensions (Cohn, Li) | 17–21 |

**PackingStar version history.** v1 2025-11-17 · v2 2026-01-21 · v3 2026-02-11 · v4 2026-06-02
(latest). No "Comments:" changelog is exposed on the abs page; no journal-ref, no external DOI.
Classes cs.LG; cs.AI; math.CO; math.MG. The abstract has changed since v1 — v4 now ends "…and
directly inspire subsequent breakthroughs by mathematicians", and claims "the first explicit
spherical-code realization of the Fischer group Fi22". Neither is about dimension 27. The
dimension-27 number has not moved across versions: v1 Table 2 already gives
`K(3)+K(24)+2|S_1|+2|S_2|+Σ_{i=3..5}|S_i|` = 200044 (previously 199976), and v4 Table 1 gives the
same form and the same value. The only number that moved between versions is n = 31 (238078 →
238350).

**The claim this result contradicts** (v1 text, still in substance in v4):

> "PackingStar not only discovers larger subsets but also identifies a strictly better construction
> template than the meta-construction introduced in 2011."
>
> "Evidence from convergence and geometric consistency strongly suggests that the new template is
> optimal within this construction pathway."

So the paper *asserts*, non-rigorously, that its template is optimal, and **that assertion is wrong
for d = 3**. It should be said plainly that the paper makes no formal optimality claim for the 25–31
constructions: its Theorems 1–2 certify optimality only for prescribed-inner-product settings
elsewhere in the paper.

### 3.3 The two data repositories

**PackingStar.** Their README describes the triangle idea for dimensions 31 and 29 only:

> "In 31 dimensions, PackingStar uncovers a novel assembly pattern by partitioning a 7-dimensional
> kissing configuration into 42 (optimal) disjoint unit-radius equilateral triangles, resulting in an
> 84-fold weighted S_i, surpassing the previous 75-fold weighted S_i. Similarly, in 29 dimensions,
> PackingStar embeds 12 (optimal) disjoint unit-radius equilateral triangles into the 5-dimensional
> kissing configuration to generate a 26-fold weighted S_i, exceeding the prior 24-fold weighted S_i."

Dimension 27 (d = 3) is conspicuously absent from that sentence, and there is no file, script or note
in the repository about partitioning the R³ configuration. `partitioned_D5.npy` and
`partitioned_E7.npy` are shipped (the d = 5 and d = 7 partitions, in partition order); there is no
`partitioned_A3.npy` or any analogue for d = 3, 4, 6. That is consistent with d = 3 never having
been revisited.

**Independent re-derivation of their dimension-27 structure**, recomputed from
`27D_200044_coordinates.npy` (200044 × 27, float64):

```
eq 194080   lift 5952   extra 12
distinct T directions: 12, each with multiplicity 496
T Gram cosines: {-1, -1/2, 0, +1/2}          → the cuboctahedron / A3 = D3 roots
distinct S sets: 5, all |S_i| = 496
  |T_1| = 3, internal cosines (-1/2, -1/2, -1/2)    triangle
  |T_2| = 3, internal cosines (-1/2, -1/2, -1/2)    triangle
  |T_3| = 2, internal cosine  -1                     antipodal pair
  |T_4| = 2, internal cosine  -1                     antipodal pair
  |T_5| = 2, internal cosine  -1                     antipodal pair
weight = 2+2+1+1+1 = 7;  12 + 196560 + 7·496 = 200044  ✓
```

This confirms, from the raw data, the earlier summary "27D = 5 sets, 2 triangles + 3 pairs, 12
extras". Verified directly on **their own** twelve T-directions:

```
triples with all pairwise cos = -1/2 : 8
maximum disjoint such triples (exhaustive DFS) : 4
  [(0,7,10), (1,6,9), (2,3,11), (4,5,8)]   — covers all 12 directions
max cos(extra, T)      = 0.853554  ≤ √3/2 = 0.866025
max cos(extra, extra)  = 0.5
```

So the four-triangle partition exists inside the very configuration they published, and their twelve
extra spheres remain admissible under it — the extra-sphere constraint is against T as a set, not
against the grouping T_i, so re-partitioning cannot break it.

**Kallal–Kan–Wang.** The repository holds data files only (`S_1..S_59.txt`, `minvects.txt`,
`Vbasis.txt`) plus a four-line README. There is no code, no partition data, and no mention of R³,
triangles or the cuboctahedron anywhere in it. Their paper's Table 3 gives the dimension-27 form as

> "196560 + 2|S_1| + 2|S_2| + Σ_{i=3}^{5} |S_i|"

— the same weight-7 grouping, and without the K(3) = 12 extra spheres (199976 = 196560 + 3416 with
all five |S_i| = 488; the +12 is PackingStar's addition). They say only

> "Note that |T_i| ≤ 3, where equality holds when T_i contains three vectors arranged in an
> equilateral triangle in the same plane as the origin."

and never discuss optimising the partition. Their "almost certainly suboptimal" remark is about the
computer-generated subsets S_i, not about the T_i grouping.

### 3.4 Where the weight-7 partition comes from

This is the decisive find. From the published Geometry & Topology text of Cohn–Jiao–Kumar–Torquato
2011, §7 (pp. 2266–2267, extracted from the arXiv v2 PDF):

> "We do not know the best way to apply Theorem 7.5, but we can use it as follows. Given any kissing
> configuration in R^d of size K, we want to partition it into antipodal pairs and equilateral
> triangles (or singletons if necessary). Of course, if it is antipodal we can simply partition it
> into K/2 antipodal pairs, but that will generally not be optimal."
>
> "If there is an Eisenstein structure, as is the case for A_2, D_4, E_6 and E_8, then we partition it
> into regular hexagons using that structure and divide each hexagon into two equilateral triangles.
> **For A_3, D_5 and E_7, we partition a cross section using its Eisenstein structure and then fill
> the rest with antipodal pairs.** This yields the bounds shown in Table 4."
>
> "We do not expect that these bounds are anywhere close to being optimal…"

Their Table 4, dimension-27 row: `196560 + 2|S_1| + 2|S_2| + Σ_{i=3}^{5}|S_i|`, giving 198576
(|S| = 288) and 199912 (|S| = 480).

So for A₃ the recipe is literally "hexagonal cross-section (→ 2 triangles) + 3 antipodal pairs",
weight 7. That heuristic is optimal for A₂, D₄, E₆ and E₈ and suboptimal for A₃, D₅ and E₇.
PackingStar fixed D₅ (24 → 26) and E₇ (75 → 84) but left A₃ at 7. **A₃ is the leftover case of a 2011
heuristic that its own authors flagged as provisional**, and no source was found anywhere that
revisits it.

### 3.5 Bibliography check

The elementary fact underlying the observation is of course classical: a decomposition of the
complete symmetric digraph on four points into directed triangles is a Mendelsohn triple system of
order 4, and the note cites Mendelsohn 1971 for it. What no source does is apply it to Theorem 7.5.

Findings on the individual references, and their current state in `docs/note/references.bib`:

* **Kallal–Kan–Wang 2017 — confirmed correct.** SIAM J. Discrete Math. **31** (2017), no. 3,
  1895–1908, doi `10.1137/16M1095810`, authors Kenz Kallal, Tomoka Kan, Eric Wang; verified against
  the SIAM page. One nit: the published title capitalises "through" in lower case — "Improved Lower
  Bounds for Kissing Numbers in Dimensions 25 through 31" — so the sentence-case title in the entry
  is fine for most styles and no change is required.
* **Cohn–Jiao–Kumar–Torquato 2011 — confirmed correct.** "Rigidity of spherical codes", Geometry &
  Topology **15** (2011) 2235–2273, doi `10.2140/gt.2011.15.2235`, arXiv:1102.5060 (v1 2011-02-24,
  v2 2012-04-06). The `number = {4}` field is consistent with the journal's issue numbering. The
  note should cite **v2** explicitly, since the §7 text quoted above is from v2; the shipped note
  does.
* **Ma et al. — a correction to two other people's metadata, not to ours.** arXiv's canonical display
  order is "Chengdong Ma, Théo Tao Zhaowei, Pengyu Li, Minghao Liu, Haojun Chen, Zihao Mao, Bo Li,
  Yuan Cheng, Yuan Qi, Yaodong Yang" — **ten** authors in v4. So `Zhaowei, Th{\'e}o Tao` is the
  correct BibTeX family-first rendering of "Théo Tao Zhaowei", and the inclusion and ordering of Bo
  Li are right. The local entry is therefore *better* than both upstream renderings: the PackingStar
  repository's own BibTeX mangles the second author to `Th{\u{A}}{\v{S}}o Tao` and drops Bo Li, and
  **Cohn's table renders the reference with a nine-author list — "C. Ma, T. Tao Z., P. Li, M. Liu,
  H. Chen, Z. Mao, Y. Cheng, Y. Qi, and Y. Yang" — also dropping Bo Li.** Two improvements were
  suggested to the local entry and are still outstanding: annotate the version and date (keep
  `year = {2025}` for the first posting, but record that v4 is 2026-06-02, so the reader knows which
  numbers are cited), and record the arXiv DOI `10.48550/arXiv.2511.13391` and that no journal
  version exists (checked 2026-08-27). The given/family split of the second author should still be
  confirmed against the paper itself.
* **Mendelsohn 1971 — corrected.** The entry was malformed: declared `@article` while carrying
  `booktitle`, `publisher`, and a `journal` field duplicating the book title. It is now
  `@incollection{Mendelsohn1971}` — *A natural generalization of Steiner triple systems*, in
  *Computers in Number Theory*, ed. Atkin and Birch, Academic Press, London, 1971, pp. 323–338.
* **de Laat–Leijenhorst 2024 — still outstanding.** Cohn's table cites it as *Math. Program. Comput.*
  **16** (2024), **no. 3**, 503–534; the local entry still omits `number = {3}`.
* **Cohn's table entry.** The page carries no last-updated stamp, so the access date is the only
  dating available. The `.bib` entry records 26 August 2026 while the note's inline bibliography
  records 27 August 2026; the access date should be made consistent and set to whenever the note is
  finalised. Adding a note that the page gives no revision date would be reasonable.
* **Not re-verified against publisher records in this pass:** Conway–Sloane (SPLAG),
  Schütte–van der Waerden 1953, Musin 2008, Leech 1967. They were checked only for their appearance
  in Cohn's reference list.

### 3.6 Cohn's table, as re-fetched on 2026-08-27

| n | lower | upper | lower source | upper source | change since 2026-08-25 |
|---|---|---|---|---|---|
| 24 | 196560 | 196560 | Leech 1967 | Levenšteĭn 1979, Odlyzko–Sloane 1979 | none |
| 25 | 197056 | 265006 | Ma et al. 2025, arXiv:2511.13391 | de Laat–Leijenhorst 2024 | none |
| 26 | 198550 | 367775 | Ma et al. | de Laat–Leijenhorst | none |
| **27** | **200044** | **522212** | **Ma et al.** | **de Laat–Leijenhorst** | **none** |
| 28 | 204520 | 752292 | Ma et al. | de Laat–Leijenhorst | none |
| 29 | 209496 | 1075991 | Ma et al. | de Laat–Leijenhorst | none |
| 30 | 220440 | 1537707 | Ma et al. | de Laat–Leijenhorst | none |
| 31 | 238350 | 2213487 | Ma et al. | de Laat–Leijenhorst | none |

The page carries **no "last updated" stamp**: no date, no revision note, and no date in the HTML
metadata. The only dating handle is indirect — it cites arXiv:2607.20359, first posted 2026-07-22,
so the page is at least that recent. The values are byte-identical to those recorded on 2026-08-25;
that is evidence of stability, not of a recent refresh. The table was re-fetched again on 2026-08-31
and n = 27 still read 200044.

### 3.7 What could not be checked

1. **Cohn's page has no last-updated date.** It cannot be shown that the table was refreshed after
   2026-08-25 — only that the values are byte-identical to what was recorded then. The DSpace
   archival copy (`1721.1/153312/table.pdf`), which might carry a version date, returned HTTP 405
   and was not retrieved.
2. **Non-arXiv literature.** No access to MathSciNet, zbMATH or Google Scholar. Journal-only or
   conference-only work, and anything behind a paywall, is unchecked. The Semantic Scholar citation
   graph was used but is demonstrably incomplete: a CJKT citation query returned zero results, and
   the PackingStar query was rate-limited on the first attempt and returned a single citing paper
   (Cohn–Li, 17–21) on retry, which is clearly an undercount. Treat the citation-graph evidence in
   §3.1 as weak. For the record, Semantic Scholar lists exactly four papers citing Kallal–Kan–Wang:
   PackingStar; "Noncommutative Spherical Codes" (arXiv:2602.01638); a p-adic DGS/KLP-bound paper
   (arXiv:2503.05654); and a 2020 modular-bootstrap paper. None improves 25–31.
3. **In-progress or unposted work.** Nothing can be said about preprints not yet on arXiv, private
   communications, or work in review. Given the PackingStar abstract's new line about "subsequent
   breakthroughs by mathematicians", there may be follow-up work in preparation by that group or
   their collaborators; no such preprint about dimensions 25–31 existed on arXiv as of 2026-08-27.
4. **PackingStar v2 and v3 full text** was not read line by line; only v1 and v4 HTML were queried,
   plus the abs-page metadata for all four. The dimension-27 form and value agree in v1 and v4, so an
   intermediate change that was then reverted is unlikely but not excluded.
5. **The DSpace data archive** linked from Cohn's table (`hdl.handle.net/1721.1/153312`) was not
   downloaded; if it contains per-dimension configuration files, they were not inspected.
6. **Priority in the strict sense.** It can be shown that no *published* source found here applies
   the four-triangle partition in this template. It cannot be ruled out that the observation is
   folklore among specialists — it is a one-line remark about twelve points in R³, and CJKT
   explicitly said they did not know the best partition. This is exactly the gap in which the
   independent discovery described in §4 sits.

---

## 4. Independent co-discovery by Alexey Kravatskiy

**The same observation — that the cuboctahedron partitions into four disjoint equilateral triangles,
giving weight 8 in the dimension-27 template and hence K(27) ≥ 200540 — was found independently, and
at about the same time, by Alexey Kravatskiy.** Neither discovery derives from the other.

The dates on record here are:

| date | event |
|---|---|
| 2026-08-26 | the four-triangle partition found in this project, and K(27) ≥ 200540 verified end to end the same day |
| 2026-08-27 | the priority check of §3 run; the note and certificate assembled |
| by 2026-08-30 | Kravatskiy's independent discovery became known to this project; from that point the working notes refer to it as co-discovered, and to a Cohn–Kravatskiy collaboration |

No precise date for Kravatskiy's own discovery is recorded in this repository, so "at about the same
time" is as strong a statement as the material here supports; he is the authority on his own
timeline. What is claimed to be co-discovered is specifically the R³ partition and the resulting
bound. The 496-element sets, the extra spheres and the template itself are prior work by Ma et al.
and by Cohn–Jiao–Kumar–Torquato in every account.

This bears directly on how §3 should be read. The check of 2026-08-27 searched published papers,
preprints, public data repositories and online record tables. **An unpublished independent discovery
is invisible to every one of those channels, and in this case one existed.** "No prior claim was
found" is therefore exactly what the check establishes, and no more.

---

## 5. A verified +2 in dimension 25: K(25) ≥ 197058

A separate and much smaller observation, obtained while examining alternative shells in the template
and reported here because it is exact and rigid rather than because it is significant.

Treating the CJKT lift height as a free parameter rather than a fixed one, the lowered lift h = 1
(the caps at latitude 60°) leaves room for the two poles. With a single 496-set S, the equator keeps
the 196560 − 496 minimal vectors outside S, two lifted caps carry 496 each, and both poles fit:

```
EXACT COUNT = (196560 − 496) + 2·496 + 2 = 197058
```

**Verified** by exact casework plus an explicit float64 pass over all non-(equator, equator) pairs:

```
max <x, s> over x in C\S, s in S = 16
eq-cap worst inner product = (sqrt3/2)*16/8 = 1.732051 <= 2 : True
cap-cap same sign: needs max ip within S <= 8; S496 independent: True
cap-cap opposite sign worst = (3/4)*32/8 - 1 = 2.000000 <= 2 : True
pole-cap = 2 exactly (touching); pole-eq = 0; pole-pole = -4
  |eq| = 196064   |cap| = 992   |pole| = 2      (all norms 4.0)
  max off-diagonal among cap+pole (994 pts) = 2.000000000000
  max inner product equator x (cap+pole)     = 1.732050807569
RESULT dim25_h1 ok=1 count=197058
```

That is **+2 on the published 197056**. The mechanism is exhausted: lowering the lift height yields
exactly this +2 and nothing more in dimension 25, and nothing at all for d ≥ 2. It is also very
likely subsumed by unpublished work in this range (§6), and it has been treated throughout as a
framework check to offer a specialist — a cheap way to confirm that two models of the same geometry
agree — rather than as a record claim. It is included here because it is exact.

---

## 6. Caveats

* **"No prior claim was found" is not "no prior claim exists."** The check of §3 covered arXiv, two
  public data repositories, Cohn's online table and general web search. It did not cover MathSciNet,
  zbMATH, Google Scholar, paywalled or journal-only literature, unposted preprints, or private
  correspondence. A genuine independent discovery went undetected by it (§4).
* **Record tables can be out of date, and this one is undated.** Cohn's table carries no
  last-updated stamp and no revision note; its archival copy could not be fetched. Any statement of
  the form "the record is X" rests on that page plus an arXiv sweep, and both have latency. The
  table should be re-checked at the moment of any submission.
* **There is unpublished work in dimensions 25–27.** The author has been informed, in private
  correspondence, that a construction not yet posted exceeds the published values in these
  dimensions. No configuration, certificate or preprint is available here, and nothing about it is
  verifiable from this repository, so it is not described further. The record claim in §1 is stated
  against the published 200044 and would be superseded if that work appears.
* **Verified is not the same as published.** K(27) ≥ 200540 and K(25) ≥ 197058 are both verified
  here by exact computation with certificates a reader can re-run; neither has been refereed. As
  recorded in `docs/note/README.md` on 2026-08-27, the note itself had not at that date been
  submitted, posted or emailed anywhere; a handover package for external readers
  (`docs/handover/findings.pdf` and `leech-496-artifacts.zip`) was assembled subsequently. That
  README also predates the compiled `docs/note/kissing27.pdf`, which now exists.
* **The published 200044 is itself a claim this project has re-checked, not taken on trust.** It was
  independently reproduced from Ma et al.'s published coordinates (§1), which is what makes the
  +496 comparison meaningful.

---

### See also

* [`05-lifting-template-and-families.md`](05-lifting-template-and-families.md) — the template, the
  four-triangle lemma, the disjoint families, the dimension-N verifiers, and the proof that the
  partitions in the other dimensions are already optimal.
* [`01-lattice-and-verification.md`](01-lattice-and-verification.md) — the Leech minimal vectors, the
  coordinate map to Ma et al.'s labelling, and the verifiers.
* [`REPRODUCE.md`](../../REPRODUCE.md) — how to re-run every headline claim from a fresh clone.

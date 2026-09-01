# `docs/note/` — research note: K(27) >= 200540

**Status:** the note below reports K(27) >= 200540. The result was communicated to
Henry Cohn, who confirmed it is correct and reported that Alexey Kravatsky had found
the same improvement independently at about the same time. It is not posted to arXiv.

| File | Content |
|---|---|
| `kissing27.tex` | the note (amsart, self-contained `thebibliography`, no external `.bst` needed) |
| `references.bib` | the same references in BibTeX, for whoever prefers `\bibliography` |
| `data/` | the complete certificate (see below) |

## Building the PDF

**Building the PDF.** `kissing27.pdf` is committed. To rebuild it you need a LaTeX
installation providing `amsart`, `amsmath`/`amssymb`/`amsthm`, `booktabs`, `geometry`,
`hyperref` and `url` (on Debian/Ubuntu: `texlive-latex-base texlive-latex-recommended
texlive-fonts-recommended`). The bibliography is inline, so no BibTeX pass is needed:

```
pdflatex kissing27 && pdflatex kissing27
```
## The certificate (`data/`)

Self-contained copy of `data/families/dim27_improved/`, plus two human-readable
files and a standalone checker.

| File | Content |
|---|---|
| `S_01.txt` … `S_04.txt` | the four sets S_i, 496 vectors each, one per line, 24 integers of squared norm 32 (√8-integer Leech coordinates). Header lines record the upstream source file and its sha256. |
| `T.txt` | the 12 cuboctahedron vectors in the D3 model (±1,±1,0), the four triangles as index triples and as explicit vectors, the equivalent A3 = {e_i − e_j} description, and the isometry between the two models |
| `extra.txt` | the 12 extra spheres in exact form y' = (a + b√2)/2, plus their normalised form and the exact maxima (2+√2)/4 and 1/2 |
| `family.json` | machine-readable version of all of the above (schema v1 of `data/families/SCHEMA.md`), including sha256 digests of the four set files and the count breakdown |
| `verify.py` | standalone checker — Python 3 and NumPy only, no project imports |

### Checking it

```
$ python3 docs/note/data/verify.py
```

runs in about **2 seconds** and prints

```
RESULT ok=1 dim=27 sets=4 weight=8 count=200540
```

It rebuilds the Golay code and all 196560 Leech minimal vectors from scratch, checks
their weight distribution / membership / inner-product histogram, then checks the four
sets, the T partition, the extra spheres (exactly, in Z[√2]), the count, and every one
of the eight pair classes — whose sizes sum to exactly C(200540, 2) = 20108045530, so
no pair goes unexamined.

Two optional, slower passes:

```
$ python3 docs/note/data/verify.py --full    # ~90 s: brute-force max over all
                                             # 19317818520 pairs of C (gives 16),
                                             # instead of the Leech minimum argument
$ python3 docs/note/data/verify.py --float   # ~2 min extra: assembles the 200540
                                             # explicit rows in R^27 and checks the
                                             # max off-diagonal inner product
                                             # -> exactly 2.000000000000
```

Both were run; both pass.

### Cross-checks with the project's own tools

`data/` is a valid family directory, so the project verifiers accept it directly:

```
$ .venv/bin/python python/verify_dimN.py docs/note/data --strict
$ build/t42/tools/disjoint_family --verify docs/note/data
```

(the same commands work on `data/families/dim27_improved/`, which holds identical
files). Reference output of the first, on the committed family:

```
RESULT ok=1 dim=27 d=3 sets=4 count=200540 count_exact=200540 count_float=200540
       max_offdiag=2.000000000000 max_class=lift-lift-same-x extra=12 K=12
```

## Provenance and credit

Reused, unchanged:

* the lifting template and all of its inequalities — Cohn–Jiao–Kumar–Torquato 2011,
  Theorem 7.5 (arXiv:1102.5060);
* the extra spheres in the last d coordinates, and the idea of preferring triangles to
  antipodal pairs — Ma et al. 2025 ("PackingStar", arXiv:2511.13391);
* the four 496-element subsets of the Leech minimal vectors — sets 1–4 of the 42
  published by Ma et al., converted to our coordinate labelling by a permutation
  (see the S file headers, and `data/external/coordinate_map.json`);
* the 12 extra spheres — Ma et al.'s dimension-27 extras, verbatim (they depend on T
  as a set, not on how T is partitioned).

New in this note:

* the observation that the cuboctahedron partitions into **four** disjoint equilateral
  triangles rather than two triangles + three antipodal pairs (weight 8 instead of 7)
  — a statement about 12 points in R^3;
* the resulting bound K(27) ≥ 200540;
* the optimality analysis for the other dimensions (weight ≤ ⌊2K(d)/3⌋ for any
  configuration whatsoever, plus the zero-sum argument for D5).

Not claimed: any improvement on |S| = 496, or on dimensions 25, 26, 28–31.

## Before submitting — open items

1. **Build the PDF.** Nothing has been compiled here.
1b. **Length.** The body is ~3100 words plus two tables and nine displays, which at
   amsart 11pt / 1in margins should land around 5–6 pages — over the 2–4 page target
   in README §6. Everything present was explicitly asked for (template inequalities,
   lemma, theorem, bounds table, optimality section, verification section); if it must
   be shorter, §5 (verification) compresses most easily — move the checker's
   step-by-step description into the data README and leave only the pair-class total
   and the two tight cases.
2. **Author line.** The note is drafted under `Ben Lindow <benlindow@gmail.com>`. Change
   if the authorship should differ.
3. **Check the Ma et al. author list.** The upstream BibTeX in the PackingStar repo has a
   mangled second author (`Zhaowei, Th{\u{A}}{\v{S}}o Tao`); the arXiv listing reads
   "Théo Tao Zhaowei". The bibliography here uses the arXiv form, but the given/family
   name split should be confirmed against the paper itself before citing.
4. **Re-check Cohn's table** at submission time in case a record has moved.
5. The Kallal–Kan–Wang journal reference (SIAM J. Discrete Math. **31** (2017), no. 3,
   1895–1908, doi 10.1137/16M1095810) was taken from a web search, not from the article;
   worth a glance at the actual page numbers.

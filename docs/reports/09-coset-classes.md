# Coset classes: how many lattice points fit on a small sphere

The classical lifting template of Cohn–Jiao–Kumar–Torquato builds a kissing configuration in
`ℝ²⁴⁺ᵈ` from a **class**: a set of unit vectors in `ℝ²⁴` with pairwise cosine at most `1/4`.
The classes used in the literature consist of Leech *minimal vectors*, where the record is
`496` and the best upper bound is `837`
([`02-upper-bounds.md`](02-upper-bounds.md), [`07-lines-vs-vectors.md`](07-lines-vs-vectors.md)).

This note asks the same question one step out: what if the class is allowed to be a set of
lattice points on a **sphere about an arbitrary centre**, rather than a subset of a shell about
the origin? The answer turns out to be governed by a single number, and the whole search
collapses to a question about centres.

## Contents

1. [The criterion](#1-the-criterion)
2. [A ceiling on every forced class at once](#2-a-ceiling-on-every-forced-class-at-once)
3. [The sweep over centres](#3-the-sweep-over-centres)
4. [891, and what it is](#4-891-and-what-it-is)
5. [What is open](#5-what-is-open)
6. [Files and reproduction](#6-files-and-reproduction)

---

## 1. The criterion

Fix a centre `c ∈ ℝ²⁴` and let the class be the set of lattice points on a sphere,
`P(c, ρ) = {x − c : x ∈ Λ, |x − c|² = ρ}`, normalised. For two points on that sphere,
`⟨x,c⟩ = (|x|² + |c|² − ρ)/2`, and substituting into
`⟨x−c, y−c⟩ = ⟨x,y⟩ − ⟨x,c⟩ − ⟨y,c⟩ + |c|²` gives

```
                    <x - c, y - c>  =  rho - |x - y|^2 / 2
```

**independent of `c`.** So the normalised cosine is `1 − |x−y|²/(2ρ)`, and because distinct
lattice points satisfy `|x − y|² ≥ 4`,

> **The class is *forced* — every pair automatically at cosine `≤ 1/4`, with no subset to
> choose — if and only if `ρ ≤ 8/3`.**

Equality is attained exactly when some pair has `|x − y|² = 4`. For `ρ > 8/3` those pairs break
the class condition and one is back to a packing problem; for `ρ ≤ 8/3` the entire sphere is a
class, and its size is simply how many lattice points lie on it.

**Nothing is left free but the centre.** That is the useful consequence: an open-ended search
for a good class becomes a concrete question — *which centre puts the most lattice points on a
sphere of radius² `8/3`?*

## 2. A ceiling on every forced class at once

At `ρ = 8/3` the cosine `1 − 3|x−y|²/16` takes values in `{1/4, −1/8, −1/2, −7/8}` (larger
`|x−y|²` would exceed the sphere's diameter), so a forced class normalises to a spherical code
in `ℝ²⁴` with inner products in a **four-element set**. The Delsarte–Goethals–Seidel LP
therefore applies to all forced classes simultaneously:

```
RESULT a forced class normalises to a spherical code in R^24 with inner products in
       {1/4, -1/8, -1/2, -7/8}
RESULT EXACT Delsarte bound on ANY forced coset class: 13405743/14959 = 896.16...
```

**At most `896`**, whatever the centre. The certificate is exact rational, produced by the same
`rationalise` / `verify_certificate` path as the bounds of
[`02-upper-bounds.md`](02-upper-bounds.md) §3.

## 3. The sweep over centres

For `c = (a/b)v` the sphere condition reads `|bx − av|² = b²ρ`, an integer, so the enumeration
is a Fincke–Pohst run on `bΛ` — exact, and complete for the centre it is given. Sweeping `v`
over norms `4, 6, 8, 10, 12, 16, 24` and `a/b` with denominators `2, 3, 4, 6`, at every radius
the ball of radius² `8/3` actually realises:

| `\|v\|²` | `α` | `ρ` | class size | norms present |
|---|---|---|---|---|
| 4 | any | — | ≤ 2 | — |
| 6 | 1/2, 3/2, 5/2 | 5/2 | 552 | one shell |
| **6** | **1/3, 2/3, 4/3, 5/3** | **8/3** | **553** | two shells |
| 8 | 1/2 | 2 | 48 | — (a deep hole) |
| 10 | 1/2, 3/2, 5/2 | 5/2 | 552 | up to four shells |
| 10 | four values | 22/9 | 275 | one shell |
| **12** | **1/3, 2/3, 4/3, 5/3** | **8/3** | **891** | one shell |
| 16 | 1/2 | — | 1 | — |

Two remarks. The `|v|² = 8`, `α = 1/2` row recovers a **deep hole** of `Λ`: `ρ = 2` is the
covering radius squared, and 48 is the vertex count of the `A₁²⁴` hole. And the `|v|² = 6`,
`α = 2/3` sphere is the one case in the sweep where the **origin itself lies on it**
(`|c|² = ρ = 8/3`), so the class contains a lattice point of norm 0.

**Completeness.** For a given centre the sphere enumeration is complete and exact. The
*centre* space is not exhausted: only centres on the line through a lattice vector were swept.
A genuine characterisation would range over all subsets of `Λ` lying on a common sphere of
radius² `≤ 8/3` — finite, but a real piece of work. See §5.

## 4. `891`, and what it is

The largest forced class the sweep finds is **891**, within 5 of the ceiling of 896. It is not
an accident of search:

> Take `v = a + b` with `a, b` minimal and `⟨a,b⟩ = 2` — that is, an **edge of the 60°
> conflict graph** — and `c = v/3`. The sphere condition `3|x|² = 2⟨x,v⟩ + 4` admits only
> `|x|² = 4`, and then `⟨x,a⟩ + ⟨x,b⟩ = 4` with both terms in `{0,±1,±2,±4}` forces
> `⟨x,a⟩ = ⟨x,b⟩ = 2`. So **the class is exactly the set of triangles through that edge**, and
> its size is the scheme's intersection number `p⁵₅₅ = 891`.

```
RESULT (<x,a>, <x,b>) over the class: {(2, 2): 891}
RESULT common neighbours of the edge {a,b} at inner product +16 (= p^5_55): 891
RESULT the class IS exactly the set of triangles through that edge: True
```

Its members have pairwise `⟨x,y⟩ ∈ {0,1,2}` — all non-negative, and including 60° pairs — so
geometrically it is a tight cluster about `v̂` rather than a spread-out set. Size alone is
therefore not the right figure of merit for a class: how *concentrated* it is matters too, and
the two pull against each other. Which of them dominates depends on the application, and is
not pursued here.

## 5. What is open

* **Completeness of the centre search.** Only centres on a lattice line were swept. Can the
  centres admitting a large forced class be characterised?
* **Is `891` the maximum forced class?** It sits 5 below the Delsarte ceiling of `896.16`. An
  exact answer would be a clean self-contained statement about `Λ`.
* **The trade-off between size and concentration**, made precise. The sweep suggests the two
  are in tension, but that is an observation, not a theorem.

## 6. Files and reproduction

| file | role |
|---|---|
| [`tools/coset/coset_sweep.py`](../../tools/coset/coset_sweep.py) | the criterion, the Delsarte ceiling, and the sweep over centres |
| [`tools/coset/fincke_pohst.py`](../../tools/coset/fincke_pohst.py) | LLL + exact-rational Fincke–Pohst |
| [`tools/coset/fp_fast.py`](../../tools/coset/fp_fast.py) | the same with float pruning and an exact final check |

```
PYTHONPATH=python .venv/bin/python tools/coset/coset_sweep.py     # ~10 min
```

Both regenerate the Leech minimal vectors from the Golay code and read no stored copy. The
Fincke–Pohst enumerations are exact: `fp_fast.py` uses floating point for pruning only, and
every surviving point is confirmed with integer arithmetic, so over-collection is harmless and
under-collection is guarded by an inflated radius.

# Lines vs vectors: does `|S| ≤ 837` improve the 425-line bound?

**Status: yes, rigorously.** `α ≤ 837` **vectors** implies `≤ 418` **lines** for the Leech class
problem, against the 425 that every previously recorded method gives. This note establishes the
units reconciliation with the external record of A. Kravatskiy
(<https://github.com/alexlegeartis/KissingNumbers>, `verifications/closed/class-problem-upper-bound/`),
because the two projects state the same family of results in different units and, at first sight,
about two different problems.

---

## 1. The two problems are genuinely different — and the difference is in our favour

| | **our** problem (vectors) | **his** problem (lines) |
|---|---|---|
| ground set | the `N = 196560` Leech minimal vectors `C` | the `98280` lines `{±x}`, `x ∈ C` |
| forbidden | `⟨x,y⟩ = +16` only (`cos = ½`, 60°) | `\|⟨x,y⟩\| = 16` (`\|cos\| = ½`), i.e. **both** `±16` |
| allowed inner products | `{−32, −16, −8, 0, 8}` | `{−8, 0, 8}` across distinct lines |
| unit | vectors | lines = antipodal pairs |
| record | `496` | `248` |
| previous best bound | `⌊9360/11⌋ = 850` (CJKT 2011) | `⌊4680/11⌋ = 425` |

So his relation is **strictly finer** than ours: our conflict graph forbids `+16` only, his forbids
`±16`. The record set `S496` happens to avoid `−16` too, but the *relaxations* differ.

## 2. The reduction, and why it goes the right way

> **Doubling lemma.** Let `T` be a set of `L` lines of Leech minimal vectors, pairwise at
> `|cos| ≤ ¼`. Pick a representative for each line and let `S = {±x : x ∈ T} ⊆ C`. Then `|S| = 2L`
> and `S` contains no pair at inner product `+16`.
>
> *Proof.* Distinct lines `{±x}, {±y}` contribute the four inner products `±⟨x,y⟩`, and
> `|⟨x,y⟩| ∈ {0, 8}` by hypothesis. Within a line the only inner product is `⟨x,−x⟩ = −32`. Neither
> is `+16`. ∎

Hence `2L ≤ α(G)` for every admissible line set, so

```
L_max  ≤  α(G) / 2  ≤  837 / 2  =  418.5   ⇒   L_max ≤ 418.
```

The direction matters: the line problem is a **restriction** of ours, so an upper bound for our
(weaker) constraint is *a fortiori* an upper bound for his after doubling. There is no gap in the
argument and no need for our relaxation to know about `−16` at all.

The same doubling is what turns the published CJKT vector bound `850` into the quoted `425` lines,
so we are following exactly the conversion already in use.

## 3. Running our SDP on *his* problem directly

For completeness the three-point SDP was re-run with **both** `±16` forbidden (classes 1 and 5 of
`CLASS_DOTS = (−32,−16,−8,0,8,16,32)`). This is the relaxation of the line problem in vector units.

```
RESULT sdp3 forbid={16}    : numeric 837.5332376939015  status optimal  exact 837.535866  floor 837
RESULT sdp3 forbid={-16,16}: numeric 837.5332597758825  status optimal  exact 837.538406  floor 837
```

The two relaxation optima agree to `2.2e−5`, which is solver tolerance (adding constraints cannot
raise an optimum, so the true values coincide). This is predicted by §8 of
[`02-upper-bounds.md`](02-upper-bounds.md): at the `{16}` optimum the pair density of `−16` is
already `0`, and `y = x_diag(k(u)) − x_u ≥ 0` with `x_diag(1) = 0` then forces every triple orbit
touching class 1 to zero — so the extra constraints are inactive. **The two problems agree at three
points exactly as they agree at two points.** Both give `837` vectors, i.e. `418` lines.

The two-point LP was re-run the same way and is unchanged:

```
RESULT two-point LP forbid=+16      exact=9360/11 = 850.909091 floor=850
RESULT two-point LP forbid=±16      exact=9360/11 = 850.909091 floor=850
RESULT two-point LP forbid=±16 anti exact=9360/11 = 850.909091 floor=850
```

## 4. Scope of the claim — an important caveat

The 425 bound has **two** proofs, and only one of them is Leech-specific:

* the Hoffman ratio bound on the 4-class line scheme (`98280` lines, valency `4600`,
  `λ_min = −20`) — Leech-specific;
* the degree-4 Gegenbauer certificate `f(t) = (4992/11)·t²(t² − 1/16)` — **lattice-free**: it bounds
  *any* line system in `ℝ²⁴` with `|cos| ≤ ¼`.

Our `837` is Leech-specific: it is built from the `148` `Co₀`-orbitals of ordered triples of minimal
vectors. Therefore:

* **418 bounds the class problem** ("lines *of Leech minimal vectors* pairwise at `|cos| ≤ ¼`"),
  which is the problem that feeds the dimension 25–31 constructions. ✔
* **418 does not bound an arbitrary line system in `ℝ²⁴` with `|cos| ≤ ¼`.** For that, 425 remains
  the best. This distinction must be stated whenever 418 is quoted.

## 5. Relation to the "a three-point bound cannot do it" finding

Kravatskiy's `bv3pt.py` implements the **Bachoc–Vallentin** three-point bound in prescribed
inner-product form for antipodal codes in `S²³` with inner products in `{−1, −¼, 0, ¼}`, at
truncations `d_max = 4 … 12`, and reports `M ≤ 9360/11 → N ≤ 425.4545` with the `Y_k` positivity
never active. His conclusion — "closing the 248–425 gap needs a four-point / Lasserre-3 relaxation,
or a lattice-structural argument. A three-point bound cannot do it" — is correct **for the
lattice-free three-point bound**, and his validation of that code (276 for the `1/5` equiangular
problem, 144 for `n = 16`) is convincing.

Our `837` is not a counterexample to that statement, it is the "lattice-structural argument" he
names, cast as a three-point relaxation: the Schrijver/Terwilliger SDP over the **full
148-dimensional centraliser algebra of `Stab(x₀)` in `Co₀`**, whose extra content over the spherical
relaxation is precisely that an orthogonal triple of minimal vectors comes in two `Co₀`-inequivalent
flavours (`p³₃₃ = 43164 = 42240 + 924`). A polynomial three-point bound on `S²³` cannot see that.

## 6. Verification

Two independent verifiers, exact arithmetic throughout, per the two-verifier rule.

**Verifier A — the reduction, on independently regenerated data** (`kiss_ref`, never
`data/leech_min.*`):

```
$ .venv/bin/python tools/bounds/verify_line_vector_reduction.py
RESULT L1 antipodal_closed=True N=196560 lines=98280 expect lines=98280 -> True
RESULT L2 per-basepoint (#dot=+16, #dot=-16, #conflicting lines) = [(4600, 4600, 4600)]
         -> vector-valency 4600, line-valency 4600: True
RESULT L4 |S496|=496 antipodally_closed=True lines=248 pairs_at_+16=0 pairs_at_-16=0
         -> feasible for BOTH problems: True
RESULT L3 doubling[S496-248-lines] lines=248 vectors=496 pairs_at_+16_after_doubling=0 ok=True
RESULT L3 doubling[greedy-0..4]    lines=109..119                                     ok=True
RESULT task1-verifierA ok=1
```

L2 independently confirms the inputs of his Hoffman bound (`N = 98280`, `k = 4600`).

**Verifier B — the certificates, from scratch** (reads only raw
`data/scheme/orbitals.json` and the certificate JSON; imports nothing from `python/bounds/` and
nothing from any solver; rebuilds the `S₃` orbit partition, the objective, `Cy`, the free/forbidden
split, both `G` families from the structure constants, both PSD blocks from their integer rank-one
decompositions, `h`, and the bound, in `Fraction`):

```
$ .venv/bin/python tools/bounds/recheck_sdp3_certificate.py data/scheme/sdp3_certificate.json
RESULT B1 s3_action=True labels=True partition=True closed=True orbits_are_single=True
         c_rowtotals=N:True D=148 Q=43
RESULT B2 diag_classes=[0,1,2,3,4,5,6] objective_matches=True Cy_matches=True sum_obj=196560
RESULT B3 forbidden_classes=[5] dots=[16] free=28 matches=True e=42 e_ok=True
RESULT B4 psd_by_construction=True h_matches=True bound=837.535866161 floor=837 matches_stored=True
RESULT B5 set=S496.txt n=496 obj.x=496 x_e=1 zero_on_forbidden=True chain obj.x<=mid<=bound: True
RESULT B5 set=S488.txt n=488 obj.x=488 x_e=1 zero_on_forbidden=True chain obj.x<=mid<=bound: True
RESULT task1-verifierB file=sdp3_certificate.json forbidden_dots=[16] bound_floor=837 ok=1

$ .venv/bin/python tools/bounds/recheck_sdp3_certificate.py data/scheme/sdp3_certificate_pm16.json
RESULT B3 forbidden_classes=[1,5] dots=[-16,16] free=18 matches=True e=42 e_ok=True
RESULT B4 psd_by_construction=True h_matches=True bound=837.538406359 floor=837 matches_stored=True
RESULT task1-verifierB file=sdp3_certificate_pm16.json forbidden_dots=[-16,16] bound_floor=837 ok=1
```

The `±16` certificate is `data/scheme/sdp3_certificate_pm16.json` (new); the `+16` one is unchanged.

## 7. Bottom line

```
RESULT task1 vector_bound=837 line_bound=418 previous_line_bound=425 improvement_lines=7
       applies_to=both_problems scope=Leech_minimal_vectors_only record_lines=248
```

The class-problem interval narrows from `[248, 425]` to **`[248, 418]`** — for line systems inside
the Leech minimal vectors. In vectors: `[496, 837]` instead of `[496, 850]`.

## 8. Reproduce

```
PYTHONPATH=python .venv/bin/python tools/bounds/run_pm16.py          # the two SDP runs + certificates
.venv/bin/python tools/bounds/verify_line_vector_reduction.py        # verifier A
.venv/bin/python tools/bounds/recheck_sdp3_certificate.py data/scheme/sdp3_certificate.json
.venv/bin/python tools/bounds/recheck_sdp3_certificate.py data/scheme/sdp3_certificate_pm16.json
```

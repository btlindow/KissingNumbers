# 60°-free subsets of the Leech minimal vectors, and kissing numbers in dimensions 25–31

The best known kissing-number lower bounds in dimensions 25 through 31 all come from one
construction (Cohn–Jiao–Kumar–Torquato 2011, refined by Kallal–Kan–Wang and by Ma et al.): take
the 196,560 minimal vectors of the Leech lattice, choose a subset **S** in which no two vectors
subtend exactly 60°, and lift it into the extra dimensions. Every extra element of *S* is worth
one or more spheres in *every* one of those seven dimensions at once.

This repository is a computational study of how large such an *S* can be, and of what the largest
known examples actually look like.

## Results

| | |
|---|---|
| **K(27) ≥ 200540** | A new lower bound, +496 on the previous record of 200044. The R³ kissing configuration in the lifting template is the cuboctahedron, and both prior papers partitioned it into 2 triangles + 3 antipodal pairs (weight 7); it in fact partitions into **4 disjoint triangles** (weight 8). Found independently and at about the same time by Alexey Kravatsky. |
| **496 ≤ max&#124;S&#124; ≤ 837** | The upper bound is a three-point semidefinite program over the Terwilliger algebra of the 6-class association scheme, with an exact-rational dual certificate. *Every* two-point method — Delsarte LP, scheme LP, Lovász ϑ, Schrijver ϑ′, the ratio bound — gives exactly 9360/11 = 850.90…, so 837 is the first bound below that ceiling. |
| **At least 64 pairwise Co₀-inequivalent maximal 496-sets** | No plateau move between them is realised by any Leech automorphism; they fall into 8 congruence classes of 8. Relevant to the suggestion that 496 is optimal: whatever the maximum is, it is not attained uniquely. |
| **A structural account of the 496** | The known 496-sets live inside a Turyn decomposition ℝ²⁴ = B₁ ⊥ B₂ ⊥ B₃ with Λ ∩ Bᵢ ≅ √2·E₈, consist entirely of "duads", and are governed by a PG(3,2) of 15 *channels* carrying 8×8 rook grids, giving 496 = 4·(3·12 + 9·8 + 2·8). Every plateau move is a single channel-pair rewiring. |
| **K(25) ≥ 197058** | A verified +2, by lowering the lift height until the polar caps fit. Small, and probably subsumed by unpublished work; included because it is exact and rigid. |

No 497-element set was found, by any method tried. That negative is characterised rather than
merely asserted: see [`docs/reports/03-search-for-497.md`](docs/reports/03-search-for-497.md).

## Start here

* **[`docs/handover/findings.pdf`](docs/handover/findings.pdf)** — a 13-page summary written for
  a specialist reader. Every claim carries one of three labels (certificate attached / certificate
  held / search evidence), and the attached ones can be checked from the archive below.
* **[`leech-496-artifacts.zip`](leech-496-artifacts.zip)** — a self-contained archive (430 KB):
  the record set, the 64 maximal sets, the SDP dual certificate, and one NumPy-only verifier that
  regenerates the Golay code and all 196,560 minimal vectors from scratch and re-checks
  everything. It has no dependency on this repository.
* **[`REPRODUCE.md`](REPRODUCE.md)** — how to rebuild and re-verify every headline claim from a
  fresh clone, with expected outputs and timings.

## Layout

```
docs/reports/      detailed working notes, by topic (start with its README)
docs/handover/     the summary paper (LaTeX + PDF) and its clean-room test transcript
docs/note/         the dimension-27 record: note, certificate and standalone verifier
docs/background.md the problem, the construction template, and the literature
docs/design.md     data formats, APIs and the GPU engine design (cited from source comments)
data/              the Leech vectors, the record sets, certificates, disjoint families
python/            reference implementations, bounds (LP/SDP), verifiers, analysis tools
src/ include/ cuda/ tools/   C++ and CUDA: lattice generation, adjacency, GPU local search
tests/             ctest and pytest suites
```

## Verifying anything

Independent verification was the working rule throughout: every headline claim has at least two
implementations, in two languages where practical, and all certified statements are in exact
integer or rational arithmetic. The fastest check is the standalone one:

```
unzip leech-496-artifacts.zip && cd leech-496-artifacts && python3 verify.py --all
```

which needs only Python 3 and NumPy, takes about two minutes, and ends in a single `RESULT` line.

## Provenance and method

The work was carried out by AI agents (Claude) directed by the repository owner, over roughly a
week. Certified claims rest on exact arithmetic and on certificates a reader can re-check, not on
the agents' assertions; a two-verifier rule applied to every headline number, and it caught at
least one false positive (a driver bug that produced apparent 497–568 element sets, rejected by
both verifiers before anything was claimed). Several early conclusions were later corrected by
subsequent work, and the corrections are recorded in place rather than quietly fixed.

Ben Lindow · benlindow@gmail.com

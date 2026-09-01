# Background: the problem and the construction

This is the mathematical background the project started from, kept because it states the
problem, the construction template and the literature in one place. It was written before any
of the work below was done, so **where it disagrees with the results, the results win** — see
[`docs/reports/`](reports/) and [`docs/handover/findings.pdf`](handover/findings.pdf).

Two figures in the original text are now known to be wrong, and are corrected here:

* The number of "frames" of minimal vectors is **not** 8,292,375 — that is the count of Conway
  crosses. Minimal-vector frames number roughly 8.5x10^20 (the vectors of shape (±4,±4,0^22)
  alone give one frame per perfect matching of the 24 coordinates, i.e. 23!! = 316,234,143,225).
* The dimension-27 entry of the template table is not optimal: the cuboctahedron partitions into
  four disjoint triangles, not two triangles and three antipodal pairs, which is what gives
  K(27) >= 200540.

## 1. The problem

The kissing number K(n) is the maximum number of non-overlapping unit balls that can touch a central unit ball in R^n. Exact values are known only for n = 1, 2, 3, 4, 8, 24. In dimensions 25–31 every known record is built from the Leech lattice by one shared template (Cohn–Jiao–Kumar–Torquato 2011): take the 196,560 minimal vectors of the Leech lattice, choose a subset S in which no two vectors meet at exactly 60°, and "lift" S into the extra dimensions. The record in dimension 25 is exactly K(24) + |S| = 196560 + |S|. The size of the best known S has gone 480 (2011) → 488 (2018, high-schoolers with simulated annealing) → 496 (Nov 2025, PackingStar, a game-theoretic RL system). **The core task is a maximum-independent-set problem on a 196,560-vertex graph, currently at 496, with an instant exact verifier.** Finding 497 sets a new record in dimension 25 immediately and propagates to 26–31. A secondary deliverable is a rigorous upper bound on |S| (a small LP), which tells us whether 497 is even possible before we burn GPU time on it.

---

## 1. Background and state of the art

### 1.1 Kissing numbers

Equivalent formulation used throughout: K(n) is the maximum number of unit vectors in R^n with pairwise inner products ≤ 1/2 (pairwise angles ≥ 60°). Lower bounds are explicit constructions; upper bounds come from LP/SDP relaxations (Delsarte, Bachoc–Vallentin, de Laat–Leijenhorst).

Current bounds in the dimensions we care about (Henry Cohn's table, https://cohn.mit.edu/kissing-numbers/, last updated June 2026 — **re-fetch it before starting; it moves**):

| n  | lower bound | upper bound | source of lower bound |
|----|-------------|-------------|------------------------|
| 24 | 196560      | 196560      | Leech lattice (exact) |
| 25 | 197056      | 265006      | PackingStar 2025 |
| 26 | 198550      | 367775      | PackingStar 2025 |
| 27 | 200044      | 522212      | PackingStar 2025 |
| 28 | 204520      | 752292      | PackingStar 2025 |
| 29 | 209496      | 1075991     | PackingStar 2025 |
| 30 | 220440      | 1537707     | PackingStar 2025 |
| 31 | 238350      | 2213487     | PackingStar 2025 |

Upper bounds are astronomically far away. We are not solving anything; we are nudging lower bounds, which is a real, publishable contribution (short arXiv note + entry in Cohn's table).

### 1.2 The Leech lattice Λ24 and its minimal vectors

Use integer coordinates scaled by √8 so everything is an integer. In this scaling a Leech lattice vector x ∈ Z^24 satisfies:

1. all coordinates have the same parity m ∈ {0,1};
2. the set {i : x_i ≡ 2 (mod 4)} (if m = 0) or {i : x_i ≡ 3 (mod 4)} (if m = 1) is a codeword of the extended binary Golay code G24 (equivalently the complementary residue class is also a codeword);
3. Σ x_i ≡ 4m (mod 8).

The minimal vectors C have squared norm 32 (= 4 in the "norm 4" scaling used by the papers). There are exactly |C| = 196560 of them, in three shapes:

| shape | description | count |
|-------|-------------|-------|
| (±2^8, 0^16) | ±2 on the 8 coordinates of an octad (a weight-8 Golay codeword), even number of minus signs | 759 × 2^7 = 97152 |
| (∓3, ±1^23) | start from (−3, 1, …, 1) with the −3 in coordinate i; negate the coordinates lying in a Golay codeword c | 24 × 4096 = 98304 |
| (±4, ±4, 0^22) | two ±4s in any two coordinates | C(24,2) × 4 = 1104 |

Total 97152 + 98304 + 1104 = 196560. Self-checks: every generated vector must pass the membership test above, have squared norm 32, and the generated set must be closed under negation.

Inner products between two *distinct* minimal vectors take only the values {−32, −16, −8, 0, 8, 16} (i.e. cosines {−1, −1/2, −1/4, 0, 1/4, 1/2}). For a fixed minimal vector x the counts are:

| ⟨x,y⟩ | cos | count |
|-------|-----|-------|
| 32 | 1 | 1 (y = x) |
| 16 | 1/2 | 4600 |
| 8 | 1/4 | 47104 |
| 0 | 0 | 93150 |
| −8 | −1/4 | 47104 |
| −16 | −1/2 | 4600 |
| −32 | −1 | 1 (y = −x) |

(1 + 4600 + 47104 + 93150 + 47104 + 4600 + 1 = 196560.) Use this histogram as a unit test on the generator.

**Generating the Golay code.** Simplest robust route: build the cyclic (23,12,7) binary Golay code from a generator polynomial, e.g. g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1 (or its reciprocal x^11 + x^9 + x^7 + x^6 + x^5 + x + 1 — both generate a Golay code), enumerate all 2^12 codewords, append an overall parity bit to get length 24. Self-check: weight distribution must be exactly 1 (w=0), 759 (w=8), 2576 (w=12), 759 (w=16), 1 (w=24). If that check passes, the octads are the 759 weight-8 words and everything downstream is right.

Alternative data source for cross-checking: the Kallal–Kan–Wang repo (https://github.com/kenzkallal/Kissing-Numbers) contains `minvects.txt` (their ordering of the 196560 vectors) and `Vbasis.txt`. Their coordinate convention may differ from the one above; compare via the membership test and inner-product histogram, not coordinate-by-coordinate.

### 1.3 The construction template (CJKT 2011, Theorem 7.5; restated as Theorem 1.1 in Kallal–Kan–Wang)

Work in the norm-4 scaling for this statement (divide integer coordinates by √8). Let S_1, …, S_n be **mutually disjoint** subsets of C such that within each S_i, all distinct x, y satisfy ⟨x,y⟩ ≤ 1 (i.e. never 2 — never 60°). Let T_1, …, T_n be disjoint subsets of a kissing configuration in S^{d−1} (unit vectors, pairwise cos ≤ 1/2 across the whole configuration) such that within each T_i all pairs satisfy ⟨y,y'⟩ ≤ −1/2. (So |T_i| ≤ 3: an antipodal pair, or an equilateral triangle through the origin's plane.) Then the following set of norm-4 vectors in R^{24+d} has all pairwise inner products ≤ 2, i.e. is a kissing configuration:

- (x, 0) for x ∈ C \ ∪_i S_i,
- (x·√(2/3), y·√(4/3)) for x ∈ S_i, y ∈ T_i.

Count: 196560 + Σ_i (|T_i| − 1)·|S_i|.

Why each constraint is needed (useful when designing verifiers and when looking for slack):
- same S_i, same x, two y's in T_i: (2/3)·4 + (4/3)⟨y,y'⟩ ≤ 2 ⟺ ⟨y,y'⟩ ≤ −1/2;
- same S_i, distinct x,x': (2/3)⟨x,x'⟩ + (4/3)·1 ≤ 2 ⟺ ⟨x,x'⟩ ≤ 1;
- different S_i, S_j: (2/3)·2 + (4/3)·(1/2) = 2, so disjointness plus the 60° condition on T is exactly enough;
- equatorial vs lifted: √(2/3)·2 < 2, automatic.

**Dimension 25** (d = 1, T = {+1, −1}, one set S): K(25) ≥ 196560 + |S|. So |S| is the whole game there. Record: |S| = 496 → 197056.

**Dimensions 26–31.** The 2018 template was (Kallal–Kan–Wang Table 3):

| n | bound |
|---|-------|
| 26 | 196560 + 2|S_1| + 2|S_2| |
| 27 | 196560 + 2|S_1| + 2|S_2| + Σ_{i=3..5}|S_i| |
| 28 | 196560 + 2Σ_{i=1..8}|S_i| |
| 29 | 196560 + 2Σ_{i=1..8}|S_i| + Σ_{i=9..16}|S_i| |
| 30 | 196560 + 2Σ_{i=1..24}|S_i| |
| 31 | 196560 + 2Σ_{i=1..24}|S_i| + Σ_{i=25..51}|S_i| |

PackingStar (2025) improved the template in two ways: (a) it adds an extra K(n−24) spheres of the form (0, 2y') sitting purely in the extra d dimensions, with y' chosen at angle ≥ 30° from every T-vector (check: ⟨(0,2y'), (x√(2/3), y√(4/3))⟩ = (4/√3)⟨y',y⟩ ≤ 2 ⟺ ⟨y',y⟩ ≤ √3/2); (b) it partitions the R^d kissing configuration into more equilateral triangles (|T_i| = 3, weight 2 per S_i) — 12 triangles from the 40-point configuration in R^5, 42 triangles from the 126-point configuration in R^7. Their reported forms:

| n | form | value |
|---|------|-------|
| 25 | K(24) + |S_1| | 197056 |
| 26 | K(2) + K(24) + 2|S_1| + 2|S_2| | 198550 |
| 27 | K(3) + K(24) + 2|S_1| + 2|S_2| + Σ_{3..5}|S_i| | 200044 |
| 28 | K(4) + K(24) + 2Σ_{1..8}|S_i| | 204520 |
| 29 | K(5) + K(24) + 2Σ_{1..12}|S_i| + Σ_{13..14}|S_i| | 209496 |
| 30 | K(6) + K(24) + 2Σ_{1..24}|S_i| | 220440 |
| 31 | K(7) + K(24) + 2Σ_{1..42}|S_i| | 238350 (v1 of the paper only reached 238078 "due to computational limits"; Cohn's table lists 238350, so confirm which S_i data actually exist) |

With |S_i| = 496 for all i these forms evaluate to exactly the listed values (e.g. 31: 126 + 196560 + 2·42·496 = 238350). Every one of those numbers is linear in the |S_i|, so **a 497 propagates to all seven dimensions.**

### 1.4 The record 496 and its structure (important)

PackingStar reports that its 496-vector S is not random-looking: it is 28 "cross" structures in 8-dimensional subspaces (16 vectors each = 8 mutually orthogonal antipodal pairs) plus one 24-dimensional cross (48 vectors = a full orthogonal frame of the Leech lattice, a "Conway–Curtis cross"). 28·16 + 48 = 496. Inside a cross all inner products are 0 or −4, which is allowed; the constraint only bites between crosses. **They conjecture 496 is optimal.** Take that seriously — it's the reason Workstream 1 (upper bound) comes before Workstream 2 (search). Their data repository: https://github.com/CDM1619/PackingStar.

---

## 2. The core discrete problem, stated precisely

**Conflict graph G.** Vertices: the 196560 minimal vectors (integer coordinates, squared norm 32). Edge between x and y iff ⟨x,y⟩ = 16 (cos = 1/2, angle 60°). G is regular of degree 4600 and vertex-transitive (the automorphism group Co_0 of the Leech lattice, order ≈ 8.3 × 10^18, acts transitively on C and preserves inner products). Note ⟨x,y⟩ = 16 ⟺ x − y is also a minimal vector, which is a handy structural fact.

**Objective.** Find an independent set S in G (no two vectors at 60°) of size ≥ 497. Antipodal pairs {x, −x} are *not* adjacent (⟨x,−x⟩ = −32), so S may and typically will be antipodal.

**Verifier (exact, integers).** With V the (|S| × 24) int matrix of chosen vectors: every row has squared norm 32, all rows distinct, and every off-diagonal entry of V·Vᵀ is ≤ 8. That's the whole certificate for the dimension-25 claim K(25) ≥ 196560 + |S|.

**Secondary objective (dims 26–31).** Given the largest S found, produce up to 42 mutually disjoint independent sets S_1..S_42, each as large as possible (ideally all equal to |S|, obtained as images g·S under Leech automorphisms g ∈ Co_0). Any total Σ|S_i| exceeding what the current record data achieves is a record in the corresponding dimension.

---


## 5. Verification protocol and certificate format

For any claimed S:
1. `S.txt`: one vector per line, 24 integers, squared norm 32, in our coordinates.
2. `verify_S.py`: regenerates C independently, checks every line of S.txt is in C, checks distinctness, computes the Gram matrix in integer arithmetic, asserts all off-diagonal entries ≤ 8, prints |S|.
3. `verify_dim25.py`: builds the explicit 25-dimensional configuration of 196560 + |S| vectors and checks all pairwise cosines ≤ 1/2 (exact casework + float sanity check), prints the count.
4. For 26–31: `S_i.txt` files plus a verifier that checks disjointness, independence of each S_i, and builds and checks the full 24+d configuration explicitly, including the extra K(d) spheres and the T_i partition.

A record that can't be verified by a stranger in under a minute from the text files is not a record.

---

## 6. If we find something

1. Verify twice with independently written code.
2. Write a 2–4 page note in the style of Kallal–Kan–Wang: statement, construction, the vector list as a data file, verification script. arXiv math.MG (cross-list math.CO).
3. Email Henry Cohn (cohn@mit.edu) so the table gets updated; he maintains it and explicitly asks to be told of improvements.
4. Same process for a rigorous |S| ≤ 496 bound if W1 lands there — it settles a conjecture stated in the PackingStar paper.

---

## 7. References

- H. Cohn, Table of kissing number bounds — https://cohn.mit.edu/kissing-numbers/ (data archive: https://hdl.handle.net/1721.1/153312)
- H. Cohn, Y. Jiao, A. Kumar, S. Torquato, *Rigidity of spherical codes*, Geom. Topol. 15 (2011) — arXiv:1102.5060. Section 7 has the construction (Thm 7.5, Lemma 7.4).
- K. Kallal, T. Kan, E. Wang, *Improved lower bounds for kissing numbers in dimensions 25 through 31* — arXiv:1608.07270. Data: https://github.com/kenzkallal/Kissing-Numbers. Source of the 488, the greedy/annealing baselines, and the disjoint-family construction.
- C. Ma et al., *Finding kissing numbers with game-theoretic reinforcement learning* — arXiv:2511.13391. Source of the 496 and the improved template. Data: https://github.com/CDM1619/PackingStar.
- J. H. Conway, N. J. A. Sloane, *Sphere Packings, Lattices and Groups*, 3rd ed. — Leech lattice construction (Ch. 4, 10), Golay code, crosses/frames.
- F. Celler, C. R. Leedham-Green, S. H. Murray, A. C. Niemeyer, E. A. O'Brien, *Generating random elements of a finite group*, Comm. Algebra 23 (1995) — product replacement, for random Co_0 elements.
- D. de Laat, N. Leijenhorst, *Solving clustered low-rank semidefinite programs arising from polynomial optimization*, Math. Program. Comput. 16 (2024) — the current upper bounds; relevant if W1 goes to SDP.
- For MIS local search: Andrade–Resende–Werneck, *Fast local search for the maximum independent set problem*, J. Heuristics 18 (2012); Cai et al., NuMVC (JAIR 2013).

---

## 8. Parking lot (ideas not yet vetted)

- The conflict edge {x,y} always produces a third minimal vector x − y with ⟨x, x−y⟩ = 16 as well; conflict edges come in structured triples. Might give a cheap way to enumerate maximal independent sets with prescribed symmetry.
- An S with no inner product 16 *or* 8 (pairwise cos ≤ 0, i.e. only 0, −1/4, −1/2, −1) would allow a different lift with a bigger scale factor; check whether any variant of the template rewards "more orthogonal" S enough to beat 496 with a smaller set.
- Non-antipodal S has never been seriously searched (both record efforts restricted to ±pairs). The 13-dimensional results in the PackingStar paper show fully non-antipodal record-matching configurations exist elsewhere.
- If W1's LP bound is far above 496, the graph may simply be hard; consider exact MIS on the quotient by a large subgroup (W2d) as a way to get *provable* statements about symmetric solutions.
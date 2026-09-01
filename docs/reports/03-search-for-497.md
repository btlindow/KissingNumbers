# The search for a 497-point configuration

The kissing number of the Leech lattice is 196560, and every one of its 196560 minimal vectors is a
unit vector at 60° or more from the others. The question pursued in this document is the *local* one
that governs the record for dimension 25 and above: how many of those 196560 minimal vectors can be
chosen so that no two of them are at exactly 60° to each other? Such a set `S` lifts to a kissing
configuration in dimension 25 (`K(25) ≥ 196560 + 2·|S| + ...`; see
`05-lifting-template-and-families.md` for the lifting machinery and
`06-record-and-priority.md` for the record it produces). The published record configuration has
`|S| = 496`. The whole of the work recorded here is an attempt — by six independent methods — to reach
**497**, and every one of them failed. The point of this document is to say exactly *what* was
searched, exactly what was proved along the way, and exactly where the boundary between proof and
search evidence lies.

Nothing in this document produces a 497. `runs/found/` — the repository's record-custody directory —
is empty of anything above 496 at the end of every run described below.

Companion documents: `01-lattice-and-verification.md` (the Golay/Leech construction, the canonical
order on the 196560 minimal vectors, the adjacency table `data/adj.u32`, the GPU tightness kernel and
the two independent verifiers), `02-upper-bounds.md` (Delsarte LP, the two-point limit 9360/11 = 850,
the three-point Terwilliger SDP bound 837), `04-structure-of-the-496.md` (the Turyn √2·E8³
decomposition of the record, its monads/duads/triads, the PG(3,2) of channels, the rook grids, and
the structural reading of the plateau moves established below), and `REPRODUCE.md` at the repository
root for how to re-run everything.

---

## 1. Notation, and what a local move can be

Write `C` for the canonical list of the 196560 minimal vectors (in the norm-32 / √8-integer scaling,
so that two minimal vectors are at 60° exactly when their inner product is ±16). Two vectors are
**adjacent** (in conflict) when their inner product is 16. For a candidate set `S ⊆ C`:

* `conf(v) = {s ∈ S : ⟨v, s⟩ = 16}` — the members of `S` that block `v`;
* `tight[v] = |conf(v)|` — the **tightness** of `v`;
* `v` is **free** when `tight[v] = 0` and `v ∉ S`; a free vertex is exactly a one-vector extension of
  `S`, i.e. a 497 when `|S| = 496`.

A **(k, m)-swap** removes a subset `R ⊂ S` with `|R| = k` and adds `m` mutually non-adjacent vertices
from the pool `P(R) = {v ∉ S : conf(v) ⊆ R}`. Its gain is `m − k`. A **plateau move** is an
equal-size swap (`m = k`, necessarily with the added vertices' conf sets covering `R` exactly). A
plateau move is **connected** when the bipartite graph "added ↔ removed" is connected; otherwise it is
a disjoint union of smaller plateau moves.

Both reference sets used throughout are **antipodal** (closed under `v ↦ −v`), which implies
`conf(−v) = −conf(v)`; the tooling checks this on every set it touches (`antipodal : ... 0 of N
candidates` violate it on both records).

The two reference sets are `data/S496.txt` (the record, 496 vectors) and `data/S488.txt` (a second,
smaller maximal set kept as a control, 488 vectors).

**The reduction that makes exhaustive swap search possible.** If `I ⊆ P(R)` is independent then
`I ⊆ P(R′)` for `R′ = ⋃_{v∈I} conf(v) ⊆ R`. So *every* swap with `|R| ≤ K` is witnessed by an `R`
that is a **union of conf sets**, and a search is complete for removal size `K` iff every such union
of size ≤ K is visited and its pool solved exactly. This is what turns "all `C(496, 12)` removals"
(astronomically large) into "all unions of conf sets of size ≤ 12" (900,672 of them).

---

## 2. Exhaustive k-swap neighbourhoods

The classical local-search neighbourhoods for maximum independent set are the (1,2)- and (2,3)-swaps.
On the 496 they are **empty for a trivial reason**: the minimum tightness outside the 496 is 4, so
there is no vertex of tightness 1, 2 or 3 to build them from. The search was therefore generalised to
exhaustive `(k, m)`-swap neighbourhoods for all `k ≤ 12`.

### 2.1 Implementation

| File | Purpose |
|---|---|
| `include/kiss/swaps.h`, `src/swaps.cpp` | Library: abstract `SwapGraph` (Leech conflict graph via mmap rows plus exact `dot`, or explicit lists for tests), `tightness_from_graph`, histograms, `greedy_maximal_set`, exact bitset MIS (`max_independent_subset`), `find_swap12` / `find_swap23` (the classical routines), the generalised `kswap_search`, `group_by_conf` (conf-set structure of a tightness class), `apply_swap`, `swap_is_valid`. CPU only, OpenMP. |
| `tools/swapsearch.cpp` | `swapsearch <S.txt> \| --greedy SEED [--kmax K=8] [--time-limit s=120] [--max-sets N=1e7] [--list N] [--struct T] [--no-classical] [--data DIR] [--dump-plateau FILE]`. Prints the histogram, the antipodal check, (1,2)/(2,3) statistics, the conf-set structure of the least-tight class, one line per k, plateau examples, and a `RESULT ...` line. Any improving swap is applied, re-verified (Gram ≤ 8) and written to `runs/found/S_<size>_<utc>_swap.txt`. |
| `tests/test_swaps.cpp`, `tests/tasks/T3_1.cmake` | Acceptance test `test_swaps` (CPU; exit 77 if `data/adj.u32` is absent, synthetic checks still run). |
| `runs/swaps/logs/*.log` (gitignored) | Full outputs of the runs below. `runs/swaps/greedy/` holds the five improved greedy sets, moved out of `runs/found/`, which is reserved for records. |

`kswap_search` works level by level: the unions of size `k` are generated from the stored unions of
size `r < k` by adding one conf set with `|conf \ R| = k − r` (candidates of tightness `k − r` must be
disjoint from `R`, scanned from the per-tightness list; candidates of larger tightness must meet `R`,
found through the per-`S`-member candidate lists), deduplicated in a hash set. Each union's pool is
the set of candidates whose first conf member is in `R` and whose conf set lies in `R`; the pool's
maximum independent set is computed exactly by a bitset branch-and-bound (pools here are ≤ 26).
OpenMP parallelises over unions (chunks of 4096) and over the generation. A level cut short by the
per-level time limit or the `max_sets` cap is reported non-exhaustive, as are all higher levels;
`kmax_exhaustive` is the largest `k` for which levels 1..k are complete.

Each union `R` with `MIS(P(R)) ≥ k` is classified by the *actual* union `R′` of the chosen vertices'
conf sets: gain > 0 gives an improving swap, gain 0 (necessarily `R′ = R`) a plateau move. Plateau
moves are counted per `R` (one maximum independent set per pool), split into connected and composite.

### 2.2 The 496 is swap-optimal for every k ≤ 12

Tightness histogram outside `S` (identical to the verification tooling's):

```
4:80 6:640 7:256 8:2704 9:8064 10:31424 11:52672 12:49552 13:31360 14:13440 15:2560
16:1480 17:256 18:704 19:64 20:528 22:128 24:152
```

free = 0, minimum tightness 4. The classical routines confirm the triviality: (1,2) has 0 hits, all
`L_x` empty; (2,3) has 0 hits, all 122,760 pools empty.

`swapsearch data/S496.txt --kmax 12 --time-limit 600`; all levels exhaustive. Times are from a 36 s
run on an idle machine (the 156 s run overlapped with other jobs).

| k | cand tight=k | distinct conf sets | unions R examined | largest pool | mean pool | best m = MIS(P(R)) | plateau (connected) | improving | exhaustive | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | – | – | – | 0 | 0 | yes | 0 |
| 2 | 0 | 0 | 0 | – | – | – | 0 | 0 | yes | 0 |
| 3 | 0 | 0 | 0 | – | – | – | 0 | 0 | yes | 0 |
| 4 | 80 | 80 | 80 | 1 | 1.00 | 1 | 0 | 0 | yes | 0.01 |
| 5 | 0 | 0 | 0 | – | – | – | 0 | 0 | yes | 0 |
| 6 | 640 | 640 | 720 | 2 | 1.11 | 2 | 0 | 0 | yes | 0.04 |
| 7 | 256 | 256 | 448 | 2 | 1.43 | 2 | 0 | 0 | yes | 0.05 |
| 8 | 2704 | 2704 | 5620 | 4 | 1.54 | 4 | 0 | 0 | yes | 0.5 |
| 9 | 8064 | 8064 | 11360 | 3 | 1.33 | 3 | 0 | 0 | yes | 1.0 |
| 10 | 31424 | 31424 | 91808 | 6 | 1.73 | 6 | 0 | 0 | yes | 8 |
| 11 | 52672 | 52672 | 123944 | 8 | 1.72 | 8 | 0 | 0 | yes | 11 |
| 12 | 49552 | 49552 | 666692 | 12 | 2.14 | 12 | **4 (4)** | 0 | yes | 15 |

Unions of size ≤ 12: 900,672.

```
RESULT ok=1 size=496 improving=0 new_size=496 kmax=12 kmax_exhaustive=12
plateau_moves=4 plateau_connected=4 free=0 min_tight=4 unions=900672 antipodal=1 found_file=- total_s=36.53
```

(`kmax 8` alone: `... kmax_exhaustive=8 plateau_moves=0 ... unions=6868 ... total_s=0.27`.)

Two observations:

* Every vertex of tightness ≤ 12 has a **distinct** conf set (distinct = cand at every k). The 496 is
  extremely rigid: for `k ≤ 11` the best a k-removal can buy back is `m < k`; the best ratio is
  `m = k − 3`, attained at `k = 11, m = 8` and at `k = 8, m = 4`.
* The **only (k,k)-moves with k ≤ 12 are four (12,12) moves**, each removing 12 members of `S` and
  adding 12 vertices of tightness exactly 4:

```
plateau (12,12): remove S pos {8,14,101,103,127,209,286,368,392,394,481,487}  add [4479 4901 26476 27189 52296 85215 111344 144263 169370 170083 191658 192080]
plateau (12,12): remove S pos {30,36,119,144,148,172,323,347,351,376,459,465} add [13131 17682 24333 39173 55836 94360 102199 140723 157386 172226 178877 183428]
plateau (12,12): remove S pos {43,51,53,89,223,234,261,272,406,442,444,452}  add [7694 19999 45656 61824 70844 87859 108700 125715 134735 150903 176560 188865]
plateau (12,12): remove S pos {84,133,138,162,189,203,292,306,333,357,362,411} add [33755 45348 63855 70602 73008 79976 116583 123551 125957 132704 151211 162804]
```

(vertex indices into the canonical order; `S` positions into the sorted index list of
`data/S496.txt`). They are pairwise disjoint — 48 members of `S`, 48 added vertices — and each is
closed under negation. These four moves are the seed of everything in §3 and §4.

### 2.3 Structure of the 80 tightness-4 vertices

From `swapsearch data/S496.txt --struct 4 --list 80` (analysis in
`runs/swaps/logs/run496_struct.log`):

* 80 vertices, **80 distinct conf sets**, each a set of 4 **mutually orthogonal** members of `S`
  (`gram(conf) = 0` for all six pairs, in every group). Shapes of the 80: 40 of octad type
  (2⁸, 0¹⁶)/√8-scaled and 40 of type (∓3, ±1²³) — none of type (±4, ±4).
* Gram among the 80: `-32:40 -8:640 0:1840 8:640` — they form **40 antipodal pairs and are mutually
  non-adjacent** (no inner product 16). So the 80 tightness-4 vertices are themselves an independent
  set; adding them all would require removing the 112 members of `S` they touch.
* Conf-set overlaps: over the 3160 pairs, `|conf_a ∩ conf_b|` = 0 in 2888 cases, 1 in 192, 2 in 80.
  They cover **112 of the 496** members: 48 members lie in 4 conf sets each, 64 in 2 each.
* The intersection graph of the conf sets has **12 connected components**, closed under negation:
  * **4 components of 12 conf sets on 12 members of S** (each member in exactly 4 sets; pairwise
    overlaps 0:6, 1:48, 2:12) — exactly the four (12,12) plateau moves above. Each is a 12-point
    structure with 12 blocks of size 4, every point on 4 blocks.
  * **8 components of 4 conf sets on 8 members** (each member in 2 sets; overlaps: two disjoint
    pairs, four pairs meeting in 2): two disjoint 4-sets `A ∪ B = C ∪ D` = the 8 members. Their
    pools have MIS 4 against `|R| = 8`, i.e. an (8,4) move — the reason the `k = 8` level has largest
    pool 4 and best `m = 4`.

  The eight small components, as `S` positions: {2,130,194,214,281,301,365,493},
  {15,86,94,171,324,401,409,480}, {24,59,134,218,277,361,436,471},
  {41,109,142,211,284,353,386,454}, {66,71,121,156,339,374,424,429},
  {96,155,168,225,270,327,340,399}, {108,117,132,140,355,363,378,387},
  {116,131,179,183,312,316,364,379}; the four 12-sets are those listed with the plateau moves.
* No conf set contains an antipodal pair (`conf(v) antipode-free for 80`), consistent with the
  members being orthogonal.

### 2.4 The 488 (control)

Tightness histogram: `2:48 4:208 5:256 6:608 7:576 8:3850 9:16376 10:35712 11:47136 12:43636
13:26648 14:13408 15:3216 16:2698 17:736 18:340 19:32 20:348 22:12 24:228`; free 0, minimum
tightness 2 (48 vertices). (1,2): 0 hits (no tightness-1 vertex). (2,3): 0 hits; of 118,828 pairs,
24 have `|P| = 2` and none has `|P| ≥ 3`. The tightness-2 structure: 48 vertices in **24 distinct
conf sets** (2 members each, pairwise orthogonal conf members, pairwise disjoint conf sets covering
48 members of `S` once); the two members of a group are non-adjacent, giving 24 connected
**(2,2)-plateau moves** and no (2,3).

`swapsearch data/S488.txt --kmax 12 --time-limit 600 --max-sets 1e7`; level 12 hit the 10 M union cap
during generation and is reported non-exhaustive. (`kmax 8` alone takes 1.2 s.)

| k | cand tight=k | distinct conf | unions examined | largest pool | mean pool | best m | plateau (connected) | improving | exhaustive | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | – | – | – | 0 | 0 | yes | 0 |
| 2 | 48 | 24 | 24 | 2 | 2.00 | 2 | 24 (24) | 0 | yes | 0.01 |
| 3 | 0 | 0 | 0 | – | – | – | 0 | 0 | yes | 0.01 |
| 4 | 208 | 204 | 480 | 4 | 2.73 | 4 | 276 (0) | 0 | yes | 0.04 |
| 5 | 256 | 256 | 256 | 1 | 1.00 | 1 | 0 | 0 | yes | 0.06 |
| 6 | 608 | 604 | 8164 | 6 | 3.53 | 6 | 2024 (0) | 0 | yes | 0.14 |
| 7 | 576 | 576 | 7936 | 4 | 2.76 | 4 | 0 | 0 | yes | 0.8 |
| 8 | 3850 | 3850 | 129337 | 8 | 4.16 | 8 | 10639 (**13**) | 0 | yes | 6.8 |
| 9 | 16376 | 16376 | 208536 | 6 | 3.40 | 6 | 0 | 0 | yes | 31 |
| 10 | 35712 | 35712 | 1981780 | 10 | 4.71 | 10 | 42816 (0) | 0 | yes | 142 |
| 11 | 47136 | 47136 | 4365552 | 9 | 4.25 | 8 | 0 | 0 | yes | 333 |
| 12 | 43636 | 43636 | 0 of 3486900 | – | – | – | – | – | **no** (cap) | – |

```
RESULT ok=1 size=488 improving=0 new_size=488 kmax=12 kmax_exhaustive=11 plateau_moves=55779
plateau_connected=37 free=0 min_tight=2 unions=10188965 antipodal=1 found_file=- total_s=518.91
```

(`kmax 8`: `... kmax_exhaustive=8 plateau_moves=12963 plateau_connected=37 ... unions=146197 ... total_s=1.17`.)

The composite counts are exactly the disjoint unions of the primitive moves —
276 = C(24,2), 2024 = C(24,3), 10626 = C(24,4), 42816 = C(24,5) + 24·13 — so the 488 has **37
primitive plateau moves** for `k ≤ 11`: 24 of size 2 (adding two tightness-2 vertices) and 13 of size
8 (adding eight tightness-4 vertices). Examples (all 24 (2,2) moves and 8 of the 13 (8,8) moves are
in `runs/swaps/logs/final488_k8.log`):

```
plateau (2,2): remove [24489 83334] (S pos {62,203}) add [5170 141462]
plateau (2,2): remove [1389 112420] (S pos {2,281}) add [8948 54571]
plateau (8,8): remove S pos {32,100,103,164,323,384,387,455} add [4620 58050 61151 74868 121691 135408 138509 191939] tight [4 ×8]
plateau (8,8): remove S pos {54,96,190,212,275,297,391,433} add [21793 40645 78257 87297 109262 118302 155914 174766] tight [4 ×8]
```

So the 488 is (2,3)-swap-optimal and in fact swap-optimal for all `k ≤ 11`; its (2,2) moves give a
plateau of ≥ 2²⁴ equal-size neighbours reachable by commuting moves.

### 2.5 Random greedy maximal sets, for contrast

`swapsearch --greedy SEED --kmax 8 --time-limit 30`:

| seed | \|S\| | tight-1 | (1,2) hits | (2,3) hits | k=1 improving / plateau | k=2 R / improving / plateau (conn.) | k=3 exhaustive | best swap found | new size | verified |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 228 | 377 | 97 | 13128 | 97 / 69 | 14832 / 11910 / 2619 (486) | no (time) | (2,5) | 231 | verify_s ok, verify_S.py ok |
| 2 | 231 | 355 | 78 | 11361 | 78 / 92 | 15429 / 10621 / 4464 (570) | no (time) | (2,4) | 233 | ok, ok |
| 3 | 236 | 244 | 54 | 7502 | 54 / 89 | 11448 / 6646 / 4261 (631) | no (time) | (3,7) | 240 | ok, ok |
| 4 | 234 | 281 | 70 | 9021 | 70 / 72 | 11398 / 7916 / 2963 (558) | yes (22 s; k=4 hit the 10 M cap) | (3,6) | 237 | ok, ok |
| 5 | 225 | 466 | 111 | 16417 | 111 / 78 | 18533 / 15235 / 3097 (341) | no (time) | (2,5) | 228 | ok, ok |

The generalised search agrees with the classical routine at `k = 1` (improving-`R` count equals the
(1,2) hit count, also checked in the test), finds far better moves at `k = 2, 3` (gain up to 4), and
the neighbourhood of a greedy set explodes (≈ 10⁶ unions of size 3, ≥ 10⁷ of size 4) — the exact
opposite of the records. Each improved set was re-verified with `build/t31/tools/verify_s` and
`.venv/bin/python python/verify_S.py` (all `RESULT ok=1`, `max_offdiag=8`) and moved to
`runs/swaps/greedy/`.

### 2.6 Acceptance test

`tests/test_swaps.cpp` covers: (1) synthetic graphs — A (planted (1,2), MIS of the pool
{1,2,3,4,5} is {1,2,5}), B (planted (2,3), no (1,2), two (1,1) plateaus), C (K4: only a plateau),
D (free vertex reported), E (connected (2,2) plateau), F (composite (2,2) plateau = two (1,1) moves)
— classical and generalised routines agree, every reported swap validates with `swap_is_valid` and
`apply_swap` + `is_independent`; (2) the 496 — histogram matches the verification tooling's string
exactly, tight = 0 on `S`, no (1,2)/(2,3), kswap exhaustive to `k ≤ 4` with 80 distinct conf sets,
pools of size 1, no plateau, no improvement; (3) a greedy set (seed 20260825, `|S| = 231`) — (1,2)
finds a swap, kswap improves at `k = 1` with the same `R` count as the classical hits, best swap
valid, improved set re-checked by exact inner products.

```
$ ctest --test-dir build/t31 -R swaps
100% tests passed, 0 tests failed out of 1
$ build/t31/tests/test_swaps data | tail -1
RESULT ok=1 synthetic=1 leech=1 s496=1 hist496_match=1 greedy_size=231 greedy_gain=6 failures=0
```

### 2.7 Engineering notes

* `kswap_search` originally generated all unions up to `kmax` eagerly, which made low levels
  non-exhaustive on loose sets for no reason. It now generates level by level, so `kmax_exhaustive`
  is meaningful and memory is bounded by the unions actually needed (peak RSS 180 MB for the 496 at
  `kmax = 12`, 1.0 GB for the 488 at the 10 M cap).
* Plateau counts are per union `R` (one MIS per pool), not per distinct added set; for the records
  the pools are tiny (≤ 12) so the distinction is immaterial.
* The 496's (12,12) moves and the 488's (8,8)/(2,2) moves are "structural": they exchange a block of
  `S` for the tightness-4 (resp. tightness-2) vertices hanging off it. No such exchange with a
  surplus exists up to `k = 12` / `k = 11`. **Any improvement on the 496 must remove at least 13
  vectors.** The structural reading of these moves — that they are channel rewirings inside the
  Turyn decomposition — is in `04-structure-of-the-496.md`.

---

## 3. The plateau graph

The exhaustive swap search leaves one obvious question: the 496 has equal-size neighbours, so is one
of *them* improvable? The plateau graph has as nodes the independent sets of the current size and as
edges the connected `(k,k)`-plateau moves with `k ≤ kmax`; walking it and running the exhaustive
`(k, k+1)` search of §2 at every node answers that question for the part of the plateau the walk can
see.

### 3.1 Implementation

| File | Purpose |
|---|---|
| `include/kiss/plateau.h`, `src/plateau.cpp` | `canonical_hash` (SHA-256 of the sorted index list), `plateau_move_is_connected` (union-find on the bipartite graph add ↔ remove), `swaps_disjoint` / `merge_swaps` / `disjoint_move_combinations` (all non-empty subsets of pairwise-disjoint moves, singles first; only singles when a node has more than `combo_max_moves` moves), and `plateau_walk`. Written against `SwapGraph`, so it runs on synthetic graphs in the test and on the Leech conflict graph. |
| `tools/plateau_walk.cpp` | `plateau_walk <S.txt> [--kmax K=12] [--node-time SEC=60] [--max-nodes N=200] [--wall SEC=7200] [--out DIR] [--combos M=4] [--max-sets N] [--no-verify]`. One progress line per expanded node, a line per new fingerprint class (with its moves when ≤ 16), class summary, depth profile, `RESULT ...`. With `--out DIR`: `DIR/nodes.tsv`, `DIR/summary.txt`, `DIR/sets/node_<id>.txt` (root, first node of each fingerprint class, improved sets). An improving swap is applied at once, written to `runs/found/S_<size>_<utc>_plateau.txt`, and re-verified by running `tools/verify_s` and `python/verify_S.py` as subprocesses. |
| `tools/swapsearch.cpp` (extended) | `--dump-plateau FILE`: machine-readable dump of the stored plateau / improving moves (`plateau k=<k> connected=<0\|1> remove=<i,...> add=<u,...>`). |
| `tests/test_plateau.cpp`, `tests/tasks/T3_1b.cmake` | ctest `test_plateau` (CPU; exit 77 if `data/adj.u32` is absent, synthetic checks still run). |
| `runs/plateau/` (gitignored) | `logs/*.log`, `s496_k12/`, `s488_k8/` (nodes.tsv, summary.txt, sets/), `moves488_k8.txt` (dump). |

`plateau_walk(g, S0, opt, extra, hooks)`:

1. **Discover** a set: sort, `canonical_hash` (dedupe), tightness vector and histogram outside `S`,
   minimum tightness, counts of tightness-1/2/3 vertices, and the discovery-time invariant
   `fp0 = tightness histogram | extra(S)`, where the tool supplies `extra` = Gram histogram of `S`
   plus the Gram histogram among the minimum-tightness vertices. All of these are Co₀-invariants, so
   equal `fp0` is *necessary* for equivalence under Co₀; the monomial-only shape counts (octad /
   (∓3,±1²³) / (±4,±4)) are tracked separately as `fp_shapes`.
2. **Pick** the frontier node with the largest size, then the `fp0` expanded least often so far
   (breadth over inequivalent classes), then the shallowest, then the oldest.
3. **Expand**: `kswap_search` with `k ≤ kmax` (per-level time limit `node_time`, `keep_moves` = 4096
   stored moves per level). An improving swap is validated (`swap_is_valid`), applied, handed to
   `on_improve` (the tool writes it to `runs/found/` and runs both verifiers), and the improved set
   becomes a new node, expanded next because it is larger. Otherwise the connected plateau moves of
   all levels (each re-validated against the graph) are the node's moves; the neighbours are the
   moves themselves and, when the node has ≤ `combos` moves, every subset of pairwise-disjoint moves
   applied simultaneously (the 496's four disjoint moves give 15 neighbours). After expansion the
   full fingerprint is `fp = fp0 | per-k plateau counts | per-k connected counts | kmax_exhaustive`.
4. Stop when the frontier is empty (component closed under the moves considered), `max_nodes` nodes
   have been expanded, or the wall budget is exceeded (checked between nodes).

Every set the walk touches is independent by construction; the test re-checks this with
`is_independent`, and the tool re-verifies improved sets with `verify_independent` in process and
with the two external verifiers.

### 3.2 Summary of both walks

**No improvement was found anywhere.** 424 nodes were expanded in total, each with the exhaustive
`(k, k+1)`-swap search up to its `kmax`; none has an improving swap, and no node anywhere has a free
(tightness-0) or tightness-1 vertex. Nothing was written to `runs/found/`.

| walk | kmax | nodes expanded | sets discovered | frontier left | verdict |
|---|---|---|---|---|---|
| 496 (`data/S496.txt`) | 12 | **16 (all)** | 16 | **0** | the component under moves with `k ≤ 12` is closed and exhausted: the 4-cube on the four commuting (12,12) moves; every one of its 16 sets is swap-optimal for `k ≤ 12` |
| 488 (`data/S488.txt`) | 8 | 408 (wall budget) | 7529 | 7121 | the component is ≥ 2³⁷ sets (37 pairwise-disjoint, hence commuting, primitive moves) and cannot be exhausted; the 408-node sample is uniformly 37-regular and improvement-free |

**Scope note, and a claim that had to be weakened.** `kmax = 12` only sees plateau moves of size
`k ≤ 12`. The GPU local search (§4), run in antipodal mode, later found **two further plateau moves
of size (80,80)** on the 496 — `runs/ls_search/plateau_atoms_496.json` holds all six atoms: 4 of size
12 plus 2 of size 80 — which commute with the four (12,12) moves and with each other. **The true
plateau component of the 496 has at least 64 sets, not 16.** This walk cannot see the (80,80) moves
(an exhaustive `k ≤ 80` search is far out of reach), so what is established below is precisely the
`k ≤ 12` statement: the sub-component generated by plateau moves of size ≤ 12 is exactly the 4-cube
on 16 sets. Nothing here should be read as "the 496's plateau component is closed" — only "closed
under plateau moves of size ≤ 12". The 4-cube is the `k ≤ 12` shadow of a 6-cube.

### 3.3 The 496 walk

`plateau_walk data/S496.txt --kmax 12 --node-time 900 --max-nodes 200 --wall 7200 --out
runs/plateau/s496_k12` (log `runs/plateau/logs/s496_k12.log`, tables `runs/plateau/s496_k12/`;
10 OpenMP threads, run concurrently with the 488 walk and other jobs, load average 36–44 throughout).

The walk terminated by exhausting the frontier, not on a budget: all 16 nodes expanded,
`frontier_left=0`, `wall_hit=0`, `node_cap_hit=0`.

```
RESULT ok=1 nodes=16 discovered=16 fingerprints=5 fingerprints0=5 fp_shapes=5 best_size=496
start_size=496 improving_found=0 improvements=0 verify_ok=0 verify_fail=0 min_tight_seen=4
nodes_t123=0 frontier_left=0 wall_hit=0 node_cap_hit=0 kmax=12 found_file=- total_s=3046.3
```

(`verify_ok=0` only because there was nothing to verify — no improvement was ever applied.)

| | |
|---|---|
| structure | the root has 4 connected (12,12) moves, pairwise disjoint in both their removed and their added sets (48 members of `S`, 48 outside vertices) — so they commute and generate a **4-dimensional hypercube**: 1 + 4 + 6 + 4 + 1 = 16 sets, reached at depths 0/1 through the singles and the `--combos 4` subsets (`via` = 4 × (12,12), 6 × combo(24,24), 4 × combo(36,36), 1 × combo(48,48)) |
| every node | `plateau=12:4 connected=12:4 moves=4 nbrs=15 new=0` — every one of the 16 nodes again has exactly the same four disjoint (12,12) moves and hence 15 neighbours, **all 15 already known**. The graph is the 4-cube plus its combo chords (K₁₆ on the cube's vertex set as the walk draws it), and it is closed: applying any move to any node stays inside the 16 |
| improvement | `improving=0` at all 16, each `kexh=12` (the `k ≤ 12` search ran to completion, no per-level timeout) — so **each of the 16 sets is swap-optimal for every removal size k ≤ 12** |
| tightness | `min_tight=4`, `free=0`, `t123=0/0/0` at every node: 80 tightness-4 vertices everywhere, never a free, tightness-1, -2 or -3 vertex. The rigidity is a property of the whole 16-set component, not of the one representative |
| Gram / shapes | `gram=-32:248,-8:37504,0:47504,8:37504` and `shapes=258/232/6` (octad / (∓3,±1²³) / (±4,±4)) identical at all 16 nodes — every node is antipodal, and the (12,12) moves preserve the shape split (unlike the 488's (8,8) moves) |
| fingerprints | **5** distinct Co₀-invariant fingerprints over the 16 nodes (also 5 distinct `fp0`, 5 with shapes), split 4 + 4 + 4 + 2 + 2. They are separated only by the tightness histogram above tightness 6 (e.g. `8:2704 / 8:2800 / 8:2320 / 8:2576`); Gram, min-Gram, plateau counts and shapes are constant |
| the 5 classes | by the subset `W ⊆ {M₁,M₂,M₃,M₄}` applied: **A** = {∅, {M₁M₄}, {M₂M₃}, all four} (nodes 0, 7, 8, 15); **B** = {M₁}, {M₄} and their complements (nodes 1, 4, 11, 14); **C** = {M₂}, {M₃} and their complements (2, 3, 12, 13); **D** = 2 pairs (5, 10); **E** = 2 pairs (6, 9). Each class is a union of **complementary pairs** `W ↔ Wᶜ` |
| kswap per node | 47.2–604.7 s, mean 189.9 s (≈ 40 s on an idle machine; the spread is contention with other jobs, and the two slowest nodes ran while the 488 walk was also going). Unions 880,632–917,960 — constant to 4%, another sign the 16 nodes are near-isometric |
| depth profile | 0:1, 1:15 — the combo edges put every node at depth 1 from the root |

Representative sets, one per fingerprint class, are in
`runs/plateau/s496_k12/sets/node_{00000,00001,00002,00005,00006}.txt`; all five re-verify as genuine
496s, e.g.

```
$ build/t31b/tools/verify_s runs/plateau/s496_k12/sets/node_00005.txt --data data
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 free=0 maximal=1 tight_hist=4:80,6:640,7:256,8:2320,9:9728,10:28864,11:54848,12:47888,13:32000,14:13952,15:2176,16:1480,17:256,18:704,19:64,20:528,22:128,24:152 file="runs/plateau/s496_k12/sets/node_00005.txt" ms=64
$ .venv/bin/python python/verify_S.py runs/plateau/s496_k12/sets/node_00005.txt
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 file="runs/plateau/s496_k12/sets/node_00005.txt" s=0.11
```

Overlaps with the root are exactly what the cube predicts: 484 = 496 − 12 for a single move,
472 = 496 − 24 for a pair, and so on down to 448 for all four.

**Note on the "complementary pair" observation.** The walk noted that applying all four moves is
fingerprint-preserving (nodes 0 and 15 have identical fingerprints) and suggested this is "what one
expects if the flip-all-four map is realised by an element of Co₀". The equivalence computation of §8
later showed that it is realised by an *orthogonal map of R²⁴* which does **not** preserve the Leech
lattice; the fingerprint coincidence is real, the Co₀ reading of it is not.

### 3.4 The 488 walk

`plateau_walk data/S488.txt --kmax 8 --node-time 300 --max-nodes 3000 --wall 1800` (log
`runs/plateau/logs/s488_k8.log`, tables `runs/plateau/s488_k8/`; 6 OpenMP threads).

**No node reaches 489.** 408 nodes expanded (wall budget hit), 7529 distinct sets discovered,
frontier 7121; every expanded node was exhaustive to `k = 8`, none had an improving swap, none a free
or tightness-1 vertex.

```
RESULT ok=1 nodes=408 discovered=7529 fingerprints=408 fingerprints0=4578
fp_shapes=408 best_size=488 start_size=488 improving_found=0 improvements=0 verify_ok=0 verify_fail=0
min_tight_seen=2 nodes_t123=408 frontier_left=7121 wall_hit=1 node_cap_hit=0 kmax=8 found_file=-
total_s=1800.4
```

The root's 37 connected moves (dumped with `swapsearch data/S488.txt --kmax 8 --dump-plateau
runs/plateau/moves488_k8.txt`: 40 stored moves, 37 connected = 24 (2,2) + 13 (8,8)) are **pairwise
disjoint** in both their removed and their added sets (pairwise overlaps: 276 (2,2)/(2,2), 312
(2,2)/(8,8), 78 (8,8)/(8,8) pairs, all 0/0; together they touch 152 members of `S` and 152 outside
vertices). So they commute, and the component of the 488 contains at least the 2³⁷ ≈ 1.4·10¹¹ sets
obtained by applying any subset — no walk can exhaust it. What the walk establishes is a property of
the sampled region:

| | |
|---|---|
| depth profile (expanded) | 0:1, 1:25, 2:358, 3:24 |
| via | 162 (2,2), 245 (8,8) |
| plateau counts at every node | `2:24, 4:276, 6:2024, 8:10639`, connected `2:24, 8:13` — identical at all 408 nodes: every visited node again has exactly 24 (2,2) and 13 (8,8) primitive moves and 37 neighbours (the graph is 37-regular on the sample) |
| tightness | min 2 everywhere (48 tightness-2 vertices at every node); tightness-3 vertices at 10 nodes (64 at 8 nodes, 128 at 2), all at depth 2 reached by two (8,8) moves; free = tightness-1 = 0 everywhere |
| fingerprints | all 408 expanded nodes have distinct tightness histograms (hence 408 distinct Co₀-classes); 4578 distinct `fp0` among the 7529 discovered sets. Of the 37 depth-1 nodes only 25 were expanded before the search moved to depth 2: the 24 (2,2) moves come in 12 negation pairs `M, −M` whose two results `±S′` are isometric (equal `fp0`, so the priority rule defers them), while the 13 (8,8) moves are each negation-closed, so all 13 were expanded |
| shapes (octad / (∓3,±1²³) / (±4,±4)) | 248/236/4 at 383 nodes, 240/244/4 at 25 nodes (one (8,8) move exchanges 8 octad-type vectors for 8 of type (∓3,±1²³)) |
| kswap per node | 0.14–23.6 s, mean 3.1 s (1.2 s on an idle machine); unions 128,789–686,949 |
| new neighbours per node | 37 at the root, 12–17 typically at depth 2 (≈ 1/3 of the 35 non-parent moves — what a commutative structure predicts) |

The nodes carrying tightness-3 vertices are the most interesting ones seen (`swapsearch
runs/plateau/s488_k8/sets/node_00489.txt --kmax 8 --struct 3`, log
`runs/plateau/logs/s488_node489_struct3.log`): node 489 (root → (8,8) → (8,8), still antipodal) has
histogram `2:48 3:128 4:336 5:128 6:480 7:320 8:3002 ...`, and its 128 tightness-3 vertices have 128
distinct conf sets (3 members with Gram 8, 0, 8, i.e. not orthogonal), group MIS 1 — a (3,4)-swap
would need 4 mutually non-adjacent vertices on one conf set. The conf sets cover 144 members of `S`
(128 members in 2 sets, 16 in 8), and the exhaustive `k ≤ 8` search there again finds only the 37
primitive plateau moves (`RESULT ... improving=0 ... plateau_connected=37 min_tight=2`). The (8,8)
moves thus lower tightness locally (2-4-4 → 3) without ever creating a tightness-1 vertex or a
`(k, k+1)` swap with `k ≤ 8`.

### 3.5 Test contents and reproducibility notes

`tests/test_plateau.cpp`: (1) synthetic — the path `P₇` from `S = {1,3,5}` with `kmax = 1`:
`canonical_hash` order-independence, connectivity of single / merged / (1,2) moves,
`disjoint_move_combinations` (2 disjoint moves → 3 combinations, cap 1 → singles only, overlapping
moves are not combined); the walk has 2 plateau moves and 3 neighbours at the root, reaches the
maximum independent set {0,2,4,6} through an improving (1,2)-swap at {0,3,6}, `on_improve` fires, all
discovered nodes are independent of size 3 or 4, the component is exhausted (frontier empty, no cap
hit), ≥ 3 fingerprint classes, and the improved node is expanded with no moves and minimum tightness
2. (2) Leech — the 496 at `kmax = 4` is an isolated node (no move, min tightness 4, no tightness-1/2/3
vertex, exhaustive to `k = 4`, no improvement); the 488 at `kmax = 2` has 24 connected (2,2) moves, 24
new neighbours, 3 nodes expanded (cap) all of size 488 and independent, no improvement.

```
$ cmake -S . -B build/t31b -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build/t31b
-- GPU tests     : ON   sanitize: OFF   werror: ON
[42/42] Linking CXX executable tests/test_ls_search

$ ctest --test-dir build/t31b -R 'plateau|swaps'
    Start 10: test_swaps
1/2 Test #10: test_swaps .......................   Passed    0.29 sec
    Start 11: test_plateau
2/2 Test #11: test_plateau .....................   Passed    0.30 sec

100% tests passed, 0 tests failed out of 2

$ build/t31b/tests/test_plateau data | tail -1
RESULT ok=1 synthetic=1 leech=1 syn_best=4 syn_nodes=9 n496=1 n488=3 fp488=3 failures=0 ms=218
```

* **The first 496 attempt was under-budgeted.** `--node-time 60` (per k-level) let the root finish
  only to `k = 11` (`kexh=11`) before the level timed out, which would have made the "exhaustive to
  12" claim false. That run is kept as
  `runs/plateau/logs/s496_k12_attempt1_nodetime60.log` and the walk was redone with
  `--node-time 900`; every node of the final run reports `kexh=12`.
* **The 488 walk was stopped by its wall budget** (`--wall 1800`, `wall_hit=1`, 7121 frontier nodes
  left), as intended — its component is ≥ 2³⁷ sets, so termination was never possible. The 488
  numbers are statements about a 408-node sample, not about the whole component.
* Both walks shared the machine with other jobs (load average 36–44), so all wall-clock numbers here
  are 3–5× the idle-machine figures. Correctness numbers (moves, fingerprints, tightness, `kexh`) are
  unaffected.
* `improving_found=0` in both walks, so the improvement path of the tool (write to `runs/found/`,
  then run both verifiers as subprocesses) was never taken here; it is exercised instead by the
  synthetic case in `tests/test_plateau.cpp`, where the walk does find an improving (1,2)-swap.

---

## 4. The GPU local-search engine

The exhaustive searches above are complete but shallow: they see only what a `k ≤ 12` removal can
reach. The complementary tool is a massively parallel iterated local search (ILS) that is incomplete
but can travel arbitrarily far. It is built in three layers — chain state and move primitives, the
per-chain ILS kernel, and a host driver with seeding, checkpoints and record custody — all on one
**RTX 3070 Laptop (sm_86, 8 GB, 80 W cap), CUDA 12.8**, release builds with `-Werror` on host and
device code.

The design constraint that shapes everything: near the 496 the minimum tightness outside `S` is 4, so
the classical "(1,2)-swap" and "perturb a tightness-1 or -2 vertex" moves never fire there. The
perturbation primitive is therefore a **force-add**: insert a vertex regardless of conflict and
remove every member that conflicts with it.

### 4.1 Chain state and incremental move kernels

| File | Purpose |
|---|---|
| `cuda/ls_state.cuh` | `LSParams`, `LSState` (structure of arrays), `LSMemory`/`lsa_report_memory`, `lsa_alloc`/`lsa_free`, `lsa_init_from_sets`, `lsa_seed_rng`, `LSHostChain`/`lsa_download`, `LSCheck`/`lsa_check`, `LSOp`/`LSMove` and the three host-callable kernels' declarations. Plain C++ apart from the RNG type (`LSRng` = `curandStatePhilox4_32_10_t` under nvcc, opaque otherwise) — includable from `.cpp`. |
| `cuda/ls_state.cu` | Allocation with the memory-fit check, init (bitmap kernel + `tightness_full` + `lsk_rescan_free` + Philox seeding), copy-back, `lsa_check` (block-per-chain audit kernel over a `tightness_full` recompute in chunks of 64 chains). |
| `cuda/ls_moves.cuh` | Warp-cooperative device primitives (`__device__ __forceinline__`, no `__syncthreads`): `adjacent`/`adjacent_pre` (6 dp4a), `is_tabu`, `warp_tabu_push`, `warp_free_push1`, `warp_free_pop`, `warp_add1`/`warp_remove1`, `warp_add`/`warp_remove` (antipodal-aware), `warp_force_add`, `warp_update_best`. |
| `cuda/ls_moves.cu` | Host-callable batched kernels: `lsk_apply_moves` (scripted move list per chain), `lsk_rescan_free` (block per chain, deterministic ascending compaction), `lsk_drain_free` (pop-validate-add until empty or `max_adds`). |
| `tests/test_ls_moves.cu`, `tests/tasks/T3_2a.cmake` | Acceptance test (ctest `test_ls_moves`, label `gpu`, skip 77, timeout 1800 s, `-fopenmp` for the CPU model). |

```cpp
namespace kiss::cuda {
constexpr uint32_t LS_NONE = 0xFFFFFFFF;
struct LSParams { int B; int N = kiss::N; int SMAX = 1024; int FL = 4096; int TABU = 32; int force_cap = 64; bool antipodal = false; };
struct LSState  { LSParams p; uint16_t* tight /*[B][N]*/; uint32_t *inS /*[B][6143]*/, *S /*[B][SMAX]*/, *size, *fl /*[B][FL]*/,
                  *fl_head; uint8_t* fl_overflow; uint32_t *tabu_v, *tabu_exp /*[B][TABU]*/, *tabu_head, *iter, *best_size,
                  *best_S /*[B][SMAX]*/; LSRng* rng /*Philox4x32-10 [B]*/; };
LSMemory lsa_report_memory(const LSParams&);            // per_chain, total, device free/total, max_B(avail, margin)
LSState  lsa_alloc(const LSParams&, size_t margin = 256 MB);   // throws if total + margin > cudaMemGetInfo().free
void     lsa_free(LSState&);
void     lsa_init_from_sets(LSState&, const vector<vector<uint32_t>>& sets, const DeviceLeech&, const uint32_t* d_adj, uint64_t seed = 0, cudaStream_t = 0);
void     lsa_seed_rng(LSState&, uint64_t seed, uint64_t offset = 0, cudaStream_t = 0);
LSHostChain lsa_download(const LSState&, int b, bool with_tight = false);
std::vector<LSCheck> lsa_check(const LSState&, const DeviceLeech&, cudaStream_t = 0);   // errors() == 0 means consistent

// device side (ls_moves.cuh), all lanes of the warp call convergently with uniform arguments
__device__ int  warp_add(const LSState&, int b, uint32_t v, const uint32_t* d_adj, const uint32_t* d_neg);
__device__ int  warp_remove(const LSState&, int b, uint32_t v, const uint32_t* d_adj, const uint32_t* d_neg, uint32_t tenure);
__device__ int  warp_force_add(const LSState&, int b, uint32_t v, const uint32_t* d_adj, const DeviceLeech&, uint32_t tenure);
__device__ uint32_t warp_free_pop(const LSState&, int b, const uint32_t* d_neg);          // LS_NONE when empty
__device__ bool is_tabu(const LSState&, int b, uint32_t v, uint32_t it, const uint32_t* d_neg = nullptr);
__device__ bool adjacent(const uint32_t* packed, uint32_t u, uint32_t w);                // 6 dp4a == 16
__device__ bool warp_update_best(const LSState&, int b);

// host-callable kernels (ls_moves.cu)
enum LSOp { LS_OP_NOP, LS_OP_ADD, LS_OP_REMOVE, LS_OP_FORCE_ADD, LS_OP_POP_ADD, LS_OP_TICK, LS_OP_BEST, LS_OP_IS_TABU };
struct LSMove { uint32_t op, v; };
void lsk_apply_moves(const LSState&, const LSMove* d_moves /*[B][M]*/, int M, const uint32_t* d_adj, const DeviceLeech&, uint32_t tenure, int32_t* d_ret = nullptr, cudaStream_t = 0);
void lsk_rescan_free(const LSState&, const DeviceLeech&, cudaStream_t = 0);
void lsk_drain_free(const LSState&, const uint32_t* d_adj, const DeviceLeech&, uint32_t max_adds, uint32_t* d_adds = nullptr, cudaStream_t = 0);
}
```

**Semantics.** *One warp is one chain.* Only the owning warp touches chain `b`'s arrays, so there are
no atomics anywhere in the move path: a row of the adjacency has no duplicates, and `row(v)` and
`row(neg[v])` are disjoint, so within one move every `tight[b][n]` is read-modify-written by exactly
one lane. All collectives use the full mask; there is no `__syncthreads` in chain logic (blocks may
hold several chains; the `lsk_*` kernels use 4 warps per block).

* `warp_add(v)`: if `v ∉ S` and `|S| < SMAX`, set the bit, append to the `S` list, and `tight[n]++`
  for `n ∈ row(v)` (lanes stride the 4600 entries, coalesced 128-byte loads). Returns the number of
  vertices added (0/1, or up to 2 in antipodal mode). It does **not** check `tight[v] == 0` — adding
  a conflicting vertex keeps `tight` exact and is what `force_add` relies on.
* `warp_remove(v, tenure)`: if `v ∈ S`, clear the bit, swap-with-last in the list, `tight[n]--` over
  `row(v)`; every `n` whose tightness reaches 0 and is not in `S` is pushed on the free list in row
  (ascending) order via ballot + popc prefix, one head update per 32-entry window; `v` itself is
  pushed afterwards if `tight[v] == 0`; pushes beyond `FL` are dropped and set `fl_overflow[b]`. `v`
  goes into the tabu ring with expiry `iter[b] + tenure`.
* `warp_force_add(v, tenure)`: returns −1 if `v ∈ S`; otherwise gathers the members adjacent to `v`
  (and to `neg[v]` in antipodal mode) by scanning the `S` list with 6 dp4a per member into per-lane
  registers (first `min(force_cap, 64)` in list order), removes them in that order, and if more
  conflicts exist than the cap, falls back to "rescan the list, remove the first conflict" until none
  is left; then `warp_add(v)`. Returns the number of vertices actually removed; every removed vertex
  becomes tabu.
* `warp_free_pop()`: pops from the top (LIFO), discarding stale entries until one validates
  (`tight == 0 && !inS`, and the same for `neg[v]` in antipodal mode); returns `LS_NONE` when
  exhausted. Entries do go stale (a later add raised the vertex), so **validate-on-pop is
  mandatory**; the list is a cache of candidates, never the truth. `lsk_rescan_free` rebuilds it as
  the ascending compaction of the true free set — call it when `fl_overflow[b]` is set, or every `R`
  iterations.
* `is_tabu(v, it)`: lane `l` checks slots `l, l+32, …`, one ballot; true iff some slot holds `v` (or
  `neg[v]` in antipodal mode) with `expiry > it`. `iter[b]` is advanced by the caller.
* **Antipodal mode** (`p.antipodal`): add/remove apply to `v` and `neg[v]` (each only if its own
  precondition holds — the state stays exact even if `S` is not closed under negation; `lsa_check`
  reports closure violations as `antipodal_bad`), `force_add` gathers the conflicts of both, `pop`
  validates both, `rescan` lists only pair-free vertices.
* `lsa_check` (debug/self-check, synchronous) recomputes tightness in chunks of 64 chains and reports
  per chain `tight_mismatch`, `list_bad`, `bitmap_bad`, `conflict_pairs` (= Σ_{s∈S} tight[s] / 2,
  which must be 0 for an independent `S`), `antipodal_bad`, `fl_bad`, and informational `fl_stale`,
  `fl_missing`.

**Memory** (`lsa_report_memory`, SMAX = 1024, FL = 4096, TABU = 32), bytes per chain:

| Array | Bytes |
|---|---|
| `tight` uint16[N] | 393,120 |
| `inS` uint32[6143] | 24,572 |
| `S` + `size` | 4,100 |
| free list + head + overflow | 16,389 |
| tabu ring (v, expiry) + head | 260 |
| `iter`, `best_size` | 8 |
| `best_S` | 4,096 |
| Philox state | 64 |
| **total** | **442,609 (432.2 KB)** |

Device: 8 GB, 7.67 GB free idle, 4.22 GB free after the 3.62 GB adjacency upload → max `B` = 9,395
with a 256 MB margin. `B = 2048` → 865 MB, `B = 4096` → 1.73 GB, `B = 8192` → 3.46 GB (fits, but 0.5
GB headroom is too tight once the search layer adds its own arrays and the host wants `lsa_check`'s
25 MB scratch). Recommended default **B = 4096**; a uint8 tightness fallback is not needed for
`B ≤ 4096`.

**Correctness.** The acceptance test mirrors every device primitive in an independent CPU model
(`RefChain`, using the mmap `Adjacency` for rows and `kiss::dot` for the pair test): list order, free
list push order and overflow, tabu ring, `force_add` gather/cap/fallback, validate-on-pop. Seeds: 32
random subsets of the 496 (300..496 members; antipodal pairs in antipodal mode) plus 32 random-greedy
maximal sets (~230). Three states at `B = 64`:

| State | antipodal | force_cap | FL | rounds × moves |
|---|---|---|---|---|
| plain | no | 5 (the fallback path runs on every force-add of tightness > 5) | 4096 | 1000 × 100 |
| antipodal | yes | 64 | 4096 | 1000 × 100 |
| smallFL | no | 64 | 64 (overflows constantly; rescanned every round) | 100 × 100 |

Per round the generator draws moves from the CPU model's current state (add a free vertex; remove a
member; force-add a random non-member of tightness ≤ 8, and 1/16 of the time ≤ 40 so the second
register slot and the cap-64 fallback run; pop-add; tick; best; is_tabu — half the queries on
vertices actually in the ring), applies them to both sides, then compares **after every round**:
`tight` over all N, the `S` list (exact order), `inS`, the free list (head, entries, overflow flag),
the tabu ring (vertices, expiries, head), `iter`, `best_size`/`best_S`, and every move's return
value. Every 50 rounds (and whenever any chain overflowed) both sides rescan; every 100 rounds
`lsa_check` runs.

```
$ ctest --test-dir build/t32 -R ls_ -V
fixture         : S496 |S|=496, adjacency data/adj.u32
adjacency       : uploaded in 2240 ms; device free 7671.8 MB -> 4221.8 MB
memory          : 442609 bytes per chain (432.2 KB); free after adjacency 4221.8 MB; max B (256 MB margin) = 9395; B=2048 -> 864.5 MB, B=4096 -> 1728.9 MB
fuzz plain     : B=64 rounds=1000 moves=100 FL=4096 force_cap=5 antipodal=0 seed sizes 225..491
fuzz plain     : moves=6400000 mismatching_fields=0 ops{add=519449 rem=757944 force=673758 pop=4112383 tick=167811 best=84466 tabu=84189} force: calls=673758 removed=3657945 over_cap=327536; pop hits=3217124; tabu hits=40903/84189; sizes 204..447; fl entries=2244607 stale=1782067; rescans=20; gpu 23320 ms cpu 44463 ms
fuzz antipodal : B=64 rounds=1000 moves=100 FL=4096 force_cap=64 antipodal=1 seed sizes 216..492
fuzz antipodal : moves=6400000 mismatching_fields=0 ops{add=497490 rem=755730 force=674099 pop=4137050 tick=167367 best=84482 tabu=83782} force: calls=674099 removed=7283032 over_cap=0; pop hits=3223214; tabu hits=41677/83782; sizes 184..408; fl entries=6466552 stale=5460912; rescans=22; gpu 35147 ms cpu 35359 ms
fuzz smallFL   : B=64 rounds=100 moves=100 FL=64 force_cap=64 antipodal=0 seed sizes 222..490
fuzz smallFL   : moves=640000 mismatching_fields=0 ops{add=52960 rem=76119 force=67396 pop=409950 tick=16821 best=8387 tabu=8367} force: calls=67396 removed=368614 over_cap=0; pop hits=318687; tabu hits=4070/8367; sizes 207..433; fl entries=52226 stale=0; rescans=100; gpu 1774 ms cpu 1361 ms
download        : chain0 |S|=496 fl_head=0 best=496; chain1 |S|=400 fl_head=96; 496 tightness histogram (outside S): 4:80 6:640 7:256 8:2704 9:8064 10:31424 11:52672 12:49552 13:31360 14:13440 15:2560 16:1480 17:256 18:704 19:64 20:528 22:128 24:152
drain plain     : B=64 from 400-subsets: free-list 96..98 before, refilled to 496: 63/64, maximal: 64/64, lsa_check errors: 0; sizes: 493:1 496:63
drain antipodal : B=64 from 400-subsets: free-list 96..100 before, refilled to 496: 64/64, maximal: 64/64, lsa_check errors: 0; sizes: 496:64
bench add/remove: B=2048 x 1000 moves: init 310 ms; best 1286.55 ms -> 1.592e+06 moves/s; adjacency 29.3 GB/s, adjacency+tight r/w 58.6 GB/s
bench check     : lsa_check chains with errors = 0/2048
bench force_add : B=2048 x 100: 540.84 ms -> 3.787e+05 force_adds/s, mean removed 3.15 per call (203668 calls), 1.562e+06 row-updates/s
bench drain     : B=2048: 161.29 ms, 225479 adds (110.1 per chain) -> 1.398e+06 adds/s
bench check     : lsa_check chains with errors = 0/2048 after force_add + drain
RESULT ok=1 fuzz_moves=13440000 fuzz_mismatches=0 fl_entries=8763385 fl_stale=7242979 refill_plain=63/64 refill_antipodal=64/64 moves_per_s=1.592e+06 adj_GBps=29.3 total_GBps=58.6 force_adds_per_s=3.787e+05 drain_adds_per_s=1.398e+06 bytes_per_chain=442609 max_B=9395 fuzz_gpu_ms=60241 fuzz_cpu_ms=81184 omp_threads=16 total_s=225 failures=0
1/1 Test #10: test_ls_moves ....................   Passed  225.65 sec
```

**0 mismatching fields** over the whole fuzz (13,440,000 moves across three state configurations,
device compared with the exact CPU model after every round). `over_cap` = force-adds that hit the
register cap (327,536 in the plain state with cap 5: the rescan fallback ran a third of a million
times). `fl entries`/`stale` = free-list entries seen at the compare points and how many were no
longer free (validate-on-pop discards these; none was ever wrong). An earlier run of the same binary,
before the generator followed force-adds with drains — sets then drift to 12..344 / 0..262 members,
a harsher small-set regime with thousands of free vertices per chain, free-list overflow and a rescan
every round — also did 13,440,000 moves with 0 mismatches and the same refill and throughput figures.

`compute-sanitizer` is clean on `test_ls_moves --small` (B = 4, 5 rounds × 20 moves, all kernels
including rescan/check/drain): memcheck with `--leak-check full` reports
`LEAK SUMMARY: 0 bytes leaked in 0 allocations` and `ERROR SUMMARY: 0 errors`; racecheck reports
`0 hazards displayed (0 errors, 0 warnings)`; synccheck reports `0 errors`. Each of the three
sanitizer runs prints
`RESULT ok=1 fuzz_moves=880 fuzz_mismatches=0 fl_entries=1642 fl_stale=670 refill_plain=4/4 refill_antipodal=4/4 ... failures=0`.

**Throughput, and the correction to the planned budget.** At `B = 2048` on an otherwise idle GPU:

| Kernel | Rate | Notes |
|---|---|---|
| `lsk_apply_moves`, remove/re-add pairs, 1000 moves × B | **1.4–1.6 × 10⁶ moves/s** | 26–29 GB/s of adjacency rows; independent of B from 256 up |
| `warp_force_add` on tightness-≤8 vertices (400-subset chains) | 3.8 × 10⁵ force-adds/s | mean 3.15 removals per call → 1.56 × 10⁶ row updates/s, the same per-row cost |
| `lsk_drain_free` after those force-adds | 1.4 × 10⁶ adds/s | ~110 adds per chain, LIFO pops with validate |
| `lsa_init_from_sets` (B = 2048, \|S\| = 400) | 327 ms | dominated by `tightness_full` |

Scaling with B (`--rounds 1 --bench-B B`):

| B | moves/s | adjacency GB/s | force-adds/s | drain adds/s |
|---|---|---|---|---|
| 256 | 1.428e6 | 26.3 | 3.18e5 | 1.32e6 |
| 1024 | 1.599e6 | 29.4 | 3.78e5 | 1.48e6 |
| 2048 | 1.592e6 | 29.3 | 3.79e5 | 1.40e6 |
| 4096 | 1.583e6 | 29.1 | 3.77e5 | 1.41e6 |

Saturation at 256 warps (6 per SM) means the limit is DRAM traffic, and **the traffic is not the
18.4 KB adjacency row that the original design estimate budgeted**. Each of the 4600 `tight[n]`
read-modify-writes touches its own 32 B sector (row entries are ~43 vertices ≈ 86 B apart on average,
a chain's `tight` array is 393 KB and `B × 393 KB` is far beyond the 4 MB L2), so a move moves
≈ 18 KB (row) + ≈ 3500 sectors × 64 B (read + write) ≈ 240 KB through DRAM:
1.6 × 10⁶ × 240 KB ≈ 380 GB/s, which is this card's usable bandwidth. **The planned "~2 × 10⁷
moves/s" was therefore ~12× optimistic for this uint16-scatter layout**; the realistic aggregate
budget is **≈ 1.5 × 10⁶ row updates/s** — with `B = 4096`, about 370 row updates per chain per
second, i.e. ~30–40 ILS iterations/s per chain when an iteration is a force-add (4–8 removals) plus a
drain (2–4 adds). Options for a future optimisation pass, in order of expected gain: keep a few
chains' `tight` arrays L2-resident with a block-per-chain kernel (turns the scatter into L2 traffic
and leaves only the 18 KB stream — up to ~10× per move, but far fewer concurrent chains); uint8
tightness (≈ 25 % fewer sectors, needs saturation guards); a vertex relabelling that clusters
neighbour indices (the 4600 hits are spread by design of the canonical order — it is unclear whether
any labelling helps much).

Practical consequences carried into the ILS layer: use `warp_force_add` on tightness-4..8 vertices as
the perturbation near the 496 (~2.6 µs of aggregate GPU time per force-add with ~3 removals); after a
perturbation drain with `warp_free_pop` + `warp_add`, noting that the removed vertices are on the
free list and popped first (LIFO), which is the "re-add what you just kicked out unless something
better appeared" behaviour — so tabu entries must be skipped on pop if a perturbation is to stick;
`iter[b]` is the tabu clock; `fl_overflow[b]` must be polled and answered with a rescan (with
FL = 4096 it fired twice in 64 chains × 1000 rounds of the antipodal fuzz and never in the plain one);
more chains buy diversity, not speed, beyond `B ≈ 256`; and `lsa_check` is synchronous and allocates
25 MB, so it belongs between launches, not inside the loop.

### 4.2 The per-chain ILS kernel

| File | Purpose |
|---|---|
| `cuda/ls_search.cuh` | `LSSearchParams`, `LSChainStats`, `LSSearch` (the search's own per-chain arrays: candidate list, tabu bitmap, `n1`, `best_hash`, stall, reported, restart_req, 32 RNG streams, stats), host API, and — under nvcc — the warp-cooperative device primitives (`lss_add/remove/force_add` with bookkeeping, `lss_drain`, warp rescans of the free list and the candidate list, `lss_select`, `lss_try_swap12`, `lss_copy_best`, `lss_revert_to_best`). It never modifies the state layer's files. |
| `cuda/ls_rng.cuh` | 32 xorshift64* streams per chain (seeded by splitmix64 from (seed, chain, lane)); 2 registers per lane. The state layer's Philox state is left alone. |
| `cuda/ls_search.cu` | `run_chains_kernel` (the ILS loop), `lss_init_kernel`, `apply_swaps_kernel` (plateau-move primitive), host wrappers. |
| `tests/test_ls_search.cu`, `tests/tasks/T3_2b.cmake` | Acceptance test (ctest `test_ls_search`, label `gpu`, skip 77, timeout 2400 s, `-fopenmp` for the CPU verifier over 512 chains). `--only STAGE --launches n --K k` runs one stage longer; `--small` is the sanitizer configuration (B = 8). |
| `runs/ls_search/` (gitignored) | Logs of the runs below; `<stage>_bestsets.txt` = the distinct best sets reached per stage (one line per set: hash, size, sorted vertex indices). |

```cpp
namespace kiss::cuda {
struct LSSearchParams {
  int CL = 4096;                 // candidate-list capacity per chain
  int tmin = 1, tmax = 8;        // perturbation window: force-add v with tmin <= tight[v] <= tmax
  int list_tmax = 8;             // candidate list holds non-members with 1 <= tight <= list_tmax (<= 16)
  int tenure = 8, tenure_jitter = 4;   // tabu tenure of removed members: tenure + U{0..jitter} iterations
  int strength = 1;              // force-adds per perturbation
  int select_scan = 1;           // 1: full candidate-list argmin of tightness (random tie-break); 0: tournament
  int tournament_rounds = 4;     // tournament: rounds x 32 random list entries
  int greedy_pct = 100;          // % of selections that are greedy (min tightness) rather than uniform among valid
  unsigned strategies = LSS_CAND | LSS_NEIGH | LSS_UNIFORM;   // candidate list, then random neighbours, then uniform probes
  int neigh_rounds = 2, uniform_rounds = 8;   // fallback sampling rounds (x 32 lanes)
  int swap_enable = 1, max_x_tries = 4;       // ARW (1,2)-swap when n1 > 0; x candidates per iteration
  int tabu_drain = 1;            // drain sets tabu free vertices aside (so a perturbation "sticks")
  int max_drop = 24;             // leash: revert to best_S when |S| < best - max_drop
  int stall_limit = 1000000;     // iterations without strict improvement -> revert + restart_del deletions + restart_req
  int restart_del = 8;
  int rescan_every = 8;          // candidate-list rebuild period (iterations)
  int accept_equal = 1;          // plateau: an equal-size, different set replaces best_S
  int warps_per_block = 8;
};
struct LSChainStats { iterations, adds, removes, swaps, swap_tries, force_adds, force_removed, select_fail, reverts,
                      restarts, cand_rescans, fl_rescans, cand_pushes, plateau, improvements, min_size, best_size, size,
                      stall, n1, cand_head, sel_cand, sel_neigh, sel_uniform, sel_tight_sum; };   // all uint32
std::size_t lss_bytes_per_chain(const LSSearchParams&);                      // 41,336 bytes (CL = 4096)
LSSearch lss_alloc(const LSState&, const LSSearchParams&);                    // throws if it does not fit (128 MB margin)
void lss_free(LSSearch&);
void lss_init(LSSearch&, const LSState&, const DeviceLeech&, uint64_t seed, bool reset_stats = true, cudaStream_t = 0);
void lsk_run_chains(const LSState&, const LSSearch&, const uint32_t* d_adj, const DeviceLeech&, int K, uint32_t target,
                    uint32_t* d_flag, uint32_t* d_out_slots, int nslots, cudaStream_t = 0);   // asynchronous
void lsk_apply_swaps(const LSState&, const LSSearch&, const uint32_t* d_rem, int R, const uint32_t* d_add, int A,
                     const uint32_t* d_adj, const DeviceLeech&, uint32_t tenure, uint32_t* d_ret = nullptr, cudaStream_t = 0);
std::vector<LSChainStats> lss_download_stats(const LSSearch&);
std::vector<uint32_t> lss_download_restart_req(const LSSearch&, bool clear = true);
struct LSFound { uint32_t chain, size; std::vector<uint32_t> S; };
std::vector<LSFound> lss_download_output(uint32_t* d_out_slots, int nslots, uint32_t* d_flag, bool reset = true);
constexpr int LS_OUT_STRIDE = SMAX + 2;   std::size_t ls_out_words(int nslots);   // slots: [0] = claimed count, then {size, chain, S[size]}
}
```

`lss_init` must be called after `lsa_init_from_sets` and after any batched state-layer kernel has
touched the state (it resynchronises `n1`, the tabu bitmap and `best_hash`; the candidate list is
rebuilt at the next iteration). `lsk_run_chains` and `lsk_apply_swaps` keep everything in sync
themselves.

**One ILS iteration** (warp per chain, `run_chains_kernel`):

0. Maintenance: if `fl_overflow[b]`, do a warp-level free-list rescan (ascending compaction, same
   result as `lsk_rescan_free`); every `rescan_every` iterations (or when it is empty) rebuild the
   candidate list as the `CL` lowest-tightness non-members with `1 ≤ tight ≤ list_tmax` (two passes
   over the 393 KB `tight` row with `uint4` loads: histogram of tightness 1..16 → threshold `t*` with
   `cum(t*) ≤ CL` → compaction of `tight ≤ t*` plus the first `CL − cum(t*)` vertices of tightness
   `t* + 1` in index order).
1. Drain the free list (pop-validate-add, LIFO). With `tabu_drain`, popped vertices that are tabu are
   set aside (≤ 64 in registers) and pushed back afterwards, so the members a perturbation just
   kicked out are **not** re-added while tabu — this is what makes a force-add a perturbation rather
   than a no-op. If more than 64 tabu free vertices were seen, `fl_overflow` is set and step 0
   rebuilds the list next iteration. Best tracking after the drain.
2. If `swap_enable` and the chain has tightness-1 non-members (`n1 > 0`, an exact counter maintained
   by the search's add/remove): ARW (1,2)-swap — a tightness-1 vertex `u` is taken from the candidate
   list (random position, non-tabu), its unique conflict `x ∈ S` is found by 6 dp4a per member, `L_x`
   = the tightness-1 non-tabu neighbours of `x` are compacted into shared memory (cap 256 per warp),
   and the first non-adjacent pair `{a, c}` (lanes stride the pairs, 6 dp4a each) gives remove `x` /
   add `a` / add `c`; up to `max_x_tries` distinct `x` per iteration. On success: drain, best
   tracking, and the perturbation is skipped.
3. Otherwise perturbation: `strength` × { select `v`; force-add `v` }. Selection: the candidate list
   first (validate every entry per lane — not in `S`, `tmin ≤ tight ≤ tmax`, not tabu — and take the
   minimum tightness with a random tie-break; or a 4 × 32 tournament with `select_scan = 0`), then
   `neigh_rounds` × 32 random neighbours of random members, then `uniform_rounds` × 32 uniform random
   vertices. `greedy_pct < 100` makes some selections uniform among the valid samples instead of
   greedy. The force-add removes every conflicting member, each with tenure `tenure + U{0..jitter}`,
   then adds `v`. `min_size` is recorded, the free list is drained, best is tracked.
4. Best tracking (`lss_track_best`): strict improvement → `best_S := S`, `stall = 0`; equal size and
   a different set (order-independent hash of `S` ≠ `best_hash`) → plateau copy (`accept_equal`), so
   `best_S` drifts along plateaus and the leash below is measured from the latest plateau point. If
   `|S| ≥ target` and larger than what the chain reported before, claim an output slot (atomic
   counter), copy `S`, set the flag. Leash: `|S| < best − max_drop` → `S := best_S` by symmetric
   difference (no tabu). `stall > stall_limit` → revert, `restart_del` random tabu deletions,
   `restart_req[b] = 1` (the host may reseed instead).
5. `iter[b] += 1` (the tabu clock).

The chain's stats, RNG streams and stall counter live in registers for the `K` iterations and are
written back at the end; all lanes hold identical copies (every increment is on a uniform path).

**Antipodal mode and the tightness-1 list.** With `p.antipodal` every add/remove/force-add applies to
`{v, neg[v]}`. Because `S` stays closed under negation, `conf(neg[u]) = −conf(u)`, so tightness-1
vertices come in pairs (`u` conflicts with `x`, `neg[u]` with `neg[x]`) and **both** land in the
candidate list and are counted in `n1` (so `n1` is even). The swap takes any list entry `u` as the
representative, finds `x`, and builds `L_x` from `row(x)` only (one vertex of each antipodal pair);
the pair test additionally checks `a ≁ neg[c]` (equivalently `neg[a] ≁ c`), which is all that is
needed since `a ≁ c ⇔ neg[a] ≁ neg[c]` and `a, neg[a]` are never adjacent. The applied move is
remove `{x, neg[x]}`, add `{a, neg[a], c, neg[c]}`: a (2,4)-swap. `lss_valid_pert` requires neither
`v` nor `neg[v]` in `S` and takes `tight = max(tight[v], tight[neg[v]])` (equal when `S` is closed); a
force-add of `v` then removes `conf(v) ∪ conf(neg[v])` = `2·tight[v]` members and adds two.
Perturbations are therefore twice as expensive in antipodal mode.

**Plateau-move primitive.** `lsk_apply_swaps` removes `d_rem[b][0..R)` (each an existing member, made
tabu with `tenure`) and then adds `d_add[b][0..A)` — each only if it is free (pair-free in antipodal
mode), so independence can never be broken by a wrong list; `LS_NONE` entries are skipped;
`d_ret[b] = {removed, added}`. It goes through the search's moves, so `n1`, the candidate list and the
tabu bitmap stay in sync; `best_S` is untouched (the next iteration's plateau copy picks the new set
up).

**Memory and throughput.** Search state: **41,336 bytes per chain** (candidate list 16 KB, tabu
bitmap 24.6 KB, 32 × 8 B RNG, stats), on top of 442,609 → 483,945 bytes per chain; `B = 4096` →
1.85 GB (fits with the 3.62 GB adjacency and 2.1 GB headroom). Shared memory: 1 KB per warp (`L_x`),
registers ≈ 90 per thread. `bench` stage (non-antipodal, seeds = random 440..489-subsets of the 496,
4 launches × K = 50, GPU idle):

| B | ILS iterations/s aggregate | per chain | row-updates/s (adds + removes) | force-adds/s | (1,2)-swaps/s |
|---|---|---|---|---|---|
| 512 | 2.20 × 10⁵ | **431** | 9.08 × 10⁵ | 1.6 × 10⁵ | 6.1 × 10⁴ |
| 2048 | 2.18 × 10⁵ | 107 | 8.99 × 10⁵ | 1.6 × 10⁵ | 6.0 × 10⁴ |
| 4096 | 2.36 × 10⁵ | 58 | 9.72 × 10⁵ | 1.7 × 10⁵ | 6.5 × 10⁴ |

The aggregate saturates at `B = 512`: ≈ 9 × 10⁵ row-updates/s is ~60 % of the 1.5 × 10⁶ ceiling of the
state layer, the rest being the candidate-list scans (4096 scattered `tight` reads per selection,
≈ half a move of DRAM sectors) and the periodic rebuilds (786 KB per chain per 8 iterations). One
iteration near the 496 costs ≈ 4 row-updates (force-add: 1 add + ~3 removals; drain: ~2 adds; a swap:
3). Antipodal mode halves the iteration rate: 183–250 it/s per chain at `B = 512`.

**Results at B = 512, K = 50, 20 launches (1000 iterations per chain).** Full log
`runs/ls_search/full_run2.log`; the 20 `inv` and 10 `inv-anti` launch lines all read
`ok: lsa_check errors=0, 512 chains S/best_S independent, best monotone` and are elided.

```
fixture   : S496 |S|=496, S488 |S|=488, adjacency data/adj.u32, omp threads 16
adjacency : uploaded in 4595 ms; device free 7671.8 MB -> 4221.8 MB
memory    : search state 41336 bytes per chain (+ 442609 of the state layer)
inv      : B=512 K=50 launches=20 antipodal=0 | best min 283 mean 424.8 max 496 | at target(497) 0/512 | deepest dip 222 | best hist 283:1 284:1 285:3 286:8 287:15 288:26 289:30 290:35 291:22 292:16 293:6 294:6 295:1 488:171 496:171 
inv      : iterations 512000 (1.75e+05/s aggregate, 342.7/s per chain), adds 1060981 removes 1036208 (7.19e+05 row-updates/s), swaps 58603, force_adds 453397, reverts 972, restarts 10761, plateau copies 206766; gpu 2918 ms wall 22649 ms
inv-anti : B=512 K=50 launches=10 antipodal=1 | best min 290 mean 428.1 max 496 | at target(497) 0/512 | deepest dip 218 | best hist 290:1 292:3 294:7 296:21 298:42 300:44 302:36 304:12 306:3 308:1 488:171 496:171 
inv-anti : iterations 256000 (8.44e+04/s aggregate, 164.8/s per chain), adds 1274970 removes 1250170 (8.32e+05 row-updates/s), swaps 11084, force_adds 244916, reverts 5012, restarts 5361, plateau copies 110826; gpu 3034 ms wall 10247 ms
refill   : after  1 launches (50 iters): best min 496 mean 496.0 max 496, at target(496): 512/512, gpu 129 ms
refill   : after  5 launches (250 iters): best min 496 mean 496.0 max 496, at target(496): 512/512, gpu 581 ms
refill   : after 20 launches (1000 iters): best min 496 mean 496.0 max 496, at target(496): 512/512, gpu 2283 ms
refill   : B=512 K=50 launches=20 antipodal=0 | best min 496 mean 496.0 max 496 | at target(496) 512/512 | deepest dip 460 | best hist 496:512 
refill   : iterations 512000 (2.24e+05/s aggregate, 438.0/s per chain), adds 1027377 removes 1011912 (8.93e+05 row-updates/s), swaps 142668, force_adds 369332, reverts 0, restarts 0, plateau copies 31684; gpu 2283 ms wall 3210 ms
refill   : chains whose best_S differs from the reference set: 478/512; distinct best sets 16; overlap |best ∩ ref| hist: 448:32 460:130 472:201 484:115 
refill-a : after  1 launches (50 iters): best min 496 mean 496.0 max 496, at target(496): 512/512, gpu 290 ms
refill-a : after  5 launches (250 iters): best min 496 mean 496.0 max 496, at target(496): 512/512, gpu 1383 ms
refill-a : after 20 launches (1000 iters): best min 496 mean 496.0 max 496, at target(496): 512/512, gpu 5485 ms
refill-a : B=512 K=50 launches=20 antipodal=1 | best min 496 mean 496.0 max 496 | at target(496) 512/512 | deepest dip 460 | best hist 496:512 
refill-a : iterations 512000 (9.33e+04/s aggregate, 182.3/s per chain), adds 2620088 removes 2607862 (9.53e+05 row-updates/s), swaps 20337, force_adds 491663, reverts 2150, restarts 0, plateau copies 32197; gpu 5485 ms wall 6455 ms
refill-a : chains whose best_S differs from the reference set: 503/512; distinct best sets 64; overlap |best ∩ ref| hist: 288:5 300:27 312:49 324:31 336:7 368:18 380:76 392:96 404:65 416:13 448:7 460:23 472:40 484:46 
climb    : greedy seeds 221..239 mean 229.7
climb    : after  1 launches (50 iters): best min 257 mean 262.9 max 271, at target(497): 0/512, gpu 123 ms
climb    : after  5 launches (250 iters): best min 276 mean 281.9 max 288, at target(497): 0/512, gpu 634 ms
climb    : after 10 launches (500 iters): best min 282 mean 287.9 max 293, at target(497): 0/512, gpu 1276 ms
climb    : after 20 launches (1000 iters): best min 287 mean 292.0 max 297, at target(497): 0/512, gpu 2550 ms
climb    : B=512 K=50 launches=20 antipodal=0 | best min 287 mean 292.0 max 297 | at target(497) 0/512 | deepest dip 221 | best hist 287:1 288:6 289:33 290:59 291:91 292:104 293:119 294:67 295:21 296:9 297:2 
climb    : iterations 512000 (2.01e+05/s aggregate, 392.1/s per chain), adds 546633 removes 515155 (4.16e+05 row-updates/s), swaps 26641, force_adds 485359, reverts 0, restarts 0, plateau copies 359895; gpu 2550 ms wall 2993 ms
s488     : after  1 launches (50 iters): best min 488 mean 488.0 max 488, at target(489): 0/512, gpu 113 ms
s488     : after  5 launches (250 iters): best min 488 mean 488.0 max 488, at target(489): 0/512, gpu 552 ms
s488     : after 20 launches (1000 iters): best min 488 mean 488.0 max 488, at target(489): 0/512, gpu 2204 ms
s488     : B=512 K=50 launches=20 antipodal=0 | best min 488 mean 488.0 max 488 | at target(489) 0/512 | deepest dip 485 | best hist 488:512 
s488     : iterations 512000 (2.32e+05/s aggregate, 453.7/s per chain), adds 1023954 removes 1024000 (9.29e+05 row-updates/s), swaps 0, force_adds 512000, reverts 0, restarts 0, plateau copies 491473; gpu 2204 ms wall 3034 ms
s488     : chains whose best_S differs from the reference set: 512/512; distinct best sets 512; overlap |best ∩ ref| hist: 452:3 456:35 458:4 460:111 462:13 464:174 466:8 468:105 470:4 472:44 474:1 476:10 
s488-a   : after  1 launches (50 iters): best min 488 mean 488.0 max 488, at target(489): 0/512, gpu 202 ms
s488-a   : after  5 launches (250 iters): best min 488 mean 488.0 max 488, at target(489): 0/512, gpu 1005 ms
s488-a   : after 20 launches (1000 iters): best min 488 mean 488.0 max 488, at target(489): 0/512, gpu 4012 ms
s488-a   : B=512 K=50 launches=20 antipodal=1 | best min 488 mean 488.0 max 488 | at target(489) 0/512 | deepest dip 486 | best hist 488:512 
s488-a   : iterations 512000 (1.28e+05/s aggregate, 249.2/s per chain), adds 2048000 removes 2048000 (1.02e+06 row-updates/s), swaps 0, force_adds 512000, reverts 0, restarts 0, plateau copies 512000; gpu 4012 ms wall 4867 ms
s488-a   : chains whose best_S differs from the reference set: 512/512; distinct best sets 462; overlap |best ∩ ref| hist: 448:16 456:119 464:202 472:152 480:23 
s496     : after  1 launches (50 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 117 ms
s496     : after  5 launches (250 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 571 ms
s496     : after 20 launches (1000 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 2282 ms
s496     : B=512 K=50 launches=20 antipodal=0 | best min 496 mean 496.0 max 496 | at target(497) 0/512 | deepest dip 476 | best hist 496:512 
s496     : iterations 512000 (2.24e+05/s aggregate, 438.2/s per chain), adds 1009454 removes 1012426 (8.86e+05 row-updates/s), swaps 142713, force_adds 369287, reverts 0, restarts 0, plateau copies 31640; gpu 2282 ms wall 3042 ms
s496     : chains whose best_S differs from the reference set: 476/512; distinct best sets 16; overlap |best ∩ ref| hist: 448:33 460:137 472:189 484:117 
s496-a   : after  1 launches (50 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 279 ms
s496-a   : after  5 launches (250 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 1383 ms
s496-a   : after 20 launches (1000 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 5477 ms
s496-a   : B=512 K=50 launches=20 antipodal=1 | best min 496 mean 496.0 max 496 | at target(497) 0/512 | deepest dip 466 | best hist 496:512 
s496-a   : iterations 512000 (9.35e+04/s aggregate, 182.6/s per chain), adds 2602318 removes 2608710 (9.51e+05 row-updates/s), swaps 20511, force_adds 491489, reverts 2124, restarts 0, plateau copies 32327; gpu 5477 ms wall 6246 ms
s496-a   : chains whose best_S differs from the reference set: 502/512; distinct best sets 64; overlap |best ∩ ref| hist: 288:9 300:28 312:48 324:30 336:3 368:13 380:72 392:84 404:66 416:16 448:13 460:32 472:60 484:28 
plateau  : the four (12,12) moves validated on the CPU: ok
plateau  : chain 0 plateau move removed=12 added=12 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 1 plateau move removed=12 added=12 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 2 plateau move removed=12 added=12 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 3 plateau move removed=12 added=12 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 4 plateau move removed=24 added=24 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 5 plateau move removed=48 added=48 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 6 plateau move removed=0 added=0 -> |S|=496 independent=1 lsa_check errors=0
plateau  : chain 7 plateau move removed=0 added=0 -> |S|=496 independent=1 lsa_check errors=0
plateau  : B=8 K=50 launches=5 antipodal=0 | best min 496 mean 496.0 max 496 | at target(497) 0/8 | deepest dip 479 | best hist 496:8 
plateau  : iterations 2000 (7.3e+03/s aggregate, 912.6/s per chain), adds 4059 removes 4108 (2.98e+04 row-updates/s), swaps 557, force_adds 1443, reverts 0, restarts 0, plateau copies 131; gpu 274 ms wall 505 ms
plateau  : chains whose best_S differs from the reference set: 8/8; distinct best sets 5; overlap |best ∩ ref| hist: 460:2 472:4 484:2 
bench512 : after  4 launches (200 iters): best min 496 mean 496.0 max 496, at target(497): 0/512, gpu 465 ms
bench512 : B=512 K=50 launches=4 antipodal=0 | best min 496 mean 496.0 max 496 | at target(497) 0/512 | deepest dip 440 | best hist 496:512 
bench512 : iterations 102400 (2.2e+05/s aggregate, 430.5/s per chain), adds 217354 removes 203974 (9.07e+05 row-updates/s), swaps 28245, force_adds 74155, reverts 0, restarts 0, plateau copies 6404; gpu 465 ms wall 1052 ms
bench2048: after  4 launches (200 iters): best min 496 mean 496.0 max 496, at target(497): 0/2048, gpu 1876 ms
bench2048: B=2048 K=50 launches=4 antipodal=0 | best min 496 mean 496.0 max 496 | at target(497) 0/2048 | deepest dip 440 | best hist 496:2048 
bench2048: iterations 409600 (2.18e+05/s aggregate, 106.6/s per chain), adds 868230 removes 815218 (8.97e+05 row-updates/s), swaps 112730, force_adds 296870, reverts 0, restarts 0, plateau copies 25604; gpu 1876 ms wall 4133 ms
bench4096: after  4 launches (200 iters): best min 496 mean 496.0 max 496, at target(497): 0/4096, gpu 3467 ms
bench4096: B=4096 K=50 launches=4 antipodal=0 | best min 496 mean 496.0 max 496 | at target(497) 0/4096 | deepest dip 440 | best hist 496:4096 
bench4096: iterations 819200 (2.36e+05/s aggregate, 57.7/s per chain), adds 1736145 removes 1630677 (9.71e+05 row-updates/s), swaps 225274, force_adds 593926, reverts 0, restarts 0, plateau copies 51102; gpu 3467 ms wall 8038 ms
RESULT ok=1 stages=10 failures=0 record_max=496 records=0 inv{restarts=10761 swaps=58603} inv_anti{restarts=5361 swaps=11084} refill=512/512 refill_a=512/512 climb{mean_best=292.048828 max=297} s488{max=488 diff=512} s488_a{max=488 diff=512} s496{max=496 diff=476} s496_a{max=496 diff=502} plateau{diff=8/8} it_s_B512=2.204e+05 it_s_B2048=2.183e+05 it_s_B4096=2.363e+05 total_s=105
```

Reading the stages:

* **Invariants** (`inv`, `inv-anti`). Seeds: a third random 400..495-subsets of the 496, a third
  400..487-subsets of the 488, a third random greedy maximal sets (221..238); `stall_limit = 40`,
  `max_drop = 16` so that restarts and leash reverts fire. After **every** launch: `lsa_check`
  0 errors on all 512 chains; every `S` and every `best_S` passes `kiss::verify_independent` (CPU,
  exact); antipodal closure of `S` in antipodal mode; `best_size` monotone per chain; `best_hash`
  equals the host hash of `best_S`; on a rotating 16 chains, `n1` equals the recount of tightness-1
  non-members and the tabu bitmap equals the ring's contents; `restart_req[b]` is set iff the restart
  counter advanced. Over 512,000 + 256,000 iterations: 10,761 + 5,361 device restarts, 972 + 5,012
  leash reverts, 58,603 + 11,084 (1,2)-swaps, 453k + 245k force-adds — **0 failures**.
* **Refill from 460-subsets** (`refill`, `refill-a`). **512/512 chains back at 496 after the first
  launch (50 iterations)** in both modes; the drain alone refills most of the 36 missing members, the
  rest is the (1,2)-swap + force-add walk. Every chain claimed an output slot once (64 slots per
  launch, `reported[b]` prevents repeats) and every reported set was verified independent by the
  host.
* **Climb from random greedy sets** (`climb`). Seeds 221..238 (mean 229). After 50 iterations best
  257..271 (mean 263); after 250: 276..288 (282); after 1000: **287..297 (mean 292.0)**. The
  20,000-iteration run (`--only climb --launches 200 --K 100`, 51 s of GPU) gives 266..280 after 100,
  294..303 (296.9) after 5000, 295..303 (298.0) after 10,000, **296..303 (mean 298.7) after 20,000**
  — histogram 296:7 297:72 298:142 299:172 300:81 301:29 302:4 303:5. The climb saturates around
  300, far short of the design hope of "≥ 400 within minutes" (which assumed both a 20× higher move
  rate and a landscape that (1,2)-swaps can climb). The (1,2)-swap fires in only 1.6 % of iterations
  at that point (159k swaps in 10.2 M iterations) and the perturbation with `tmin = 1` mostly performs
  (1,1) plateau swaps (862k plateau copies). **Random greedy seeds are for diversity only**; the
  record neighbourhood must be seeded from the records and their images.
* **Seeds from the 488** (`s488`, `s488-a`). All 512 chains seeded with the full 488, target 489:
  **no 489** in either mode after 1000 iterations (deepest dip 485 / 486 — the walk stays within 3 of
  the record). Every chain ended on a *different* 488: 512 distinct sets in plain mode (overlaps with
  the seed 452..476), 462 distinct in antipodal mode (overlaps 448, 456, 464, 472, 480 = the seed
  minus 2..6 of the 24 (2,2)-moves and the 13 (8,8)-moves). The 488's plateau is huge (≥ 2²⁴ sets
  from the commuting (2,2)-moves) and the ILS moves along it freely; **none of the visited plateau
  points had a free vertex**.
* **Seeds from the 496** (`s496`, `s496-a`). All 512 chains seeded with the 496, target 497: **no
  497** in either mode, neither in 1000 iterations nor in the 20,000-iteration run
  (`runs/ls_search/s496_long.log`: 10.24 M iterations per mode, 7.4 M / 10.1 M force-adds, 2.9 M /
  0.18 M swaps, deepest dips 475 / 466). See §4.3 for what those runs found instead.
* **Plateau primitive** (`plateau`). The four (12,12) moves, validated on the CPU first (every added
  vertex's conflicts lie in the removed set, added vertices mutually non-adjacent), applied by
  `lsk_apply_swaps` to chains 0–3 (12/12 each), chain 4 (moves 0+1: 24/24), chain 5 (all four: 48/48),
  chains 6–7 (none): every chain at 496 and independent afterwards (`lsa_check` 0 errors), and the
  search continues from the new 496; 5 launches × 50 iterations with full invariants, 0 failures,
  5 distinct 496s among the 8 chains.
* **Sanitizers** (`--small`: B = 8, 3 launches × K = 5, all stages including plateau moves and output
  slots): memcheck `0 bytes leaked in 0 allocations`, `ERROR SUMMARY: 0 errors`; racecheck
  `0 hazards displayed (0 errors, 0 warnings)`; synccheck `0 errors`. The memcheck run's line:
  `RESULT ok=1 stages=10 failures=0 record_max=496 records=0 ... climb{mean_best=247.375000 max=250} ... total_s=16`.

**Parameter sweep** (climb from greedy sets, B = 512, 40 launches × K = 50 = 2000 iterations, seed
4242):

| configuration | best min / mean / max after 2000 iterations |
|---|---|
| **defaults** (strength 1, greedy 100 %, tmin 1, tmax 8, tenure 8 ± 4, full-list argmin, swap on) | 289 / **294.7** / 301 |
| `--strength 2` | 289 / 292.8 / 296 |
| `--strength 4` | 288 / 290.9 / 294 |
| `--greedy-pct 50` | 271 / 274.8 / 281 |
| `--tmin 2` | 270 / 273.7 / 280 |
| `--select-scan 0` (4 × 32 tournament) | 273 / 276.3 / 283 |
| `--max-drop 8` | 289 / 294.7 / 301 (identical: the leash never fires while best rises) |
| `--swap 0` | 290 / 294.0 / 298 |
| `--tenure 16` | 289 / 292.2 / 298 |
| `--stall-limit 100 --restart-del 16` | 287 / 292.8 / 298 |

Greedy minimum-tightness selection over the full candidate list is what drives the climb (the two
cheaper, noisier selections lose 20); everything else is second order at this horizon. The
(1,2)-swap is nearly neutral on greedy sets but it is what refills the 496 fastest (142k swaps in the
refill stage), so it stays on.

**Design notes.** The perturbation deliberately departs from the classical "tight ∈ {1,2}" recipe:
with `tmin = 1` the greedy choice is a tightness-1 vertex whenever one exists (a (1,1) plateau swap),
and the sweep shows `tmin = 2` climbs worse; near the records the minimum tightness is 4 (496) or 2
(488) anyway. The (1,2)-swap takes `u` from the candidate list rather than iterating `x`
round-robin — a tightness-1 `u` pins its `x`, the cheaper direction — capped at `max_x_tries = 4`
distinct `x` per iteration. Restart is device-side (revert to `best_S` plus `restart_del` tabu
deletions) and additionally raises `restart_req[b]` for host reseeding; the leash (`max_drop`) never
fired in the climb and fired 972–5,012 times in the invariant stages with `max_drop = 16`. Plateau
acceptance keeps best sizes monotone (checked); output slots report a chain only when `|S| ≥ target`
*and* it exceeds what that chain reported before, so plateau points at the target size are not
re-reported (the test reads them from `best_S` / the bestsets dump instead). The RNG is 32
xorshift64* streams per chain, deterministic given (seed, chain, lane); the test's stages are
reproducible run to run (identical numbers in `full_run1.log` and `full_run2.log`). `runs/found/`
stayed empty: no set ≥ 497 was produced; the custody path (`write_set` → `verify_s` → `verify_S.py`)
is in place and was exercised on 496s via the 460-subset refill.

### 4.3 What the 496-seeded runs found: six commuting plateau atoms

The 20,000-iteration runs from the 496 produced no 497, but they did map the plateau. The `best_S` of
every chain at the end of the run was collected and every set re-verified independent in exact
integer arithmetic by `runs/ls_search/analyse_plateau.py`.

* **Plain mode: exactly 16 distinct 496s**, overlaps with the seed 484:126 / 472:180 / 460:134 /
  448:36 (36 chains stayed on the seed). Their removal sets are exactly the 2⁴ unions of the four
  (12,12) moves — recovered to the vertex, positions and added vertices identical to the exhaustive
  search's list. Nothing else, consistent with "no other plateau move for `k ≤ 12`".
* **Antipodal mode: exactly 64 distinct 496s**, overlaps 484..448 (the (12,12) unions) *and* 416,
  404, 392, 380, 368 (= 496 − 80 − 12·{0..4}) and 336, 324, 312, 300, 288 (= 496 − 160 − 12·{0..4}).
  The removal sets are the 2⁶ unions of **six pairwise-disjoint atoms: the four (12,12) moves and two
  new (80,80) plateau moves**, each closed under negation, each replacing 80 members of the 496 by 80
  vertices whose tightness with respect to the 496 is 4 (16 of them), 6 (32) and 8 (32); the
  conflicts of the added vertices lie inside the removed 80, and the added 80 are mutually
  non-adjacent. The atoms (`S` positions into the ascending index list of `data/S496.txt`, and added
  vertex indices) are in `runs/ls_search/plateau_atoms_496.json` / `plateau_atoms_full.txt`; the 64
  sets in `runs/ls_search/s496-a_long_bestsets.txt`. The set with both (80,80) moves applied (overlap
  336) is saved as `runs/ls_search/S496_alt_overlap336.txt`:

```
$ build/t32b/tools/verify_s runs/ls_search/S496_alt_overlap336.txt --data data
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 free=0 maximal=1 tight_hist=4:80,6:640,7:256,8:2704,9:8064,10:31424,11:52672,12:49552,13:31360,14:13440,15:2560,16:1480,17:256,18:704,19:64,20:528,22:128,24:152 file="runs/ls_search/S496_alt_overlap336.txt" ms=326
$ .venv/bin/python python/verify_S.py runs/ls_search/S496_alt_overlap336.txt
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 file="runs/ls_search/S496_alt_overlap336.txt" s=1.00
```

* The (80,80) moves were found **only in antipodal mode** (plain mode: 20,000 iterations, 512 chains,
  16 sets). An antipodal force-add removes `2·tight` members (8..16) and adds 2, a much stronger
  structured perturbation than the plain 4..8-removal, and the plain walk never leaves the
  (12,12)-plateau. This is the concrete argument for running the record neighbourhood in antipodal
  mode. Whether a *non*-antipodal 496 exists remains open: none was seen.

> **Superseded conjecture.** At the time this alternative 496 was found, it was noted that it has the
> same Gram histogram and the same tightness histogram as the original and was therefore "plausibly a
> Co₀-image of it". **That conjecture is false.** The set in question is `S_48` in the labelling of
> §7, and the equivalence computation of §8 shows it is not even *isometric* to the record — the
> individualisation search terminates with zero Gram-graph isomorphisms. Moreover the claim that all
> 64 plateau sets share the record's tightness histogram is itself wrong: they take **five** distinct
> values (§4.8, §7, §8). The saved set happened to lie in the record's own fingerprint class, which is
> what produced the mistaken generalisation.

Recommended settings that came out of this layer, and were used by the production driver: the
parameters above as defaults, `stall_limit ≈ 2000` iterations (the 20,000-iteration runs never
improved after ~5000 on greedy sets and never at all from the records), `restart_del 8`; **antipodal
mode on** for the record neighbourhood with a minority of chains plain; seeds = the 496 and its 64
plateau images and Co₀-images minus 5–15 %, the 488 likewise, greedy sets as a diversity minority
only; plateau atoms injected with `lsk_apply_swaps` as host-side perturbations when a chain stalls at
496 (the device walk finds the 12-moves in ~50 iterations and the 80-moves within ~1000 antipodal
iterations, so injection mainly saves time); `B = 512–1024` for interactive runs, 4096 for long
unattended runs; `K = 50–100` per launch; `lsa_check` every ~50 launches. Target = 497 — the
independent upper bound in force is 850 from the two-point LP (`02-upper-bounds.md`), so the kernel's
size ceiling never triggers.

### 4.4 The host driver `gpu_mis`

| File | Purpose |
|---|---|
| `include/kiss/run_config.h` | `Json` value type, parser/dumper declarations, `RunConfig`/`SeedMix`/`EngineParams`, overrides, validation |
| `src/run_config.cpp` | hand-written JSON reader/writer (nlohmann is not installed here) plus config load/override/validate/dump |
| `tools/gpu_mis.cpp` | the driver (≈ 1500 lines): fixtures, seed factory, two engine states, launch loop, elite pool, distinct-set census, checkpoints/resume, custody chain, watchdog |
| `configs/default.json` | the diverse seed mix, B = 1024, 75 % antipodal |
| `configs/record_neighbourhood.json` | B = 2048, 75 % antipodal, seeds mostly 496 / plateau images / Co₀ images |
| `configs/soak_10min.json` | small (B = 512), self-check on, used by the test |
| `docs/runbook_gpu_mis.md` | operator documentation (CLI, config keys, run directory, recipes, troubleshooting) |
| `tests/test_gpu_mis.cpp`, `tests/tasks/T3_2c.cmake` | acceptance test (`ctest -R test_gpu_mis`) |

**Epochs, and why `--resume` is exact.** The ILS kernel keeps per-chain RNG streams *on the device*,
seeded by `lss_init(seed)`, and there is no API to read them back. The driver therefore makes the
resumable unit an **epoch** rather than a launch. An epoch starts with
`lsa_init_from_sets(sets) + lss_init(seed_epoch)`; inside an epoch the device state after *n* launches
is a deterministic function of `(sets, seed_epoch, K, n)`. Epoch boundaries occur at the start, at a
reseed pass, and at every checkpoint. A checkpoint writes the chain sets it is about to re-initialise
from, together with `epoch+1`, the host `mt19937_64` state, the elite pool, the distinct-set census
and the counters; it then performs that re-initialisation itself, and a `--resume` performs exactly
the same re-initialisation from the file, so both paths continue from bit-identical device state.
`seed_epoch = seed·0x9E3779B97F4A7C15 + epoch·0xBF58476D1CE4E5B9 + (antipodal ? … : …)`.

A **reseed pass** is the host-level restart: chains whose `restart_req` fired since the last pass get
a new seed (the elite pool with random deletions with probability `reseed_elite_prob`, else the seed
mix); every other chain restarts from its own best-ever set. The pass is triggered once
`reseed_batch_frac·B` chains of a state are pending, which amortises the ≈ 0.4 s
`lsa_init_from_sets` (a `tightness_full` recompute over all chains) over many device-side restarts.

Two engine states coexist (antipodal / plain, split by `antipodal_fraction`), each with its own
`LSState`, `LSSearch`, output slots, flag and **CUDA stream**. Running them concurrently rather than
sequentially is worth ≈ 20 % (81 k → 99 k aggregate ILS it/s at `B = 512`), because neither half
saturates the card alone.

**Seeding.** `seed_mix` weights are normalised and stratified over the chain index
(deterministically), and any kind whose inputs are missing drops out. Every seed is its base set
minus a random `delete_frac_min…delete_frac_max` fraction — whole antipodal pairs in antipodal mode.

| kind | base |
|---|---|
| `s496` / `s488` | `data/S496.txt` / `data/S488.txt` |
| `plateau496` | a random non-empty subset of the six commuting plateau moves of `runs/ls_search/plateau_atoms_496.json` applied to the 496 |
| `co0_496` | `g · (random plateau image)`, `g` from `ProductReplacement` over `co0_generator_perms` (see `05-lifting-template-and-families.md` for the Leech automorphism machinery) |
| `co0_488` | `g · 488` |
| `elite` | a random elite-pool member |
| `greedy` | a fresh random greedy maximal set (~230–240) |

At start-up the driver **verifies all 64 plateau images** with `verify_independent` (2 s) and
disables the plateau seeds if any is not a valid 496. Every run prints

```
plateau   : 64 images of the 496 from 6 commuting atoms, 0 invalid
```

so the four (12,12) moves and the two (80,80) moves are independently confirmed to be pairwise
disjoint, commuting, and each of the 2⁶ combinations an independent set of size 496.

**Custody chain.** On the device flag, in this order: `write_set` to
`runs/<run>/found/S_<size>_<utc>_<hash>.txt` → `tools/verify_s <file> --data data` →
`.venv/bin/python python/verify_S.py <file>` → `RECORD` only if both exit 0, both print `ok=1`, and
both report the same size ≥ 497. Anything else prints a `VERIFY-FAIL` banner, is counted, and makes
the process exit with `ok=0`. Duplicate sets (equal order-independent hash) are written once.
Candidates below 497 (a lowered `--target`) take the same path and are logged as "verified … (below
497, not a record)". Transcript from the test's custody stage:

```
>>> candidate |S| = 496 (anti/chain21/launch1) written to runs/gpu_mis/test_custody/found/S_496_20260827T050823Z_250398b9.txt
>>> verified |S| = 496 (below 497, not a record): runs/gpu_mis/test_custody/found/S_496_20260827T050823Z_250398b9.txt
    verify_s : RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 free=0 maximal=1
               tight_hist=4:80,6:640,7:256,8:2704,9:8064,10:31424,11:52672,12:49552,13:31360,14:13440,15:2560,16:1480,
               17:256,18:704,19:64,20:528,22:128,24:152 ms=141
    verify_S.py: RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 s=0.31
```

**Logging.** `runs/<run>/log.csv` — one row per launch: `t_s, launch, epoch, iter_per_chain,
iters_total, best, mean_best, improved, restarts, reseeds, it_per_s, elite, distinct_ge_log, found,
records, sizes_hash, gpu_temp_c, gpu_power_w, gpu_clk_mhz`, where `sizes_hash` is the determinism
fingerprint of a launch (a hash of every chain's `(best_size, size)`). `runs/<run>/sets.csv` — every
**distinct** best set ≥ `log_min_size` (490), deduped by the order-independent hash, with its file and
its tightness histogram (computed on the GPU, appended as a follow-up row); at most `hist_per_launch`
new distinct sets get the full treatment (independence re-verification + file + histogram) per
launch, and the rest are still counted (`(not sampled)`), so the census stays exact at bounded host
cost. `runs/<run>/gpu.csv` — the watchdog: `nvidia-smi --query-gpu=temperature.gpu,power.draw,
clocks.sm,clocks.mem,utilization.gpu` every `watchdog_seconds`, **log only, no action**.

**Tests** (`ctest -R test_gpu_mis`, 186.06 s, 4 stages, 0 failures):

```
== stage json ==
config    : configs/default.json ok (B=1024 K=50 antipodal=0.75 target=497 stall=2000)
config    : configs/record_neighbourhood.json ok (B=2048 K=50 antipodal=0.75 target=497 stall=2000)
config    : configs/soak_10min.json ok (B=512 K=50 antipodal=0.50 target=497 stall=2000)

== stage custody (target 490, 480-subsets of the 496, B = 64) ==
RESULT ok=1 run=test_custody best=496 launches=3 iterations=3840 records=0 verify_fail=0 found=1
       distinct=24 bad_seeds=0 elapsed_s=1.3 stop=max_launches

== stage resume (--max-launches 4 + --resume vs an uninterrupted run) ==
  launch 1  A ab8e8ff0 / 419.938   B ab8e8ff0 / 419.938
  launch 2  A 560cdf67 / 423.078   B 560cdf67 / 423.078
  launch 3  A 51399ea6 / 424.422   B 51399ea6 / 424.422
  launch 4  A 81ec226d / 425.891   B 81ec226d / 425.891      <- checkpoint here; run B exits
  launch 5  A 176928bf / 426.5     B 176928bf / 426.5        <- run B resumed from checkpoint.txt
  launch 6  A 8e0f0afe / 427.484   B 8e0f0afe / 427.484
  launch 7  A 36275c59 / 427.922   B 36275c59 / 427.922
  launch 8  A a054a26e / 428.484   B a054a26e / 428.484

== stage soak (3.0 min, --selfcheck) ==
RESULT ok=1 run=test_soak best=496 launches=613 iterations=15692800 records=0 verify_fail=0 found=0
       distinct=64 bad_seeds=0 elapsed_s=180.3 stop=wall_clock
soak      : 613 CSV rows, best 496, 613 launches

RESULT ok=1 failures=0 soak_minutes=3.00
```

The custody stage's own files in `found/` are re-run through both verifiers by the test itself, which
asserts `ok=1`, equal sizes and size ≥ 490; the resume stage matches 8/8 launches on both
`sizes_hash` and `mean_best`; the soak runs `--selfcheck` every 20 launches with all `errors=0`,
monotone CSV timestamps and a populated `gpu.csv`. (The soak is 3 minutes rather than 10 to keep
ctest fast; `--soak-minutes` sets it.)

### 4.5 Production runs: 242 M iterations, no 497

Both runs: RTX 3070 Laptop, GPU otherwise idle, release build.

**Record neighbourhood** — `configs/record_neighbourhood.json`, B = 2048, 75 % antipodal, 30 min:

```
$ build/t32c/tools/gpu_mis configs/record_neighbourhood.json --run-name prod_record_30min --minutes 30
plateau   : 64 images of the 496 from 6 commuting atoms, 0 invalid
co0       : 64 random Co_0 elements from 5 generators in 0.8 s
engines   : anti B=1536 slots=1536  plain B=512 slots=512  | 3.18 GB free
seeds anti : co0_496=384 elite=108 greedy=46 plateau496=537 s488=77 s496=384
seeds plain: co0_496=128 elite=36 greedy=15 plateau496=179 s488=26 s496=128

run          prod_record_30min          stop_reason  wall_clock     elapsed_s 1800.63
launches     1144  (K = 50, B = 2048)   iterations   117145600  (65058 it/s aggregate)
best         496                        epochs       56   reseed passes 50
elite pool   256                        found files  0
records      0   verify-fail 0   rejected seeds 0
distinct sets >= 490 by size: 492:3  496:4160
gpu          max_temp 65 C, max_power 61.7 W, mean_clk 1728 MHz, min_clk 1470 MHz over 31 samples
state anti   B=1536 best=496 iters=87859200 force_adds=84645702 swaps=3213498 device_restarts=37779
             plateau=6965843 improvements=56460
state plain  B=512  best=496 iters=29286400 force_adds=21567807 swaps=7718593 device_restarts=12500
             plateau=2533301 improvements=22638
RESULT ok=1 run=prod_record_30min best=496 launches=1144 iterations=117145600 records=0 verify_fail=0
       found=0 distinct=4163 bad_seeds=0 elapsed_s=1800.6 stop=wall_clock
```

| t (s) | launch | best | mean best | distinct ≥ 490 | it/s | GPU |
|---|---|---|---|---|---|---|
| 3.1 | 1 | 496 | 489.00 | 546 | 112 k | 55 °C, 47 W, 1470 MHz |
| 166.7 | 100 | 496 | 488.15 | 4155 | 120 k | 62 °C, 33 W, 1740 MHz |
| 336.2 | 200 | 496 | 488.79 | 4159 | 121 k | 64 °C, 45 W, 1725 MHz |
| 491.3 | 300 | 496 | 490.76 | 4160 | 124 k | 63 °C, 52 W, 1740 MHz |
| 805.3 | 500 | 496 | 490.40 | 4160 | 119 k | 65 °C, 47 W, 1740 MHz |
| 1120.6 | 700 | 496 | 490.27 | 4162 | 120 k | 62 °C, 32 W, 1740 MHz |
| 1464.9 | 900 | 496 | 489.33 | 4162 | 110 k | 61 °C, 56 W, 1740 MHz |
| 1740.2 | 1100 | 496 | 489.78 | 4162 | 120 k | 63 °C, 58 W, 1740 MHz |

**Best never moved off 496.** The 2048 chains reached 496 within the first launch (the seeds are
496-subsets) and the whole run is a plateau walk.

**Diverse mix** — `configs/default.json`, B = 1024, 30 min:

```
$ build/t32c/tools/gpu_mis configs/default.json --run-name prod_diverse_30min --minutes 30
engines   : anti B=768 slots=768  plain B=256 slots=256  | 3.65 GB free
seeds anti : co0_488=77 co0_496=154 elite=38 greedy=77 plateau496=153 s488=115 s496=154
seeds plain: co0_488=26 co0_496=51 elite=12 greedy=26 plateau496=51 s488=39 s496=51

run          prod_diverse_30min         stop_reason  wall_clock     elapsed_s 1800.27
launches     2450  (K = 50, B = 1024)   iterations   125440000  (69678 it/s aggregate)
best         496                        epochs       122  reseed passes 116
elite pool   256                        found files  0
records      0   verify-fail 0   rejected seeds 0
distinct sets >= 490 by size: 492:1  496:4160
gpu          max_temp 65 C, max_power 61.5 W, mean_clk 1700 MHz, min_clk 1590 MHz over 31 samples
state anti   B=768 best=496 iters=94080000 force_adds=91058885 swaps=3021115 device_restarts=41411
            plateau=16178099 improvements=117233
state plain  B=256 best=496 iters=31360000 force_adds=24786061 swaps=6573939 device_restarts=13339
            plateau=5826527 improvements=56380
```

| t (s) | launch | best | mean best | distinct ≥ 490 | it/s |
|---|---|---|---|---|---|
| 2.1 | 1 | 496 | 471.8 | 262 | 107 k |
| 371.3 | 500 | 496 | 472.0 | 4160 | 108 k |
| 736.3 | 1000 | 496 | 470.5 | 4160 | 107 k |
| 1103.0 | 1500 | 496 | 472.6 | 4160 | 109 k |
| 1468.3 | 2000 | 496 | 467.3 | 4161 | 110 k |
| 1652.7 | 2250 | 496 | 472.6 | 4161 | 109 k |

The lower `mean_best` (≈ 471 vs ≈ 490) is the 10 % greedy + 15 % 488 + 10 % Co₀-488 share of the mix,
which sits at ~300 / ~488 forever. Nothing in the diverse mix reached 497 either.

**Anything ≥ 497: nothing.** `found/` is empty in both runs; `best` is 496 in every one of the 3594
logged launches.

**The distinct-496 census — the interesting number.** Both runs converge to **exactly 4160 distinct
496s = 65 × 64**: the 64 plateau images of the identity seed plus the 64 plateau images of each of
the 64 random Co₀ elements. Growth is fast then flat (run 2: 262 → 2759 in 30 s → 4097 at 187 s →
4160 at ~370 s, then +1 in 25 minutes). The engine **saturates the plateau component of each seeded
496 and never leaves it**: the plateau component reachable by these moves from a 496 has exactly 64
nodes (the 4-cube on the four (12,12) moves × the two (80,80) moves = 2⁶). More diversity is bought
only by more Co₀ seeds (`co0_count`), not by more chains or more time. Three 492s (run 1) and one 492
(run 2) were also logged; they have **minimum tightness 2** (`2:8`, eight vertices of tightness 2),
i.e. they admit (2,3)-swap attempts, unlike the 496 whose minimum is 4.

**Thermal behaviour.**

| run | samples | mean temp | max temp | mean power | max power | mean SM clock | min SM clock |
|---|---|---|---|---|---|---|---|
| record neighbourhood, 30 min | 31 | 63.0 °C | 65 °C | 49.8 W | 61.7 W | 1728 MHz | 1470 MHz |
| diverse mix, 30 min | 31 | 63.3 °C | 65 °C | 57.0 W | 61.5 W | 1700 MHz | 1590 MHz |

**No thermal throttling over 30 minutes.** The card sits at 63–65 °C, well under the 80 W cap
(50–62 W), and holds 1725–1740 MHz for most of the run. The occasional 1470–1650 MHz samples coincide
with epoch boundaries (`lsa_init_from_sets` plus the host-side seed check), where the GPU is briefly
idle, not with throttling. Aggregate throughput is flat to ±3 % from the first minute to the last
(120 k it/s at B = 2048 throughout), which is the operational proof that nothing sagged. An earlier
run that ran hotter (78 °C, 78 W) was the aborted one described next, which spent its last minutes
running thousands of verifier subprocesses — CPU load, not GPU.

### 4.6 A driver bug the custody chain caught (fixed)

The first 30-minute production run produced, at launch 630, a burst of "candidates" of size 497–568.
Both verifiers rejected all of them, no `RECORD` was logged, and the run exited `ok=0`. That run is
kept as `runs/gpu_mis/ABORTED_record_bug/`:

```
!!! VERIFY-FAIL for .../found/S_514_20260827T054602Z_3f3b6799.txt (claimed |S| = 514)
!!!   build/t32c/tools/verify_s ... --data data                  -> exit 1, ok=0 size=514
!!!   .venv/bin/python python/verify_S.py ...                    -> exit 1, ok=0 size=514

$ build/t32c/tools/verify_s runs/gpu_mis/ABORTED_record_bug/found/S_497_20260827T054010Z_32017ef5.txt --data data
FAIL      : gram: line 2 (row 0) and line 5 (row 3) have inner product 16 > 8 (60 degrees) [1470 offending pair(s)]
RESULT ok=0 size=497 ... reason="gram: ... [1470 offending pair(s)]"
```

Cause: `Driver::checkpoint()` re-initialises every chain from its best-ever set. It refreshed the
`best_S` block from the device but took the *sizes* from `stats[b].best_size` downloaded during
`collect()`. When a **reseed pass** ran in the same loop iteration (it precedes the checkpoint), it
had already called `lsa_init_from_sets`, which resets `best_S`/`best_size` to the new, generally
smaller seeds. The checkpoint then read `old_size` members out of the `new_best_S` buffer: the tail
was stale data from the previous epoch, giving duplicated and conflicting vertices, so the engine was
seeded with sets that are not independent and from then on happily "grew" them past 496. The two
events only coincide when a wall-clock checkpoint lands on a launch that also triggered a reseed pass
— here, once in 1144 launches.

Fixes, all in `tools/gpu_mis.cpp`:

1. `download_best(Engine&)` reads `best_size` **and** `best_S` in one transaction into
   `Engine::h_best_size` / `h_best_S`; `chain_set(e, b)` is the only way a chain's set is extracted.
   `harvest_best`, `reseed_pass` and `checkpoint` all call `download_best` first. `stats[].best_size`
   is never used to slice the buffer again.
2. `sanitise_seeds()` (new, config key `verify_seeds`, default on) checks **every** set handed to
   `lsa_init_from_sets`: sorted, de-duplicated, indices in range, and pairwise `dot != 16` (OpenMP
   over chains; ≈ 0.3 s at B = 2048, i.e. ~1 % of run time given an epoch boundary every ~40
   launches). A failing set is replaced deterministically by the 496 and reported as
   `INVARIANT FAILURE`; the count appears in `RESULT bad_seeds=` and makes the run exit non-zero.
3. Everything entering the **elite pool** is verified with `verify_independent` — the pool is what
   propagates a set to other chains.
4. `selfcheck_every: 200` was added to both production configs, so `lsa_check` (device-side tightness
   recompute, independence, antipodal closure) aborts the run on any engine-level corruption.

After the fix, a deliberate stress run (B = 2048, `stall_limit=200`, `checkpoint_minutes=0.4`,
`selfcheck_every=20`) did 30 reseed passes and 8 checkpoints in 3 minutes with `rejected seeds 0`,
all self-checks `errors=0`, and best pinned at 496; both 30-minute production runs (56 and 122
epochs) likewise report `verify-fail 0  rejected seeds 0`. This is the second occasion in this
project on which the two-verifier custody rule caught a false record: **the engine and the driver are
not trustworthy on their own.**

### 4.7 Other engineering notes from the driver

* **Restart semantics.** "Restart stalled chains" is implemented as a batched *reseed pass* (an epoch
  boundary), not per chain: `lsa_init_from_sets` is a whole-state operation and one re-initialisation
  at B = 2048 costs ≈ 0.5 s. `reseed_batch_frac` (default 1/8) amortises it. The device-side restart
  (revert to best plus `restart_del` tabu deletions) still happens immediately; the host reseed is
  the slower, more global one.
* **Chains stall in lockstep.** Because an epoch boundary resets every chain's stall counter at once,
  `restart_req` then fires for nearly all chains simultaneously `stall_limit / K` launches later
  (visible as `restarts ≈ 2000` every 41 launches at `stall_limit = 2000`, `K = 50`). Harmless, but
  it makes the reseed passes periodic rather than staggered; jittering `stall_limit` per chain would
  fix it.
* **A header quirk.** `cuda/ls_search.cuh` opens `namespace kiss::cuda {` at line 44 and closes it
  only inside `#ifdef __CUDACC__` (lines 189/919), so a `.cpp` that includes it is left inside the
  namespace. `tools/gpu_mis.cpp` works around this with a guarded `}` after the include rather than
  editing the header; the one-line fix there would be an `#else }` next to line 188.
* **Bounded host work.** Near the 496 practically every chain produces a new distinct best set every
  launch, so the per-set work (sort, independence check, file, GPU histogram) is capped at
  `hist_per_launch` per launch; sets past the cap are still counted, so the distinct-set census stays
  exact. Hashing is order-independent, so a set that is not new costs one pass over its `best_S`
  slice and nothing else. `max_set_files` is 512 in the production configs (a run visits thousands of
  distinct 496s; the census lives in `sets.csv`, the files are only a sample).
* **Throughput.** ≈ 1.2 × 10⁵ aggregate ILS it/s at B = 2048 (75 % antipodal) and ≈ 1.1 × 10⁵ at
  B = 1024, against the kernel-only 2.2–2.4 × 10⁵ for pure non-antipodal chains; the difference is
  the antipodal majority (≈ 2× cost per iteration) plus the epoch boundaries.
* **What would have to change.** 242 M ILS iterations across the two runs, all of them within ~5
  Co₀-classes of the 496, produced nothing above 496. Every chain that starts at (or reaches) a 496
  is trapped in a 64-node plateau component. A different answer requires changing the
  *neighbourhood*, not the throughput: larger `k` in the force-add window (`tmax ≫ 8`), multi-vertex
  perturbations (`strength ≫ 1`) with a much longer tabu tenure, or the structural searches of §5–§6.

### 4.8 The 64 plateau images are NOT one Co₀-orbit — five fingerprint classes

This is the correction that came out of the production runs, and it matters for how the plateau is
described.

It had been reported that the 64 plateau images are "distinct verified 496s with identical
Gram/tightness histograms". The **Gram** histograms are indeed identical
(`{−32:248, −8:37504, 0:47504, 8:37504}` for all 64), but the **tightness histograms are not**.
Computed two ways — independently in numpy (float64 GEMM over all 196560 × 496 inner products per
image) and by the driver's GPU path (`tightness_full`) on the sets the runs actually visited — the 64
images fall into exactly **five** classes:

| class | # images | mask pattern (bits = the four (12,12) moves; the two (80,80) moves never change the class) | tightness histogram (vertices outside S) |
|---|---|---|---|
| 0 | 16 | 0000, 0101, 1010, 1111 | 4:80, 6:640, 7:256, 8:2704, 9:8064, 10:31424, 11:52672, 12:49552, 13:31360, 14:13440, 15:2560, 16:1480, 17:256, 18:704, 19:64, 20:528, 22:128, 24:152 |
| 1 | 16 | 0001, 0100, 1011, 1110 | 4:80, 6:672, 7:128, 8:2800, 9:8320, 10:31312, 11:51584, 12:51616, 13:29312, 14:14912, 15:1792, 16:1864, 18:848, 20:544, 22:128, 24:152 |
| 2 | 16 | 0010, 0111, 1000, 1101 | 4:80, 6:672, 7:128, 8:2800, 9:8320, 10:31056, 11:52864, 12:49056, 13:31872, 14:13632, 15:2048, 16:1864, 18:848, 20:544, 22:128, 24:152 |
| 3 | 8 | 0011, 1100 | 4:80, 6:640, 7:256, 8:2576, 9:8448, 10:31424, 11:52032, 12:49680, 13:31744, 14:13440, 15:2432, 16:1480, 17:256, 18:704, 19:64, 20:528, 22:128, 24:152 |
| 4 | 8 | 0110, 1001 | 4:80, 6:640, 7:256, 8:2320, 9:9728, 10:28864, 11:54848, 12:47888, 13:32000, 14:13952, 15:2176, 16:1480, 17:256, 18:704, 19:64, 20:528, 22:128, 24:152 |

Class 0 is the class of `data/S496.txt` itself. The multiplicities are visible directly in the
production runs' `sets.csv`: of the 1819 histograms computed in run 1 the counts are
**448 : 451 : 449 : 237 : 231**, and of the 2991 in run 2 **739 : 771 : 745 : 370 : 365** — both
≈ 16 : 16 : 16 : 8 : 8.

The tightness histogram of `S` is a Co₀-invariant, so:

* the two (80,80) plateau moves preserve it (they were at that point still suspected of being
  automorphic images — see §8, where that suspicion is disposed of);
* the four (12,12) moves do **not** — so **the 496 has at least five Co₀-inequivalent "plateau
  relatives" of size 496**, and the answer to "are the 64 plateau 496s Co₀-images of each other?" is
  **no** for at least 5 of the 64. (§8 sharpens this to: no two of the 64 are equivalent.)

This also means the ≈ 4160 sets each production run enumerates are at least 5 genuinely different
496-point configurations up to the symmetry of the lattice, not 4160 copies of one.

---

## 5. Frame-structured search

A different idea for escaping the plateau: search at the level of *frames* rather than vectors. The
196560 minimal vectors form **98280 antipodal classes** `{v, −v}`, and every sign-invariant relation
is a function of `|dot|` of representatives:

| relation | condition | degree |
|---|---|---|
| orthogonal | dot = 0 | 46575 |
| conflict | \|dot\| = 16 | 4600 (the vertex-level conflict degree, quotiented) |

A class set is **independent** iff no two classes conflict; its `2|X|` vectors are then an
independent set of the kissing problem, and every antipodal `S` (both records are antipodal) is of
this form. A **frame** is a 24-clique of the orthogonality graph (48 vectors); a **k-subframe** is a
k-clique. Cliques are automatically independent (dot = 0 ≠ ±16), so the frame level is a
"conflict-free coordinate system": only *cross*-clique pairs can conflict.

Note that this is **not** Conway's cross: a cross is a frame of 48 **norm-8** (Λ₄) vectors, of which
there are `|Λ₄|/48 = 398034000/48 = 8,292,375`.

### 5.1 Implementation

| File | Purpose |
|---|---|
| `include/kiss/frames.h`, `src/frames.cpp` | Antipodal classes (`make_classes`, class dot/orthogonal/conflict, `class_conflicts`, `set_classes`, `classes_to_vectors`); dense `Bitset`; induced `PoolGraph`s on class pools; clique machinery (`maximal_cliques` = Bron–Kerbosch with pivot, `k_cliques` = ordered DFS, `max_clique` = colouring branch and bound); `cross_structure`; `build_neighbourhood` + `count_cliques_through` (exact frame DFS in the 46575-class neighbourhood, OpenMP over the first level) and `estimate_cliques_through` (unbiased random-descent / Knuth estimator of the same count); `classes_of_shape`; `random_frame`; `extend_from_class` (frame-level swap search with a marginal-conflict bound) + `apply_extend_hit`; `greedy_frame_union`; `clique_local_search` (tabu). CPU + OpenMP, exact integer dots throughout. |
| `tools/frames.cpp` | `frames count \| crosses \| extend \| greedy \| ls`. Every subcommand ends with a `RESULT ok=1 …` line; anything with > 248 classes goes through the custody chain (`save_and_verify` → `runs/found/` → `verify_s` → `verify_S.py`). |
| `tests/test_frames.cpp`, `tests/tasks/T3_3.cmake` | Acceptance test `test_frames` (CPU, 21 s, `ARGS data/S496.txt`). |
| `runs/frames/logs/*.log`, `runs/frames/plateau_496_seed4479.txt` (gitignored) | Outputs below. |

Counting method: by transitivity of Co₀ on the 98280 classes, `#frames = f_c · 98280 / 24` for the
number `f_c` of frames through any one class. `count_cliques_through` is an exact ordered DFS on
46575-bit rows (the neighbourhood of class 0 costs `46575² = 2.2 × 10⁹` exact dots and ~270 MB of
bitsets), parallel over the first level; `estimate_cliques_through` is the Knuth/Rosenbluth unbiased
estimator on the same tree (`v₁` uniform in the neighbourhood, `v_{i+1}` uniform in the surviving
candidate set, weight `Π|P_i| / (k−1)!`), which also yields the mean candidate-set size per depth and
estimates of the j-clique counts for every `j ≤ 24`.

### 5.2 How many frames are there? A planning figure corrected

An early planning note in this repository quoted **8,292,375 as the number of full frames of the
minimal vectors** and proposed enumerating them all and building a conflict structure between
subframes. **That number is the count of Conway crosses (frames of norm-8 vectors), not of
minimal-vector frames, and the plan built on it is not executable — it is wrong by about 14 orders of
magnitude.**

Exact counts (`frames count --k K --exact`):

| K | K-cliques through class 0 | total = `f₀ · 98280 / K` | cost |
|---|---|---|---|
| 2 | 46,575 | 2,288,695,500 | trivial (the orthogonality degree) |
| 3 | **502,590,825** | **16,464,875,427,000** | 46,575 DFS nodes, 0.0 s (= 46575·21582/2, an independent check) |
| 4 | **1,636,707,770,775** | **40,213,909,927,941,750** | 5.03 × 10⁸ nodes, 483 s |
| 24 | ≥ 5,963,706 after 9.99 × 10⁹ nodes / 900 s | — | hopeless: the DFS had not finished the *first* of 46575 first-level branches |

`K = 4` is the practical limit of exact enumeration (`K = 5` would need ~1.6 × 10¹² nodes).

**The (±4,±4) family gives an exact lower bound.** The 552 classes of shape (±4,±4) (276 coordinate
pairs × 2 classes each: the (4,4)-type and (4,−4)-type on the same pair are orthogonal, since
dot = 16 − 16 = 0) give frames explicitly. Two shape-2 classes are orthogonal iff their coordinate
pairs are equal or disjoint, so a 24-clique of shape-2 classes uses ≥ 12 pairs, at most 2 classes per
pair, hence **exactly a perfect matching of the 24 coordinates with both classes on each matched
pair**:

* all such frames: `(2m−1)!!` with `2m = 24`, i.e. **23!! = 316,234,143,225** (48 vectors: ±(4,4),
  ±(4,−4) per pair);
* through a given shape-2 class: **21!! = 13,749,310,575**.

This was verified exactly by DFS in the shape-restricted neighbourhood for
`2m = 6, 8, 10, 12, 14, 16` (`--shape 2 --coords 2m --k 2m`): through class 0 the count is `(2m−3)!!`
and the total `(2m−1)!!` in every case; e.g. `2m = 16` gives `135,135 = 13!!` through class 0
(6.2 × 10⁸ nodes, 6.4 s). At `2m = 24` the estimator gives `1.45 × 10¹⁰ ± 0.97 × 10⁹` against the
exact `1.3749 × 10¹⁰` (0.8 σ). Class 0 (representative `(−4,−4,0²²)`) is a shape-2 class and Co₀ is
transitive on classes, so **every** class lies in at least `21!!` frames, and

> **#minimal-vector frames ≥ 13,749,310,575 · 98280 / 24 = 56,303,426,804,625 ≈ 5.63 × 10¹³**, exactly.

That alone settles the question: `8,292,375 ≪ 5.63 × 10¹³`.

**Estimate of the true count** (`frames count --probes 4000000`, four independent seeds; ~5 min each
at 12 threads, 6 min at 4):

| seed | frames through class 0 | rel. s.e. | total |
|---|---|---|---|
| 1 | 2.05184 × 10¹⁷ | 2.22 % | 8.40228 × 10²⁰ |
| 2 | 2.09003 × 10¹⁷ | 2.85 % | 8.55869 × 10²⁰ |
| 3 | 2.07582 × 10¹⁷ | 3.49 % | 8.50047 × 10²⁰ |
| 4 | 2.08965 × 10¹⁷ | 2.79 % | 8.55710 × 10²⁰ |
| **combined** (inverse-variance) | **2.073 × 10¹⁷ ± 2.8 × 10¹⁵ (1.4 %)** | | **8.489 × 10²⁰ ± 1.2 × 10¹⁹** |

So `f₀ ≈ 2.07 × 10¹⁷` frames through each class and ≈ 8.5 × 10²⁰ frames in all — 10¹⁴ times the
Conway-cross count, with the (±4,±4) family a vanishing 3.7 × 10⁻¹⁰ of them. The mean candidate-set
size by depth along a descent (identical across seeds) is 46575, 21582, 9769, 4316, 1860, 783, 322,
130, 52, 21, 9, 4, 2, 2, 1, 0, … — a uniform random descent dies around depth 14, which is why
`random_frame` uses best-of-16 sampling with restarts.

**Consequence for the search.** With ~10²⁰ frames and ~10¹⁷ through each class, global frame
enumeration is not an option. Frame-level search must be **local**: enumerate only the
frames/subframes inside the orthogonal neighbourhood of a low-conflict class, restricted to classes
that barely conflict with the current set. The number of 8-subframes is larger still (estimated
3.96 × 10²¹ through each class, ≈ 4.9 × 10²⁵ in all; even the *exact* 4-subframe count is
4.02 × 10¹⁶).

### 5.3 Cross structure of the 496

`build/t33/tools/frames crosses data/S496.txt` — **0.13 s**, reproducing in C++ what an earlier
Python analysis took 403 s to compute, exactly:

```
data/S496.txt: 496 vectors, verify_independent ok=1 (ok)
classes 248, antipodal 1
orthogonality degrees: 79:16 87:88 95:4 99:32 103:72 107:32 111:4
maximal cliques: 2718, sizes 8:1280 10:800 12:448 16:171 18:4 20:8 22:4 24:3; clique number 24
full frames (24-cliques): 3
  frame 0: classes 783 8499 11408 18292 25703 28363 37199 39518 43780 45469 45914 46135 48112 52674 52893 53571 57062 64240 71875 72948 77422 84290 90695 96727
  frame 1: classes 2821 4989 11552 13868 16692 20000 20659 33495 36862 40590 40796 46937 51386 53670 56487 58261 59485 66834 69731 75927 80861 83472 87833 91462
  frame 2: classes 3916 5105 9776 14407 16035 23970 29632 32648 33802 38033 39184 43922 44001 53705 57422 63133 63678 65870 68949 69623 83549 86081 88677 90074
frames pairwise disjoint: 1
8-cliques (all): 7069615
RESULT ok=1 cmd=crosses size=496 classes=248 maximal=2718 clique_number=24 frames=3 disjoint=1 cliques8=7069615 seconds=0.13
```

So the "one 24-cross + 28 8-crosses" description of the 496 is one admissible decomposition among
many: the rigid objects are the **three disjoint frames** and the 2718 maximal cliques.

`crosses --cover` runs the exact cover in C++ — remove a full frame, then repeatedly branch on the
uncovered class of smallest uncovered degree, enumerate the 8-cliques through it inside the uncovered
set, and prune as soon as any uncovered class has < 7 uncovered neighbours:

```
$ build/t33/tools/frames crosses data/S496.txt --cover      # 300 s budget per frame
  cover with frame 0 + 8-cliques: time limit (0 blocks, 89817712 branch nodes, 2813254656 DFS nodes, 300.0 s)
  cover with frame 1 + 8-cliques: FOUND (28 blocks, 6269 branch nodes, 191294 DFS nodes, 0.0 s)
    blocks: {3916,5105,9776,14407,16035,23970,29632,32648} {44001,8972,14796,20757,32633,43922,48927,60806}
            {65870,783,21292,25698,39184,43780,46135,52674} … (28 octuples, full list in the log)
  cover with frame 2 + 8-cliques: FOUND (28 blocks, 80623384 branch nodes, 2437954071 DFS nodes, 190.6 s)
RESULT ok=1 cmd=crosses size=496 classes=248 maximal=2718 clique_number=24 frames=3 disjoint=1 cliques8=7069615 covers=2 seconds=0.33
```

**The 496 = one full frame (48 vectors) + 28 orthogonal octuples (16 vectors each)**, reproduced
independently of the earlier Python run (2 of the 3 frames; frame #0 also fails a retry at 1800 s /
7.06 × 10⁸ branch nodes). Which frame is easy depends only on the search order — the Python run found
frame #2 in 0.2 s, #0 in 31 s, #1 in 453 s; here frame #1 took 0.0 s, #2 took 190 s, #0 not at all —
so the decomposition is highly non-canonical.

### 5.4 Frame-level extension from the 496: the frame-for-frame swap

**The search.** For a seed class `c ∉ X` (`X` = the 248 classes of the 496) with conflict count
`conf(c) = t`, build the pool `{c} ∪ {x ∈ X : x ⊥ c} ∪ {d ∉ X : d ⊥ c, conf(d) ≤ tmax}` and enumerate
the orthogonal cliques `Q ∋ c` in it, maximising

```
gain(Q) = |Q \ X| − |Conf(Q)|,   Conf(Q) = {x ∈ X : x conflicts with some q ∈ Q \ X}
```

(the class-level swap `X → (X \ Conf(Q)) ∪ Q`; gain > 0 means more than 496 vectors). Branch and
bound: at a node with `r` new classes and conflict-union `C`, sort the remaining candidates by
*marginal* cost `|conf(u) \ C|` and take the cheapest ones — an admissible bound, since adding `r′`
more classes costs at least the `r′` smallest marginal costs. The `X`-members of the pool are tracked
separately so that the number of **frames** through the current clique can be counted.

Class-conflict histogram outside the 496 (the class-level image of the tightness histogram):
`4:40 6:320 7:128 8:1352 9:4032 10:15712 11:26336 12:24776 13:15680 14:6720 15:1280 16:740 17:128
18:352 19:32 20:264 22:64 24:76` — minimum 4, so adding any class outside `X` costs ≥ 4.

Seeds are all classes outside the 496 with a given conflict count `t`; `tmax = t` caps the outsiders
admitted to the pool. Two passes were run over the same seeds: one at `--min-gain -8` (recording the
whole gain histogram down to −8) and one at `--min-gain 1` (the bound then prunes every node that
cannot reach gain ≥ 1 — much faster, and it is the pass that *proves* no improvement exists in these
pools).

| tier conf = t | seeds | outsider pool sizes | best gain | best hit | frames in pools | complete at −8? | time (−8 / +1) |
|---|---|---|---|---|---|---|---|
| 4 | 40 | 15:16, 23:12, 31:8, 39:4 | **0** | added 24, removed 24, \|Q\| = 24 | 336 | yes | 37 s / 0.5 s |
| 6 | 320 | 147:32, 177:64, 183:64, 195:64, 201:96 | −5 | added 13, removed 18 | 143 | yes | 66–228 s / 34 s |
| 7 | 128 | 220:32, 224:32, 250:64 | −6 | added 1, removed 7 | 0 | yes | 181 s / 15 s |
| 8 | 1352 | 775 … 993 | −7 | added 7, removed 14 | 25 | no (30 s/seed cap) | – / 1698 s |

1840 seed classes in total; **best gain over everything = 0**, `improved=0`, and the `--min-gain 1`
pass is complete for every tier (`incomplete=0`), so **no orthogonal clique through any class of
conflict ≤ 8 improves the 496**. Only tier 4 produces gain 0 at all, and it does so 192 times; tiers
6–8 top out at −5, −6, −7 — adding a class of conflict `t` costs `t` and the orthogonal structure
never pays it back. The tier-6 pass was re-run independently and reproduced the earlier run
bit-for-bit (same gain histogram, same 191,005,212 DFS nodes), so the enumeration is deterministic.

**The gain-0 hit is a frame-for-frame swap.** Every one of the 24 tier-4 seeds whose pool has ≥ 23
outsiders returns the **same** best clique:

```
$ build/t33/tools/frames extend data/S496.txt --tiers 4 --min-gain -8 --write-plateau
  seed 4479 (conf 4): pool 108 (23 outside X), nodes 4258086, best gain 0 (added 24, removed 24, |Q|=24), frames 12, complete, 1.06 s
    best gain by |Q\X|: 1:-3 2:-2 3:-3 4:-2 5:-1 6:0 7:-3 8:-2 9:-3 10:-2 11:-1 12:0 13:-3 14:-2 15:-3 16:-2 17:-1 18:0 19:-3 20:-2 21:-3 22:-2 23:-1 24:0
best over all tiers: gain 0 (seed 4479, added 24, removed 24, clique
  [4479 4901 7694 13131 17682 19999 24333 26476 27189 33755 39173 45348 45656 52296 55836 61824 63855 70602 70844 73008 79976 85215 87859 94360])
```

The added 24 classes are **mutually orthogonal — a full frame** — and they are exactly the union of
the four 12-vertex sets of the four disjoint (12,12) plateau moves (6 classes each). The removed 24
classes,

```
  removed classes: 2821 4989 11552 13868 16692 20000 20659 33495 36862 40590 40796 46937 51386 53670 56487 58261 59485 66834 69731 75927 80861 83472 87833 91462
```

are **frame #1 of the 496** verbatim. The gain profile `6:0 12:0 18:0 24:0` shows the four quarters
commute: any 1, 2, 3 or 4 of the (12,12) moves is a plateau move, gain 0 at 6, 12, 18 and 24 added
classes. So the four (12,12) vector moves have a single frame-level description:

> **The 496 has a (48,48)-vector plateau move that replaces one of its three disjoint frames by
> another frame, built from 24 of its 40 tightness-4 classes.**

The other 16 tightness-4 classes (the 8 "small" components of §2.3, 4 conf sets on 8 members each)
have pools with only 15 outsiders and best gain −2 (add 2 classes, remove 4) — they carry no frame.
Per seed there are exactly `8 = 2³` gain-0 cliques (the seed's own quarter, plus any subset of the
other three), 192 in total — the tier-4 gain histogram's `0:192` entry.

`--write-plateau` applied it; both verifiers accept the result and every invariant is unchanged:

```
$ build/t33/tools/verify_s runs/frames/plateau_496_seed4479.txt | tail -1
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 free=0 maximal=1
  tight_hist=4:80,6:640,7:256,8:2704,9:8064,10:31424,11:52672,12:49552,13:31360,14:13440,15:2560,16:1480,17:256,18:704,19:64,20:528,22:128,24:152 …
$ .venv/bin/python python/verify_S.py runs/frames/plateau_496_seed4479.txt | tail -1
RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504 …
$ build/t33/tools/frames crosses runs/frames/plateau_496_seed4479.txt | tail -8
maximal cliques: 2718, sizes 8:1280 10:800 12:448 16:171 18:4 20:8 22:4 24:3; clique number 24
full frames (24-cliques): 3
  frame 0: (= frame 0 of the 496)
  frame 1: (= frame 2 of the 496)
  frame 2: classes 4479 4901 7694 13131 17682 19999 24333 26476 27189 33755 39173 45348 45656 52296 55836 61824 63855 70602 70844 73008 79976 85215 87859 94360
frames pairwise disjoint: 1
8-cliques (all): 7069615
RESULT ok=1 cmd=crosses size=496 classes=248 maximal=2718 clique_number=24 frames=3 disjoint=1 cliques8=7069615 seconds=0.22
```

Identical Gram spectrum, tightness histogram, orthogonality-degree profile, maximal-clique profile
and 8-clique count. At the time this was read as support for the hypothesis that the plateau 496s are
Co₀-images of the record; **that hypothesis was later refuted** (§8) — the invariants really are
identical here (this particular set is in the record's fingerprint class), but identical invariants do
not imply equivalence.

**The swap is an involution.** Running the same search from the plateau 496
(`frames extend runs/frames/plateau_496_seed4479.txt --tiers 4`) reproduces the mirror image: same
class-conflict histogram (`4:40 6:320 …`), same 40 tightness-4 classes, same 192 gain-0 cliques, and

```
best over all tiers: gain 0 (seed 2821, added 24, removed 24, clique
  [2821 4989 11552 13868 16692 20000 20659 33495 36862 40590 40796 46937 51386 53670 56487 58261 59485 66834 69731 75927 80861 83472 87833 91462])
RESULT ok=1 cmd=extend size=496 classes=248 seeds=40 best_gain=0 best_added=24 best_removed=24 frames=336 incomplete=0 improved=0 seconds=10.9
```

— the added frame is exactly the 496's original frame #1.

**What the frame level structurally cannot see.** The two (80,80) atoms are **invisible to this
search, by construction**: they add 80 vertices = 40 classes of conflict 4 (8 classes), 6 (16) and 8
(16) which are only mutually *non-adjacent*, not mutually orthogonal, so they are not a clique of the
orthogonality graph and no frame-level move can express them. That is the one real limitation of the
frame level: it sees exactly the plateau moves whose added set is an orthogonal clique, and the 496
has (40,40)-class plateau moves that are not. A class-level "add an independent set, remove its
conflicts" search — the GPU search of §4, or `frames ls` with the clique requirement dropped — is
strictly stronger here.

### 5.5 Greedy frame unions and class-level tabu search

`frames greedy --restarts 100` builds an independent class set by repeatedly adding a maximal
orthogonal clique of the currently *free* classes (conflict count 0): the first clique is a random
full frame, then, while the free pool is large, another random frame inside it, otherwise an exact
maximum clique of the induced free graph. 100 restarts (`--restarts 100 --seed 1`, 4067 s):

```
size distribution (vectors): 192:1 196:2 200:3 202:3 204:8 206:1 208:13 210:15 212:9 214:10 216:14 218:6 220:6 222:4 224:2 226:2 228:1
second clique size: 17:1 18:13 19:51 20:18 21:5 22:7 24:5
mean 212.0 vectors, best 228 vectors (496 = the record)
RESULT ok=1 cmd=greedy restarts=100 best=228 mean=212.0 seconds=4067.7
```

* **Mean 212, best 228 vectors — 46 % of the record.** A typical run is
  `24+19+14+12+10+8+7+5+4+3+2+1`: the first frame is free, the second clique is already down to ~19,
  and the union dies at ~105 classes.
* A full frame kills a lot of the space: after one frame only ≈ 7800–11900 of the remaining 98256
  classes are still conflict-free (8–12 %), even though 46575 of them are orthogonal to any given
  class — conflict, not orthogonality, is what shrinks the pool.
* In 5 of the 100 restarts the *second* clique is another full frame (the `24:5` entry; the low
  "free classes after the first frame" values 836–1066 are that field being overwritten after the
  second frame). Two disjoint frames leave only ~1000 free classes, and those runs end with the
  **smallest** unions (192–202 vectors) — greedily stacking frames is actively bad, consistent with
  the record's own shape: the 496 contains only three frames among 2718 maximal cliques and 28 of its
  29 blocks are octuples.
* For comparison, vertex-level greedy (§2.5) reaches 225–236 vectors and is improved to 228–240 by
  small swaps, so the frame-first constraint costs rather than gains at this scale.

`frames ls` is a class-level tabu local search. The move: pick a non-tabu class `c ∉ X` of minimum
conflict count, run the clique search around it with `min_gain = −max_drop`, apply the best clique
(remove its conflicts, add it), then *drain* — greedily add every class that has become free. Removed
classes are tabu for `tenure` moves.

| start | options | moves (impr/plateau/wors/rej) | size range visited (classes) | best |
|---|---|---|---|---|
| the 496 (248 classes) | tmax 6, max-drop 2, 600 s | 6633 (134/2255/2060/2184) | 228 … 248, mode 240 | **248 = 496 vectors** (never more) |
| greedy seed 1 (105 classes) | tmax 6, max-drop 2, 600 s | 182 (0/181/1/0) | 107 … 147 | **147 = 294 vectors** |
| greedy seed 2 (107 classes) | tmax 8, max-drop 4, 600 s | 91 (0/91/0/0) | 109 … 138 | **138 = 276 vectors** |

* **From the 496 the search cannot climb.** Every move at `max_drop = 2` costs classes; the walk
  drops to ≈ 240 immediately (`240:1582` of the 4449 accepted moves; 2184 more were rejected
  outright) and returns to 248 only 14 times in 600 s, never above. This is the class-level
  counterpart of "no `(k, k+1)`-swap for `k ≤ 12`" (§2) and "no 497 in 20k × 512 GPU chains" (§4).
* **From a greedy union it climbs steadily but saturates far below.** Every accepted move has gain 0
  at the clique level; the growth comes entirely from the *drain* step (free classes released by the
  swap), and it stalls at 138–147 classes (276–294 vectors). Moves are slow (3–7 s) because the pool
  around a low-conflict class of a small set is huge; the vertex-level GPU search reaches ≈ 300
  vectors from greedy sets far faster, so the frame level buys nothing here.

Verified: `build/t33/tools/verify_s runs/frames/ls_best_294_20260827T052141Z.txt` →
`RESULT ok=1 size=294 antipodal=1 max_offdiag=8 gram=-32:147,-8:12434,0:18056,8:12434 free=0 maximal=1`.

### 5.6 Verdict on the frame level, and its RESULT lines

What the local frame level bought: (1) it exposed the 496's plateau structure as a **frame-for-frame
swap**, the single clean statement that unifies the four (12,12) moves, and showed the move is an
involution; (2) it gave the class-level restatement of the 496's rigidity — over 1840 seed classes,
complete enumeration of all orthogonal cliques through them within the low-conflict pools, **max gain
0**. What it did not buy: any set larger than 248 classes. And what it structurally cannot buy: a
plateau or improving move whose added set is independent but not orthogonal — the 496 has such moves
and no clique search can reach them. The frame level is a *restriction* of the class-level search,
not a generalisation: a much smaller search tree and an exact bound, at the price of missing the
non-orthogonal moves. It is a good way to *describe* the 496 and a poor way to improve on it.

```
crosses_496.log        RESULT ok=1 cmd=crosses size=496 classes=248 maximal=2718 clique_number=24 frames=3 disjoint=1 cliques8=7069615 seconds=0.13
crosses_496_cover.log  RESULT ok=1 cmd=crosses size=496 classes=248 maximal=2718 clique_number=24 frames=3 disjoint=1 cliques8=7069615 covers=2 seconds=0.33
count_k3_exact.log     RESULT ok=1 cmd=count class=0 k=3  shape=-1 coords=24 nbhd=46575 exact=502590825       complete=1 nodes=46575      seconds=40.0
count_k4_exact.log     RESULT ok=1 cmd=count class=0 k=4  shape=-1 coords=24 nbhd=46575 exact=1636707770775   complete=1 nodes=502637397  seconds=518.4
count_full.log         RESULT ok=1 cmd=count class=0 k=24 shape=-1 coords=24 nbhd=46575 est=2.05184e+17 est_se=4.56e+15 est_total=8.40228e+20 exact=5963706 complete=0 nodes=9992580212 seconds=1124.5
count_full_seed2.log   RESULT ok=1 cmd=count class=0 k=24 shape=-1 coords=24 nbhd=46575 est=2.09003e+17 est_se=5.95e+15 est_total=8.55869e+20 seconds=299.3
count_full_seed34.log  RESULT ok=1 cmd=count class=0 k=24 shape=-1 coords=24 nbhd=46575 est=2.07582e+17 est_se=7.25e+15 est_total=8.50047e+20 seconds=393.9
count_full_seed34.log  RESULT ok=1 cmd=count class=0 k=24 shape=-1 coords=24 nbhd=46575 est=2.08965e+17 est_se=5.83e+15 est_total=8.55710e+20 seconds=362.6
count_shape2_small.log RESULT ok=1 cmd=count class=0 k=6  shape=2 coords=6  nbhd=13  exact=3   complete=1 (= 3!!)
count_shape2_small.log RESULT ok=1 cmd=count class=0 k=8  shape=2 coords=8  nbhd=31  exact=15  complete=1 (= 5!!)
count_shape2_small.log RESULT ok=1 cmd=count class=0 k=10 shape=2 coords=10 nbhd=57  exact=105 complete=1 (= 7!!)
count_shape2_small.log RESULT ok=1 cmd=count class=0 k=12 shape=2 coords=12 nbhd=91  exact=945 complete=1 (= 9!!)
count_shape2_M14_M16   RESULT ok=1 cmd=count class=0 k=14 shape=2 coords=14 nbhd=133 exact=10395  complete=1 nodes=15532202  seconds=0.2   (= 11!!)
count_shape2_M14_M16   RESULT ok=1 cmd=count class=0 k=16 shape=2 coords=16 nbhd=183 exact=135135 complete=1 nodes=617025021 seconds=6.5   (= 13!!)
count_shape2_24.log    RESULT ok=1 cmd=count class=0 k=24 shape=2 coords=24 nbhd=463 est=1.4502e+10 est_se=9.7e+08  (exact 21!! = 1.3749e+10)
extend496_plateau.log  RESULT ok=1 cmd=extend size=496 classes=248 seeds=40   best_gain=0  best_added=24 best_removed=24 frames=336 incomplete=0 improved=0 seconds=43.6
extend496_tier6.log    RESULT ok=1 cmd=extend size=496 classes=248 seeds=320  best_gain=-5 best_added=13 best_removed=18 frames=143 incomplete=0 improved=0 seconds=66.5
extend496_mg1.log      RESULT ok=1 cmd=extend size=496 classes=248 seeds=1840 best_gain=0  best_added=12 best_removed=12 frames=75  incomplete=0 improved=0 seconds=1747.4
extend_plateau496.log  RESULT ok=1 cmd=extend size=496 classes=248 seeds=40   best_gain=0  best_added=24 best_removed=24 frames=336 incomplete=0 improved=0 seconds=10.9
greedy100.log          RESULT ok=1 cmd=greedy restarts=100 best=228 mean=212.0 seconds=4067.7
ls_496.log             RESULT ok=1 cmd=ls start=496 best=496 moves=6633 improving=134 plateau=2255 worsening=2060 rejected=2184 seconds=600.2
ls_greedy1.log         RESULT ok=1 cmd=ls start=210 best=294 moves=182 improving=0 plateau=181 worsening=1 rejected=0 seconds=649.5
ls_greedy2.log         RESULT ok=1 cmd=ls start=214 best=276 moves=91  improving=0 plateau=91  worsening=0 rejected=0 seconds=633.8
test_frames                RESULT test_frames ok=1 failures=0
```

**Nothing exceeded 248 classes / 496 vectors; `runs/found/` was not touched.**

Acceptance (`ctest --test-dir build/t33 -R frames`, `1/1 Test #14: test_frames ... Passed 90.72 sec`).
`tests/test_frames.cpp` (21 s on an idle box) covers: (1) `Bitset` against a `std::vector<bool>`
model; (2) `maximal_cliques` / `k_cliques` / `max_clique` against brute-force subset enumeration on
random graphs; (3) the class machinery on the full vector list (98280 classes, orthogonal degree
46575, conflict degree 4600, conflict counts and compatibility against brute-force dots); (4) the
(±4,±4) matching identity for `M = 6, 8, 10` — exact counts `(M−3)!!` through class 0 and `(M−1)!!`
in all, estimator within 5 s.e.; (5) the 496 → 248 classes, 3 disjoint frames, 2718 maximal cliques
with the expected histogram, 7,069,615 8-cliques; (6) `extend_from_class` on tightness-4 classes of
the 496 against brute force over all cliques of the outsider pool (best gain, clique count, gain
histogram identical; the bound prunes 32768 → 281 nodes) and `apply_extend_hit` independence; (7)
`random_frame` returns a frame through the start and `greedy_frame_union` an independent set that is
the disjoint union of the cliques it reports and is maximal.

Two implementation notes worth keeping: `random_frame` was rewritten to intersect only the *current
candidate set* with the orthogonal set of a sampled class (≈ 25× fewer dots than building a full
98280-bit orthogonal row per sample) and to restart the descent instead of exhaustively backtracking
at shallow levels (a level is abandoned after 6 tries) — 0/200 failures at the new default node
limit, 0.024 s per call, against 10 % failures and ~2 s before; this is what made
`greedy --restarts 100` and `test_frames` (900 s → 21 s) feasible. And the per-level bitset stacks of
the k-clique DFS, the extension DFS and `random_frame` are preallocated to full depth (`k+1`,
`FRAME_SIZE+2`, `FRAME_SIZE+1`), so no `std::vector<Bitset>` can reallocate while a reference into it
is live. No GPU work was needed: every pool that matters has ≤ 1000 classes, and the two expensive
global objects (the 46575-vertex neighbourhood, 270 MB; the 98280-dot conflict scans) are
OpenMP-parallel and take seconds.

---

## 6. Symmetry-restricted search: stabilisers and orbit unions

If a 497 exists with a prescribed symmetry group `H ≤ Co₀`, it is a union of `H`-orbits on the 196560
minimal vectors, and the search collapses to a weighted maximum independent set on the orbit graph —
a far smaller problem. This section computes the stabilisers of the two records exactly, then runs
that reduced search over 95 subgroups.

### 6.1 Implementation

| File | Content |
|---|---|
| `include/kiss/orbits.h`, `src/orbits.cpp` | `Subgroup` (generators as `IndexPerm`), `orbits()` (union-find, numbered by smallest vertex, CSR members), `orbits_invariant`, `is_orbit_union`, `union_of_orbits`; `OrbitGraph` (one adjacency row per orbit representative; explicit CSR when #orbits ≤ `max_explicit`, else implicit/on-demand; a synthetic constructor for tests); `greedy_orbit_mis`, `local_search_orbit_mis` (ILS: force-add a sampled orbit, remove its chosen neighbours with tabu, refill free orbits heaviest-first), `branch_and_bound_orbit_mis` (bitset adjacency, greedy clique-cover weight bound, time limit, starts from a lower bound), `orbit_set_valid`; monomial helpers (`monomial_order`, `monomial_group_order` by closure, `index_perm_order`, `random_monomial_word`, `load/save_monomial_list`). |
| `tools/orbit_mis.cpp` | Driver: one group (`--group-file` monomial list, `--perm-file` u32 index permutations, `--aut-file` den + 24×24 matrices, `--antipodal` adjoins −I) or `--suite DIR` for the whole table (`rows.txt`, `table.md`). Every best union is re-verified with `verify_independent`; anything larger than the reference set is written to `--found` (`runs/found/orbit_<H>_<size>.txt`) and announced. |
| `python/tools/stabilizer_496.py` | Exact monomial stabiliser plus Gram automorphisms and their extension to Co₀. Writes `runs/orbits/stab496/{stabiliser_monomials,stabiliser_generators}.txt`, `co0_stab_gen<k>.txt` (Aut files, den 8), `co0_stabiliser_elements.txt`. Options `--aut`, `--perm-file` stabilise `g·S` instead. Uses only `kiss_ref` + `group.m24`, independent of the C++. |
| `python/tools/m24_subgroups.py` | sympy: Sylow p-subgroups (p = 2, 3, 5, 7, 11, 23), derived subgroup / centre of Syl₂, pointwise stabilisers of 1..5 points (M23, M22, M21, M20, M19), Sylow 2-/3-subgroups of PSL(2,23), reduced to ≤ 3 random generators; exported as monomial lists to `runs/orbits/subgroups/`. |
| `python/tools/orbit_milp.py` | Exact MILP attempt on the reduced orbit graph written by `orbit_mis --export-graph` (HiGHS via `scipy.optimize.milp`, greedy edge-clique-cover constraints, time limit; reports dual bound / gap). |
| `tests/test_orbits.cpp`, `tests/tasks/T3_4.cmake` | ctest `test_orbits` (CPU; exit 77 only if `data/adj.u32` is absent). |
| `runs/orbits/` | `stab496/`, `stab488/`, `probe/` (the ξ·S496 probe), `subgroups/`, `table/` (suite log, rows, table), `deep/` (long runs on the Co₀ stabiliser: B&B logs, the exported reduced graph, MILP log and its 64 chosen orbits), `auts/` (6 random Co₀ elements used as conjugators), `run_t34.sh` (the exact command lines). |

Build: `cmake -S . -B build/t34 -G Ninja -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build
build/t34 --target kiss orbit_mis test_orbits`.

### 6.2 The monomial stabiliser is {±I} (exact)

A monomial element `m = D_c P_π` (`π ∈ M24`, `c` a Golay codeword) fixes `S` iff `mS = S`. `π` must
preserve the three shapes, hence the multiset of octad supports of the 258 (±2⁸) rows (119 distinct
octads, multiplicities {2: 109, 4: 10}), the multiset of "3-positions" of the 232 (∓3, ±1²³) rows and
the pair supports of the six (±4, ±4) rows ({7,12}, {2,22}, {1,3}). From these, coordinates get
colours `(n_oct, n_tri, n_ff)` refined three rounds by the multiset of (pair-octad count, neighbour
colour) — **all 24 coordinates end in distinct colour classes**, so `π = id` is forced before any
backtracking. (The code still runs the full machinery: base of 5 points, stabiliser chain from
sympy's Schreier–Sims with base prefix [0,1,2,3,4], chain orbit sizes 24·23·22·21·20·16·3 = |M24|, 48
elements in the 5-point stabiliser, 1 candidate 5-tuple × 48 → exactly 1 `π` preserving all support
multisets.) Signs: with `π` fixed, a (∓3)-row `x` and every `y ∈ S` with `|y| = 3` at the same
position give the only candidate `c = {j : x_j ≠ y_j}`; 12 candidates, 2 of which are codewords
fixing `S`: `c = 0` and `c = 1²⁴`.

```
$ .venv/bin/python python/tools/stabilizer_496.py data/S496.txt --out runs/orbits/stab496
S = S496.txt, |S| = 496
shape counts: octad=258 31=232 44=6
distinct octad supports: 119 multiplicities {2: 109, 4: 10}
distinct 3-positions: 24 multiplicities {12: 5, 10: 3, 14: 5, 16: 1, 4: 2, 2: 1, 6: 5, 8: 2}
44 supports: [[7, 12], [2, 22], [1, 3]]
coordinate colour classes: 24 sizes [1, 1, ..., 1]
base points [0, 1, 2, 3, 4] (colour classes [1, 1, 1, 1, 1]), full base [0, 1, 2, 3, 4, 5, 6], chain orbit sizes [24, 23, 22, 21, 20, 16, 3], product 244823040, 5-point stabiliser 48
5-tuples tried: 1 (bound 1) x 48, pi preserving all support multisets: 1, 0.0s
monomial stabiliser: 2 elements
structure: {'order': 2, 'abelian': True, 'element_orders': {1: 1, 2: 1}, 'centre_order': 2, 'derived_order': 1}
-I in stabiliser: True
distinct coordinate permutations (image in M24): 1; sign kernel order: 2
orbits on S: 248, sizes {2: 248}
orbits on C: 98280, sizes {2: 98280}
S is a union of stabiliser orbits on C: True
```

An **independent C++ check** re-derives the same coordinate invariants (`n_oct`, `n_tri`, `n_ff` and
the sorted profile of pair-octad counts), finds them pairwise distinct for the 24 coordinates (so
`π = id`), and enumerates sign masks:
`stabiliser_496: 24 distinct coordinate invariants, 12 candidate sign masks, 2 fix S -> |Stab| = 2`.
The 488 gives the identical outcome (24 singleton colour classes, stabiliser {±I}), in
`runs/orbits/stab488/`. A limited check of conjugates of the monomial group was also run
(`stabilizer_496.py data/S496.txt --aut data/group/xi.txt`, i.e. the stabiliser of `ξ·S` in 2¹²:M24,
equivalently of `S` in `ξ⁻¹(2¹²:M24)ξ`): again 24 singleton classes and {±I}
(`runs/orbits/probe/`). That probe is **superseded** by the full Co₀ computation below, which settles
the conjugates question completely.

### 6.3 The full Co₀ stabiliser is exactly 2³ (exact)

Any `g ∈ Co₀` with `gS = S` permutes `S` preserving inner products, i.e. is an automorphism of the
complete graph on `S` with edges coloured by `⟨x, y⟩ ∈ {−32, −8, 0, 8}`. Conversely, since `S`
contains a full frame (so spans R²⁴), a Gram automorphism `f` extends to at most one orthogonal map
`g_f`, and `g_f ∈ Co₀` iff it preserves Λ. `gram_automorphisms()` does colour refinement (1-WL,
vectorised) — 28 stable classes of sizes {4: 6, 8: 3, 16: 10, 32: 9} — then
individualisation–refinement, enumerating **64** automorphisms (`x ↦ −x` among them).
`extend_to_co0()` solves `g_f` by least squares on all 496 rows, rounds `8·g_f` (`8Z²⁴ ⊂ Λ` in the √8
scaling, so `8g` is integral for `g ∈ Co₀`), verifies `g_f S = S[f]` exactly in integers, and tests
`g_f` on all 196560 minimal vectors (all images minimal vectors, bijective).

```
Gram colour refinement: 28 classes, sizes {4: 6, 8: 3, 16: 10, 32: 9}
Gram automorphisms (exact): 64
|Aut(Gram(S))| = 64; x->-x among them: True
Gram automorphisms extending to Leech automorphisms: 8 of 64
full Co_0 stabiliser: |Stab_Co0(S)| = 8; structure {'order': 8, 'abelian': True, 'element_orders': {1: 1, 2: 7}, 'centre_order': 8, 'derived_order': 1, 'orbits_on_S': [(4, 124)]}
Co_0 stabiliser generators written: 3
RESULT ok=1 set='S496.txt' size=496 stab_order=2 m24_image=1 sign_kernel=2 abelian=1 orbits_on_S=248 orbits_on_C=98280 gram_aut=64 gram_classes=28 co0_stab_order=8 seconds=79.6
```

So `Stab_{Co₀}(496) = {±1} × ⟨a, b⟩ ≅ 2³`, elementary abelian of order 8. Element data
(`runs/orbits/stab496/co0_stabiliser_elements.txt`, den 8):

| element | trace | dim fix / dim(−1) | monomial? | fixes / negates in S |
|---|---|---|---|---|
| I | +24 | 24 / 0 | yes | 496 / 0 |
| a | +8 | 16 / 8 | no (entries ±4/8) | 208 / 0 |
| b | +8 | 16 / 8 | no (entries ±2/8, ±6/8) | 144 / 0 |
| ab | −8 | 8 / 16 | no | see the correction below |
| −a, −b | −8 | 8 / 16 | no | 0 / 208, 144 |
| −I | −24 | 0 / 24 | yes | 0 / 496 |

> **Correction to the trace bookkeeping.** The table as originally published listed "b, ab" both at
> trace +8 and `−a, −b, −ab` all at −8. That is **false**. The later structural analysis (see
> `04-structure-of-the-496.md`) re-verified all eight matrices from scratch and found the trace
> multiset to be **{+24, +8 × 3, −8 × 3, −24}**, with the three trace-+8 involutions `t₁, t₂, t₃`
> satisfying `tᵢtⱼ = −t_k`. So no two of them generate a Klein four-group containing the third: the
> *products* of the trace-+8 involutions have trace −8. The group is still `{±1} × ⟨t₁, t₂⟩ ≅ 2³` —
> only the trace bookkeeping was wrong, and no downstream orbit computation depended on it. The three
> trace-+8 involutions fix 208, 144 and 144 members of `S` respectively.

The three involutions with an 8-dimensional (−1)-eigenspace are of the octad-sign-change class of
Co₀; in these coordinates none of them is monomial, which is exactly why the monomial search sees
only {±I}. Their (−1)-spaces are √2·E8 sections — the entry point to the Turyn decomposition
developed in `04-structure-of-the-496.md`. On `S` the 2³ has **124 orbits of size 4**, each of the
form `{±x, ±y}` with `x ⊥ y` (Gram pattern −32, −32, 0, 0, 0, 0).

The C++ side re-checks all of this: `index_permutation(load_aut)` throws unless the matrix is an
automorphism; each generator fixes the index set of the 496; 2000 random inner products are
preserved; the 496 is a union of 124 orbits of the 3 generators and is recovered exactly.

The 488 behaves the same way: `Aut(Gram)` has order 16, 8 elements extend, giving `Stab_{Co₀} ≅ 2³`
again, with 122 orbits of size 4 of the same `{±x, ±y}` shape (`runs/orbits/stab488/`).

### 6.4 Orbit machinery and its tests

`orbits()` is one union-find pass per generator (numbering by smallest vertex, CSR members). In
`OrbitGraph`, for an orbit `o` with representative `r`, the orbit ids of `row(r)` (4600 entries) are
exactly the orbits conflicting with `o` (including `o` itself iff `o` is self-conflicting) — one
adjacency row per orbit, OpenMP over orbits; explicit CSR when #orbits ≤ `max_explicit` (default
12,000; 26,280 orbits ≈ 0.5 GB), otherwise the neighbour list is recomputed from the row on demand.
Weighted MIS: greedy (weight descending, degree ascending) → ILS (sampled force-add with loss
evaluation, tabu tenure 10, refill heaviest-first, restart from best after 2000 non-improving steps)
→ branch and bound on the explicit graph restricted to non-self-conflicting orbits (bitset adjacency,
vertices in weight-descending order, greedy clique cover with per-clique max weight as the bound,
prefix bounds, time limit, seeded with the best known lower bound — the reference set itself when it
is an orbit union).

```
bb_vs_brute_force: 8 random graphs (n=24) agree
C23: 8548 orbits, sizes 1x2 23x8546
C23x2: 4274 orbits
stabiliser_496: 24 distinct coordinate invariants, 12 candidate sign masks, 2 fix S -> |Stab| = 2
stabiliser_496: 98280 orbits on C, the 496 = union of 248 orbits (recovered exactly)
stabiliser_496 (Co_0, 3 generators): 26280 orbits on C, the 496 = union of 124 orbits, sizes 4x124
23:11: 788 orbits, 778 self-conflicting, 271866 edges (95 ms)
conflict vs brute force: 50/50 pairs agree (43 conflicting), 20/20 self flags agree
23:11 MIS: greedy 301, ls 301, bb 301 (optimal=1, 1 nodes, 0.0 s)
RESULT ok=1 failures=0 ms=2441
```

Namely: (1) branch and bound equals brute force (2²⁴ subsets) on 8 random weighted graphs with
self-conflicting vertices, with and without an initial bound, and greedy ≤ LS ≤ B&B with all unions
valid; (2) orbit invariants for `⟨α⟩` (sizes {1: 2, 23: 8546}, summing to N, invariant, numbered by
smallest vertex; the two fixed vectors are of (−3, 1²³) type with the 3 at ∞) and `⟨α, −I⟩`, the
trivial group giving N singletons, plus an `is_orbit_union` round trip and partial-orbit detection;
(3) the stabiliser of the 496 as above; (4) for 23:11, the conflict relation against brute-force
cross pairs on 50 orbit pairs (half biased to actual neighbours), self flags on 20 orbits, symmetry,
degrees, and greedy/LS/B&B unions passing `verify_independent`.

```
$ ctest --test-dir build/t34 -R orbits
1/1 Test #13: test_orbits ......................   Passed    2.46 sec
100% tests passed, 0 tests failed out of 1
```

### 6.5 The subgroup table: 95 groups, nothing above 496

`runs/orbits/run_t34.sh` → `orbit_mis --suite runs/orbits/table --subgroups-dir runs/orbits/subgroups
--ls-seconds 10 --bb-seconds 30 --seed 1` (log `runs/orbits/table/suite.log`, rows `rows.txt`, table
`table.md`). **95 groups, 0 errors, 1868 s**:
`RESULT ok=1 groups=95 errors=0 best=496 best_group=Stab(496) reference=496 exceeded=0`.

Naming: `Ck` = the cyclic group of a random M24 element of order `k` (pure coordinate permutation);
`Ck±` = the same with −I adjoined; `Cks<m>` = the same permutation composed with a random Golay sign
change (element order `m`). `Syl*`, `M19..M23`, `PSL_Syl*` are the sympy subgroups (plain and `±`).
`D2n=<xi,m>` are the dihedral groups generated by Conway's ξ and a monomial involution `m`.
`g_k.H.g_k^{-1}` are conjugates by random Co₀ elements (`runs/orbits/auts/`) — a sanity check: they
must and do reproduce the rows of `C23` and `23:11` (orbit structure, edges, best), since the
conflict graph is Co₀-invariant. Columns: `B&B` = branch-and-bound value (only when the orbit graph
is explicit, i.e. ≤ 12,000 orbits); `opt` = B&B finished within 30 s (the value is then the exact
maximum-weight orbit union); `best` = max of greedy/LS/B&B/reference; every `best` union was
re-verified with `verify_independent`.

Summary:

* **Only the two stabilisers reach 496** — trivially, since the 496 is a union of their orbits (248
  antipodal pairs under {±I}; 124 orthogonal quadruples under the 2³). No other `H` among the 95 has
  the 496 as an orbit union (`union496 = no` everywhere else), consistent with §6.2–6.3: any `H` with
  an invariant 496 must lie in the 2³.
* **Best non-trivial unions** (none proven optimal within 30 s; all far below 496): `C14±` (|H| = 28)
  402, `C21±`/`C21s42` (|H| = 42) 384, `D4=<ξ,−I>` 344 (LS only, implicit graph), `C8±` 342, `C7s14`
  339, `C14` 337, `C11±` 334, `C23`/`C23±`/`Syl23` 324. Adjoining −I never hurts and usually helps
  (the search space has half the orbits and the 496 is antipodal anyway).
* **Exact rows (23 with `opt = yes`)**: the large groups have tiny or empty answers — `M24`,
  `PSL(2,23)`, `2¹²:M24`, `2¹²:23`, `23:11.s`: **0** (every orbit is self-conflicting); `M23` 2,
  `M22` 4–5, `M21` 6, `M20` 8–9, `M19` 48; `23:11` (order 253): **301** exactly (and the same for
  both Co₀-conjugates); `Syl3±` 264; everything containing a big 2-group (`2¹²`, `Syl2`, `Syl2±`,
  `Syl2_derived±`, `23:11±`, `M19±`) is capped at **48 = one coordinate frame** (the only
  2¹²-invariant independent sets are unions of sign classes, and the sign classes of the (±2⁸) and
  (∓3, ±1²³) shapes are self-conflicting).
* Nothing exceeds 496: `exceeded=0`, `runs/found/` empty.

| H | \|H\| | #orbits | self-conf. | edges | greedy | LS | B&B | best | opt | S496 union? | s | orbit sizes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stab(496) | 2 | 98280 | 0 | 226044000 | 266 | 266 | — | **496** | — | yes | 18.6 | 2x98280 |
| Stab_Co0(496) | 8 | 26280 | 23040 | 47988000 | 396 | 422 | — | **496** | — | yes | 12.6 | 8x23040,4x2880,2x360 |
| 2^12 | 4096 | 1059 | 783 | 152628 | 48 | 48 | 48 | **48** | no | no | 40.2 | 4096x24,128x759,4x276 |
| 2^12:23 | 94208 | 47 | 47 | 1037 | 0 | 0 | 0 | **0** | yes | no | 0.1 | 94208x1,4096x1,2944x33,92x12 |
| 2^12:M24 | >3000000 | 3 | 3 | 3 | 0 | 0 | 0 | **0** | yes | no | 2.0 | 98304x1,97152x1,1104x1 |
| C2 | 2 | 98412 | 8448 | 222267744 | 258 | 258 | — | **258** | — | no | 16.2 | 2x98148,1x264 |
| C2± | 4 | 49272 | 8448 | 109245744 | 270 | 270 | — | **270** | — | no | 13.2 | 4x49008,2x264 |
| C2s4 | 4 | 50244 | 31296 | 104911080 | 214 | 232 | — | **232** | — | no | 10.8 | 4x48060,2x2136,1x48 |
| C3 | 3 | 65520 | 2160 | 147359880 | 255 | 258 | — | **258** | — | no | 13.2 | 3x65520 |
| C3± | 6 | 32760 | 1200 | 72081480 | 252 | 264 | — | **264** | — | no | 12.1 | 6x32760 |
| C3s6 | 6 | 32760 | 1200 | 72081480 | 258 | 264 | — | **264** | — | no | 12.6 | 6x32760 |
| C4 | 4 | 49212 | 4728 | 108826584 | 244 | 254 | — | **254** | — | no | 14.1 | 4x49074,2x126,1x12 |
| C4± | 8 | 24642 | 4728 | 52332072 | 274 | 274 | — | **274** | — | no | 12.2 | 8x24504,4x126,2x12 |
| C4s8 | 8 | 24570 | 756 | 53516862 | 256 | 264 | — | **264** | — | no | 12.6 | 8x24570 |
| C5 | 5 | 39408 | 9360 | 81348960 | 254 | 259 | — | **259** | — | no | 12.8 | 5x39288,1x120 |
| C5± | 10 | 19704 | 4800 | 39998880 | 224 | 274 | — | **274** | — | no | 12.3 | 10x19644,2x60 |
| C5s10 | 10 | 19732 | 6432 | 38436008 | 273 | 273 | — | **273** | — | no | 13.2 | 10x19618,5x52,2x58,1x4 |
| C6 | 6 | 32804 | 4176 | 70965096 | 240 | 264 | — | **264** | — | no | 13.2 | 6x32716,3x88 |
| C6± | 12 | 16424 | 3664 | 33444120 | 228 | 276 | — | **276** | — | no | 13.1 | 12x16336,6x88 |
| C6s12 | 12 | 16380 | 1200 | 34153800 | 240 | 276 | — | **276** | — | no | 11.7 | 12x16380 |
| C7 | 7 | 28116 | 6384 | 56871636 | 262 | 262 | — | **262** | — | no | 14.3 | 7x28074,1x42 |
| C7± | 14 | 14058 | 3327 | 27528048 | 262 | 276 | — | **276** | — | no | 11.7 | 14x14037,2x21 |
| C7s14 | 14 | 14370 | 10463 | 25436428 | 230 | 339 | — | **339** | — | no | 12.8 | 14x13729,7x616,2x17,1x8 |
| C8 | 8 | 25190 | 16842 | 47978417 | 265 | 336 | — | **336** | — | no | 12.7 | 8x24030,4x1015,2x115,1x30 |
| C8± | 16 | 12612 | 8584 | 23124604 | 238 | 342 | — | **342** | — | no | 11.8 | 16x12000,8x536,4x60,2x16 |
| C8s8 | 8 | 25190 | 16842 | 47978417 | 250 | 250 | — | **250** | — | no | 13.0 | 8x24030,4x1015,2x115,1x30 |
| C10 | 10 | 19732 | 6432 | 38436008 | 231 | 261 | — | **261** | — | no | 13.1 | 10x19618,5x52,2x58,1x4 |
| C10± | 20 | 9880 | 4056 | 17846200 | 236 | 300 | 314 | **314** | no | no | 42.0 | 20x9796,10x52,4x28,2x4 |
| C10s20 | 20 | 9868 | 3688 | 17470941 | 248 | 301 | 310 | **310** | no | no | 41.0 | 20x9809,10x25,5x2,4x29,... |
| C11 | 11 | 17880 | 4032 | 34786338 | 253 | 268 | — | **268** | — | no | 11.6 | 11x17868,1x12 |
| C11± | 22 | 8940 | 2256 | 16250034 | 246 | 290 | 334 | **334** | no | no | 42.3 | 22x8934,2x6 |
| C11s11 | 11 | 17880 | 4032 | 34786338 | 242 | 270 | — | **270** | — | no | 12.6 | 11x17868,1x12 |
| C12 | 12 | 16404 | 2696 | 33343480 | 240 | 270 | — | **270** | — | no | 11.7 | 12x16358,6x42,3x4 |
| C12± | 24 | 8214 | 2344 | 14800236 | 228 | 306 | 312 | **312** | no | no | 41.2 | 24x8168,12x42,6x4 |
| C12s12 | 12 | 16404 | 2696 | 33343480 | 240 | 261 | — | **261** | — | no | 12.7 | 12x16358,6x42,3x4 |
| C14 | 14 | 14370 | 10463 | 25436428 | 316 | 337 | — | **337** | — | no | 12.3 | 14x13729,7x616,2x17,1x8 |
| C14± | 28 | 7194 | 5383 | 11727188 | 360 | 374 | 402 | **402** | no | no | 41.8 | 28x6856,14x325,4x8,2x5 |
| C14s28 | 28 | 7038 | 2109 | 12223010 | 260 | 290 | 318 | **318** | no | no | 41.5 | 28x7010,14x17,4x10,2x1 |
| C15 | 15 | 13240 | 6806 | 23142838 | 243 | 269 | — | **269** | — | no | 12.7 | 15x13046,5x150,3x38,1x6 |
| C15± | 30 | 6620 | 3596 | 10615993 | 208 | 310 | 328 | **328** | no | no | 40.9 | 30x6523,10x75,6x19,2x3 |
| C15s15 | 15 | 13240 | 6806 | 23142838 | 246 | 261 | — | **261** | — | no | 11.2 | 15x13046,5x150,3x38,1x6 |
| C21 | 21 | 9372 | 2936 | 16545077 | 273 | 303 | 303 | **303** | no | no | 41.3 | 21x9358,3x14 |
| C21± | 42 | 4686 | 1780 | 6998622 | 258 | 342 | 384 | **384** | no | no | 40.6 | 42x4679,6x7 |
| C21s42 | 42 | 4686 | 1780 | 6998622 | 258 | 300 | 384 | **384** | no | no | 40.6 | 42x4679,6x7 |
| C23 | 23 | 8548 | 2272 | 14833648 | 253 | 301 | 324 | **324** | no | no | 41.1 | 23x8546,1x2 |
| C23± | 46 | 4274 | 1444 | 6223819 | 232 | 324 | 324 | **324** | no | no | 40.6 | 46x4273,2x1 |
| C23s46 | 46 | 4274 | 1444 | 6223819 | 232 | 324 | 324 | **324** | no | no | 40.6 | 46x4273,2x1 |
| 23:11 | 253 | 788 | 778 | 271866 | 301 | 301 | 301 | **301** | yes | no | 10.1 | 253x776,23x10,1x2 |
| 23:11± | 506 | 394 | 391 | 76124 | 48 | 48 | 48 | **48** | yes | no | 10.1 | 506x388,46x5,2x1 |
| 23:11.s | 518144 | 11 | 11 | 39 | 0 | 0 | 0 | **0** | yes | no | 0.0 | 47104x2,32384x2,16192x2,2048x2,... |
| PSL(2,23) | 6072 | 50 | 50 | 1099 | 0 | 0 | 0 | **0** | yes | no | 0.1 | 6072x24,3036x14,1518x3,759x2,... |
| PSL(2,23)± | 12144 | 31 | 31 | 445 | 0 | 0 | 0 | **0** | yes | no | 0.0 | 12144x10,6072x9,3036x4,1518x4,... |
| M24 | >3000000 | 16 | 16 | 82 | 0 | 0 | 0 | **0** | yes | no | 3.4 | 53130x1,30912x2,21252x2,12144x2,... |
| M19 | 48 | 5949 | 5795 | 4571189 | 47 | 48 | 48 | **48** | no | no | 40.7 | 48x3128,24x1200,16x464,12x715,... |
| M19± | 96 | 3012 | 2921 | 2153006 | 46 | 48 | 48 | **48** | yes | no | 10.7 | 96x1544,48x640,32x232,24x340,... |
| M20 | 960 | 891 | 859 | 142657 | 8 | 9 | 9 | **9** | yes | no | 10.1 | 960x47,480x152,320x58,240x87,... |
| M20± | 1920 | 456 | 440 | 62447 | 8 | 8 | 8 | **8** | yes | no | 10.1 | 1920x20,960x83,640x28,480x40,... |
| M21 | 20160 | 270 | 252 | 14081 | 6 | 6 | 6 | **6** | yes | no | 10.1 | 6720x1,3360x16,2520x15,2016x6,... |
| M21± | 40320 | 138 | 129 | 6186 | 6 | 6 | 6 | **6** | yes | no | 10.1 | 6720x9,5040x6,4032x3,3360x7,... |
| M22 | 443520 | 105 | 97 | 2406 | 5 | 5 | 5 | **5** | yes | no | 10.1 | 18480x1,9240x2,7392x6,6160x6,... |
| M22± | 887040 | 54 | 50 | 1031 | 4 | 4 | 4 | **4** | yes | no | 10.1 | 18480x2,14784x3,12320x3,9240x1,... |
| M23 | >3000000 | 42 | 40 | 460 | 2 | 2 | 2 | **2** | yes | no | 13.3 | 35420x1,15456x2,14168x4,8855x2,... |
| M23± | >3000000 | 22 | 21 | 185 | 2 | 2 | 2 | **2** | yes | no | 11.0 | 35420x1,30912x1,28336x2,17710x1,... |
| PSL_Syl2 | 8 | 24738 | 9204 | 51028114 | 254 | 258 | — | **258** | — | no | 11.6 | 8x24411,4x310,2x15,1x2 |
| PSL_Syl2± | 16 | 12453 | 7724 | 23159103 | 276 | 276 | — | **276** | — | no | 11.3 | 16x12132,8x293,4x24,2x4 |
| PSL_Syl3 | 3 | 65520 | 2160 | 147359880 | 255 | 258 | — | **258** | — | no | 13.8 | 3x65520 |
| PSL_Syl3± | 6 | 32760 | 1200 | 72081480 | 252 | 276 | — | **276** | — | no | 12.0 | 6x32760 |
| Syl11 | 11 | 17880 | 4032 | 34786338 | 243 | 269 | — | **269** | — | no | 11.4 | 11x17868,1x12 |
| Syl11± | 22 | 8940 | 2256 | 16250034 | 246 | 312 | 312 | **312** | no | no | 41.1 | 22x8934,2x6 |
| Syl2 | 1024 | 685 | 663 | 122360 | 48 | 48 | 48 | **48** | yes | no | 10.1 | 1024x48,512x165,256x154,128x133,... |
| Syl2± | 2048 | 375 | 359 | 46250 | 48 | 48 | 48 | **48** | yes | no | 10.1 | 2048x23,1024x81,512x79,256x71,... |
| Syl23 | 23 | 8548 | 2272 | 14833648 | 230 | 301 | 324 | **324** | no | no | 41.1 | 23x8546,1x2 |
| Syl23± | 46 | 4274 | 1444 | 6223819 | 232 | 324 | 324 | **324** | no | no | 40.6 | 46x4273,2x1 |
| Syl2_centre | 2 | 100440 | 61440 | 214582560 | 237 | 237 | — | **237** | — | no | 15.3 | 2x96120,1x4320 |
| Syl2_centre± | 4 | 50280 | 30720 | 107163360 | 210 | 258 | — | **258** | — | no | 13.4 | 4x48000,2x2280 |
| Syl2_derived | 64 | 4888 | 4744 | 3604783 | 48 | 48 | 48 | **48** | no | no | 40.5 | 64x2056,32x1556,16x756,8x286,... |
| Syl2_derived± | 128 | 2540 | 2440 | 1517245 | 48 | 48 | 48 | **48** | yes | no | 15.0 | 128x1016,64x772,32x432,16x120,... |
| Syl3 | 27 | 7672 | 7108 | 10095152 | 108 | 270 | 270 | **270** | no | no | 40.8 | 27x7108,9x492,3x72 |
| Syl3± | 54 | 3836 | 3572 | 4406506 | 210 | 264 | 264 | **264** | yes | no | 10.6 | 54x3554,18x246,6x36 |
| Syl5 | 5 | 39408 | 9360 | 81348960 | 254 | 273 | — | **273** | — | no | 12.9 | 5x39288,1x120 |
| Syl5± | 10 | 19704 | 4800 | 39998880 | 234 | 280 | — | **280** | — | no | 11.3 | 10x19644,2x60 |
| Syl7 | 7 | 28116 | 6384 | 56871636 | 248 | 256 | — | **256** | — | no | 12.0 | 7x28074,1x42 |
| Syl7± | 14 | 14058 | 3327 | 27528048 | 328 | 328 | — | **328** | — | no | 11.2 | 14x14037,2x21 |
| D24=<xi,gamma> | 24 | 8570 | 7330 | 13955157 | 217 | 260 | 260 | **260** | no | no | 41.1 | 24x7874,12x554,8x81,6x29,... |
| D4=<xi,-I> | 4 | 50280 | 30720 | 107163360 | 226 | 344 | — | **344** | — | no | 12.7 | 4x48000,2x2280 |
| D4=<xi,flip(oct0)> | 4 | 50340 | 30720 | 107047152 | 224 | 230 | — | **230** | — | no | 12.6 | 4x47952,2x2364,1x24 |
| D12=<xi,flip(oct1)> | 12 | 17646 | 16302 | 30879711 | 126 | 148 | — | **148** | — | no | 11.6 | 12x15246,6x2142,4x120,2x138 |
| D12=<xi,flip(oct2)> | 12 | 17668 | 16326 | 30551764 | 158 | 186 | — | **186** | — | no | 10.9 | 12x15236,6x2153,4x120,3x18,... |
| D4=<xi,inv0> | 4 | 50346 | 33504 | 105361368 | 212 | 224 | — | **224** | — | no | 12.9 | 4x47964,2x2322,1x60 |
| D8=<xi,inv1> | 8 | 26253 | 24356 | 48884090 | 152 | 174 | — | **174** | — | no | 11.6 | 8x23058,4x2856,2x333,1x6 |
| D24=<xi,inv2> | 24 | 8567 | 7361 | 13828257 | 226 | 296 | 296 | **296** | no | no | 41.0 | 24x7871,12x561,8x84,6x28,... |
| g0.C23.g^-1 | 23 | 8548 | 2272 | 14833648 | 253 | 301 | 324 | **324** | no | no | 41.2 | 23x8546,1x2 |
| g0.(23:11).g^-1 | 253 | 788 | 778 | 271866 | 301 | 301 | 301 | **301** | yes | no | 10.2 | 253x776,23x10,1x2 |
| g1.C23.g^-1 | 23 | 8548 | 2272 | 14833648 | 230 | 278 | 324 | **324** | no | no | 41.1 | 23x8546,1x2 |
| g1.(23:11).g^-1 | 253 | 788 | 778 | 271866 | 301 | 301 | 301 | **301** | yes | no | 10.1 | 253x776,23x10,1x2 |

### 6.6 Exact search over 2³-invariant sets — the one genuinely open question

Under `Stab_{Co₀}(496) ≅ 2³` the orbit graph has 26,280 orbits (23,040 of size 8, 2880 of size 4, 360
of size 2). **All 23,040 size-8 orbits are self-conflicting** (each contains a 60° pair), so only the
3240 orbits of the smaller sizes — `{±x, ±y}` with `x ⊥ y` (size 4) and `{±x}` (size 2, the 720
vectors fixed up to sign by `⟨a, b⟩`) — can enter a union. The reduced graph on these 3240 orbits has
456,480 conflict edges (mean degree 282; `--export-graph` writes it to
`runs/orbits/deep/stab496_reduced_graph.txt`). The 496 is 124 of the 2880 size-4 orbits
(`union496=1`, recovered exactly from the orbit ids).

| run | LS | B&B | optimal? | nodes | s | log |
|---|---|---|---|---|---|---|
| table row (ls 10 s, no B&B: implicit graph at 12,000) | 422 | — | — | — | 12.6 | `table/suite.log` |
| `Stab_Co0_496_deep` (ls 300 s, bb 120 s, seed 2) | 480 | 496 | no | 134,400 | 424 | `deep/stab496.log` |
| `Stab_Co0_496_exact` (ls 60 s, **bb 1800 s**, seed 3) | 480 | 496 | **no** | 1,640,192 | 1861 | `deep/stab496_exact.log` |
| `Stab_Co0_496_export` (ls 5 s, bb 5 s) | 476 | 496 | no | 35,840 | 10.6 | export run |
| HiGHS MILP, clique-cover formulation, **1800 s** (`python/tools/orbit_milp.py`) | — | 256 (heuristic) | no, LP bound 1476 | 0 | 1857 | `deep/stab496_milp.log` |

```
ROW name=Stab_Co0_496_exact order=? orbits=26280 self=23040 edges=47988000 explicit=1 greedy=396 ls=480 bb=496 optimal=0 nodes=1640192 best=496 union496=1 seconds=1861.4 sizes=8x23040,4x2880,2x360
RESULT ok=1 groups=1 errors=0 best=496 best_group=Stab_Co0_496_exact reference=496 exceeded=0 seconds=1861.7
```

```
$ .venv/bin/python python/tools/orbit_milp.py runs/orbits/deep/stab496_reduced_graph.txt --time 1800
graph: 26280 orbits, 3240 usable (weights {2: 360, 4: 2880}), 456480 edges, reference = union of 124 orbits (weight 496)
greedy edge clique cover: 43620 cliques, sizes min/mean/max = 15/15.6/23, 55.8s
  Status            Time limit reached
  Primal bound      -256
  Dual bound        -1476
  LP iterations     39873 (total)
best union: 256 vectors from 64 orbits; valid=True; dual bound 1476.000; gap 4.766; status=1 (Time limit reached.); optimal=0; 1857.0s
RESULT ok=1 usable=3240 edges=456480 cliques=43620 best=256 dual_bound=1476.000 upper=1476 optimal=0 reference=496 seconds=1857.0
```

**Conclusion.** The 496 is a 2³-invariant independent set and is the largest orbit union found by
every method (ILS from scratch reaches 480 in 5 min; B&B seeded with 480 finds 496 quickly and then
never improves it in 1.6 M nodes; B&B seeded with the 496 finds nothing better). **Whether 496 is the
exact maximum over 2³-invariant sets was left open here** — the greedy clique-cover bound of the B&B
is too weak to close the 3240-vertex reduced graph in 30 min, and the HiGHS MILP did not get past its
root LP in 30 min (the LP bound 1476 is useless). Closing it needs a stronger relaxation. §9 is the
dedicated attack on exactly this question.

The same pipeline on the 488 (`Stab_Co0_488_deep`, 2³, 122 size-4 orbits) gives ls 484, B&B 488 (not
proven), and no 2³-invariant set larger than the 488:

```
ROW name=Stab_Co0_488_deep order=? orbits=26280 self=23040 edges=47988000 explicit=1 greedy=408 ls=484 bb=488 optimal=0 nodes=121088 best=488 union496=1 seconds=183.7 sizes=8x23040,4x2880,2x360
```

Notes: the local search on the huge quotients (`⟨−I⟩`: 98,280 orbits, implicit mode) is weak (≈ 266)
— it exists only to seed B&B on the small quotients. `highspy` is not installed here, but
`scipy.optimize.milp` (HiGHS 1.8) is; it was tried on the one graph where B&B stalls and was strictly
worse than the B&B. All other stalled rows are far below 496 and were not pushed further.
`runs/found/` remains empty.

```
# python/tools/stabilizer_496.py
RESULT ok=1 set='S496.txt' size=496 stab_order=2 m24_image=1 sign_kernel=2 abelian=1 orbits_on_S=248 orbits_on_C=98280 gram_aut=64 gram_classes=28 co0_stab_order=8 seconds=79.6
RESULT ok=1 set='S488.txt' size=488 stab_order=2 m24_image=1 sign_kernel=2 abelian=1 orbits_on_S=244 orbits_on_C=98280 seconds=0.5
RESULT ok=1 set='S496.txt mapped by xi.txt' size=496 stab_order=2 m24_image=1 sign_kernel=2 abelian=1 orbits_on_S=248 orbits_on_C=-1 seconds=0.2
# random Co_0 conjugators (runs/orbits/auts)
RESULT ok=1 seed=3404 count=6 slots=10 burnin=100 gens=5 out=runs/orbits/auts ip_fail=0 ms=96.4
# tools/orbit_mis --suite (95 groups)
RESULT ok=1 groups=95 errors=0 best=496 best_group=Stab(496) reference=496 exceeded=0 seconds=1868.4
# tools/orbit_mis on Stab_Co0(496) / Stab_Co0(488)
RESULT ok=1 groups=1 errors=0 best=496 best_group=Stab_Co0_496_deep reference=496 exceeded=0 seconds=424.4
RESULT ok=1 groups=1 errors=0 best=496 best_group=Stab_Co0_496_exact reference=496 exceeded=0 seconds=1861.7
RESULT ok=1 groups=1 errors=0 best=496 best_group=Stab_Co0_496_export reference=496 exceeded=0 seconds=10.8
RESULT ok=1 groups=1 errors=0 best=488 best_group=Stab_Co0_488_deep reference=488 exceeded=0 seconds=184.2
# python/tools/orbit_milp.py (HiGHS, 1800 s)
RESULT ok=1 usable=3240 edges=456480 cliques=43620 best=256 dual_bound=1476.000 upper=1476 optimal=0 reference=496 seconds=1857.0
# tests/test_orbits.cpp
RESULT ok=1 failures=0 ms=2441
```

---

## 7. The 64 certified maximal 496-sets

The plateau of the record was, up to this point, a search artefact: the four (12,12) atoms and the
two (80,80) atoms were known, the 2⁶ = 64 combinations had been walked by the GPU engine, and the 16
sets of the 4-cube had been certified maximal by the exhaustive `k ≤ 12` search. The **48 sets
involving an (80,80) atom were known to be maximal only through the GPU engine** — not certified by
an independent verifier. That gap is now closed.

> **None of the 64 plateau 496-sets extends. Every one of the 64 has 0 free vertices, no vertex of
> tightness 1, 2 or 3, and minimum tightness exactly 4 (80 tightness-4 vertices each), certified by
> two independent CPU verifiers in exact arithmetic over all 196560 minimal vectors.** No 497 exists
> anywhere on the plateau by single-vector extension, and no `(k, k+1)`-swap with `k ≤ 3` exists at
> any of the 64 sets. `runs/found/` stays empty.

### 7.1 Deliverables

| File | Content |
|---|---|
| `data/S496_family/S_00.txt` … `S_63.txt` | The 64 sets in the standard text format (24 integers per row, norm-32 scaling), rows sorted by canonical Leech index, each with a provenance header naming the applied atom mask, the sha256 of both inputs, and the set's invariants. |
| `data/S496_family/INDEX.md` | Per-set table: atoms applied, overlap with `S_00`, sha256 of the file, min tightness, free-vertex count, tightness-1/2/3 counts, fingerprint class (0–4), isometry class (0–7), full tightness histogram. |
| `python/tools/materialize_64.py` | Deterministic regenerator: `data/S496.txt` + `runs/ls_search/plateau_atoms_496.json` → the whole directory, byte for byte (verified: two runs diff clean). Sanity-checks the atoms, verifies every set, computes all tightness vectors (float64 GEMM — exact for inner products ≤ 32), checks atom applicability on every set, re-derives the fingerprint classes. Exits non-zero on any check failure or any free vertex. |
| `python/tools/check_64.py` | Standalone checker, numpy + `python/kiss_ref` only (the minimal vectors are regenerated; nothing from the C++ or from the materialiser is read — an independent arithmetic path, int64/int32 matmul). Per file: 496 rows, all Leech minimal vectors (lookup plus arithmetic membership cross-check), distinct, off-diagonal Gram ≤ 8, antipodal; then the full tightness vector over all 196560 vertices → free vertices, min tightness, t1/t2/t3, histogram. A free vertex is reported loudly and exits 2 (it would mean a 497). |

```
$ .venv/bin/python python/tools/materialize_64.py            # regenerate data/S496_family/ (26 s)
$ .venv/bin/python python/tools/check_64.py data/S496_family # independent check (89 s)
$ ls data/S496_family/S_*.txt | xargs -P 8 -I{} .venv/bin/python python/verify_S.py {}
```

### 7.2 Atom reconstruction and sanity checks

`runs/ls_search/plateau_atoms_496.json` (sha256 `d565cd97f1b8…`) holds six atoms with fields `size`,
`remove_S_pos`, `remove_vertices`, `add_vertices` (canonical indices; `remove_S_pos` indexes the
ascending index list of `data/S496.txt`). All checks pass:

* sizes exactly (12, 12, 12, 12, 80, 80);
* `remove_vertices` equals the base set's ascending index list at `remove_S_pos` for every atom;
* all six remove sets pairwise disjoint, all six add sets pairwise disjoint, no remove set meets any
  add set, no add set meets `S`;
* every remove set and every add set is closed under negation;
* the 208 added vectors (of all six atoms together) are mutually non-adjacent — so every union of
  atoms is applicable simultaneously, and the 6-cube is well defined;
* each atom is a valid plateau move on `S`: the union of the added vectors' conflict sets is exactly
  the atom's removed set.

### 7.3 Generation and certification

For each mask `T ⊆ {0..5}`: `S_T = (S ∖ ⋃ removes) ∪ (⋃ adds)` over the atoms in `T`. Every `S_T`
verified: 496 rows, all minimal vectors, distinct, Gram off-diagonal histogram exactly
`{−32:248, −8:37504, 0:47504, 8:37504}` (the record's), antipodal, and
`|S_T ∩ S| = 496 − 12·#(12-atoms in T) − 80·#(80-atoms in T)` — the cube prediction, 64/64.

The two-verifier rule is honoured here with **three** independent implementations: `materialize_64.py`
(float64 GEMM tightness), `check_64.py` (integer matmul tightness over regenerated minimal vectors),
and `python/verify_S.py` as a subprocess on each of the 64 files. Per-file statistics from the first
two are **identical on all 64 files** (free, min tight, t1/t2/t3, full histogram); no disagreement
anywhere.

```
$ .venv/bin/python python/tools/materialize_64.py
atoms     : 6 atoms (12,12,12,12,80,80), pairwise disjoint, negation-closed, valid plateau moves on S; ...
...
fp check  : 5 distinct tightness histograms, sizes 16/16/16/8/8, partition identical to the equivalence run's fingerprint classes
RESULT ok=1 sets=64 free_total=0 min_tight_all=4 t123_total=0 fp_classes=5 sizes=16/16/16/8/8 out=data/S496_family

$ .venv/bin/python python/tools/check_64.py data/S496_family
summary   : 64/64 files pass; free vertices total 0; min-tightness values [(4, 64)]; distinct tightness histograms 5
RESULT ok=1 files=64 pass=64 fail=0 free_total=0 min_tight=4 distinct_hists=5

$ ls data/S496_family/S_*.txt | xargs -P 8 -I{} .venv/bin/python python/verify_S.py {} \
    | grep -c 'RESULT ok=1 size=496 antipodal=1 max_offdiag=8 gram=-32:248,-8:37504,0:47504,8:37504'
64        # and zero 'RESULT ok=0' lines
```

Per set, over all 196560 vertices:

| quantity | value, at **all 64 sets** |
|---|---|
| free vertices (tightness 0 outside S) | **0** |
| tightness-1, -2, -3 vertices | **0, 0, 0** |
| minimum tightness outside S | **4** (exactly 80 tightness-4 vertices) |

So (a) no plateau set extends to 497 by adding any single vector — in particular the 48 sets touching
an (80,80) atom, whose maximality had only the search engine behind it, are now certified maximal by
independent exact arithmetic; and (b) no `(k, k+1)`-swap with `k ≤ 3` exists at any of the 64, so any
local improvement anywhere on the plateau must remove ≥ 4 vectors.

**Fingerprint classes.** The 64 tightness histograms take exactly **5** values, in class sizes
**16/16/16/8/8**, and the partition — re-derived here purely from the computed histograms — is
*identical*, member for member, to the classes of §4.8 and §8 (asserted in-run; a mismatch fails the
tool). Class 0 (the record's histogram, 16 sets) consists of the masks with
`T ∩ {0,1,2,3} ∈ {0000, 0101, 1010, 1111}` — insensitive to the two (80,80) atoms, i.e.
`fp(T) = fp(T^15) = fp(T^48)`. Min-tightness distribution over the 64 sets: 4 at all 64.

### 7.4 The 6-cube structure, checked move by move

For every `S_T` and every atom `a`, checked numerically (inner products of the candidate added
vectors against all of `S_T`):

* `a ∉ T`: the **forward** move applies (removes ⊆ `S_T`, adds disjoint from `S_T`, added conflicts
  lie exactly in the removed set) — 64 × per-atom, all pass;
* `a ∈ T`: the forward move does **not** apply (its remove set is absent) and the **reverse** move
  (swapping the remove/add roles) applies — all pass.

Exactly as the 6-cube predicts: from any `S_T` the six toggles `⊕a` are available and lead to
`S_{T⊕a}`; 384/384 applicability checks pass, no exception.

### 7.5 Cross-checks against independent artefacts

* **`S_48` (both (80,80) atoms) equals `runs/ls_search/S496_alt_overlap336.txt`** as a set of
  canonical indices — exact match.
* **The 64 sets are exactly the 64 distinct best sets of the 20,000-iteration antipodal GPU run**
  (`runs/ls_search/s496-a_long_bestsets.txt`): both families have 64 distinct index sets and they are
  identical as families. The CPU reconstruction from the atoms and the GPU walk agree set for set.
* **Determinism:** a second `materialize_64.py` run into a scratch directory produced a
  byte-identical tree (`diff -r` clean), so the committed directory is reproducible from its two
  inputs.
* **Overlaps** with `S_00` are 496/484/472/460/448 (12-atom combinations), 416…368 (one 80-atom),
  336…288 (both).
* **Isometry classes** in `INDEX.md` are carried from the equivalence computation of §8
  (`runs/plateau_equiv/isometry_classes.json`, 8 classes of 8, embedded as constants with citation);
  they are provenance, not recomputed here. The fingerprint partition, by contrast, is re-derived
  from the computed histograms on every run and compared.

Two further notes: `check_64.py` checks Gram off-diagonal ≤ 8 (the same criterion as `verify_S.py` /
`verify.h`), which is strictly stronger than "≠ 16" and is the correct independence condition in this
scaling; and both tools would have failed loudly (exit 2,
`*** … POTENTIAL NEW RECORD ***`) had any free vertex existed — that branch never fired.

**What this settles.** The plateau of the record is now a fully materialised, fully certified object:
64 committed certificate files, each independently verified as a 496-point kissing configuration in
Λ₂₄, each proven non-extendable and `(k ≤ 3)`-swap-rigid by exact arithmetic. Together with §8
(pairwise Co₀-inequivalent) this gives a clean citable artefact: 64 inequivalent, individually
certified maximal 496s, 5 fingerprint classes (16/16/16/8/8), 8 isometry classes of 8.

---

## 8. Are the 64 plateau 496s Co₀-equivalent? No — 64 distinct classes

The question: for each of the 64 plateau sets `S_T` (`T` a subset of the six atoms, `S_0 = S` the
record of `data/S496.txt`), is there `g ∈ Co₀` with `g·S = S_T`?

> **None of the 63 non-trivial plateau sets is a Co₀-image of the 496, and no two of the 64 are
> Co₀-equivalent to each other: the 64 sets fall into 64 distinct Co₀-classes.** Fifty-six of them
> are not even *isometric* to the record; the remaining seven are congruent copies of it in R²⁴ that
> no automorphism of the Leech lattice can reach. The ILS plateau of the 496 is therefore **not a
> group orbit**, and the earlier conjecture that the alternative 496
> `runs/ls_search/S496_alt_overlap336.txt` is "plausibly a Co₀-image" of the record is **refuted**.

| File | Content |
|---|---|
| `python/tools/plateau_equivalence.py` | The whole computation, exact, `.venv/bin/python` (numpy + sympy, `kiss_ref.leech` for the 196560 minimal vectors; independent of the C++). |
| `runs/plateau_equiv/log.txt` | Full log of the run reported here. |
| `runs/plateau_equiv/results.json` | Machine-readable: WL invariants, Co₀ fingerprints (per set), per-T verdicts, pair tests, class partition, stabiliser cross-checks. |
| `runs/plateau_equiv/isometry_closure.py` | Follow-up that closes the isometry classification (73 cross-class tests); writes `isometry_classes.json`, `isometry_closure.log`. |

```
$ .venv/bin/python python/tools/plateau_equivalence.py --out runs/plateau_equiv --jobs 12 --stab 15,16,48
$ .venv/bin/python runs/plateau_equiv/isometry_closure.py --jobs 12
```

(Options of the main tool: `--set --atoms --data --out --auts-out --stab --jobs --quick --no-orbits
--no-auts-out --rep-pairs`.)

### 8.1 Method: a complete decision procedure

Everything is exact integer arithmetic on the 196560 minimal vectors of Λ₂₄ in the norm-32 scaling
(`data/leech_min.i8`, checked against `kiss_ref.leech.leech_min_vectors()` on load).

**The reduction.** If `g ∈ Co₀` has `g·S = S′` then `g` restricts to a bijection `f : S → S′`
preserving every inner product, i.e. an **isomorphism of the complete graphs on 496 vertices coloured
by `⟨x,y⟩ ∈ {−32, −8, 0, 8}`**. Conversely `S` spans R²⁴ (verified: every one of the 64 sets has rank
24), so `f` determines **at most one** orthogonal map `g_f`, and `S′ = g·S` for some `g ∈ Co₀` **iff**
some such `f` extends to a map that preserves Λ. The procedure is therefore complete in both
directions:

1. **Enumerate all Gram-graph isomorphisms** `f : S → S_T`. Colour refinement (1-WL) is run on the
   *disjoint union* of the two coloured graphs (cross edges get their own colour, so the class labels
   are shared and comparable), followed by individualisation–refinement: a class whose two sides have
   different sizes prunes the branch; a colouring in which every class is one `S`-vertex plus one
   `S_T`-vertex yields a candidate `f`, verified by comparing the two Gram matrices exactly. Since
   the cross colour is uniform, the map (`f` on `S`, `f⁻¹` on `S_T`) is an automorphism of the
   coloured union whenever `f` is an isomorphism, so WL colours cannot separate a vertex from its
   image: **no branch containing a genuine isomorphism is ever pruned**, and the enumeration is
   exhaustive. Taking `S_T = S` reproduces `Aut(Gram(S))`.
2. **Extend and test.** For each `f`, `8·g_f` is obtained by least squares on all 496 rows and
   rounded; the result is accepted only after the exact integer checks `S·(8g)ᵀ ≡ 0 (mod 8)`,
   `S·gᵀ = S_T[f]` and `(8g)(8g)ᵀ = 64·I`. Then `g` is applied to **all 196560 minimal vectors**:
   `g ∈ Co₀` iff every image is again a minimal vector and the map is a bijection. The extending `f`
   then form a coset of `Stab_{Co₀}(S)`, so their number is 8 or 0 — checked.

**Pre-filters, used only where provably sound.** The 1-WL invariant of each Gram graph on its own
(stable class sizes plus the class quotient) is a necessary condition for **isometry**, hence for
Co₀-equivalence. The Co₀-invariant fingerprint is: the histogram of `tight[v]` over `v` outside `S`;
the histogram of the **full profile** `(#{⟨v,s⟩=16}, #{⟨v,s⟩=8})` (a strict refinement — `S` is
antipodal, so the counts at −16, −8 mirror those at 16, 8 and the profile is determined by that
pair); the Gram histogram of `S`; and the Gram histogram of the minimum-tightness vertices. Every
entry is defined purely by inner products between minimal vectors, so `g ∈ Co₀` carries it unchanged.

**Not used as a Co₀ filter: the shape counts** (octad / (∓3, ±1²³) / (±4, ±4)). Those are invariants
of the *monomial* subgroup 2¹²:M24 only — Conway's ξ mixes the three shapes — so they can certify
monomial-inequivalence but never Co₀-inequivalence. The run demonstrates this directly: `ξ·S` has the
same Co₀ fingerprint as `S` (it must: it *is* a Co₀-image) but shape counts 242/248/6 instead of
`S`'s 258/232/6. (An earlier draft did include the shape counts, which splits the 64 into 15 classes;
10 of those splits are spurious for Co₀. The 15-way split is reported for what it is: a lower bound
for the monomial subgroup, not for Co₀.)

**Positive controls.** Before any verdict is drawn, the pipeline is required to *find* equivalences
that are known to exist: `g·S` for `g ∈ {ξ, ξ², γ (an M24 coordinate permutation), ξ·γ, −I}`. All
five are recognised — 64 isomorphisms each, exactly 8 extending to Co₀ — including `ξ·S` and `ξγ·S`,
which meet `S` in only 8 and 4 vectors respectively. A "not equivalent" verdict is therefore not an
artefact of a search that fails to look far enough.

### 8.2 The "64 = 64" coincidence is a coincidence

`|Aut(Gram(S))| = 64` and `|Stab_{Co₀}(S)| = 8` are both reproduced here (`n_iso 64, n_ext 8`, 28
stable WL classes of sizes {4:6, 8:3, 16:10, 32:9}); the plateau also has 64 sets. The two 64s are
**unrelated**:

* All 64 Gram automorphisms map **every atom `R_k` onto itself** — the induced permutation of the six
  atoms is the identity for all 64, and each atom is a union of `Aut(Gram(S))`-orbits (Aut has 40
  orbits on `S`, of sizes 4 and 16; the 8-element Co₀ stabiliser has 124 orbits, all of size 4).
* Consequently the 56 non-Leech Gram automorphisms, applied to `S_T`, give back **`S_T` itself** for
  every one of the 63 non-trivial `T` (the run checks all 56 × 63 images land in the minimal vectors
  and identifies the resulting set: always the same `T`). So `Aut(Gram(S))` acts **trivially** on the
  64 plateau sets; it cannot be the source of the 2⁶, and nothing in it moves `S` to another plateau
  set.
* The 2⁶ = 64 of the plateau comes from six independent commuting plateau moves; the 2⁶ = 64 of
  `Aut(Gram(S))` is the order of an elementary abelian 2-group of orthogonal maps, only 2³ of which
  preserve Λ. The equality is numerical only.

The three non-trivial involutions of `Stab_{Co₀}(S)` (the octad-sign-change class) fix 208, 144 and
144 members of `S`; on the atoms they fix 4 of the 12 vectors of each (12,12) atom and 16 or 32 of the
80 vectors of each (80,80) atom — so the stabiliser preserves each atom but acts non-trivially inside
it, which again is why it never carries `S` to a different `S_T`.

### 8.3 Cross-check against the independent C++ plateau walk

`plateau_equivalence.py` reads the five saved representatives of the plateau walk (§3.3) and locates
each one among the 64:

| plateau-walk file | overlap with S | equals | fingerprint class |
|---|---|---|---|
| `node_00000.txt` | 496 | `S_0` = S | fp#0 |
| `node_00001.txt` | 484 | `S_8` (atom 3) | fp#2 |
| `node_00002.txt` | 484 | `S_1` (atom 0) | fp#1 |
| `node_00005.txt` | 472 | `S_9` (atoms 0, 3) | fp#4 |
| `node_00006.txt` | 472 | `S_12` (atoms 2, 3) | fp#3 |

* The 16 sets with `T ⊆ {atoms 0,1,2,3}` carry exactly **5** fingerprints, in sizes 4+4+4+2+2 — the
  walk's count and split — and *the five tightness histograms agree string for string* with
  `runs/plateau/s496_k12/summary.txt` (asserted in the run). Two independently written
  implementations agree on the invariant.
* Over the **whole** 6-cube there are still exactly **5** Co₀-invariant fingerprints, in sizes
  16, 16, 16, 8, 8. Their structure is completely explicit: `fp(T)` depends only on the pair
  `{T ∩ {0,1,2,3}, its complement}` and is *totally insensitive to the two (80,80) atoms* —

  ```
  fp(T) = fp(T ^ 15) = fp(T ^ 48) = fp(T ^ 63)   for all 64 T
  ```

  which extends the walk's "each class is a union of complementary pairs" to the two big moves.
* **This corrects the earlier claim** that the 64 sets all share the 496's Gram **and** tightness
  histogram. The Gram histogram (−32:248, −8:37504, 0:47504, 8:37504) really is constant over all 64,
  and so are the minimum tightness (4), the number of tightness-4 vertices (80) and their Gram
  histogram — but the **tightness histogram is not**: it takes 5 different values. The earlier claim
  had checked only `S496_alt_overlap336.txt` = `S_48`, which does happen to lie in `S`'s own
  fingerprint class (both (80,80) moves leave the fingerprint alone).
* The **shape counts are not constant either**: 258/232/6 when neither (80,80) move is applied,
  274/216/6 for atom 4 alone, 242/248/6 for atom 5 alone, and back to 258/232/6 for both. Each
  (80,80) move trades 16 vectors between the octad and the (∓3, ±1²³) shape, in opposite directions.
  This is a monomial invariant only, so it is reported but never used to certify Co₀-inequivalence.

### 8.4 Result 1 — none of the 63 non-trivial plateau sets is a Co₀-image of the 496

```
summary: 0 of 63 non-trivial S_T are Co_0-equivalent to S; 7 are isometric to S (Gram graphs isomorphic)
```

For **every** `T ≠ 0` the answer is no, and for 56 of the 63 it is no for the strongest possible
reason — there is no inner-product-preserving bijection `S → S_T` at all, so the two 496-sets are not
even **isometric**, let alone related by a lattice automorphism. In particular all **six generators**:

| T | atoms | move | overlap with S | Gram isomorphisms | extend to Co₀ | verdict |
|---|---|---|---|---|---|---|
| 1 | {0} | (12,12) | 484 | **0** | 0 | not even isometric |
| 2 | {1} | (12,12) | 484 | **0** | 0 | not even isometric |
| 4 | {2} | (12,12) | 484 | **0** | 0 | not even isometric |
| 8 | {3} | (12,12) | 484 | **0** | 0 | not even isometric |
| 16 | {4} | (80,80) | 416 | **0** | 0 | not even isometric |
| 32 | {5} | (80,80) | 416 | **0** | 0 | not even isometric |
| 48 | {4,5} | both (80,80) | 336 | **0** | 0 | not even isometric |
| 63 | all six | | 288 | **0** | 0 | not even isometric |

The 32 masks with `|T ∩ {0,1,2,3}|` **odd** are refuted in ≈ 1 s each: joint colour refinement splits
the two sides into 56 classes with unequal side sizes, so no isomorphism can exist. The 31 masks with
even `|T ∩ {0,1,2,3}|` survive refinement (28 balanced classes, the same profile as `S`) and need the
full individualisation search, 39–112 s each; 24 of them terminate with **zero** isomorphisms and 7
with 64.

**The 7 isometric ones.** For `T ∈ {15, 22, 25, 35, 44, 53, 58}` the search finds exactly 64 Gram
isomorphisms `S → S_T` — the same number as `|Aut(Gram(S))|`, as it must be, since the isomorphisms
form a coset of `Aut(Gram(S))` — and **not one of the 64 extends to a Leech automorphism** (all 64
give an orthogonal map with `8g` integral and `(8g)(8g)ᵀ = 64 I`, but each moves some minimal vector
off the lattice). Together with `T = 0` these masks form an **elementary abelian subgroup of order 8**
of the 6-cube,

```
{0, 15, 22, 25, 35, 44, 53, 58} = ⟨15, 22, 35⟩ ≤ (Z/2)⁶
```

(15 ⊕ 22 = 25, 15 ⊕ 35 = 44, 22 ⊕ 35 = 53, 25 ⊕ 35 = 58), so the isometry class of `S` inside the
plateau is a subgroup, not just a set. There are therefore eight **congruent** copies of the 496
configuration sitting in the Leech lattice at overlaps 496, 448, 392, 392, 392, 392, 312, 312 —
pairwise related by orthogonal maps of R²⁴, none of which preserves Λ. Four of the seven
(`T = 22, 25, 35, 44`) even carry a **different Co₀-invariant tightness histogram** from `S`, which is
an independent second proof that they are not Co₀-images: isometric, provably inequivalent, by two
unrelated arguments.

### 8.5 Result 2 — the 64 are pairwise Co₀-inequivalent

```
Co_0-inequivalent classes among the 64 tested sets: 64; sizes [1, 1, 1, ... , 1]
RESULT ok=1 sets=64 equivalent_to_S=1 classes=64 iso_classes=18 aut_gram=64 stab=8 seconds=2848.2
```

(`equivalent_to_S=1` counts `S` itself. `iso_classes=18` is *not* the isometry-class count: it is the
refinement this run can see, because it only tests pairs inside a fingerprint class; the true count is
8, established by the closure run in §8.7.)

The Co₀-classification of all 64 is complete, and it is the finest possible: **every one of the 64
plateau 496s is its own Co₀-class.** The argument has two exhaustive halves:

1. *Across fingerprints.* The 64 fall into **5** Co₀-invariant fingerprint classes, of sizes
   16, 16, 16, 8, 8. Sets with different fingerprints cannot be Co₀-equivalent, full stop.
2. *Inside a fingerprint class.* Every one of the 401 same-fingerprint pairs — 120 + 120 + 120 + 28 +
   28, minus the 15 already covered by the `S`-versus-`S_T` pass — was decided by the full isomorphism
   search plus the Leech test. **Not one produced a Co₀ element**; 77 of them turned out
   isometric-but-not-equivalent and the remaining 324 have no Gram isomorphism at all.

This refines the plateau walk's lower bound as far as it can go: the walk established ≥ 5
inequivalent 496s among the 16 sets of the 4-cube (by the tightness fingerprint, a Co₀-invariant);
here the 4-cube's 16 sets are 16 distinct classes and the whole 6-cube's 64 sets are 64. The 5-way
fingerprint split is exactly recovered — it is the coarse invariant, and the isomorphism search
resolves everything it leaves open. No pair with different fingerprints ever produced a Co₀ element
(a consistency check the run makes explicitly:
`contradictions with the Co_0-invariant fingerprints: 0`).

### 8.6 Result 3 — stabiliser cross-checks

`Stab_{Co₀}(S_T)` was recomputed from scratch (full `Aut(Gram(S_T))` enumeration plus the Leech test)
for `T = 15` (all four (12,12) moves), `T = 16` (one (80,80) move) and `T = 48` (both (80,80) moves):

| T | \|Aut(Gram(S_T))\| | WL classes | \|Stab_{Co₀}(S_T)\| | structure | orbits on S_T | same matrices as Stab_{Co₀}(S)? |
|---|---|---|---|---|---|---|
| 0 (= S) | 64 | 28 | **8** | 2³ elementary abelian | 124 × size 4 | — |
| 15 | 64 | 28 | **8** | 2³ elementary abelian | 124 × size 4 | **yes** |
| 16 | 64 | 28 | **8** | 2³ elementary abelian | 124 × size 4 | **yes** |
| 48 | 64 | 28 | **8** | 2³ elementary abelian | 124 × size 4 | **yes** |

Each of the four sets has the *same* stabiliser — not merely an isomorphic or a conjugate one, but
literally the same eight matrices (equal sha256 of `8g`). That is forced by the structure above:
`Stab_{Co₀}(S)` preserves every atom setwise, so it preserves every union `S_T`. It is also the
reason the usual consistency check `Stab(S_T) = g_T Stab(S) g_T⁻¹` is vacuous here (reported as
`None`): there is no `g_T`, because no `g_T` exists. So the 2³ is a common stabiliser of all 64 sets,
and each of the 64 Co₀-classes has the same stabiliser order 8 — a further sign that the 64 are
"equally symmetric" copies without being equivalent.

### 8.7 Result 4 — the isometry classification: 8 classes of 8

The Co₀ answer does not need the isometry classes, but they came out clean, so
`runs/plateau_equiv/isometry_closure.py` finished the job. The main run decides every pair inside one
fingerprint class and every pair `(S, S_T)`; the closure adds the 73 cross-class pairs that share a
1-WL invariant (the other 80 pairs of representatives differ in it and are therefore provably
non-isometric), and isometry being transitive, that decides **every** one of the 2016 pairs.

```
ISOMETRY classes among the 64: 8; sizes [8, 8, 8, 8, 8, 8, 8, 8]
RESULT ok=1 iso_classes=8 tests=73 seconds=207.0
```

| class rep | the 8 sets | T ⊕ rep |
|---|---|---|
| S_0 | 0, 15, 22, 25, 35, 44, 53, 58 | K₀ |
| S_3 | 3, 12, 21, 26, 32, 47, 54, 57 | K₀ |
| S_5 | 5, 10, 19, 28, 38, 41, 48, 63 | K₀ |
| S_6 | 6, 9, 16, 31, 37, 42, 51, 60 | K₀ |
| S_1 | 1, 14, 17, 30, 33, 46, 49, 62 | K₁ |
| S_2 | 2, 13, 18, 29, 34, 45, 50, 61 | K₁ |
| S_4 | 4, 11, 20, 27, 36, 43, 52, 59 | K₁ |
| S_7 | 7, 8, 23, 24, 39, 40, 55, 56 | K₁ |

with two **different** elementary abelian subgroups of order 8 of the 6-cube,

```
K₀ = {0, 15, 22, 25, 35, 44, 53, 58} = ⟨15, 22, 35⟩      (the four classes with |T ∩ {0,1,2,3}| even)
K₁ = {0, 15, 16, 31, 32, 47, 48, 63} = ⟨15, 16, 32⟩      (the four classes with |T ∩ {0,1,2,3}| odd)
```

So each isometry class is a coset — but of `K₀` on one half of the plateau and of `K₁` on the other,
and `K₀ ∪ K₁` is not a group. The isometry relation is genuinely **not** invariant under translation
of the 6-cube: `S_1 ≅ S_49` (difference 48 = both (80,80) moves) while `S_0 ≇ S_48`. Everything the
two subgroups share is `⟨15⟩`, and that one is explained:

* **The "flip all four (12,12) moves" isometry is global.** All 64 orthogonal maps `g` with
  `g·S = S_15` also satisfy `g·S_T = S_{T ⊕ 15}` for **every one of the 64 T** (checked exactly,
  `64 of 64`). So `⊕15` is realised by a single orthogonal map of R²⁴ acting on the whole plateau —
  it just does not preserve Λ. (This is the honest version of the plateau walk's observation that
  applying all four moves preserves the fingerprint.)
* **The other generators are not.** Of the 64 maps realising `S → S_22`, **none** translates the cube
  by `⊕22`; the same for `⊕35`. Those isometries are local to their coset.

Numerically the plateau is therefore a very tidy 8 × 8 grid: **8 congruence classes, each containing
8 Leech-inequivalent realisations of one and the same 496-point configuration.** The 5 fingerprint
classes cut across this grid (sizes 16/16/16/8/8). `Aut(Gram(S))` (order 64) preserves every `S_T`,
so it is contained in every `Aut(Gram(S_T))`; for the four `T` checked in full (0, 15, 16, 48) the two
groups coincide and the Co₀ stabiliser is literally the same 2³.

### 8.8 What this means

The ILS plateau of the 496 is **not** an orbit of a group. The six plateau moves are not induced by
lattice automorphisms; the 64 sets are genuinely different 496-point configurations that happen to
share the Gram histogram, the minimum tightness (4), the number of tightness-4 vertices (80), the
maximality and the antipodality of the record. The local search wanders between inequivalent optima,
not between images of one optimum.

Nothing here contradicts the numerical-optimality intuition for the 496 — every one of the 64 is
again a 496 and none reaches 497 (§3, §4, §7) — but "the" record 496 is **not unique up to the
symmetry of the ambient lattice**: there are at least **64** pairwise Co₀-inequivalent 496-point
kissing configurations in Λ₂₄ reachable from it by plateau moves, and they realise only **8**
congruence classes, so each shape occurs in 8 Leech-inequivalent positions, and seven sets are
congruent to the record itself without being lattice-equivalent to it. Congruence of the
configuration does not imply equivalence inside the lattice, so "the 496" is well defined only up to
a choice among 8 — or, counting all shapes on the plateau, 64 — inequivalent representatives.

Two procedural notes. `data/group/plateau_auts/` was reserved for the Co₀ elements realising the six
plateau moves; since no plateau move is realised by a Co₀ element, there is nothing to write, and the
tool creates the directory only when all six generators turn out to be Co₀-images. And three runs
were made, all agreeing: a first full run was stopped in its orbit phase because that phase used an
O(64²) all-pairs scan (≈ 4 h projected); a second added a shortcut that tries to prove the isometry
classes are cosets of one subgroup, and the shortcut **failed** — correctly, since the classes are
cosets of *two* different subgroups; the third took the bounded fallback (exhaustive inside each
fingerprint class, 401 tests, which is all the Co₀ question needs, plus the separate 73-test closure
for the isometry classes). All three produced identical numbers everywhere they overlap:
`Aut(Gram(S)) = 64`, `Stab = 8`, the same 63 per-`T` verdicts, the same 7 isometric masks, and the
same pairwise verdicts. `runs/plateau_equiv/log.txt` is the third run. **The failed shortcut is
itself a result**: "is the plateau move `⊕k` realised by an orthogonal map that acts on the whole
6-cube?" is answered yes for `k = 15` and no for `k = 22` and `k = 35`, and that asymmetry is exactly
what makes the isometry partition use two different subgroups. (The run shared the machine with other
jobs at load average 15–25, so the per-test timings of 39–112 s for a full individualisation search
on 2 × 496 vertices are 2–3× the idle-machine figures; correctness is unaffected.)

---

## 9. The Turyn slice: 496 ≤ opt ≤ 720 over 2³-invariant sets

§6.6 left one question genuinely open: **is 496 the maximum weight of an independent union of
2³-orbits?** — where the 2³ is `Stab_{Co₀}(496)`, the sign group of a Turyn `√2·E8³` decomposition of
Λ₂₄. Equivalently, in the channel/rook model of `04-structure-of-the-496.md`: is 124 the maximum
independent duad set, allowing also the 360 weight-2 monad orbits? This section is the dedicated
attack, with the full exact-solver arsenal. All computations are in
`python/tools/channel_opt.py` (subcommands named below); artefacts in `runs/a2/`.

Labels used below: **VERIFIED** = exact computation with assertions and a `RESULT` line;
**PROVED (solver)** = exact optimisation run to proven optimality or infeasibility by the named
solver.

> **Status: OPEN, but fully instrumented. Certified: 496 ≤ opt ≤ 720**, both ends by machine proof
> (the 720 is the per-channel matching LP, reproduced independently by HiGHS and CP-SAT). More than
> 10 hours of exact anytime search never found a 497 — heuristic evidence that the answer is 496.
> The gap did not close, because every relaxation tried is blind below the flat profile `m ≡ 4`. Nine
> kissat refutation instances that would each certify a strict improvement — one of them the full
> theorem, with a DRAT proof — were **left running** at the end of the session; see §9.7.

### 9.1 The model, rebuilt from scratch

**VERIFIED** (`channel_opt.py build`, 3.4 s, every claim an `assert`):

* Orbits of 2³ = {±1}³ on the 196560 minimal vectors: 23,040 of size 8 (every one self-conflicting —
  it contains an internal dot +16), 2880 of size 4 (duads `{±x, ±y}`, `x ⊥ y`), 360 of size 2 (monads
  `{±m}`). Usable orbits: 3240, total weight `2880·4 + 360·2 = 12,240` = all duads plus all monads.
* Orbit-level conflict = any cross dot of absolute value 16 (negation-closure makes −16 and +16
  equivalent at orbit level, so this is *exactly* the vector-level 60°-freeness of the union).
  Edges: **456,480**, equal to the reduced-graph count of §6.6.
* Duads = 3 pairs × 15 channels × 64 rook cells (8×8); monads = 3 blocks × 120. The 15 channels are
  the nonzero points of a 4-dimensional GF(2)-space, i.e. PG(3,2). The reference `S` is 124 orbits of
  weight 496, independent, with a channel table identical to the structural analysis's.

**Edge census** (VERIFIED, `census.json`):

| class | count | meaning |
|---|---|---|
| rook (same pair, same channel) | 20,160 | shared row or column line |
| xchan (same pair, different channel) | 322,560 | exactly 1024 per (pair, chan<chan′), 16 per duad per foreign channel |
| xpair_same_chan | 23,040 | shared line in the common block |
| xpair_xchan | **0** | cross-pair, cross-channel conflicts do not exist |
| monad–duad | 80,640 | factor through 14 lines per monad (2 in each of 7 channels) |
| monad–monad | 10,080 | within-block only, 56-regular |

**The whole conflict graph factors through block-level line geometry** (**VERIFIED**, `structure`,
0.3 s, `runs/a2/lines.npz`): each (channel, block) is a frame of 8 orthogonal norm-64 lines
(15 × 3 × 8 = 360 lines); a duad is a pair of same-channel lines in two blocks; and with `H_b` = the
60°-graph on block `b`'s 120 lines (`|dot| = 32`; cross-channel line dots are only 0 or ±32, each
line seeing exactly 4 of the 8 lines of every foreign channel):

* same channel: conflict ⇔ **shared line** (360 line-16-cliques cover rook + xpair);
* same pair, different channels: conflict ⇔ **`H_j`(rows) AND `H_k`(cols)** (both blocks at 60°);
* different pair, different channel: never;
* monad–duad: conflict ⇔ `|m·u| = 32` (45°) for the duad's line `u` in the monad's block;
* monad–monad: same block and `|m·m′| = 16` (60°).

Derivation (proved by exhaustive check over all pairs): cross dots of duads
`(x,y) = ((u±w)/2), (x′,y′)` are `(±u·u′ ± w·w′)/4` with block dots in `{0, ±16, ±32, ±64}`;
`|a ± b| = 64` forces `(|a|,|b|) ∈ {(64,0), (0,64), (32,32)}`.

A bonus fact (**VERIFIED**, spectra computed exactly): per block, both the 120 channel-lines under
`H_b` *and* the 120 monad-lines under 60°-adjacency are strongly regular graphs **srg(120, 56, 28,
24)** with spectrum `{56¹, 8³⁵, (−4)⁸⁴}` — the E8 root-line graph. Its ratio bound is
`120·4/(56+4) = 8` exactly, which is why "at most 8 monads per block" and "frames are maximum
orthogonal sets" are tight.

**Symmetry of the reduced model** (**VERIFIED**, pynauty on the kind-coloured conflict graph):
`|Aut| = 31,708,938,240`, with exactly **2 vertex orbits** (all 2880 duads equivalent, all 360 monads
equivalent). Every generator respects the channel and pair partitions; the induced action on the 15
channels is the **full GL(4,2)** (order 20,160) and on the 3 pairs the full `S₃`. So all PG(3,2)
lines/planes of channels are equivalent, and channel-subset questions reduce to GL(4,2)-orbit
representatives. Moreover (**VERIFIED**, `autprod.log`) the image of Aut on (channels, pairs) is not
merely surjective on each factor but is the full **GL(4,2) × S₃** (order 120,960, by BFS closure of
the pynauty generators' images) — the justification for replicating each proved cut over all 45
(pair-pair, plane) instances and the like.

### 9.2 The cap rule

Within a single pair, the maximum number of duads supported on a channel subset `C` obeys
**max = 8 · (largest cap of PG(3,2) contained in C)** — verified exactly, each cap individually
PROVED by CP-SAT:

* every channel: 8 (rook); every 2-subset: 16 (105/105);
* every PG-line `{c, c′, c⊕c′}`: **16**, not 24 (35 lines × 3 pairs, each proved);
* every non-collinear triple: 24 (420/420);
* 4/5-subsets: 8 spot-checks all equal `8·capmax`;
* every plane (7 channels): **32** = 8·4 (15 planes × 3 pairs, each proved);
* the whole pair (15 channels): **64** = 8·8 — and the maximum caps of PG(3,2) are exactly the affine
  hyperplane-complements of size 8, which is why the single-pair optima concentrate on such an
  8-channel affine set with a full 8-permutation each.

These caps become entailed linear cuts (`runs/a2/chancuts.json`, 150 cuts + 3 pair caps + 3
pair-with-monad caps). They do **not** by themselves force the answer: the "profile IP" over the 45
counts `m_{p,c}` with all proven caps (`m ≤ 8`, lines ≤ 16, planes ≤ 32, pair ≤ 64, channel ≤ 12
across pairs) still reaches 180 at the flat profile `m ≡ 4` — the obstruction to 125+ is genuinely
global, not any local cap. Global (all-pairs) channel triples reach 36 = 3 × 12 even on PG-lines, so
there is no cross-pair analogue of the line cap.

### 9.3 Exact subproblem optima

(`pairs`, CP-SAT; PROVED where marked OPTIMAL.)

| subproblem | vertices | optimum | orbits | status |
|---|---|---|---|---|
| single pair, duads only (each of 3) | 960 | **256** | 64 | OPTIMAL (6–38 s) |
| single pair + its 2 blocks' monads (each of 3) | 1200 | **288** | 80 = 64 + 16 | OPTIMAL |
| two pairs, duads only (each of 3, S₃-equivalent) | 1920 | in **[96, 120] orbits** | 96 found | TIMEOUT (2.5 h CP-SAT with all 117 proven cuts; bound stuck at the per-channel LP 15 × 8) |
| duads only, global | 2880 | in [124, 180] orbits | 124 found | TIMEOUT (5 h CP-SAT, bound 720 = 180 orbits) |
| monads only, global | 360 | **48** | 24 = 3 × 8 | OPTIMAL (frame per block; ratio bound tight) |

The single-pair optimum is **64 orbits = 256** — far above the record's largest pair usage (52 orbits
= 208) — achieved by choosing the 8 channels of an **affine hyperplane-complement of PG(3,2)** and
giving each a full 8-permutation. A single pair plus the monads of its two blocks reaches
**288 = 64 duads + 16 monads** (8 monads = a full frame per block). **The cross-pair interactions are
what pull the global optimum down to 496.**

Smaller channel-subset maxima within one pair (all PROVED): every one of the 105 channel pairs
reaches 16 = 8 + 8; three-channel subsets split by PG(3,2) collinearity — every one of the 35 lines
caps at **16** (= 8 × cap 2) while every one of the 420 non-collinear triples reaches 24 (the cap rule
`8 · capmax`). So the "≤ 64 of 120 possible" per-pair cap is a genuinely high-order (≥ 9 channels)
phenomenon, mirrored by the affine support of the optima.

The two-pair instance is where the hardness lives: the incumbent 96 matches the value of the
structural construction 64 + 32 (one pair full on an affine 8-set at 8 each; the other confined by
the shared-block frames to the complementary plane, where its plane cap is 32), but neither CP-SAT
(2.5 h, 117 proven cuts, `pairs01_exact.log`) nor the LP can certify anything below 120 = 15 × 8 (the
shared-block frame cap per channel; the flat profile 4 + 4 per channel is LP-feasible). A kissat
decision run on "two-pair ≥ 97 orbits" (`sub_pairs2_ge97.cnf`, entailed cuts encoded) is the closure
attempt — UNSAT would pin the two-pair optimum at 96 and certify duads ≤ `(3/2)·96·4 = 576` globally
(each duad lives in 2 of the 3 two-pair subproblems).

### 9.4 Decomposition facts

**Monads live on planes** (**VERIFIED**, `_monad_plane_assignment`): each monad's 14 conflict lines
lie in its own block, 2 in each of 7 channels, and those 7 channels always form a **plane of
PG(3,2)**; each of the 15 planes receives exactly 24 monads (8 per block), and those 24 are mutually
independent. So *all* monad–duad conflicts factor through the plane geometry, and the whole conflict
graph is PG(3,2)-local.

**The exact kill function** (**PROVED**, `monadkill`, 1.2 s, every value OPTIMAL): `u(k)` := the
minimum number of distinct block lines conflicting with some independent `k`-set of one block's
monads = **14, 26, 36, 44, 50, 54, 56, 56** for `k = 1..8` — exactly the pairwise
inclusion–exclusion bound `14k − k(k−1)` (independent same-block monads share 0 or 2 conflict lines,
verified exhaustively), achieved *inside a single plane*: the 8 same-block monads of one plane kill
only that plane's 7 × 8 = 56 lines. Consequence (a valid cut, since the duads touching a block use
pairwise distinct lines of its 120): for every block `b`,
`#duads touching b ≤ 120 − u(#monads in b)`. At full monads (24 = 3 × 8) this forces duads ≤
`(360 − 3·56)/2 = 96` orbits, i.e. weight ≤ 384 + 48 = 432 < 496: **maximum-monad solutions are
provably sub-optimal**, the first certified statement that monads must be scarce at the optimum.

**Sub-optimum ledger** (`subopt`, CP-SAT, exact where OPTIMAL, ×3 by symmetry): two pairs on a
PG-line = **24** (0.6 s, equal to the flat value, so no cut); two pairs on an affine 8-set = **64**
(2.7 s; equals the flat value 8 × 8, so again no profile cut — but it is the two-pair total cap
restricted to the affine support). Two pairs on a plane and three pairs on a plane / an affine set:
see §9.7.

**Plane decomposition**: a duad's channel lies in 7 of the 15 planes, so
`Σ_planes (duads-with-channel-in-π) = 7 × #duads`, and `#duads ≤ ⌊15·Q/7⌋` where `Q` = the plane
subproblem optimum (duads on one plane's 7 channels, all 3 pairs, 1344 orbits; all planes
Aut-equivalent). `Q ≥ 64` (the record restricted to its best plane). Proving `Q = 64` would give
duads ≤ `⌊960/7⌋ = 137` orbits (548), hence weight ≤ 548 + 48 = 596 by naive addition — and ≤ **560**
through the `combine` ILP, which trades the monads against the kill cut instead of adding their
maximum.

### 9.5 The certified state, and the payoff table

The obstruction found by every relaxation is the **flat profile** `m ≡ 4` (every (pair, channel) cell
at 4, no monads): it satisfies the rook caps, the shared-frame caps, every per-pair line/plane/total
cap and the kill cut, and pays 720. Every certified improvement must therefore prove some
*sub*problem optimum strictly below its flat value. `combine` assembles all PROVED caps into a
48-variable CP-SAT profile maximisation whose optimum is a valid upper bound for the whole model (any
independent orbit set induces a feasible profile; every cut is replicated over its full GL(4,2) × S₃
orbit, legitimate by the product-transitivity fact of §9.1).

Certified now (all ingredients PROVED; `combine` OPTIMAL in 0.2 s): **496 ≤ opt ≤ 720**, and at any
profile of weight 720 the monad count is forced to 0 (the kill cut — checked by a separate exact
solve maximising monads subject to weight = 720). The 720 did not move because every flat-breaking
subproblem timed out in CP-SAT with its dual bound stuck exactly at its own per-channel LP, and none
of the kissat refutations finished.

**The profile-cap ledger** (`g(k)` = max duads in ONE pair with every channel ≤ k; incumbents are
witnesses, dual bounds are proven):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| g(k) ≥ (witness) | **15** | 22 | 27 | 36 | 40 | 48 | 56 | **64** |
| g(k) ≤ (proved) | **15** | 30 | 45 | 60 | 62 | 62 | 62 | **64** |

`g(1) = 15` and `g(8) = 64` are exact; the `k = 5..7` dual bounds 62 < 64 (monotonised — the raw
CP-SAT bounds are 62/63/62) are the only non-trivial ones and do not touch the flat profile. Note the
enormous witness-versus-LP gaps at `k = 2..4`: a pair genuinely cannot spread thin, but no relaxation
can see it.

Conditional payoffs, computed by running `combine` with each cap asserted (each line reads: "if the
corresponding pending kissat instance is UNSAT, the certified bound for the 2³-invariant optimum
becomes"):

| pending closure (incumbent, flat value) | certified bound after |
|---|---|
| g(2) = 22 (flat 30) alone | ≤ 630 |
| g(2) = 22 and g(3) = 27 | ≤ 588, incumbent 572 |
| two-pair total = 96 orbits (flat 120) | 582 |
| plane (3 pairs, 7 channels) = 64 orbits (flat 84) | **560** |
| two-pair plane = 48 orbits (flat 56) | 614 |
| plane 64 + two-pair plane 48 (+ pairs2 96 + perfect ≤ 8) | **558** |
| `sat_ge497_cuts.cnf` UNSAT | **496 = opt, with DRAT proof** |

The wall is the same at every scale: each subproblem's LP sits at its per-channel matching cap, the
flat profile is feasible for every local cap, and only combinatorial (SAT-style) refutation moves
below it — which is why the endgame was converted into nine kissat decision instances rather than
more CP-SAT hours.

### 9.6 Perfect channels — a structural conjecture refuted

A channel is *perfect* when it carries 12 orbits, i.e. a perfect matching of its `K₈,₈,₈`. The record
has exactly 3 perfect channels (ids {2,3,7} in the model's channel order), and the structural
analysis had **conjectured an obstruction to a 4th**. CP-SAT feasibility (channel counts forced = 12),
duad-only, exact, organised by GL(4,2)-orbit representatives of channel subsets (legitimate because
the induced channel action is all of GL(4,2)):

* **all 105 channel pairs** and **all 455 channel triples** can be simultaneously perfect — and the
  perfect triple is NOT forced to be the record's frame triple (its {2,3,7} is not even a PG(3,2)
  line: `c₂ ⊕ c₃ = c₆`, `c₂ ⊕ c₃ ⊕ c₇ = c₁₄`);
* **4-subsets: perfect-feasible** (both orbit types) — **there is no obstruction to a 4th perfect
  channel; the conjecture is refuted.** A 4-perfect independent duad family exists in the model. The
  reason the record has only 3 is an optimisation trade-off, not a hard constraint;
* **5-subsets: perfect-feasible** (all 4 orbit types);
* 6-subsets: at least 4 of the 5 orbit types feasible (representative (0,1,2,3,4,5) hit the per-test
  time cap) — **six simultaneously perfect channels exist**;
* 7-subsets: 2 of 6 orbit types proved feasible ((0,1,2,3,4,7,13) and (0,1,3,6,7,10,12); the plane
  type (0,1,2,3,4,5,6) itself timed out UNKNOWN);
* 8-subsets: representative (0,1,3,6,7,10,12,13) — verified to be the **affine
  hyperplane-complement** orbit (its complement is a plane; orbit size 15), the same channel-set
  shape as the single-pair optima — proved feasible: **eight simultaneously perfect channels exist**
  (96 orbits from the perfect channels alone). The other five 8-orbit types timed out UNKNOWN (900 s
  each);
* 9-subsets: **all five orbit representatives timed out UNKNOWN** (900 s each).

> **A recorded result that had to be weakened.** The original run's break logic recorded
> `max_simultaneously_perfect = 8`. That is only a **proved lower bound**: infeasibility at `k = 9`
> was **not** established by those runs. The code and `perfect.json` now say so explicitly
> (`k_break_proved: false`). Five kissat decision instances (`sub_perfect9_rep*.cnf`: independence
> plus per-channel exactly-12) were written to close it. If all five are UNSAT, "≤ 8 perfect
> channels" becomes a theorem and yields the occupancy cut `Σ_c t_c ≤ 8·12 + 7·11 = 173` orbits
> (692 weight) on its own.

### 9.7 Solver log — what worked, what did not, and what was left running

* **SAT decision "≥ 497"** (encoding soundness unit-tested by `selftest`, 660 checks). Three
  instances written: raw (`sat_ge497.cnf`, 25 MB), with all entailed cuts (`sat_ge497_cuts.cnf`,
  88 MB: channel ≤ 12, pair-line ≤ 16, pair-plane ≤ 32, pair ≤ 64, monads/block ≤ 8), and the cuts
  instance with monads forced off (`sat_ge497_cuts_duadsonly.cnf`). cadical195 on the raw instance
  was abandoned after ~1 h (its partial DRAT is `sat_ge497.drat`); the two cuts instances were handed
  to **kissat 4.0.4** with DRAT proof logging. An UNSAT on the cuts instance is the theorem "no
  2³-invariant 60°-free set of weight ≥ 497" (modulo the proven cut lemmas), with a machine-checkable
  proof; the duads-only variant would pin the duad-only optimum at 496.
* **RC2/RC2Stratified MaxSAT** (pysat): no answer within 500 s on even the 960-vertex single-pair
  subproblem — dead end.
* **pysat decision ascent on subproblems**: pair-0 took > 20 min without settling ≥ 260 vs 256 — dead
  end (these instances are hard for plain CDCL; the refutations are pigeonhole-flavoured).
* **LP over the 13,151-clique cover** (`cliques` + `lp`): bound **720.000** = exactly the
  360-line-clique bound `4 × (15 × 12)`; the greedy cross-channel and monad cliques add nothing
  fractionally (the flat profile `m ≡ 4` is LP-feasible). Two-point and spectral methods are equally
  blind: the duad-only conflict graph is 254-regular with `λmin = −18` exactly, Hoffman bound 190.6
  orbits (762.4 weight); and the **symmetry-reduced weighted Lovász theta** — the coherent
  configuration of Aut on the 3240 orbits has rank just **27** (2 fibers, 6 edge orbitals in 5
  symmetric classes), so `θ(G, w)` reduces to a 5-parameter eigenvalue optimisation — comes out
  ≈ **1031**, *worse* than the clique LP. The −16 ⇔ +16 orbit symmetry really is invisible to every
  quadratic relaxation tried.
* **scipy HiGHS MILP** over the clique cover (2 h): never left the root — the time went into the root
  LP / analytic centre; primal incumbent 4 (!), dual bound 720.000. Dead end (`milp.log`).
* **SCIP, 2 h** (`scip.log`): worse — one node, LP unfinished, dual bound stuck at the trivial 12,240
  (total weight). Dead end.
* **OR-Tools CP-SAT 9.15**, the strongest general tool tried, two 5-hour runs with all proven cuts
  and the 496 hint (`cpsat_full.log`, `cpsat_duads.log`): both end **best = 496, best_bound = 720** —
  the bound never moved off the clique LP, in the full and the duads-only model alike (identical,
  because the LP optimum uses no monads: the flat duad profile `m ≡ 4` already pays 720, so deleting
  the monads changes nothing the relaxation can see). 10 h of anytime search never found 497, which
  is (only) heuristic evidence that 496 is optimal.

```
RESULT section=cpsat duads_only=False status=FEASIBLE best=496 upper_bound=720.0 seconds=18000.8
RESULT section=cpsat duads_only=True  status=FEASIBLE best=496 upper_bound=720.0 seconds=18000.8
RESULT section=milp primal=4.0 dual_bound=719.99… status=timelimit seconds=7202.8
RESULT section=scip status=timelimit primal=304.0 dual_bound=12240.0 nodes=1 seconds=7209.9
RESULT pairs01_exact status: FEASIBLE best: 96.0 bound: 120.0 9002s   (units: ORBITS)
RESULT section=perfect max_simultaneously_perfect=8 S_perfect=[2, 3, 7] seconds=12658.3
       (>= 8 PROVED; the k = 9 break is NOT proved — all five reps UNKNOWN at 900 s)
RESULT section=monadkill u=[14, 26, 36, 44, 50, 54, 56, 56] statuses=[OPTIMAL x8] seconds=1.2
RESULT autprod full_product=True (image on (channels, pairs) = GL(4,2) x S3, order 120960)
RESULT section=subopt key=pairs0,1_line0    status=OPTIMAL best=24 (0.6 s)
RESULT section=subopt key=pairs0,1_affine0  status=OPTIMAL best=64 (2.7 s)
```

**Long-horizon solver status as last recorded (2026-08-30 ~19:10; all LEFT RUNNING and, as of this
writing, unresolved).** Every instance is a *decision* CNF in `runs/a2/`; an UNSAT verdict has the
stated meaning (encodings soundness-checked against explicit witnesses: each witness satisfies its
"≥ value" instance and falsifies "≥ value+1"). SAT means a better incumbent — harvest the model,
expand to vectors, verify, and follow the custody protocol.

| instance (kissat, `runs/a2/`) | UNSAT would prove | solver clock at end |
|---|---|---|
| `sat_ge497_cuts.cnf` → `.drat` | **opt = 496, the full theorem** (modulo proven lemmas), DRAT-checkable | 6.5 h |
| `sat_ge497_cuts_duadsonly.cnf` → `.drat` | duads-only opt = 496 | 5.9 h |
| `sub_pairs2_ge97.cnf` → `.drat` | two-pair = 96 orbits ⇒ opt ≤ 582 | 2.5 h |
| `sub_plane_ge65.cnf` → `.drat` | plane = 64 orbits ⇒ opt ≤ 560 | 2.4 h |
| `sub_pairs2plane_ge49.cnf` → `.drat` | two-pair plane = 48 ⇒ opt ≤ 614 (558 with plane) | 2.2 h |
| `sub_g2_ge23.cnf` → `.drat` | g(2) = 22 ⇒ opt ≤ 630 | 2.2 h |
| `sub_g3_ge28.cnf` → `.drat` | g(3) = 27 (with g(2): opt ≤ 588) | 2.2 h |
| `sub_g2_ge23_sb.cnf`, `sub_pairs2plane_ge49_sb.cnf` | same as their parents (BreakID symmetry-broken: 47 / 189 generators; the verdict transfers, the DRAT does not) | 2.0 h |
| `sub_perfect9_rep{0..4}.cnf` | (all five) no 9 simultaneously perfect channels ⇒ occupancy cut 692 | written; runs stopped after 15 min to free cores |

`sub_plane_ge65_sb.cnf` (995 BreakID generators — the plane stabiliser is huge, so this is the
variant most likely to crack first) and `sub_g3_ge28_sb.cnf` are generated and ready but were not
launched (core budget). A `drat-trim` binary for proof checking is a 30-second build from
github.com/marijnheule/drat-trim when a verdict lands.

**The obvious next moves**, in order: (1) harvest the nine running kissat instances
(`grep "^s " runs/a2/kissat_*.log`); any UNSAT immediately upgrades the certified bound per the table
above, and `combine` recomputes the composite in seconds
(`combine --plane 64 --pairs2plane 48 --pairs2 96 ...`, passing exactly the caps that were closed).
(2) If cores free up, launch kissat on `sub_plane_ge65_sb.cnf` — the highest payoff (560) per
instance size among the sub-questions. (3) The structural route that would beat 558 without the
monolith: close `g(2)`/`g(3)`/`g(4)` (the witness-versus-LP gaps 22/30, 27/45, 36/60 are the largest
known), then rerun `combine`; the `g(2..4)`-closed hypothetical already reaches ≤ 588 with two
ingredients. (4) If a SAT verdict appears anywhere with weight ≥ 497: expand orbits to vectors, write
to `runs/found/`, verify with `build/release/tools/verify_s` AND `.venv/bin/python
python/verify_S.py`, and report loudly. **No solver produced any incumbent above 496 in ~40
CPU-hours.**

### 9.8 Files

| file | content |
|---|---|
| `python/tools/channel_opt.py` | everything: `build`, `census`, `structure`, `cliques`, `lp`, `milp`, `sat`, `maxsat`, `cpsat`, `pairs`, `perfect`, `selftest`, and the decomposition subcommands `profilecaps`, `monadkill`, `subcnf`, `subopt`, `plane`, `combine` |
| `runs/a2/model.npz`, `census.json` | the verified orbit model |
| `runs/a2/lines.npz` | line vectors, `H_b` graphs, monad-line incidences |
| `runs/a2/cliques.npz` | 13,151-clique cover of all 456,480 edges |
| `runs/a2/pairs.json` | exact subproblem optima |
| `runs/a2/perfect.json` | which channel subsets can be simultaneously perfect (with the `k = 9` caveat) |
| `runs/a2/chancuts.json` | 150 proven per-(pair, line/plane) caps |
| `runs/a2/profilecaps.json` | `g(k)`: single-pair maxima under per-channel caps |
| `runs/a2/monadkill.json` | `u(k)`: exact minimum block-lines killed by k monads |
| `runs/a2/sat_ge497_cuts*.cnf/.drat` | the two long-horizon kissat decision instances plus proofs-in-progress |
| `runs/a2/sub_pairs2_ge97.cnf`, `sub_plane_ge65.cnf`, `sub_pairs2plane_ge49.cnf`, `sub_g2_ge23.cnf`, `sub_perfect9_rep{0..4}.cnf` | decomposition decision instances (kissat) |
| `runs/a2/subopt.json` | generic sub-optimum ledger (two-pair line = 24, two-pair affine = 64, …) |
| `runs/a2/pairs2_witness96.json` | explicit independent 96-orbit two-pair witness (64 affine + 32 plane) |
| `runs/a2/autprod.log` | proof that Aut acts as the full GL(4,2) × S₃ on (channels, pairs) |
| `runs/a2/cpsat_best.json`, `cpsat_best_duads.json` | the 496 incumbents of the 5-hour runs |
| `runs/a2/*.log` | every solver transcript with `RESULT` lines |

---

## 10. Corrections and superseded claims

Every one of these was found by a later computation in this repository and is recorded here in its
corrected form. They are collected in one place because several of them changed how the plateau is
described.

1. **"The 64 plateau 496s have identical Gram and tightness histograms" — WRONG.** The Gram histogram
   `{−32:248, −8:37504, 0:47504, 8:37504}` really is constant over all 64, as are the minimum
   tightness (4), the number of tightness-4 vertices (80) and their Gram histogram. The **tightness
   histogram is not constant**: it takes exactly **5 values**, in class sizes 16/16/16/8/8 (§4.8, and
   independently re-derived in §7.3 and §8.3). The original claim had checked only
   `S496_alt_overlap336.txt`, which happens to lie in the record's own fingerprint class because both
   (80,80) moves preserve the fingerprint (`fp(T) = fp(T^15) = fp(T^48) = fp(T^63)`).

2. **"The alternative 496 is plausibly a Co₀-image of the record" — REFUTED.** Verbatim, the original
   claim was that this set has "the same Gram histogram and the same tightness histogram as the
   original, so it is plausibly a Co₀-image of it". The full equivalence computation (§8) shows that
   `S496_alt_overlap336.txt` = `S_48` is **not even isometric** to the record: the individualisation
   search terminates with zero Gram-graph isomorphisms after 45.71 s. More generally none of the 63
   non-trivial plateau sets is a Co₀-image of the record, and **the 64 sets fall into 64 distinct
   Co₀-classes**. The plateau is not a group orbit.

3. **Trace bookkeeping in the Co₀ stabiliser — WRONG in the original table.** The element table
   listed "b, ab" both at trace +8. In fact the eight matrices have trace multiset
   **{+24, +8 × 3, −8 × 3, −24}**, and the three trace-+8 involutions `t₁, t₂, t₃` satisfy
   `tᵢtⱼ = −t_k`, so the *product* of two of them has trace −8. The group is unchanged —
   `{±1} × ⟨t₁, t₂⟩ ≅ 2³` — and nothing downstream depended on the traces; only the bookkeeping was
   wrong. Detail and re-verification in `04-structure-of-the-496.md`. (§6.3)

4. **"The 496's plateau component is closed" — TOO STRONG as first stated.** The exhaustive plateau
   walk established closure only *under plateau moves of size k ≤ 12*, where the component is exactly
   the 4-cube on 16 sets. Two further (80,80) plateau moves, invisible to any `k ≤ 12` search, were
   found later by the antipodal GPU search, so the true component has at least **64** sets. The
   `k ≤ 12` statement stands unchanged; the general statement was withdrawn. (§3.2)

5. **"8,292,375 full frames of the minimal vectors" — WRONG by ~14 orders of magnitude.** That number
   is `|Λ₄|/48`, the count of **Conway crosses** (frames of norm-8 vectors). Minimal-vector frames
   number **≥ 5.63 × 10¹³ exactly** and **≈ 8.5 × 10²⁰ estimated**. The plan built on the smaller
   figure — enumerate all frames and build a conflict structure between subframes — is not executable
   as written; only the local version is. (§5.2)

6. **"The plateau 496 has identical invariants, consistent with the plateau sets being Co₀-images" —
   the invariants are right, the inference is not.** The frame-swap image
   `runs/frames/plateau_496_seed4479.txt` genuinely has the identical Gram spectrum, tightness
   histogram, orthogonality-degree profile, maximal-clique profile and 8-clique count as the record;
   it lies in the record's fingerprint class. But identical invariants do not imply equivalence, and
   §8 settles the matter negatively. (§5.4)

7. **"There is an obstruction to a 4th perfect channel" — REFUTED.** Exact CP-SAT feasibility shows
   4-, 5-, 6-, 7- and 8-subsets of channels can all be simultaneously perfect; the record has only 3
   perfect channels as an optimisation trade-off, not because of a hard constraint. (§9.6)

8. **"Maximum simultaneously perfect channels = 8" — a lower bound, not an equality.** The original
   run's break logic recorded 8, but `k = 9` infeasibility was **not** proved (all five orbit
   representatives returned UNKNOWN at 900 s). The artefact `perfect.json` now carries
   `k_break_proved: false`, and five kissat instances were written to close it. (§9.6)

9. **The design estimate of ~2 × 10⁷ GPU moves/s was ~12× optimistic.** The real ceiling for the
   uint16-scatter layout is ≈ 1.5 × 10⁶ row updates/s, DRAM-bound on ~240 KB of scattered sector
   traffic per move rather than on the 18.4 KB adjacency row that had been budgeted. All search
   budgets in §4 are stated against the measured figure. (§4.1)

10. **A checkpoint/reseed race produced false "records" of size 497–568.** Both verifiers rejected
    every one of them and the run exited `ok=0`; the cause was a driver bug (stale `best_S` tail),
    since fixed, with three layers of new guards. The evidence files are kept in
    `runs/gpu_mis/ABORTED_record_bug/`. No claimed record from that run survives. (§4.6)

11. **The ξ-conjugate probe of the monomial stabiliser is superseded.** It found {±I} for the
    stabiliser of `ξ·S` in 2¹²:M24; the full Co₀ computation subsumes it and settles the conjugates
    question completely. (§6.2)

---

## 11. What is proved, and what is search evidence

The distinction matters more here than usual, because the headline of the whole document is a
negative result and negative results are easy to overstate. Nothing below says "no 497 exists".

### 11.1 Proved (exact computation, exhaustive within a stated scope)

* **The 496 admits no improving `(k, m)`-swap for any removal size `k ≤ 12`**, and its only equal-size
  moves with `k ≤ 12` are four pairwise-disjoint, negation-closed (12,12) plateau moves. Exhaustive:
  900,672 unions of conf sets, every pool solved exactly by branch and bound. Any improvement must
  remove ≥ 13 vectors. (§2.2)
* **The 488 admits no improving swap for any `k ≤ 11`** (level 12 hit a cap and is *not* exhaustive),
  with 37 primitive plateau moves: 24 of size 2 and 13 of size 8. (§2.4)
* **The sub-component of the 496's plateau generated by moves of size ≤ 12 is exactly the 4-cube on
  16 sets**, each of which is again 496, antipodal, maximal, free-vertex-free and swap-optimal for
  every `k ≤ 12`. Frontier exhausted, not budget-limited. (§3.3)
* **All 64 plateau sets are maximal**: 0 free vertices, no tightness-1/2/3 vertex, minimum tightness
  exactly 4 with exactly 80 tightness-4 vertices — certified over all 196560 minimal vectors by three
  independent implementations in exact arithmetic. Hence no 497 by single-vector extension anywhere
  on the plateau, and no `(k, k+1)`-swap with `k ≤ 3` at any of the 64. (§7.3)
* **The 6-cube structure of the plateau**: six pairwise-disjoint, negation-closed, commuting atoms
  (four of size 12, two of size 80); all 384 forward/reverse applicability checks pass; the 64
  overlaps match the cube prediction exactly. (§7.2, §7.4)
* **`Stab_{2¹²:M24}(496) = {±I}`** (order 2) and **`Stab_{Co₀}(496) ≅ 2³`** (order 8), both exact,
  the first by two independent derivations. The same for the 488. On `S` the 2³ has 124 orbits of
  size 4, each `{±x, ±y}` with `x ⊥ y`. (§6.2, §6.3)
* **The 64 plateau sets are pairwise Co₀-inequivalent — 64 distinct classes** — and realise exactly
  **8 isometry (congruence) classes of 8**. The decision procedure is complete in both directions and
  passes five positive controls. Seven sets are congruent to the record without being
  lattice-equivalent to it. (§8.5, §8.7)
* **Minimal-vector frame counts**: exactly 502,590,825 triangles through a class (16,464,875,427,000
  in total) and 1,636,707,770,775 4-cliques (40,213,909,927,941,750 in total); and the exact lower
  bound **#frames ≥ 56,303,426,804,625** from the (±4,±4) family, whose count `23!!` /
  `21!!` is proved by the perfect-matching argument and verified by DFS at `2m ≤ 16`. (§5.2)
* **No orthogonal clique through any class of conflict ≤ 8 improves the 496**: the `--min-gain 1`
  pass is complete for all four tiers over all 1840 seed classes, `incomplete=0`. Best gain 0,
  attained only by the frame-for-frame swap, which is an involution. (§5.4)
* **The 496 is one full frame plus 28 orthogonal octuples** (recovered for 2 of its 3 frames within
  budget; the decomposition is highly non-canonical). (§5.3)
* **Exact maximum-weight orbit unions for 23 of the 95 subgroups**, including `23:11` = 301 and the
  cap of 48 for everything containing a large 2-group. (§6.5)
* **In the Turyn slice**: the single-pair optimum is exactly 256 (64 orbits), the single-pair-plus-its-
  monads optimum exactly 288, the monads-only optimum exactly 48; the cap rule
  `max = 8 · capmax(C)` for every channel subset shape tested; the kill function
  `u(k) = 14, 26, 36, 44, 50, 54, 56, 56`; `|Aut|` of the reduced model = 31,708,938,240 acting as the
  full GL(4,2) × S₃; and the certified sandwich **496 ≤ opt ≤ 720** for 2³-invariant sets, with the
  720 reproduced independently by HiGHS and CP-SAT. Maximum-monad solutions are provably sub-optimal
  (≤ 432). Four, five, six, seven and eight simultaneously perfect channels all exist. (§9)

### 11.2 Search evidence (no 497 found; not a proof that none exists)

* **The GPU ILS never reached 497**, from any seed, in any mode: 512 chains × 20,000 iterations from
  the 496 in both plain and antipodal mode; 242 M ILS iterations across two 30-minute production runs
  over a mixed seed pool; 512 chains × 1000 iterations from the 488 never reached 489. Deepest dips
  475 / 466 from the 496. The engine saturates the 64-node plateau component of each seeded 496 and
  never leaves it; both production runs converge to exactly 4160 = 65 × 64 distinct 496s. (§4)
* **The plateau walk found no improvement at any of 424 expanded nodes**, and the 488 walk's 408-node
  sample is a *sample*: that component has ≥ 2³⁷ sets and cannot be exhausted. (§3)
* **Class-level tabu search from the 496 cannot climb** (600 s, 6633 moves, best 248 classes = 496
  vectors, never more); from greedy unions it saturates at 138–147 classes (276–294 vectors). (§5.5)
* **Greedy constructions saturate far below the record**: vertex-level greedy 225–236 (improved to
  228–240 by small swaps), frame-union greedy mean 212 / best 228 vectors, GPU climb from greedy
  seeds 296–303 after 20,000 iterations. (§2.5, §4.2, §5.5)
* **No subgroup among 95 yields an invariant set above 496**; the best non-stabiliser union is 402
  (`C14±`), and none of the non-exact rows is proven optimal. (§6.5)
* **496 is the best 2³-invariant set found by every method** (ILS, branch and bound seeded and
  unseeded, MILP, 10 h of CP-SAT anytime search), but **optimality over 2³-invariant sets is not
  proved**: the certified upper bound is 720. (§6.6, §9.5)
* **No `(k, k+1)`-swap with `k ≤ 8` was found at any node of the 488 walk**, but that is per-node
  exhaustive only to the stated `kmax`, over a sampled region.

### 11.3 Open

* Is 497 attainable at all? Unknown. The independent upper bounds are 850 (two-point LP) and 837
  (three-point Terwilliger SDP) — see `02-upper-bounds.md` — so the gap between what is achieved
  (496) and what is excluded (838) remains very wide.
* Is 496 the exact maximum over 2³-invariant (Turyn-symmetric) sets? Open, sandwiched as
  `496 ≤ opt ≤ 720`. Nine kissat decision instances that would each certify a strict improvement —
  including one whose UNSAT is the full theorem with a DRAT proof — were left running unresolved.
  (§9.5, §9.7)
* Does a *non*-antipodal 496 exist? None was ever seen; all searches that reached 496 produced
  antipodal sets.
* Is there an improving move that removes ≥ 13 vectors? Nothing in this document rules one out; the
  exhaustive guarantee stops at `k = 12`, and the GPU search's neighbourhood, though unbounded in
  principle, is in practice confined by its perturbation window.
* How many Co₀-classes of maximal 496-point configurations are there in total? At least 64 are known
  (the plateau of this one record), plus the 488 and the sporadic 492s seen in the production runs.
  No classification is claimed.











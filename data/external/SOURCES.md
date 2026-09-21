# data/external — provenance (T1.3, fetch half)

Fetched **2026-08-25** (local time; the Cohn page's HTTP `date` header read 2026-08-26 02:47 UTC).
Everything under `data/external/PackingStar/` and `data/external/Kissing-Numbers/` is a verbatim
upstream clone plus the in-place extraction of two zip archives (see §1.2). The clones are gitignored
(`data/external/.gitignore`); re-create them from the commit hashes below. Nothing was converted to our
coordinates yet — that is the second half of T1.3 and needs T1.2.

Inspection script (read-only, system `python3` + numpy 1.21): `python/tools/inspect_external.py`
(≈13 s). Its full output is pasted in `docs/reports/T1.3-fetch.md`.

---

## 1. PackingStar (Ma et al. 2025, arXiv:2511.13391)

| item | value |
|---|---|
| URL | https://github.com/CDM1619/PackingStar |
| clone | `git clone --depth 1` (shallow; 57 commits upstream, history read via the GitHub API) |
| commit | `50ea645a9805d4f29b96180550186d26a166c3be` — "Add files via upload", 2026-06-08 08:18:21 UTC |
| size | 82 MB checked out (433 MB after unzipping §1.2; `.git` 27 MB) |
| paper versions | v1 2025-11-17, v2 2026-01-21, v3 2026-02-11, v4 2026-06-02 (arXiv listing) |
| commits touching `25D-31D/` | `8f7070416ded` 2025-11-20 (initial), `bb57cf509703` + `1e9851a5e095` 2026-01-15 (zip deleted and re-uploaded — this is when the 31D file went from 238078 to 238350; the 31D `.npy` inside is dated 2026-01-14, the other six 2025-11-04..17) |

### 1.1 Relevant files (as shipped)

| path (under `PackingStar/`) | bytes | sha256[:16] | content |
|---|---|---|---|
| `25D-31D/Si_configurations.zip` | 1,092,405 | `8177b8b8bd9a042a` | the two 24-D files below |
| `25D-31D/25D-31D_new_bounds_configurations.zip` | 7,354,971 | `4f1342d906758b89` | the seven record configurations below (+ macOS junk) |
| `25D-31D/partitioned_D5.npy` | 1,728 | `ea7ddb4bcb9c9e22` | int64 (40,5): D5 root system (norm 2) = 40-point kissing config in R^5, in the order used for the T-partition |
| `25D-31D/partitioned_E7.npy` | 7,184 | `4e4b59821551c8c8` | float64 (126,7): E7 roots as unit vectors, entries ∈ {0, ±1/2, ±1/√2, ±1/(2√2), ±1} |
| `README.md`, `verify_coordinates.py`, `verify_cosmatrix.py` | — | — | their verifier: normalise rows, check pairwise cos ≤ 0.5 + 1e-6 in batches |

Not relevant to us: `12D_14D_15D/`, `13D_rational_configurations/`, `New generalized kissing configurations/`.

### 1.2 Files extracted locally from the zips (not in the upstream tree)

`unzip -o` into `25D-31D/`; `__MACOSX/` and `.DS_Store` deleted.

| path (under `PackingStar/25D-31D/`) | bytes | sha256[:16] | shape / dtype | format |
|---|---|---|---|---|
| `Si_configurations/24D_496_Si.npy` | 23,936 | `0121a72731e48490` | (496, 24) **int16** | **the 496-vector S**, √8-integer scaling: entries ∈ {−4..4}, every row has squared norm 32 |
| `Si_configurations/24D_196560_coordinates.npy` | 37,739,648 | `6ecda160033e634f` | (196560, 24) float64, all values integral | all Leech minimal vectors, same scaling (norm 32) |
| `25D-31D_new_bounds_configurations/25D_197056_coordinates.npy` | 39,411,328 | `0ccb95be9c1a58d2` | (197056, 25) float64 | unit vectors |
| `…/26D_198550_coordinates.npy` | 41,298,528 | `054e7d24a0824749` | (198550, 26) float64 | unit vectors |
| `…/27D_200044_coordinates.npy` | 43,209,632 | `b7dcb05ea2c90438` | (200044, 27) float64 | unit vectors |
| `…/28D_204520_coordinates.npy` | 45,812,608 | `e46fae5ac48e804f` | (204520, 28) float64 | unit vectors |
| `…/29D_209496_coordinates.npy` | 48,603,200 | `bf29e6c156c41738` | (209496, 29) float64 | unit vectors |
| `…/30D_220440_coordinates.npy` | 52,905,728 | `066aae16d5259bcf` | (220440, 30) float64 | unit vectors |
| `…/31D_238350_coordinates.npy` | 59,110,928 | `9ed11766c8e536c5` | (238350, 31) float64 | unit vectors (dated 2026-01-14; the only file from the January re-upload) |

**There are no per-dimension `S_i` files.** The only explicit S is `24D_496_Si.npy`. For n = 26..31 the
S_i are implicit in the unit-vector configurations and were recovered exactly (see §1.4): a row is
equatorial `(x,0)/√32`, lifted `(x/(4√3), y/√3)` (x integer norm-32, y a unit vector of the R^d kissing
configuration), or an extra sphere `(0, y')`. Rescaling by √32 resp. 4√3 gives integers with residual
≤ 8.9e−16.

### 1.3 Gram-level findings (their own coordinates)

* `24D_496_Si.npy`: 496 distinct rows, diag 32, off-diagonal multiset {−32: 496, −8: 75008, 0: 95008, 8: 75008},
  max 8 → an independent set of the 60°-conflict graph; antipodal-closed (248 ± pairs). No entry 16 or −16.
  Shape counts (#(±2^8), #(∓3,±1^23), #(±4,±4)) = see report. All 496 rows lie in `24D_196560_coordinates.npy`.
* `24D_196560_coordinates.npy`: 196560 distinct integral rows, norm 32, inner-product histogram against
  row 0 = {−32:1, −16:4600, −8:47104, 0:93150, 8:47104, 16:4600, 32:1} (README §1.2 histogram reproduced).
  **As a set it is identical to Kallal–Kan–Wang `minvects.txt`** (different row order).
* The 496 file is **not** the S lifted in `25D_197056_coordinates.npy` (same Gram histogram, different
  set); it **is** S_8 of the 29D, 30D and 31D configurations.

### 1.4 Implicit S_i / T_i content per configuration (all recovered S_i have |S_i| = 496)

| file | equatorial | lifted | extra (=K(d)) | #S_i | T_i sizes | check |
|---|---|---|---|---|---|---|
| 25D_197056 | 196064 | 992 | 0 (K(1)=2 not usable) | 1 | {2:1} | 196560 + 496 |
| 26D_198550 | 195568 | 2976 | 6 | 2 | {3:2} | 6 + 196560 + 4·496 |
| 27D_200044 | 194080 | 5952 | 12 | 5 | {3:2, 2:3} | 12 + 196560 + 7·496 |
| 28D_204520 | 192592 | 11904 | 24 | 8 | {3:8} | 24 + 196560 + 16·496 |
| 29D_209496 | 189616 | 19840 | 40 | 14 | {3:12, 2:2} | 40 + 196560 + 26·496 |
| 30D_220440 | 184656 | 35712 | 72 | 24 | {3:24} | 72 + 196560 + 48·496 |
| 31D_238350 | 175728 | 62496 | 126 | 42 | {3:42} | 126 + 196560 + 84·496 |

In every file: the S_i are pairwise disjoint, each has norm 32 / off-diagonal ≤ 8 / the same Gram
histogram as the 496, equatorial set = C \ ∪S_i exactly, T-vectors within a T_i have inner product −1/2
(or −1 for pairs), across T_i ≤ 1/2, extra spheres pairwise ≤ 1/2 and ≤ √3/2 against every T-vector
(28D and 30D sit exactly at √3/2 up to 1e-6).

---

## 2. Kallal–Kan–Wang (2016/2018, arXiv:1608.07270)

| item | value |
|---|---|
| URL | https://github.com/kenzkallal/Kissing-Numbers |
| clone | `git clone --depth 1` (5 commits upstream, all 2016-08-02) |
| commit | `548a09282ea1037077a435be4157e7f464b1b146` — "Update README.md", 2016-08-01 20:49:50 −0400 |
| size | 14 MB |

| path (under `Kissing-Numbers/`) | bytes | sha256[:16] | format |
|---|---|---|---|
| `minvects.txt` | 11,004,241 | `540ea1a0fc3a809e` | 196560 lines × 24 space-separated ints, norm 32, √8-integer scaling; shape counts 97152 / 98304 / 1104; ip histogram vs row 0 as in README §1.2 |
| `Vbasis.txt` | 1,359 | `510e1a92a6316d96` | 24 lines × 24 ints: a Leech basis of minimal vectors (all rows in `minvects.txt`, det Gram = 8^24) |
| `S_1.txt … S_59.txt` | 26,064–28,066 each | combined (sorted -V, concatenated) `45ab00655d63eb29` | one vector per line, 24 ints, norm 32; **the 488-set is `S_1.txt`** |

Sizes |S_1..S_59| = 488 ×24 (S_1–S_24), 486, 484 ×4, 482 ×4, 480 ×4, 478 ×3, 476, 474 ×5, 472, 468 ×6,
466 ×2, 464 ×2, 462, 460 (S_59). Sum 28324; all 59 pairwise disjoint; every set independent
(off-diagonal ∈ {−32, −8, 0, 8}), antipodal-closed, contained in `minvects.txt`.
S_1 Gram multiset: {−32: 488, −8: 73088, 0: 90992, 8: 73088}. Their 2018 template applied to these sizes
gives 198512 / 199976 / 204368 / 208272 / 219984 / 232878 for n = 26..31.

---

## 3. Coordinate convention (both repos)

Both repositories use the **same** integer coordinates: √8 scaling, minimal vectors of norm 32, entries
in {−4,…,4}, the three shapes of README §1.2. The 759 octads read off the (±2^8, 0^16) vectors span a
12-dimensional GF(2) code (a Golay code), but **none of them is an octad of the cyclic Golay code
generated by g(x) = x^11+x^10+x^6+x^5+x^4+x^2+1 (README §1.2) with identity coordinate labelling**.
So mapping into our canonical `C` will need at least a permutation of the 24 coordinates (an M24-coset /
Golay-code equivalence), possibly signs; this is the conversion half's job (or T4.1 if a non-monomial
automorphism turns out to be needed).

---

## 4. Henry Cohn's table

| item | value |
|---|---|
| URL | https://cohn.mit.edu/kissing-numbers/ (data archive: https://hdl.handle.net/1721.1/153312, plot: https://cohn.mit.edu/kissingplot/) |
| fetched | 2026-08-25 (server date header 2026-08-26 02:47 UTC) |
| "last updated" | **no update date is stated on the page or in its HTML metadata.** Newest reference cited is arXiv:2607.20359 (July 2026), so the page was edited in or after July 2026. |

Rows n = 24..31 (lower / upper / references as numbered on the page):

| n | lower | upper | refs | lower-bound source | upper-bound source |
|---|---|---|---|---|---|
| 24 | 196560 | 196560 | [12, 14, 20] | Leech 1967 | Levenšteĭn 1979; Odlyzko–Sloane 1979 |
| 25 | 197056 | 265006 | [15, 10] | Ma et al. 2025 (PackingStar) | de Laat–Leijenhorst 2024 |
| 26 | 198550 | 367775 | [15, 10] | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 27 | 200044 | 522212 | [15, 10] | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 28 | 204520 | 752292 | [15, 10] | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 29 | 209496 | 1075991 | [15, 10] | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 30 | 220440 | 1537707 | [15, 10] | Ma et al. 2025 | de Laat–Leijenhorst 2024 |
| 31 | 238350 | 2213487 | [15, 10] | Ma et al. 2025 | de Laat–Leijenhorst 2024 |

Identical to README §1.1 — nothing has moved past 496 in dimensions 24–31.

---

## 5. Kravatskiy's repository (added September 2026)

| item | value |
|---|---|
| URL | https://github.com/alexlegeartis/KissingNumbers (MIT) |
| location | git submodule `external/kravatskiy` (since 2026-09-21; before that a gitignored clone under `data/external/`) |
| pinned commits | `52fa09d16e20394f06c1d19b7a1bdc967c865d9f` (2026-09-18) and `c349d565362f39f8492e55bda7129cd0787a1d6e` (2026-09-20), checked out on demand by `tools/kravatskiy/pinned.py` |
| used | `verifications/improved/dim25-lens-heads/` and `verifications/improved/dim26-27-iota-triangles/`, read by the independent verifiers in `tools/kravatskiy/` (REPRODUCE.md, last section) |

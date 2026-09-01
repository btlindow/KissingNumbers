"""Orbitals of the point stabiliser and the Terwilliger-type algebra T(x) of
the Leech minimal-vector scheme (docs/design.md T2.3; docs/reports/T2.3.md section 1).

Inputs (both written by C++ tools, both re-verified here in exact integer
arithmetic):

* data/scheme/orbitals.json  (tools/triple_orbitals): for x = C[0] and
  Gamma = <five explicit Co_0 generators>, the orbitals of Stab_Gamma(x) on
  C x C as LABELS computed from explicit stabiliser elements — so two pairs
  with the same label are provably in the same orbital — with their
  intersection numbers c[u][s][t] = #{w : (y,w) in s, (w,z) in t} for
  (y,z) in u, the S_3 action (transpose, swap of x and y) on the orbitals,
  and the ordered-triple counts of S496 / S488 per orbital.
* data/scheme/triples.json (tools/triple_stats, GPU brute force): for each
  class-cell (i,j,k) the distinct 4-point histograms H_z over all z, with
  the number of z carrying each.  H_z is Stab(x,y)-invariant, so the total
  number of distinct histograms is a LOWER bound on the number of orbitals.

The proof that the labels are exactly the orbitals: labels refine orbitals
(explicit elements), #labels = D, #orbitals >= #variants; D == #variants
forces equality.  Then span{B_u} (B_u the 0/1 matrix of orbital u) is the
centraliser algebra of Stab(x) — closed under products and transposition —
and c[u][s][t] are its structure constants: B_s B_t = sum_u c[u][s][t] B_u.
This is the algebra the three-point SDP (sdp3_scheme.py) is block-
diagonalised in; its dimension 148 exceeds the number 147 of nonzero
intersection numbers p^i_{jk} (the "triple-class matrices" span only a
147-dimensional subspace that is NOT an algebra: the cell (3,3,3) of
mutually orthogonal triples splits into two orbitals of sizes
93150*42240 and 93150*924).

Everything here is Python int / Fraction; numpy is used only for the dense
integer tensor c.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import numpy as np

NCLASS = 7
IDENTITY = 6
CONFLICT = 5
CLASS_DOTS = (-32, -16, -8, 0, 8, 16, 32)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ORBITALS_JSON = os.path.join(_HERE, "..", "..", "data", "scheme", "orbitals.json")
DEFAULT_TRIPLES_JSON = os.path.join(_HERE, "..", "..", "data", "scheme", "triples.json")
DEFAULT_SCHEME_JSON = os.path.join(_HERE, "..", "..", "data", "scheme", "intersection_numbers.json")


class OrbitalError(ValueError):
    pass


@dataclass(frozen=True)
class Orbital:
    id: int
    i: int          # class(x, y)
    o: int          # orbit index within class i
    z: int          # representative z (y = y_i, the class representative)
    j: int          # class(x, z)
    k: int          # class(y, z)
    size: int       # |u| = number of pairs (y, z) in the orbital (x fixed)
    osize: int      # number of z for fixed y = |u| / v_i
    transpose: int  # orbital of (z, y)
    swap: int       # orbital of the triple (y, x, z) moved back to x


@dataclass
class Orbitals:
    N: int
    x: int
    valencies: tuple[int, ...]
    reps: tuple[int, ...]              # y_i per class
    orbitals: list[Orbital]
    c: np.ndarray                      # int64 [D, D, D]: c[u, s, t]
    sets: dict[str, tuple[int, list[int]]] = field(default_factory=dict)   # path -> (|S|, triples per orbital)
    source: dict | None = None

    @property
    def D(self) -> int:
        return len(self.orbitals)

    def diag(self, k: int) -> Orbital:
        """The orbital of (y_k, y_k): pairs (y, y) with class(x, y) = k."""
        for u in self.orbitals:
            if u.i == k and u.z == self.reps[k]:
                return u
        raise OrbitalError(f"diagonal orbital of class {k} not found")

    def by_class(self, i: int, j: int, k: int) -> list[Orbital]:
        return [u for u in self.orbitals if (u.i, u.j, u.k) == (i, j, k)]


def load_orbitals(path: str | os.PathLike = DEFAULT_ORBITALS_JSON) -> Orbitals:
    with open(path) as f:
        d = json.load(f)
    orbs = [Orbital(*[int(v) for v in row]) for row in d["orbitals"]]
    for n, u in enumerate(orbs):
        if u.id != n:
            raise OrbitalError("orbital ids are not 0..D-1 in order")
    D = len(orbs)
    c = np.zeros((D, D, D), dtype=np.int64)
    for u, s, t, v in d["c"]:
        c[u, s, t] = int(v)
    sets = {}
    for s in d.get("sets", []):
        sets[os.path.basename(s["path"])] = (int(s["n"]), [int(v) for v in s["triples"]])
    O = Orbitals(
        N=int(d["N"]), x=int(d["x"]), valencies=tuple(int(v) for v in d["valencies"]),
        reps=tuple(int(v) for v in d["representatives"]), orbitals=orbs, c=c, sets=sets, source=d,
    )
    check_orbitals(O)
    return O


# --------------------------------------------------------------------------- checks


def check_orbitals(O: Orbitals) -> None:
    """Re-verify, in exact arithmetic, everything the SDP relies on.
    Raises OrbitalError with the first failure."""
    v, D = O.valencies, O.D
    if sum(v) != O.N or v[IDENTITY] != 1:
        raise OrbitalError("valencies")
    # sizes: |u| = v_i * osize; sum over orbitals with first class i of |u| = v_i * N
    for u in O.orbitals:
        if u.size != v[u.i] * u.osize or u.osize <= 0:
            raise OrbitalError(f"orbital {u.id}: size {u.size} != v_i * osize")
    for i in range(NCLASS):
        if sum(u.osize for u in O.orbitals if u.i == i) != O.N:
            raise OrbitalError(f"class {i}: orbit sizes do not sum to N")
        for j in range(NCLASS):
            if sum(u.osize for u in O.orbitals if u.i == i and u.j == j) != v[j]:
                raise OrbitalError(f"(i,j)=({i},{j}): orbit sizes do not sum to v_j")
    # the diagonal orbitals exist and have osize 1; the identity orbital (x, x)
    for k in range(NCLASS):
        dk = O.diag(k)
        if dk.osize != 1 or dk.j != k or dk.k != IDENTITY:
            raise OrbitalError(f"diagonal orbital of class {k} malformed")
    # S_3 action: involutions, sizes, class patterns
    for u in O.orbitals:
        t, s = O.orbitals[u.transpose], O.orbitals[u.swap]
        if t.transpose != u.id or s.swap != u.id:
            raise OrbitalError(f"orbital {u.id}: transpose/swap not involutive")
        if t.size != u.size or s.size != u.size:
            raise OrbitalError(f"orbital {u.id}: S_3 image has a different size")
        if (t.i, t.j, t.k) != (u.j, u.i, u.k) or (s.i, s.j, s.k) != (u.i, u.k, u.j):
            raise OrbitalError(f"orbital {u.id}: S_3 image has the wrong class pattern")
    # structure constants: support, row/column sums, the |u| c^u_st = |s| c^s_{u t^T} identity
    c = O.c
    I = np.array([u.i for u in O.orbitals])
    J = np.array([u.j for u in O.orbitals])
    T = np.array([u.transpose for u in O.orbitals])
    size = np.array([u.size for u in O.orbitals], dtype=np.int64)
    osize = np.array([u.osize for u in O.orbitals], dtype=np.int64)
    # support: c[u,s,t] != 0 only if i(s) = i(u), j(t) = j(u), j(s) = i(t)
    supp = (I[:, None, None] == I[None, :, None]) & (J[:, None, None] == J[None, None, :]) & (J[None, :, None] == I[None, None, :])
    if np.any((c != 0) & ~supp):
        raise OrbitalError("structure constant outside its support")
    if np.any(c < 0):
        raise OrbitalError("negative structure constant")
    rows = c.sum(axis=2)                       # [u, s]
    exp_rows = np.where(I[:, None] == I[None, :], osize[None, :], 0)
    if np.any(rows != exp_rows):
        raise OrbitalError("row sums of c are not |s|/v_i")
    cols = c.sum(axis=1)                       # [u, t]
    exp_cols = np.where(J[:, None] == J[None, :], (size // np.array(v)[J])[None, :], 0)
    if np.any(cols != exp_cols):
        raise OrbitalError("column sums of c are not |t|/v_j")
    lhs = size[:, None, None] * c
    rhs = np.transpose(size[:, None, None] * c[:, :, T], (1, 0, 2))   # rhs[u,s,t] = |s| c[s,u,T[t]]
    if np.any(lhs != rhs):
        raise OrbitalError("|u| c^u_st != |s| c^s_{u t^T}")
    # the identity relation is the union of the 7 diagonal orbitals d_k = (y_k, y_k):
    # sum_k B_{d_k} = I, so sum_k c[u, d_k, t] = [t == u] and sum_k c[u, s, d_k] = [s == u]
    dg = [O.diag(k).id for k in range(NCLASS)]
    if np.any(c[:, dg, :].sum(axis=1) != np.eye(D, dtype=np.int64)) or np.any(c[:, :, dg].sum(axis=2) != np.eye(D, dtype=np.int64)):
        raise OrbitalError("the diagonal orbitals do not sum to the unit of the algebra")
    # associativity of the algebra, B_s (B_t B_r) = (B_s B_t) B_r:
    #   sum_u c[u,t,r] c[w,s,u] = sum_u c[u,s,t] c[w,u,r]   for all s, t, r, w
    # (checked per s: D^3 work each, D^4 = 4.8e8 in total; no D^4 tensor is materialised)
    # float64 BLAS products are exact here: every partial sum is < D * max(c)^2 < 2^53.
    if int(c.max()) ** 2 * D >= 2 ** 53:
        raise OrbitalError("structure constants too large for the exact float64 associativity check")
    cf = c.astype(np.float64)
    cflat = cf.reshape(D, D * D)                                  # [u, (t, r)]
    cT = np.ascontiguousarray(np.transpose(cf, (0, 2, 1))).reshape(D * D, D)   # [(w, r), u] = c[w, u, r]
    for s in range(D):
        cs = np.ascontiguousarray(cf[:, s, :])                     # cs[u, t] = c[u, s, t]
        left = (cs @ cflat).reshape(D, D, D)                       # left[w,t,r] = sum_u c[w,s,u] c[u,t,r]
        right = np.transpose((cT @ cs).reshape(D, D, D), (0, 2, 1))  # right[w,t,r] = sum_u c[w,u,r] c[u,s,t]
        if not np.array_equal(left, right):
            raise OrbitalError(f"structure constants are not associative (s = {s})")
    # the recorded checks of the tool
    ch = (O.source or {}).get("checks", {})
    if any(int(val) != 0 for val in ch.values()):
        raise OrbitalError(f"tool-side checks failed: {ch}")
    if int((O.source or {}).get("orbit_of_x", 0)) != O.N or int((O.source or {}).get("stab_x_orbits", 0)) != NCLASS:
        raise OrbitalError("transitivity / class-transitivity record missing")


def check_against_triples(O: Orbitals, triples_path: str | os.PathLike = DEFAULT_TRIPLES_JSON) -> dict:
    """Cross-check the group-theoretic structure constants against the GPU
    brute-force 4-point histograms, and establish labels == orbitals:
      * every orbital's class-marginal histogram
            H_u(a,b,c) = sum_{s: j(s)=a, k(s)=b} sum_{t: k(t)=c} c[u,s,t]
        must be one of the recorded variants of the cell (i,j,k) of u, with
        the variant's z-count equal to the sum of osize over the orbitals of
        that cell mapping to it;
      * D == total number of variants (the invariant lower bound).
    Returns a summary dict; raises OrbitalError on any mismatch."""
    with open(triples_path) as f:
        T = json.load(f)
    if int(T["x"]) != O.x or [int(r) for r in T["representatives"]] != list(O.reps):
        raise OrbitalError("triples.json was computed for a different base point / representatives")
    if not T.get("checks_ok") or not T.get("reps_agree"):
        raise OrbitalError("triples.json failed its own checks")
    cells = {tuple(cell["ijk"]): cell for cell in T["cells"]}
    J = [u.j for u in O.orbitals]
    K = [u.k for u in O.orbitals]
    matched = {}
    for u in O.orbitals:
        H = {}
        cu = O.c[u.id]
        nz = np.nonzero(cu)
        for s, t in zip(*nz):
            key = (J[s], K[s], K[t])
            H[key] = H.get(key, 0) + int(cu[s, t])
        cell = cells.get((u.i, u.j, u.k))
        if cell is None:
            raise OrbitalError(f"orbital {u.id}: cell {(u.i, u.j, u.k)} missing from triples.json")
        hit = None
        for vi, var in enumerate(cell["variants"]):
            q = {(a, b, cc): int(n) for a, b, cc, n in var["q"]}
            if q == H:
                hit = vi
                break
        if hit is None:
            raise OrbitalError(f"orbital {u.id}: its 4-point histogram matches no GPU variant of cell {(u.i, u.j, u.k)}")
        matched.setdefault((u.i, u.j, u.k, hit), []).append(u)
    for (i, j, k, vi), us in matched.items():
        zc = int(cells[(i, j, k)]["variants"][vi]["z_count"])
        if sum(u.osize for u in us) != zc:
            raise OrbitalError(f"cell {(i,j,k)} variant {vi}: z-count {zc} != sum of orbit sizes")
    # every variant of every cell must be hit
    for (i, j, k), cell in cells.items():
        for vi in range(len(cell["variants"])):
            if (i, j, k, vi) not in matched:
                raise OrbitalError(f"cell {(i,j,k)} variant {vi} is not the histogram of any orbital")
    total_variants = sum(len(cell["variants"]) for cell in cells.values())
    if total_variants != int(T["total_variants"]):
        raise OrbitalError("total_variants inconsistent")
    if total_variants != O.D:
        raise OrbitalError(f"D = {O.D} labels but {total_variants} variants: labels are not proven to be the orbitals")
    split = {ijk: [int(v["z_count"]) for v in cell["variants"]] for ijk, cell in cells.items() if len(cell["variants"]) > 1}
    return {
        "cells": len(cells), "variants": total_variants, "D": O.D, "split_cells": split,
        "regular": bool(T["regular"]), "labels_are_orbitals": True,
    }


def check_against_scheme(O: Orbitals, scheme_path: str | os.PathLike = DEFAULT_SCHEME_JSON) -> None:
    """Orbit sizes per cell must sum to the intersection numbers p^i_{jk}."""
    with open(scheme_path) as f:
        S = json.load(f)
    p = S["p"]
    if tuple(int(v) for v in S["valencies"]) != O.valencies:
        raise OrbitalError("valencies differ from intersection_numbers.json")
    for i in range(NCLASS):
        for j in range(NCLASS):
            for k in range(NCLASS):
                tot = sum(u.osize for u in O.by_class(i, j, k))
                if tot != int(p[i][j][k]):
                    raise OrbitalError(f"cell {(i,j,k)}: orbit sizes {tot} != p^i_jk = {p[i][j][k]}")


# --------------------------------------------------------------------------- triple orbits (S_3)


def triple_orbits(O: Orbitals) -> tuple[list[int], list[list[int]]]:
    """Orbits of S_3 = <transpose, swap> on the orbitals = Gamma-orbits of
    unordered triples {x,y,z} (with repetitions).  Returns (tv, members):
    tv[u] = triple-orbit index of orbital u; members[q] = orbitals of orbit q."""
    D = O.D
    parent = list(range(D))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u in O.orbitals:
        for w in (u.transpose, u.swap):
            a, b = find(u.id), find(w)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = sorted({find(u) for u in range(D)})
    idx = {r: q for q, r in enumerate(roots)}
    tv = [idx[find(u)] for u in range(D)]
    members = [[] for _ in roots]
    for u in range(D):
        members[tv[u]].append(u)
    # consistency: all members of a triple orbit have the same size and the same multiset of classes
    for ms in members:
        sizes = {O.orbitals[u].size for u in ms}
        cls = {tuple(sorted((O.orbitals[u].i, O.orbitals[u].j, O.orbitals[u].k))) for u in ms}
        if len(sizes) != 1 or len(cls) != 1:
            raise OrbitalError("triple orbit with inconsistent members")
    return tv, members


def set_point(O: Orbitals, name: str) -> tuple[int, list[Fraction]]:
    """The SDP point of a recorded set S: x_u = n_O(S) / (|S| |u|), with the
    S_3 consistency of the counts checked.  Returns (|S|, [x_u])."""
    if name not in O.sets:
        raise OrbitalError(f"set {name} not recorded in orbitals.json (have {sorted(O.sets)})")
    n, cnt = O.sets[name]
    if sum(cnt) != n ** 3:
        raise OrbitalError("triple counts do not sum to |S|^3")
    for u in O.orbitals:
        if cnt[u.transpose] != cnt[u.id] or cnt[u.swap] != cnt[u.id]:
            raise OrbitalError(f"triple counts of {name} are not S_3-symmetric at orbital {u.id}")
    return n, [Fraction(cnt[u.id], n * u.size) for u in O.orbitals]


def describe(O: Orbitals) -> str:
    lines = [f"orbitals of Stab(x), x = {O.x}: D = {O.D}; per class i: " +
             ", ".join(f"{i}:{sum(1 for u in O.orbitals if u.i == i)}" for i in range(NCLASS))]
    tv, members = triple_orbits(O)
    lines.append(f"S_3-orbits (Gamma-orbits of unordered triples): {len(members)}")
    for u in O.orbitals:
        if (u.i, u.j, u.k) == (3, 3, 3):
            lines.append(f"  orthogonal-triple orbital {u.id}: z = {u.z}, orbit size {u.osize}, |u| = {u.size}")
    return "\n".join(lines)


if __name__ == "__main__":
    O = load_orbitals()
    print(describe(O))
    print("exact checks: sizes, S_3 action, support/row/column sums, symmetry, unit, associativity: OK")
    check_against_scheme(O)
    print("orbit sizes per cell = intersection numbers p^i_jk: OK")
    r = check_against_triples(O)
    print(f"GPU 4-point histograms: every orbital matches a variant, z-counts agree; D = {r['D']} = variants = {r['variants']} "
          f"(cells {r['cells']}, split cells {r['split_cells']}) -> labels are the orbitals: {r['labels_are_orbitals']}")
    for name in O.sets:
        n, xpt = set_point(O, name)
        print(f"set {name}: |S| = {n}, x_e = {xpt[O.diag(IDENTITY).id]}, sum_i v_i x_diag_i = {sum(O.valencies[k] * xpt[O.diag(k).id] for k in range(NCLASS))}")
    print(f"RESULT ok=1 D={O.D} variants={r['variants']} triple_orbits={len(triple_orbits(O)[1])} labels_are_orbitals=1")

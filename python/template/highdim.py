"""The CJKT / PackingStar lifting template above dimension 31 (PLAN T5.2).

Question: does the template

    K(24 + d)  >=  |E|  +  196560  +  sum_i (|T_i| - 1) * |S_i|

beat the recorded lower bound in any dimension n = 24 + d with n >= 32?

The three ingredients and their hard caps (README section 1.3, T5.1):

* ``T_1..T_g`` are disjoint groups of a spherical 60-degree code in R^d with pairwise
  cos <= -1/2 inside a group, so |T_i| <= 3 and the weight of a group is |T_i| - 1 <= 2.
  Over a K-point configuration the total weight is at most

      weight_upper_bound(K) = 2*floor(K/3) + [K mod 3 == 2]

  (T5.1; a counting bound valid for every K-point set, lattice or not).  A configuration
  with an Eisenstein (Z[omega]) structure attains 2K/3 exactly: its minimal vectors split
  into K/6 hexagons {+-r, +-wr, +-w^2 r} and each hexagon splits into the two triangles
  (r, wr, w^2 r) and (-r, -wr, -w^2 r).  E8 (K = 240 -> 80 triangles) is the case used here.
* ``E`` is a 60-degree code in R^d (at >= 30 degrees from every T-vector), so |E| <= K(d).
* **The S_i are pairwise DISJOINT 60-degree-free subsets of the 196560 Leech minimal
  vectors.**  This is the constraint that decides the whole question above n = 31:

      sum_i |S_i| <= 196560,  |S_i| <= 496 (best known),  so
      lifted term <= 2 * 196560 = 393120  and  at most 396 groups can hold a 496-set.

  T4.2 constructed 42 pairwise disjoint 496-sets (one Co_0 element of order 42); nothing
  larger has been constructed, so every count is reported twice: at ``n_sets = 42``
  (what exists) and at the theoretical ceiling (unlimited disjoint sets, only the
  196560-vector budget binding).

Everything here is arithmetic on published numbers plus one exact integer construction
(E8), so it is cheap; ``analyse()`` is the entry point and ``__main__`` prints the table.
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import small_kissing as sk  # noqa: E402

__all__ = [
    "LEECH", "BEST_S", "CONSTRUCTED_SETS", "MAX_SETS", "LIFT_CEILING",
    "COHN_LOWER", "COHN_UPPER", "COHN_SOURCE", "ECHOLS_2026",
    "record_lower", "weight_upper_bound", "max_lifted", "template_value",
    "sets_needed", "row", "analyse", "max_lifted_from_sizes", "KKW_SIZES", "KKW_TOTAL", "kkw_family_sizes", "e8_roots", "e8_eisenstein_sigma",
    "eisenstein_triangle_partition", "eisenstein_extra_spheres", "apply_sigma",
]

LEECH = 196560          # K(24) = |C|
BEST_S = 496            # largest known 60-degree-free subset of C (PackingStar 2025)
CONSTRUCTED_SETS = 42   # pairwise disjoint 496-sets we can actually build (T4.2)

# The largest EXPLICIT pairwise-disjoint family in the literature or in this repository:
# Kallal-Kan-Wang 2018 (data/external/Kissing-Numbers/S_1..S_59.txt) -- 59 pairwise disjoint
# 60-degree-free subsets of C, sizes 488 down to 460, total 28324 vectors.  Verified here
# (kkw_family_sizes): all in C, all independent, pairwise disjoint.  T4.2's 42 x 496 = 20832
# is larger per set but smaller in total, so 28324 is the best constructed lifting budget.
KKW_SIZES: tuple[int, ...] = (
    (488,) * 24 + (486,) + (484,) * 4 + (482,) * 4 + (480,) * 4 + (478,) * 3 + (476,)
    + (474,) * 5 + (472,) + (468,) * 6 + (466,) * 2 + (464,) * 2 + (462,) + (460,))
KKW_TOTAL = sum(KKW_SIZES)          # 28324
MAX_SETS = LEECH // BEST_S          # 396: sum_i |S_i| <= 196560
LIFT_CEILING = 2 * LEECH            # 393120: every group has weight <= 2


# ---------------------------------------------------------------------------
# published bounds
# ---------------------------------------------------------------------------
# Henry Cohn's table, https://cohn.mit.edu/kissing-numbers/, fetched 2026-08-27.
# (lower, upper) and the source of the lower bound.
COHN_LOWER: dict[int, int] = {
    1: 2, 2: 6, 3: 12, 4: 24, 5: 40, 6: 72, 7: 126, 8: 240,
    9: 306, 10: 510, 11: 604, 12: 841, 13: 1154, 14: 1932, 15: 2564, 16: 4320,
    17: 5730, 18: 7654, 19: 11948, 20: 19448, 21: 29768, 22: 49896, 23: 93150,
    24: 196560,
    25: 197056, 26: 198550, 27: 200044, 28: 204520, 29: 209496, 30: 220440,
    31: 238350,
    32: 345408, 33: 360640, 34: 380868, 35: 409548, 36: 484568, 37: 494312,
    38: 566652, 39: 755988, 40: 1064368, 41: 1170384, 42: 1250676, 43: 2060399,
    44: 2948552, 45: 3047160, 46: 5318060, 47: 9741412, 48: 52416000,
}
COHN_UPPER: dict[int, int] = {
    24: 196560, 25: 265006, 26: 367775, 27: 522212, 28: 752292, 29: 1075991,
    30: 1537707, 31: 2213487, 32: 3162316, 33: 4494570, 34: 6422593,
    35: 9162403, 36: 13017098, 37: 18498316, 38: 26496684, 39: 37826766,
    40: 53589200, 41: 76287040, 42: 108404055, 43: 153813582, 44: 220788272,
    45: 316735249, 46: 441900184, 47: 621658419, 48: 867897072,
}
COHN_SOURCE: dict[int, str] = {
    8: "Korkine-Zolotareff 1873", 9: "Leech-Sloane 1971", 10: "Ganzhinov 2025",
    11: "Bianchi et al. 2026", 12: "Takhanov et al. 2026", 13: "Zinoviev-Ericson 1999",
    14: "Ganzhinov 2025", 15: "Leech-Sloane 1971", 16: "Barnes-Wall 1959",
    17: "Cohn-Li 2024", 18: "Cohn-Li 2024", 19: "Ho 2026", 20: "Cohn-Li 2024",
    21: "Cohn-Li 2024", 22: "Leech 1967", 23: "Leech 1967", 24: "Leech 1967",
    25: "Ma et al. 2025", 26: "Ma et al. 2025", 27: "Ma et al. 2025",
    28: "Ma et al. 2025", 29: "Ma et al. 2025", 30: "Ma et al. 2025",
    31: "Ma et al. 2025",
    32: "Brouwer / Edel-Rains-Sloane 1998", 33: "Brouwer / Edel-Rains-Sloane 1998",
    34: "Brouwer / Edel-Rains-Sloane 1998", 35: "Brouwer / Edel-Rains-Sloane 1998",
    36: "Brouwer / Edel-Rains-Sloane 1998", 37: "Brouwer / Edel-Rains-Sloane 1998",
    38: "Brouwer / Edel-Rains-Sloane 1998", 39: "Brouwer / Edel-Rains-Sloane 1998",
    40: "Brouwer / Edel-Rains-Sloane 1998", 41: "Brouwer / Edel-Rains-Sloane 1998",
    42: "Brouwer / Edel-Rains-Sloane 1998", 43: "Sun-Wang 2026",
    44: "Edel-Rains-Sloane 1998", 45: "Brouwer / Edel-Rains-Sloane 1998",
    46: "Brouwer / Edel-Rains-Sloane 1998", 47: "Brouwer / Edel-Rains-Sloane 1998",
    48: "Leech-Sloane 1971",
}

# arXiv:2608.13906 (W. Echols, 2026-08-14), Table 2: improved constant-weight codes A(n,8,8)
# fed into Edel-Rains-Sloane's bound
#     tau_n >= 2^17 + 2^7 * A(n,8,8) + 2n(n-1)        (n >= 32)
# (2^17 = A(32,8) from the Cheng-Sloane [32,17,8] code; 2n(n-1) = 2^2 * C(n,2)).
# Not yet on Cohn's table (fetched 2026-08-27), so these are the current records for
# n = 32, 33, 34, 37.  The Cohn/Brouwer values in COHN_LOWER are the same formula with
# the older A(n,8,8) = 1659, 1777, 1934, 2817.
ECHOLS_2026: dict[int, int] = {32: 346432, 33: 362048, 34: 381124, 37: 496232}


def record_lower(n: int, include_echols: bool = True) -> tuple[int, str]:
    """Best recorded lower bound for K(n) and its source."""
    best, src = COHN_LOWER[n], COHN_SOURCE.get(n, "")
    if include_echols and n in ECHOLS_2026 and ECHOLS_2026[n] > best:
        best, src = ECHOLS_2026[n], "Echols 2026 (arXiv:2608.13906)"
    return best, src


# ---------------------------------------------------------------------------
# the template arithmetic
# ---------------------------------------------------------------------------
def weight_upper_bound(K: int) -> int:
    """max sum (|T_i| - 1) over disjoint groups of size <= 3 in a K-point set (T5.1)."""
    return 2 * (K // 3) + (1 if K % 3 == 2 else 0)


def max_lifted(K: int, n_sets: int | float = math.inf, set_size: int = BEST_S,
               budget: int = LEECH) -> dict:
    """Maximum of sum_i (|T_i| - 1)*|S_i| given a K-point R^d configuration.

    ``n_sets`` disjoint 60-degree-free subsets of C are available, each of size at most
    ``set_size``, and their sizes sum to at most ``budget`` (they are disjoint subsets of
    the 196560 minimal vectors).  Groups are taken weight-first (triangles before pairs),
    which is optimal because every group takes at most ``set_size`` from the budget.
    """
    n_tri = K // 3
    n_pair = 1 if K % 3 == 2 else 0
    groups = [2] * n_tri + [1] * n_pair
    lifted = used_sets = used_budget = 0
    tri_used = pair_used = 0
    for w in groups:
        if used_sets >= n_sets or used_budget >= budget:
            break
        s = min(set_size, budget - used_budget)
        lifted += w * s
        used_sets += 1
        used_budget += s
        if w == 2:
            tri_used += 1
        else:
            pair_used += 1
    return {"lifted": lifted, "weight": 2 * tri_used + pair_used, "sets_used": used_sets,
            "vectors_used": used_budget, "triangles_used": tri_used, "pairs_used": pair_used,
            "weight_bound": weight_upper_bound(K), "triangles_available": n_tri}


def max_lifted_from_sizes(K: int, sizes) -> dict:
    """Lifted term for an EXPLICIT family of disjoint sets of the given sizes.

    Largest sets go to the heaviest groups (rearrangement inequality, T5.1 assign.py).
    """
    n_tri = K // 3
    n_pair = 1 if K % 3 == 2 else 0
    weights = [2] * n_tri + [1] * n_pair
    sz = sorted(sizes, reverse=True)
    lifted = sum(w * s for w, s in zip(weights, sz))
    used = min(len(weights), len(sz))
    return {"lifted": lifted, "sets_used": used, "sets_available": len(sz),
            "vectors_used": sum(sz[:used]), "weight": sum(weights[:used])}


def template_value(d: int, K: int | None = None, n_sets: int | float = math.inf,
                   extras: int | None = None) -> dict:
    """Template count in dimension n = 24 + d: |E| + 196560 + lifted."""
    if K is None:
        K = COHN_LOWER[d]
    if extras is None:
        extras = K                      # |E| <= K(d); K(d) is the generous choice
    res = max_lifted(K, n_sets=n_sets)
    res.update({"d": d, "n": 24 + d, "K": K, "extras": extras,
                "value": extras + LEECH + res["lifted"]})
    return res


def sets_needed(d: int, target: int, K: int | None = None) -> dict:
    """How many disjoint 496-sets the template needs to reach ``target`` in n = 24 + d.

    Returns ``feasible`` = whether that many weight-2 groups even exist in a K-point
    configuration (and whether the 196560-vector budget allows them).
    """
    if K is None:
        K = COHN_LOWER[d]
    need_lifted = target - LEECH - K
    if need_lifted <= 0:
        return {"sets": 0, "feasible": True, "need_lifted": need_lifted}
    k = -(-need_lifted // (2 * BEST_S))          # ceil, all groups triangles
    return {"sets": k, "feasible": k <= min(K // 3, MAX_SETS),
            "need_lifted": need_lifted, "triangles_available": K // 3,
            "max_sets": MAX_SETS}


def row(d: int, n_sets: int | float = CONSTRUCTED_SETS) -> dict:
    """One line of the comparison table for n = 24 + d."""
    n = 24 + d
    K = COHN_LOWER[d]
    rec, src = record_lower(n)
    at_k = template_value(d, K=K, n_sets=n_sets)
    ceil_ = template_value(d, K=K, n_sets=math.inf)
    kkw = max_lifted_from_sizes(K, KKW_SIZES)
    return {
        "n": n, "d": d, "K": K, "K_source": COHN_SOURCE.get(d, ""),
        "triangles_available": K // 3, "weight_bound": weight_upper_bound(K),
        "extras": K,
        "n_sets": n_sets, "weight_at_n_sets": at_k["weight"], "value_at_n_sets": at_k["value"],
        "weight_ceiling": ceil_["weight"], "sets_at_ceiling": ceil_["sets_used"],
        "value_ceiling": ceil_["value"],
        "lifted_kkw": kkw["lifted"], "sets_kkw": kkw["sets_used"],
        "value_kkw": K + LEECH + kkw["lifted"],
        "record": rec, "record_source": src,
        "diff_at_n_sets": at_k["value"] - rec, "diff_ceiling": ceil_["value"] - rec,
        "diff_kkw": K + LEECH + kkw["lifted"] - rec,
        "sets_to_beat": sets_needed(d, rec + 1, K=K),
        "absolute_ceiling": K + LEECH + LIFT_CEILING,
    }


def analyse(d_range=range(8, 25), n_sets: int | float = CONSTRUCTED_SETS) -> list[dict]:
    return [row(d, n_sets=n_sets) for d in d_range]


def crossover(d_range=range(8, 25)) -> int | None:
    """Smallest n >= 32 from which the template ceiling never beats the record again."""
    rows = analyse(d_range)
    last_win = None
    for r in rows:
        if r["diff_ceiling"] > 0:
            last_win = r["n"]
    if last_win is None:
        return rows[0]["n"] if rows else None
    for r in rows:
        if r["n"] > last_win:
            return r["n"]
    return None


# ---------------------------------------------------------------------------
# the largest explicitly constructed disjoint family (Kallal-Kan-Wang 2018)
# ---------------------------------------------------------------------------
def kkw_family_sizes(root: str | None = None, verify: bool = True) -> dict:
    """Read data/external/Kissing-Numbers/S_*.txt and verify the disjoint family.

    Each file is a flat list of 24 integer coordinates per vector.  Checks: squared norm
    32 (our Leech scaling), no two vectors in a set at 60 degrees (Gram off-diagonal <= 8),
    and pairwise disjointness across all 59 sets.
    """
    import glob
    import re
    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                            "data", "external", "Kissing-Numbers")
    files = sorted(glob.glob(os.path.join(root, "S_*.txt")),
                   key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    sets = [np.array(open(f).read().split(), dtype=np.int64).reshape(-1, 24) for f in files]
    sizes = [len(V) for V in sets]
    out = {"files": len(files), "sizes": sizes, "total": sum(sizes), "ok": True, "why": ""}
    if not verify:
        return out
    for V in sets:
        if not np.all((V * V).sum(axis=1) == 32):
            out.update(ok=False, why="norm")
            return out
        G = V @ V.T
        np.fill_diagonal(G, 0)
        if G.max() > 8:
            out.update(ok=False, why="60-degree pair inside a set")
            return out
    keys = [set(map(tuple, V.tolist())) for V in sets]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] & keys[j]:
                out.update(ok=False, why=f"sets {i} and {j} overlap")
                return out
    return out


# ---------------------------------------------------------------------------
# the one explicit R^d configuration we need: E8 and its Eisenstein structure
# ---------------------------------------------------------------------------
def e8_roots() -> np.ndarray:
    """The 240 E8 roots, integer, squared norm 8 (small_kissing's scaling)."""
    return sk._e8_roots()


def e8_eisenstein_sigma(seed: int = 0) -> tuple[np.ndarray, int]:
    """An order-3 fixed-point-free isometry sigma of E8 with sigma^2 + sigma + 1 = 0.

    Returned as ``(M, den)`` with ``sigma = M / den`` (den = 2: an E8 automorphism is only
    half-integral in the ambient coordinates, because E8 does not contain Z^8).

    Built from four mutually orthogonal A2 subsystems (E8 = A2^4 + tetracode glue): sigma
    rotates each A2 plane by 120 degrees.  That rotation lies in W(A2), which acts trivially
    on A2*/A2, so the glue is preserved and sigma is an automorphism of E8 -- checked here
    directly by mapping all 240 roots.
    """
    R = e8_roots()
    G = R @ R.T
    idx = {tuple(v): i for i, v in enumerate(R.tolist())}
    planes: list[np.ndarray] = []
    used = np.zeros(len(R), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(R))
    for i in order:
        if used[i]:
            continue
        for j in np.flatnonzero(G[i] == -4):
            if used[j]:
                continue
            r, s = R[i], R[j]
            hexagon = [r, s, -(r + s), -r, -s, r + s]
            hidx = [idx[tuple(h.tolist())] for h in hexagon]
            if used[hidx].any():
                continue
            if planes and any(np.any(P @ np.array([r, s]).T != 0) for P in planes):
                continue
            planes.append(np.array([r, s], dtype=np.int64))
            used[hidx] = True
            break
        if len(planes) == 4:
            break
    if len(planes) != 4:
        raise RuntimeError("could not find A2^4 in E8")
    # r -> s -> -(r+s) -> r is the 120-degree rotation of the plane (cos<r,s> = -1/2).
    B = np.vstack(planes).astype(np.float64)
    img = np.vstack([np.array([p[1], -(p[0] + p[1])]) for p in planes]).astype(np.float64)
    sigma = np.linalg.solve(B, img).T
    den = 2
    M = np.rint(den * sigma).astype(np.int64)
    I = np.eye(8, dtype=np.int64)
    assert np.allclose(den * sigma, M), "sigma is not half-integral"
    assert np.array_equal(M @ M.T, den * den * I), "sigma is not orthogonal"
    assert np.array_equal(M @ M + den * M + den * den * I, np.zeros((8, 8), dtype=np.int64)), \
        "sigma^2 + sigma + 1 != 0"
    return M, den


def apply_sigma(V: np.ndarray, sigma: tuple[np.ndarray, int]) -> np.ndarray:
    """sigma applied to integer rows, exactly (the result must stay integral)."""
    M, den = sigma
    W = (M @ np.asarray(V, dtype=np.int64).T).T
    assert np.all(W % den == 0), "sigma does not preserve the integer lattice of these rows"
    return W // den


def eisenstein_triangle_partition(V: np.ndarray, sigma) -> list[tuple[int, int, int]]:
    """Partition the K rows of V into K/3 triangles using the order-3 isometry sigma.

    Every sigma-orbit on the rows has size 3, and {v, sigma v, sigma^2 v} sums to zero
    (sigma^2 + sigma + 1 = 0), i.e. is a triangle (pairwise cos = -1/2).  This is the
    "hexagons, halved" partition: a hexagon {+-v, +-sv, +-s^2 v} gives two triangles.
    """
    V = np.asarray(V, dtype=np.int64)
    idx = {tuple(v): i for i, v in enumerate(V.tolist())}
    W1 = apply_sigma(V, sigma)
    W2 = apply_sigma(W1, sigma)
    seen: set[int] = set()
    out: list[tuple[int, int, int]] = []
    for i in range(len(V)):
        if i in seen:
            continue
        j = idx[tuple(W1[i].tolist())]
        k = idx[tuple(W2[i].tolist())]
        t = tuple(sorted((i, j, k)))
        if len(set(t)) != 3:
            raise RuntimeError("sigma has a short orbit on these rows")
        if np.any(V[list(t)].sum(axis=0) != 0):
            raise RuntimeError("orbit is not a triangle")
        seen.update(t)
        out.append(t)
    return out


def eisenstein_extra_spheres(V: np.ndarray, sigma) -> dict:
    """|E| = K extra spheres from the 30-degree rotation Rot = (sigma + 2)/sqrt(3).

    J = (2 sigma + 1)/sqrt(3) satisfies J^2 = -1 (from sigma^2 + sigma + 1 = 0) and is
    orthogonal, so Rot = cos(30)*I + sin(30)*J = (sigma + 2)/sqrt(3) is an isometry and
    E = Rot(V) is again a 60-degree code of size |V|.  The extra-sphere condition is
    cos(Rot v, w) <= sqrt(3)/2 for all v, w in V.  With N the common squared norm and
    a = <sigma v, w> + 2 <v, w> (an integer), <Rot v, w> = a / sqrt(3), so the condition is
    a <= 0 or 4 a^2 <= 3 * 3 N^2 -- and since E must also stay 30 degrees away on the other
    side (T is antipodal here) the two-sided form |a| <= 3N/2 is what is checked.
    """
    V = np.asarray(V, dtype=np.int64)
    N = int((V[0] * V[0]).sum())
    assert np.all((V * V).sum(axis=1) == N)
    A = apply_sigma(V, sigma) @ V.T + 2 * (V @ V.T)
    lim2 = 9 * N * N                      # (3N/2)^2 * 4
    ok = bool(np.all(4 * A * A <= lim2))
    chk = sk.check_kissing(V)
    return {"ok": ok and bool(chk["ok"]), "size": len(V), "max_abs_a": int(np.abs(A).max()),
            "limit_2a": 3 * N, "touching": int(np.sum(4 * A * A == lim2)),
            "code_ok": bool(chk["ok"])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt(rows: list[dict]) -> str:
    hdr = (f"{'n':>3} {'d':>3} {'K(d)':>7} {'tri':>5} {'wmax':>5} "
           f"{'w@42':>5} {'val@42':>9} {'val@kkw59':>10} {'w@ceil':>7} {'sets':>5} {'val@ceil':>9} "
           f"{'record':>10} {'diff@42':>10} {'diff@kkw':>10} {'diff@ceil':>11}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['n']:>3} {r['d']:>3} {r['K']:>7} {r['triangles_available']:>5} "
            f"{r['weight_bound']:>5} {r['weight_at_n_sets']:>5} {r['value_at_n_sets']:>9} "
            f"{r['value_kkw']:>10} "
            f"{r['weight_ceiling']:>7} {r['sets_at_ceiling']:>5} {r['value_ceiling']:>9} "
            f"{r['record']:>10} {r['diff_at_n_sets']:>+10} {r['diff_kkw']:>+10} {r['diff_ceiling']:>+11}")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    lo = int(argv[1]) if len(argv) > 1 else 8
    hi = int(argv[2]) if len(argv) > 2 else 24
    rows = analyse(range(lo, hi + 1))
    print(f"template ceiling anywhere = K(d) + {LEECH} + {LIFT_CEILING} = K(d) + {LEECH + LIFT_CEILING}")
    print(f"constructed disjoint 496-sets = {CONSTRUCTED_SETS}; absolute max = {MAX_SETS}\n")
    print(_fmt(rows))
    wins42 = [r["n"] for r in rows if r["diff_at_n_sets"] > 0]
    winskkw = [r["n"] for r in rows if r["diff_kkw"] > 0]
    winsc = [r["n"] for r in rows if r["diff_ceiling"] > 0]
    print(f"\nbeats the record at 42 x 496 = 20832 : {wins42 or 'NONE'}")
    print(f"beats the record at KKW 59 sets (28324) : {winskkw or 'NONE'}")
    print(f"beats the record at ceiling : {winsc or 'NONE'}")
    print(f"crossover (ceiling never wins again from) : n = {crossover(range(lo, hi + 1))}")
    for r in rows:
        s = r["sets_to_beat"]
        print(f"  n={r['n']:>3}: needs {s['sets']:>4} disjoint 496-sets to beat {r['record']}"
              f"  (triangles available {s.get('triangles_available', '-')}, feasible={s['feasible']})")

    kk = kkw_family_sizes()
    print(f"\nKallal-Kan-Wang disjoint family: {kk['files']} sets, total {kk['total']} vectors, "
          f"verified ok={kk['ok']} {kk['why']}")

    print("\n-- explicit R^8 configuration (E8) --")
    R = e8_roots()
    sigma = e8_eisenstein_sigma()
    tri = eisenstein_triangle_partition(R, sigma)
    pts = sorted(p for t in tri for p in t)
    ex = eisenstein_extra_spheres(R, sigma)
    print(f"E8 roots K=240, sigma order 3 fixed-point-free, hexagon partition: "
          f"{len(tri)} disjoint triangles covering {len(set(pts))}/240 points, "
          f"weight {2 * len(tri)} = counting bound {weight_upper_bound(240)}")
    print(f"extra spheres via R=(sigma+2I)/sqrt3: ok={ex['ok']} |E|={ex['size']} "
          f"max|a|={ex['max_abs_a']} 2*limit={ex['limit_2a']} touching={ex['touching']} (equality pairs = exact 30 deg)")
    print(f"RESULT wins42={len(wins42)} wins_ceiling={len(winsc)} crossover={crossover(range(lo, hi + 1))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

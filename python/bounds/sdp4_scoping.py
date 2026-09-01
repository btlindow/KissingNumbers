"""B2 — scoping study for a four-point (Lasserre level-3 / Schrijver-quadruple
style) upper bound on |S| for the Leech minimal-vector scheme.

This module does NOT solve a four-point SDP.  It answers, with data, the
question "how big would one be, and is it worth it?", from three inputs:

1. data/scheme/quad_orbits.json (tools/quad_stats + cuda/quad_stats.cu):
   for x = C[0] and one representative y_i per pair class, the orbits of
   Stab(x, y_i) on z and, per z-orbit representative z_r, the orbits of
   Stab(x, y_i, z_r) on w — each bracketed by the number of distinct GPU
   five-point histograms (a Stab(x,y,z)-invariant: a LOWER bound) and the
   orbit count of explicit random stabiliser elements (labels REFINE orbits:
   an UPPER bound); exact where the two agree, which is the same proof
   pattern that established D = 148 in T2.3.

       dim T(x, y_i) = # Stab(x, y_i)-orbitals on ordered pairs (z, w)
                     = sum over z-orbit reps r of #orbits of Stab(x,y_i,z_r) on w.

   This is the dimension of the centraliser algebra of Stab(x, y_i) — the
   algebra a four-point SDP's PSD blocks live in via the left-regular-
   representation trick of T2.3 (whose blocks were 148 x 148 because
   dim T(x) = 148).

2. data/scheme/orbitals.json (T2.3): the level-2 scheme, used for anchor
   checks (classes 0 and 6 have Stab(x, y) = Stab(x), so their dim T(x,y)
   must be exactly 148; z-orbit counts and sizes must reproduce the 148
   orbitals).

3. The three-point SDP itself (bounds.sdp3_scheme): re-solved here to read
   off the slack structure of the 837 optimum — which constraints are tight,
   how far the optimal x* is from the 496's moment vector — the empirical
   basis for guessing how much a level-3 bound could move.

Run:  PYTHONPATH=python .venv/bin/python -m bounds.sdp4_scoping
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Sequence

import numpy as np

from .orbitals import CONFLICT, NCLASS, load_orbitals

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUAD_JSON = os.path.join(_HERE, "..", "..", "data", "scheme", "quad_orbits.json")

# Pair classes that may occur inside an admissible S (dot 16 forbidden).
ALLOWED_PAIR_CLASSES = tuple(i for i in range(NCLASS) if i != CONFLICT)
# Unordered base-pair classes a four-point SDP needs blocks for: the class of
# (x, y) with x != y allowed inside S, up to y -> -y conjugacy (Stab(x, y) =
# Stab(x, -y), classes 1<->5 (5 forbidden anyway), 2<->4), plus the diagonal
# y = x (class 6) whose blocks are T2.3's.


def load_quad(path: str = DEFAULT_QUAD_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


class QuadError(ValueError):
    pass


def check_quad(q: dict, O=None) -> dict:
    """Anchor checks of quad_orbits.json against orbitals.json.  Raises
    QuadError on any failure; returns a summary dict."""
    if O is None:
        O = load_orbitals()
    if int(q["N"]) != O.N or int(q["x"]) != O.x:
        raise QuadError("quad_orbits.json computed for a different scheme/base point")
    if tuple(int(v) for v in q["valencies"]) != O.valencies:
        raise QuadError("valencies differ")
    if [int(v) for v in q["representatives"]] != list(O.reps):
        raise QuadError("class representatives differ from orbitals.json")
    if int(q["spot_checks"]["mismatches"]) != 0:
        raise QuadError("GPU spot checks failed")
    for key, val in q["checks"].items():
        if int(val) != 0:
            raise QuadError(f"tool-side check failed: {key} = {val}")
    by_i = {int(c["i"]): c for c in q["classes"]}
    if sorted(by_i) != list(range(NCLASS)):
        raise QuadError(f"not all 7 classes computed: {sorted(by_i)}")

    nlab_expected = [sum(1 for u in O.orbitals if u.i == i) for i in range(NCLASS)]
    for i, c in by_i.items():
        if int(c["z_orbit_count"]) != nlab_expected[i]:
            raise QuadError(f"class {i}: {c['z_orbit_count']} z-orbits, orbitals.json has {nlab_expected[i]}")
        # z-orbit (j, k, size) multiset must reproduce the orbitals of class i
        got = Counter((int(z["j"]), int(z["k"]), int(z["size"])) for z in c["z_orbits"])
        want = Counter((u.j, u.k, u.osize) for u in O.orbitals if u.i == i)
        if got != want:
            raise QuadError(f"class {i}: z-orbit cells/sizes do not match the orbitals")
        for z in c["z_orbits"]:
            if int(z["label_orbits"]) < int(z["hist_orbits"]):
                raise QuadError(f"class {i} z={z['z']}: labels < histograms (impossible)")
            if sum(int(w[1]) for w in z["w_orbits"]) != O.N:
                raise QuadError(f"class {i} z={z['z']}: w-orbit sizes do not sum to N")
        if sum(int(z["hist_orbits"]) for z in c["z_orbits"]) != int(c["dimT_hist"]):
            raise QuadError(f"class {i}: dimT_hist is not the sum over z-orbits")
        if sum(int(z["label_orbits"]) for z in c["z_orbits"]) != int(c["dimT_labels"]):
            raise QuadError(f"class {i}: dimT_labels is not the sum over z-orbits")

    # anchors: Stab(x, +-x) = Stab(x)  =>  dim T = D = 148 exactly
    for i in (0, 6):
        if int(by_i[i]["dimT_hist"]) != O.D or int(by_i[i]["dimT_labels"]) != O.D:
            raise QuadError(f"class {i}: dim T(x,y) = "
                            f"[{by_i[i]['dimT_hist']}, {by_i[i]['dimT_labels']}] != D = {O.D}")
    # anchors: y -> -y conjugates Stab(x, y_1) ~ Stab(x, y_5), Stab(x, y_2) ~ Stab(x, y_4).
    # Histogram counts are deterministic invariants, so they must agree exactly;
    # label counts are upper bounds from random subgroups, so only the brackets
    # [hist, labels] must overlap.
    for a, b in ((1, 5), (2, 4)):
        for key in ("dimT_hist", "z_orbit_count"):
            if int(by_i[a][key]) != int(by_i[b][key]):
                raise QuadError(f"classes {a}/{b}: {key} differ "
                                f"({by_i[a][key]} != {by_i[b][key]}) — antipodal conjugacy broken")
        lo = max(int(by_i[a]["dimT_hist"]), int(by_i[b]["dimT_hist"]))
        hi = min(int(by_i[a]["dimT_labels"]), int(by_i[b]["dimT_labels"]))
        if lo > hi:
            raise QuadError(f"classes {a}/{b}: dim T brackets do not overlap")
    exact = all(int(c["dimT_hist"]) == int(c["dimT_labels"]) for c in q["classes"])
    return {
        "classes": NCLASS,
        "dimT": {i: (int(by_i[i]["dimT_hist"]), int(by_i[i]["dimT_labels"])) for i in range(NCLASS)},
        "all_exact": exact,
        "anchors_ok": True,
    }


# --------------------------------------------------------------------- sizes


def scoping(q: dict) -> dict:
    """All the size/cost numbers of the four-point relaxation, from the data."""
    by_i = {int(c["i"]): c for c in q["classes"]}
    dimT = {i: int(by_i[i]["dimT_labels"]) for i in range(NCLASS)}
    dimT_lo = {i: int(by_i[i]["dimT_hist"]) for i in range(NCLASS)}

    # ---- variables: Co_0-orbits of admissible 4-tuples -------------------
    # Each Co_0-orbit of ordered 4-tuples (x,y,z,w) determines class(x,y) = i
    # and one Stab(x,y_i)-orbit of (z,w); summing dim T(x,y_i) over i counts
    # every ordered-4-tuple orbit exactly once (repetition patterns included:
    # y = x is i = 6, w = z is class(z,w) = 6, etc.).
    ordered = sum(dimT.values())
    ordered_lo = sum(dimT_lo.values())
    admissible = 0            # no pairwise class 5 among the six pairs
    admissible_distinct = 0   # ... and all four points distinct
    clique4 = 0               # all six pairs in class 5 (a 4-clique of the conflict graph)
    triangle_orbits = 0       # Stab(x,y_5)-orbits of z with (j,k) = (5,5): ordered conflict triangles
    for i in range(NCLASS):
        for z in by_i[i]["z_orbits"]:
            j, k = int(z["j"]), int(z["k"])
            if i == CONFLICT and j == CONFLICT and k == CONFLICT:
                triangle_orbits += 1
            for w in z["w_orbits"]:
                a, b, c, forb, dist = int(w[2]), int(w[3]), int(w[4]), int(w[5]), int(w[6])
                if not forb:
                    admissible += 1
                    if dist:
                        admissible_distinct += 1
                if i == j == k == a == b == c == CONFLICT:
                    clique4 += 1
    # S_4 symmetrisation: each unordered orbit collects between 1 and 24
    # ordered orbits, so the variable count is bracketed by
    # [ceil(admissible/24), admissible]; experience one level down (148
    # orbitals -> 43 S_3-orbits, ratio 3.44 vs |S_3| = 6) suggests the truth
    # sits well above the lower end for the degenerate patterns and near
    # /24 for the generic ones.
    vars_lo = math.ceil(admissible / 24)

    # ---- PSD blocks ------------------------------------------------------
    # T2.3 pattern one level up: per base configuration, one block for
    # "base in S" and one for "base not in S".  At level 3 the base is the
    # pair (x, y); membership splits into (x,y in S), (x in, y out), (both
    # out) -> 3 blocks of size dim T(x,y_i) per usable unordered pair class.
    # Classes 1~5 and 2~4 are conjugate (y -> -y), class 5 is forbidden for
    # the "both in" block; the diagonal class 6 reproduces T2.3.  A concrete
    # accounting for the class list {0,1,2,3,4} + diagonal:
    block_classes = [0, 1, 2, 3, 4]           # distinct base-pair geometries with y != x
    blocks = []
    for i in block_classes:
        blocks += [("class %d both-in" % i, dimT[i]), ("class %d one-in" % i, dimT[i]),
                   ("class %d both-out" % i, dimT[i])]
    blocks += [("diag (T2.3 M1)", 148), ("diag (T2.3 M2)", 148)]
    sum_d2 = sum(d * d for _, d in blocks)
    sum_d3 = sum(d ** 3 for _, d in blocks)
    max_d = max(d for _, d in blocks)

    # ---- costs -----------------------------------------------------------
    # Left-regular representation: needs the structure-constant tensor
    # c[u,s,t] of the centraliser algebra of Stab(x,y_i), u,s,t < dim T.
    # T2.3's tensor (D = 148) had 11382 nonzeros ~ D^2 * 0.52; the support
    # condition (three cell constraints) keeps it sparse; extrapolating the
    # same fill gives the estimate below.  Computing it costs one O(N) pass
    # per orbital u with two label lookups (as tools/triple_orbitals step 7).
    t23_fill = 11382 / 148 ** 2
    nnz_c_est = {i: int(t23_fill * dimT[i] ** 2) for i in block_classes}

    # Interior-point cost model (Clarabel).  Per iteration the PSD cones need
    # several dense O(d^3) operations per block (eig/Cholesky of the NT
    # scaling, congruences); forming the Schur complement over the Q variables
    # adds Q sparse congruences W G_q W per block.  Memory: the solver carries
    # ~8 vectors over the cone scalars sum d(d+1)/2 plus a few dense d x d
    # work matrices per block.
    psd_scalars = sum(d * (d + 1) // 2 for _, d in blocks)
    flops_iter = 10 * sum_d3
    iters = 60
    eff_flops = 5e10                             # ~what threaded LAPACK reaches on this box
    return {
        "dimT": dimT, "dimT_lower": dimT_lo,
        "ordered_orbits": ordered, "ordered_orbits_lower": ordered_lo,
        "admissible_ordered": admissible, "admissible_distinct4": admissible_distinct,
        "vars_bracket": (vars_lo, admissible),
        "triangle_orbits": triangle_orbits, "clique4_orbits": clique4,
        "blocks": blocks, "sum_d2": sum_d2, "sum_d3": sum_d3, "max_block": max_d,
        "nnz_c_est": nnz_c_est,
        "psd_scalars": psd_scalars,
        "mem_gb_est": (8 * psd_scalars * 8 + 4 * sum_d2 * 8) / 2 ** 30,
        "hours_est": flops_iter * iters / eff_flops / 3600,
    }


def print_scoping(q: dict, s: dict) -> None:
    print("dim T(x, y_i) per pair class (PSD block dimension of a four-point SDP):")
    print("  class  dot   z-orbits  dim T [hist, labels]  exact")
    by_i = {int(c["i"]): c for c in q["classes"]}
    dots = q["classes_dots"]
    for i in range(NCLASS):
        c = by_i[i]
        lo, hi = int(c["dimT_hist"]), int(c["dimT_labels"])
        print(f"  {i}     {dots[i]:4d}   {int(c['z_orbit_count']):5d}    [{lo}, {hi}]"
              f"{'':6s}{'yes' if lo == hi else 'NO'}")
    print(f"ordered 4-tuple orbits of Co_0 (all patterns): {s['ordered_orbits']}"
          f" (lower bound {s['ordered_orbits_lower']})")
    print(f"  admissible (no dot-16 pair): {s['admissible_ordered']}; "
          f"with 4 distinct points: {s['admissible_distinct4']}")
    print(f"  S_4-symmetrised variable count in [{s['vars_bracket'][0]}, {s['vars_bracket'][1]}]")
    print(f"  conflict-graph structure: {s['triangle_orbits']} ordered-triangle orbits, "
          f"{s['clique4_orbits']} ordered-4-clique orbits")
    print(f"PSD blocks ({len(s['blocks'])}): " +
          ", ".join(f"{name} {d}x{d}" for name, d in s['blocks']))
    print(f"  sum d^2 = {s['sum_d2']:.3e}, sum d^3 = {s['sum_d3']:.3e}, largest block {s['max_block']}")
    print(f"  structure-constant tensors (LRR trick), est. nonzeros per class: " +
          ", ".join(f"{i}: {n:.1e}" for i, n in s['nnz_c_est'].items()))
    print(f"  cone scalars sum d(d+1)/2 = {s['psd_scalars']:.2e}; interior-point memory "
          f"~ {s['mem_gb_est']:.1f} GB; time ~ {s['hours_est']:.0f} h (60 iterations, "
          f"10 sum d^3 flops each at 5e10/s) — EXCLUDING Schur-complement formation")


# --------------------------------------------- slack structure of the 837 run


def tight_constraints(solver: str = "CLARABEL", tol: float = 1e-6) -> dict:
    """Solve the three-point relaxation (forbidden = {16}) and report which
    constraints are tight at the numerical optimum — the empirical basis for
    guessing what a four-point bound could add."""
    from .sdp3_scheme import Sdp3

    S = Sdp3()
    res = S.solve((CONFLICT,), solver=solver)
    x = res.x
    free = S.free_orbits((CONFLICT,))
    zeroed = S.forbidden_orbits((CONFLICT,))
    at0 = [q for q in free if q != S.e and x[q] < tol]
    at1 = [q for q in free if q != S.e and x[q] > 1 - tol]
    interior = [q for q in free if q != S.e and tol <= x[q] <= 1 - tol]
    y = S.Cy @ x
    y_tight = int(np.sum(np.abs(y) < tol))
    eig = {}
    for tag, B in (("block1", S.Zp), ("block2", S.Zyp)):
        M = np.tensordot(x, B, axes=(0, 0))
        M = (M + M.T) / 2
        w = np.linalg.eigvalsh(M)
        scale = max(abs(w).max(), 1.0)
        eig[tag] = {
            "null": int(np.sum(w < 1e-7 * scale)),
            "dim": len(w),
            "min": float(w.min()),
            "max": float(w.max()),
        }
    # per-class diagonal profile: x_{diag(k)} = fraction of class-k pairs
    diag = {int(k): float(x[S.tv[S.dg[k]]]) for k in range(NCLASS)}
    # the 496's moment vector, for comparison of the pair-class profile
    n496, x496 = S.x_of_set("S496.txt")
    x496f = np.array([float(t) for t in x496])
    diag496 = {int(k): float(x496f[S.tv[S.dg[k]]]) for k in range(NCLASS)}
    return {
        "value": res.value, "status": res.status,
        "n_free": len(free), "n_forbidden": len(zeroed),
        "at_zero": at0, "at_one": at1, "interior": interior,
        "y_dim": len(y), "y_tight": y_tight,
        "eig": eig, "diag": diag, "diag496": diag496,
        "x": x, "x496": x496f, "dist_496": float(np.max(np.abs(x - x496f))),
        "S": S,
    }


def print_tight(t: dict) -> None:
    S = t["S"]
    print(f"three-point optimum: {t['value']:.6f} ({t['status']})")
    print(f"  variables: {t['n_free']} free (+{t['n_forbidden']} forced to 0 by the forbidden class)")
    print(f"  x at 0: {len(t['at_zero'])}  {t['at_zero']}")
    print(f"  x at 1: {len(t['at_one'])}  {t['at_one']}")
    print(f"  x interior: {len(t['interior'])}")
    print(f"  y = Cy x >= 0: {t['y_tight']} of {t['y_dim']} tight")
    for tag, e in t["eig"].items():
        print(f"  {tag}: {e['null']} (near-)zero eigenvalues of {e['dim']} "
              f"[min {e['min']:.2e}, max {e['max']:.2e}]")
    print("  x_diag per class (pair-class density: optimum vs the 496, and their ratio):")
    for k, v in t["diag"].items():
        v496 = t["diag496"][k]
        ratio = f"{v / v496:8.3f}" if v496 > 1e-12 else ("     inf" if v > 1e-6 else "       -")
        print(f"    class {k}: {v:.6f}   496: {v496:.6f}   ratio {ratio}")
    print(f"  max|x* - x(S496)| = {t['dist_496']:.4f}; the optimal pseudo-set is antipodal "
          "(x_diag(0) = 1), shares the 496's support (no dot -16 or +16 pairs) and is "
          "close to a |bound|/496 scaling of the 496's pair distribution")


# ------------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quad", default=DEFAULT_QUAD_JSON)
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--skip-sdp", action="store_true",
                    help="only the size scoping, no three-point re-solve")
    a = ap.parse_args(argv)

    q = load_quad(a.quad)
    O = load_orbitals()
    chk = check_quad(q, O)
    print(f"quad_orbits.json checks: anchors OK (classes 0/6 give dim T = {O.D}; "
          f"1~5 and 2~4 agree); all classes exact: {chk['all_exact']}")
    s = scoping(q)
    print_scoping(q, s)
    ok = chk["anchors_ok"]
    if not a.skip_sdp:
        t = tight_constraints(a.solver)
        print_tight(t)
        ok = ok and t["value"] is not None
    dims = ",".join(str(s["dimT"][i]) for i in range(NCLASS))
    print(f"RESULT ok={1 if ok else 0} dimT={dims} ordered={s['ordered_orbits']} "
          f"admissible={s['admissible_ordered']} vars_lo={s['vars_bracket'][0]} "
          f"max_block={s['max_block']} clique4={s['clique4_orbits']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

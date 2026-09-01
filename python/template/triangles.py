"""Triangle packings and group partitions of the R^d kissing configurations (PLAN T5.1 items 1-2).

Template recap (README 1.3): the R^d half is a kissing configuration T (unit vectors, pairwise
cos <= 1/2) partitioned into groups T_i with pairwise cos <= -1/2 inside a group; group i is worth
(|T_i| - 1)·|S_i| lifted vectors. Four unit vectors with pairwise ip <= -1/2 cannot exist
(|sum|^2 = 4 + 2·sum ip <= 4 - 6 < 0), so |T_i| <= 3: a group is a "triangle" (three coplanar
vectors at 120 degrees, weight 2) or a "pair" (cos <= -1/2, weight 1).

Two exact optimisation problems on a given configuration V (integer rows, common squared norm):

  max_triangle_packing(V)   max number of pairwise disjoint triangles (binary ILP, HiGHS);
  max_weight_partition(V)   max sum of weights over disjoint groups (triangles weight 2, admissible
                            pairs weight 1) — this is the quantity the template count depends on.

Both return the solver's proven optimality (mip_gap == 0, dual bound) and, independently, the
human-readable bounds:

  * weight <= 2·floor(K/3) + [K mod 3 == 2]   (counting: groups are disjoint subsets of size <= 3)
    — valid for EVERY K-point configuration, lattice or not (weight_upper_bound);
  * triangles <= floor(K/3), and if the configuration sums to zero (all six lattice models are
    antipodal) and K ≡ 1 (mod 3), triangles <= floor(K/3) - 1: each triangle sums to zero
    (|x+y+z|^2 = 3N + 2·3·(-N/2) = 0), so the leftover points sum to zero, and a single leftover
    point would be the zero vector (zero_sum_bound).

The continuous search (continuous_triangle_search) looks for n unit vectors in R^d containing k
prescribed triangles with all other pairwise cosines <= 1/2: SLSQP on the coordinates, triangle
constraints as equalities, min over restarts of the max non-triangle cosine; a value <= 1/2 + 1e-9
is a (numerical) kissing configuration with k triangles. It is only a sanity tool here: the counting
bound above shows the lattice configurations already attain the maximum weight for every d.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import small_kissing as sk  # noqa: E402

__all__ = ["triangles_of", "admissible_pairs", "max_triangle_packing", "max_weight_partition",
           "weight_upper_bound", "zero_sum_bound", "record_partition", "analyse", "continuous_triangle_search"]


# ---------------------------------------------------------------------------
# combinatorics on an integer configuration
# ---------------------------------------------------------------------------
def triangles_of(V: np.ndarray) -> list[tuple[int, int, int]]:
    """All index triples with pairwise cos exactly -1/2 (delegates to kiss_ref.small_kissing)."""
    return sk.triangles_of(V)


def admissible_pairs(V: np.ndarray) -> list[tuple[int, int]]:
    """All index pairs with cos <= -1/2 (antipodal pairs and the edges of triangles)."""
    V = np.asarray(V, dtype=np.int64)
    N = (V * V).sum(axis=1)
    G = V @ V.T
    ok = sk.cos_le_neg_half(G, N[:, None], N[None, :])
    return [(int(i), int(j)) for i, j in np.argwhere(np.triu(ok, 1))]


def _solve_packing(n: int, blocks: list[tuple[int, ...]], weights: np.ndarray, time_limit: float | None) -> dict:
    """max weights·x, x binary, each point in <= 1 chosen block. Returns solution + certificate data."""
    m = len(blocks)
    if m == 0:
        return {"value": 0, "chosen": [], "lp_bound": 0.0, "dual_bound": 0.0, "gap": 0.0, "nodes": 0,
                "status": "trivial", "seconds": 0.0}
    rows, cols = [], []
    for k, b in enumerate(blocks):
        for p in b:
            rows.append(p)
            cols.append(k)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))
    cons = LinearConstraint(A, -np.inf, 1.0)
    bounds = Bounds(0, 1)
    c = -weights.astype(float)
    t0 = time.time()
    lp = milp(c, constraints=cons, bounds=bounds, integrality=np.zeros(m))  # LP relaxation
    lp_bound = float(-lp.fun) if lp.success else float("nan")
    opts = {"disp": False}
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)
    res = milp(c, constraints=cons, bounds=bounds, integrality=np.ones(m), options=opts)
    if not res.success:
        raise RuntimeError(f"milp failed: {res.message}")
    x = np.rint(res.x).astype(int)
    chosen = [blocks[k] for k in np.flatnonzero(x)]
    # feasibility re-check independent of the solver
    used = np.zeros(n, dtype=int)
    for b in chosen:
        used[list(b)] += 1
    assert used.max() <= 1, "solver returned overlapping blocks"
    value = int(round(sum(weights[k] for k in np.flatnonzero(x))))
    assert abs(value + res.fun) < 1e-6
    dual = float(getattr(res, "mip_dual_bound", -res.fun))
    return {"value": value, "chosen": chosen, "lp_bound": lp_bound, "dual_bound": -dual if dual < 0 else dual,
            "gap": float(getattr(res, "mip_gap", 0.0)), "nodes": int(getattr(res, "mip_node_count", 0)),
            "status": str(res.message), "seconds": time.time() - t0}


def max_triangle_packing(V: np.ndarray, time_limit: float | None = None) -> dict:
    """Maximum number of pairwise disjoint triangles in V (exact ILP). Keys: value, chosen (triangles),
    leftover, lp_bound (= K/3 in every case here), dual_bound, gap (0 = proven), nodes, seconds."""
    V = np.asarray(V, dtype=np.int64)
    n = len(V)
    tris = triangles_of(V)
    out = _solve_packing(n, tris, np.ones(len(tris)), time_limit)
    covered = {p for t in out["chosen"] for p in t}
    out["leftover"] = [i for i in range(n) if i not in covered]
    out["n_triangles_total"] = len(tris)
    out["proven"] = out["gap"] <= 1e-9
    return out


def max_weight_partition(V: np.ndarray, time_limit: float | None = None) -> dict:
    """Maximum of sum_i (|T_i| - 1) over disjoint groups (triangles: 2, pairs with cos <= -1/2: 1).
    Keys: value (the weight), triangles, pairs, leftover, lp_bound, dual_bound, gap, nodes, seconds."""
    V = np.asarray(V, dtype=np.int64)
    n = len(V)
    tris = triangles_of(V)
    pairs = admissible_pairs(V)
    blocks = [tuple(t) for t in tris] + [tuple(p) for p in pairs]
    w = np.array([2.0] * len(tris) + [1.0] * len(pairs))
    out = _solve_packing(n, blocks, w, time_limit)
    out["triangles"] = [b for b in out["chosen"] if len(b) == 3]
    out["pairs"] = [b for b in out["chosen"] if len(b) == 2]
    covered = {p for b in out["chosen"] for p in b}
    out["leftover"] = [i for i in range(n) if i not in covered]
    out["n_admissible_pairs"] = len(pairs)
    out["proven"] = out["gap"] <= 1e-9
    return out


# ---------------------------------------------------------------------------
# bounds that need no solver
# ---------------------------------------------------------------------------
def weight_upper_bound(K: int) -> int:
    """max sum (|T_i|-1) over disjoint groups of size <= 3 among K points: 2·floor(K/3) + [K mod 3 == 2]."""
    return 2 * (K // 3) + (1 if K % 3 == 2 else 0)


def zero_sum_bound(V: np.ndarray) -> dict:
    """Upper bound on disjoint triangles: floor(K/3), minus 1 when the configuration sums to zero and
    K ≡ 1 (mod 3) (a single leftover point would have to be the zero vector)."""
    V = np.asarray(V, dtype=np.int64)
    K = len(V)
    sums_to_zero = bool(np.all(V.sum(axis=0) == 0))
    b = K // 3
    reason = "floor(K/3)"
    if sums_to_zero and K % 3 == 1:
        b -= 1
        reason = "floor(K/3) - 1: configuration sums to zero, triangles sum to zero, one leftover point impossible"
    return {"bound": b, "sums_to_zero": sums_to_zero, "reason": reason}


# ---------------------------------------------------------------------------
# the record's partitions (data/families/dim<n>/family.json)
# ---------------------------------------------------------------------------
def record_partition(d: int, families_dir: str | None = None) -> dict:
    """Groups of the record family for dimension 24 + d, mapped onto config(d) by exact matching of
    the T vectors (the record's T block IS config(d) up to row order; the extractor mapped it)."""
    import json

    if families_dir is None:
        families_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "families")
    with open(os.path.join(families_dir, f"dim{24 + d}", "family.json")) as f:
        fam = json.load(f)
    V, _ = sk.config(d)
    index = {tuple(r): i for i, r in enumerate(V.tolist())}
    T = fam["T"]["vectors"]
    m = [index[tuple(int(x) for x in v)] for v in T]  # raises if the record's T is not our model
    groups = [[m[i] for i in g] for g in fam["T"]["groups"]]
    sizes = [int(s["size"]) for s in fam["sets"]]
    return {"groups": groups, "n_triangles": sum(len(g) == 3 for g in groups),
            "n_pairs": sum(len(g) == 2 for g in groups), "weight": sum(len(g) - 1 for g in groups),
            "sizes": sizes, "count": int(fam["count"]), "extra": int(fam["extra"]["count"])}


def analyse(d: int, time_limit: float | None = None) -> dict:
    """Everything for one d: packing optimum, weight optimum, bounds, record comparison."""
    V, _ = sk.config(d)
    K = len(V)
    pack = max_triangle_packing(V, time_limit)
    part = max_weight_partition(V, time_limit)
    zs = zero_sum_bound(V)
    rec = record_partition(d)
    wub = weight_upper_bound(K)
    assert pack["value"] <= zs["bound"] and part["value"] <= wub
    return {"d": d, "K": K, "n_triangles": pack["n_triangles_total"], "n_admissible_pairs": part["n_admissible_pairs"],
            "packing": pack, "partition": part, "zero_sum_bound": zs, "weight_upper_bound": wub, "record": rec,
            "packing_optimal_and_proven": pack["proven"] and pack["value"] == zs["bound"],
            "weight_optimal_and_proven": part["proven"] and part["value"] == wub,
            "record_is_optimal": rec["weight"] == part["value"],
            "record_slack_weight": part["value"] - rec["weight"]}


# ---------------------------------------------------------------------------
# continuous search: n points in R^d with k prescribed triangles, minimise the max other cosine
# ---------------------------------------------------------------------------
def _penalty_phase(X0: np.ndarray, ia, ib, ta, tb, maxiter: int = 3000) -> np.ndarray:
    """Phase 1: L-BFGS-B on sum_free max(0, cos - 1/2)^2 + sum_tri (cos + 1/2)^2 over normalised
    vectors (smooth, finds the basin); SLSQP then polishes the minimax value with exact constraints."""
    n, d = X0.shape

    def pen(z):
        X = z.reshape(n, d)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        U = X / nr
        cf = (U[ia] * U[ib]).sum(axis=1)
        ct = (U[ta] * U[tb]).sum(axis=1) if len(ta) else np.zeros(0)
        v1 = np.maximum(0.0, cf - 0.5)
        v2 = ct + 0.5
        f = (v1 ** 2).sum() + (v2 ** 2).sum()
        gU = np.zeros_like(U)
        np.add.at(gU, ia, (2 * v1)[:, None] * U[ib])
        np.add.at(gU, ib, (2 * v1)[:, None] * U[ia])
        if len(ta):
            np.add.at(gU, ta, (2 * v2)[:, None] * U[tb])
            np.add.at(gU, tb, (2 * v2)[:, None] * U[ta])
        gX = (gU - (gU * U).sum(axis=1, keepdims=True) * U) / nr
        return f, gX.ravel()

    res = minimize(pen, X0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 0, "gtol": 1e-12})
    X = res.x.reshape(n, d)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def continuous_triangle_search(d: int, n: int, k: int, restarts: int = 20, seed: int = 0,
                               maxiter: int = 400, init: np.ndarray | None = None) -> dict:
    """Points 3j, 3j+1, 3j+2 (j < k) are constrained to be a triangle (pairwise ip = -1/2, unit norm);
    all points unit; minimise t subject to <x_a, x_b> <= t for every unconstrained pair.
    Returns best t (<= 0.5 + 1e-9 means a kissing configuration with k triangles), the points, the
    per-restart values. Random starts (or ``init`` plus noise)."""
    if 3 * k > n:
        raise ValueError("3k > n")
    rng = np.random.default_rng(seed)
    tri_pairs = [(3 * j + a, 3 * j + b) for j in range(k) for a, b in ((0, 1), (0, 2), (1, 2))]
    tri_set = set(tri_pairs)
    free_pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in tri_set]
    ia = np.array([p[0] for p in free_pairs])
    ib = np.array([p[1] for p in free_pairs])
    ta = np.array([p[0] for p in tri_pairs], dtype=int)
    tb = np.array([p[1] for p in tri_pairs], dtype=int)
    nv = n * d

    def unpack(z):
        return z[:nv].reshape(n, d), z[nv]

    def obj(z):
        return z[nv]

    def obj_grad(z):
        g = np.zeros_like(z)
        g[nv] = 1.0
        return g

    def ineq(z):  # t - <xa,xb> >= 0
        X, t = unpack(z)
        return t - (X[ia] * X[ib]).sum(axis=1)

    def ineq_jac(z):
        X, _ = unpack(z)
        J = np.zeros((len(ia), nv + 1))
        for r, (a, b) in enumerate(zip(ia, ib)):
            J[r, a * d:(a + 1) * d] = -X[b]
            J[r, b * d:(b + 1) * d] = -X[a]
        J[:, nv] = 1.0
        return J

    def eq(z):  # norms = 1, triangle ips = -1/2
        X, _ = unpack(z)
        return np.concatenate([(X * X).sum(axis=1) - 1.0, (X[ta] * X[tb]).sum(axis=1) + 0.5])

    def eq_jac(z):
        X, _ = unpack(z)
        J = np.zeros((n + len(ta), nv + 1))
        for i in range(n):
            J[i, i * d:(i + 1) * d] = 2 * X[i]
        for r, (a, b) in enumerate(zip(ta, tb)):
            J[n + r, a * d:(a + 1) * d] = X[b]
            J[n + r, b * d:(b + 1) * d] = X[a]
        return J

    best = None
    values = []
    for r in range(restarts):
        if init is not None:
            X0 = np.asarray(init, dtype=float) + 0.05 * rng.standard_normal((n, d))
        else:
            X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # seed the triangles exactly: rotate each prescribed triple into a random plane
        for j in range(k):
            Bm = np.linalg.qr(rng.standard_normal((d, 2)))[0]
            for a, ang in enumerate((0.0, 2 * math.pi / 3, 4 * math.pi / 3)):
                X0[3 * j + a] = Bm[:, 0] * math.cos(ang) + Bm[:, 1] * math.sin(ang)
        X0 = _penalty_phase(X0, ia, ib, ta, tb)
        t0 = float((X0[ia] * X0[ib]).sum(axis=1).max())
        z0 = np.concatenate([X0.ravel(), [t0]])
        res = minimize(obj, z0, jac=obj_grad, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": ineq, "jac": ineq_jac},
                                    {"type": "eq", "fun": eq, "jac": eq_jac}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
        X, _ = unpack(res.x)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # project; then measure honestly
        G = X @ X.T
        tri_err = float(np.abs(G[ta, tb] + 0.5).max()) if k else 0.0
        t = float(G[ia, ib].max()) if len(ia) else -1.0
        values.append((t, tri_err))
        if tri_err < 1e-7 and (best is None or t < best["max_cos"]):
            best = {"max_cos": t, "triangle_err": tri_err, "X": X, "restart": r}
    return {"d": d, "n": n, "k": k, "best": best, "values": values,
            "kissing": best is not None and best["max_cos"] <= 0.5 + 1e-9}


def _main(argv: list[str]) -> int:
    import json

    ds = [int(x) for x in argv[1:]] or list(range(2, 8))
    for d in ds:
        a = analyse(d)
        p, w, z, r = a["packing"], a["partition"], a["zero_sum_bound"], a["record"]
        print(f"d={d} K={a['K']} triangles={a['n_triangles']} admissible_pairs={a['n_admissible_pairs']}")
        print(f"  packing : max disjoint triangles = {p['value']}  lp_bound={p['lp_bound']:.4f} dual={p['dual_bound']:.4f} "
              f"gap={p['gap']} nodes={p['nodes']} leftover={len(p['leftover'])}  zero_sum_bound={z['bound']} ({z['reason']})"
              f"  {'PROVEN' if a['packing_optimal_and_proven'] else 'solver-only'}  {p['seconds']:.2f}s")
        print(f"  weight  : max = {w['value']} = 2*{len(w['triangles'])} + {len(w['pairs'])}  counting bound={a['weight_upper_bound']} "
              f"gap={w['gap']} nodes={w['nodes']}  {'PROVEN' if a['weight_optimal_and_proven'] else 'solver-only'}  {w['seconds']:.2f}s")
        print(f"  record  : {r['n_triangles']} triangles + {r['n_pairs']} pairs, weight {r['weight']}, count {r['count']}  "
              f"-> {'OPTIMAL' if a['record_is_optimal'] else 'SUBOPTIMAL by ' + str(a['record_slack_weight']) + ' weight = ' + str(a['record_slack_weight'] * min(r['sizes'])) + ' vectors'}")
        if not a["record_is_optimal"]:
            print("  optimal groups:", json.dumps(w["triangles"] + w["pairs"]))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))

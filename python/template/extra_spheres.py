"""Extra spheres of the PackingStar template (T5.1 item 3).

Given the R^d kissing configuration T (as a set — the partition is irrelevant here), the extra
vectors (0, 2y') need unit y' with cos <= 1/2 among themselves and cos <= sqrt3/2 (>= 30 degrees)
to every T vector. PackingStar uses K(d) of them (a rotated copy of T).

Upper bound (exact, no computation): the extras are themselves a kissing configuration in R^d
(pairwise cos <= 1/2), so #extra <= K(d). For d = 2, 3, 4 the kissing number is known
(6, 12, 24: Newton/Schütte-van der Waerden, Musin), so K(d) extras is optimal. For d = 5, 6, 7 the
bound is the kissing-number upper bound (44, 77, 134: Korkine-Zolotareff / Mittelmann-Vallentin,
de Laat-Leijenhorst-de Muinck Keizer; Cohn's table 2026-08-26), and any configuration of more than
K(d) = 40, 72, 126 extras would be a new kissing-number record in R^d before being anything about
the template — with the extra 30-degree-avoidance constraint on top. The searches below are
therefore expected to (and do) stop at K(d); they quantify how the constraint set behaves.

  (a) exact/combinatorial: candidate pool = integer vectors of the span with small entries
      (kiss_ref.small_kissing.extra_sphere_candidates), the record's extras (Q(sqrt M)/D form) and the
      1-plane exact rotations of T; conflict graph with exact predicates (float with 1e-9 margin for
      the pool, the chosen set re-checked exactly in the field); maximum independent set by ILP
      (HiGHS, gap 0) — the maximum over the pool, with certificate.
  (b) continuous: n = K(d)+1 points, penalty descent (L-BFGS-B) on the squared violations of
      cos <= 1/2 and cos <= sqrt3/2, random restarts; a zero-violation hit would be followed by
      an exactness attempt (none occurs).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import small_kissing as sk  # noqa: E402
from kiss_ref.exact import Q, cos_le_half, cos_le_sqrt3_half, dot, norm2  # noqa: E402

__all__ = ["KISSING_UPPER", "upper_bound", "record_extras", "candidate_pool", "max_extras_ilp",
           "continuous_extra_search", "check_exact"]

# Cohn's table (https://cohn.mit.edu/kissing-numbers/, fetched 2026-08-26): lower = upper for d <= 4, 8.
KISSING_UPPER = {2: 6, 3: 12, 4: 24, 5: 44, 6: 77, 7: 134}
SQRT3_2 = math.sqrt(3) / 2


def upper_bound(d: int) -> dict:
    """#extra <= K(d)_upper because the extras form a kissing configuration in R^d."""
    exact = KISSING_UPPER[d] == sk.K[d]
    return {"d": d, "K_lower": sk.K[d], "K_upper": KISSING_UPPER[d], "bound": KISSING_UPPER[d],
            "tight": exact,
            "argument": "extras have pairwise cos <= 1/2, i.e. form a kissing configuration in R^d"
                        + ("; K(d) is known exactly, so K(d) extras is optimal" if exact else
                           f"; exceeding {sk.K[d]} would be a new kissing-number record in R^{d}")}


# ---------------------------------------------------------------------------
# exact vectors in the field (lists of Q) and their float images
# ---------------------------------------------------------------------------
def _block_to_Q(block: dict) -> list[list[Q]]:
    """The schema's (a + b sqrt M)/D rows -> lists of Q."""
    M, D = int(block.get("sqrt", 1)), int(block.get("den", 1))
    A, B = block["a"], block["b"]
    out = []
    for a, b in zip(A, B):
        out.append([(Q(int(x)) + Q.sqrt(M, int(y))) * Q(1) / Q(D) for x, y in zip(a, b)])
    return out


def record_extras(d: int, families_dir: str | None = None) -> tuple[list[list[Q]], np.ndarray]:
    """The record family's extras for dimension 24+d, exact (Q rows) and as floats, in the ambient
    coordinates of config(d) (the extractor already mapped them onto our model)."""
    if families_dir is None:
        families_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "families")
    with open(os.path.join(families_dir, f"dim{24 + d}", "family.json")) as f:
        fam = json.load(f)
    rows = _block_to_Q(fam["extra"])
    return rows, np.array([[float(x) for x in r] for r in rows])


def _plane_rotations(T: np.ndarray, angles=(math.pi / 6, math.pi / 4, math.pi / 3), rng=None, tries: int = 60) -> list[np.ndarray]:
    """Float copies of T rotated in random coordinate-pair / root-pair planes of the span (candidate generator)."""
    rng = np.random.default_rng(0) if rng is None else rng
    Tf = np.asarray(T, dtype=float)
    out = []
    n, m = Tf.shape
    for _ in range(tries):
        i, j = rng.choice(n, 2, replace=False)
        u = Tf[i] / np.linalg.norm(Tf[i])
        v = Tf[j] - (Tf[j] @ u) * u
        if np.linalg.norm(v) < 1e-9:
            continue
        v /= np.linalg.norm(v)
        th = rng.choice(angles) * rng.choice((-1, 1))
        P = np.outer(u, u) + np.outer(v, v)
        R = np.eye(m) - P + math.cos(th) * P + math.sin(th) * (np.outer(v, u) - np.outer(u, v))
        out.append(Tf @ R.T)
    return out


def candidate_pool(d: int, max_entry: int = 2, extra_rotations: int = 60, seed: int = 0,
                   max_candidates: int = 6000) -> dict:
    """Unit-normalised float candidates y' in the span of config(d) at >= 30 degrees from every T
    vector, from: integer grid (entries in [-max_entry, max_entry]), the record's extras, plane
    rotations of T. Deduplicated by direction (1e-7). Returns floats + the exact rows where known."""
    T, _ = sk.config(d)
    Tf = np.asarray(T, dtype=float)
    Tu = Tf / np.linalg.norm(Tf, axis=1, keepdims=True)
    exact_rows: dict[int, list[Q]] = {}
    src = []
    C = sk.extra_sphere_candidates(d, max_entry=max_entry)
    Cf = np.asarray(C, dtype=float)
    Cf = Cf / np.linalg.norm(Cf, axis=1, keepdims=True)
    src += ["grid"] * len(Cf)
    exact_grid = [[Q(int(x)) for x in r] for r in np.asarray(C).tolist()]
    rows_Q, rows_f = record_extras(d)
    rows_f = rows_f / np.linalg.norm(rows_f, axis=1, keepdims=True)
    rot = _plane_rotations(T, rng=np.random.default_rng(seed), tries=extra_rotations)
    rotf = np.vstack(rot) if rot else np.zeros((0, Tf.shape[1]))
    rotf = rotf / np.linalg.norm(rotf, axis=1, keepdims=True)
    allf = np.vstack([Cf, rows_f, rotf])
    exact_list = exact_grid + rows_Q + [None] * len(rotf)
    src += ["record"] * len(rows_f) + ["rotation"] * len(rotf)
    # 30-degree filter and dedup
    ok = (allf @ Tu.T).max(axis=1) <= SQRT3_2 + 1e-9
    keep, seen = [], []
    for i in np.flatnonzero(ok):
        v = allf[i]
        if seen and (np.array(seen) @ v > 1 - 1e-7).any():
            continue
        seen.append(v)
        keep.append(i)
        if len(keep) >= max_candidates:
            break
    P = allf[keep]
    for r, i in enumerate(keep):
        if exact_list[i] is not None:
            exact_rows[r] = exact_list[i]
    return {"d": d, "P": P, "sources": [src[i] for i in keep], "exact": exact_rows, "Tu": Tu,
            "n_raw": int(len(allf)), "n_filtered": int(ok.sum()), "n": int(len(P))}


def max_extras_ilp(pool: dict, time_limit: float | None = 600.0) -> dict:
    """Maximum independent set of the conflict graph (cos > 1/2 + 1e-9) on the pool, HiGHS ILP with
    edge constraints plus greedy-clique constraints (stronger LP). Certificate: gap == 0."""
    P = pool["P"]
    n = len(P)
    G = P @ P.T
    conf = np.triu(G > 0.5 + 1e-9, 1)
    edges = np.argwhere(conf)
    t0 = time.time()
    # clique cover of the edges (greedy) for tighter constraints
    adj = conf | conf.T
    rows, cols, r = [], [], 0
    covered = np.zeros_like(conf)
    order = np.argsort(-adj.sum(axis=1))
    for i in order:
        for j in np.flatnonzero(adj[i]):
            if j < i or covered[i, j]:
                continue
            clique = [i, j]
            cand = np.flatnonzero(adj[i] & adj[j])
            for k in cand:
                if all(adj[k, c] for c in clique):
                    clique.append(k)
            for a in clique:
                for b in clique:
                    if a < b:
                        covered[a, b] = True
            for a in clique:
                rows.append(r)
                cols.append(a)
            r += 1
    if r == 0:
        return {"value": n, "chosen": list(range(n)), "gap": 0.0, "dual_bound": float(n), "nodes": 0,
                "n_edges": 0, "n_cliques": 0, "lp_bound": float(n), "seconds": 0.0, "sources": list(pool["sources"])}
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, n))
    cons = LinearConstraint(A, -np.inf, 1.0)
    c = -np.ones(n)
    lp = milp(c, constraints=cons, bounds=Bounds(0, 1), integrality=np.zeros(n))
    opts = {"disp": False}
    if time_limit:
        opts["time_limit"] = float(time_limit)
    res = milp(c, constraints=cons, bounds=Bounds(0, 1), integrality=np.ones(n), options=opts)
    if not res.success:
        raise RuntimeError(res.message)
    x = np.rint(res.x).astype(int)
    chosen = [int(i) for i in np.flatnonzero(x)]
    S = P[chosen]
    assert (np.triu(S @ S.T, 1) <= 0.5 + 1e-9).all()
    return {"value": len(chosen), "chosen": chosen, "gap": float(getattr(res, "mip_gap", 0.0)),
            "dual_bound": float(abs(getattr(res, "mip_dual_bound", res.fun))), "nodes": int(getattr(res, "mip_node_count", 0)),
            "n_edges": int(len(edges)), "n_cliques": int(r), "lp_bound": float(-lp.fun), "seconds": time.time() - t0,
            "sources": [pool["sources"][i] for i in chosen]}


def check_exact(rows: list[list[Q]], d: int) -> dict:
    """Exact check of extras (Q rows) against config(d): pairwise cos <= 1/2, cos <= sqrt3/2 to T."""
    T, _ = sk.config(d)
    TQ = [[Q(int(x)) for x in r] for r in T.tolist()]
    NT = [norm2(t) for t in TQ]
    NR = [norm2(r) for r in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if not cos_le_half(dot(rows[i], rows[j]), NR[i], NR[j]):
                return {"ok": False, "why": f"extras {i},{j} closer than 60 degrees"}
        for k, t in enumerate(TQ):
            if not cos_le_sqrt3_half(dot(rows[i], t), NR[i], NT[k]):
                return {"ok": False, "why": f"extra {i} within 30 degrees of T[{k}]"}
    return {"ok": True, "n": len(rows)}


# ---------------------------------------------------------------------------
# continuous search for K(d)+1
# ---------------------------------------------------------------------------
def continuous_extra_search(d: int, n: int | None = None, restarts: int = 20, seed: int = 0,
                            maxiter: int = 2000, init_from_record: bool = True) -> dict:
    """n unit vectors (default K(d)+1) in R^d minimising the penalty
       sum_{i<j} max(0, cos_ij - 1/2)^2 + sum_{i,t} max(0, cos(y_i, t) - sqrt3/2)^2,
    L-BFGS-B over free coordinates (vectors normalised inside). Reports the best max violation."""
    T, _ = sk.config(d)
    n = sk.K[d] + 1 if n is None else n
    E = sk.orthonormal_embedding(T, d)  # (K, d) unit-scale embedding of T's span... rows are T in R^d
    Tu = E / np.linalg.norm(E, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    _, recf = record_extras(d)
    if init_from_record:
        # embed the record extras in the same R^d coordinates as Tu: least squares onto the span
        Tf = np.asarray(T, dtype=float)
        M, *_ = np.linalg.lstsq(Tf, E, rcond=None)  # ambient -> R^d map on the span
        rec_d = recf @ M
        rec_d /= np.linalg.norm(rec_d, axis=1, keepdims=True)
    iu = np.triu_indices(n, 1)

    def penalty(z):
        X = z.reshape(n, d)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        U = X / nr
        G = U @ U.T
        v1 = np.maximum(0.0, G[iu] - 0.5)
        H = U @ Tu.T
        v2 = np.maximum(0.0, H - SQRT3_2)
        f = (v1 ** 2).sum() + (v2 ** 2).sum()
        # gradient w.r.t. U, then chain through the normalisation
        gU = np.zeros_like(U)
        W = np.zeros((n, n))
        W[iu] = 2 * v1
        W = W + W.T
        gU += W @ U
        gU += (2 * v2) @ Tu
        # d(U)/d(X): (I - u u^T)/|x|
        gX = (gU - (gU * U).sum(axis=1, keepdims=True) * U) / nr
        return f, gX.ravel()

    best = None
    hist = []
    for r in range(restarts):
        if init_from_record and r % 2 == 0:
            X0 = np.vstack([rec_d, rng.standard_normal((n - len(rec_d), d))]) if n > len(rec_d) else rec_d[:n].copy()
            X0 = X0 + 0.02 * rng.standard_normal(X0.shape)
        else:
            X0 = rng.standard_normal((n, d))
        res = minimize(penalty, X0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 0, "gtol": 1e-14})
        U = res.x.reshape(n, d)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        G = U @ U.T
        viol = max(float((G[iu] - 0.5).max()), float((U @ Tu.T - SQRT3_2).max()))
        hist.append((float(res.fun), viol))
        if best is None or viol < best["max_violation"]:
            best = {"max_violation": viol, "penalty": float(res.fun), "restart": r, "U": U}
    return {"d": d, "n": n, "best_max_violation": best["max_violation"], "best_penalty": best["penalty"],
            "hit": best["max_violation"] <= 1e-9, "history": hist, "U": best["U"]}


def _main(argv: list[str]) -> int:
    ds = [int(x) for x in argv[1:]] or list(range(2, 8))
    for d in ds:
        ub = upper_bound(d)
        pool = candidate_pool(d, max_entry=2 if d <= 5 else 1, extra_rotations=60 if d <= 5 else 12,
                              max_candidates=6000 if d <= 5 else 2500)
        ilp = max_extras_ilp(pool)
        rows_Q, _ = record_extras(d)
        rec_ok = check_exact(rows_Q, d)
        exact_chosen = [pool["exact"][i] for i in ilp["chosen"] if i in pool["exact"]]
        print(f"d={d} K={sk.K[d]} bound={ub['bound']} tight={ub['tight']}: {ub['argument']}")
        print(f"  pool: raw={pool['n_raw']} after-30deg={pool['n_filtered']} dedup={pool['n']} "
              f"sources={ {s: pool['sources'].count(s) for s in set(pool['sources'])} }")
        print(f"  ILP max independent set = {ilp['value']}  lp_bound={ilp['lp_bound']:.3f} dual={ilp['dual_bound']:.3f} gap={ilp['gap']} "
              f"nodes={ilp['nodes']} edges={ilp['n_edges']} cliques={ilp['n_cliques']} {ilp['seconds']:.1f}s  "
              f"chosen sources={ {s: ilp['sources'].count(s) for s in set(ilp['sources'])} } exact-known={len(exact_chosen)}/{ilp['value']}")
        print(f"  record extras exact check: {rec_ok}")
        cs = continuous_extra_search(d, restarts=10 if d <= 5 else 6)
        print(f"  continuous n=K+1={cs['n']}: best max violation {cs['best_max_violation']:.3e} (penalty {cs['best_penalty']:.3e}) hit={cs['hit']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))

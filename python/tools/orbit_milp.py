#!/usr/bin/env python3
"""T3.4 — exact maximum-weight orbit union by MILP (HiGHS via scipy.optimize.milp).

Input: the reduced orbit graph written by `orbit_mis --export-graph F` (the
non-self-conflicting orbits with their sizes as weights, the conflict edges
among them, and the orbit ids of the reference set if it is an orbit union).

Model: x_o ∈ {0,1} per usable orbit, maximise Σ |o|·x_o subject to Σ_{o∈K} x_o ≤ 1
for every clique K of a greedy edge clique cover (every conflict edge lies in
at least one K, so feasible x are exactly the independent orbit unions; clique
constraints give a much stronger LP relaxation than edge constraints).

Prints the best union found, HiGHS's dual bound / MIP gap and whether the
solve proved optimality within the time limit. Optionally writes the chosen
orbit ids (one per line) with --out-orbits.

Usage:
  .venv/bin/python python/tools/orbit_milp.py runs/orbits/deep/stab496_reduced_graph.txt --time 1800
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


def read_graph(path: str):
    with open(path) as f:
        tok = f.readline().split()
        assert tok[0] == "ORBITS", tok
        n_total, n_usable = int(tok[1]), int(tok[2])
        ids = np.empty(n_usable, dtype=np.int64)
        w = np.empty(n_usable, dtype=np.int64)
        for k in range(n_usable):
            a, b = f.readline().split()
            ids[k], w[k] = int(a), int(b)
        tok = f.readline().split()
        assert tok[0] == "EDGES", tok
        m = int(tok[1])
        E = np.loadtxt(f, dtype=np.int64, max_rows=m).reshape(m, 2) if m else np.zeros((0, 2), dtype=np.int64)
        tok = f.readline().split()
        assert tok[0] == "REF", tok
        k = int(tok[1])
        ref = np.array([int(f.readline()) for _ in range(k)], dtype=np.int64)
    return n_total, ids, w, E, ref


def greedy_clique_cover(n: int, E: np.ndarray, adj: np.ndarray) -> list[np.ndarray]:
    """Cliques covering every edge: grow each uncovered edge to a maximal clique
    (candidates = common neighbours, taken in order of most uncovered edges)."""
    covered = np.zeros(len(E), dtype=bool)
    # edge index lookup
    key = E[:, 0] * n + E[:, 1]
    order = np.argsort(key)
    key_sorted = key[order]

    def edge_index(a, b):
        lo, hi = (a, b) if a < b else (b, a)
        i = np.searchsorted(key_sorted, lo * n + hi)
        return order[i]

    cliques = []
    # process edges in order; skip covered
    for e in range(len(E)):
        if covered[e]:
            continue
        u, v = E[e]
        K = [u, v]
        cand = np.flatnonzero(adj[u] & adj[v])
        while len(cand):
            # pick the candidate with the largest common neighbourhood with the remaining candidates
            sub = adj[np.ix_(cand, cand)]
            deg = sub.sum(axis=1)
            j = int(np.argmax(deg))
            c = cand[j]
            K.append(c)
            cand = cand[adj[c, cand]]
        K = np.array(K, dtype=np.int64)
        for i in range(len(K)):
            for j in range(i + 1, len(K)):
                covered[edge_index(K[i], K[j])] = True
        cliques.append(K)
    assert covered.all()
    return cliques


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--time", type=float, default=1800.0, help="HiGHS time limit (s)")
    ap.add_argument("--gap", type=float, default=0.0, help="relative MIP gap to stop at")
    ap.add_argument("--out-orbits", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    n_total, ids, w, E, ref = read_graph(args.graph)
    n = len(ids)
    pos = {int(o): k for k, o in enumerate(ids)}
    Ei = np.array([[pos[int(a)], pos[int(b)]] for a, b in E], dtype=np.int64).reshape(len(E), 2)
    adj = np.zeros((n, n), dtype=bool)
    adj[Ei[:, 0], Ei[:, 1]] = True
    adj[Ei[:, 1], Ei[:, 0]] = True
    print(f"graph: {n_total} orbits, {n} usable (weights {dict(zip(*np.unique(w, return_counts=True)))}), "
          f"{len(E)} edges, reference = union of {len(ref)} orbits (weight {int(w[[pos[int(o)] for o in ref]].sum()) if len(ref) else 0})")
    ref_x = np.zeros(n)
    if len(ref):
        for o in ref:
            ref_x[pos[int(o)]] = 1
        # sanity: the reference is independent in the reduced graph
        assert not (adj[Ei[:, 0], Ei[:, 1]] & (ref_x[Ei[:, 0]] * ref_x[Ei[:, 1]] > 0)).any()

    cliques = greedy_clique_cover(n, Ei, adj)
    sizes = np.array([len(K) for K in cliques])
    print(f"greedy edge clique cover: {len(cliques)} cliques, sizes min/mean/max = {sizes.min()}/{sizes.mean():.1f}/{sizes.max()}, "
          f"{time.time() - t0:.1f}s")
    rows = np.concatenate([np.full(len(K), i) for i, K in enumerate(cliques)])
    cols = np.concatenate(cliques)
    A = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(cliques), n))
    cons = LinearConstraint(A, -np.inf, 1.0)
    # trivial clique-cover bound: Σ_K max_{o∈K} w_o over a clique *partition* would bound; the cover is not a
    # partition, so just report the LP bound from HiGHS.
    res = milp(c=-w.astype(float), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": args.time, "mip_rel_gap": args.gap, "disp": not args.quiet})
    elapsed = time.time() - t0
    if res.x is None:
        print(f"RESULT ok=0 status={res.status} message='{res.message}' seconds={elapsed:.1f}")
        return 1
    x = np.rint(res.x).astype(int)
    chosen = np.flatnonzero(x)
    best = int(w[chosen].sum())
    # validity
    bad = adj[np.ix_(chosen, chosen)].any()
    dual = -float(res.mip_dual_bound) if res.mip_dual_bound is not None else float("nan")
    gap = float(res.mip_gap) if res.mip_gap is not None else float("nan")
    optimal = res.status == 0 and res.success
    print(f"best union: {best} vectors from {len(chosen)} orbits; valid={not bad}; dual bound {dual:.3f}; "
          f"gap {gap:.4g}; status={res.status} ({res.message}); optimal={int(optimal)}; {elapsed:.1f}s")
    if args.out_orbits:
        with open(args.out_orbits, "w") as f:
            for k in chosen:
                f.write(f"{int(ids[k])}\n")
    print(f"RESULT ok={int(not bad)} usable={n} edges={len(E)} cliques={len(cliques)} best={best} "
          f"dual_bound={dual:.3f} upper={int(np.floor(dual + 1e-6)) if dual == dual else -1} optimal={int(optimal)} "
          f"reference={int(w[[pos[int(o)] for o in ref]].sum()) if len(ref) else 0} seconds={elapsed:.1f}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

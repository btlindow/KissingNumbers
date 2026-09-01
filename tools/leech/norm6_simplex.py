#!/usr/bin/env python3
"""The largest regular simplex of edge sqrt(6) in the Leech lattice.

Equivalently: how many vectors of norm 6 can be pairwise at inner product 3?  Their pairwise
differences then also have norm 6, so {0, v_1, ..., v_k} is a set of k+1 lattice points at
mutual squared distance 6, and the lattice they span is a copy of sqrt(3) A_k.

    THE MAXIMUM IS EXACTLY 24, AND IT IS ATTAINED.

Upper bound, with no computation: the Gram of k such vectors is 3(I_k + J_k), with eigenvalues
3(k+1) once and 3 with multiplicity k-1.  It is positive definite, so the v_i are linearly
independent and k <= dim R^24 = 24.  Lower bound: an explicit 24-system, stored in
data/norm6_simplex24.json and re-verified here; randomised greedy in the link of one vector
finds one in seconds, so the configuration is common rather than rare.

This script also reports the quantities such a system determines: the sets
A(v_i) = { u minimal : <u, v_i> = -3 } of size 552, the bijection w -> w - v_i, and their
pairwise, triple and higher intersections -- for which see
docs/reports/08-the-second-shell.md.

Scaling: sqrt-8 integers, minimal norm 32, norm 6 is 48, inner product 3 is 24.

    python tools/leech/norm6_simplex.py
"""

from __future__ import annotations
import os, sys, json, time
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
from kiss_ref.leech import leech_min_vectors  # noqa: E402
from norm6_shell import load as load_norm6, in_leech  # noqa: E402

CERT = os.path.join(ROOT, "data", "norm6_simplex24.json")
SEED = 20260901


def find_system(V, seed=SEED, target=24, budget=120.0):
    """Randomised greedy in the link of V[0]; returns a maximal system of size `target`."""
    N = len(V)
    v1 = np.asarray(V[0], dtype=np.int32)
    d = np.empty(N, dtype=np.int32)
    for s in range(0, N, 2_000_000):
        d[s:s + 2_000_000] = V[s:s + 2_000_000].astype(np.int32) @ v1
    link = V[np.nonzero(d == 24)[0]].astype(np.int32)
    rng = np.random.default_rng(seed)
    t0, best = time.time(), None
    while time.time() - t0 < budget:
        cur, chain = link, [v1.copy()]
        while len(cur):
            w = cur[int(rng.integers(len(cur)))]
            chain.append(w.copy())
            cur = cur[cur @ w == 24]
        if best is None or len(chain) > len(best):
            best = np.array(chain, dtype=np.int64)
            if len(best) >= target:
                break
    return best, len(link)


def main():
    ok = True
    V = load_norm6()
    print(f"RESULT norm6-shell loaded={len(V)} (expected 16773120: {len(V) == 16773120})")
    ok &= len(V) == 16773120

    # ---- the inner-product spectrum of a norm-6 vector against the shell ----------
    v0 = np.asarray(V[0], dtype=np.int32)
    d = np.empty(len(V), dtype=np.int32)
    for s in range(0, len(V), 2_000_000):
        d[s:s + 2_000_000] = V[s:s + 2_000_000].astype(np.int32) @ v0
    vals, cnts = np.unique(d, return_counts=True)
    spec = dict(zip(map(int, vals), map(int, cnts)))
    print(f"RESULT norm6-vs-norm6 ip spectrum (x8 scaling) = {spec}")
    print(f"RESULT   -> standard inner products {sorted(k // 8 for k in spec)}; "
          f"NO +-5: {40 not in spec and -40 not in spec}; |L(v)| at ip 3 = {spec.get(24)}")
    ok &= (40 not in spec) and spec.get(24) == 257600

    # ---- (a) the system -----------------------------------------------------------
    if os.path.exists(CERT):
        S = np.array(json.load(open(CERT))["vectors"], dtype=np.int64)
        link = spec[24]
    else:
        S, link = find_system(V)
    k = len(S)
    G = S @ S.T
    off = ~np.eye(k, dtype=bool)
    good = bool(np.all(np.diag(G) == 48) and np.all(G[off] == 24))
    inlat = bool(np.all(in_leech(S)))
    rank = int(np.linalg.matrix_rank(G.astype(float)))
    print(f"RESULT dim28-system k={k} gram_is_24(I+J)={good} all_in_Leech={inlat} "
          f"rank={rank} link_size={link}")
    ok &= good and inlat and rank == k

    # upper bound, exactly: eigenvalues of 24(I_k + J_k)
    ev = sorted(set(np.round(np.linalg.eigvalsh(G.astype(float)), 6)))
    print(f"RESULT dim28-upper-bound Gram 24(I+J) eigenvalues={ev} "
          f"(24(k+1)={24*(k+1)}, 24 with multiplicity {k-1}) -> positive definite, "
          f"rank k, so k <= 24. MAXIMUM = 24, ATTAINED: {k == 24}")
    ok &= k == 24

    # ---- (b) the classes and their deduplication ----------------------------------
    C = leech_min_vectors().astype(np.int64)
    key = {tuple(r): i for i, r in enumerate(map(tuple, C))}
    W, A = [], []
    for i in range(k):
        ip = C @ S[i]
        Wi = np.nonzero(ip == 24)[0]
        Ai = np.nonzero(ip == -24)[0]
        W.append(Wi); A.append(Ai)
    sizes = [len(x) for x in W]
    print(f"RESULT W-sizes |W_i| = {sorted(set(sizes))} for all {k} classes "
          f" -> {set(sizes) == {552}}")
    ok &= set(sizes) == {552}
    # w -> w - v_i is a bijection W_i -> A_i
    bij = True
    for i in range(k):
        shifted = {key[tuple(r)] for r in (C[W[i]] - S[i])}
        bij &= shifted == set(map(int, A[i]))
    print(f"RESULT the map w -> w - v_i is a bijection W_i -> A_i = {{u : <u,v_i> = -3}} "
          f"for every i: {bij}")
    ok &= bij

    As = [set(map(int, a)) for a in A]
    pair = sorted({len(As[i] & As[j]) for i, j in combinations(range(k), 2)})
    trip = sorted({len(As[i] & As[j] & As[l]) for i, j, l in combinations(range(k), 3)})
    quad = sorted({len(As[i] & As[j] & As[l] & As[m])
                   for i, j, l, m in combinations(range(k), 4)})
    print(f"RESULT intersections |A_i ∩ A_j| values={pair}; triples={trip}; quadruples={quad}")

    seen, s = set(), []
    for i in range(k):
        s.append(len(As[i] - seen))
        seen |= As[i]
    print(f"RESULT dedup sizes s_1..s_{k} = {s}")
    print(f"RESULT   greedy deduplication sizes s_m = |A_m \\ (A_1 u ... u A_(m-1))|; the "
          f"total sum_m s_m = |A_1 u ... u A_k| is order-independent")
    pred = [3 * m * m - 42 * m + 591 for m in range(1, k + 1)]
    agree = [i + 1 for i in range(k) if s[i] == pred[i]]
    print(f"RESULT   inclusion-exclusion prediction 3m^2-42m+591 (pairwise 33, triple 6, "
          f"quadruple 0) = {pred[:8]}...; matches s_m for m in {agree}")
    print(f"RESULT   cumulative sum_1^m s_i = {[sum(s[:m]) for m in range(1, k + 1)]}")
    print(f"RESULT   union |A_1 u ... u A_{k}| = {len(seen)} of 196560")

    if not os.path.exists(CERT):
        os.makedirs(os.path.dirname(CERT), exist_ok=True)
        json.dump({"task": "T4a", "description":
                   "24 Leech vectors of norm 6 (squared norm 48 in the sqrt-8 integer scaling) "
                   "pairwise at inner product 3 (24 in this scaling); the maximum possible, "
                   "since their Gram 24(I+J) is positive definite of rank k <= 24.",
                   "scaling": "sqrt8-integer, minimal norm 32", "k": int(k),
                   "gram": "24*(I_24 + J_24)", "seed": SEED,
                   "vectors": S.tolist()}, open(CERT, "w"))
        print(f"certificate written to {CERT}")
    print(f"RESULT task4ab ok={int(ok)} max_norm6_system=24")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Complete stars: |A(v) n A(w)| = 33 by exhaustion, as corroboration for the moment
proof in `second_shell_intersections.py`.

A(v) = { u minimal : <u, v> = -3 }, |A(v)| = 552, for v of norm 6.

PAIRS.  For a FIXED v, this computes |A(v) n A(w)| for EVERY w in the link
L(v) = { w norm 6 : <v,w> = 3 }, all 257600 of them -- not a sample.  That settles the
value for every pair containing v.  Repeating over several v then makes the result
independent of Co_0-transitivity on the norm-6 shell (which is classical, Conway/SPLAG
ch. 10, but is not assumed here).

TRIPLES.  For a fixed pair (v1, v2) this computes |A_1 n A_2 n A_3| for EVERY v3 in the
common link, all 32571 of them, and checks a candidate invariant against them.

Scaling: sqrt-8 integers; minimal norm 32, norm 6 is 48, inner product 3 is 24, -3 is -24.
"""
from __future__ import annotations
import os, sys, json
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "python", "kiss_ref")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
from kiss_ref.leech import leech_min_vectors, is_lattice_vector_rows  # noqa: E402
from norm6_shell import load as load_norm6                            # noqa: E402

N_BASE = 8


def dots(V, v):
    out = np.empty(len(V), dtype=np.int32)
    vi = np.asarray(v, dtype=np.int32)
    for s in range(0, len(V), 2_000_000):
        out[s:s + 2_000_000] = V[s:s + 2_000_000].astype(np.int32) @ vi
    return out


def main():
    ok = True
    C = leech_min_vectors().astype(np.int64)
    V = load_norm6()
    rng = np.random.default_rng(5)
    bases = [0] + [int(rng.integers(len(V))) for _ in range(N_BASE - 1)]

    # ---------------------------------------------------------------- PAIRS
    hist_all = Counter()
    for b in bases:
        v = np.asarray(V[b], dtype=np.int64)
        A = C[C @ v == -24]
        d = dots(V, v)
        L = V[d == 24].astype(np.int32)
        cnt = (A.astype(np.int32) @ L.T == -24).sum(axis=0)
        h = Counter(map(int, cnt))
        hist_all.update(h)
        print(f"RESULT pair-star base={b} |A(v)|={len(A)} |L(v)|={len(L)} "
              f"|A(v) n A(w)| over ALL w in L(v): {dict(sorted(h.items()))}")
        ok &= set(h) == {33} and len(A) == 552 and len(L) == 257600
    print(f"RESULT pairs COMPLETE over {N_BASE} full stars = "
          f"{sum(hist_all.values())} ordered pairs (not a sample): "
          f"values {dict(sorted(hist_all.items()))} -> |A_i n A_j| = 33 always: "
          f"{set(hist_all) == {33}}")

    # -------------------------------------------------------------- TRIPLES
    v1 = np.asarray(V[0], dtype=np.int64)
    d1 = dots(V, v1)
    L1 = V[d1 == 24].astype(np.int64)
    v2 = L1[0]
    L12 = L1[L1 @ v2 == 24]
    A1 = C[C @ v1 == -24]
    A2 = C[C @ v2 == -24]
    A12 = A1[np.isin([tuple(r) for r in map(tuple, A1)],
                     [tuple(r) for r in map(tuple, A2)])] if False else None
    s1 = {tuple(r) for r in map(tuple, A1)}
    A12 = np.array([r for r in A2 if tuple(r) in s1], dtype=np.int64)
    print(f"RESULT triple-star |A_1 n A_2| = {len(A12)}; common link |L(v1) n L(v2)| = {len(L12)}")
    cnt3 = (A12.astype(np.int32) @ L12.astype(np.int32).T == -24).sum(axis=0)
    h3 = Counter(map(int, cnt3))
    print(f"RESULT triple values over ALL {len(L12)} completions v3: {dict(sorted(h3.items()))}")

    # the invariant: is (v1 + v2 + v3) / 3 a lattice vector?
    S = (v1 + v2 + L12)
    div = np.all(S % 3 == 0, axis=1)
    third = np.zeros(len(S), dtype=bool)
    if div.any():
        third[div] = is_lattice_vector_rows(S[div] // 3)
    tab = Counter(zip(map(int, cnt3), map(bool, third)))
    print(f"RESULT invariant (|A1 n A2 n A3|, (v1+v2+v3)/3 in Lambda) = "
          f"{dict(sorted(tab.items()))}")
    byval = {}
    for (c, t), n in tab.items():
        byval.setdefault(c, set()).add(t)
    # it SEPARATES only if no truth value is shared between the two intersection classes;
    # a value that is constant across them separates nothing.
    vals = list(byval.values())
    separates = all(a.isdisjoint(b) for i, a in enumerate(vals) for b in vals[i + 1:])
    constant = len(set().union(*vals)) == 1
    print(f"RESULT the invariant (v1+v2+v3)/3 in Lambda separates the two triple types: "
          f"{separates}{'  (it is CONSTANT, so it separates nothing)' if constant else ''} "
          f"({ {k: sorted(v) for k, v in sorted(byval.items())} })")
    print(f"RESULT   -> the separating invariant is the common second-shell link size, "
          f"7780 vs 7966; see docs/reports/08-the-second-shell.md section 4")
    if div.any():
        n6 = ((S[div] // 3) ** 2).sum(1)
        print(f"RESULT when (v1+v2+v3)/3 is in Lambda its squared norm (sqrt-8 scaling) is "
              f"{sorted(set(map(int, n6)))} -> a MINIMAL vector: {set(map(int, n6)) == {32}}")

    print(f"RESULT second-shell-stars ok={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

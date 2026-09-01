#!/usr/bin/env python3
"""Extremal minimal vectors against the second shell of the Leech lattice.

For v in the Leech lattice of norm 6 and u a minimal vector, |v - u|^2 = 10 - 2<u,v> lies in
the lattice, so it is 0 or at least 4; hence |<u,v>| <= 3, and 3 is attained.  Call

    A(v) = { u minimal : <u, v> = -3 }        the vectors EXTREMAL against v.

|A(v)| = 552 for every v (Co_0 is transitive on the second shell).  This script answers, in
exact arithmetic, how many minimal vectors are extremal against SEVERAL second-shell vectors
at once, when those vectors are themselves as close together as the lattice allows -- norm 6
and pairwise inner product 3, i.e. all pairwise differences again of norm 6.

    THEOREM.  |A(v) n A(w)| = 33 for every pair v, w of norm 6 with <v,w> = 3.

    Proof (implemented below).  Write B(u) = { v of norm 6 : <v,u> = -3 } for u minimal;
    |B(u)| = 47104.  Let P be the set of ordered pairs (v,w) of norm-6 vectors at inner
    product 3.  Exchanging the order of summation twice,

        E := sum_{(v,w) in P} |A(v) n A(w)|
           = sum_u #{(v,w) in B(u)^2 : <v,w> = 3}                    = 196560 * N
        Q := sum_{(v,w) in P} |A(v) n A(w)|^2
           = sum_{u,u'} #{(v,w) in (B(u) n B(u'))^2 : <v,w> = 3}     = sum_c 196560 v_c M_c

    using only that Gamma = <the five Co_0 generators of data/group> is transitive on the
    196560 minimal vectors and has exactly seven orbitals on ordered pairs of them -- both
    verified in docs/reports/02-upper-bounds.md section 4.  So E and Q follow from ONE
    minimal vector and SEVEN representative pairs.  The mean is 33 and the variance is 0,
    which forces the count to be constant.

    HIGHER INTERSECTIONS.  For v_1..v_k of norm 6, pairwise at inner product 3, the Gram is
    3(I_k + J_k) and (I_k + J_k)^{-1} = I - J/(k+1), so every u in A(v_1) n ... n A(v_k) has
    the SAME projection p = -(v_1 + ... + v_k)/(k+1) onto their span, with |p|^2 = 3k/(k+1)
    and |u - p|^2 = 4 - 3k/(k+1).  Cauchy-Schwarz then pins <u,u'> to a short list and
    0 <= |sum (u - p)|^2 bounds the size:

        k = 2 : no bound (the max cross term is 0 -- which is why k = 2 needed the argument above)
        k = 3 : <= 8      k = 4 : <= 5      k = 5 : <= 4      k = 6 : <= 3

    TWO FLAVOURS OF TRIPLE.  Over all completions of a fixed pair, |A_1 n A_2 n A_3| takes
    exactly two values, 4 and 6, and they are separated by the size of the triple's common
    second-shell link |L(v_1) n L(v_2) n L(v_3)| in {7780, 7966}.  This is the analogue, one
    shell up, of the split p^3_{33} = 43164 = 42240 + 924 of orthogonal triples of minimal
    vectors (docs/reports/02-upper-bounds.md section 6.1).

    python tools/leech/second_shell_intersections.py
"""
from __future__ import annotations
import os, sys
from fractions import Fraction as F
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
from kiss_ref.leech import leech_min_vectors      # noqa: E402
from norm6_shell import load as load_norm6        # noqa: E402

DOTS = (-32, -16, -8, 0, 8, 16, 32)
VAL = (1, 4600, 47104, 93150, 47104, 4600, 1)


def count_pairs_at(X, target=24, chunk=1500):
    X = np.ascontiguousarray(X, dtype=np.int32)
    return sum(int((X[s:s + chunk] @ X.T == target).sum()) for s in range(0, len(X), chunk))


def bound(k):
    p2 = F(3 * k, k + 1)
    r = F(4) - p2
    ips = [t for t in (-4, -2, -1, 0, 1, 2, 4) if p2 - r <= t <= p2 + r and t != 4]
    mx = max(F(t) - p2 for t in ips)
    return (p2, r, ips, mx, None if mx >= 0 else int(1 - r / mx))


def main():
    ok = True
    C = leech_min_vectors().astype(np.int64)
    V = load_norm6()
    print(f"RESULT shell |second shell| = {len(V)} (theta series: 16773120): "
          f"{len(V) == 16773120}")
    ok &= len(V) == 16773120

    v0 = np.asarray(V[0], dtype=np.int32)
    d = np.empty(len(V), dtype=np.int32)
    for s in range(0, len(V), 2_000_000):
        d[s:s + 2_000_000] = V[s:s + 2_000_000].astype(np.int32) @ v0
    spec = dict(zip(*[list(map(int, a)) for a in np.unique(d, return_counts=True)]))
    print(f"RESULT shell inner-product spectrum of a norm-6 vector against the shell "
          f"(sqrt-8 scaling) = {spec}")
    print(f"RESULT shell   -> standard values {sorted(k // 8 for k in spec)}; +-5 absent: "
          f"{40 not in spec}; |L(v)| at inner product 3 = {spec.get(24)}")
    ok &= 40 not in spec and spec.get(24) == 257600

    # ---- the theorem ---------------------------------------------------------
    u = np.asarray(C[0], dtype=np.int32)
    for s in range(0, len(V), 2_000_000):
        d[s:s + 2_000_000] = V[s:s + 2_000_000].astype(np.int32) @ u
    Bu = V[d == -24].astype(np.int32)
    print(f"RESULT thm |B(u)| = {len(Bu)} (47104): {len(Bu) == 47104}", flush=True)
    N = count_pairs_at(Bu)
    G = C @ C[0]
    M = []
    for c, dot in enumerate(DOTS):
        j = int(np.nonzero(G == dot)[0][0])
        Bup = Bu[Bu @ np.asarray(C[j], dtype=np.int32) == -24]
        M.append(count_pairs_at(Bup))
        print(f"RESULT thm class {c} (<u,u'> = {dot:>3}, valency {VAL[c]:>5}): "
              f"|B(u) n B(u')| = {len(Bup):>5}, M_c = {M[-1]}", flush=True)
    P = F(len(V)) * 257600
    E = F(len(C)) * N
    Q = sum(F(len(C)) * VAL[c] * M[c] for c in range(7))
    mean, var = E / P, Q / P - (E / P) ** 2
    print(f"RESULT thm N = {N}; |P| = {int(P)}; E = {int(E)}; Q = {int(Q)}")
    print(f"RESULT thm mean = {mean}; variance = {var} -> |A(v) n A(w)| = 33 for EVERY pair: "
          f"{var == 0 and mean == 33}")
    ok &= var == 0 and mean == 33

    # ---- higher intersections and the two flavours ---------------------------
    for k in range(2, 7):
        p2, r, ips, mx, n = bound(k)
        print(f"RESULT higher k={k}: |p|^2={p2} |u-p|^2={r} admissible <u,u'>={ips} "
              f"max<u-p,u'-p>={mx} -> bound {'none' if n is None else '<= %d' % n}")

    L1 = V[np.nonzero(np.array([1]))[0]] if False else None
    v1 = np.asarray(V[0], dtype=np.int64)
    for s in range(0, len(V), 2_000_000):
        d[s:s + 2_000_000] = V[s:s + 2_000_000].astype(np.int32) @ v1.astype(np.int32)
    L1 = V[d == 24].astype(np.int64)
    v2 = L1[0]
    L12 = L1[L1 @ v2 == 24]
    A1 = C[C @ v1 == -24]
    s1 = {tuple(r) for r in map(tuple, A1)}
    A12 = np.array([r for r in C[C @ v2 == -24] if tuple(r) in s1], dtype=np.int64)
    n3 = (A12.astype(np.int32) @ L12.astype(np.int32).T == -24).sum(axis=0)
    L32 = L12.astype(np.int32)
    link3 = np.empty(len(L12), dtype=np.int32)
    for s in range(0, len(L12), 4000):
        link3[s:s + 4000] = (L32[s:s + 4000] @ L32.T == 24).sum(axis=1)
    tab = Counter(zip(map(int, n3), map(int, link3)))
    print(f"RESULT flavours |A_1 n A_2| = {len(A12)}; over ALL {len(L12)} completions v_3, "
          f"(|A_1 n A_2 n A_3|, |L_1 n L_2 n L_3|) = {dict(sorted(tab.items()))}")
    ok &= len(A12) == 33 and set(k for k, _ in tab) == {4, 6}
    print(f"RESULT second-shell-intersections ok={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

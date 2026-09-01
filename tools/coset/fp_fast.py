#!/usr/bin/env python3
"""Fast Fincke-Pohst: float Gram-Schmidt for PRUNING only; every surviving point is then
checked with exact integer arithmetic, so the OUTPUT is exact.  Over-collection is harmless
(a slightly loose float bound enumerates a few extra candidates, which the exact check
discards); under-collection is guarded against by inflating the radius by a small epsilon at
every level.
"""
from __future__ import annotations
import math


def enumerate_ball(B, target, R2):
    """All x in the lattice spanned by the integer rows of B with |x - target|^2 <= R2.

    B      : list of integer rows (a basis, ideally LLL-reduced)
    target : integer vector
    R2     : integer bound
    Returns (points, distances2) with exact integer distances.
    """
    n, N = len(B), len(B[0])
    Bs, mu = [], [[0.0] * n for _ in range(n)]
    for i in range(n):
        v = [float(t) for t in B[i]]
        for j in range(i):
            nj = sum(x * x for x in Bs[j])
            mu[i][j] = sum(B[i][k] * Bs[j][k] for k in range(N)) / nj
            v = [v[k] - mu[i][j] * Bs[j][k] for k in range(N)]
        Bs.append(v)
    Bn = [sum(x * x for x in b) for b in Bs]

    # coordinates of the target in the basis, via the Gram-Schmidt residual
    c = [0.0] * n
    res = [float(t) for t in target]
    for i in range(n - 1, -1, -1):
        c[i] = sum(res[k] * Bs[i][k] for k in range(N)) / Bn[i]
        res = [res[k] - c[i] * B[i][k] for k in range(N)]

    out, x = [], [0] * n
    EPS = 1e-6

    def rec(i, rem, off):
        if i < 0:
            out.append(list(x))
            return
        centre, bn = off[i], Bn[i]
        lim = math.sqrt(max(rem, 0.0) / bn) + 1e-9
        for xi in range(math.floor(centre - lim), math.ceil(centre + lim) + 1):
            d = (xi - centre) ** 2 * bn
            if d > rem + EPS:
                continue
            x[i] = xi
            noff = list(off)
            for j in range(i):
                noff[j] = off[j] - (xi - c[i]) * mu[i][j]
            rec(i - 1, rem - d, noff)
        x[i] = 0

    rec(n - 1, float(R2) + EPS, list(c))
    pts, d2 = [], []
    for coef in out:
        p = [sum(coef[i] * B[i][k] for i in range(n)) for k in range(N)]
        dd = sum((p[k] - target[k]) ** 2 for k in range(N))       # exact integers
        if dd <= R2:
            pts.append(p)
            d2.append(dd)
    return pts, d2

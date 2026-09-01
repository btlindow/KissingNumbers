#!/usr/bin/env python3
"""Enumerate the lattice points on a sphere by Fincke-Pohst, assuming nothing.

A derivation can tell you which Leech norms are admissible on a given sphere and hence how
many points it carries; this script checks such a derivation by enumeration instead.  It scales by 3 -- so the problem becomes
"enumerate y in 3L with |y - 2v|^2 <= 192", all integer -- LLL-reduces a basis of 3L, and
runs a standard Fincke-Pohst enumeration in exact rational arithmetic.  It then reports
every point found, its norm, and whether it lies on the sphere.

L is the integer lattice spanned by the 196560 minimal vectors in the sqrt-8 scaling
(det 2^36); a modular-HNF basis is built from scratch here.
"""
from __future__ import annotations
import os, sys, json
from fractions import Fraction as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "python", "kiss_ref")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools", "structure"))
from kiss_ref.leech import leech_min_vectors, is_lattice_vector_rows  # noqa: E402
from r_subspace_vs_channels import modular_hnf                        # noqa: E402
import sympy as sp

SYS = os.path.join(ROOT, "data", "norm6_simplex24.json")
N = 24


def lll(B, delta=F(3, 4)):
    """LLL on integer rows, exact rationals."""
    B = [list(map(int, r)) for r in B]
    n = len(B)

    def gso():
        Bs, mu = [], [[F(0)] * n for _ in range(n)]
        for i in range(n):
            v = [F(t) for t in B[i]]
            for j in range(i):
                nj = sum(x * x for x in Bs[j])
                mu[i][j] = (sum(F(B[i][k]) * Bs[j][k] for k in range(N)) / nj) if nj else F(0)
                v = [v[k] - mu[i][j] * Bs[j][k] for k in range(N)]
            Bs.append(v)
        return Bs, mu

    Bs, mu = gso()
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = mu[k][j]
            r = int(q + F(1, 2)) if q >= 0 else -int(-q + F(1, 2))
            if r:
                B[k] = [B[k][t] - r * B[j][t] for t in range(N)]
                Bs, mu = gso()
        nk = sum(x * x for x in Bs[k])
        nk1 = sum(x * x for x in Bs[k - 1])
        if nk >= (delta - mu[k][k - 1] ** 2) * nk1:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gso()
            k = max(k - 1, 1)
    return B


def fincke_pohst(B, target, R2):
    """all x in the lattice spanned by rows of B with |x - target|^2 <= R2 (exact)."""
    n = len(B)
    Bs, mu = [], [[F(0)] * n for _ in range(n)]
    for i in range(n):
        v = [F(t) for t in B[i]]
        for j in range(i):
            nj = sum(x * x for x in Bs[j])
            mu[i][j] = sum(F(B[i][k]) * Bs[j][k] for k in range(N)) / nj
            v = [v[k] - mu[i][j] * Bs[j][k] for k in range(N)]
        Bs.append(v)
    Bn = [sum(x * x for x in b) for b in Bs]
    # coordinates of the target in the basis (rational)
    import sympy as sp
    T = sp.Matrix([[int(t) for t in row] for row in B])
    tv = sp.Matrix([[F(int(t)).numerator for t in target]])
    sol = T.T.solve(tv.T)                       # target = sum c_i b_i
    c = [F(int(sp.nsimplify(x).p), int(sp.nsimplify(x).q)) for x in sol]

    out = []
    x = [0] * n

    def rec(i, rem, off):
        """off[j] = c_j + sum_{k>i} (x_k - c_k) mu[k][j]; walk i = n-1 .. 0."""
        if i < 0:
            out.append(list(x))
            return
        centre = off[i]
        # exact integer range: (xi - centre)^2 * Bn[i] <= rem
        import math
        w = rem / Bn[i]                       # >= 0
        num, den = w.numerator, w.denominator
        rt = F(math.isqrt(num * den) + 1, den)   # >= sqrt(w)
        lo = math.floor(centre - rt)
        hi = math.ceil(centre + rt)
        for xi in range(lo, hi + 1):
            d = (F(xi) - centre) ** 2 * Bn[i]
            if d > rem:
                continue
            x[i] = xi
            noff = list(off)
            for j in range(i):
                # centre_j = c_j - sum_{k>j} (x_k - c_k) mu[k][j]
                noff[j] = off[j] - (F(xi) - c[i]) * mu[i][j]
            rec(i - 1, rem - d, noff)
        x[i] = 0

    rec(n - 1, F(R2), list(c))
    return [[sum(coef[i] * B[i][k] for i in range(n)) for k in range(N)] for coef in out]


def main():
    ok = True
    C = leech_min_vectors().astype(np.int64)
    v = np.array(json.load(open(SYS))["vectors"], dtype=np.int64)[0]
    DET = 1 << 36
    rows = [[DET if i == j else 0 for j in range(N)] for i in range(N)]
    rng = np.random.default_rng(3)
    rows += C[rng.choice(len(C), 400, replace=False)].tolist()
    B = modular_hnf(rows)
    B3 = [[3 * t for t in r] for r in B]                     # a basis of 3L
    B3 = lll(B3)
    det = abs(int(sp.Matrix([[int(t) for t in r] for r in B3]).det()))
    print(f"RESULT A2 LLL-reduced basis of 3L: |det| ~ {det} (expected 3^24 * 2^36 = "
          f"{3**24 * 2**36}); shortest row norm^2 = "
          f"{min(sum(t*t for t in r) for r in B3)} (min of 3L is 9*32 = 288)")

    s = (2 * v).tolist()
    pts = fincke_pohst(B3, s, 192)
    print(f"RESULT A2 Fincke-Pohst found {len(pts)} points y in 3L with |y - 2v|^2 <= 192",
          flush=True)
    d2 = [sum((pts[i][k] - s[k]) ** 2 for k in range(N)) for i in range(len(pts))]
    on = [i for i, t in enumerate(d2) if t == 192]
    print(f"RESULT A2 distances^2 found: {sorted(set(d2))}; on the sphere (= 192): {len(on)}")
    X = np.array([[pts[i][k] // 3 for k in range(N)] for i in on],
                 dtype=np.int64).reshape(-1, N)
    div = all(pts[i][k] % 3 == 0 for i in on for k in range(N))
    inlat = bool(np.all(is_lattice_vector_rows(X)))
    norms = sorted(set(map(int, (X * X).sum(1))))
    zero = int((np.abs(X).sum(1) == 0).sum())
    print(f"RESULT A2 y/3 integral: {div}; all in Lambda: {inlat}; |x|^2 values {norms}; "
          f"x = 0 present: {zero == 1}")
    W = C[C @ v == 24]
    match = {tuple(r) for r in map(tuple, X)} == ({tuple(r) for r in map(tuple, W)} |
                                                  {tuple([0] * N)})
    print(f"RESULT A2 ENUMERATION: {len(X)} points = {{0}} u W with |W| = {len(W)}: {match}")
    ok &= len(X) == 553 and match and inlat and div
    print(f"RESULT task6-A2 ok={int(ok)} sphere_size={len(X)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

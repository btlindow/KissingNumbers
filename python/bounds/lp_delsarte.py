"""Delsarte-Goethals-Seidel LP bound for spherical codes in R^n with a
restricted set A of allowed pairwise inner products (docs/design.md T2.1).

Theorem (DGS 1977).  Let X be a finite set of unit vectors in R^n whose
pairwise inner products (x != y) all lie in A subset [-1, 1).  If
f(t) = sum_{k=0}^d f_k G_k(t) with f_0 > 0, f_k >= 0 for k >= 1 and
f(t) <= 0 for every t in A, then |X| <= f(1) / f_0.

(Proof sketch: sum_{x,y in X} f(<x,y>) = |X| f(1) + sum_{x != y} f(<x,y>)
<= |X| f(1); and the same sum equals sum_k f_k sum_{x,y} G_k(<x,y>)
>= f_0 |X|^2 by positive-definiteness of the G_k on the sphere.)

LP (f_0 = 1): minimise 1 + sum_{k>=1} f_k subject to f_k >= 0 and
1 + sum_k f_k G_k(t) <= 0 for each t in A.  HiGHS gives a float solution;
`rationalise` turns it into an EXACT rational certificate by rounding the
f_k (k >= 1) with Fraction.limit_denominator and then recomputing f_0
exactly as f_0 = -max_{t in A} sum_{k>=1} f_k G_k(t), so that f(t) <= 0 holds
on A by construction.  `verify_certificate` re-checks every condition of the
theorem independently in exact arithmetic and returns floor(f(1)/f_0).

CLI:  .venv/bin/python python/bounds/lp_delsarte.py  [--grid]
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

if __package__ in (None, ""):  # run as a script: python python/bounds/lp_delsarte.py
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from bounds.gegenbauer import DEFAULT_N, gegenbauer_values  # noqa: E402
else:
    from .gegenbauer import DEFAULT_N, gegenbauer_values

F = Fraction

# README section 3 W1 item 1: S/2 has pairwise inner products in A (cos 60 deg = 1/2 excluded).
A_RESTRICTED: tuple[Fraction, ...] = (F(-1), F(-1, 2), F(-1, 4), F(0), F(1, 4))
# All inner products between distinct Leech minimal vectors (no restriction).
A_ALL_LEECH: tuple[Fraction, ...] = A_RESTRICTED + (F(1, 2),)
# README section 8 parking-lot item 2: "cos <= 0" variant.
A_NONPOS: tuple[Fraction, ...] = A_RESTRICTED[:4]

DEGREES = (10, 20, 30, 40, 60)
DENOMINATORS = (10**3, 10**4, 10**6, 10**8, 10**10, 10**12, 10**15, 10**18)


def grid_A(step_denominator: int = 2000, upper: Fraction = F(1, 2)) -> tuple[Fraction, ...]:
    """{-1, -1 + 1/step, ..., upper} -- the classical 'cos <= 1/2' constraint on a grid."""
    lo = -step_denominator
    hi = int(upper * step_denominator)
    assert F(hi, step_denominator) == upper, "upper must be a grid point"
    return tuple(F(i, step_denominator) for i in range(lo, hi + 1))


def gegenbauer_matrix(A: Sequence[Fraction], d: int, n: int = DEFAULT_N) -> list[list[Fraction]]:
    """Exact matrix M[t][k] = G_k(t) for t in A, k = 0..d."""
    return [gegenbauer_values(d, t, n) for t in A]


# --------------------------------------------------------------------------- float LP


@dataclass
class LPResult:
    A: tuple[Fraction, ...]
    d: int
    n: int
    status: int
    message: str
    bound: float  # 1 + sum f_k (float), or inf if not solved
    f: np.ndarray  # f_0 .. f_d as floats (f_0 = 1)
    matrix: list[list[Fraction]] = field(repr=False)


def solve_lp(A: Sequence[Fraction], d: int, n: int = DEFAULT_N) -> LPResult:
    """HiGHS solve of: min 1 + sum_{k=1}^d f_k, s.t. f_k >= 0,
    sum_{k=1}^d f_k G_k(t) <= -1 for all t in A."""
    A = tuple(Fraction(t) for t in A)
    M = gegenbauer_matrix(A, d, n)
    Af = np.array([[float(M[i][k]) for k in range(1, d + 1)] for i in range(len(A))])
    bf = -np.ones(len(A))
    c = np.ones(d)
    res = linprog(c, A_ub=Af, b_ub=bf, bounds=[(0, None)] * d, method="highs")
    if res.status == 0:
        f = np.concatenate([[1.0], res.x])
        bound = 1.0 + float(res.x.sum())
    else:
        f = np.full(d + 1, np.nan)
        bound = math.inf
    return LPResult(A, d, n, res.status, res.message, bound, f, M)


# --------------------------------------------------------------------------- exact side


@dataclass
class Certificate:
    """An exact DGS certificate: f(t) = sum_k f[k] G_k(t)."""

    A: tuple[Fraction, ...]
    n: int
    f: list[Fraction]  # f[0] = f_0 > 0, f[k] >= 0

    @property
    def d(self) -> int:
        return len(self.f) - 1

    @property
    def bound(self) -> Fraction:
        """f(1)/f_0 exactly (G_k(1) = 1 so f(1) = sum f_k)."""
        return sum(self.f, Fraction(0)) / self.f[0]

    @property
    def int_bound(self) -> int:
        return math.floor(self.bound)

    def values_on_A(self) -> list[Fraction]:
        return [
            sum((fk * gk for fk, gk in zip(self.f, gegenbauer_values(self.d, t, self.n))), Fraction(0))
            for t in self.A
        ]

    def support(self) -> list[int]:
        return [k for k, fk in enumerate(self.f) if fk != 0]


def verify_certificate(f: Sequence, A: Sequence, n: int = DEFAULT_N) -> Fraction:
    """Independent exact check of every hypothesis of the DGS theorem.
    Returns the exact bound f(1)/f_0; raises ValueError if any condition fails.
    Does not use anything from `rationalise` except the numbers themselves."""
    f = [Fraction(x) for x in f]
    A = [Fraction(t) for t in A]
    if not f or f[0] <= 0:
        raise ValueError("f_0 must be > 0")
    if any(fk < 0 for fk in f[1:]):
        raise ValueError("all f_k (k >= 1) must be >= 0")
    if any(not (-1 <= t < 1) for t in A):
        raise ValueError("A must lie in [-1, 1)")
    d = len(f) - 1
    for t in A:
        g = gegenbauer_values(d, t, n)
        val = sum((fk * gk for fk, gk in zip(f, g)), Fraction(0))
        if val > 0:
            raise ValueError(f"f({t}) = {val} > 0")
    return sum(f, Fraction(0)) / f[0]


def rationalise(
    lp: LPResult,
    denominators: Sequence[int] = DENOMINATORS,
    zero_tol: float = 1e-12,
) -> Certificate:
    """Turn a float LP solution into an exact certificate.

    For each candidate denominator D: round f_k (k >= 1) to Fraction with
    limit_denominator(D) (values below zero_tol are snapped to 0), then set
    f_0 := -max_{t in A} sum_{k>=1} f_k G_k(t) exactly.  f_0 > 0 is required
    (it is ~1 since the LP enforced the sum <= -1); with this f_0 the
    condition f(t) <= 0 on A holds by construction, and f(1)/f_0 =
    (f_0 + sum f_k)/f_0 is the exact bound.  Returns the candidate with the
    smallest exact bound.  Every returned certificate passes verify_certificate."""
    if lp.status != 0:
        raise ValueError(f"LP not solved: {lp.message}")
    best: Certificate | None = None
    for D in denominators:
        fk = [Fraction(0)] + [
            Fraction(0) if abs(x) < zero_tol else Fraction(float(x)).limit_denominator(D)
            for x in lp.f[1:]
        ]
        g_on_A = [
            sum((fk[k] * lp.matrix[i][k] for k in range(1, lp.d + 1)), Fraction(0))
            for i in range(len(lp.A))
        ]
        f0 = -max(g_on_A)
        if f0 <= 0:
            continue
        fk[0] = f0
        cert = Certificate(lp.A, lp.n, fk)
        verify_certificate(cert.f, cert.A, cert.n)  # must not raise
        if best is None or cert.bound < best.bound:
            best = cert
    if best is None:
        raise ValueError("no denominator produced a feasible exact certificate")
    return best


def exact_vertex(lp: LPResult, support_tol: float = 1e-9, active_tol: float = 1e-7) -> Certificate | None:
    """Try to recover the *exact* LP optimum: the LP optimum is a vertex, so
    the support K of f (k with f_k > 0) and the set T of active constraints
    (f(t) = 0) satisfy |K| <= |T|.  If |K| == |T| we solve the square system
    sum_{k in K} f_k G_k(t) = -1 (t in T) exactly (Fraction Gauss-Jordan) and
    verify feasibility.  Returns None if the structure is not clean."""
    if lp.status != 0:
        return None
    K = [k for k in range(1, lp.d + 1) if lp.f[k] > support_tol]
    g = np.array([[float(lp.matrix[i][k]) for k in range(1, lp.d + 1)] for i in range(len(lp.A))])
    vals = 1.0 + g @ lp.f[1:]
    T = [i for i in range(len(lp.A)) if abs(vals[i]) < active_tol * max(1.0, abs(lp.f[1:]).max())]
    if len(K) != len(T) or not K:
        return None
    # Gauss-Jordan on the |T| x |K| system.
    m = len(K)
    aug = [[lp.matrix[i][k] for k in K] + [Fraction(-1)] for i in T]
    for col in range(m):
        piv = next((r for r in range(col, m) if aug[r][col] != 0), None)
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(m):
            if r != col and aug[r][col] != 0:
                fac = aug[r][col]
                aug[r] = [x - fac * y for x, y in zip(aug[r], aug[col])]
    f = [Fraction(0)] * (lp.d + 1)
    f[0] = Fraction(1)
    for j, k in enumerate(K):
        f[k] = aug[j][m]
    try:
        verify_certificate(f, lp.A, lp.n)
    except ValueError:
        return None
    return Certificate(lp.A, lp.n, f)


# --------------------------------------------------------------------------- driver


@dataclass
class BoundRow:
    d: int
    lp: LPResult
    cert: Certificate | None  # limit_denominator certificate
    vertex: Certificate | None  # exact vertex re-solve, if the structure was clean

    @property
    def best(self) -> Certificate | None:
        cands = [c for c in (self.cert, self.vertex) if c is not None]
        return min(cands, key=lambda c: c.bound) if cands else None


def delsarte_bound(A: Sequence[Fraction], degrees: Sequence[int] = DEGREES, n: int = DEFAULT_N) -> list[BoundRow]:
    rows = []
    for d in degrees:
        lp = solve_lp(A, d, n)
        cert = rationalise(lp) if lp.status == 0 else None
        vtx = exact_vertex(lp)
        rows.append(BoundRow(d, lp, cert, vtx))
    return rows


def format_certificate(cert: Certificate) -> str:
    lines = [f"n = {cert.n}, degree {cert.d}, A = {{{', '.join(str(t) for t in cert.A)}}}"]
    for k in cert.support():
        lines.append(f"  f_{k:<3d} = {cert.f[k]}")
    lines.append("  values f(t) on A:")
    for t, v in zip(cert.A, cert.values_on_A()):
        lines.append(f"    f({str(t):>4s}) = {v}")
    lines.append(f"  f(1)/f_0 = {cert.bound} = {float(cert.bound):.6f}  ->  floor = {cert.int_bound}")
    return "\n".join(lines)


def print_table(name: str, A: Sequence[Fraction], degrees: Sequence[int] = DEGREES, n: int = DEFAULT_N, show_cert: bool = True) -> list[BoundRow]:
    print(f"== {name}: A = {{{', '.join(str(t) for t in A)}}}, n = {n}")
    print(f"{'d':>4s}  {'HiGHS float':>16s}  {'exact (limit_denominator)':>28s}  {'exact vertex':>28s}  floor")
    rows = delsarte_bound(A, degrees, n)
    for r in rows:
        c = r.cert
        v = r.vertex
        cs = f"{float(c.bound):.6f}" if c else "n/a"
        vs = f"{float(v.bound):.6f}" if v else "n/a"
        fl = r.best.int_bound if r.best else "n/a"
        st = "" if r.lp.status == 0 else f"  [LP status {r.lp.status}: {r.lp.message}]"
        print(f"{r.d:4d}  {r.lp.bound:16.6f}  {cs:>28s}  {vs:>28s}  {fl}{st}")
    ok = [r for r in rows if r.best is not None]
    if ok and show_cert:
        b = min(ok, key=lambda r: r.best.bound)
        print(f"-- best exact certificate (degree {b.d}):")
        print(format_certificate(b.best))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--grid", action="store_true", help="also run the classical cos<=1/2 grid sanity LP")
    p.add_argument("--degrees", type=int, nargs="*", default=list(DEGREES))
    args = p.parse_args(argv)

    rows = print_table("restricted (README W1)", A_RESTRICTED, args.degrees)
    print()
    rows_all = print_table("all Leech inner products", A_ALL_LEECH, args.degrees, show_cert=False)
    print()
    rows_np = print_table("cos <= 0 variant (README section 8)", A_NONPOS, args.degrees)
    grid_val = None
    if args.grid:
        print()
        G = grid_A(2000)
        print(f"== classical Delsarte kissing LP on grid, |A| = {len(G)}")
        for d in (20, 30, 40):
            lp = solve_lp(G, d)
            print(f"{d:4d}  {lp.bound:16.6f}  status={lp.status}")
            if lp.status == 0:
                grid_val = lp.bound if grid_val is None else max(grid_val, lp.bound)

    def best_int(rs):
        ok = [r.best.int_bound for r in rs if r.best]
        return min(ok) if ok else "nan"

    print()
    print(
        f"RESULT bound_restricted={best_int(rows)} bound_all_leech={best_int(rows_all)} "
        f"bound_nonpos={best_int(rows_np)} grid_classical={grid_val if grid_val is not None else 'skipped'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

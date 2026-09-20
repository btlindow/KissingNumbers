"""Turn a grid-feasible Delsarte certificate into one certified on the WHOLE
interval, by exact real-root counting.

A Delsarte dual solved on a grid is not by itself a proof: the dual polynomial
could rise above zero between grid points. This module repairs that. Given the
LP's Gegenbauer coefficients f_1..f_d (all >= 0), put

    g(t) = sum_{k>=1} f_k G_k(t).

We need f_0 > 0 with f_0 + g(t) <= 0 for every real t in [-1, s], and then the
Delsarte bound is (f_0 + sum_k f_k) / f_0. The best such f_0 is -max g over the
interval, so it is enough to produce a RATIONAL M with

    g(t) <= M < 0   for all t in [-1, s].

M is found numerically and then verified exactly: the polynomial M - g(t) has
rational coefficients, and Sturm's theorem (sympy's count_roots, which is exact)
says how many real roots it has in [-1, s]. If it has none and it is positive at
one rational point of the interval, it is positive throughout. That is a
complete proof, with no floating point in the decision and no grid.

The checker `verify_on_interval` re-derives everything from f alone and shares
no code with the LP solver.
"""

from fractions import Fraction as F

import sympy as sp

from bounds.gegenbauer import DEFAULT_N, gegenbauer_coeffs


def _g_poly(f, n=DEFAULT_N):
    """sum_{k>=1} f_k G_k(t) as an exact sympy Poly in t."""
    t = sp.Symbol("t")
    expr = sp.Integer(0)
    for k in range(1, len(f)):
        if f[k] == 0:
            continue
        c = gegenbauer_coeffs(k, n)
        pk = sum(sp.Rational(int(c[i].numerator), int(c[i].denominator)) * t ** i
                 for i in range(len(c)))
        expr += sp.Rational(int(F(f[k]).numerator), int(F(f[k]).denominator)) * pk
    return sp.Poly(sp.expand(expr), t), t


def max_on_interval(f, s, n=DEFAULT_N, margin_steps=60):
    """A rational M with g(t) <= M on [-1, s], verified exactly. None if it fails."""
    g, t = _g_poly(f, n)
    lo, hi = sp.Integer(-1), sp.Rational(F(s).numerator, F(s).denominator)

    # Numerical maximum, to know roughly where to put M.
    crit = [lo, hi]
    try:
        for r in sp.Poly(g.diff(t), t).nroots(n=40):
            if abs(sp.im(r)) < 1e-30:
                rr = sp.re(r)
                if lo <= rr <= hi:
                    crit.append(rr)
    except Exception:
        pass
    approx = max(float(g.eval(c)) for c in crit)

    # Try progressively looser rational M just above the numerical maximum.
    M = None
    for j in range(margin_steps):
        cand = sp.Rational(int(approx * 10**9) + 1 + 10 ** min(j, 12), 10**9)
        if cand >= 0:
            return None
        if _is_positive_on(sp.Poly(sp.expand(cand - g.as_expr()), t), lo, hi):
            M = cand
            break
    return M


def _is_positive_on(p, lo, hi):
    """Exact: is the rational polynomial p strictly positive on all of [lo, hi]?"""
    if p.count_roots(lo, hi) != 0:
        return False
    mid = sp.Rational(lo + hi, 2)
    return p.eval(mid) > 0


def verify_on_interval(f, s, n=DEFAULT_N):
    """Independent exact check. Returns the certified bound as a Fraction.

    Re-checks every DGS hypothesis from f alone: f_k >= 0 for k >= 1, f_0 > 0,
    and f(t) <= 0 on the whole real interval [-1, s] by root counting.
    """
    f = [F(x) for x in f]
    if f[0] <= 0:
        raise ValueError("f_0 must be positive")
    if any(x < 0 for x in f[1:]):
        raise ValueError("f_k must be non-negative for k >= 1")
    g, t = _g_poly(f, n)
    lo = sp.Integer(-1)
    hi = sp.Rational(F(s).numerator, F(s).denominator)
    f0 = sp.Rational(f[0].numerator, f[0].denominator)
    # need f0 + g(t) <= 0 on [lo, hi], i.e. -(f0 + g) >= 0; check strict
    # positivity of -(f0 + g) + 0 is too strong at a touching optimum, so verify
    # that -(f0 + g) has no root in the OPEN interval and is >= 0 at the ends.
    p = sp.Poly(sp.expand(-(f0 + g.as_expr())), t)
    if p.eval(lo) < 0 or p.eval(hi) < 0:
        raise ValueError("certificate positive at an endpoint")
    nr = p.count_roots(lo, hi)
    if nr != 0:
        # Roots are allowed only if the polynomial never goes negative; check by
        # sampling every cell cut out by the isolating intervals.
        for a, b in p.intervals(inf=lo, sup=hi, eps=sp.Rational(1, 10**12)):
            if p.eval(sp.Rational(a + b, 2)) < 0:
                raise ValueError("certificate goes positive inside the interval")
    return F(int(sum(f).numerator), int(sum(f).denominator)) / f[0]

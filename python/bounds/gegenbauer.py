"""Exact rational Gegenbauer polynomials for the sphere S^{n-1} (docs/design.md T2.1).

Convention (Delsarte-Goethals-Seidel): G_k = G_k^{(n)} is the Gegenbauer
polynomial C_k^{(alpha)} with alpha = (n-2)/2, normalised so that G_k(1) = 1.
They are orthogonal on [-1, 1] with weight (1 - t^2)^{(n-3)/2}.

Three-term recurrence (exact in Fractions):

    G_0(t) = 1,  G_1(t) = t,
    (k + n - 2) G_{k+1}(t) = (2k + n - 2) t G_k(t) - k G_{k-1}(t).

For n = 3 this is Legendre's (k+1) P_{k+1} = (2k+1) t P_k - k P_{k-1}.

Polynomials are represented as lists of Fractions, index i = coefficient of t^i.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Sequence

DEFAULT_N = 24  # S^23: the sphere in R^24, alpha = 11


def _frac(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


@lru_cache(maxsize=None)
def _table(d: int, n: int) -> tuple[tuple[Fraction, ...], ...]:
    if d < 0:
        raise ValueError("degree must be >= 0")
    if n < 3:
        raise ValueError("n must be >= 3")
    rows: list[list[Fraction]] = [[Fraction(1)]]
    if d >= 1:
        rows.append([Fraction(0), Fraction(1)])
    for k in range(1, d):
        gk, gk1 = rows[k], rows[k - 1]
        a = Fraction(2 * k + n - 2, k + n - 2)  # multiplies t*G_k
        b = Fraction(k, k + n - 2)  # multiplies G_{k-1}
        new = [Fraction(0)] * (k + 2)
        for i, c in enumerate(gk):
            new[i + 1] += a * c
        for i, c in enumerate(gk1):
            new[i] -= b * c
        rows.append(new)
    return tuple(tuple(r) for r in rows)


def gegenbauer_table(d: int, n: int = DEFAULT_N) -> list[list[Fraction]]:
    """Monomial coefficients of G_0 .. G_d for S^{n-1}; row k has length k+1."""
    return [list(r) for r in _table(d, n)]


def gegenbauer_coeffs(k: int, n: int = DEFAULT_N) -> list[Fraction]:
    """Monomial coefficients (index i = coefficient of t^i) of G_k^{(n)}."""
    return list(_table(k, n)[k])


def poly_eval(coeffs: Sequence[Fraction], t) -> Fraction:
    """Exact Horner evaluation of a coefficient list at a rational t."""
    t = _frac(t)
    acc = Fraction(0)
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def gegenbauer_eval(k: int, t, n: int = DEFAULT_N) -> Fraction:
    """G_k^{(n)}(t) exactly, by running the recurrence at the point t."""
    t = _frac(t)
    if k == 0:
        return Fraction(1)
    prev, cur = Fraction(1), t
    for j in range(1, k):
        prev, cur = cur, ((2 * j + n - 2) * t * cur - j * prev) / (j + n - 2)
    return cur


def gegenbauer_values(d: int, t, n: int = DEFAULT_N) -> list[Fraction]:
    """[G_0(t), ..., G_d(t)] exactly, one pass of the recurrence."""
    t = _frac(t)
    out = [Fraction(1)]
    if d >= 1:
        out.append(t)
    for j in range(1, d):
        out.append(((2 * j + n - 2) * t * out[j] - j * out[j - 1]) / (j + n - 2))
    return out


def to_gegenbauer_basis(coeffs: Sequence, n: int = DEFAULT_N) -> list[Fraction]:
    """Exact change of basis: monomial coefficients -> [f_0, ..., f_d] with
    sum_k f_k G_k(t) == the input polynomial.  Back-substitution on the
    upper-triangular matrix of leading coefficients (G_k has degree exactly k)."""
    c = [_frac(x) for x in coeffs]
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    d = len(c) - 1
    table = _table(d, n)
    f = [Fraction(0)] * (d + 1)
    rem = list(c)
    for k in range(d, -1, -1):
        lead = table[k][k]
        f[k] = rem[k] / lead
        for i in range(k + 1):
            rem[i] -= f[k] * table[k][i]
    assert all(r == 0 for r in rem)
    return f


def from_gegenbauer_basis(f: Sequence, n: int = DEFAULT_N) -> list[Fraction]:
    """Inverse of to_gegenbauer_basis: [f_0..f_d] -> monomial coefficients."""
    f = [_frac(x) for x in f]
    d = len(f) - 1
    table = _table(d, n)
    out = [Fraction(0)] * (d + 1)
    for k, fk in enumerate(f):
        if fk:
            for i, c in enumerate(table[k]):
                out[i] += fk * c
    return out


def weight_exponent(n: int = DEFAULT_N) -> Fraction:
    """Exponent e in the orthogonality weight (1 - t^2)^e, e = (n-3)/2."""
    return Fraction(n - 3, 2)

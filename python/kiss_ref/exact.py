"""Exact arithmetic in multi-quadratic number fields Q(sqrt p_1, sqrt p_2, ...) (docs/design.md T4.3).

The R^d half of the CJKT / PackingStar template lives in small number fields:
our own integer models are rational, PackingStar's extra spheres are copies of
the kissing configuration rotated by a matrix over Q(sqrt 3) (d even) or
Q(sqrt 2) (d odd). Every angle condition of the template is a comparison of
the form ``4 p^2 <= c * Na * Nb`` with p = <a,b>, Na = |a|^2, Nb = |b|^2 and
c in {1, 3}; all of these are decided here exactly.

Representation: an element is a dict {r: Fraction} meaning sum_r c_r * sqrt(r)
over square-free positive integers r (r = 1 is the rational part). The set
{sqrt r : r square-free} is linearly independent over Q, so the normalised
dict (zero coefficients dropped) is a canonical form and the zero test is
exact. Products use sqrt(r1) sqrt(r2) = s sqrt(r) with r1 r2 = s^2 r.

Sign of a nonzero element: rational -> trivial; a + b sqrt(r) -> compare a^2
with r b^2; more radicals -> rational interval arithmetic with isqrt-based
bounds on the square roots, refined until the interval excludes 0 (an element
with a canonical nonzero representation is nonzero, so this terminates; a
guard raises after 400 digits).

Pure Python; no numpy needed. Everything here is O(1)-sized (vectors of
length <= 8, at most a few hundred of them), so speed is irrelevant.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, Sequence

__all__ = ["Q", "squarefree_split", "parse_entry", "parse_vector", "dot", "norm2", "rank",
           "cos_le_half", "cos_le_neg_half", "cos_le_sqrt3_half", "is_antipodal", "is_same_direction",
           "to_float"]

Number = int | Fraction


@lru_cache(maxsize=None)
def squarefree_split(n: int) -> tuple[int, int]:
    """n = s^2 * r with r square-free; returns (s, r). n >= 1."""
    if n < 1:
        raise ValueError("squarefree_split needs n >= 1")
    s, r, m, p = 1, 1, n, 2
    while p * p <= m:
        while m % (p * p) == 0:
            s *= p
            m //= p * p
        if m % p == 0:
            r *= p
            m //= p
        p += 1
    r *= m
    return s, r


@lru_cache(maxsize=None)
def _primes_of(r: int) -> tuple[int, ...]:
    out, p, m = [], 2, r
    while p * p <= m:
        if m % p == 0:
            out.append(p)
            while m % p == 0:
                m //= p
        p += 1
    if m > 1:
        out.append(m)
    return tuple(out)


class Q:
    """An element of Q(sqrt p_1, ..., sqrt p_k); immutable."""

    __slots__ = ("c",)

    def __init__(self, c=None):
        if c is None:
            self.c = {}
        elif isinstance(c, Q):
            self.c = c.c
        elif isinstance(c, dict):
            self.c = {int(r): Fraction(v) for r, v in c.items() if v != 0}
        else:
            v = Fraction(c)
            self.c = {1: v} if v != 0 else {}

    # ---- constructors --------------------------------------------------
    @staticmethod
    def sqrt(m: int, coeff=1) -> "Q":
        """coeff * sqrt(m) for an integer m >= 0."""
        m = int(m)
        if m < 0:
            raise ValueError("sqrt of a negative integer")
        if m == 0:
            return Q()
        s, r = squarefree_split(m)
        return Q({r: Fraction(coeff) * s})

    # ---- basic properties ----------------------------------------------
    def is_zero(self) -> bool:
        return not self.c

    def is_rational(self) -> bool:
        return set(self.c) <= {1}

    def rational(self) -> Fraction:
        if not self.is_rational():
            raise ValueError(f"{self} is not rational")
        return self.c.get(1, Fraction(0))

    def radicals(self) -> tuple[int, ...]:
        return tuple(sorted(r for r in self.c if r != 1))

    def __float__(self) -> float:
        return float(sum(float(v) * math.sqrt(r) for r, v in self.c.items()))

    def __repr__(self) -> str:
        if not self.c:
            return "0"
        parts = []
        for r in sorted(self.c):
            v = self.c[r]
            parts.append(f"{v}" if r == 1 else f"{v}*sqrt({r})")
        return " + ".join(parts)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.c.items())))

    # ---- arithmetic ------------------------------------------------------
    def __add__(self, o) -> "Q":
        o = o if isinstance(o, Q) else Q(o)
        c = dict(self.c)
        for r, v in o.c.items():
            c[r] = c.get(r, 0) + v
        return Q(c)

    __radd__ = __add__

    def __neg__(self) -> "Q":
        return Q({r: -v for r, v in self.c.items()})

    def __sub__(self, o) -> "Q":
        return self + (-(o if isinstance(o, Q) else Q(o)))

    def __rsub__(self, o) -> "Q":
        return Q(o) - self

    def __mul__(self, o) -> "Q":
        o = o if isinstance(o, Q) else Q(o)
        c: dict[int, Fraction] = {}
        for r1, v1 in self.c.items():
            for r2, v2 in o.c.items():
                s, r = squarefree_split(r1 * r2)
                c[r] = c.get(r, 0) + v1 * v2 * s
        return Q(c)

    __rmul__ = __mul__

    def conjugates(self) -> list["Q"]:
        """All images under the Galois group generated by sqrt p -> -sqrt p (p prime)."""
        primes = sorted({p for r in self.c for p in _primes_of(r)})
        out = []
        for mask in range(1 << len(primes)):
            flip = {p for k, p in enumerate(primes) if (mask >> k) & 1}
            c = {}
            for r, v in self.c.items():
                sign = -1 if sum(1 for p in _primes_of(r) if p in flip) % 2 else 1
                c[r] = sign * v
            out.append(Q(c))
        return out

    def inv(self) -> "Q":
        if self.is_zero():
            raise ZeroDivisionError("inverse of 0")
        if self.is_rational():
            return Q(1 / self.c[1])
        conj = self.conjugates()
        num = Q(1)
        for z in conj[1:]:
            num = num * z
        den = num * self  # the field norm, rational
        if not den.is_rational():
            raise ArithmeticError("field norm is not rational (bug)")
        return num * Q(1 / den.c[1])

    def __truediv__(self, o) -> "Q":
        o = o if isinstance(o, Q) else Q(o)
        return self * o.inv()

    def __rtruediv__(self, o) -> "Q":
        return Q(o) * self.inv()

    def __pow__(self, n: int) -> "Q":
        out = Q(1)
        for _ in range(int(n)):
            out = out * self
        return out

    # ---- comparisons -----------------------------------------------------
    def sign(self) -> int:
        """-1, 0, +1; exact."""
        c = self.c
        if not c:
            return 0
        if set(c) <= {1}:
            v = c[1]
            return (v > 0) - (v < 0)
        keys = sorted(c)
        if len(keys) == 1:  # b sqrt r
            v = c[keys[0]]
            return (v > 0) - (v < 0)
        if len(keys) == 2 and keys[0] == 1:  # a + b sqrt r
            a, b, r = c[1], c[keys[1]], keys[1]
            sa, sb = (a > 0) - (a < 0), (b > 0) - (b < 0)
            if sa == sb:
                return sa
            # opposite signs: compare a^2 with r b^2
            lhs, rhs = a * a, r * b * b
            if lhs == rhs:
                return 0  # impossible for r square-free > 1, kept for safety
            return sa if lhs > rhs else sb
        # general: rational interval arithmetic
        for digits in (30, 60, 120, 240, 400):
            scale = 10 ** digits
            lo = hi = Fraction(0)
            for r, v in c.items():
                if r == 1:
                    lo += v
                    hi += v
                    continue
                root_lo = Fraction(math.isqrt(r * scale * scale), scale)
                root_hi = root_lo + Fraction(1, scale)
                if v > 0:
                    lo += v * root_lo
                    hi += v * root_hi
                else:
                    lo += v * root_hi
                    hi += v * root_lo
            if lo > 0:
                return 1
            if hi < 0:
                return -1
        raise ArithmeticError(f"sign of {self} undecided at 400 digits")

    def __eq__(self, o) -> bool:
        o = o if isinstance(o, Q) else Q(o)
        return self.c == o.c

    def __ne__(self, o) -> bool:
        return not self.__eq__(o)

    def __lt__(self, o) -> bool:
        return (self - o).sign() < 0

    def __le__(self, o) -> bool:
        return (self - o).sign() <= 0

    def __gt__(self, o) -> bool:
        return (self - o).sign() > 0

    def __ge__(self, o) -> bool:
        return (self - o).sign() >= 0


# ---------------------------------------------------------------------------
# parsing of coordinate entries
# ---------------------------------------------------------------------------
_MAX_FLOAT_DEN = 4096
_TERM = re.compile(r"^\s*(?P<coef>[+-]?\s*(?:\d+(?:/\d+)?)?)\s*(?:\*?\s*sqrt\(?\s*(?P<rad>\d+)\s*\)?)?\s*$")


def _parse_term(t: str) -> Q:
    m = _TERM.match(t)
    if not m or not t.strip():
        raise ValueError(f"cannot parse term {t!r}")
    coef = m.group("coef").replace(" ", "")
    rad = m.group("rad")
    if coef in ("", "+", "-"):
        if rad is None:
            raise ValueError(f"cannot parse term {t!r}")
        f = Fraction(-1 if coef == "-" else 1)
    else:
        f = Fraction(coef)
    return Q.sqrt(int(rad), f) if rad is not None else Q(f)


def _parse_string(s: str) -> Q:
    """'-1/2', '3/4*sqrt(2)', 'sqrt2/2', '(2+sqrt(2))/4', '1/2 - 1/2*sqrt(3)' -> Q (exact)."""
    s = s.strip().replace("√", "sqrt")
    s = re.sub(r"sqrt\s*(\d+)", r"sqrt(\1)", s)
    # (expr)/k  or  (expr)*k
    m = re.match(r"^\((.*)\)\s*([/*])\s*(\d+(?:/\d+)?)$", s)
    if m:
        inner = _parse_string(m.group(1))
        k = Fraction(m.group(3))
        return inner * Q(1 / k if m.group(2) == "/" else k)
    # a/b*sqrt(r)/c
    m = re.match(r"^(.*sqrt\(\d+\))\s*/\s*(\d+)$", s)
    if m:
        return _parse_string(m.group(1)) * Q(Fraction(1, int(m.group(2))))
    # split on top-level +/- (no parentheses left)
    if "(" in s.replace("sqrt(", ""):
        raise ValueError(f"cannot parse {s!r}")
    terms = re.findall(r"[+-]?[^+-]+", s.replace(" ", ""))
    if not terms:
        raise ValueError(f"cannot parse {s!r}")
    out = Q()
    for t in terms:
        out = out + _parse_term(t)
    return out


def parse_entry(e, radical: int | None = None) -> Q:
    """One coordinate entry of a family.json vector -> Q.

    Accepted forms: int; float that is a dyadic rational with denominator
    <= 4096 (anything else is rejected as inexact); str as in
    ``_parse_string``; a 2-list [a, b] meaning a + b*sqrt(radical) (a, b any
    of the above); a dict {"1": a, "2": b, ...} meaning sum a*sqrt(key).
    """
    if isinstance(e, Q):
        return e
    if isinstance(e, bool):
        raise ValueError("boolean entry")
    if isinstance(e, (int, Fraction)):
        return Q(e)
    if isinstance(e, float):
        f = Fraction(e)
        if f.denominator > _MAX_FLOAT_DEN:
            raise ValueError(f"float entry {e!r} is not an exact small rational; use a string like '1/3' or 'sqrt(2)/2'")
        return Q(f)
    if isinstance(e, str):
        return _parse_string(e)
    if isinstance(e, (list, tuple)):
        if len(e) != 2:
            raise ValueError(f"pair entry must have 2 parts: {e!r}")
        if radical is None:
            raise ValueError(f"pair entry {e!r} needs the block's 'sqrt' radical")
        return parse_entry(e[0]) + parse_entry(e[1]) * Q.sqrt(int(radical))
    if isinstance(e, dict):
        out = Q()
        for k, v in e.items():
            out = out + parse_entry(v) * Q.sqrt(int(k))
        return out
    raise ValueError(f"unsupported entry {e!r}")


def parse_vector(v: Sequence, radical: int | None = None, scale: Q | None = None) -> list[Q]:
    """A list of entries -> list of Q, divided by ``scale`` if given."""
    out = [parse_entry(e, radical) for e in v]
    if scale is not None and not (scale.is_rational() and scale.rational() == 1):
        inv = scale.inv()
        out = [x * inv for x in out]
    return out


# ---------------------------------------------------------------------------
# vector helpers and the angle predicates
# ---------------------------------------------------------------------------
def dot(a: Sequence[Q], b: Sequence[Q]) -> Q:
    if len(a) != len(b):
        raise ValueError("dot of vectors of different lengths")
    out = Q()
    for x, y in zip(a, b):
        out = out + x * y
    return out


def norm2(a: Sequence[Q]) -> Q:
    return dot(a, a)


def to_float(rows: Iterable[Sequence[Q]]) -> list[list[float]]:
    return [[float(x) for x in r] for r in rows]


def cos_le_half(p: Q, Na: Q, Nb: Q) -> bool:
    """cos <= 1/2  <=>  p <= 0  or  4 p^2 <= Na Nb."""
    return p.sign() <= 0 or (Na * Nb - p * p * 4).sign() >= 0


def cos_le_neg_half(p: Q, Na: Q, Nb: Q) -> bool:
    """cos <= -1/2  <=>  p < 0  and  4 p^2 >= Na Nb."""
    return p.sign() < 0 and (p * p * 4 - Na * Nb).sign() >= 0


def cos_le_sqrt3_half(p: Q, Na: Q, Nb: Q) -> bool:
    """cos <= sqrt(3)/2  <=>  p <= 0  or  4 p^2 <= 3 Na Nb."""
    return p.sign() <= 0 or (Na * Nb * 3 - p * p * 4).sign() >= 0


def is_antipodal(p: Q, Na: Q, Nb: Q) -> bool:
    return p.sign() < 0 and (p * p - Na * Nb).is_zero()


def is_same_direction(p: Q, Na: Q, Nb: Q) -> bool:
    return p.sign() > 0 and (p * p - Na * Nb).is_zero()


def rank(rows: Sequence[Sequence[Q]]) -> int:
    """Exact rank by Gaussian elimination over the field."""
    M = [list(r) for r in rows]
    if not M:
        return 0
    ncol = len(M[0])
    r = 0
    for col in range(ncol):
        piv = next((i for i in range(r, len(M)) if not M[i][col].is_zero()), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = M[r][col].inv()
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and not M[i][col].is_zero():
                f = M[i][col]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        r += 1
        if r == len(M):
            break
    return r

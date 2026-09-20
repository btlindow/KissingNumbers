"""Exact comparisons for the small algebraic numbers these verifications need.

Everything here decides a strict/non-strict inequality between a rational and a
number of the form  a + b*sqrt(r)  with no floating point anywhere.  The pattern
is always the same: isolate the surd, branch on the sign of each side, and then
compare squares, which is exact because both sides are then non-negative
rationals.

Used by verify25_independent.py and verify2627_independent.py.  Written here,
sharing no code with A. Kravatskiy's verify.py.
"""

from fractions import Fraction

import numpy as np


def le_rat_times_sqrt(L, M, r):
    """Decide  L <= M * sqrt(r)  for rationals/ints L, M and a non-negative int r.

    Both branches reduce to a comparison of non-negative rationals, so the test
    is exact.
    """
    if M >= 0:
        if L <= 0:
            return True
        return L * L <= r * M * M
    # M < 0, so the right-hand side is negative.
    if L >= 0:
        return False
    return L * L >= r * M * M


def le_rat_times_sqrt_vec(L, M, r):
    """Vectorised `le_rat_times_sqrt` over int64 numpy arrays."""
    L = np.asarray(L, dtype=np.int64)
    M = np.asarray(M, dtype=np.int64)
    pos = M >= 0
    sq = L * L <= r * M * M
    sq_rev = L * L >= r * M * M
    return np.where(
        pos,
        np.where(L <= 0, True, sq),
        np.where(L >= 0, False, sq_rev),
    )


def sign_a_plus_b_sqrt(a, b, r):
    """Exact sign of  a + b*sqrt(r)  for rationals a, b and non-negative int r."""
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    # Opposite signs: compare a^2 against r*b^2.
    lhs = a * a
    rhs = r * b * b
    if a > 0:  # b < 0, value positive iff a^2 > r b^2
        if lhs > rhs:
            return 1
        if lhs < rhs:
            return -1
        return 0
    # a < 0, b > 0: value positive iff r b^2 > a^2
    if rhs > lhs:
        return 1
    if rhs < lhs:
        return -1
    return 0


def le_s_sqrt2_plus_t_sqrt6(s, t, R):
    """Decide  s*sqrt(2) + t*sqrt(6) <= R  for rationals s, t, R.

    Factor the left side as sqrt(2) * (s + t*sqrt(3)) and branch on the exact
    sign of the inner Q(sqrt 3) element, which reduces the whole test to
    rational comparisons and one further  rational <= rational*sqrt(3)  call.
    """
    sgn = sign_a_plus_b_sqrt(s, t, 3)
    if sgn <= 0:
        # Left side is <= 0.
        return R >= 0 or _neg_branch(s, t, R)
    # Left side > 0; need R > 0 and 2*(s + t sqrt3)^2 <= R^2.
    if R <= 0:
        return False
    # 2*(s^2 + 3t^2) + 4*s*t*sqrt(3) <= R^2
    return le_rat_times_sqrt(2 * (s * s + 3 * t * t) - R * R, -4 * s * t, 3)


def _neg_branch(s, t, R):
    """Left side <= 0 and R < 0: need 2*(s + t sqrt3)^2 >= R^2."""
    # -sqrt(2)*|F| <= R < 0  <=>  2*F^2 >= R^2
    return not le_rat_times_sqrt(2 * (s * s + 3 * t * t) - R * R, -4 * s * t, 3)


def frac_vec(rows):
    """Turn a sequence of Fractions into (integer numerators, common denom)."""
    den = 1
    for x in rows:
        den = den * x.denominator // _gcd(den, x.denominator)
    return [int(x.numerator * (den // x.denominator)) for x in rows], den


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def as_fraction_list(v):
    return [x if isinstance(x, Fraction) else Fraction(x) for x in v]

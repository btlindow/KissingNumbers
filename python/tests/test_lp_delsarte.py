"""Acceptance tests for python/bounds (docs/design.md T2.1): exact Gegenbauer
polynomials and the Delsarte LP bound with restricted inner products."""

from __future__ import annotations

import os
import sys
from fractions import Fraction as F

import pytest
import sympy
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bounds.gegenbauer import (  # noqa: E402
    from_gegenbauer_basis,
    gegenbauer_coeffs,
    gegenbauer_eval,
    gegenbauer_table,
    gegenbauer_values,
    poly_eval,
    to_gegenbauer_basis,
)
from bounds.lp_delsarte import (  # noqa: E402
    A_ALL_LEECH,
    A_NONPOS,
    A_RESTRICTED,
    Certificate,
    delsarte_bound,
    exact_vertex,
    grid_A,
    main,
    rationalise,
    solve_lp,
    verify_certificate,
)

N = 24
ALPHA = F(N - 2, 2)  # 11


def _sympy_normalised(k: int, n: int) -> list[F]:
    t = sympy.symbols("t")
    alpha = sympy.Rational(n - 2, 2)
    p = sympy.Poly(sympy.expand(sympy.gegenbauer(k, alpha, t) / sympy.gegenbauer(k, alpha, 1)), t)
    coeffs = p.all_coeffs()[::-1]  # ascending powers
    return [F(int(c.p), int(c.q)) for c in coeffs] + [F(0)] * (k + 1 - len(coeffs))


def _poly_mul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _poly_from_roots(roots):
    p = [F(1)]
    for r in roots:
        p = _poly_mul(p, [-F(r), F(1)])
    return p


# ----------------------------------------------------------------- Gegenbauer


@pytest.mark.parametrize("k", range(0, 13))
def test_recurrence_matches_sympy_n24(k):
    assert gegenbauer_coeffs(k, N) == _sympy_normalised(k, N)


@pytest.mark.parametrize("n", [3, 4, 8])
def test_recurrence_matches_sympy_other_dimensions(n):
    for k in range(0, 9):
        assert gegenbauer_coeffs(k, n) == _sympy_normalised(k, n)


def test_n3_is_legendre():
    t = sympy.symbols("t")
    for k in range(0, 9):
        p = sympy.Poly(sympy.legendre(k, t), t).all_coeffs()[::-1]
        assert gegenbauer_coeffs(k, 3) == [F(int(c.p), int(c.q)) for c in p]


def test_table_and_coeffs_agree():
    tab = gegenbauer_table(15, N)
    assert len(tab) == 16
    for k in range(16):
        assert tab[k] == gegenbauer_coeffs(k, N)
        assert len(tab[k]) == k + 1 and tab[k][k] != 0


def test_point_evaluation_is_exact_and_consistent():
    pts = [F(-1), F(-1, 2), F(-1, 4), F(0), F(1, 4), F(1, 2), F(1), F(3, 7)]
    for k in range(0, 41):
        coeffs = gegenbauer_coeffs(k, N)
        for t in pts:
            v = gegenbauer_eval(k, t, N)
            assert isinstance(v, F)
            assert v == poly_eval(coeffs, t)
    for t in pts:
        assert gegenbauer_values(40, t, N) == [gegenbauer_eval(k, t, N) for k in range(41)]


def test_endpoints():
    for k in range(0, 61):
        assert gegenbauer_eval(k, 1, N) == 1
        assert gegenbauer_eval(k, -1, N) == (-1) ** k


@pytest.mark.parametrize("j,k", [(0, 1), (1, 2), (2, 5), (3, 7), (4, 10), (0, 12), (6, 9)])
def test_orthogonality_numeric(j, k):
    # weight (1 - t^2)^{(n-3)/2} = (1 - t^2)^{10.5} for n = 24
    cj = [float(c) for c in gegenbauer_coeffs(j, N)]
    ck = [float(c) for c in gegenbauer_coeffs(k, N)]
    w = lambda t: (1.0 - t * t) ** 10.5  # noqa: E731
    ev = lambda cs, t: sum(c * t**i for i, c in enumerate(cs))  # noqa: E731
    # The normalised G_k have tiny L2 norms (G_k(1) = 1 but |G_k| << 1 inside), so use the
    # relative criterion <G_j, G_k> / (|G_j| |G_k|): norms first (relative tolerance), then the
    # off-diagonal integral, whose true value is 0, with an absolute tolerance scaled to them.
    nj, _ = quad(lambda t: ev(cj, t) ** 2 * w(t), -1.0, 1.0, epsabs=0, epsrel=1e-10, limit=200)
    nk, _ = quad(lambda t: ev(ck, t) ** 2 * w(t), -1.0, 1.0, epsabs=0, epsrel=1e-10, limit=200)
    assert nj > 0 and nk > 0
    scale = (nj * nk) ** 0.5
    off, _ = quad(lambda t: ev(cj, t) * ev(ck, t) * w(t), -1.0, 1.0, epsabs=1e-10 * scale, epsrel=0, limit=200)
    assert abs(off) / scale < 1e-8


def test_basis_change_roundtrip():
    poly = [F(3, 7), F(-2), F(0), F(5, 3), F(1, 11), F(-4, 9), F(2)]
    f = to_gegenbauer_basis(poly, N)
    assert from_gegenbauer_basis(f, N) == poly
    # sanity: sum f_k G_k(t) == poly(t) at a random rational point
    t = F(2, 9)
    assert sum(fk * gk for fk, gk in zip(f, gegenbauer_values(len(f) - 1, t, N))) == poly_eval(poly, t)


# ----------------------------------------------------------------- known exact certificates


def test_levenshtein_polynomial_gives_196560_exactly():
    # Odlyzko-Sloane / Levenshtein: f(t) = (t+1)(t+1/2)^2 (t+1/4)^2 t^2 (t-1/4)^2 (t-1/2)
    roots = [-1, F(-1, 2), F(-1, 2), F(-1, 4), F(-1, 4), 0, 0, F(1, 4), F(1, 4), F(1, 2)]
    poly = _poly_from_roots(roots)
    f = to_gegenbauer_basis(poly, N)
    assert len(f) == 11
    assert f[0] > 0 and all(fk >= 0 for fk in f[1:])
    # f <= 0 on [-1, 1/2] since it has double roots inside and simple roots at the ends;
    # verify_certificate checks the discrete A_ALL_LEECH, which is all we need for the theorem.
    assert verify_certificate(f, A_ALL_LEECH, N) == 196560


def test_degree4_polynomial_gives_restricted_bound_exactly():
    # f(t) = (t+1)(t+1/4) t (t-1/4): zeros exactly on A \ {-1/2}, negative at -1/2.
    poly = _poly_from_roots([-1, F(-1, 4), 0, F(1, 4)])
    f = to_gegenbauer_basis(poly, N)
    assert all(fk >= 0 for fk in f[1:]) and f[0] > 0
    assert verify_certificate(f, A_RESTRICTED, N) == F(9360, 11)
    # normalised to f_0 = 1 this is the certificate the LP finds
    g = [fk / f[0] for fk in f]
    assert g == [F(1), F(24), F(5083, 77), F(4416, 11), F(27600, 77)]


# ----------------------------------------------------------------- LP + rationalisation


def test_antipodal_bound_is_two():
    for d in (1, 5, 10):
        lp = solve_lp([F(-1)], d, N)
        assert lp.status == 0
        assert abs(lp.bound - 2.0) < 1e-9
        cert = rationalise(lp)
        assert cert.bound == 2
        assert verify_certificate(cert.f, [F(-1)], N) == 2


@pytest.mark.parametrize("d", [10, 20, 30, 40, 60])
def test_restricted_bound_per_degree(d):
    lp = solve_lp(A_RESTRICTED, d, N)
    assert lp.status == 0, lp.message
    assert abs(lp.bound - 9360 / 11) < 1e-6
    cert = rationalise(lp)
    # independent exact re-verification of the rounded certificate
    b = verify_certificate(cert.f, A_RESTRICTED, N)
    assert b == cert.bound
    assert b >= F(9360, 11) - F(1, 10**6)  # cannot beat the true LP value by more than rounding
    assert b.__floor__() == 850
    vtx = exact_vertex(lp)
    assert vtx is not None and vtx.bound == F(9360, 11)
    assert verify_certificate(vtx.f, A_RESTRICTED, N) == F(9360, 11)


def test_restricted_bound_headline_and_496():
    rows = delsarte_bound(A_RESTRICTED, (10, 20, 40), N)
    best = min((r.best for r in rows), key=lambda c: c.bound)
    assert best.int_bound == 850
    assert best.int_bound > 496  # this LP alone does not close the gap; T2.2 must


def test_all_leech_inner_products_reproduce_196560():
    for d in (10, 20, 40):
        lp = solve_lp(A_ALL_LEECH, d, N)
        assert lp.status == 0
        assert abs(lp.bound - 196560) < 1e-3
        vtx = exact_vertex(lp)
        assert vtx is not None and vtx.bound == 196560
        cert = rationalise(lp)
        assert 196560 - F(1, 100) <= cert.bound <= 196560 + F(1, 100)
        assert cert.int_bound in (196559, 196560)


def test_nonpositive_variant_is_orthoplex_bound_2n():
    for d in (10, 30):
        lp = solve_lp(A_NONPOS, d, N)
        assert abs(lp.bound - 2 * N) < 1e-8
        vtx = exact_vertex(lp)
        assert vtx is not None and vtx.bound == 2 * N
        assert verify_certificate(rationalise(lp).f, A_NONPOS, N).__floor__() == 2 * N


def test_grid_reproduces_classical_kissing_lp_bound():
    # A_grid = {-1, -1+1/2000, ..., 1/2}.  Since A_ALL_LEECH is a subset of A_grid and
    # A_grid is a subset of [-1, 1/2], the LP value is sandwiched:
    #   196560 = bound(A_ALL_LEECH) <= bound(A_grid) <= bound([-1,1/2]) = 196560,
    # so the grid LP must return 196560 up to solver tolerance (for d >= 10).
    G = grid_A(2000)
    assert len(G) == 3001 and set(A_ALL_LEECH) <= set(G)
    for d in (20, 30):
        lp = solve_lp(G, d, N)
        assert lp.status == 0, lp.message
        assert abs(lp.bound - 196560) / 196560 < 1e-6


def test_verify_certificate_rejects_bad_input():
    with pytest.raises(ValueError):
        verify_certificate([1, -1], A_RESTRICTED, N)
    with pytest.raises(ValueError):
        verify_certificate([0, 1], A_RESTRICTED, N)
    with pytest.raises(ValueError):
        verify_certificate([1, 1], A_RESTRICTED, N)  # 1 + t > 0 at t = 1/4
    assert verify_certificate([1, 1], [F(-1)], N) == 2


def test_certificate_helpers():
    c = Certificate(A_RESTRICTED, N, [F(1), F(24), F(5083, 77), F(4416, 11), F(27600, 77)])
    assert c.d == 4 and c.support() == [0, 1, 2, 3, 4]
    assert c.bound == F(9360, 11) and c.int_bound == 850
    assert max(c.values_on_A()) == 0


def test_cli_result_line(capsys):
    assert main(["--degrees", "10", "20"]) == 0
    out = capsys.readouterr().out
    assert "RESULT bound_restricted=850 bound_all_leech=196560 bound_nonpos=48" in out

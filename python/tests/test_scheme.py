"""T2.2 — association scheme of the Leech minimal vectors, exact eigenmatrices,
scheme LP bound with exact certificate, theta / theta' / Hoffman cross-checks.

All numeric expectations below were established by tools/scheme_numbers (GPU
brute force, all 196560 y for x = C[0]) and are re-derived here from the
committed data/scheme/intersection_numbers.json in exact arithmetic.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from fractions import Fraction as F
from math import comb

import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bounds.gegenbauer import gegenbauer_eval  # noqa: E402
from bounds.lp_scheme import (  # noqa: E402
    VARIANTS,
    ExactLP,
    LPError,
    make_lp,
    rationalise,
    run_variants,
    solve_scheme_lp,
    solve_square,
)
from bounds.lp_scheme import main as lp_main  # noqa: E402
from bounds.scheme import (  # noqa: E402
    ANTIPODE,
    CLASS_DOTS,
    CONFLICT,
    DEFAULT_JSON,
    IDENTITY,
    SchemeError,
    check_bose_mesner,
    eigenmatrices,
    hoffman_bound,
    intersection_matrices,
    load_scheme,
    to_fractions,
)
from bounds.scheme import main as scheme_main  # noqa: E402
from bounds.theta_prime import hoffman_certificate, theta_prime, theta_sym  # noqa: E402
from bounds.theta_prime import main as theta_main  # noqa: E402

N = 196560
VALENCIES = (1, 4600, 47104, 93150, 47104, 4600, 1)
MULTIPLICITIES = (1, 24, 299, 2576, 17250, 95680, 80730)
CONFLICT_EIGENVALUES = (4600, 2300, 1000, 350, 76, -10, -20)
BOUND = F(9360, 11)
P_EXPECTED = [
    [1, 4600, 47104, 93150, 47104, 4600, 1],
    [-1, -2300, -11776, 0, 11776, 2300, 1],
    [1, 1000, 1024, -4050, 1024, 1000, 1],
    [-1, -350, 704, 0, -704, 350, 1],
    [1, 76, -320, 486, -320, 76, 1],
    [-1, 10, -16, 0, 16, -10, 1],
    [1, -20, 64, -90, 64, -20, 1],
]

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_JSON), reason="data/scheme/intersection_numbers.json missing")


@pytest.fixture(scope="module")
def scheme():
    return load_scheme(DEFAULT_JSON)


@pytest.fixture(scope="module")
def eig(scheme):
    return eigenmatrices(scheme)


# --------------------------------------------------------------------------- the json / scheme property


def test_json_records_full_verification(scheme):
    d = scheme.source
    assert d["N"] == N and d["x"] == 0
    assert d["scheme"] is True and d["identities_ok"] is True
    assert d["y_checked"] == N
    assert d["distinct_matrices_per_class"] == [1] * 7
    assert d["classes"] == list(CLASS_DOTS)
    assert tuple(d["valencies"]) == VALENCIES
    assert d["identity_class"] == IDENTITY and d["conflict_class"] == CONFLICT
    # vertex-transitivity computed, not cited
    assert d["orbit_checked"] is True and d["orbit_size"] == N and d["xi_orthogonal"] is True
    assert d["vertex_transitive_verified"] is True
    assert d["extra_x_agree"] == len(d["extra_x"])


def test_valencies_and_sum(scheme):
    assert scheme.valencies == VALENCIES
    assert sum(scheme.valencies) == N


def test_intersection_numbers_spot_values(scheme):
    p = scheme.p
    # conflict graph: lambda (common neighbours of an edge) and the mu's
    assert p[CONFLICT][CONFLICT][CONFLICT] == 891
    assert p[3][CONFLICT][CONFLICT] == 44        # orthogonal pair
    assert p[2][CONFLICT][CONFLICT] == 0 and p[4][CONFLICT][CONFLICT] == 275   # <x,y> = -8 / +8
    assert p[1][CONFLICT][CONFLICT] == 1         # <x,y> = -16
    assert p[ANTIPODE][CONFLICT][CONFLICT] == 0  # antipodes share no neighbour
    # antipode class permutes: p^0_{ij} = [j == 6 - i] v_i
    for i in range(7):
        for j in range(7):
            assert p[ANTIPODE][i][j] == (VALENCIES[i] if j == 6 - i else 0)


def test_identities_rejected_when_broken(scheme):
    import copy
    bad = copy.deepcopy(scheme)
    p = [[list(r) for r in m] for m in bad.p]
    p[3][1][2] += 1
    bad.p = tuple(tuple(tuple(r) for r in m) for m in p)
    with pytest.raises(SchemeError):
        from bounds.scheme import check_identities
        check_identities(bad)


def test_bose_mesner_algebra(scheme):
    L = intersection_matrices(scheme)
    check_bose_mesner(scheme, L)
    for i in range(7):
        for j in range(7):
            assert L[i] * L[j] == L[j] * L[i]
    assert L[IDENTITY] == sp.eye(7)


# --------------------------------------------------------------------------- eigenmatrices


def test_eigenmatrix_P(eig):
    assert eig.P == sp.Matrix(P_EXPECTED)
    assert eig.m == MULTIPLICITIES
    assert sum(eig.m) == N


def test_PQ_equals_NI_and_Q_formula(eig, scheme):
    assert eig.P * eig.Q == N * sp.eye(7)
    assert eig.Q * eig.P == N * sp.eye(7)
    for i in range(7):
        for r in range(7):
            assert eig.Q[i, r] == sp.Rational(eig.m[r] * eig.P[r, i], scheme.valencies[i])
    assert all(eig.Q[i, 0] == 1 for i in range(7))
    assert [eig.Q[IDENTITY, r] for r in range(7)] == list(MULTIPLICITIES)
    assert [eig.P[r, IDENTITY] for r in range(7)] == [1] * 7


def test_orthogonality_relations(eig, scheme):
    v, m = scheme.valencies, eig.m
    for r in range(7):
        for s in range(7):
            assert sum(eig.P[r, i] * eig.P[s, i] / v[i] for i in range(7)) == (sp.Rational(N, m[r]) if r == s else 0)


def test_conflict_spectrum(eig):
    spec = eig.conflict_spectrum(CONFLICT)
    assert tuple(int(l) for l, _ in spec) == CONFLICT_EIGENVALUES
    assert tuple(m for _, m in spec) == MULTIPLICITIES
    assert sum(l * m for l, m in spec) == 0            # trace of A_5
    assert sum(l * l * m for l, m in spec) == N * 4600  # trace of A_5^2


def test_multiplicities_are_harmonic_dimensions_and_Q_is_gegenbauer(eig):
    """The minimal vectors form a spherical 11-design, so E_1..E_5 are the
    harmonic spaces of degree 1..5 and Q_{ir} = m_r G_r(cos_i) for r <= 5."""
    def harm_dim(k, n=24):
        return comb(k + n - 1, n - 1) - (comb(k + n - 3, n - 1) if k >= 2 else 0)
    for r in range(1, 6):
        assert eig.m[r] == harm_dim(r)
    Q = to_fractions(eig.Q)
    for i, dot in enumerate(CLASS_DOTS):
        for r in range(6):
            assert Q[i][r] == eig.m[r] * gegenbauer_eval(r, F(dot, 32), 24)
    assert eig.m[6] == N - sum(eig.m[:6])


def test_hoffman(eig):
    assert hoffman_bound(eig, CONFLICT) == BOUND
    assert BOUND == F(N * 20, 4600 + 20)


# --------------------------------------------------------------------------- exact LP machinery


def test_solve_square():
    A = [[F(2), F(1)], [F(1), F(3)]]
    assert solve_square(A, [F(3), F(4)]) == [F(1), F(1)]
    assert solve_square([[F(1), F(2)], [F(2), F(4)]], [F(1), F(2)]) is None


def test_exact_lp_toy():
    # max x + y s.t. x + 2y <= 4, 3x + y <= 6, x,y >= 0  -> (8/5, 6/5), value 14/5
    lp = ExactLP(c=[F(1), F(1)],
                 A_ub=[[F(1), F(2)], [F(3), F(1)], [F(-1), F(0)], [F(0), F(-1)]],
                 b_ub=[F(4), F(6), F(0), F(0)])
    val, x, verts = lp.solve()
    assert val == F(14, 5) and x == [F(8, 5), F(6, 5)]
    assert len(verts) == 4


def test_rationalise():
    assert rationalise([0.5, 1e-12, 1 / 3]) == [F(1, 2), F(0), F(1, 3)]


# --------------------------------------------------------------------------- the bound


@pytest.fixture(scope="module")
def results(eig):
    return run_variants(eig)


def test_main_bound_exact_and_float(results):
    r = results["main"]
    assert r.exact_value == BOUND
    assert r.floor == 850
    assert abs(r.highs_value - float(BOUND)) < 1e-6
    assert r.dual_value == BOUND
    assert r.n_optimal_vertices == 1


def test_main_optimal_distribution(results):
    r = results["main"]
    assert r.a_star == [F(1), F(0), F(2944, 11), F(3450, 11), F(2944, 11), F(0), F(1)]
    assert sum(r.a_star) == BOUND
    aq = r.lp.aQ(r.a_star)
    assert aq[0] == BOUND and aq[1:6] == [F(0)] * 5 and aq[6] > 0


def test_main_certificate_is_hoffmans(results, eig):
    r = results["main"]
    expected = [F(0), F(116, 231), F(17, 77), F(37, 462), F(8, 385), F(1, 462), F(0)]
    assert r.beta == expected
    assert r.beta == hoffman_certificate(eig, CONFLICT)
    lam = CONFLICT_EIGENVALUES
    for s in range(1, 7):
        assert r.beta[s] == F(lam[s] + 20, 4600 + 20)
    assert r.lp.verify_certificate(r.beta) == BOUND
    assert r.beta_highs == expected
    # F(i) = -1 on every free class: the certificate is tight everywhere
    Q = r.lp.Q
    for i in r.lp.free:
        assert sum(r.beta[s] * Q[i][s] for s in range(7)) == -1


def test_certificate_by_hand(results):
    """The whole certificate re-checked from the numbers a reader would copy."""
    r = results["main"]
    Q = r.lp.Q
    beta = [F(0), F(116, 231), F(17, 77), F(37, 462), F(8, 385), F(1, 462), F(0)]
    assert all(b >= 0 for b in beta)
    for i in (0, 1, 2, 3, 4):           # allowed non-identity classes
        assert sum(beta[s] * Q[i][s] for s in range(7)) <= -1
    assert 1 + sum(beta[s] * MULTIPLICITIES[s] for s in range(7)) == BOUND


def test_T21_delsarte_polynomial_is_also_a_certificate(results):
    """T2.1's degree-4 polynomial f = 1 + 24 G_1 + (5083/77) G_2 + (4416/11) G_3
    + (27600/77) G_4 gives beta_r = f_r / m_r, another optimal dual."""
    r = results["main"]
    beta = [F(0), F(24, 24), F(5083, 77 * 299), F(4416, 11 * 2576), F(27600, 77 * 17250), F(0), F(0)]
    assert beta == [F(0), F(1), F(17, 77), F(12, 77), F(8, 385), F(0), F(0)]
    assert r.lp.verify_certificate(beta) == BOUND


def test_bad_certificates_rejected(results):
    r = results["main"]
    with pytest.raises(LPError):
        r.lp.verify_certificate([F(0)] * 7)             # proves nothing (F = 0 > -1)
    with pytest.raises(LPError):
        r.lp.verify_certificate([F(-1)] + [F(1)] * 6)   # negative multiplier
    with pytest.raises(LPError):
        r.lp.check_primal([F(1)] * 7)                   # a_16 must be 0


def test_variants(results):
    assert results["antipodal"].exact_value == BOUND
    assert results["antipodal"].a_star[ANTIPODE] == 1
    assert results["forbid_16_8"].exact_value == 48          # = T2.1's Rankin/orthoplex value
    assert results["forbid_16_8"].a_star == [F(1), F(0), F(0), F(46), F(0), F(0), F(1)]
    assert results["none"].exact_value == N
    assert results["none"].a_star == [F(v) for v in VALENCIES]
    assert results["only_pm32"].exact_value == 2
    for r in results.values():
        assert r.lp.verify_certificate(r.beta) == r.exact_value
        assert abs(r.highs_value - float(r.exact_value)) <= 1e-6 * max(1.0, float(r.exact_value))


def test_antipodal_constraint_is_free(eig):
    """Forcing a_-32 = 1 does not change the optimum (a* already has it)."""
    r = solve_scheme_lp(make_lp(eig, (CONFLICT,), {ANTIPODE: 1}, name="anti"))
    assert r.exact_value == BOUND


def test_theta_theta_prime_hoffman_coincide(eig):
    tp, ts = theta_prime(eig), theta_sym(eig)
    assert tp.exact_value == BOUND
    assert ts.exact_value == BOUND
    assert not ts.lp.nonneg and tp.lp.nonneg
    assert ts.lp.verify_certificate(ts.beta) == BOUND
    assert hoffman_bound(eig, CONFLICT) == BOUND


def test_bound_exceeds_496():
    assert BOUND > 496   # gate G1: the LP does not close dimension 25


# --------------------------------------------------------------------------- CLIs


def _run(main):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([])
    return rc, buf.getvalue()


def test_cli_scheme():
    rc, out = _run(scheme_main)
    assert rc == 0
    assert "RESULT ok=1 scheme=1 N=196560 PQ_eq_NI=1 multiplicities=1,24,299,2576,17250,95680,80730 " \
           "conflict_eigenvalues=4600,2300,1000,350,76,-10,-20 hoffman=9360/11" in out


def test_cli_lp_scheme():
    rc, out = _run(lp_main)
    assert rc == 0
    last = [l for l in out.splitlines() if l.startswith("RESULT")][-1]
    assert last == ("RESULT bound_exact=9360/11 bound_floor=850 highs=850.909091 antipodal=9360/11 "
                    "antipodal_floor=850 forbid_16_8=48 none=196560 only_pm32=2")


def test_cli_theta_prime():
    rc, out = _run(theta_main)
    assert rc == 0
    assert "RESULT hoffman=9360/11 theta_sym=9360/11 theta_prime=9360/11 lp=9360/11 hoffman_cert_ok=1 all_equal=1" in out

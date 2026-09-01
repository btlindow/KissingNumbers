"""T2.3 — acceptance tests for the three-point (Schrijver/Terwilliger) SDP bound.

The tests are ordered from "structure of the algebra" through "known feasible
points" (the decisive test of the normalisation: the record 496 must satisfy
every constraint) to "the bound and its exact certificate".
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bounds.orbitals import (  # noqa: E402
    CONFLICT,
    check_against_scheme,
    check_against_triples,
    load_orbitals,
    triple_orbits,
)
from bounds.sdp3_scheme import (  # noqa: E402
    Sdp3,
    class_cells,
    psd_exact,
    terwilliger_dimension,
    verify_certificate,
)

cvxpy = pytest.importorskip("cvxpy")


@pytest.fixture(scope="module")
def O():
    return load_orbitals()          # runs orbitals.check_orbitals (exact)


@pytest.fixture(scope="module")
def S(O):
    return Sdp3(O)


# ------------------------------------------------------------------ structure


def test_orbitals_are_the_orbitals(O):
    """148 labels == 148 distinct GPU 4-point histograms => the labels are
    exactly the Stab(x)-orbitals; and Co_0 is NOT transitive on ordered triples
    with prescribed pairwise classes."""
    r = check_against_triples(O)
    check_against_scheme(O)
    assert r["labels_are_orbitals"] is True
    assert r["D"] == r["variants"] == 148
    assert r["cells"] == 147
    assert r["split_cells"] == {(3, 3, 3): [42240, 924]}


def test_terwilliger_dimension(O):
    """T(x) = <E_i*, A_k> is the WHOLE 148-dimensional centraliser algebra of
    Stab(x); the 147 triple-class matrices E_i* A_k E_j* span a proper
    subspace of it (and do not form an algebra)."""
    assert class_cells(O) == 147
    assert terwilliger_dimension(O) == O.D == 148


def test_s3_orbits(O):
    tv, members = triple_orbits(O)
    assert len(members) == 43
    assert sum(len(m) for m in members) == O.D


def test_coefficient_matrices(S):
    """G_q (exact integers) and Z_q (floats) are the same operator in two
    bases: G = Delta^{1/2} Z Delta^{1/2}."""
    sq = S.sqsize
    for q in (0, S.e, 7, 20, S.Q - 1):
        G = np.array([[float(v) for v in row] for row in S.G[q]])
        Z = S.Z[q]
        ref = sq[:, None] * Z * sq[None, :]
        scale = max(np.abs(G).max(), 1.0)
        assert np.abs(G - ref).max() <= 1e-6 * scale
        assert np.array_equal(S.G[q], S.G[q].T)
        assert np.array_equal(S.Gy[q], S.Gy[q].T)


def test_psd_exact_sanity():
    assert psd_exact([[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]) == (True, 2)
    assert psd_exact([[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]]) == (True, 1)
    assert psd_exact([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]])[0] is False
    assert psd_exact([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]])[0] is False


# ------------------------------------------------ known feasible points (decisive)


def test_s496_is_feasible_exactly(S):
    """THE decisive test of the formulation and the normalisation: the record
    set's own triple distribution must satisfy every constraint of the
    relaxation, in exact arithmetic, with objective exactly 496."""
    n, x = S.x_of_set("S496.txt")
    assert n == 496
    r = S.check_point(x, [CONFLICT], exact_psd=True)
    assert r["objective"] == 496
    assert r["x_e"] == 1
    assert r["nonneg"] and r["at_most_one"] and r["forbidden_zero"] and r["y_nonneg"]
    assert r["psd_block1"] and r["psd_block2"]
    assert r["feasible"]


def test_s488_is_feasible(S):
    n, x = S.x_of_set("S488.txt")
    assert n == 488
    r = S.check_point(x, [CONFLICT], exact_psd=False)
    assert r["objective"] == 488
    assert r["feasible"]


def test_whole_scheme_is_feasible_exactly(S):
    """S = C with nothing forbidden: x == 1, objective == N, and the complement
    block is identically zero (the complement of S is empty)."""
    ones = np.array([Fraction(1)] * S.Q, dtype=object)
    r = S.check_point(ones, [], exact_psd=True)
    assert r["objective"] == S.N == 196560
    assert r["feasible"]
    assert r["rank_block1"] == 7        # M1 = J restricted to the 7 fibres
    assert r["rank_block2"] == 0        # M2 = 0


def test_infeasible_point_is_rejected(S):
    """Negative control: a point that is not the distribution of a set must be
    caught by the PSD test (here the 496's x with one entry inflated)."""
    n, x = S.x_of_set("S496.txt")
    bad = list(x)
    q = max((qq for qq in range(S.Q) if qq != S.e and bad[qq] > 0), key=lambda t: bad[t])
    bad[q] = bad[q] * 4 + 1
    r = S.check_point(np.array(bad, dtype=object), [CONFLICT], exact_psd=True)
    assert not (r["psd_block1"] and r["psd_block2"])


# ------------------------------------------------------------------- the bound


@pytest.fixture(scope="module")
def res_conflict(S):
    r = S.solve((CONFLICT,))
    return S.certify(r, bits=52)


def test_bound_forbidding_16(S, res_conflict):
    """The headline: forbidding inner product 16 (the 60-degree condition)."""
    r = res_conflict
    assert r.value is not None
    assert 496.0 <= r.value <= 851.0
    assert r.bound_floor is not None
    assert 496 <= r.bound_floor < 850, "the three-point bound must beat T2.2's 850"
    assert float(r.bound) >= r.value - 1e-6
    assert r.bound_floor == 837


def test_bound_nothing_forbidden(S):
    """Sanity: with no forbidden class the relaxation must not exclude S = C."""
    r = S.certify(S.solve(()), bits=52)
    assert r.bound_floor >= S.N


def test_bound_forbidding_16_and_8(S):
    """Sanity: forbidding inner products 16 and 8 leaves only cos <= 0, whose
    optimum is the orthoplex bound 2*24 = 48 (T2.1)."""
    r = S.certify(S.solve((CONFLICT, 4)), bits=52)
    assert r.bound_floor == 48


def test_certificate_recheck_independently(S, res_conflict):
    """Re-derive the bound from the stored certificate with a straightforward,
    independent code path, and check the Lagrangian inequality directly at the
    496's point: sum_q h_q x_q must be >= the objective 496."""
    cert = S.certificate(res_conflict)
    free = set(S.free_orbits(res_conflict.forbidden))
    h = cert["h"]
    assert len(h) == S.Q
    bound = h[S.e] + sum(cert["rho"][q] for q in free) \
        + sum(hq for q, hq in enumerate(h) if q in free and q != S.e and hq > 0)
    # `h` in the certificate already has rho subtracted (see Sdp3.certificate)
    assert bound == res_conflict.bound
    n, x = S.x_of_set("S496.txt")
    lag = sum(h[q] * x[q] for q in range(S.Q)) + sum(cert["rho"][q] * (1 - x[q]) for q in free)
    assert lag >= 496
    assert lag <= res_conflict.bound


def test_certificate_file_roundtrip(S, res_conflict, tmp_path):
    """Write the certificate and re-check it from scratch: the verifier
    rebuilds G_q and Gy_q from orbitals.json, recomputes every h_q in exact
    rational arithmetic and re-derives the bound, using nothing from the
    solver."""
    path = str(tmp_path / "cert.json")
    S.save_certificate(res_conflict, path)
    r = verify_certificate(path)
    assert r["matches_stored"] is True
    assert r["bound_floor"] == res_conflict.bound_floor == 837
    assert r["forbidden_dots"] == [16]


def test_shipped_certificate_verifies():
    """The committed certificate in data/scheme/ must still prove |S| <= 837."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "data", "scheme", "sdp3_certificate.json")
    if not os.path.exists(path):
        pytest.skip("data/scheme/sdp3_certificate.json not generated")
    r = verify_certificate(path)
    assert r["bound_floor"] == 837
    assert r["forbidden_dots"] == [16]

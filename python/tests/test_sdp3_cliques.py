"""B1 acceptance tests: clique number input, lifted clique cuts, exact
feasibility of the record sets, certificate with cuts (docs/reports/B1.md)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bounds.orbitals import CONFLICT, OrbitalError
from bounds.sdp3_cliques import (
    K24,
    CliqueCutter,
    Sdp3Cuts,
    class_of,
    load_vectors,
    verify_cuts_certificate,
    verify_k24,
)

RNG = np.random.default_rng(20260829)


@pytest.fixture(scope="module")
def C():
    return load_vectors()


@pytest.fixture(scope="module")
def S():
    return Sdp3Cuts()


@pytest.fixture(scope="module")
def cutter(S, C):
    return CliqueCutter(S, C)


# --------------------------------------------------------------- omega(G) = 24


def test_k24_is_a_24_clique(C):
    assert len(K24) == 24 and len(set(K24)) == 24
    verify_k24(C)  # exact integer arithmetic


def test_k24_gram_is_16_I_plus_J(C):
    V = C[list(K24)].astype(np.int64)
    G = V @ V.T
    assert np.array_equal(G, 16 * (np.eye(24, dtype=np.int64) + np.ones((24, 24), dtype=np.int64)))
    # rank argument for omega <= 24: 16(I+J_k) is positive definite for all k,
    # so k pairwise-16 vectors are linearly independent and k <= dim = 24.
    w = np.linalg.eigvalsh(G.astype(float))
    assert w.min() > 15.9  # eigenvalues {16, 16*25}


def test_verify_k24_rejects_a_non_clique(C):
    bad = list(K24[:23]) + [2]  # vertex 2 is not adjacent to all of them
    with pytest.raises(OrbitalError):
        verify_k24(C, bad)


def test_common_neighbourhood_of_an_edge(C):
    d0 = C @ C[0]
    d1 = C @ C[1]
    assert int(C[0] @ C[1]) == 16
    common = np.where((d0 == 16) & (d1 == 16))[0]
    assert len(common) == 891  # = p^5_55, triangles through an edge
    sub = C[common]
    A = (sub @ sub.T) == 16
    np.fill_diagonal(A, False)
    deg = A.sum(1)
    assert deg.min() == deg.max() == 336  # 336-regular


# --------------------------------------------------------------- the classifier


def test_classifier_selfcheck(cutter):
    cutter.selfcheck_classifier(RNG)  # exact 924 + 42240 split


def test_singleton_clique_on_own_base_gives_zero_row(cutter):
    # K = {b}: triple (a,b,b) is the S_3 image of the diagonal orbital, so
    # (C1) reads x_diag(i) <= x_diag(i) — the row must be exactly zero.
    a, b = 0, 196559  # antipodal pair (class 0)
    (cut,) = cutter.cuts(a, b, [b], kinds=("C1",))
    assert not np.any(cut.row) and cut.rhs == 0


def test_cut_rows_reference_forbidden_orbits_only_when_conflicting(S, cutter):
    # a triangle of K24 seen from a random pair: coefficients sum to |K| on the
    # triple side (minus the diagonal), spread over valid triple orbits
    a, b = 5000, int(np.where(class_of(cutter.C @ cutter.C[5000]) == 2)[0][0])
    (cut,) = cutter.cuts(a, b, K24[:3], kinds=("C1",))
    qdiag_i = cutter.qdiag[2]
    assert cut.row[qdiag_i] <= -1 + 3  # -1 from the rhs term (+ any conflicts)
    assert cut.row.sum() == 3 - 1  # total +|K|, -1 for x_diag(i)


def test_base_edge_is_rejected(cutter):
    with pytest.raises(OrbitalError):
        cutter.cuts(0, 1, K24[:3])  # (0,1) is an edge: class 5


# ------------------------------------------------- MANDATORY validity: the 496


def test_496_and_488_exactly_feasible_under_cuts(cutter):
    rng = np.random.default_rng(3)
    tri = [tuple(int(K24[t]) for t in rng.choice(24, 3, replace=False)) for _ in range(2)]
    pool = cutter.sample_pool(rng, tri + [K24], pairs_per_class=3, kinds=("C1", "C2"))
    assert len(pool) >= 20
    stats = cutter.check_sets_exact(pool)  # raises on any violation
    assert set(stats) == {"S488.txt", "S496.txt"}
    for r in stats.values():
        assert r["eq"] + r["lt"] == len(pool)


def test_corrupted_cut_is_caught_by_the_496(S, cutter):
    rng = np.random.default_rng(4)
    pool = cutter.sample_pool(rng, [K24], pairs_per_class=2, classes=(2,), kinds=("C1",))
    cut = pool[0]
    # inflate the row on the diagonal variable of class 2, which is strictly
    # positive on the 496 (a_{-8} = 37504/496 pairs), so the corrupted "cut"
    # is guaranteed to be violated: val >= 999 * x_diag(2) > 0.
    cut.row[cutter.qdiag[2]] += 1000
    with pytest.raises(OrbitalError, match="VIOLATED"):
        cutter.check_sets_exact([cut])


# ------------------------------------------------------- SDP + certificate


def test_solve_and_certify_with_cuts_forbid_16_8(S, cutter, tmp_path):
    # small, fast instance: forbid {16, 8} -> orthoplex bound 48; omega cuts
    rng = np.random.default_rng(5)
    pool = cutter.sample_pool(rng, [K24], pairs_per_class=2, classes=(0, 3), kinds=("C1",))
    cutter.check_sets_exact(pool)
    res = S.solve_with_cuts(pool, (CONFLICT, 4))
    S.certify_with_cuts(res)
    assert res.bound_floor == 48
    # certificate round-trip with independent re-derivation of the cut rows
    p = str(tmp_path / "cert.json")
    S.save_cuts_certificate(res, p)
    v = verify_cuts_certificate(p)
    assert v["matches_stored"] and v["bound_floor"] == 48 and v["cuts"] == len(pool)


def test_verify_rejects_a_tampered_certificate(S, cutter, tmp_path):
    import json

    rng = np.random.default_rng(6)
    pool = cutter.sample_pool(rng, [K24], pairs_per_class=1, classes=(3,), kinds=("C1",))
    res = S.solve_with_cuts(pool, (CONFLICT, 4))
    S.certify_with_cuts(res)
    p = str(tmp_path / "cert.json")
    S.save_cuts_certificate(res, p)
    with open(p) as f:
        doc = json.load(f)
    doc["cuts"][0]["row"][0] = int(doc["cuts"][0]["row"][0]) + 1
    with open(p, "w") as f:
        json.dump(doc, f)
    with pytest.raises(OrbitalError):
        verify_cuts_certificate(p)


def test_cuts_do_not_separate_the_baseline_optimum(S, cutter):
    # the headline finding of B1: the T2.3 optimum satisfies every clique cut
    res = S.solve_with_cuts([], (CONFLICT,))
    assert res.value == pytest.approx(837.5332, abs=2e-3)
    rng = np.random.default_rng(7)
    tri = [tuple(int(K24[t]) for t in rng.choice(24, 3, replace=False)) for _ in range(3)]
    pool = cutter.sample_pool(rng, tri + [K24], pairs_per_class=6, kinds=("C1", "C2"))
    A = np.stack([c.row for c in pool]).astype(float)
    t = np.array([c.rhs for c in pool], dtype=float)
    viol = A @ res.x - t
    assert viol.max() <= 1e-7  # no violated cut (equality up to solver accuracy)

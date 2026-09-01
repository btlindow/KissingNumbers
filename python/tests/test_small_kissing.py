"""Tests for python/kiss_ref/small_kissing.py and python/kiss_ref/exact.py (docs/design.md T4.3)."""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import exact as ex  # noqa: E402
from kiss_ref import small_kissing as sk  # noqa: E402
from kiss_ref.exact import Q  # noqa: E402

TOUCHING = {2: 6, 3: 24, 4: 96, 5: 240, 6: 720, 7: 2016}      # unordered pairs at exactly 60 degrees
TRIANGLES = {2: 2, 3: 8, 4: 32, 5: 80, 6: 240, 7: 672}       # triples with pairwise cos = -1/2


# ---------------------------------------------------------------------------
# the configurations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", range(2, 8))
def test_config_counts_and_kissing(d):
    V, norm2 = sk.config(d)
    assert V.shape == (sk.K[d], sk.AMBIENT[d])
    assert V.dtype == np.int64
    assert np.all((V * V).sum(axis=1) == norm2)
    assert len({tuple(r) for r in V.tolist()}) == sk.K[d]
    res = sk.check_kissing(V)
    assert res["ok"], res
    assert res["max_cos"] == 0.5                      # touching attained ...
    assert res["touching_pairs"] == TOUCHING[d]       # ... this often
    # antipodal, and every vector has exactly 2 (d=2), 4, 8, 12, 20, 32 neighbours at 60 degrees
    G = V @ V.T
    assert np.all(np.sort(G, axis=1)[:, 0] == -norm2)
    assert np.all((G == norm2 // 2).sum(axis=1) == 2 * TOUCHING[d] // sk.K[d])


def test_config_is_a_copy():
    V, _ = sk.config(3)
    V[0, 0] = 99
    assert sk.config(3)[0][0, 0] != 99


def test_config_bad_dimension():
    with pytest.raises(ValueError):
        sk.config(8)


@pytest.mark.parametrize("d", range(2, 8))
def test_embed_in_Rd(d):
    V, norm2 = sk.config(d)
    X = sk.embed_in_Rd(d)
    assert X.shape == (sk.K[d], d)
    assert np.abs(X @ X.T - V @ V.T).max() < 1e-9
    U = sk.embed_in_Rd(d, unit=True)
    assert np.abs((U * U).sum(axis=1) - 1).max() < 1e-12
    assert np.abs(U @ U.T - (V @ V.T) / norm2).max() < 1e-9


def test_embedding_rejects_too_small_d():
    V, _ = sk.config(4)
    with pytest.raises(ValueError):
        sk.orthonormal_embedding(V, 3)


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------
def test_predicates_exact_ties():
    # (1,-1,0) vs (2,-1,-1): cos = 3/sqrt(12) = sqrt3/2 exactly
    assert sk.cos_le_sqrt3_half(3, 2, 6)
    assert not sk.cos_le_sqrt3_half(4, 2, 6)
    assert sk.cos_le_half(1, 2, 2) and not sk.cos_le_half(2, 2, 2)
    assert sk.cos_le_neg_half(-1, 2, 2) and not sk.cos_le_neg_half(0, 2, 2)
    assert sk.is_antipodal(-2, 2, 2) and not sk.is_antipodal(-1, 2, 2)
    assert sk.is_same_direction(4, 2, 8) and not sk.is_same_direction(-4, 2, 8)
    assert sk.cmp_sqrt(3, 1, 8) == 1 and sk.cmp_sqrt(3, 1, 9) == 0 and sk.cmp_sqrt(3, 1, 10) == -1
    assert sk.cmp_sqrt(-3, -1, 8) == -1 and sk.cmp_sqrt(-3, 1, 2) == -1 and sk.cmp_sqrt(3, -1, 2) == 1


def test_predicates_vs_float():
    rng = np.random.default_rng(1)
    A = rng.integers(-5, 6, size=(300, 5))
    B = rng.integers(-5, 6, size=(300, 5))
    A = A[np.any(A != 0, axis=1)]
    B = B[np.any(B != 0, axis=1)]
    Na, Nb = (A * A).sum(1), (B * B).sum(1)
    P = A @ B.T
    cosv = P / np.sqrt(np.outer(Na, Nb))
    for thr, fn in ((0.5, sk.cos_le_half), (math.sqrt(3) / 2, sk.cos_le_sqrt3_half)):
        clear = np.abs(cosv - thr) > 1e-9
        assert np.array_equal(fn(P, Na[:, None], Nb[None, :])[clear], (cosv <= thr)[clear])
    clear = np.abs(cosv + 0.5) > 1e-9
    assert np.array_equal(sk.cos_le_neg_half(P, Na[:, None], Nb[None, :])[clear], (cosv <= -0.5)[clear])


def test_predicates_big_integers_do_not_overflow():
    # norms ~ 1e10 each: Na*Nb ~ 1e20 > int64
    a = np.array([10**5, 3, 0], dtype=np.int64)
    b = np.array([10**5, -4, 1], dtype=np.int64)
    Na, Nb, p = int(a @ a), int(b @ b), int(a @ b)
    want = p / math.sqrt(Na * Nb) <= 0.5
    assert bool(sk.cos_le_half(p, Na, Nb)) == want


# ---------------------------------------------------------------------------
# triangles, partitions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", range(2, 8))
def test_triangles(d):
    V, norm2 = sk.config(d)
    tris = sk.triangles(d)
    assert len(tris) == TRIANGLES[d]
    for a, b, c in tris:
        assert a < b < c
        assert V[a] @ V[b] == V[b] @ V[c] == V[a] @ V[c] == -norm2 // 2
        assert not np.any(V[a] + V[b] + V[c])  # coplanar equilateral: sum zero


@pytest.mark.parametrize("d,want,pairs", [(2, 2, 0), (3, 4, 0), (4, 8, 0), (5, 12, 2), (6, 24, 0), (7, 42, 0)])
def test_triangle_partition(d, want, pairs):
    res = sk.triangle_partition(d, want)
    assert res is not None and sk.triangle_partition.last_status == "done"
    assert len(res["triangles"]) == want and len(res["pairs"]) == pairs and not res["singles"]
    used = [i for t in res["triangles"] for i in t] + [i for p in res["pairs"] for i in p]
    assert sorted(used) == list(range(sk.K[d]))  # exact cover of the configuration
    V, norm2 = sk.config(d)
    for i, j in res["pairs"]:
        assert V[i] @ V[j] == -norm2
    assert res["weight"] == 2 * want + pairs


def test_triangle_partition_d5_13_is_infeasible():
    assert sk.triangle_partition(5, 13) is None
    assert sk.triangle_partition.last_status == "done"  # exhaustive, not a node limit


def test_max_triangle_partition():
    for d, best in ((2, 2), (3, 4), (5, 12), (7, 42)):
        res = sk.max_triangle_partition(d)
        assert res["want"] == best and res["exhaustive"]


# ---------------------------------------------------------------------------
# extra spheres
# ---------------------------------------------------------------------------
def test_extra_spheres_d2_and_d4_reach_K_with_integers():
    for d in (2, 4):
        r = sk.lattice_extra_spheres(d, restarts=3, iters=2000)
        assert r["reached"] and r["n"] == sk.K[d]
        chk = sk.check_extra(r["vectors"], sk.config(d)[0])
        assert chk["ok"], chk
        assert abs(chk["max_cos_to_T"] - {2: math.sqrt(3) / 2, 4: math.sqrt(0.5)}[d]) < 1e-12


def test_extra_spheres_d3_lattice_partial():
    r = sk.lattice_extra_spheres(3, max_entry=2, restarts=3, iters=2000)
    assert 8 <= r["n"] <= 12
    assert sk.check_extra(r["vectors"], sk.config(3)[0])["ok"]


def test_check_extra_rejects_a_T_vector():
    V, _ = sk.config(3)
    chk = sk.check_extra(V[:1], V)
    assert not chk["ok"] and "within 30 degrees" in chk["message"]
    chk = sk.check_extra(np.array([[1, 1, 0], [1, 1, 1]]), V[:0])
    assert not chk["ok"] and "cos > 1/2" in chk["message"]


@pytest.mark.parametrize("d", (3, 5))
def test_exact_rotation_extras(d):
    r = sk.rotated_extra_spheres_exact(d, tries=400, max_len=2, seed=0)
    assert r["reached"] and r["n"] == sk.K[d] and r["sqrt"] == 2
    V, N = sk.config(d)
    chk = sk.check_extra_ab(np.array(r["a"]), np.array(r["b"]), r["sqrt"], r["den"], V, N)
    assert chk["ok"], chk
    # cross-check in the general field arithmetic
    E = r["vectors_exact"]
    Tq = [[Q(int(x)) for x in row] for row in V]
    for e in E[:20]:
        assert ex.norm2(e) == Q(N)
        for t in Tq:
            assert ex.cos_le_sqrt3_half(ex.dot(e, t), Q(N), Q(N))


def test_sign_ab():
    A = np.array([3, -3, 0, 2, -2, 3, 0])
    B = np.array([-2, 2, 0, 1, 1, 2, -1])
    want = [int(np.sign(a + b * math.sqrt(2))) for a, b in zip(A, B)]
    assert sk._sign_ab(A, B, 2).tolist() == want


# ---------------------------------------------------------------------------
# schema blocks
# ---------------------------------------------------------------------------
def test_schema_blocks():
    Tb = sk.schema_T_block(2)
    assert Tb["K"] == 6 and Tb["norm2"] == 2 and Tb["ambient"] == 3 and len(Tb["groups"]) == 2
    E = sk.lattice_extra_spheres(2)["vectors"]
    Eb = sk.schema_extra_block(E, 2)
    assert Eb["count"] == 6 and Eb["sqrt"] == 3 and Eb["den"] == 3
    for a, b in zip(Eb["a"], Eb["b"]):
        y = [Q(x) + Q(z) * Q.sqrt(Eb["sqrt"]) for x, z in zip(a, b)]
        y = [v * Q(Fraction(1, Eb["den"])) for v in y]
        assert ex.norm2(y) == Q(2)
    with pytest.raises(ValueError):
        sk.schema_extra_block(np.array([[1, 0, 0], [1, 1, 0]]), 2)


# ---------------------------------------------------------------------------
# isometry finder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", (2, 3, 4))
def test_isometry_to_model(d):
    rng = np.random.default_rng(d)
    X = sk.embed_in_Rd(d)
    Qm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    perm = rng.permutation(len(X))
    Y = (X @ Qm.T)[perm] * 0.37
    p = sk.isometry_to_model(Y, d)
    assert p is not None
    V, norm2 = sk.config(d)
    G = (V[p] @ V[p].T)
    assert np.abs(G - Y @ Y.T / (Y[0] @ Y[0]) * norm2).max() < 1e-9


# ---------------------------------------------------------------------------
# exact field arithmetic
# ---------------------------------------------------------------------------
def test_exact_basic_arithmetic():
    r2, r3 = Q.sqrt(2), Q.sqrt(3)
    assert Q.sqrt(8) == r2 * 2 and Q.sqrt(12) == r3 * 2
    assert r2 * r3 == Q.sqrt(6) and (r2 * r3) * (r2 * r3) == Q(6)
    x = (Q(2) + r2) * Q(Fraction(1, 4))
    assert abs(float(x) - (2 + math.sqrt(2)) / 4) < 1e-15
    assert x <= r3 * Q(Fraction(1, 2))  # 0.8536 <= 0.8660
    assert (r3 - r2).sign() == 1 and (r2 * 3 - r3 * 2).sign() == 1 and (r2 + r3 - Q.sqrt(6) * 2).sign() == -1
    y = Q(1) + r2 + r3 + Q.sqrt(6)
    assert y * y.inv() == Q(1) and y / y == Q(1)
    assert (Q(1) - r2).inv() == -(Q(1) + r2)
    with pytest.raises(ZeroDivisionError):
        Q().inv()


def test_exact_parse():
    assert ex.parse_entry("-1/2") == Q(Fraction(-1, 2))
    assert ex.parse_entry("3/4*sqrt(2)") == Q.sqrt(2) * Q(Fraction(3, 4))
    assert ex.parse_entry("sqrt2/2") == Q.sqrt(2) * Q(Fraction(1, 2))
    assert ex.parse_entry("(2+sqrt(2))/4") == (Q(2) + Q.sqrt(2)) * Q(Fraction(1, 4))
    assert ex.parse_entry("1/2 - 1/2*sqrt(3)") == Q(Fraction(1, 2)) - Q.sqrt(3) * Q(Fraction(1, 2))
    assert ex.parse_entry([1, "1/2"], 2) == Q(1) + Q.sqrt(2) * Q(Fraction(1, 2))
    assert ex.parse_entry({"1": 1, "3": "1/2"}) == Q(1) + Q.sqrt(3) * Q(Fraction(1, 2))
    assert ex.parse_entry(0.5) == Q(Fraction(1, 2)) and ex.parse_entry(-2) == Q(-2)
    with pytest.raises(ValueError):
        ex.parse_entry(0.7071067811865476)
    with pytest.raises(ValueError):
        ex.parse_entry("foo")
    assert ex.parse_vector([1, "sqrt(2)"], scale=Q.sqrt(2)) == [Q.sqrt(2) * Q(Fraction(1, 2)), Q(1)]


def test_exact_rank_and_predicates():
    r2 = Q.sqrt(2)
    assert ex.rank([[Q(1), Q(0), r2], [Q(0), Q(1), Q(0)], [Q(1), Q(1), r2]]) == 2
    assert ex.rank([[r2, Q(2)], [Q(1), r2]]) == 1
    assert ex.rank([]) == 0
    assert ex.cos_le_sqrt3_half(Q(3), Q(2), Q(6)) and not ex.cos_le_sqrt3_half(Q(4), Q(2), Q(6))
    assert ex.cos_le_half(Q(1), Q(2), Q(2)) and not ex.cos_le_half(Q(2), Q(2), Q(2))
    assert ex.cos_le_neg_half(Q(-1), Q(2), Q(2)) and ex.is_antipodal(Q(-2), Q(2), Q(2))
    assert ex.is_same_direction(Q(4), Q(2), Q(8))
    # PackingStar's d=3 value (2+sqrt2)/4 against unit vectors: within 30 degrees? no (31.4 degrees)
    p = (Q(2) + r2) * Q(Fraction(1, 4))
    assert ex.cos_le_sqrt3_half(p, Q(1), Q(1))
    assert not ex.cos_le_sqrt3_half(p + Q(Fraction(1, 50)), Q(1), Q(1))


def test_exact_rank_matches_numpy_on_configs():
    for d in range(2, 8):
        V, _ = sk.config(d)
        rows = [[Q(int(x)) for x in r] for r in V[:40]]
        assert ex.rank(rows) == np.linalg.matrix_rank(V[:40]) == min(d, 40)

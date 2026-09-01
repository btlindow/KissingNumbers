"""Tests for python/template/highdim.py (PLAN T5.2): the template above dimension 31."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import small_kissing as sk  # noqa: E402
from template import highdim as hd, triangles as tr  # noqa: E402


# ---------------------------------------------------------------------------
# the arithmetic reproduces everything already verified for n = 25..31
# ---------------------------------------------------------------------------
# T5.1 / T4.2: max weight per d, and the resulting count with 496-sets.
# d = 3 is our improved value 200540 (T5.1), not PackingStar's 200044.
KNOWN = {2: (4, 198550), 3: (8, 200540), 4: (16, 204520),
         5: (26, 209496), 6: (48, 220440), 7: (84, 238350)}


@pytest.mark.parametrize("d", sorted(KNOWN))
def test_reproduces_verified_counts(d):
    weight, count = KNOWN[d]
    K = sk.K[d]
    assert hd.weight_upper_bound(K) == weight
    # 42 disjoint 496-sets are enough for every d <= 7 (at most 42 groups exist)
    res = hd.template_value(d, K=K, n_sets=hd.CONSTRUCTED_SETS)
    assert res["weight"] == weight
    assert res["value"] == count


def test_weight_upper_bound_edges():
    assert hd.weight_upper_bound(0) == 0
    assert hd.weight_upper_bound(2) == 1          # one antipodal pair
    assert hd.weight_upper_bound(3) == 2          # one triangle
    assert hd.weight_upper_bound(4) == 2
    assert hd.weight_upper_bound(5) == 3          # triangle + pair
    assert hd.weight_upper_bound(240) == 160      # E8: 2K/3, attained (Eisenstein)


def test_d1_extras_are_not_K1():
    """d = 1 is the one dimension where |E| = K(d) fails: R^1 has no direction 30
    degrees away from +-1, so the 25-dimensional record has no extra spheres."""
    assert hd.template_value(1, K=2, n_sets=1, extras=0)["value"] == 197056


# ---------------------------------------------------------------------------
# the caps that decide the question above n = 31
# ---------------------------------------------------------------------------
def test_lifting_ceiling():
    assert hd.LIFT_CEILING == 2 * hd.LEECH == 393120
    assert hd.MAX_SETS == 396
    # unlimited groups: the 196560-vector budget binds, never more than 393120
    for K in (1932, 4320, 196560):
        res = hd.max_lifted(K, n_sets=math.inf)
        assert res["lifted"] == hd.LIFT_CEILING
        assert res["vectors_used"] == hd.LEECH
    # absolute ceiling of the template in n = 24 + d
    assert hd.template_value(16, n_sets=math.inf)["value"] == 4320 + 196560 + 393120


def test_max_lifted_is_monotone_and_capped():
    prev = -1
    for k in range(0, 500, 7):
        v = hd.max_lifted(4320, n_sets=k)["lifted"]
        assert v >= prev
        assert v <= hd.LIFT_CEILING
        prev = v


def test_42_sets_gives_weight_84():
    for d in range(7, 25):
        res = hd.max_lifted(hd.COHN_LOWER[d], n_sets=42)
        assert res["weight"] == 84
        assert res["lifted"] == 84 * hd.BEST_S == 41664


def test_max_lifted_from_sizes_matches_uniform():
    assert hd.max_lifted_from_sizes(1932, [496] * 42)["lifted"] == 84 * 496
    assert hd.max_lifted_from_sizes(12, [496] * 42)["lifted"] == 8 * 496   # only 4 groups exist
    # rearrangement: the largest sets must land on the weight-2 groups
    sizes = [100, 400, 300]
    assert hd.max_lifted_from_sizes(5, sizes)["lifted"] == 2 * 400 + 1 * 300


# ---------------------------------------------------------------------------
# the headline answer
# ---------------------------------------------------------------------------
def test_no_dimension_above_31_is_beaten_with_constructible_families():
    for r in hd.analyse(range(8, 25)):
        assert r["diff_at_n_sets"] < 0, r          # 42 x 496
        assert r["diff_kkw"] < 0, r                # 59 KKW sets, 28324 vectors


def test_ceiling_wins_only_at_37_and_38_and_crossover_is_39():
    rows = hd.analyse(range(8, 25))
    assert [r["n"] for r in rows if r["diff_ceiling"] > 0] == [37, 38]
    assert hd.crossover(range(8, 25)) == 39


def test_sets_needed_is_consistent():
    for r in hd.analyse(range(8, 25)):
        need = r["sets_to_beat"]
        if need["sets"] == 0:
            continue
        # "feasible" == that many weight-2 groups exist AND the 196560-vector budget allows
        # them; where it is False the template cannot reach the record at any set count.
        assert need["feasible"] == (r["diff_ceiling"] > 0), r
        if not need["feasible"]:
            assert r["value_ceiling"] <= r["record"]
            continue
        got = hd.template_value(r["d"], K=r["K"], n_sets=need["sets"])["value"]
        assert got > r["record"]
        one_less = hd.template_value(r["d"], K=r["K"], n_sets=need["sets"] - 1)["value"]
        assert one_less <= r["record"]
    # the two winnable dimensions need far more than the 42 (or 59) that exist
    assert hd.sets_needed(13, hd.record_lower(37)[0] + 1)["sets"] == 301
    assert hd.sets_needed(14, hd.record_lower(38)[0] + 1)["sets"] == 372


def test_record_table_uses_echols_where_it_improves_cohn():
    assert hd.record_lower(32) == (346432, "Echols 2026 (arXiv:2608.13906)")
    assert hd.record_lower(37)[0] == 496232
    assert hd.record_lower(35)[0] == hd.COHN_LOWER[35] == 409548
    # Edel-Rains-Sloane: tau_n >= 2^17 + 2^7 * A(n,8,8) + 4 * C(n,2), with A(32,8) >= 2^17
    # from the Cheng-Sloane [32,17,8] code and Echols' new A(n,8,8).
    for n, a in ((32, 1667), (33, 1788), (34, 1936), (37, 2832)):
        assert hd.ECHOLS_2026[n] == 2 ** 17 + 2 ** 7 * a + 2 * n * (n - 1)
    # and the prior values come from the prior A(n,8,8) (Brouwer's table)
    for n, a, prior in ((32, 1659, 345408), (33, 1777, 360640), (34, 1934, 380868),
                        (37, 2817, 494312)):
        assert hd.COHN_LOWER[n] == 2 ** 17 + 2 ** 7 * a + 2 * n * (n - 1) == prior


# ---------------------------------------------------------------------------
# the explicit R^8 configuration
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e8():
    R = hd.e8_roots()
    return R, hd.e8_eisenstein_sigma()


def test_e8_is_a_kissing_configuration(e8):
    R, _ = e8
    assert R.shape == (240, 8)
    assert sk.check_kissing(R)["ok"]


def test_e8_sigma_is_an_automorphism_of_order_3(e8):
    R, sigma = e8
    M, den = sigma
    I = np.eye(8, dtype=np.int64)
    assert np.array_equal(M @ M.T, den * den * I)                 # orthogonal
    assert np.array_equal(M @ M + den * M + den * den * I, np.zeros((8, 8), dtype=np.int64))
    img = hd.apply_sigma(R, sigma)
    assert set(map(tuple, img.tolist())) == set(map(tuple, R.tolist()))
    assert np.array_equal(hd.apply_sigma(hd.apply_sigma(img, sigma), sigma), R)  # sigma^3 = 1


def test_e8_hexagon_partition_is_optimal(e8):
    R, sigma = e8
    tri = hd.eisenstein_triangle_partition(R, sigma)
    assert len(tri) == 80
    pts = [p for t in tri for p in t]
    assert sorted(pts) == list(range(240))                        # a perfect partition
    for t in tri:
        assert np.all(R[list(t)].sum(axis=0) == 0)                # genuine triangles
    assert 2 * len(tri) == hd.weight_upper_bound(240) == 160      # attains the counting bound


def test_e8_ilp_confirms_80_triangles(e8):
    R, _ = e8
    res = tr.max_triangle_packing(R, time_limit=600)
    assert res["value"] == 80 and res["gap"] == 0.0 and res["proven"]


def test_e8_extra_spheres_reach_K8(e8):
    R, sigma = e8
    ex = hd.eisenstein_extra_spheres(R, sigma)
    assert ex["ok"] and ex["size"] == 240 and ex["code_ok"]
    assert ex["max_abs_a"] == ex["limit_2a"] // 2 == 12           # exactly 30 degrees
    assert ex["touching"] > 0


# ---------------------------------------------------------------------------
# the largest explicitly constructed disjoint family
# ---------------------------------------------------------------------------
KKW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "data", "external", "Kissing-Numbers")


@pytest.mark.skipif(
    not os.path.isdir(KKW_DIR),
    reason="data/external/ is gitignored (an upstream clone); re-create it from "
           "data/external/SOURCES.md to run this test",
)
def test_kkw_family_is_disjoint_and_independent():
    res = hd.kkw_family_sizes()
    assert res["ok"], res["why"]
    assert res["files"] == 59
    assert res["total"] == hd.KKW_TOTAL == 28324
    assert sorted(res["sizes"], reverse=True) == sorted(hd.KKW_SIZES, reverse=True)
    assert max(res["sizes"]) == 488 and min(res["sizes"]) == 460
    # more sets than T4.2's 42, but a smaller total, and still 7x short of the ceiling
    assert res["total"] > 42 * hd.BEST_S
    assert res["total"] < hd.LEECH

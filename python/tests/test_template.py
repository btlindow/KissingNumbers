"""Tests for python/template/ (docs/design.md T5.1): triangle packings, extra spheres, assignment."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import small_kissing as sk  # noqa: E402
from template import assign, extra_spheres as es, triangles as tr  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FAMILIES = os.path.join(ROOT, "data", "families")

MAX_TRIANGLES = {2: 2, 3: 4, 4: 8, 5: 12, 6: 24, 7: 42}
MAX_WEIGHT = {2: 4, 3: 8, 4: 16, 5: 26, 6: 48, 7: 84}
RECORD_WEIGHT = {2: 4, 3: 7, 4: 16, 5: 26, 6: 48, 7: 84}


# ---------------------------------------------------------------------------
# triangles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", range(2, 8))
def test_max_triangle_packing_exact(d):
    V, _ = sk.config(d)
    res = tr.max_triangle_packing(V)
    assert res["value"] == MAX_TRIANGLES[d]
    assert res["gap"] == 0.0 and res["proven"]
    assert abs(res["dual_bound"] - res["value"]) < 1e-6
    assert res["value"] == tr.zero_sum_bound(V)["bound"]  # the hand-proof matches the solver
    # chosen triangles are disjoint, genuine (sum zero), and cover K - leftover points
    pts = [p for t in res["chosen"] for p in t]
    assert len(pts) == len(set(pts)) == 3 * res["value"]
    for t in res["chosen"]:
        assert np.all(V[list(t)].sum(axis=0) == 0)
    assert len(res["leftover"]) == sk.K[d] - 3 * res["value"]
    assert res["lp_bound"] == pytest.approx(sk.K[d] / 3)


@pytest.mark.parametrize("d", range(2, 8))
def test_max_weight_partition_exact(d):
    V, _ = sk.config(d)
    res = tr.max_weight_partition(V)
    assert res["value"] == MAX_WEIGHT[d] == tr.weight_upper_bound(sk.K[d])
    assert res["proven"]
    assert 2 * len(res["triangles"]) + len(res["pairs"]) == res["value"]
    N = (V * V).sum(axis=1)
    for g in res["triangles"] + res["pairs"]:
        for a in g:
            for b in g:
                if a < b:
                    assert sk.cos_le_neg_half(int(V[a] @ V[b]), int(N[a]), int(N[b]))


def test_zero_sum_bound_d5():
    V, _ = sk.config(5)
    z = tr.zero_sum_bound(V)
    assert z["sums_to_zero"] and z["bound"] == 12 and "leftover" in z["reason"]
    # and a non-antipodal 40-point set would only get the trivial bound
    assert tr.zero_sum_bound(V[:40][np.arange(40) != 0].tolist() + [V[1].tolist()])["bound"] == 13


def test_weight_upper_bound():
    assert [tr.weight_upper_bound(K) for K in (6, 12, 24, 40, 72, 126)] == [4, 8, 16, 26, 48, 84]
    assert tr.weight_upper_bound(41) == 27 and tr.weight_upper_bound(43) == 28 and tr.weight_upper_bound(44) == 29


@pytest.mark.parametrize("d", range(2, 8))
def test_record_partition_vs_optimum(d):
    rec = tr.record_partition(d, FAMILIES)
    assert rec["weight"] == RECORD_WEIGHT[d]
    a = tr.analyse(d)
    assert a["record_is_optimal"] == (d != 3)
    assert a["record_slack_weight"] == (1 if d == 3 else 0)


def test_admissible_pairs_counts():
    # antipodal pairs K/2 plus one pair per triangle edge (3 per triangle)
    for d in (2, 3, 4, 5):
        V, _ = sk.config(d)
        assert len(tr.admissible_pairs(V)) == sk.K[d] // 2 + 3 * len(tr.triangles_of(V))


def test_continuous_search_recovers_cuboctahedron():
    # 12 points in R^3 with 4 prescribed triangles: the minimum of the max other cosine is 1/2
    res = tr.continuous_triangle_search(3, 12, 4, restarts=6, seed=3)
    assert res["best"] is not None and res["best"]["triangle_err"] < 1e-7
    assert res["best"]["max_cos"] <= 0.5 + 1e-6 and res["kissing"]
    # 13 points can never reach 60 degrees (K(3) = 12)
    res = tr.continuous_triangle_search(3, 13, 4, restarts=3, seed=3)
    assert not res["kissing"]


# ---------------------------------------------------------------------------
# extra spheres
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", range(2, 8))
def test_upper_bound(d):
    ub = es.upper_bound(d)
    assert ub["bound"] >= sk.K[d] and ub["tight"] == (d <= 4)


@pytest.mark.parametrize("d", range(2, 8))
def test_record_extras_exact(d):
    rows, f = es.record_extras(d, FAMILIES)
    assert len(rows) == sk.K[d]
    assert es.check_exact(rows, d)["ok"]


def test_record_extras_rejects_violation():
    rows, _ = es.record_extras(3, FAMILIES)
    T, _ = sk.config(3)
    from kiss_ref.exact import Q
    bad = rows[:-1] + [[Q(int(x)) for x in T[0]]]  # a T vector itself (0 degrees)
    assert not es.check_exact(bad, 3)["ok"]


@pytest.mark.parametrize("d", (2, 3, 4))
def test_ilp_pool_reaches_K_and_not_more(d):
    pool = es.candidate_pool(d, max_entry=2, extra_rotations=20)
    ilp = es.max_extras_ilp(pool)
    assert ilp["value"] == sk.K[d] and ilp["gap"] <= 1e-9
    S = pool["P"][ilp["chosen"]]
    assert (np.triu(S @ S.T, 1) <= 0.5 + 1e-9).all()
    assert (S @ pool["Tu"].T <= es.SQRT3_2 + 1e-9).all()


def test_continuous_extra_search_fails_for_K_plus_1_in_R2():
    res = es.continuous_extra_search(2, restarts=4, seed=0)
    assert not res["hit"] and res["best_max_violation"] > 1e-3
    # and succeeds for K itself
    res = es.continuous_extra_search(2, n=6, restarts=4, seed=0)
    assert res["best_max_violation"] <= 1e-6


# ---------------------------------------------------------------------------
# assignment / family rewriting
# ---------------------------------------------------------------------------
def test_family_count_and_assignment():
    assert assign.family_count([496] * 5, [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9], [10, 11]], 12) == 200044
    assert assign.family_count([496] * 4, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], 12) == 200540
    a = assign.assign_sets_to_groups([10, 30, 20], [[0, 1], [2, 3, 4], [5, 6]])
    assert a["order"][1] == 1 and sorted(a["order"]) == [0, 1, 2]
    a = assign.assign_sets_to_groups([1, 5, 3, 4], [[0, 1, 2], [3, 4]])
    assert a["sizes"] == [5, 4]


@pytest.mark.parametrize("n", range(26, 32))
def test_record_families_count(n):
    raw, files = assign.load_raw(os.path.join(FAMILIES, f"dim{n}"))
    sizes = [int(s["size"]) for s in raw["sets"]]
    assert assign.family_count(sizes, raw["T"]["groups"], int(raw["extra"]["count"])) == raw["count"]
    opt = assign.optimal_groups(n - 24)
    best = assign.assign_sets_to_groups(sizes, opt)["count_without_extra"] + int(raw["extra"]["count"])
    assert best - raw["count"] == (496 if n == 27 else 0)


def test_rewrite_family_dim27(tmp_path):
    out = tmp_path / "dim27"
    new = assign.rewrite_family(os.path.join(FAMILIES, "dim27"), str(out), assign.optimal_groups(3))
    assert new["count"] == 200540 and len(new["sets"]) == 4
    assert all(len(g) == 3 for g in new["T"]["groups"])
    assert new["extra"]["count"] == 12
    for s in new["sets"]:
        assert (out / s["file"]).exists() and assign.sha256_file(str(out / s["file"])) == s["sha256"]
    j = json.load(open(out / "family.json"))
    assert j["count"] == 200540

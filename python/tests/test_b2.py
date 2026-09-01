"""B2 — light acceptance tests for the quadruple orbit data (scoping study).

Checks data/scheme/quad_orbits.json (tools/quad_stats + cuda/quad_stats.cu)
against data/scheme/orbitals.json (T2.3) and the intersection numbers, plus
the internal consistency of bounds.sdp4_scoping.  Nothing here solves an SDP.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bounds.orbitals import CONFLICT, load_orbitals  # noqa: E402
from bounds.sdp4_scoping import (  # noqa: E402
    DEFAULT_QUAD_JSON,
    check_quad,
    load_quad,
    scoping,
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_QUAD_JSON),
    reason="data/scheme/quad_orbits.json not generated (run tools/quad_stats)",
)


@pytest.fixture(scope="module")
def quad():
    return load_quad()


@pytest.fixture(scope="module")
def O():
    return load_orbitals()


def test_check_quad_passes(quad, O):
    chk = check_quad(quad, O)
    assert chk["anchors_ok"]


def test_diagonal_classes_reproduce_T23(quad, O):
    """Stab(x, x) = Stab(x, -x) = Stab(x): dim T must equal T2.3's D = 148."""
    by_i = {int(c["i"]): c for c in quad["classes"]}
    for i in (0, 6):
        assert int(by_i[i]["dimT_hist"]) == O.D == 148
        assert int(by_i[i]["dimT_labels"]) == O.D
        # per-z-orbit w-orbit counts are the labels per class of orbitals.json
        nlab = Counter(sum(1 for u in O.orbitals if u.i == j) for j in range(7))
        got = Counter(int(z["label_orbits"]) for z in by_i[i]["z_orbits"])
        assert got == nlab


def test_conjugate_classes_agree(quad):
    """y -> -y conjugacy: the deterministic histogram counts of classes 1/5
    and 2/4 must agree, per z-orbit cell multiset."""
    by_i = {int(c["i"]): c for c in quad["classes"]}
    for a, b in ((1, 5), (2, 4)):
        assert int(by_i[a]["dimT_hist"]) == int(by_i[b]["dimT_hist"])
        ha = Counter((int(z["size"]), int(z["hist_orbits"])) for z in by_i[a]["z_orbits"])
        hb = Counter((int(z["size"]), int(z["hist_orbits"])) for z in by_i[b]["z_orbits"])
        assert ha == hb


def test_orthogonal_split_visible_at_level3(quad):
    """T2.3's split cell: class 3 has exactly two z-orbits with cell (3,3,3),
    of sizes 924 and 42240, and they have different w-orbit counts (the
    lattice's two flavours of orthogonal triple stay distinguishable)."""
    c3 = next(c for c in quad["classes"] if int(c["i"]) == 3)
    ortho = [z for z in c3["z_orbits"] if int(z["j"]) == 3 and int(z["k"]) == 3]
    assert sorted(int(z["size"]) for z in ortho) == [924, 42240]
    counts = {int(z["size"]): int(z["hist_orbits"]) for z in ortho}
    assert counts[924] != counts[42240]


def test_triangle_and_clique_orbits(quad, O):
    """Co_0 is transitive on ordered conflict-graph triangles (one z-orbit
    with cell (5,5,5), size p^5_55 = 891) and on ordered 4-cliques."""
    s = scoping(quad)
    assert s["triangle_orbits"] == 1
    assert s["clique4_orbits"] == 1
    c5 = next(c for c in quad["classes"] if int(c["i"]) == CONFLICT)
    tri = [z for z in c5["z_orbits"] if int(z["j"]) == CONFLICT and int(z["k"]) == CONFLICT]
    assert len(tri) == 1 and int(tri[0]["size"]) == 891


def test_scoping_totals_consistent(quad):
    s = scoping(quad)
    assert s["ordered_orbits"] == sum(s["dimT"].values())
    assert s["ordered_orbits_lower"] == sum(s["dimT_lower"].values())
    assert s["admissible_distinct4"] <= s["admissible_ordered"] <= s["ordered_orbits"]
    assert s["vars_bracket"][0] <= s["vars_bracket"][1]
    assert s["max_block"] == max(s["dimT"].values())
    # totals stored by the tool must match a recount from the detail
    t = quad["totals"]
    assert int(t["ordered_orbits_upper"]) == s["ordered_orbits"]
    assert int(t["admissible"]) == s["admissible_ordered"]
    assert int(t["clique4_orbits"]) == s["clique4_orbits"]


def test_z_orbits_match_intersection_numbers(quad):
    """Per class i, the z-orbit sizes summed per (j, k) cell must equal the
    intersection numbers p^i_{jk} of T2.2."""
    path = os.path.join(os.path.dirname(DEFAULT_QUAD_JSON), "intersection_numbers.json")
    with open(path) as f:
        p = json.load(f)["p"]
    for c in quad["classes"]:
        i = int(c["i"])
        cell = Counter()
        for z in c["z_orbits"]:
            cell[(int(z["j"]), int(z["k"]))] += int(z["size"])
        for (j, k), n in cell.items():
            assert n == int(p[i][j][k])

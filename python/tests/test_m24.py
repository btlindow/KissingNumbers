"""Acceptance tests for python/group/m24.py (docs/design.md T4.1 item 1 / T3.4)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from group.m24 import (  # noqa: E402
    M24_ORDER,
    PSL_ORDER,
    XI_TETRAD_SIGNS,
    delta,
    find_sextet,
    group_order,
    m24_generators,
    permute_mask,
    preserves_code,
    psl_generators,
    verify_aut,
    xi_numerators,
)
from kiss_ref.golay import golay_codewords, octads  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "group")


@pytest.fixture(scope="module")
def gens():
    return m24_generators()


def test_generators_are_permutations(gens):
    assert len(gens) == 3
    for g in gens:
        assert sorted(g) == list(range(24))


def test_every_generator_preserves_the_code(gens):
    words = golay_codewords()
    for g in gens:
        assert preserves_code(g, words)


def test_a_random_permutation_does_not_preserve_the_code():
    import random

    rng = random.Random(1)
    perm = list(range(24))
    rng.shuffle(perm)
    assert not preserves_code(perm)
    # and the transposition (0 1) neither
    t = list(range(24))
    t[0], t[1] = 1, 0
    assert not preserves_code(t)


def test_swapped_delta_variant_fails():
    assert not preserves_code(delta(swap=True))


def test_orders():
    a, g = psl_generators()
    assert group_order([a, g]) == PSL_ORDER
    assert group_order(m24_generators()) == M24_ORDER == 244_823_040


def test_generator_element_orders(gens):
    from sympy.combinatorics import Permutation

    assert [Permutation(g).order() for g in gens] == [23, 2, 5]


def test_permute_mask():
    ident = list(range(24))
    assert permute_mask(0b1011, ident) == 0b1011
    a, _ = psl_generators()
    assert permute_mask(1 << 22, a) == 1  # 22 -> 0
    assert permute_mask(1 << 23, a) == 1 << 23  # inf fixed


def test_sextet():
    s = find_sextet()
    assert len(s) == 6 and sorted(i for t in s for i in t) == list(range(24))
    o = set(octads())
    for i in range(6):
        for j in range(i + 1, 6):
            assert sum(1 << k for k in s[i] + s[j]) in o
    assert s[0] == [0, 1, 2, 3]


def test_xi_is_a_bijection_of_the_minimal_vectors():
    s = find_sextet()
    n_in, bij = verify_aut(xi_numerators(s, XI_TETRAD_SIGNS), 2)
    assert (n_in, bij) == (196560, True)


def test_uniform_sign_xi_is_not_an_automorphism():
    s = find_sextet()
    n_in, bij = verify_aut(xi_numerators(s, (1, 1, 1, 1, 1, 1)), 2)
    assert not bij
    assert n_in == 49104


def test_data_files_match(gens):
    path = os.path.join(DATA, "m24_generators.txt")
    if not os.path.exists(path):
        pytest.skip("data/group not generated")
    rows = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            rows.append([int(v) for v in line.split()])
    assert rows == gens
    xi_path = os.path.join(DATA, "xi.txt")
    lines = [ln.split("#", 1)[0].strip() for ln in open(xi_path)]
    lines = [ln for ln in lines if ln]
    assert lines[0] == "den 2"
    num = [[int(v) for v in ln.split()] for ln in lines[1:]]
    assert num == xi_numerators(find_sextet(), XI_TETRAD_SIGNS)

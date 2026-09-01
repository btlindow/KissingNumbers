"""Acceptance tests for python/kiss_ref/golay.py (docs/design.md T1.1)."""

from __future__ import annotations

import os
import random
import sys
from itertools import combinations

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref.golay import (  # noqa: E402
    GENERATOR_POLY,
    carry_less_multiply,
    golay_codewords,
    is_codeword,
    octads,
    weight_distribution,
)

EXPECTED_WEIGHTS = {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}
ALL_ONES = (1 << 24) - 1


@pytest.fixture(scope="module")
def words() -> list[int]:
    return golay_codewords()


def test_generator_polynomial_divides_x23_minus_1():
    # g(x) must divide x^23 + 1 over GF(2); long division.
    n = (1 << 23) | 1
    g = GENERATOR_POLY
    r = n
    while r and r.bit_length() >= g.bit_length():
        r ^= g << (r.bit_length() - g.bit_length())
    assert r == 0


def test_count_and_distinct(words):
    assert len(words) == 4096
    assert len(set(words)) == 4096
    assert words == sorted(words)
    assert all(0 <= w < (1 << 24) for w in words)


def test_weight_distribution(words):
    assert weight_distribution(words) == EXPECTED_WEIGHTS


def test_parity_bit_is_overall_parity(words):
    for w in words:
        assert bin(w & ((1 << 23) - 1)).count("1") % 2 == (w >> 23) & 1
        assert bin(w).count("1") % 2 == 0


def test_all_ones_and_zero_present(words):
    assert words[0] == 0
    assert ALL_ONES in set(words)
    assert is_codeword(ALL_ONES)


def test_closed_under_xor_random_pairs(words):
    rng = random.Random(20260825)
    sample = rng.sample(words, 300)
    s = set(words)
    for a, b in combinations(sample, 2):
        assert (a ^ b) in s


def test_linear_span_of_basis_equals_code(words):
    # Basis: x^k * g(x) for k = 0..11, extended by parity.  Their span must be
    # exactly the codeword set (full closure check of a 12-dim GF(2) space).
    basis = []
    for k in range(12):
        c = carry_less_multiply(1 << k, GENERATOR_POLY)
        c |= (bin(c).count("1") & 1) << 23
        basis.append(c)
    span = {0}
    for b in basis:
        span |= {x ^ b for x in span}
    assert len(span) == 4096
    assert span == set(words)


def test_minimum_distance_is_8(words):
    # Linear code: min distance = min nonzero weight.
    assert min(bin(w).count("1") for w in words if w) == 8
    # And directly on a random sample of pairs.
    rng = random.Random(7)
    sample = rng.sample(words, 200)
    for a, b in combinations(sample, 2):
        assert bin(a ^ b).count("1") >= 8


def test_octads(words):
    o = octads()
    assert len(o) == 759
    assert all(bin(w).count("1") == 8 for w in o)
    assert o == sorted(o)
    # Complement of an octad is a weight-16 codeword.
    s = set(words)
    assert all((ALL_ONES ^ w) in s for w in o)
    # Any two distinct octads meet in 0, 2 or 4 points (Steiner system S(5,8,24)).
    rng = random.Random(3)
    sample = rng.sample(o, 120)
    for a, b in combinations(sample, 2):
        assert bin(a & b).count("1") in (0, 2, 4)


def test_is_codeword_negative_cases(words):
    s = set(words)
    assert not is_codeword(-1)
    assert not is_codeword(1 << 24)
    rng = random.Random(11)
    hits = 0
    for _ in range(2000):
        m = rng.getrandbits(24)
        assert is_codeword(m) == (m in s)
        hits += m in s
    # 4096 / 2^24 ~ 0.024%: with 2000 draws we expect ~0.5 hits.
    assert hits <= 10
    # Flipping one bit of a codeword never yields a codeword (d = 8).
    for w in rng.sample(words, 50):
        for i in range(24):
            assert not is_codeword(w ^ (1 << i))


def test_weight_distribution_helper():
    assert weight_distribution([]) == {}
    assert weight_distribution([0, 1, 3, 3]) == {0: 1, 1: 1, 2: 2}


def test_sorted_dump_roundtrip(words, tmp_path):
    # Write the sorted masks to a temp file (outside the repo) and read back.
    p = tmp_path / "golay_py.txt"
    p.write_text("\n".join(f"{w:06x}" for w in words) + "\n")
    back = [int(line, 16) for line in p.read_text().split()]
    assert back == words

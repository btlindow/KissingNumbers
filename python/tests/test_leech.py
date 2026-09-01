"""Acceptance tests for python/kiss_ref/leech.py (docs/design.md T1.2).

Everything here is computed from the independent Python generator; the only
test that touches the C++ output is test_cross_check_leech_min_txt, which is
skipped when data/leech_min.txt is absent.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref.golay import is_codeword, octads  # noqa: E402
from kiss_ref.leech import (  # noqa: E402
    DIM,
    EXPECTED_HISTOGRAM,
    N_MIN,
    canonical_sort,
    ip_histogram,
    is_lattice_vector,
    is_lattice_vector_rows,
    leech_min_vectors,
    shape_four_four,
    shape_octad,
    shape_three_one,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEECH_TXT = os.path.join(REPO, "data", "leech_min.txt")


@pytest.fixture(scope="module")
def C() -> np.ndarray:
    return leech_min_vectors()


def _rowset(a: np.ndarray) -> set[tuple[int, ...]]:
    return set(map(tuple, np.asarray(a, dtype=np.int64).tolist()))


def test_shape_counts():
    a, b, c = shape_octad(), shape_three_one(), shape_four_four()
    assert a.shape == (97152, DIM) and a.dtype == np.int8
    assert b.shape == (98304, DIM) and b.dtype == np.int8
    assert c.shape == (1104, DIM) and c.dtype == np.int8
    assert 97152 + 98304 + 1104 == N_MIN
    # Shapes as multisets of |coordinate| values.
    assert np.all(np.sort(np.abs(a), axis=1)[:, -8:] == 2) and np.all(np.sort(np.abs(a), axis=1)[:, :16] == 0)
    assert np.all(np.sort(np.abs(b), axis=1)[:, -1] == 3) and np.all(np.sort(np.abs(b), axis=1)[:, :23] == 1)
    assert np.all(np.sort(np.abs(c), axis=1)[:, -2:] == 4) and np.all(np.sort(np.abs(c), axis=1)[:, :22] == 0)
    # Octad shape: even number of minus signs, support is an octad.
    assert np.all((a < 0).sum(axis=1) % 2 == 0)
    octad_set = set(octads())
    supports = ((a != 0).astype(np.int64) << np.arange(DIM)).sum(axis=1)
    assert set(supports.tolist()) == octad_set
    assert len(set(supports.tolist())) == 759


def test_count_distinct_and_canonical_order(C):
    assert C.shape == (N_MIN, DIM)
    assert C.dtype == np.int8
    assert len(_rowset(C)) == N_MIN
    # Strictly increasing in signed lexicographic order, coordinate 0 first.
    as_list = C.astype(np.int64).tolist()
    assert all(as_list[i] < as_list[i + 1] for i in range(N_MIN - 1))
    # Agreement with Python's own tuple sort (independent of np.lexsort).
    assert as_list == sorted(as_list)


def test_canonical_sort_helper():
    rows = np.array([[1, 0], [-1, 5], [1, -2], [-1, -128]], dtype=np.int8)
    got = canonical_sort(rows).tolist()
    assert got == [[-1, -128], [-1, 5], [1, -2], [1, 0]]


def test_norms(C):
    norms = (C.astype(np.int64) ** 2).sum(axis=1)
    assert np.all(norms == 32)


def test_negation_closure(C):
    s = _rowset(C)
    neg = _rowset(-C.astype(np.int64))
    assert neg == s
    # Also as a permutation: -C[i] is at a unique index; involution.
    index = {row: i for i, row in enumerate(map(tuple, C.astype(np.int64).tolist()))}
    neg_idx = np.array([index[tuple((-np.array(r)).tolist())] for r in C.astype(np.int64).tolist()])
    assert np.all(neg_idx[neg_idx] == np.arange(N_MIN))
    assert np.all(neg_idx != np.arange(N_MIN))


def test_all_pass_membership_vectorised(C):
    ok = is_lattice_vector_rows(C)
    assert ok.shape == (N_MIN,)
    assert bool(ok.all())
    # Scalar version on a sample agrees.
    rng = np.random.default_rng(1)
    for i in rng.choice(N_MIN, 300, replace=False):
        assert is_lattice_vector(C[i])
    # Complementary residue class is a codeword as well.
    for i in rng.choice(N_MIN, 300, replace=False):
        x = C[i].astype(int)
        m = x[0] % 2
        r = 1 if m else 0
        mask = sum(1 << k for k in range(DIM) if x[k] % 4 == r)
        assert is_codeword(mask)


def test_membership_negative_and_non_minimal_cases():
    z = [0] * DIM
    assert is_lattice_vector(z)
    assert not is_lattice_vector([1] * DIM)  # sum 24 = 0 mod 8, needs 4
    mixed = [1] * DIM
    mixed[3] = 2
    assert not is_lattice_vector(mixed)
    assert not is_lattice_vector([2] * 7 + [0] * 17)  # weight-7 support, no such codeword
    assert not is_lattice_vector([2] * 9 + [0] * 15)
    # (2^8, 0^16) on coordinates 0..7 iff {0..7} is an octad.
    assert is_lattice_vector([2] * 8 + [0] * 16) == is_codeword(0xFF)
    # A real octad, then spoiled by moving one coordinate.
    o = octads()[0]
    x = [2 if (o >> k) & 1 else 0 for k in range(DIM)]
    assert is_lattice_vector(x)
    k_in = next(k for k in range(DIM) if (o >> k) & 1)
    k_out = next(k for k in range(DIM) if not (o >> k) & 1)
    y = list(x)
    y[k_in], y[k_out] = 0, 2
    assert not is_lattice_vector(y)
    y = list(x)
    y[k_in] = -2  # odd number of minus signs -> sum = 12 = 4 mod 8
    assert not is_lattice_vector(y)
    # (-3, 1^23) and (3, -1^23) are both in the lattice (they are negatives).
    assert is_lattice_vector([-3] + [1] * 23)
    assert is_lattice_vector([3] + [-1] * 23)
    # Non-minimal lattice vectors (norm 64): in the lattice, not minimal.
    C = leech_min_vectors()
    s = _rowset(C)
    assert is_lattice_vector([8] + [0] * 23)
    assert tuple([8] + [0] * 23) not in s
    assert is_lattice_vector([4, 4, 4, 4] + [0] * 20)
    assert tuple([4, 4, 4, 4] + [0] * 20) not in s
    assert not is_lattice_vector([4] + [0] * 23)  # sum 4, needs 0 mod 8
    assert not is_lattice_vector([4, 4, 4] + [0] * 21)
    assert not is_lattice_vector([1] * 23)  # wrong length
    # Sums of two minimal vectors are lattice vectors.
    rng = np.random.default_rng(2)
    for _ in range(500):
        i, j = rng.integers(0, N_MIN, 2)
        assert is_lattice_vector(C[i].astype(int) + C[j].astype(int))
    # Random small vectors are (essentially never) lattice vectors.
    hits = 0
    for _ in range(5000):
        hits += is_lattice_vector(rng.integers(-4, 5, DIM))
    assert hits == 0


@pytest.mark.parametrize("shape_fn", [shape_octad, shape_three_one, shape_four_four])
def test_histogram_from_each_shape(C, shape_fn):
    base = shape_fn()[0]
    assert ip_histogram(C, base) == EXPECTED_HISTOGRAM


def test_histogram_random_bases(C):
    rng = np.random.default_rng(20260825)
    idx = rng.choice(N_MIN, 16, replace=False)
    # Batched int64 matmul, one column per base.
    ips = C.astype(np.int64) @ C[idx].astype(np.int64).T
    assert ips.shape == (N_MIN, 16)
    for col in range(16):
        vals, counts = np.unique(ips[:, col], return_counts=True)
        assert {int(v): int(c) for v, c in zip(vals, counts)} == EXPECTED_HISTOGRAM
    assert sum(EXPECTED_HISTOGRAM.values()) == N_MIN


def test_ip_values_only_expected_classes(C):
    # 256 random pairs of distinct vectors: inner product in the six classes.
    rng = np.random.default_rng(3)
    i = rng.integers(0, N_MIN, 256)
    j = rng.integers(0, N_MIN, 256)
    ips = (C[i].astype(np.int64) * C[j].astype(np.int64)).sum(axis=1)
    assert set(ips.tolist()) <= {-32, -16, -8, 0, 8, 16, 32}


@pytest.mark.skipif(not os.path.exists(LEECH_TXT), reason="data/leech_min.txt not generated")
def test_cross_check_leech_min_txt(C):
    """Explicit cross-check against the C++ generator's text output."""
    file_rows = np.loadtxt(LEECH_TXT, dtype=np.int64)
    assert file_rows.shape == (N_MIN, DIM)
    assert file_rows.min() >= -128 and file_rows.max() <= 127
    # Same set.
    assert _rowset(file_rows) == _rowset(C)
    # File order is the canonical sorted order (checked with Python's tuple
    # sort, independent of numpy's lexsort).
    as_list = file_rows.tolist()
    assert as_list == sorted(as_list)
    # And hence identical row-by-row to the Python array.
    assert np.array_equal(file_rows, C.astype(np.int64))

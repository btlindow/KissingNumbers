"""Leech lattice minimal vectors — independent numpy reference (docs/design.md T1.2).

Written without reading src/leech.cpp. The only shared input is the spec in
README.md section 1.2 / docs/design.md section 2.2 and the (independent) Golay code in
kiss_ref.golay. Nothing here reads the C++ generator's output; the explicit
cross-check lives in python/tests/test_leech.py.

Integer (sqrt 8) scaling: every minimal vector has squared norm 32.

Three shapes:

* (+-2^8, 0^16): +-2 on the coordinates of an octad, even number of minus
  signs                                                  759 * 2^7 = 97152
* (-+3, +-1^23): start from (-3 at i, +1 elsewhere) and negate the
  coordinates lying in a Golay codeword c                24 * 4096 = 98304
* (+-4, +-4, 0^22): two +-4 on any two coordinates       C(24,2) * 4 = 1104

Canonical order (PLAN section 2.2): rows sorted lexicographically ascending
as signed integers, coordinate 0 most significant.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

from .golay import golay_codewords, is_codeword, octads

__all__ = [
    "N_MIN",
    "DIM",
    "shape_octad",
    "shape_three_one",
    "shape_four_four",
    "leech_min_vectors",
    "canonical_sort",
    "is_lattice_vector",
    "is_lattice_vector_rows",
    "ip_histogram",
]

N_MIN = 196560
DIM = 24
EXPECTED_HISTOGRAM = {32: 1, 16: 4600, 8: 47104, 0: 93150, -8: 47104, -16: 4600, -32: 1}


def _mask_to_indices(mask: int) -> list[int]:
    return [i for i in range(DIM) if (mask >> i) & 1]


def shape_octad() -> np.ndarray:
    """(+-2^8, 0^16) with an even number of minus signs: 97152 rows, int8."""
    rows = np.zeros((759 * 128, DIM), dtype=np.int8)
    # Sign patterns on 8 coordinates with even parity: enumerate all 256 and
    # keep those with an even number of ones (independent of the C++ trick of
    # fixing the last sign).
    patterns = np.array([[(s >> j) & 1 for j in range(8)] for s in range(256)], dtype=np.int8)
    even = patterns[patterns.sum(axis=1) % 2 == 0]
    assert even.shape == (128, 8)
    signs = (1 - 2 * even).astype(np.int8) * np.int8(2)  # +2 / -2
    r = 0
    for o in octads():
        idx = _mask_to_indices(o)
        rows[r : r + 128, idx] = signs
        r += 128
    return rows


def shape_three_one() -> np.ndarray:
    """(-+3, +-1^23): 98304 rows, int8."""
    words = golay_codewords()
    # sign matrix: row c, column j = -1 if j in c else +1
    flip = np.array([[-1 if (c >> j) & 1 else 1 for j in range(DIM)] for c in words], dtype=np.int8)
    rows = np.empty((DIM * len(words), DIM), dtype=np.int8)
    for i in range(DIM):
        base = np.ones(DIM, dtype=np.int8)
        base[i] = -3
        rows[i * len(words) : (i + 1) * len(words)] = base[None, :] * flip
    return rows


def shape_four_four() -> np.ndarray:
    """(+-4, +-4, 0^22): 1104 rows, int8."""
    rows = np.zeros((1104, DIM), dtype=np.int8)
    r = 0
    for i, j in combinations(range(DIM), 2):
        for si in (4, -4):
            for sj in (4, -4):
                rows[r, i] = si
                rows[r, j] = sj
                r += 1
    assert r == 1104
    return rows


def canonical_sort(rows: np.ndarray) -> np.ndarray:
    """Rows sorted lexicographically ascending, coordinate 0 most significant."""
    rows = np.asarray(rows, dtype=np.int8)
    # np.lexsort sorts by the LAST key first, so pass the columns reversed.
    order = np.lexsort(rows.T[::-1])
    return rows[order]


@lru_cache(maxsize=None)
def _leech_min_cached() -> np.ndarray:
    all_rows = np.concatenate([shape_octad(), shape_three_one(), shape_four_four()], axis=0)
    out = canonical_sort(all_rows)
    out.setflags(write=False)
    return out


def leech_min_vectors() -> np.ndarray:
    """All 196560 minimal vectors, int8 (N, 24), canonical order. Read-only view."""
    return _leech_min_cached()


def is_lattice_vector(x) -> bool:
    """README section 1.2 membership test for an arbitrary integer vector.

    1. all coordinates share a parity m;
    2. {i : x_i = 2 mod 4} (m = 0) or {i : x_i = 3 mod 4} (m = 1) is a Golay
       codeword;
    3. sum(x) = 4m mod 8.
    """
    x = [int(v) for v in x]
    if len(x) != DIM:
        return False
    m = x[0] % 2
    if any(v % 2 != m for v in x):
        return False
    r = 3 if m else 2
    mask = 0
    for i, v in enumerate(x):
        if v % 4 == r:  # Python's % is non-negative for positive modulus
            mask |= 1 << i
    if not is_codeword(mask):
        return False
    return sum(x) % 8 == 4 * m


def is_lattice_vector_rows(rows: np.ndarray) -> np.ndarray:
    """Vectorised membership test over an (n, 24) integer array -> bool (n,)."""
    rows = np.asarray(rows, dtype=np.int64)
    m = rows[:, 0] & 1
    same_parity = np.all((rows & 1) == m[:, None], axis=1)
    r = np.where(m == 1, 3, 2)
    bits = ((rows & 3) == r[:, None]).astype(np.int64)
    masks = (bits << np.arange(DIM, dtype=np.int64)[None, :]).sum(axis=1)
    codeword_set = set(golay_codewords())
    in_code = np.fromiter((int(mk) in codeword_set for mk in masks), dtype=bool, count=len(masks))
    sum_ok = (rows.sum(axis=1) % 8) == 4 * m
    return same_parity & in_code & sum_ok


def ip_histogram(rows: np.ndarray, base) -> dict[int, int]:
    """Histogram of <base, row> over all rows, exact int64 arithmetic."""
    ips = np.asarray(rows, dtype=np.int64) @ np.asarray(base, dtype=np.int64)
    vals, counts = np.unique(ips, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}

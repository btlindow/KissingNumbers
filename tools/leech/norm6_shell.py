#!/usr/bin/env python3
"""The second shell of the Leech lattice: the 16773120 vectors of norm 6 (squared norm 48
in the sqrt-8 integer scaling), generated from the Golay code and verified against the theta
series.

Shapes (x in Z^24, |x|^2 = 48).  Membership (README 1.2): all coordinates share a parity m;
{i : x_i = 2 mod 4} (m = 0) resp. {i : x_i = 3 mod 4} (m = 1) is a Golay codeword; and
sum(x) = 4m mod 8.  Writing x = 2z for m = 0, |z|^2 = 12 and {i : z_i odd} must be a
codeword, which forces z of shape 1^12 on a DODECAD, 2*1^8 with the 1s on an OCTAD, or
2^3 (excluded below: sum(x) = +-4, +-12 is never 0 mod 8).  For m = 1 the shapes are
(5, 1^23) and (3^3, 1^21).

    (+-2^12, 0^12)   on a dodecad, even number of minus signs      2576 * 2048 = 5275648
    (+-4, +-2^8, 0^15)  the +-2s on an octad, the +-4 off it        759 * 4096 = 3108864
    (-+5, +-1^23)    base 5 at one place, signs flipped on a word    24 * 4096 =   98304
    (-+3^3, +-1^21)  base -3 on three places, ditto                2024 * 4096 = 8290304
                                                                   ----------------------
                                                                              16773120

Every candidate is put through the independent membership test, so the shape analysis is
checked rather than trusted.

MEMORY.  The shell is 384 MiB on disk and is written STREAMING: each shape is generated in
chunks of at most ~200k rows, filtered, and appended to a memory-mapped .npy, so the peak
working set is a few hundred MiB rather than the whole shell plus its int64 intermediates.
`load()` returns a read-only memmap, so callers do not pull 384 MiB into RAM either.

    python tools/leech/norm6_shell.py            # build, verify, cache
"""
from __future__ import annotations
import os, sys
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
from kiss_ref.golay import golay_codewords, octads  # noqa: E402

DIM, NORM, TOTAL = 24, 48, 16773120
CHUNK = 200_000
CACHE = os.environ.get("LEECH_NORM48_CACHE",
                       os.path.join(ROOT, ".cache", "leech_norm48.npy"))


def _mask_rows(masks):
    m = np.asarray(masks, dtype=np.int64)[:, None]
    return ((m >> np.arange(DIM, dtype=np.int64)[None, :]) & 1).astype(np.int8)


def _even_sign_patterns(k):
    p = np.array([[(s >> j) & 1 for j in range(k)] for s in range(1 << k)], dtype=np.int8)
    return p[p.sum(1) % 2 == 0]


def _codeword_lookup():
    tab = np.zeros(1 << DIM, dtype=bool)
    tab[np.array(golay_codewords(), dtype=np.int64)] = True
    return tab


def in_leech(rows, tab=None):
    """Vectorised README-1.2 membership test (no per-row Python)."""
    tab = _codeword_lookup() if tab is None else tab
    r = np.asarray(rows, dtype=np.int64)
    m = r[:, 0] & 1
    ok = np.all((r & 1) == m[:, None], axis=1)
    tgt = np.where(m == 1, 3, 2)
    bits = ((r & 3) == tgt[:, None]).astype(np.int64)
    masks = (bits << np.arange(DIM, dtype=np.int64)[None, :]).sum(axis=1)
    ok &= tab[masks]
    ok &= (r.sum(axis=1) % 8) == 4 * m
    return ok


# --------------------------------------------------------------- shape generators
# Each yields int8 chunks of candidates; the caller filters and appends.

def _gen_dodecad():
    words = np.array(golay_codewords(), dtype=np.int64)
    wt = np.array([bin(int(w)).count("1") for w in words])
    dod = words[wt == 12]
    assert len(dod) == 2576, len(dod)
    sgn = (1 - 2 * _even_sign_patterns(12)) * np.int8(2)      # 2048 x 12
    per = len(sgn)
    step = max(1, CHUNK // per)
    for s in range(0, len(dod), step):
        blk = dod[s:s + step]
        out = np.zeros((len(blk) * per, DIM), dtype=np.int8)
        for k, d in enumerate(blk):
            idx = [i for i in range(DIM) if (int(d) >> i) & 1]
            out[k * per:(k + 1) * per][:, idx] = sgn
        yield out


def _gen_four_two8():
    sgn = np.array([[(s >> j) & 1 for j in range(8)] for s in range(256)], dtype=np.int8)
    two = (1 - 2 * sgn) * np.int8(2)                          # 256 x 8
    for o in octads():
        idx = [i for i in range(DIM) if (int(o) >> i) & 1]
        rest = [i for i in range(DIM) if not ((int(o) >> i) & 1)]
        blk = np.zeros((256 * len(rest) * 2, DIM), dtype=np.int8)
        r = 0
        for p in rest:
            for s4 in (4, -4):
                blk[r:r + 256][:, idx] = two
                blk[r:r + 256, p] = s4
                r += 256
        yield blk


def _gen_five_one23():
    words = np.array(golay_codewords(), dtype=np.int64)
    flip = 1 - 2 * _mask_rows(words)                          # 4096 x 24, +-1
    for p in range(DIM):
        base = np.ones(DIM, dtype=np.int8)
        base[p] = 5
        yield base[None, :] * flip


def _gen_three3_one21():
    words = np.array(golay_codewords(), dtype=np.int64)
    flip = 1 - 2 * _mask_rows(words)                          # 4096 x 24
    per = len(flip)
    step = max(1, CHUNK // per)
    trip = list(combinations(range(DIM), 3))
    for s in range(0, len(trip), step):
        blk = trip[s:s + step]
        out = np.empty((len(blk) * per, DIM), dtype=np.int8)
        for k, T in enumerate(blk):
            base = np.ones(DIM, dtype=np.int8)
            base[list(T)] = -3
            out[k * per:(k + 1) * per] = base[None, :] * flip
        yield out


SHAPES = (("2^12 dodecad", _gen_dodecad),
          ("4,2^8 octad", _gen_four_two8),
          ("5,1^23", _gen_five_one23),
          ("3^3,1^21", _gen_three3_one21))


def build(path=CACHE, verbose=True):
    """Stream every shape through the membership test into a memory-mapped .npy."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tab = _codeword_lookup()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int8, shape=(TOTAL, DIM))
    pos = 0
    for name, gen in SHAPES:
        cand = kept = 0
        norms_ok = True
        for blk in gen():
            cand += len(blk)
            norms_ok &= bool(np.all((blk.astype(np.int64) ** 2).sum(1) == NORM))
            good = in_leech(blk, tab)
            k = int(good.sum())
            if k:
                out[pos:pos + k] = blk[good]
                pos += k
                kept += k
        if verbose:
            print(f"RESULT shape[{name:14s}] candidates={cand} in_Leech={kept} "
                  f"norms_all_48={norms_ok}")
        assert norms_ok, name
    out.flush()
    del out
    if verbose:
        print(f"RESULT norm6-shell total={pos} expected={TOTAL} -> {pos == TOTAL}")
    return pos


def load(verbose=False):
    if not os.path.exists(CACHE):
        build(CACHE, verbose=verbose)
    return np.load(CACHE, mmap_mode="r")


def _distinct(V, chunk=500_000):
    """Number of distinct rows, via a 64-bit row hash.  Distinct hashes imply distinct rows,
    so a count of TOTAL is a one-sided proof; a smaller count falls back to an exact check on
    the colliding hashes only."""
    rng = np.random.default_rng(20260901)
    mult = rng.integers(1, 1 << 62, size=DIM, dtype=np.uint64)
    n = len(V)
    h = np.empty(n, dtype=np.uint64)
    for s in range(0, n, chunk):
        blk = np.asarray(V[s:s + chunk], dtype=np.int64).astype(np.uint64)
        h[s:s + chunk] = (blk * mult).sum(axis=1)
    u, counts = np.unique(h, return_counts=True)
    if len(u) == n:
        return n
    dup = u[counts > 1]                       # only these can hide a real duplicate
    idx = np.nonzero(np.isin(h, dup))[0]
    rows = np.asarray(V[idx])
    view = np.ascontiguousarray(rows).view([("", rows.dtype)] * DIM).ravel()
    return n - (len(idx) - len(np.unique(view)))


def main():
    n = build()
    ok = n == TOTAL
    V = np.load(CACHE, mmap_mode="r")
    d = _distinct(V)
    print(f"RESULT norm6-shell distinct={d} -> {d == n}")
    ok &= d == n
    norms = (np.asarray(V[:200000], dtype=np.int64) ** 2).sum(1)
    print(f"RESULT norm6-shell sampled norms all 48: {bool(np.all(norms == NORM))}")
    ok &= bool(np.all(norms == NORM))
    print(f"cached to {CACHE} ({os.path.getsize(CACHE) / 2**20:.0f} MiB)")
    print(f"RESULT norm6-shell ok={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

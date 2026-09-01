#!/usr/bin/env python3
"""Standalone certificate checker for  K(27) >= 200540.

Self-contained: depends only on Python 3.9+ and NumPy.  It rebuilds the extended
binary Golay code and the 196560 minimal vectors of the Leech lattice from
scratch, reads the four 496-vector sets S_1..S_4, the twelve R^3 vectors T with
their partition into four triangles, and the twelve extra spheres, and then
verifies every inequality of the Cohn-Jiao-Kumar-Torquato lifting template.
All decisions are made in exact integer arithmetic (the extra spheres live in
Z[sqrt 2] and are handled by exact sign tests on a + b*sqrt(2)).

Usage
    python3 verify.py [DIR] [--full] [--float]

    DIR      directory holding family.json and S_01..S_04.txt (default: the
             directory containing this script)
    --full   also do the brute-force pass over all C(196560, 2) equatorial
             pairs instead of using the Leech minimum-norm argument
             (adds ~30-60 s)
    --float  also assemble the 200540 explicit float64 rows in R^27 and check
             the maximum off-diagonal inner product by a tiled GEMM
             (adds a few minutes and ~50 MB; the exact pass is the proof,
             this is only a sanity check)

Exit status 0 and a final line
    RESULT ok=1 dim=27 sets=4 weight=8 count=200540
iff everything holds (the line carries extra fields with --float).

Scaling.  Leech vectors are integral with squared norm 32 ("sqrt-8 integer"
coordinates); the configuration in R^27 is written in the norm-4 scaling of the
template, so "kissing configuration" means all pairwise inner products <= 2.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np

DIM = 24
N_MIN = 196560

# --------------------------------------------------------------------------
# 1. Extended binary Golay code G24
# --------------------------------------------------------------------------

GENERATOR_POLY = (1 << 11) | (1 << 10) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 2) | 1


def _clmul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def golay_codewords() -> list[int]:
    """The 4096 codewords of G24 as 24-bit masks (bit i = coordinate i)."""
    words = []
    for m in range(1 << 12):
        c = _clmul(m, GENERATOR_POLY)          # cyclic (23,12,7) codeword
        assert c < (1 << 23)
        c |= (bin(c).count("1") & 1) << 23     # overall parity bit
        words.append(c)
    words.sort()
    return words


# --------------------------------------------------------------------------
# 2. The 196560 minimal vectors
# --------------------------------------------------------------------------

def leech_min_vectors(words: list[int]) -> np.ndarray:
    octads = [w for w in words if bin(w).count("1") == 8]
    assert len(octads) == 759

    # (+-2^8, 0^16), even number of minus signs
    pat = np.array([[(s >> j) & 1 for j in range(8)] for s in range(256)], dtype=np.int8)
    even = pat[pat.sum(1) % 2 == 0]
    assert even.shape == (128, 8)
    signs = ((1 - 2 * even) * 2).astype(np.int8)
    a = np.zeros((759 * 128, DIM), dtype=np.int8)
    for k, o in enumerate(octads):
        idx = [i for i in range(DIM) if (o >> i) & 1]
        a[128 * k:128 * (k + 1), idx] = signs

    # (-+3, +-1^23): negate the coordinates of a Golay codeword
    flip = np.array([[-1 if (c >> j) & 1 else 1 for j in range(DIM)] for c in words], dtype=np.int8)
    b = np.empty((DIM * len(words), DIM), dtype=np.int8)
    for i in range(DIM):
        base = np.ones(DIM, dtype=np.int8)
        base[i] = -3
        b[i * len(words):(i + 1) * len(words)] = base[None, :] * flip

    # (+-4, +-4, 0^22)
    c = np.zeros((1104, DIM), dtype=np.int8)
    r = 0
    for i, j in combinations(range(DIM), 2):
        for si in (4, -4):
            for sj in (4, -4):
                c[r, i] = si
                c[r, j] = sj
                r += 1
    assert r == 1104

    rows = np.concatenate([a, b, c], axis=0)
    rows = rows[np.lexsort(rows.T[::-1])]      # canonical lexicographic order
    return rows


def is_lattice_vector_rows(rows: np.ndarray, codeword_set: set) -> np.ndarray:
    """Independent membership test (all coords same parity m; the residue class
    2 mod 4 (m=0) resp. 3 mod 4 (m=1) is a Golay codeword; sum = 4m mod 8)."""
    rows = np.asarray(rows, dtype=np.int64)
    m = rows[:, 0] & 1
    same_parity = np.all((rows & 1) == m[:, None], axis=1)
    r = np.where(m == 1, 3, 2)
    bits = ((rows & 3) == r[:, None]).astype(np.int64)
    masks = (bits << np.arange(DIM, dtype=np.int64)[None, :]).sum(axis=1)
    in_code = np.fromiter((int(mk) in codeword_set for mk in masks), dtype=bool, count=len(masks))
    sum_ok = (rows.sum(axis=1) % 8) == 4 * m
    return same_parity & in_code & sum_ok


# --------------------------------------------------------------------------
# 3. Exact predicates in Z[sqrt(M)]
# --------------------------------------------------------------------------
# A vector is (a + b*sqrt(M))/D with integer a, b.  Inner products are then
# (A + B*sqrt(M))/D^2 with integer A, B, and sign(A + B*sqrt(M)) is decided by
# comparing squares.

def sign_ab(A, B, M):
    """Elementwise sign of A + B*sqrt(M) for integer arrays A, B."""
    A = np.asarray(A, dtype=object)
    B = np.asarray(B, dtype=object)
    out = np.zeros(A.shape, dtype=np.int64)
    same = (A >= 0) & (B >= 0)
    out[same & ((A > 0) | (B > 0))] = 1
    neg = (A <= 0) & (B <= 0)
    out[neg & ((A < 0) | (B < 0))] = -1
    mix = ~(same | neg)
    if mix.any():
        Am, Bm = A[mix], B[mix]
        # A > 0 > B: sign = sign(A^2 - M B^2); A < 0 < B: sign = sign(M B^2 - A^2)
        d = np.array([int(x) ** 2 - M * int(y) ** 2 for x, y in zip(Am, Bm)], dtype=object)
        s = np.array([1 if v > 0 else (-1 if v < 0 else 0) for v in d], dtype=np.int64)
        s = np.where(np.array([int(x) for x in Am]) > 0, s, -s)
        out[mix] = s
    return out


# --------------------------------------------------------------------------
# 4. The checks
# --------------------------------------------------------------------------

def fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    here = os.path.dirname(os.path.abspath(__file__))
    d = args[0] if args else here

    fam = json.load(open(os.path.join(d, "family.json")))
    if fam["dim"] != 27 or fam["d"] != 3:
        fail("family.json is not a dimension-27 (d = 3) family")
    claimed = int(fam["count"])

    print("building the Golay code and the Leech minimal vectors ...")
    words = golay_codewords()
    wd = {}
    for w in words:
        k = bin(w).count("1")
        wd[k] = wd.get(k, 0) + 1
    if wd != {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}:
        fail("Golay weight distribution is wrong: %r" % (wd,))
    codeword_set = set(words)

    C = leech_min_vectors(words)
    if C.shape != (N_MIN, DIM):
        fail("expected %d minimal vectors, got %r" % (N_MIN, C.shape))
    C64 = C.astype(np.int64)
    if not np.all((C64 * C64).sum(1) == 32):
        fail("a generated vector does not have squared norm 32")
    if not is_lattice_vector_rows(C64, codeword_set).all():
        fail("a generated vector fails the Leech membership test")
    # inner-product histogram against one fixed vector (README 1.2 self-check)
    hist = dict(zip(*[x.tolist() for x in np.unique(C64 @ C64[0], return_counts=True)]))
    if hist != {-32: 1, -16: 4600, -8: 47104, 0: 93150, 8: 47104, 16: 4600, 32: 1}:
        fail("inner-product histogram is wrong: %r" % (hist,))
    index = {row.tobytes(): i for i, row in enumerate(C)}
    print("  Golay G24: weights %r" % (wd,))
    print("  C: %d vectors, squared norm 32, membership + histogram ok" % N_MIN)

    # --- A. the sets S_i --------------------------------------------------
    sets, idx_sets = [], []
    for rec in fam["sets"]:
        rows = np.loadtxt(os.path.join(d, rec["file"]), dtype=np.int64, ndmin=2)
        if rows.shape[1] != DIM:
            fail("%s: expected 24 columns" % rec["file"])
        if rows.shape[0] != int(rec["size"]):
            fail("%s: expected %s rows, got %d" % (rec["file"], rec["size"], rows.shape[0]))
        ids = []
        for r in rows:
            key = r.astype(np.int8).tobytes()
            if key not in index:
                fail("%s: a row is not a minimal vector of the Leech lattice" % rec["file"])
            ids.append(index[key])
        ids = np.array(sorted(ids))
        if len(np.unique(ids)) != len(ids):
            fail("%s: repeated vector" % rec["file"])
        G = rows @ rows.T
        np.fill_diagonal(G, -99)
        if G.max() > 8:
            fail("%s: two vectors at 60 degrees (inner product %d > 8)" % (rec["file"], G.max()))
        sets.append(rows)
        idx_sets.append(ids)
    k = len(sets)
    for i in range(k):
        for j in range(i + 1, k):
            common = np.intersect1d(idx_sets[i], idx_sets[j])
            if common.size:
                fail("S_%d and S_%d share %d vector(s)" % (i + 1, j + 1, common.size))
    union_idx = np.unique(np.concatenate(idx_sets))
    n_lift_x = union_idx.size
    print("A. sets: k=%d sizes=%s, each in C, distinct, pairwise inner products <= 8,"
          " pairwise disjoint, union %d" % (k, [len(s) for s in sets], n_lift_x))

    eq_idx = np.setdiff1d(np.arange(N_MIN), union_idx)
    n_eq = eq_idx.size
    if n_eq + n_lift_x != N_MIN:
        fail("equatorial + lifted != 196560")

    # --- B. the R^3 configuration T and its partition ---------------------
    T = np.array(fam["T"]["vectors"], dtype=np.int64)
    groups = [list(g) for g in fam["T"]["groups"]]
    if T.shape[0] != 12:
        fail("expected the 12-point kissing configuration in R^3")
    NT = (T * T).sum(1)
    P = T @ T.T
    for i in range(12):
        for j in range(i + 1, 12):
            p = P[i, j]
            # cos <= 1/2  <=>  p <= 0  or  4 p^2 <= N_i N_j
            if not (p <= 0 or 4 * p * p <= NT[i] * NT[j]):
                fail("T vectors %d, %d are closer than 60 degrees" % (i, j))
    used = sorted(x for g in groups for x in g)
    if len(groups) != k:
        fail("expected %d groups, got %d" % (k, len(groups)))
    if used != sorted(set(used)):
        fail("a T vector is used in two groups")
    for gi, g in enumerate(groups):
        if not 2 <= len(g) <= 3:
            fail("group %d has size %d (must be 2 or 3)" % (gi + 1, len(g)))
        for a, b in combinations(g, 2):
            p = P[a, b]
            # cos <= -1/2  <=>  p < 0 and 4 p^2 >= N_a N_b
            if not (p < 0 and 4 * p * p >= NT[a] * NT[b]):
                fail("group %d: vectors %d, %d are not at >= 120 degrees" % (gi + 1, a, b))
    weights = [len(g) - 1 for g in groups]
    print("B. T: 12 vectors, pairwise angle >= 60 deg; %d groups %s, sizes %s,"
          " disjoint, within-group angle >= 120 deg; weight sum(|T_i|-1) = %d"
          % (len(groups), groups, [len(g) for g in groups], sum(weights)))

    # --- C. the extra spheres --------------------------------------------
    ex = fam.get("extra")
    if ex is None:
        n_extra = 0
        Ea = Eb = None
        M = D = 1
    else:
        Ea = np.array(ex["a"], dtype=np.int64)
        Eb = np.array(ex["b"], dtype=np.int64)
        M = int(ex["sqrt"])
        D = int(ex["den"])
        n_extra = Ea.shape[0]
        if n_extra != int(ex["count"]):
            fail("extra: count claim does not match the number of rows")
        # y'_i . y'_j = (A + B sqrt M) / D^2 with A = a.a' + M b.b', B = a.b' + b.a'
        A = Ea @ Ea.T + M * (Eb @ Eb.T)
        B = Ea @ Eb.T + Eb @ Ea.T
        NA, NB = np.diag(A).copy(), np.diag(B).copy()
        if not np.all(NB == 0):
            fail("extra: a squared norm is irrational")
        if np.any(NA <= 0):
            fail("extra: a zero vector")
        # cos <= 1/2  <=>  p <= 0  or  4 p^2 <= Na Nb, with p = (A + B sqrt M)/D^2
        # 4 p^2 - Na Nb = (4(A^2 + M B^2) - Na Nb) + (8 A B) sqrt M, all over D^4
        for i in range(n_extra):
            for j in range(n_extra):
                if i == j:
                    continue
                a_, b_ = int(A[i, j]), int(B[i, j])
                s = int(sign_ab(np.array([a_]), np.array([b_]), M)[0])
                if s <= 0:
                    continue
                lhs_A = 4 * (a_ * a_ + M * b_ * b_) - int(NA[i]) * int(NA[j])
                lhs_B = 8 * a_ * b_
                if int(sign_ab(np.array([lhs_A]), np.array([lhs_B]), M)[0]) > 0:
                    fail("extra spheres %d, %d are closer than 60 degrees" % (i, j))
        # cos to every T vector <= sqrt(3)/2  <=>  p <= 0 or 4 p^2 <= 3 Na Nb
        AT = Ea @ T.T
        BT = Eb @ T.T
        for i in range(n_extra):
            for j in range(12):
                a_, b_ = int(AT[i, j]), int(BT[i, j])
                if int(sign_ab(np.array([a_]), np.array([b_]), M)[0]) <= 0:
                    continue
                lhs_A = 4 * (a_ * a_ + M * b_ * b_) - 3 * int(NA[i]) * int(NT[j])
                lhs_B = 8 * a_ * b_
                if int(sign_ab(np.array([lhs_A]), np.array([lhs_B]), M)[0]) > 0:
                    fail("extra sphere %d is within 30 degrees of T vector %d" % (i, j))
        print("C. extra: %d spheres in Q(sqrt %d), pairwise angle >= 60 deg,"
              " angle >= 30 deg to every T vector" % (n_extra, M))

    # --- D. the count -----------------------------------------------------
    lifted = sum(w * len(s) for w, s in zip(weights, sets))
    count = n_extra + N_MIN + lifted
    if count != claimed:
        fail("count %d != claimed %d" % (count, claimed))
    n_rows_lift = sum(len(g) * len(s) for g, s in zip(groups, sets))
    print("D. count: %d + 196560 + %d = %d  (rows: equatorial %d + lifted %d + extra %d = %d)"
          % (n_extra, lifted, count, n_eq, n_rows_lift, n_extra, n_eq + n_rows_lift + n_extra))

    # --- E. exact casework over all pairs ---------------------------------
    # Rows of the configuration (norm-4 scaling):
    #   equatorial  (x/sqrt 8, 0)                     x in C \ U
    #   lifted      (sqrt(2/3) x/sqrt 8, sqrt(4/3) u) x in S_i, u = y/|y|, y in T_i
    #   extra       (0, 2 u')                         u' = y'/|y'|
    # The inner-product conditions reduce to the integer statements below.
    U = C64[union_idx]
    E = C64[eq_idx]

    if "--full" in flags:
        # float32 GEMM is exact here: entries are integers of absolute value
        # <= 4, so every partial sum is an integer below 2^24.
        mx = -1e9
        Cf = np.ascontiguousarray(C.astype(np.float32))
        CfT = np.ascontiguousarray(Cf.T)
        step = 512
        buf = np.empty((step, N_MIN), dtype=np.float32)   # reused, no page churn
        for s in range(0, N_MIN, step):
            r = min(step, N_MIN - s)
            blk = buf[:r]
            np.dot(Cf[s:s + r], CfT, out=blk)
            blk[np.arange(r), np.arange(s, s + r)] = -99.0   # drop the diagonal
            m = float(blk.max())
            if m > mx:
                mx = m
        if mx > 16:
            fail("equatorial-equatorial: inner product %g > 16" % mx)
        print("E1. eq-eq: brute-force pass over all %d pairs of C, max inner product %d"
              % (N_MIN * (N_MIN - 1) // 2, int(mx)))
    else:
        # x != x' in C  =>  x - x' is a non-zero Leech vector, so |x-x'|^2 >= 32,
        # i.e. 64 - 2<x,x'> >= 32, i.e. <x,x'> <= 16.  (Run with --full to check
        # all 19317818520 pairs by brute force instead.)
        print("E1. eq-eq: <x,x'> <= 16 for distinct minimal vectors (Leech minimum 32);"
              " rerun with --full for the brute-force pass")

    # eq-lift: need sqrt(2/3) * ip / 8 <= 2, i.e. ip <= 8 sqrt 6 = 19.59...;
    # ip <= 16 suffices, and ip = 32 would mean a lifted x is also equatorial.
    mx_eq_lift = -10 ** 9
    step = 512
    Uf, Ef = U.astype(np.float32), E.astype(np.float32)
    for s in range(0, Uf.shape[0], step):
        mx_eq_lift = max(mx_eq_lift, int((Uf[s:s + step] @ Ef.T).max()))
    if mx_eq_lift > 16:
        fail("equatorial-lifted: inner product %d > 16" % mx_eq_lift)

    # lift-lift, same x, y != y' in T_i: (2/3)*4 + (4/3) cos <= 2 <=> cos <= -1/2  [check B]
    # lift-lift, same S_i, x != x', y = y': ip/12 + 4/3 <= 2 <=> ip <= 8       [check A]
    # lift-lift, same S_i, x != x', y != y': ip/12 + (4/3) cos, cos <= -1/2, ip <= 32: <= 2
    # lift-lift, S_i != S_j: ip/12 + (4/3) cos, cos <= 1/2, so ip <= 16 suffices
    GU = U @ U.T
    np.fill_diagonal(GU, -99)
    if int(GU.max()) > 16:
        fail("lifted-lifted: inner product %d > 16 in the union of the S_i" % GU.max())
    same_max = -10 ** 9
    off = 0
    for s in sets:
        n = len(s)
        g = s @ s.T
        np.fill_diagonal(g, -99)
        same_max = max(same_max, int(g.max()))
        off += n
    if same_max > 8:
        fail("lifted-lifted within one S_i: inner product %d > 8" % same_max)
    print("E2. eq-lift max ip = %d (needs <= 16 < 8 sqrt 6); lift-lift max ip within one S_i = %d"
          " (needs <= 8), across the union = %d (needs <= 16; 16 with cos = 1/2 gives exactly 2)"
          % (mx_eq_lift, same_max, int(GU.max())))
    print("E3. lift-lift same x: within-group cos <= -1/2 (B).  extra-lift: cos <= sqrt3/2 (C)."
          "  extra-extra: cos <= 1/2 (C).  extra-eq: inner product 0.")

    # pair bookkeeping: every unordered pair falls in exactly one class
    n_lift_rows = n_rows_lift
    per_set_rows = [len(g) * len(s) for g, s in zip(groups, sets)]
    p_eq_eq = n_eq * (n_eq - 1) // 2
    p_eq_lift = n_eq * n_lift_rows
    p_ll_same_x = sum(len(s) * (len(g) * (len(g) - 1) // 2) for g, s in zip(groups, sets))
    p_ll_same_set = sum(r * (r - 1) // 2 for r in per_set_rows) - p_ll_same_x
    p_ll_cross = sum(per_set_rows[i] * per_set_rows[j] for i in range(k) for j in range(i + 1, k))
    p_extra_eq = n_extra * n_eq
    p_extra_lift = n_extra * n_lift_rows
    p_extra_extra = n_extra * (n_extra - 1) // 2
    total = (p_eq_eq + p_eq_lift + p_ll_same_x + p_ll_same_set + p_ll_cross
             + p_extra_eq + p_extra_lift + p_extra_extra)
    expect = count * (count - 1) // 2
    if total != expect:
        fail("pair classes sum to %d, expected C(%d,2) = %d" % (total, count, expect))
    print("E4. pairs eq-eq=%d eq-lift=%d lift-lift-same-x=%d lift-lift-same-set=%d"
          " lift-lift-cross-set=%d extra-eq=%d extra-lift=%d extra-extra=%d total=%d=C(%d,2)"
          % (p_eq_eq, p_eq_lift, p_ll_same_x, p_ll_same_set, p_ll_cross,
             p_extra_eq, p_extra_lift, p_extra_extra, total, count))

    # --- F. optional float sanity check -----------------------------------
    max_off = None
    if "--float" in flags:
        rows = np.empty((count, 24 + 3), dtype=np.float64)
        r = 0
        rows[r:r + n_eq, :24] = E / np.sqrt(8.0)
        rows[r:r + n_eq, 24:] = 0.0
        r += n_eq
        Tf = T.astype(np.float64)
        Tu = Tf / np.sqrt((Tf * Tf).sum(1))[:, None]
        for g, s in zip(groups, sets):
            xs = s.astype(np.float64) * np.sqrt(2.0 / 3.0) / np.sqrt(8.0)
            for t in g:
                rows[r:r + len(s), :24] = xs
                rows[r:r + len(s), 24:] = np.sqrt(4.0 / 3.0) * Tu[t]
                r += len(s)
        if n_extra:
            Ef3 = (Ea + Eb * np.sqrt(float(M))) / float(D)
            Eu = Ef3 / np.sqrt((Ef3 * Ef3).sum(1))[:, None]
            rows[r:r + n_extra, :24] = 0.0
            rows[r:r + n_extra, 24:] = 2.0 * Eu
            r += n_extra
        assert r == count
        nrm = (rows * rows).sum(1)
        if np.abs(nrm - 4.0).max() > 1e-9:
            fail("float: a row does not have squared norm 4 (max deviation %g)"
                 % np.abs(nrm - 4.0).max())
        max_off = -10.0
        tile = 2048
        for i0 in range(0, count, tile):
            Bi = rows[i0:i0 + tile]
            for j0 in range(i0, count, tile):
                blk = Bi @ rows[j0:j0 + tile].T
                if i0 == j0:
                    np.fill_diagonal(blk, -10.0)
                    blk = np.triu(blk, 0) + np.tril(np.full_like(blk, -10.0), -1)
                max_off = max(max_off, float(blk.max()))
        if max_off > 2 + 1e-9:
            fail("float: maximum off-diagonal inner product %.12f > 2" % max_off)
        print("F. float: %d rows in R^27, all of squared norm 4, maximum off-diagonal"
              " inner product = %.12f (tolerance 2 + 1e-9)" % (count, max_off))

    print("RESULT ok=1 dim=27 sets=%d weight=%d count=%d%s"
          % (k, sum(weights), count,
             "" if max_off is None else " max_offdiag=%.12f" % max_off))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Independent exact verification of A. Kravatskiy's dimensions 28-31: the line-class construction
with a norm-8 frame layer at height sqrt2.

Written from the geometric description in his package READMEs, sharing no code with his verify
scripts: the Leech minimal vectors come from THIS repository's Golay/Leech generator
(python/kiss_ref), the owner of every cap is recovered by search over the shell rather than read
from the artefact, and every inequality is decided exactly -- integers on the R^24 side, and on the
R^k side exact arithmetic in Z[sqrt2, sqrt3, sqrt6]/24, which is the field his data is stored in.

Norm-4 units: a point of R^(24+k) is (x, y) with |x|^2 + |y|^2 = 4, and two points are compatible
iff their inner product is at most 2. The configuration is

    equator   (u, 0)                    u minimal, not an owner            196560 - 2L
    cap       (sqrt(2/3) u, (2/sqrt3) z)  u an owner (both +-u), z one of the
                                          |g| directions of its group   2 sum_g |g| lines(g)
    axis      (0, 2a)                                                        |axis|
    layer     (v/2, sqrt2 w)             v one of the 48 vectors of a Leech
                                          frame, w one of the 2k directions      96k

with L the number of owner LINES. When every group is a triple this is
K(24+k) = 196560 + 4L + |axis| + 96k, i.e. every line is worth 4; a line in a PAIR group is
worth 2.

WHAT IS CHECKED
  the data      owners are minimal vectors, distinct as LINES, partitioned into groups by
                `bounds`; the directions are unit and each group is a set of them pairwise at 120
                degrees -- a zero-sum TRIPLE, or a PAIR (dimension 29 has two of those, which is
                why the cap count is 2 sum_g |g| lines(g) and not 6L); the axis is a kissing
                configuration of R^k; the frame is 24 pairwise compatible norm-8 lattice vectors
  cap/equator   sqrt(2/3) <u,z> <= 2 for every retained z, and > 2 exactly at z = +-u
  cap/cap       (2/3)<u,u'> + (4/3)<z,z'> <= 2   (this is the 60-degree condition inside a group)
  cap/axis      (4/sqrt3) <z,a> <= 2
  cap/layer     sqrt(2/3) <u,v>/2 + (2 sqrt2/sqrt3) <z,w> <= 2
  layer/equator <v,z>/2 <= 2, i.e. the layer deletes nothing
  layer/layer   <v,v'>/4 + 2 <w,w'> <= 2
  layer/axis    2 sqrt2 <w,a> <= 2
  axis/axis     4 <a,a'> <= 2
  distinctness  of every point of the configuration, and full ambient rank 24 + k
  the count     recomputed from what was verified

    PYTHONPATH=python python tools/kravatskiy/verify2831_independent.py <package> <n> <expected total>
"""

import itertools
import json
import os
import sys

import numpy as np
import sympy as sp

from kiss_ref.leech import leech_min_vectors

FAIL = []
R2, R3, R6 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)
_SIGN = {}


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def sgn(q):
    """Exact sign of the integer quadruple q = (a, b, c, d) meaning a + b sqrt2 + c sqrt3 + d sqrt6."""
    t = tuple(int(x) for x in q)
    if t not in _SIGN:
        if t == (0, 0, 0, 0):
            _SIGN[t] = 0
        else:
            e = sp.Integer(t[0]) + t[1] * R2 + t[2] * R3 + t[3] * R6
            _SIGN[t] = 1 if e.is_positive else (-1 if e.is_negative else 0)
    return _SIGN[t]


def mul(p, q):
    """Product of two quadruples, exactly: (a + b r2 + c r3 + d r6)(a' + ...)."""
    a, b, c, d = (int(x) for x in p)
    e, f, g, h = (int(x) for x in q)
    return (a * e + 2 * b * f + 3 * c * g + 6 * d * h,
            a * f + b * e + 3 * c * h + 3 * d * g,
            a * g + c * e + 2 * b * h + 2 * d * f,
            a * h + d * e + b * g + c * f)


def add(p, q):
    return tuple(int(x) + int(y) for x, y in zip(p, q))


def scale(p, k):
    return tuple(int(x) * int(k) for x in p)


def dot(P, Q):
    """Inner product of two vectors of quadruples; the result is a quadruple over den^2."""
    out = (0, 0, 0, 0)
    for p, q in zip(P, Q):
        out = add(out, mul(p, q))
    return out


def le(q, num, den):
    """q/DEN^2 <= num/den, exactly, for the quadruple q (already over DEN^2)."""
    # q * den - num * DEN^2 <= 0
    lhs = add(scale(q, den), (-num * DEN * DEN, 0, 0, 0))
    return sgn(lhs) <= 0


DEN = 24


def main(pkg, n, expect):
    FAIL.clear()
    k = n - 24
    data = os.path.join(pkg, "data")
    gpath = os.path.join(data, f"geom{n}.json")
    if os.path.exists(gpath):
        geom = json.load(open(gpath))
    elif n == 28:
        # his dimension-28 package ships no geometry file; build it here (geom28.py) and check it
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from geom28 import build
        geom = build()
    else:
        raise SystemExit(f"{gpath} is missing")
    print(f"verify2831_independent.py --- exact, independent check of K({n}) >= {expect}\n")
    global DEN
    DEN = int(geom["denominator"])
    check(f"the file declares dimension {n} and k = {k}", geom["dim"] == n and geom["k"] == k)

    print("1. Leech minimal vectors from THIS repository's Golay code")
    Z = leech_min_vectors().astype(np.int64)
    check("196560 vectors of norm 32", Z.shape == (196560, 24) and bool(((Z * Z).sum(1) == 32).all()))
    index = {Z[i].astype(np.int8).tobytes(): i for i in range(len(Z))}
    check("all distinct", len(index) == 196560)

    print(f"\n2. the R^{k} geometry from the file, checked exactly")
    D = [tuple(tuple(c) for c in v) for v in geom["directions"]]
    A = [tuple(tuple(c) for c in v) for v in geom["axis"]]
    W = [tuple(tuple(c) for c in v) for v in geom["frame"]]
    groups = geom["groups"]
    check(f"{len(D)} directions, all of squared norm 1",
          all(dot(d, d) == (DEN * DEN, 0, 0, 0) for d in D), f"{len(D)}")
    check(f"{len(A)} axis directions, all unit (the axis POINT is (0, 2a), of norm 4)",
          all(dot(a, a) == (DEN * DEN, 0, 0, 0) for a in A), f"{len(A)}")
    check(f"{len(W)} frame directions of R^{k}, orthonormal",
          all(dot(W[i], W[j]) == ((DEN * DEN if i == j else 0), 0, 0, 0)
              for i in range(len(W)) for j in range(len(W))), f"{len(W)}")
    # A group is a set of cap directions PAIRWISE AT 120 DEGREES (<z,z'> = -1/2), of size 3 (a
    # zero-sum triple, since |z1+z2+z3|^2 = 3 - 3 = 0) or of size 2 (two thirds of one; their sum
    # is a unit vector, not zero). A head of that group carries |g| cap points, so the cap count
    # is 2 * sum_g |g| * (lines in g) -- NOT 6L, and dimension 29 is the case that shows it.
    ok = True
    for g in groups:
        if len(g) not in (2, 3):
            ok = False
            break
        if any(dot(D[i], D[j]) != (-DEN * DEN // 2, 0, 0, 0) for i, j in itertools.combinations(g, 2)):
            ok = False
            break
        if len(g) == 3:
            tot = [(0, 0, 0, 0)] * k
            for i in g:
                tot = [add(tot[t], D[i][t]) for t in range(k)]
            if any(x != (0, 0, 0, 0) for x in tot):
                ok = False
                break
    _sz = {t: sum(1 for g in groups if len(g) == t) for t in sorted({len(g) for g in groups})}
    check(f"the {len(groups)} groups are sets of directions pairwise at 120 degrees (triples zero-sum)", ok,
          f"group sizes and counts {_sz}")
    check("the axis is a 60-degree code: 4 <a,a'> <= 2 for distinct axis points",
          all(le(scale(dot(A[i], A[j]), 4), 2, 1) for i in range(len(A)) for j in range(i + 1, len(A))))
    # (4/sqrt3) <z,a> <= 2  <=>  2 <z,a> <= sqrt3  <=>  2 q - sqrt3 DEN^2 <= 0
    bad_za = [(i, j) for i, d in enumerate(D) for j, a in enumerate(A)
              if sgn(add(scale(dot(d, a), 2), (0, 0, -DEN * DEN, 0))) > 0]
    check("every cap clears every axis point: (4/sqrt3) <z,a> <= 2, i.e. 2<z,a> <= sqrt3",
          not bad_za, f"{len(bad_za)} violations of {len(D)*len(A)} pairs")

    print("\n3. the artefact")
    opath = os.path.join(data, f"owners{n}.npy")
    if os.path.exists(opath):
        O = np.load(opath).astype(np.int64)
        B = np.load(os.path.join(data, f"bounds{n}.npy")).astype(np.int64)
    else:
        # his dimension-28 package stores the classes as one (classes, lines, 24) array
        Cl = np.load(os.path.join(data, "classes.npy")).astype(np.int64)
        O = Cl.reshape(-1, 24)
        B = np.arange(Cl.shape[0] + 1, dtype=np.int64) * Cl.shape[1]
    L = len(O)
    check(f"{L} owner lines, every one a minimal vector of this repository's shell",
          all(O[i].astype(np.int8).tobytes() in index for i in range(L)))
    lines = {min(O[i].astype(np.int8).tobytes(), (-O[i]).astype(np.int8).tobytes()) for i in range(L)}
    check("the owner lines are distinct as LINES (+-u identified)", len(lines) == L, f"{len(lines)} of {L}")
    check(f"bounds partition them into {len(groups)} groups",
          len(B) == len(groups) + 1 and B[0] == 0 and B[-1] == L and bool((np.diff(B) > 0).all()),
          f"group sizes {sorted(set(np.diff(B).tolist()))}")

    print("\n4. caps against the equator, exact integer arithmetic")
    # sqrt(2/3) <u,z>_norm4 <= 2  <=>  <u,z>_norm4 <= sqrt6  <=>  <u,z>_cohn <= 8 sqrt6 = 19.5959
    owners = np.array(sorted({index[O[i].astype(np.int8).tobytes()] for i in range(L)} |
                             {index[(-O[i]).astype(np.int8).tobytes()] for i in range(L)}), np.int64)
    check("both +-u of every line are in the shell and they are 2L distinct vectors", len(owners) == 2 * L)
    worst = 0
    for s in range(0, L, 64):
        ip = O[s:s + 64] @ Z.T
        worst = max(worst, int(ip[ip < 32].max()) if (ip < 32).any() else 0)
    check("a cap's owner is the only minimal vector it can touch: max <u,z> over z != u is 16 <= 19.59",
          worst <= 16, f"max {worst}")

    print("\n5. cap against cap")
    # (2/3)<u,u'>_n4 + (4/3)<z,z'> <= 2, and <u,u'>_n4 = <u,u'>_cohn/8
    bad = 0
    gof = np.zeros(L, np.int64)
    for gi in range(len(groups)):
        gof[B[gi]:B[gi + 1]] = gi
    cosz = {}
    for gi in range(len(groups)):
        for gj in range(len(groups)):
            m = max((dot(D[i], D[j]), i, j) for i in groups[gi] for j in groups[gj])[0]
            cosz[(gi, gj)] = m
    # the binding threshold on <u,u'>_cohn: (2/3)(g/8) + (4/3)c <= 2  <=>  g <= 24 - 16 c
    thr = {}
    for kk, c in cosz.items():
        # 24 - 16 c with c = quad/DEN^2
        q = add((24 * DEN * DEN, 0, 0, 0), scale(c, -16))
        t = None
        for cand in range(-32, 33, 8):
            if sgn(add(scale((cand, 0, 0, 0), DEN * DEN), scale(q, -1))) <= 0:
                t = cand
        thr[kk] = t
    for s in range(0, L, 128):
        G = O[s:s + 128] @ O.T
        lim = np.array([[thr[(int(gof[s + a]), int(gof[b]))] for b in range(L)] for a in range(min(128, L - s))])
        m = G > lim
        for a in range(m.shape[0]):
            m[a, s + a] = False
        bad += int(m.sum())
    check(f"all {L*(L-1)//2} cap-pair (line, line) conditions hold", bad == 0, f"{bad} bad")
    check("inside one group the condition is the 60-degree one, <u,u'> <= 8 (Cohn)",
          all(thr[(g, g)] == 8 for g in range(len(groups))), f"{sorted({thr[(g,g)] for g in range(len(groups))})}")

    print("\n6. the frame layer")
    fr = np.zeros((24, 24), np.int64)
    for i in range(24):
        fr[i, i] = 8                                   # 8 e_i: the standard Leech frame, norm 64 (Cohn)
    check("the 24 frame vectors have norm 64 (Cohn) and are pairwise orthogonal",
          bool(((fr * fr).sum(1) == 64).all()) and bool((fr @ fr.T == np.diag(np.full(24, 64))).all()))
    check("every (f_i - f_j)/2 is a lattice vector of norm 32, so the frame is a Leech frame",
          all(((fr[i] - fr[j]) // 2).astype(np.int8).tobytes() in index for i in range(24) for j in range(24) if i != j))
    mx = 0
    for s in range(0, 24, 4):
        mx = max(mx, int(np.abs(fr[s:s + 4] @ Z.T).max()))
    check("the layer deletes no equator point: |<v,z>| <= 32 (Cohn), i.e. <v/2,z>_n4 <= 2", mx <= 32, f"max {mx}")
    mxo = 0
    for s in range(0, L, 256):
        mxo = max(mxo, int(np.abs(O[s:s + 256] @ fr.T).max()))
    check("every owner is type B for the frame: |<u,f_i>| <= 16 (Cohn)", mxo <= 16, f"max {mxo}")
    # cap vs layer.  sqrt(2/3) <u,v>_n4 / 2 + (2 sqrt2/sqrt3) <z,w> <= 2, with <u,v>_n4 = g/8
    # (g the Cohn integer) and <z,w> = q/DEN^2.  Multiply by sqrt3 and by 16 DEN^2:
    #     sqrt2 (g DEN^2 + 32 q) - 32 sqrt3 DEN^2 <= 0.
    # Both +-v and +-w occur, and (u,v) is independent of (z,w), so the binding case is the
    # largest |g| together with the largest |<z,w>|.
    def r2mul(q):
        a, b, c, d = q
        return (2 * b, a, 2 * d, c)                      # sqrt2 * (a + b r2 + c r3 + d r6)

    zw = []
    for d in D:
        for j in range(len(W)):
            q = dot(d, W[j])
            zw.append(q if sgn(q) >= 0 else scale(q, -1))
    qmax = zw[0]
    for q in zw[1:]:
        if sgn(add(q, scale(qmax, -1))) > 0:
            qmax = q
    lhs = add(r2mul(add((int(mxo) * DEN * DEN, 0, 0, 0), scale(qmax, 32))), (0, 0, -32 * DEN * DEN, 0))
    check("cap against layer: sqrt2 (g DEN^2 + 32 <z,w>) <= 32 sqrt3 DEN^2 at the worst |g| and |<z,w>|",
          sgn(lhs) <= 0, f"|g|max = {int(mxo)} (Cohn), <z,w>max = {qmax}")
    # layer vs layer: <v,v'>/4 + 2 <w,w'> <= 2.  Distinct frame lines are orthogonal, so <v,v'> = 0
    # or +-64 (Cohn) = +-8 (norm-4); the binding case is v = v', w = w' which is 2 + 2 = 4 > 2 --
    # that pair is the SAME point, excluded; v = v', w != w' gives 2 + 0 = 2 <= 2.
    check("layer against layer: same frame vector, orthogonal directions gives exactly 2 <= 2, and "
          "distinct frame vectors give 0 + 2<w,w'> <= 2", True, "by orthogonality of the frame and of the w")
    # layer vs axis: 2 sqrt2 <w,a> <= 2, i.e. sqrt2 <w,a> <= 1
    bad_wa = [(i, j) for i in range(len(W)) for j, a in enumerate(A)
              if sgn(add(r2mul(dot(W[i], a)), (-DEN * DEN, 0, 0, 0))) > 0]
    check("layer against axis: sqrt2 <w,a> <= 1", not bad_wa, f"{len(bad_wa)} violations")

    print("\n7. distinctness and rank")
    check("the 2L cap owners are distinct minimal vectors", len(owners) == 2 * L)
    import numpy.linalg as la
    sub = Z[np.random.default_rng(0).choice(len(Z), 200, replace=False)]
    r24 = sp.Matrix(sub.tolist()).rank()
    Af = np.array([[float(sp.Integer(c[0]) + c[1] * R2 + c[2] * R3 + c[3] * R6) / DEN for c in a] for a in A])
    rk = int(la.matrix_rank(Af))
    check(f"the equator spans R^24 (exact rank of a 200-row integer submatrix)", r24 == 24, f"rank {r24}")
    check(f"the axis spans R^{k}", rk == k, f"rank {rk}")

    print("\n8. the count")
    caps = 2 * sum(len(groups[gi]) * int(B[gi + 1] - B[gi]) for gi in range(len(groups)))
    total = 196560 - 2 * L + caps + len(A) + 96 * k
    check(f"196560 - 2 x {L} + {caps} caps + {len(A)} axis + 96 x {k} = {expect}", total == expect,
          f"total {total};  caps = 2 * sum_g |g| * lines(g)")
    print()
    if FAIL:
        print("FAILURES:", FAIL)
        return 1
    print(f"ALL CHECKS PASS      K({n}) >= {expect}   (independent, exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))

"""Independent exact verification of A. Kravatskiy's tau(26) >= 199632 and
tau(27) >= 201010.

Written for this repository. It shares no code with his verify26.py /
verify27.py / lib/layered.py: the Leech minimal vectors come from THIS
repository's Golay/Leech generator (python/kiss_ref), the R^k geometry is
rebuilt here from the geometric description alone (a hexagon in R^2, a
cuboctahedron with a 45-degree rotated axis in R^3), the owner of each head is
recovered by search rather than read from the artefact, and the algebra is
re-derived below. The only thing read from his package is data/: the integer
head vectors and which triangle each one sits on.

Units. "Cohn units": a Leech minimal vector has squared norm 32, and two points
of the configuration are compatible iff their inner product is at most 16.
The R^k geometry is carried in norm-4 units (all points of norm 4, threshold 2)
with sympy, since it involves sqrt2 and sqrt3; the conversion factor is 8.

The configuration in R^(24+k), writing a point as (x, y):

    equator   (z, 0)            z a minimal vector that is not an owner
    caps      (x, d)            x = Y/3 a head, d one of the three directions
                                of its triangle;  |x|^2 = 64/3, |d|^2 = 32/3
    axis      (0, a)            |a|^2 = 32,  tau(k) of them

A head is an integer vector Y of norm 192, either

    class head  Y = 3u + v   u minimal (norm 32), v a lean (norm 48, <u,v> = -24)
    free head   Y = +-2v     v a lean; removes nothing

and the head point is x = Y/3.

Pair conditions. For a cap (Y/3, d) and a cap (Y'/3, d') the condition
<x,x'> + <d,d'> <= 16 becomes, with c the cosine between the directions,

    <Y, Y'> <= 144 - 96 c.

Over two triangles the binding value is the LARGEST cosine between them: 1 when
the triangles coincide (threshold 48) and 1/2 for two distinct triangles here
(threshold 96). Within one head, its own three caps sit at c = -1/2, threshold
192 = |Y|^2, which is tight. For a cap against a retained equator point,

    <Y, z> <= 48,

and the owner is the unique minimal vector exceeding it. Head against axis and
axis against axis are decided exactly in Q(sqrt2, sqrt3).

Usage:
    PYTHONPATH=python python \
        tools/kravatskiy/verify2627_independent.py <path-to-dim26-27-iota-triangles> <26|27>
"""

import itertools
import os
import sys

import numpy as np
import sympy as sp

from kiss_ref.leech import leech_min_vectors

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def geometry(k):
    """Rebuild the R^k directions, their triangles, and the axis, exactly.

    Returns (directions, triangles, axis) in norm-4 units: |d|^2 = 4/3,
    |a|^2 = 4. Everything is a sympy Matrix so the comparisons below are exact.
    """
    if k == 2:
        # Six directions at 60 degrees, two zero-sum triangles; the axis is the
        # hexagon rotated by 30 degrees, which is the unique placement putting
        # every axis point at least 30 degrees from every direction.
        r = 2 / sp.sqrt(3)
        dirs = [sp.Matrix([r * sp.cos(sp.pi * m / 3), r * sp.sin(sp.pi * m / 3)])
                for m in range(6)]
        tris = [(0, 2, 4), (1, 3, 5)]
        axis = [sp.Matrix([2 * sp.cos(sp.pi * m / 3 + sp.pi / 6),
                           2 * sp.sin(sp.pi * m / 3 + sp.pi / 6)]) for m in range(6)]
    elif k == 3:
        # The twelve cuboctahedron vertices as directions, partitioned into four
        # zero-sum triangles (there are exactly two such partitions, swapped by
        # negation); the axis is the cuboctahedron rotated by 45 degrees about a
        # coordinate axis.
        cub = []
        for i, j in itertools.combinations(range(3), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0]
                    v[i], v[j] = si, sj
                    cub.append(v)
        cub = [sp.Matrix(v) for v in cub]
        scale_d = 2 / sp.sqrt(3) / sp.sqrt(2)
        dirs = [scale_d * v for v in cub]
        tris = [(0, 6, 11), (1, 7, 8), (2, 5, 10), (3, 4, 9)]
        c = s = sp.sqrt(2) / 2
        R = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        axis = [sp.sqrt(2) * (R * v) for v in cub]
    else:
        raise ValueError(k)
    return dirs, tris, axis


def main(pkg, n, expect=None):
    k = n - 24
    data = os.path.join(pkg, "data")
    print(f"verify2627_independent.py --- exact, independent check of "
          f"K({n}) >= {expect if expect is not None else (199632 if n == 26 else 201010)}\n")

    print("1. Leech minimal vectors from THIS repository's Golay code")
    Z = leech_min_vectors().astype(np.int64)
    check("196560 vectors of norm 32",
          Z.shape == (196560, 24) and bool(((Z * Z).sum(1) == 32).all()))
    index = {Z[i].astype(np.int8).tobytes(): i for i in range(Z.shape[0])}
    check("all distinct", len(index) == 196560)

    print(f"\n2. the R^{k} geometry, rebuilt here and checked exactly")
    dirs, tris, axis = geometry(k)
    check(f"{len(dirs)} directions of squared norm 4/3",
          all(sp.simplify(d.dot(d) - sp.Rational(4, 3)) == 0 for d in dirs))
    check(f"{len(axis)} axis points of squared norm 4 (tau({k}) = {len(axis)})",
          all(sp.simplify(a.dot(a) - 4) == 0 for a in axis)
          and len(axis) == (6 if k == 2 else 12))
    check(f"the directions split into {len(tris)} zero-sum triangles covering all of them",
          all(sp.simplify(sum((dirs[i] for i in t), sp.zeros(k, 1)).norm()) == 0
              for t in tris)
          and sorted(i for t in tris for i in t) == list(range(len(dirs))))
    amax = max(sp.nsimplify(axis[i].dot(axis[j]))
               for i in range(len(axis)) for j in range(len(axis)) if i != j)
    check("axis is a 60-degree code: max <a,a'> = 2 <= 2", sp.simplify(amax - 2) == 0,
          f"max {amax}")
    damax = max(sp.nsimplify(d.dot(a)) for d in dirs for a in axis)
    check("every cap clears every axis point: max <d,a> <= 2",
          sp.simplify(damax) <= 2, f"max {sp.simplify(damax)} = {float(damax):.6f}")

    # threshold(s, s') = 144 - 96 * (largest cosine between triangle s and s')
    thr = {}
    for a in range(len(tris)):
        for b in range(len(tris)):
            cmax = max(sp.nsimplify(dirs[i].dot(dirs[j]) / sp.Rational(4, 3))
                       for i in tris[a] for j in tris[b])
            thr[(a, b)] = sp.nsimplify(144 - 96 * cmax)
    same = {thr[(a, a)] for a in range(len(tris))}
    cross = {thr[(a, b)] for a in range(len(tris)) for b in range(len(tris)) if a != b}
    check("same-triangle threshold is 48, cross-triangle 96",
          same == {48} and cross == {96}, f"same {same}, cross {cross}")
    selfthr = min(sp.nsimplify(144 - 96 * dirs[i].dot(dirs[j]) / sp.Rational(4, 3))
                  for t in tris for i in t for j in t if i != j)
    check("a head's own three caps are mutually compatible (192 <= 192)", selfthr == 192,
          f"threshold {selfthr}")

    print("\n3. the artefact")
    Y = np.load(os.path.join(data, f"heads{n}_Y.npy")).astype(np.int64)
    side = np.load(os.path.join(data, f"heads{n}_side.npy")).astype(np.int64)
    H = Y.shape[0]
    check(f"{H} heads with a triangle each", Y.shape[0] == side.shape[0])
    check("every head is an integer vector of norm 192",
          bool(((Y * Y).sum(1) == 192).all()))
    check(f"every triangle index is in range 0..{len(tris)-1}",
          bool(((side >= 0) & (side < len(tris))).all()))

    print("\n4. owners, recovered here by search over the whole shell")
    owners = np.full(H, -1, np.int64)
    multi = 0
    for s in range(0, H, 64):
        ip = Y[s:s + 64] @ Z.T
        bad = ip > 48
        cnt = bad.sum(1)
        for t in range(bad.shape[0]):
            if cnt[t] == 1:
                owners[s + t] = int(np.argmax(bad[t]))
            elif cnt[t] > 1:
                multi += 1
    check("no head conflicts with more than one minimal vector", multi == 0,
          f"{multi} heads with several")
    cls = np.where(owners >= 0)[0]
    free = np.where(owners < 0)[0]
    nc, nf = len(cls), len(free)
    check("every head is either a class head or a free head", nc + nf == H,
          f"{nc} class, {nf} free")
    check("the owners are distinct", len(set(owners[cls].tolist())) == nc)

    U = Z[owners[cls]]
    V = Y[cls] - 3 * U
    check("every class head is 3u + v with v a lean: |v|^2 = 48 and <u,v> = -24",
          bool(((V * V).sum(1) == 48).all()) and bool(((U * V).sum(1) == -24).all()))
    W = U + V
    check("w = u + v is a minimal vector with <u,w> = 8",
          bool(((W * W).sum(1) == 32).all()) and bool(((U * W).sum(1) == 8).all()))
    Vf = Y[free]
    okfree = all(int((Vf[t] * Vf[t]).sum()) == 192 and (Vf[t] % 2 == 0).all()
                 and int(((Vf[t] // 2) * (Vf[t] // 2)).sum()) == 48 for t in range(nf))
    check("every free head is +-2v for a lean v, and removes nothing", okfree)

    print("\n5. heads against the equator, exact integer arithmetic")
    retained = np.ones(196560, bool)
    retained[owners[cls]] = False
    viol = 0
    for s in range(0, H, 64):
        ip = Y[s:s + 64] @ Z.T
        viol += int(((ip > 48) & retained[None, :]).sum())
    check("no head conflicts with a retained equator point", viol == 0,
          f"violations {viol}")

    print("\n6. head against head, exact integer arithmetic")
    G = Y @ Y.T
    T = np.array([[int(thr[(a, b)]) for b in range(len(tris))]
                  for a in range(len(tris))], dtype=np.int64)
    lim = T[side[:, None], side[None, :]]
    bad = G > lim
    np.fill_diagonal(bad, False)
    nbad = int(bad.sum()) // 2
    check(f"all {H*(H-1)//2} head pairs are within their threshold", nbad == 0,
          f"{nbad} bad")

    print("\n7. equator against equator, over the whole retained shell")
    # The only pair class not otherwise touched. Computed, not assumed.
    mx = -10**9
    Zf = Z.astype(np.float32)
    for s in range(0, len(Z), 512):
        blk = Zf[s:s + 512] @ Zf.T
        for t in range(blk.shape[0]):
            blk[t, s + t] = -10**9          # drop the diagonal only
        mx = max(mx, int(blk.max()))
    check("distinct minimal vectors have inner product at most 16", mx <= 16,
          f"max {mx}")

    print("\n8. distinctness of every point of the configuration")
    check("all head vectors Y are distinct", len({Y[i].tobytes() for i in range(H)}) == H,
          f"{len({Y[i].tobytes() for i in range(H)})} of {H}")
    check("all equator points are distinct (the shell is)", len(index) == 196560)
    check("the three families cannot collide: caps have |d|^2 = 32/3 != 0 while "
          "equator and axis have zero auxiliary part; axis has zero Leech part "
          "while caps have |x|^2 = 64/3 != 0", True)
    # caps are indexed by (head, direction); distinct heads have distinct Y, and a
    # head's own three directions are distinct, so the 3H cap points are distinct.
    check("the 3 x H cap points are distinct",
          len({(Y[i].tobytes(), d) for i in range(H) for d in tris[side[i]]}) == 3 * H)

    print("\n9. full ambient rank")
    # Equator rows span R^24 exactly (integer rank over Q); axis rows span R^k.
    import sympy as _sp
    sub = Z[np.random.default_rng(0).choice(len(Z), 200, replace=False)]
    r24 = _sp.Matrix(sub.tolist()).rank()
    rk = _sp.Matrix([[sp.nsimplify(x) for x in a] for a in axis]).rank()
    check(f"the equator spans R^24 (exact rank of a 200-row integer submatrix)",
          r24 == 24, f"rank {r24}")
    check(f"the axis spans R^{k} (exact rank in Q(sqrt2, sqrt3))", rk == k,
          f"rank {rk}")
    check(f"the configuration spans R^{n}: {24} + {k} = {n}", r24 + rk == n)

    print("\n10. the count")
    eq = 196560 - nc
    total = eq + 3 * H + len(axis)
    want = expect if expect is not None else (199632 if n == 26 else 201010)
    check(f"equator {eq} + 3 x {H} caps + {len(axis)} axis = {want}", total == want,
          f"total {total}")

    print()
    if FAIL:
        print("FAILURES:", FAIL)
        return 1
    print(f"ALL CHECKS PASS      K({n}) >= {want}   (independent, exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]),
                  int(sys.argv[3]) if len(sys.argv) > 3 else None))

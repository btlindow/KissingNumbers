"""Independent exact verification of A. Kravatskiy's TWO-LAYER configurations in dimensions
26 and 27 (his tau(26) >= 199806 and tau(27) >= 201509 of 2026-09-20), and of any
configuration stored in the same format.

Written from the geometric description, sharing no code with his verify26.py / verify27.py /
lib/layered.py. The first layer is checked by this repository's verify2627_independent.py
(imported, not copied); this file adds the second layer and the combined count.

Units: "Cohn units" (minimal vector norm 32, compatible iff inner product <= 16).

THE SECOND LAYER. For a minimal vector u (the owner) and a unit line direction l in R^k, the two
points

    ( (sqrt3/2) u ,  +- sqrt8 l )          |x|^2 = 24,  |y|^2 = 8,  total 32.

The lines are the three lines through the hexagon's VERTICES (k = 2: the axis directions, 30
degrees from the nearest first-layer direction) or through the cuboctahedron's SQUARE-FACE
CENTRES +-e1, +-e2, +-e3 (k = 3: 45 degrees from the nearest direction). Only which heads share
a line matters to any condition below, never which line carries which index; that is checked
(every line meets every triangle of directions at the same largest cosine, and every two lines
meet at the same angle).

Pair conditions, each decided exactly (integers, or sympy signs in Q(sqrt2, sqrt3)):

  second layer / equator     (sqrt3/2) <u,z> <= 16. For z = u it is 16 sqrt3 > 16, so u is removed;
                             for z != u, <u,z> <= 16 and 8 sqrt3 < 16.
  second / first-layer cap   (sqrt3/6) <Y,u> + (16/sqrt3) c <= 16, c the largest cosine between
                             +-l and the head's three directions; i.e. <Y,u> <= 32 sqrt3 - 32 c,
                             turned into an exact integer bar per (line, triangle).
  second / second, one line  (3/4) <u,u'> + 8 <= 16 (same sign; the opposite sign is weaker)
  second / second, two lines (3/4) <u,u'> + 8 |cos(l,l')| <= 16
  second / axis              <sqrt8 l, a> <= 16
  a head's own two points    24 - 8 = 16 <= 16

Owners are recovered and checked here: every second-layer u must be a minimal vector of THIS
repository's shell, all distinct, and disjoint from the first layer's owners (found by search).

Usage:
    PYTHONPATH=python python tools/kravatskiy/verify2627_layers_independent.py <pkg> <26|27> <expected total>
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify2627_independent as first  # noqa: E402
from kiss_ref.leech import leech_min_vectors  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def exact_floor(expr):
    """floor of a real algebraic number, confirmed by two exact sign tests."""
    f = int(sp.floor(expr))
    assert sp.simplify(expr - f).is_nonnegative and sp.simplify(f + 1 - expr).is_positive
    return f


def lines_of(k, axis):
    if k == 2:
        L = [axis[m] / 2 for m in range(3)]                  # through the hexagon's vertices
    else:
        L = [sp.Matrix([1 if i == j else 0 for i in range(3)]) for j in range(3)]   # +-e_j
    assert all(sp.simplify(l.dot(l) - 1) == 0 for l in L)
    return L


def main(pkg, n, expect):
    FAIL.clear()
    k = n - 24
    data = os.path.join(pkg, "data")
    print(f"verify2627_layers_independent.py --- exact, independent check of K({n}) >= {expect}\n")

    Y = np.load(os.path.join(data, f"heads{n}_Y.npy")).astype(np.int64)
    side = np.load(os.path.join(data, f"heads{n}_side.npy")).astype(np.int64)
    fu = os.path.join(data, f"heads{n}_layer2_u.npy")
    if os.path.exists(fu):
        U2 = np.load(fu).astype(np.int64).reshape(-1, 24)
        line = np.load(os.path.join(data, f"heads{n}_layer2_line.npy")).astype(np.int64).reshape(-1)
    else:
        U2, line = np.zeros((0, 24), np.int64), np.zeros(0, np.int64)
    L2 = len(U2)

    print("A. THE FIRST LAYER, by verify2627_independent.py (its own count is checked in F)")
    Z = leech_min_vectors().astype(np.int64)
    own1 = np.full(len(Y), -1, np.int64)
    for s in range(0, len(Y), 64):
        hit = (Y[s:s + 64] @ Z.T) > 48
        own1[s:s + 64] = np.where(hit.sum(1) == 1, hit.argmax(1), -1)
    nc = int((own1 >= 0).sum())
    axis_n = 6 if k == 2 else 12
    first_total = 196560 - nc + 3 * len(Y) + axis_n
    rc = first.main(pkg, n, first_total)
    check(f"first layer passes every check of verify2627_independent.py (alone it gives {first_total})", rc == 0)

    print(f"\nB. the second layer: {L2} heads")
    dirs, tris, axis = first.geometry(k)                    # norm-4 units: |d|^2 = 4/3, |a|^2 = 4
    lines = lines_of(k, axis)
    check("line indices are in range", L2 == 0 or (line.min() >= 0 and line.max() < 3 and len(line) == L2))
    index = {Z[i].astype(np.int8).tobytes(): i for i in range(len(Z))}
    ids = [index.get(U2[i].astype(np.int8).tobytes(), -1) if np.abs(U2[i]).max() < 128 else -1 for i in range(L2)]
    check("every second-layer owner is a minimal vector of this repository's shell", all(i >= 0 for i in ids),
          f"{sum(1 for i in ids if i < 0)} are not")
    check("the second-layer owners are distinct", len(set(ids)) == L2)
    check("no second-layer owner is a first-layer owner", not (set(ids) & set(own1[own1 >= 0].tolist())))
    check("a second-layer point has norm 32: (3/4) 32 + 8", sp.Rational(3, 4) * 32 + 8 == 32)

    print("\nC. second layer against the equator and the axis")
    check("(sqrt3/2) <u,u> = 16 sqrt3 > 16: the owner is removed", sp.simplify(16 * sp.sqrt(3) - 16).is_positive)
    check("(sqrt3/2) 16 = 8 sqrt3 <= 16: no other minimal vector is touched (distinct minimal "
          "vectors meet at <= 16, checked over the whole shell in A.7)", sp.simplify(16 - 8 * sp.sqrt(3)).is_nonnegative)
    amax = max(sp.simplify(sg * (sp.sqrt(8) * l).dot(a * sp.sqrt(8))) for l in lines for a in axis for sg in (1, -1))
    check("every second-layer point clears every axis point: max <sqrt8 l, a> <= 16",
          sp.simplify(16 - amax).is_nonnegative, f"max {amax}")

    print("\nD. second layer against first-layer caps: <Y,u> <= 32 sqrt3 - 32 c, exact integer bars")
    bar = np.zeros((3, len(tris)), np.int64)
    cs = set()
    for a, l in enumerate(lines):
        for t, tri in enumerate(tris):
            c = max(sp.simplify(sg * l.dot(dirs[i]) / sp.sqrt(sp.Rational(4, 3))) for i in tri for sg in (1, -1))
            cs.add(c)
            bar[a, t] = exact_floor(32 * sp.sqrt(3) - 32 * c)
    check("every line meets every triangle of directions at the same largest cosine",
          len(cs) == 1, f"cos = {cs}, bar <Y,u> <= {sorted(set(bar.ravel().tolist()))}")
    if L2:
        M = Y @ U2.T                                         # H x L2, exact integers
        lim = bar[line][:, side].T                           # H x L2
        nbad = int((M > lim).sum())
        check(f"all {M.size} (first-layer head, second-layer head) pairs are within the bar", nbad == 0,
              f"{nbad} bad; max <Y,u> = {int(M.max())}")

    print("\nE. second layer against itself")
    same = exact_floor(sp.Rational(32, 3))                   # (3/4) g + 8 <= 16
    cross = {}
    for a in range(3):
        for b in range(3):
            if a != b:
                cross[(a, b)] = exact_floor((16 - 8 * abs(sp.simplify(lines[a].dot(lines[b])))) * sp.Rational(4, 3))
    check("every two lines meet at the same angle", len(set(cross.values())) == 1,
          f"one line: <u,u'> <= {same}; two lines: <u,u'> <= {sorted(set(cross.values()))}")
    if L2:
        G = U2 @ U2.T
        lim2 = np.where(line[:, None] == line[None, :], same, list(cross.values())[0])
        bad = G > lim2
        np.fill_diagonal(bad, False)
        check(f"all {L2*(L2-1)//2} second-layer pairs are within their threshold", not bad.any(),
              f"{int(bad.sum())//2} bad")
    check("a head's own two points meet at 24 - 8 = 16 <= 16", 24 - 8 <= 16)
    check("the 2 x L second-layer points are distinct from each other (distinct owners, two signs) and from "
          "every other family (|x|^2 = 24 with a non-zero auxiliary part)", len(set(ids)) == L2)

    print("\nF. the count")
    total = 196560 - nc - L2 + 3 * len(Y) + 2 * L2 + axis_n
    check(f"196560 - {nc} - {L2} + 3 x {len(Y)} + 2 x {L2} + {axis_n} = {expect}", total == expect, f"total {total}")

    print()
    if FAIL or first.FAIL:
        print("FAILURES:", FAIL + first.FAIL)
        return 1
    print(f"ALL CHECKS PASS      K({n}) >= {expect}   (independent, exact, two layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))

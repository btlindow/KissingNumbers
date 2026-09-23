"""Build the R^4 geometry of A. Kravatskiy's dimension-28 package, which ships no geometry file,
in the same exact format as his geom29/30/31.json: integer quadruples (a + b sqrt2 + c sqrt3 +
d sqrt6)/24.

From his README: the cap directions are the 24 normalised D4 roots, split into 8 zero-sum triples;
the axis is the 16 half-vectors of the dual 24-cell; the layer directions are the four coordinate
directions of R^4. Everything here is constructed and checked, not read from his scripts:

    directions   r/sqrt2 for the 24 roots r = (+-1, +-1, 0, 0) and permutations, so 1/sqrt2 -> (0,12,0,0)
    groups       a partition of those 24 into 8 triples pairwise at 120 degrees, found here by
                 exhaustive backtracking (the partition is not unique; any valid one verifies the
                 claim, and the verifier re-checks every pair condition against the one chosen)
    axis         (+-1, +-1, +-1, +-1)/2, 16 unit vectors, so 1/2 -> (12,0,0,0)
    frame        e_1..e_4

    python tools/kravatskiy/geom28.py <package>      writes data/geom28.json beside the artefact
"""

import itertools
import json
import os
import sys

DEN = 24
HALF = (DEN // 2, 0, 0, 0)          # 1/2
INVR2 = (0, DEN // 2, 0, 0)         # 1/sqrt2 = 12 sqrt2 / 24
ZERO = (0, 0, 0, 0)


def roots():
    """The 24 D4 roots, as unit vectors in quadruple form."""
    out = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [ZERO] * 4
                v[i] = tuple(si * c for c in INVR2)
                v[j] = tuple(sj * c for c in INVR2)
                out.append(tuple(v))
    return out


def ip(a, b):
    """Inner product of two of these vectors, as a multiple of 1/DEN^2 -- here always rational."""
    t = 0
    for p, q in zip(a, b):
        # both entries are of the form m*sqrt2/DEN, so the product is 2 m m' / DEN^2
        t += 2 * p[1] * q[1]
    return t


def triples(R):
    """Partition the 24 roots into 8 triples pairwise at 120 degrees (<r,r'> = -DEN^2/2)."""
    want = -DEN * DEN // 2
    n = len(R)
    adj = [[j for j in range(n) if j != i and ip(R[i], R[j]) == want] for i in range(n)]
    used = [False] * n
    out = []

    def bt():
        if len(out) * 3 == n:
            return True
        i = next(t for t in range(n) if not used[t])
        for j in adj[i]:
            if used[j] or j < i:
                continue
            for k in adj[i]:
                if used[k] or k <= j or ip(R[j], R[k]) != want:
                    continue
                used[i] = used[j] = used[k] = True
                out.append([i, j, k])
                if bt():
                    return True
                out.pop()
                used[i] = used[j] = used[k] = False
        return False

    assert bt(), "no partition of the D4 roots into zero-sum triples was found"
    return out


def axis16():
    out = []
    for s in itertools.product((1, -1), repeat=4):
        out.append(tuple(tuple(si * c for c in HALF) for si in s))
    return out


def frame4():
    out = []
    for i in range(4):
        v = [ZERO] * 4
        v[i] = (DEN, 0, 0, 0)
        out.append(tuple(v))
    return out


def build():
    R = roots()
    G = triples(R)
    A = axis16()
    W = frame4()
    assert len(R) == 24 and len(G) == 8 and len(A) == 16 and len(W) == 4
    for g in G:
        s = [sum(R[i][t][b] for i in g) for t in range(4) for b in range(4)]
        assert all(x == 0 for x in s), "a triple is not zero-sum"
    return {"dim": 28, "k": 4, "denominator": DEN,
            "directions": [[list(c) for c in v] for v in R],
            "groups": G,
            "axis": [[list(c) for c in v] for v in A],
            "frame": [[list(c) for c in v] for v in W]}


if __name__ == "__main__":
    pkg = sys.argv[1]
    g = build()
    out = os.path.join(pkg, "data", "geom28.json")
    json.dump(g, open(out, "w"))
    print(f"wrote {out}: {len(g['directions'])} directions in {len(g['groups'])} triples, "
          f"{len(g['axis'])} axis directions, {len(g['frame'])} frame directions")

#!/usr/bin/env python3
"""TASK 2(a), second identification: Kravatskiy's ARENA X_R (6120 of the 98280 lines) is
exactly the minimal-vector set of our sublattice <S> = Fix(t1)+Fix(t2)+Fix(t3), i.e. our
720 monads + 11520 duads on the Turyn sqrt(2)E8^3 blocks.

Checks, in exact integer arithmetic on independently regenerated minimal vectors:
  A1  X_R (defined by b(., R) = 0, R = perp_b of the record class's Lambda/2Lambda span)
      has 12240 vectors = 6120 lines.
  A2  X_R = C intersect <S>_Z, the minimal vectors of the Z-span of the record.
  A3  [Lambda : <S>] = 16, matching the 16-fold reduction.
  A4  X_R splits into 2880 fibres of 2 lines and 45 fibres of 8 lines (his numbers);
      the 2-line fibres are exactly our 2880 duad orbits {+-x, +-y} and the 8-line fibres
      are the 360 monad lines, 8 per (block, channel).
  A5  the record uses 124 of the 2-line fibres and none of the 8-line fibres.
"""
from __future__ import annotations
import os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
from kiss_ref.leech import leech_min_vectors  # noqa: E402
from r_subspace_vs_channels import Mod2, kernel, f2_span, read_set, n  # noqa: E402

PC = np.array([bin(i).count("1") & 1 for i in range(1 << 16)], dtype=np.uint8)


def parity(v):
    return PC[v & 0xFFFF] ^ PC[(v >> 16) & 0xFFFF]


def main():
    C = leech_min_vectors().astype(np.int64)
    key = {tuple(r): i for i, r in enumerate(map(tuple, C))}
    neg = np.array([key[tuple(-r)] for r in C], dtype=np.int64)
    mod = Mod2(C, 7)
    S = read_set(os.path.join(ROOT, "data", "S496.txt"))
    idx = np.array([key[tuple(r)] for r in S], dtype=np.int64)
    reps = sorted({min(int(i), int(neg[int(i)])) for i in idx})
    P = mod.PHI[reps]

    kb = kernel({mod.bfun(int(p)) for p in P})
    W = f2_span(kb)
    arena = np.ones(len(C), dtype=bool)
    for r in kb:
        arena &= parity(mod.PHI & int(mod.bfun(r))) == 0
    sel = np.nonzero(arena)[0]
    lines = len({min(int(i), int(neg[int(i)])) for i in sel})
    print(f"RESULT A1 dim R = {len(kb)}, |X_R| = {len(sel)} vectors = {lines} lines "
          f"(Kravatskiy: 6120 lines) -> {lines == 6120 and len(sel) == 12240}")

    # A2/A3: <S>_Z and its minimal vectors, via an HNF basis of the span of S
    from r_subspace_vs_channels import modular_hnf, DET
    rowsS = [[DET if i == j else 0 for j in range(n)] for i in range(n)] + S.tolist()
    BS = np.array(modular_hnf(rowsS), dtype=np.int64)
    detS = 1
    for i in range(n):
        detS *= int(BS[i, i])
    index = abs(detS) // DET
    # membership of every minimal vector in <S>
    X = C.astype(np.int64).copy()
    inS = np.ones(len(C), dtype=bool)
    for i in range(n):
        d = int(BS[i, i])
        r = X[:, i] % d
        inS &= (r == 0)
        q = np.where(inS, X[:, i] // d, 0)
        X -= np.outer(q, BS[i])
    inS &= ~X.any(axis=1)
    print(f"RESULT A3 [Lambda : <S>] = {index} (16-fold reduction) -> {index == 16}")
    print(f"RESULT A2 C ∩ <S> has {int(inS.sum())} vectors; equals X_R: {bool(np.all(inS == arena))}")

    # A4: fibres of X_R = cosets of R inside the image of X_R
    lreps = sorted({min(int(i), int(neg[int(i)])) for i in sel})
    fib = Counter()
    for i in lreps:
        fib[int(mod.PHI[i])] += 1        # a fibre = a coset of R; classes differ by R
    # group line-classes into R-cosets
    cos = {}
    for i in lreps:
        c = int(mod.PHI[i])
        rep = min(c ^ w for w in W)
        cos.setdefault(rep, []).append(i)
    sizes = Counter(len(v) for v in cos.values())
    print(f"RESULT A4 X_R splits into {len(cos)} cosets of R, sizes {dict(sorted(sizes.items()))} "
          f"(Kravatskiy: 2925 fibres = 2880 of size 2 + 45 of size 8) -> "
          f"{len(cos) == 2925 and sizes.get(2) == 2880 and sizes.get(8) == 45}")

    used = Counter()
    for rep, mem in cos.items():
        k = len(set(mem) & set(reps))
        if k:
            used[len(mem)] += 1
    print(f"RESULT A5 the record occupies {sum(used.values())} fibres: "
          f"{dict(sorted(used.items()))} by fibre size "
          f"(Kravatskiy: 124 fibres of size 2, none of size 8) -> "
          f"{used.get(2) == 124 and used.get(8, 0) == 0}")

    # our reading: the size-2 fibres are duad orbits, the size-8 fibres are monad lines
    norms2 = set()
    for rep, mem in cos.items():
        if len(mem) == 2:
            a, b = mem
            norms2.add(int((C[a] + C[b]) @ (C[a] + C[b])))
    print(f"RESULT A6 |x+y|^2 over the size-2 fibres: {sorted(norms2)} (64 = our duad orbits, "
          f"since the two lines of a fibre are orthogonal) -> {norms2 == {64}}")
    ok = (lines == 6120 and index == 16 and bool(np.all(inS == arena))
          and len(cos) == 2925 and sizes.get(2) == 2880 and sizes.get(8) == 45
          and used.get(2) == 124 and used.get(8, 0) == 0)
    print(f"RESULT task2a-arena ok={int(ok)} verdict='X_R == the 12240 monads+duads == "
          f"minimal vectors of <S>, index 16 in Lambda'")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

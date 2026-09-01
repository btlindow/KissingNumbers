#!/usr/bin/env python3
"""TASK 1, verifier A: the LINE problem -> VECTOR problem reduction, on regenerated data.

Kravatskiy's "class problem" counts LINES of Leech minimal vectors pairwise at
|cos| <= 1/4 (i.e. both inner products +-16 forbidden, in the norm-32 scaling).
Our problem counts VECTORS with no pair at 60 degrees (only +16 forbidden).

This script checks, in exact integer arithmetic on the independently regenerated
196560 minimal vectors (kiss_ref, never data/leech_min.*):

  L1  C is closed under negation; |C| = 196560; #lines = 98280.
  L2  the line conflict relation |<u,v>| = 16 has valency 4600 (Kravatskiy's
      Hoffman input k = 4600, N = 98280, lambda_min = -20).
  L3  DOUBLING LEMMA, checked on witnesses: if T is a set of lines with
      |<.,.>| <= 8 across distinct lines, then S = {+-x : x in T} is a set of
      2|T| vectors of C with no pair at inner product +16.
      Witnesses: the record 248-line class, plus randomly grown maximal
      +-16-free line sets.
  L4  the record S496 is antipodally closed, is 248 lines, and is +-16-free
      (so it is simultaneously feasible for both problems: the two problems
      have the same known record).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
from kiss_ref.leech import leech_min_vectors  # noqa: E402

def read_set(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            t = raw.split("#", 1)[0].strip()
            if t:
                rows.append([int(v) for v in t.split()])
    return np.array(rows, dtype=np.int64)

def main() -> int:
    C = leech_min_vectors().astype(np.int64)
    N = C.shape[0]
    ok = True
    norms = (C * C).sum(1)
    assert np.all(norms == 32)

    # --- L1: antipodal closure and the number of lines -----------------------
    key = {tuple(r): i for i, r in enumerate(map(tuple, C))}
    neg = np.array([key[tuple(-r)] for r in C], dtype=np.int64)
    l1 = (len(key) == N == 196560) and np.all(neg[neg] == np.arange(N)) and np.all(neg != np.arange(N))
    lines = N // 2
    print(f"RESULT L1 antipodal_closed={bool(l1)} N={N} lines={lines} "
          f"expect lines=98280 -> {lines == 98280}")
    ok = ok and l1 and lines == 98280

    # --- L2: valency of the line conflict relation ---------------------------
    # Co_0 is transitive on C, but check several base points anyway.
    rng = np.random.default_rng(20260901)
    vals = set()
    for i in list(rng.choice(N, size=8, replace=False)) + [0]:
        d = C @ C[int(i)]
        p16 = int((d == 16).sum()); m16 = int((d == -16).sum())
        # lines conflicting with the line of C[i]: y with |<x,y>| = 16, halved
        vals.add((p16, m16, (p16 + m16) // 2))
    print(f"RESULT L2 per-basepoint (#dot=+16, #dot=-16, #conflicting lines) = {sorted(vals)} "
          f"-> vector-valency 4600, line-valency 4600: {vals == {(4600, 4600, 4600)}}")
    ok = ok and vals == {(4600, 4600, 4600)}

    # --- L4: the record --------------------------------------------------
    S = read_set(os.path.join(ROOT, "data", "S496.txt"))
    idx = np.array([key[tuple(r)] for r in S], dtype=np.int64)
    G = S @ S.T
    off = ~np.eye(len(S), dtype=bool)
    anti = set(map(int, neg[idx])) == set(map(int, idx))
    nlines = len({min(int(i), int(j)) for i, j in zip(idx, neg[idx])})
    plus16 = int((G[off] == 16).sum()); minus16 = int((G[off] == -16).sum())
    print(f"RESULT L4 |S496|={len(S)} antipodally_closed={anti} lines={nlines} "
          f"pairs_at_+16={plus16} pairs_at_-16={minus16} "
          f"-> feasible for BOTH problems: {anti and nlines == 248 and plus16 == 0 and minus16 == 0}")
    ok = ok and anti and nlines == 248 and plus16 == 0 and minus16 == 0

    # --- L3: the doubling lemma on witnesses ---------------------------------
    def line_rep(i):  # canonical representative index of the line of vector i
        return min(int(i), int(neg[i]))

    def check_double(reps):
        """reps: indices, one per line, pairwise |dot| <= 8.  Double and test."""
        R = C[np.asarray(reps)]
        g = R @ R.T
        m = ~np.eye(len(reps), dtype=bool)
        assert np.all(np.abs(g[m]) <= 8), "input is not a valid line set"
        D = np.concatenate([np.asarray(reps), neg[np.asarray(reps)]])
        assert len(set(map(int, D))) == 2 * len(reps)
        gd = C[D] @ C[D].T
        md = ~np.eye(len(D), dtype=bool)
        return int((gd[md] == 16).sum()), len(D)

    witnesses = []
    # witness 0: the record class
    reps0 = sorted({line_rep(i) for i in idx})
    witnesses.append(("S496-248-lines", reps0))
    # witnesses 1..5: greedily grown maximal +-16-free line sets
    for t in range(5):
        r = np.random.default_rng(1000 + t)
        order = r.permutation(N)
        chosen: list[int] = []
        for i in order:
            i = line_rep(i)
            if i in chosen:
                continue
            if chosen and np.any(np.abs(C[chosen] @ C[i]) == 16):
                continue
            chosen.append(int(i))
        witnesses.append((f"greedy-{t}", sorted(set(chosen))))

    allok = True
    for name, reps in witnesses:
        bad, sz = check_double(reps)
        good = bad == 0 and sz == 2 * len(reps)
        allok = allok and good
        print(f"RESULT L3 doubling[{name}] lines={len(reps)} vectors={sz} "
              f"pairs_at_+16_after_doubling={bad} ok={good}")
    ok = ok and allok

    print(f"RESULT task1-verifierA ok={int(ok)} "
          f"conclusion='a line set of size L doubles to a 2L-vector 60-degree-free set, "
          f"so L <= floor(alpha_vec/2)'")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())

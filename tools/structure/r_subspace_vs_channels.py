#!/usr/bin/env python3
"""TASK 2(a): are Kravatskiy's "124 orthogonal pairs whose differences span a 4-dimensional
totally isotropic subspace R of Lambda/2Lambda" and our "15 channels" the same object?

Everything is rebuilt from scratch in exact integer arithmetic:
  * a modular-HNF basis B of the integer lattice L spanned by the 196560 minimal vectors
    (norm-32 scaling); |det B| = 8^12 = 2^36 certifies that B is a basis of L, so the
    F2 coordinates it induces really are Lambda/2Lambda;
  * b(x,y) = (<x,y>/8) mod 2 and its Gram matrix in those coordinates;
  * for each recorded 496: the perp W = { v : b(v, [s]) = 0 for every class [s] of the set },
    the induced pairing of the 248 lines, the channel of each pair, and the arena.

Two verifiers: the whole computation is run under TWO independent HNF bases (different
random seeds -> different coordinates on Lambda/2Lambda). Every reported invariant is
basis-free, so the two runs must agree exactly.

    python tools/structure/r_subspace_vs_channels.py
"""
from __future__ import annotations
import os, sys, json
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))
from kiss_ref.leech import leech_min_vectors  # noqa: E402

n, DET = 24, 1 << 36


def egcd(a, b):
    x, x0, y, y0 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x, x0 = x0, x - q * x0
        y, y0 = y0, y - q * y0
    return a, x, y


def modular_hnf(rows):
    """Upper-triangular basis of the lattice spanned by `rows`, worked modulo DET
    (the rows DET*e_i are included by the caller), so entries stay small."""
    piv = [None] * n

    def backreduce():
        for i in range(n):
            if piv[i] is None:
                continue
            for j in range(i + 1, n):
                if piv[j] is None:
                    continue
                q = piv[i][j] // piv[j][j]
                if q:
                    for k in range(j, n):
                        piv[i][k] -= q * piv[j][k]

    for r in rows:
        r = [int(t) % DET for t in r]
        c = 0
        while c < n:
            r[c] %= DET
            if r[c] == 0:
                c += 1
                continue
            if piv[c] is None:
                piv[c] = r
                break
            p = piv[c]
            a, b = p[c], r[c]
            g, x, y = egcd(a, b)
            piv[c] = [(x * p[k] + y * r[k]) % DET for k in range(n)]
            r = [((a // g) * r[k] - (b // g) * p[k]) % DET for k in range(n)]
        backreduce()
    return piv


def read_set(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.split("#", 1)[0].strip()
            if t:
                out.append([int(v) for v in t.split()])
    return np.array(out, dtype=np.int64)


class Mod2:
    """Lambda/2Lambda in the coordinates of one HNF basis."""

    def __init__(self, C, seed):
        rng = np.random.default_rng(seed)
        rows = [[DET if i == j else 0 for j in range(n)] for i in range(n)]
        rows += C[rng.choice(len(C), 400, replace=False)].tolist()
        self.B = np.array(modular_hnf(rows), dtype=np.int64)
        self.det = 1
        for i in range(n):
            self.det *= int(self.B[i, i])
        assert abs(self.det) == DET, f"|det| = {self.det}, not 2^36: not a basis of L"
        G = self.B @ self.B.T
        assert np.all(G % 8 == 0)
        M = (G // 8) % 2
        self.Mrows = [int(sum(int(M[i, j]) << j for j in range(n))) for i in range(n)]
        self.PHI = self.phi(C)

    def phi(self, X):
        X = X.astype(np.int64).copy()
        out = np.zeros(len(X), dtype=np.int64)
        for i in range(n):
            d = int(self.B[i, i])
            assert np.all(X[:, i] % d == 0)
            ci = X[:, i] // d
            out |= (ci & 1) << i
            X -= np.outer(ci, self.B[i])
        assert not X.any(), "vector not in L"
        return out

    def b(self, u, v):
        acc, uu = 0, int(u)
        while uu:
            i = (uu & -uu).bit_length() - 1
            acc ^= bin(self.Mrows[i] & int(v)).count("1") & 1
            uu &= uu - 1
        return acc

    def bfun(self, v):
        """mask of the functional u -> b(u, v)"""
        acc, vv = 0, int(v)
        while vv:
            i = (vv & -vv).bit_length() - 1
            acc ^= self.Mrows[i]
            vv &= vv - 1
        return acc


def f2_span(basis):
    S = {0}
    for b_ in basis:
        S |= {s ^ b_ for s in S}
    return S


def f2_rank(vals):
    bas = []
    for v in vals:
        v = int(v)
        for b_ in bas:
            v = min(v, v ^ b_)
        if v:
            bas.append(v)
            bas.sort(reverse=True)
    return bas


def kernel(masks):
    piv = {}
    for m in masks:
        mm = int(m)
        for c in sorted(piv, reverse=True):
            if (mm >> c) & 1:
                mm ^= piv[c]
        if mm:
            piv[mm.bit_length() - 1] = mm
    for c in sorted(piv, reverse=True):
        for c2 in list(piv):
            if c2 != c and (piv[c2] >> c) & 1:
                piv[c2] ^= piv[c]
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        v = 1 << f
        for c, pm in piv.items():
            if (pm >> f) & 1:
                v |= 1 << c
        out.append(v)
    return out


def analyse(mod, C, neg, key, S, name):
    idx = np.array([key[tuple(r)] for r in S], dtype=np.int64)
    reps = sorted({min(int(i), int(neg[int(i)])) for i in idx})
    assert len(reps) * 2 == len(S), f"{name}: not antipodally closed"
    P = mod.PHI[reps]
    Gm = C[reps] @ C[reps].T
    np.fill_diagonal(Gm, 0)
    assert not (Gm == 16).any() and not (Gm == -16).any(), f"{name}: has a +-16 pair"

    rank = len(f2_rank(P))
    W = f2_span(kernel({mod.bfun(int(p)) for p in P}))
    iso = all(mod.b(u, v) == 0 for u in W for v in W)
    perp = all(mod.b(w, int(p)) == 0 for w in W for p in P)

    cand = [[j for j in range(len(reps))
             if j != i and Gm[i, j] == 0 and (int(P[i]) ^ int(P[j])) in W]
            for i in range(len(reps))]
    degs = Counter(len(c) for c in cand)
    matched = set(degs) == {1}
    pairs = sorted({tuple(sorted((i, cand[i][0]))) for i in range(len(reps))}) if matched else []
    ch = Counter(int(P[a]) ^ int(P[b]) for a, b in pairs)
    prof = dict(sorted(Counter(ch.values()).items()))

    # x+y and x-y have norm 64 and share the class phi(x)^phi(y): the channel, our definition
    sums = np.array([C[reps[a]] + C[reps[b]] for a, b in pairs], dtype=np.int64)
    difs = np.array([C[reps[a]] - C[reps[b]] for a, b in pairs], dtype=np.int64)
    norms = set(map(int, (sums * sums).sum(1))) | set(map(int, (difs * difs).sum(1)))
    same = bool(np.all(mod.phi(sums) == mod.phi(difs)))
    agree = bool(np.all(mod.phi(sums) == np.array([int(P[a]) ^ int(P[b]) for a, b in pairs])))

    arena = np.ones(len(C), dtype=bool)
    pc = np.array([bin(i).count("1") & 1 for i in range(1 << 16)], dtype=np.uint8)

    def parity(v):                      # v: int64 array of 24-bit masks
        return pc[v & 0xFFFF] ^ pc[(v >> 16) & 0xFFFF]

    for r in kernel({mod.bfun(int(p)) for p in P}):
        arena &= parity(mod.PHI & int(mod.bfun(r))) == 0
    alines = len({min(int(i), int(neg[int(i)])) for i in np.nonzero(arena)[0]})

    mv = sorted({int((mod.PHI == w).sum()) for w in W})
    return {
        "name": name, "vectors": len(S), "lines": len(reps), "class_rank": rank,
        "dimW": 24 - rank, "W_size": len(W), "isotropic": iso, "perp_ok": perp,
        "perfect_matching": matched, "pairs": len(pairs),
        "channels_used": len(ch), "channel_profile": prof,
        "norms_of_x_plus_minus_y": sorted(norms), "sum_diff_same_class": same,
        "channel_is_xor": agree, "arena_lines": alines,
        "min_vectors_per_channel_class": mv,
    }


def main():
    C = leech_min_vectors().astype(np.int64)
    key = {tuple(r): i for i, r in enumerate(map(tuple, C))}
    neg = np.array([key[tuple(-r)] for r in C], dtype=np.int64)

    sets = [("S496.txt", os.path.join(ROOT, "data", "S496.txt"))]
    fam = os.path.join(ROOT, "data", "S496_family")
    if os.path.isdir(fam):
        sets += [(f, os.path.join(fam, f)) for f in sorted(os.listdir(fam)) if f.endswith(".txt")]

    runs = {}
    for seed in (7, 20260901):
        mod = Mod2(C, seed)
        print(f"RESULT basis seed={seed}: |det|={abs(mod.det)} = 2^36 -> a genuine basis of L; "
              f"rank of b = {len(f2_rank(mod.Mrows))}/24")
        out = [analyse(mod, C, neg, key, read_set(p), nm) for nm, p in sets]
        runs[seed] = out
        a = out[0]
        print(f"RESULT seed={seed} S496: class rank={a['class_rank']} dim W={a['dimW']} "
              f"|W|={a['W_size']} isotropic={a['isotropic']} perp={a['perp_ok']} "
              f"matching={a['perfect_matching']} pairs={a['pairs']} "
              f"channels_used={a['channels_used']}/15 profile={a['channel_profile']} "
              f"arena={a['arena_lines']} lines")

    A, B_ = runs[7], runs[20260901]
    keys = [k for k in A[0] if k != "channel_profile"]
    same = all(all(x[k] == y[k] for k in keys) and x["channel_profile"] == y["channel_profile"]
               for x, y in zip(A, B_))
    print(f"RESULT two-verifier: the two independent Lambda/2Lambda coordinate systems agree on "
          f"every invariant for all {len(A)} sets: {same}")

    inv = Counter((x["dimW"], x["W_size"], x["isotropic"], x["perfect_matching"], x["pairs"],
                   x["channels_used"], tuple(sorted(x["channel_profile"].items())),
                   x["arena_lines"], x["class_rank"]) for x in A)
    print(f"RESULT across all {len(A)} recorded 496s the invariant "
          f"(dimW, |W|, isotropic, matching, pairs, channels, profile, arena, rank) takes "
          f"{len(inv)} distinct value(s):")
    for k, v in inv.items():
        print(f"          {v:3d} x  dimW={k[0]} |W|={k[1]} isotropic={k[2]} matching={k[3]} "
              f"pairs={k[4]} channels_used={k[5]} profile={dict(k[6])} arena={k[7]} rank={k[8]}")
    a = A[0]
    print(f"RESULT channel identity: |x+y|^2 and |x-y|^2 in {a['norms_of_x_plus_minus_y']}, "
          f"[x+y]==[x-y]: {a['sum_diff_same_class']}, equals phi(x)^phi(y): {a['channel_is_xor']}, "
          f"minimal vectors in a channel class: {a['min_vectors_per_channel_class']}")
    ok = same and len(inv) == 1 and a["dimW"] == 4 and a["arena_lines"] == 6120
    print(f"RESULT task2a ok={int(ok)} verdict='R (Kravatskiy) == T (our 15 channels)'")
    out = os.environ.get("KISS_OUT", os.path.join(ROOT, "runs"))
    os.makedirs(out, exist_ok=True)
    json.dump(A, open(os.path.join(out, "r_vs_channels.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""T1.3 (fetch half): read-only inspection of the external data in data/external/.

Reports, at the Gram level and in each repository's OWN coordinates (no conversion):
  * PackingStar 24D_496_Si.npy, 24D_196560_coordinates.npy
  * Kallal-Kan-Wang S_1..S_59.txt, minvects.txt
  * the implicit S_i / T_i / extra-sphere decomposition of each PackingStar
    <n>D_<count>_coordinates.npy for n = 25..31 (unit-vector configurations)
  * whether the two 196560-vector lists are the same set, and whether their
    octads coincide with the cyclic Golay code from README.md section 1.2.

Run:  python3 python/tools/inspect_external.py [--no-configs]
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "data", "external")
PS = os.path.join(EXT, "PackingStar", "25D-31D")
KKW = os.path.join(EXT, "Kissing-Numbers")
KISSING = {1: 2, 2: 6, 3: 12, 4: 24, 5: 40, 6: 72, 7: 126}


def rows(a):
    return {tuple(int(v) for v in r) for r in a}


def gram_report(name, a):
    a = np.asarray(a)
    G = a @ a.T
    diag = np.unique(np.diag(G))
    off = G[~np.eye(len(a), dtype=bool)]
    u, c = np.unique(off, return_counts=True)
    keys = rows(a)
    antipodal = all(tuple(-v for v in r) in keys for r in keys)
    hist = dict(zip(u.tolist(), c.tolist()))
    print(f"  {name}: n={len(a)} dim={a.shape[1]} dtype={a.dtype} coords={sorted(set(a.ravel().tolist()))}")
    print(f"      diag={diag.tolist()} off-diag multiset={hist} max_off={off.max()} "
          f"distinct={len(keys)} antipodal={antipodal} independent(max<=8)={off.max() <= 8}")
    return hist


def orth_relation_report(name, a):
    """Cross diagnostic: relation x~y iff <x,y> in {0,-32}. Is it transitive?"""
    G = a @ a.T
    A = (G == 0) | (G == -32)
    np.fill_diagonal(A, True)
    deg = A.sum(1) - 1
    dd = collections.Counter(deg.tolist())
    Ai = A.astype(np.int32)
    # transitivity: A[x,y] & A[y,z] => A[x,z]
    reach = (Ai @ Ai) > 0
    transitive = not np.any(reach & ~A)
    print(f"  {name}: |{{y: <x,y> in {{0,-32}}}}| distribution = {dict(sorted(dd.items()))}; relation transitive = {transitive}")
    if transitive:
        # connected components = equivalence classes
        seen = -np.ones(len(a), dtype=int)
        k = 0
        for i in range(len(a)):
            if seen[i] < 0:
                seen[A[i]] = k
                k += 1
        sizes = collections.Counter(collections.Counter(seen.tolist()).values())
        print(f"      classes: {dict(sorted(sizes.items()))}")


def cyclic_golay_octads():
    """Extended Golay code from g(x) = x^11+x^10+x^6+x^5+x^4+x^2+1 (README 1.2); returns octads as 24-bit masks."""
    g = 0
    for e in (0, 2, 4, 5, 6, 10, 11):
        g |= 1 << e
    gens = [g << i for i in range(12)]  # degree <= 22, no wraparound needed
    words = [0]
    for gen in gens:
        words += [w ^ gen for w in words]
    ext = []
    for w in words:
        p = bin(w).count("1") & 1
        ext.append(w | (p << 23))
    wd = collections.Counter(bin(w).count("1") for w in ext)
    assert dict(wd) == {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}, wd
    return {w for w in ext if bin(w).count("1") == 8}


def octads_of(minvecs):
    oct_ = set()
    for r in minvecs:
        if np.max(np.abs(r)) == 2:
            m = 0
            for i, v in enumerate(r):
                if v != 0:
                    m |= 1 << i
            oct_.add(m)
    return oct_


def decompose_config(path, d, ref_hist=None):
    """Split a PackingStar unit-vector configuration in R^{24+d} into
    equatorial (x,0)/sqrt32, lifted (x*sqrt(2/3), y*sqrt(4/3))/2, extra (0,y') and
    recover the S_i (integer sqrt8 scaling) and T_i."""
    a = np.load(path)
    n, dim = a.shape
    assert dim == 24 + d, (dim, d)
    head, tail = a[:, :24], a[:, 24:]
    hn, tn = (head ** 2).sum(1), (tail ** 2).sum(1)
    eq = tn < 1e-9
    extra = hn < 1e-9
    lifted = ~eq & ~extra
    print(f"  {os.path.basename(path)}: n={n} dim={dim} | equatorial={eq.sum()} lifted={lifted.sum()} extra={extra.sum()} (K({d})={KISSING[d]})")
    print(f"      lifted head-norm^2 values={np.unique(np.round(hn[lifted], 6)).tolist()} (expect 2/3=0.666667), tail-norm^2={np.unique(np.round(tn[lifted], 6)).tolist()} (expect 1/3)")
    xe = head[eq] * np.sqrt(32)
    xl = head[lifted] * 4 * np.sqrt(3)
    print(f"      integrality residual: equatorial*sqrt32 {np.abs(xe - np.rint(xe)).max():.2e}, lifted*4sqrt3 {np.abs(xl - np.rint(xl)).max():.2e}")
    xe = np.rint(xe).astype(int)
    xl = np.rint(xl).astype(int)
    yl = tail[lifted] * np.sqrt(3)
    ykey = [tuple(np.round(y, 6)) for y in yl]
    by_y = collections.defaultdict(list)
    for k, x in zip(ykey, xl):
        by_y[k].append(tuple(x))
    # group y's by identical S
    by_S = collections.defaultdict(list)
    for k, xs in by_y.items():
        by_S[frozenset(xs)].append(k)
    groups = sorted(by_S.items(), key=lambda kv: (-len(kv[1]), -len(kv[0])))
    Tsizes = collections.Counter(len(v) for _, v in groups)
    Ssizes = collections.Counter(len(k) for k, _ in groups)
    print(f"      distinct T-vectors={len(by_y)}; groups (S_i,T_i)={len(groups)}; |T_i| distribution={dict(sorted(Tsizes.items()))}; |S_i| distribution={dict(sorted(Ssizes.items()))}")
    nx = int(extra.sum())
    total = nx + 196560 + sum((len(T) - 1) * len(S) for S, T in groups)
    print(f"      formula #extra + 196560 + sum(|T_i|-1)|S_i| = {nx} + 196560 + {sum((len(T) - 1) * len(S) for S, T in groups)} = {total}  (file count {n}, match={total == n}; K({d})={KISSING[d]})")
    # disjointness, independence, fingerprint
    allS = [k for k, _ in groups]
    union = set().union(*allS)
    disjoint = sum(len(S) for S in allS) == len(union)
    eqset = rows(xe)
    print(f"      S_i pairwise disjoint={disjoint}; |union S_i|={len(union)}; equatorial∩union={len(eqset & union)}; |eq|+|union|={len(eqset) + len(union)}")
    same_as_ref = 0
    indep = True
    for S, T in groups:
        M = np.array(sorted(S))
        G = M @ M.T
        off = G[~np.eye(len(M), dtype=bool)]
        if off.max() > 8 or np.any(np.diag(G) != 32):
            indep = False
        u, c = np.unique(off, return_counts=True)
        if ref_hist is not None and dict(zip(u.tolist(), c.tolist())) == ref_hist:
            same_as_ref += 1
    print(f"      every S_i independent (norm 32, off-diag<=8)={indep}; S_i with Gram histogram identical to the 496 = {same_as_ref}/{len(groups)}")
    # T structure
    Y = np.array([np.array(k) for k in by_y])
    Yg = np.round(Y @ Y.T, 6)
    within = collections.Counter()
    across = collections.Counter()
    yidx = {k: i for i, k in enumerate(by_y)}
    gid = {}
    for gi, (_, T) in enumerate(groups):
        for k in T:
            gid[yidx[k]] = gi
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            (within if gid[i] == gid[j] else across)[float(Yg[i, j])] += 1
    print(f"      T inner products within T_i={dict(sorted(within.items()))}; across T_i max={max(across) if across else None}")
    if extra.sum():
        E = tail[extra]
        EG = np.round(E @ E.T, 6)
        eo = EG[~np.eye(len(E), dtype=bool)]
        EY = np.round(E @ Y.T, 6)
        print(f"      extra spheres: {len(E)}; pairwise cos max={eo.max() if len(E) > 1 else None}; max cos(extra, T)={EY.max()} (need <= sqrt3/2=0.866025)")
    return groups, union, eqset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-configs", action="store_true", help="skip the 25D-31D configuration decomposition")
    args = ap.parse_args()
    np.set_printoptions(linewidth=200)

    print("== PackingStar: Si_configurations/24D_496_Si.npy")
    s496 = np.load(os.path.join(PS, "Si_configurations", "24D_496_Si.npy"))
    h496 = gram_report("24D_496_Si", s496)
    orth_relation_report("24D_496_Si", s496.astype(np.int64))

    print("== PackingStar: Si_configurations/24D_196560_coordinates.npy")
    c_ps = np.load(os.path.join(PS, "Si_configurations", "24D_196560_coordinates.npy"))
    print(f"  shape={c_ps.shape} dtype={c_ps.dtype} integral={np.abs(c_ps - np.rint(c_ps)).max() == 0} norms={np.unique((c_ps ** 2).sum(1)).tolist()}")
    c_ps = np.rint(c_ps).astype(np.int64)
    c_ps_set = rows(c_ps)
    g0 = c_ps @ c_ps[0]
    u, c = np.unique(g0, return_counts=True)
    print(f"  distinct={len(c_ps_set)}; ip histogram vs row 0: {dict(zip(u.tolist(), c.tolist()))}")
    print(f"  496 subset of PackingStar 196560: {rows(s496) <= c_ps_set}")

    print("== Kallal-Kan-Wang: minvects.txt, Vbasis.txt")
    c_kkw = np.loadtxt(os.path.join(KKW, "minvects.txt"), dtype=np.int64)
    c_kkw_set = rows(c_kkw)
    g0 = c_kkw @ c_kkw[0]
    u, c = np.unique(g0, return_counts=True)
    print(f"  minvects shape={c_kkw.shape} distinct={len(c_kkw_set)} norms={np.unique((c_kkw ** 2).sum(1)).tolist()} ip histogram vs row 0: {dict(zip(u.tolist(), c.tolist()))}")
    shapes = collections.Counter()
    for r in c_kkw:
        m = int(np.max(np.abs(r)))
        shapes[{2: "(±2^8,0^16)", 3: "(∓3,±1^23)", 4: "(±4,±4,0^22)"}[m]] += 1
    print(f"  shape counts: {dict(shapes)}")
    print(f"  PackingStar 196560 set == KKW minvects set: {c_ps_set == c_kkw_set}; same row order: {bool(np.array_equal(c_ps, c_kkw))}")
    print(f"  496 subset of KKW minvects: {rows(s496) <= c_kkw_set}")
    vb = np.loadtxt(os.path.join(KKW, "Vbasis.txt"), dtype=np.int64)
    Gb = vb @ vb.T
    print(f"  Vbasis shape={vb.shape} det(Gram)={np.linalg.det(Gb.astype(float)):.6g} (expect 8^24={8**24:.6g} for sqrt8-scaled Leech); diag={np.unique(np.diag(Gb)).tolist()}")
    print(f"  Vbasis rows all in minvects: {rows(vb) <= c_kkw_set}")

    print("== Octads vs cyclic Golay code of README 1.2 (identity coordinates)")
    oc = cyclic_golay_octads()
    ok = octads_of(c_kkw)
    print(f"  octads in minvects: {len(ok)}; cyclic-Golay octads: {len(oc)}; common: {len(ok & oc)}")
    # is the octad set the weight-8 part of SOME linear code? (span dimension)
    basis = []
    for w in ok:
        x = w
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    print(f"  GF(2)-span of minvects octads has dimension {len(basis)} (expect 12)")

    print("== Kallal-Kan-Wang: S_1..S_59")
    kk = {}
    for i in range(1, 60):
        p = os.path.join(KKW, f"S_{i}.txt")
        kk[i] = np.loadtxt(p, dtype=np.int64)
    for i in (1, 2, 24, 25, 59):
        gram_report(f"S_{i}", kk[i])
    sizes = [len(kk[i]) for i in range(1, 60)]
    print(f"  sizes S_1..S_59: {sizes}")
    bad = [i for i in kk if (kk[i] @ kk[i].T)[~np.eye(len(kk[i]), dtype=bool)].max() > 8 or np.any((kk[i] ** 2).sum(1) != 32)]
    print(f"  sets failing norm-32 / off-diag<=8: {bad}")
    ks = {i: rows(kk[i]) for i in kk}
    ov = [(i, j) for i in ks for j in ks if i < j and ks[i] & ks[j]]
    print(f"  overlapping pairs: {ov}; all subsets of minvects: {all(ks[i] <= c_kkw_set for i in ks)}; sum={sum(sizes)}")
    orth_relation_report("S_1 (488)", kk[1])
    # KKW Table-3 style totals
    S = sizes
    t = {26: 196560 + 2 * sum(S[:2]), 27: 196560 + 2 * sum(S[:2]) + sum(S[2:5]), 28: 196560 + 2 * sum(S[:8]),
         29: 196560 + 2 * sum(S[:8]) + sum(S[8:16]), 30: 196560 + 2 * sum(S[:24]), 31: 196560 + 2 * sum(S[:24]) + sum(S[24:51])}
    print(f"  KKW 2018 template values from these sizes: {t}")

    if args.no_configs:
        return
    print("== PackingStar 25D-31D configurations: implicit S_i / T_i decomposition")
    s496set = rows(s496)
    all_sets = {}  # (dim, i) -> frozenset
    for d in range(1, 8):
        (path,) = glob.glob(os.path.join(PS, "25D-31D_new_bounds_configurations", f"{24 + d}D_*_coordinates.npy"))
        groups, union, eqset = decompose_config(path, d, ref_hist=h496)
        for i, (S, T) in enumerate(groups):
            all_sets[(24 + d, i + 1)] = S
        print(f"      S_i equal (as sets) to 24D_496_Si.npy: {[i + 1 for i, (S, T) in enumerate(groups) if S == s496set]}")
        print(f"      union of S_i subset of KKW minvects: {union <= c_kkw_set}; equatorial subset: {eqset <= c_kkw_set}; equatorial == C \\ union: {eqset == (c_kkw_set - union)}")
    print("== Identity of S_i sets across dimensions")
    distinct = {}
    for k, S in all_sets.items():
        distinct.setdefault(S, []).append(k)
    print(f"  total S_i extracted: {len(all_sets)}; distinct as sets: {len(distinct)}; 24D_496_Si.npy among them: {frozenset(s496set) in distinct}")
    reused = {tuple(v): len(v) for v in distinct.values() if len(v) > 1}
    print(f"  sets appearing in more than one place: {reused if reused else 'none'}")
    print(f"  antipodal-closed: {sum(all(tuple(-c for c in x) in S for x in S) for S in distinct)}/{len(distinct)}")
    print("== Monomial-invariant shape counts (#(±2^8), #(∓3,±1^23), #(±4,±4)) per set")
    def shapes(S):
        c = collections.Counter(max(abs(v) for v in x) for x in S)
        return (c[2], c[3], c[4])
    inv = collections.Counter(shapes(S) for S in distinct)
    print(f"  24D_496_Si.npy: {shapes(s496set)}; KKW S_1: {shapes(ks[1])}; distinct S_i of 25D-31D: {dict(inv)}")
    print(f"  KKW S_1..S_59 shape counts: {collections.Counter(shapes(ks[i]) for i in ks)}")


if __name__ == "__main__":
    main()

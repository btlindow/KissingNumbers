#!/usr/bin/env python3
"""Exact setwise stabiliser of a vector set S inside the monomial group 2^12:M24 (T3.4 item 1).

    python python/tools/stabilizer_496.py [S.txt] [--out runs/orbits/stab496]
        [--aut data/group/xi.txt]        # first map S by this Aut (den + 24x24), i.e. compute Stab(xi S)
        [--perm-file aut_0001.u32]       # or by an index permutation (needs data/leech_min.i8)

A monomial element m = D_c P_pi (pi in M24 a coordinate permutation, c a Golay
codeword sign mask) fixes S iff m S = S.  The search is exact:

1. Invariants of pi.  The three shapes are preserved, so pi must map the
   multiset of octad supports of the (+-2^8) rows to itself, likewise the
   multiset of "3-positions" of the (-+3,+-1^23) rows and the pair supports
   of the (+-4,+-4) rows.  From these we colour the 24 coordinates and the
   coordinate pairs (number of support octads through i, through {i,j}, ...).
2. M24 is 5-transitive with a 5-point stabiliser of order 48, so pi is one of
   48 elements once the images of 5 base points are fixed.  We enumerate
   colour-consistent injective images of 5 base points (chosen in the rarest
   colour classes), reconstruct one element of M24 with those images from a
   stabiliser chain (base = the 5 points + sympy's extension, transversals
   from a strong generating set), multiply by the 48 elements of the
   pointwise stabiliser, and keep the pi that preserve the three support
   multisets.
3. Signs.  For a surviving pi take one full-support row x of S; P_pi x has
   its +-3 at pi(i); every row y of S with |y| = 3 there gives the only
   candidate c = {j : (P_pi x)_j != y_j}; keep c if it is a codeword and
   D_c P_pi S = S as a set.

Everything found is the whole stabiliser (the enumeration is exhaustive over
M24 for pi and over S for c).  The result is written as a monomial list that
tools/orbit_mis reads (`--group-file`), plus the orbits of the stabiliser on
S and on C.  Uses only kiss_ref (independent Golay/Leech) + group.m24.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from group.m24 import M24_ORDER, m24_generators, permute_mask, preserves_code  # noqa: E402
from kiss_ref.golay import golay_codewords  # noqa: E402
from kiss_ref.leech import leech_min_vectors  # noqa: E402


# ---------------------------------------------------------------------------
# small permutation helpers (perm[i] = image of i)
# ---------------------------------------------------------------------------
def compose(a, b):  # a o b (apply b first)
    return tuple(a[b[i]] for i in range(len(b)))


def inverse(a):
    r = [0] * len(a)
    for i, ai in enumerate(a):
        r[ai] = i
    return tuple(r)


IDENT = tuple(range(24))


def read_set(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if not s:
                continue
            v = [int(t) for t in s.split()]
            assert len(v) == 24, f"{path}: bad row {s}"
            rows.append(v)
    return np.array(rows, dtype=np.int64)


def row_shape(v):
    a = np.abs(v)
    if a.max() == 2:
        return 0
    if a.max() == 3:
        return 1
    return 2


def mask_of(bits):
    m = 0
    for i in bits:
        m |= 1 << int(i)
    return m


def set_key(rows):
    return frozenset(tuple(int(x) for x in r) for r in rows)


# ---------------------------------------------------------------------------
# stabiliser chain of M24 for a given base (5 points)
# ---------------------------------------------------------------------------
def stabiliser_chain(gens, base):
    """Transversals for the chain G > G_{b0} > G_{b0,b1} > ... using sympy's strong generating set."""
    from sympy.combinatorics import Permutation, PermutationGroup

    G = PermutationGroup([Permutation(list(g)) for g in gens])
    full_base, strong = G.schreier_sims_incremental(base=list(base))
    full_base = list(full_base)
    assert full_base[: len(base)] == list(base), (full_base, base)
    strong = [tuple(p.array_form) + tuple(range(len(p.array_form), 24)) for p in strong]
    chain = []
    for i, bi in enumerate(full_base):
        fixed = full_base[:i]
        level_gens = [g for g in strong if all(g[x] == x for x in fixed)]
        trans = {bi: IDENT}
        frontier = [bi]
        while frontier:
            nxt = []
            for pt in frontier:
                u = trans[pt]
                for g in level_gens:
                    q = g[pt]
                    if q not in trans:
                        trans[q] = compose(g, u)  # maps bi -> q
                        nxt.append(q)
            frontier = nxt
        chain.append(trans)
    # the pointwise stabiliser of the first len(base) base points: all products of
    # transversal elements of the deeper levels
    K = [IDENT]
    for lvl in range(len(base), len(full_base)):
        K = [compose(k, u) for k in K for u in chain[lvl].values()]
    return G, chain, full_base, K


def element_with_images(chain, base, images):
    """The element g of the group with g(base[i]) = images[i], or None."""
    h = IDENT
    hinv = IDENT
    for i, (bi, ci) in enumerate(zip(base, images)):
        t = hinv[ci]
        u = chain[i].get(t)
        if u is None:
            return None
        h = compose(h, u)
        hinv = inverse(h)
    return h


# ---------------------------------------------------------------------------
# the stabiliser search
# ---------------------------------------------------------------------------
def monomial_stabiliser(S, gens, codeword_set, verbose=True):
    n = len(S)
    shapes = np.array([row_shape(v) for v in S])
    oct_rows = np.where(shapes == 0)[0]
    tri_rows = np.where(shapes == 1)[0]
    ff_rows = np.where(shapes == 2)[0]
    oct_supp = Counter(mask_of(np.nonzero(S[r])[0]) for r in oct_rows)
    tri_pos = Counter(int(np.argmax(np.abs(S[r]))) for r in tri_rows)
    ff_supp = Counter(mask_of(np.nonzero(S[r])[0]) for r in ff_rows)
    if verbose:
        print(f"shape counts: octad={len(oct_rows)} 31={len(tri_rows)} 44={len(ff_rows)}")
        print(f"distinct octad supports: {len(oct_supp)} multiplicities {dict(Counter(oct_supp.values()))}")
        print(f"distinct 3-positions: {len(tri_pos)} multiplicities {dict(Counter(tri_pos.values()))}")
        print(f"44 supports: {[sorted(np.nonzero((m >> np.arange(24)) & 1)[0].tolist()) for m in ff_supp]}")

    # coordinate and pair colours (pi-invariant)
    n_oct = [0] * 24
    n_tri = [0] * 24
    n_ff = [0] * 24
    pair_oct = [[0] * 24 for _ in range(24)]
    for m, k in oct_supp.items():
        bits = [i for i in range(24) if (m >> i) & 1]
        for i in bits:
            n_oct[i] += k
            for j in bits:
                if i != j:
                    pair_oct[i][j] += k
    for i, k in tri_pos.items():
        n_tri[i] += k
    for m, k in ff_supp.items():
        for i in range(24):
            if (m >> i) & 1:
                n_ff[i] += k
    colour = [(n_oct[i], n_tri[i], n_ff[i]) for i in range(24)]
    # refine by the multiset of pair colours (one round of WL)
    for _ in range(3):
        colour = [(colour[i], tuple(sorted((pair_oct[i][j], colour[j]) for j in range(24) if j != i)))
                  for i in range(24)]
        ids = {c: k for k, c in enumerate(sorted(set(colour)))}
        colour = [ids[c] for c in colour]
    classes = Counter(colour)
    if verbose:
        print(f"coordinate colour classes: {len(classes)} sizes {sorted(classes.values())}")

    # base: 5 points in the rarest classes
    order = sorted(range(24), key=lambda i: (classes[colour[i]], i))
    base = order[:5]
    G, chain, full_base, K = stabiliser_chain(gens, base)
    orbit_sizes = [len(t) for t in chain]
    prod = 1
    for s in orbit_sizes:
        prod *= s
    assert G.order() == M24_ORDER and prod == M24_ORDER and orbit_sizes[:5] == [24, 23, 22, 21, 20], (orbit_sizes, prod)
    assert len(K) == 48 and len(set(K)) == 48 and all(all(k[b] == b for b in base) for k in K)
    if verbose:
        print(f"base points {base} (colour classes {[classes[colour[b]] for b in base]}), full base {full_base}, "
              f"chain orbit sizes {orbit_sizes}, product {prod}, 5-point stabiliser {len(K)}")

    # enumerate colour-consistent images
    cands = [[c for c in range(24) if colour[c] == colour[b]] for b in base]
    tried = 0
    pis = []
    t0 = time.time()

    def rec(i, images):
        nonlocal tried
        if i == 5:
            tried += 1
            h = element_with_images(chain, base, images)
            if h is None:
                return
            for k in K:
                g = compose(h, k)
                if any(oct_supp.get(permute_mask(m, g), 0) != kk for m, kk in oct_supp.items()):
                    continue
                if any(tri_pos.get(g[p], 0) != kk for p, kk in tri_pos.items()):
                    continue
                if any(ff_supp.get(permute_mask(m, g), 0) != kk for m, kk in ff_supp.items()):
                    continue
                pis.append(g)
            return
        bi = base[i]
        for c in cands[i]:
            if c in images:
                continue
            if any(pair_oct[c][images[j]] != pair_oct[bi][base[j]] for j in range(i)):
                continue
            rec(i + 1, images + [c])

    rec(0, [])
    if verbose:
        print(f"5-tuples tried: {tried} (bound {np.prod([len(c) for c in cands])}) x 48, "
              f"pi preserving all support multisets: {len(pis)}, {time.time() - t0:.1f}s")
    for g in pis:
        assert preserves_code(g)

    # signs
    Skey = set_key(S)
    x_ref = S[tri_rows[0]]
    elements = []  # (pi, c)
    for g in pis:
        ginv = inverse(g)
        X = S[:, list(ginv)]  # (P_g S)[:, g[i]] = S[:, i]
        xg = X[tri_rows[0]]
        ip = int(np.argmax(np.abs(xg)))
        seen = set()
        for r in tri_rows:
            y = S[r]
            if abs(y[ip]) != 3:
                continue
            c = mask_of(np.nonzero(xg != y)[0])
            if c in seen or c not in codeword_set:
                continue
            seen.add(c)
            sgn = np.array([-1 if (c >> j) & 1 else 1 for j in range(24)], dtype=np.int64)
            if set_key(X * sgn) == Skey:
                elements.append((g, c))
    return elements, base


# ---------------------------------------------------------------------------
# the FULL Co_0 stabiliser via the Gram matrix (not only the monomial part)
# ---------------------------------------------------------------------------
def gram_automorphisms(S, verbose=True):
    """Automorphism group of the Gram-coloured complete graph on S, exactly.

    Any g in Co_0 (indeed any orthogonal map) with gS = S permutes S preserving
    all inner products, i.e. is an automorphism of the complete graph on S with
    edge colours <x,y>. Conversely such an automorphism extends to at most one
    orthogonal map (S spans R^24). So Stab_{Co_0}(S) embeds in Aut(Gram(S)).
    Colour refinement (1-WL, vectorised) on the vertices, then
    individualisation-refinement to enumerate all automorphisms explicitly.
    Returns (list of automorphisms as index arrays, number of WL classes).
    """
    n = len(S)
    Gm = S @ S.T
    vals = sorted(set(Gm.ravel().tolist()))
    E = np.searchsorted(np.array(vals), Gm)  # edge colours 0..len(vals)-1
    K = len(vals)
    offdiag = ~np.eye(n, dtype=bool)

    def refine(col):
        col = np.asarray(col, dtype=np.int64)
        while True:
            ncol = int(col.max()) + 1
            M = E * ncol + col[None, :]
            M = np.where(offdiag, M, -1)
            M.sort(axis=1)
            sig = np.concatenate([col[:, None], M], axis=1)
            _, new = np.unique(sig, axis=0, return_inverse=True)
            new = new.ravel()
            if len(np.unique(new)) == ncol:
                return new
            col = new

    base = refine(np.zeros(n, dtype=np.int64))
    n_classes = len(np.unique(base))
    if verbose:
        sizes = Counter(np.bincount(base).tolist())
        print(f"Gram colour refinement: {n_classes} classes, sizes {dict(sorted(sizes.items()))}")

    auts = []

    def search(col_a, col_b):
        if len(np.unique(col_a)) == n:
            pos = np.empty(n, dtype=np.int64)
            pos[col_b] = np.arange(n)
            f = pos[col_a]
            if np.array_equal(E, E[np.ix_(f, f)]):
                auts.append(f)
            return
        cnt = np.bincount(col_a)
        cands = [c for c in range(len(cnt)) if cnt[c] > 1]
        c = min(cands, key=lambda k: (cnt[k], k))
        i = int(np.where(col_a == c)[0][0])
        m = int(col_a.max()) + 1
        for j in np.where(col_b == c)[0]:
            a = col_a.copy()
            a[i] = m
            b = col_b.copy()
            b[int(j)] = m
            ra, rb = refine(a), refine(b)
            if not np.array_equal(np.sort(np.bincount(ra)), np.sort(np.bincount(rb))):
                continue
            search(ra, rb)

    search(base, base)
    if verbose:
        print(f"Gram automorphisms (exact): {len(auts)}")
    return auts, n_classes


def extend_to_co0(S, auts, verbose=True):
    """For each Gram automorphism f the unique orthogonal g with g S[i] = S[f[i]].

    S spans R^24; 8 g is integral whenever g preserves the Leech lattice
    (8 Z^24 is a sublattice in the sqrt8 scaling), so g is computed in floats,
    8 g rounded, and then verified exactly: g S = S[f] and g maps all 196560
    minimal vectors bijectively onto themselves (group.m24.verify_aut).
    Returns the list of (f, num) with den = 8 for the automorphisms that extend.
    """
    from group.m24 import verify_aut

    X = S.astype(np.float64)
    # least squares on all rows (exact solution exists): g^T = argmin |X g^T - Y|
    out = []
    for f in auts:
        Y = S[f].astype(np.float64)
        gT, *_ = np.linalg.lstsq(X, Y, rcond=None)
        num8 = np.rint(8 * gT.T)
        if not np.allclose(8 * gT.T, num8, atol=1e-6):
            continue
        num = num8.astype(np.int64)
        img = S @ num.T
        if np.any(img % 8 != 0) or not np.array_equal(img // 8, S[f]):
            continue
        n_in, bij = verify_aut(num.tolist(), 8)
        if n_in == 196560 and bij:
            out.append((f, num))
    if verbose:
        print(f"Gram automorphisms extending to Leech automorphisms: {len(out)} of {len(auts)}")
    return out


# ---------------------------------------------------------------------------
# structure of the found group
# ---------------------------------------------------------------------------
def signed_perm(pi, c):
    """Monomial element as a permutation of the 48 signed coordinates (i, s) -> 2i + s."""
    p = [0] * 48
    for i in range(24):
        j = pi[i]
        flip = (c >> j) & 1
        p[2 * i] = 2 * j + flip
        p[2 * i + 1] = 2 * j + (1 - flip)
    return p


def monomial_apply(pi, c, rows):
    ginv = inverse(pi)
    X = rows[:, list(ginv)]
    sgn = np.array([-1 if (c >> j) & 1 else 1 for j in range(24)], dtype=np.int64)
    return X * sgn


def index_perm_of(C, C8_sorted_keys, order, rows):
    """Indices (in C's row order) of the given rows."""
    r8 = rows.astype(np.int8)
    k = np.ascontiguousarray(r8).view(np.dtype((np.void, 24))).ravel()
    pos = np.searchsorted(C8_sorted_keys, k)
    assert np.all(C8_sorted_keys[pos] == k), "image left C"
    return order[pos]


def orbits_of(n, perms):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in perms:
        for i in range(n):
            a, b = find(i), find(int(p[i]))
            if a != b:
                parent[a] = b
    roots = [find(i) for i in range(n)]
    ids = {}
    out = []
    for i, r in enumerate(roots):
        if r not in ids:
            ids[r] = len(out)
            out.append([])
        out[ids[r]].append(i)
    return out


def describe_group(elements):
    from sympy.combinatorics import Permutation, PermutationGroup

    perms = [Permutation(signed_perm(pi, c)) for pi, c in elements]
    G = PermutationGroup(perms)
    order = int(G.order())
    el_orders = Counter(int(p.order()) for p in perms)
    info = {
        "order": order,
        "abelian": bool(G.is_abelian),
        "element_orders": dict(sorted(el_orders.items())),
        "centre_order": int(G.center().order()),
        "derived_order": int(G.derived_subgroup().order()),
    }
    return G, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("set", nargs="?", default=os.path.join(HERE, "..", "..", "data", "S496.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "runs", "orbits", "stab496"))
    ap.add_argument("--aut", default=None, help="Aut file (den + 24x24 numerators): stabilise (aut S) instead")
    ap.add_argument("--perm-file", default=None, help="uint32[196560] index permutation: stabilise (g S) instead")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "data"))
    ap.add_argument("--no-orbits", action="store_true", help="skip the orbit computation on C")
    args = ap.parse_args(argv)

    t0 = time.time()
    S = read_set(args.set)
    label = os.path.basename(args.set)
    if args.aut:
        num, den = [], None
        with open(args.aut) as f:
            for line in f:
                s = line.split("#", 1)[0].split()
                if not s:
                    continue
                if s[0] == "den":
                    den = int(s[1])
                else:
                    num.append([int(t) for t in s])
        M = np.array(num, dtype=np.int64)
        img = S @ M.T
        assert np.all(img % den == 0)
        S = img // den
        label += f" mapped by {os.path.basename(args.aut)}"
    if args.perm_file:
        C8 = np.fromfile(os.path.join(args.data, "leech_min.i8"), dtype=np.int8).reshape(-1, 24)
        g = np.fromfile(args.perm_file, dtype=np.uint32)
        keys = np.ascontiguousarray(C8).view(np.dtype((np.void, 24))).ravel()
        k = np.ascontiguousarray(S.astype(np.int8)).view(np.dtype((np.void, 24))).ravel()
        pos = np.searchsorted(keys, k)
        assert np.all(keys[pos] == k)
        S = C8[g[pos]].astype(np.int64)
        label += f" mapped by {os.path.basename(args.perm_file)}"
    assert np.all((S * S).sum(1) == 32)
    n = len(S)
    print(f"S = {label}, |S| = {n}")

    codeword_set = set(golay_codewords())
    gens = [tuple(g) for g in m24_generators()]
    elements, base = monomial_stabiliser(S, gens, codeword_set)
    print(f"monomial stabiliser: {len(elements)} elements")
    G, info = describe_group(elements)
    assert info["order"] == len(elements), (info, len(elements))
    print(f"structure: {info}")
    neg_present = any(pi == IDENT and c == (1 << 24) - 1 for pi, c in elements)
    print(f"-I in stabiliser: {neg_present}")
    pis = sorted(set(pi for pi, _ in elements))
    print(f"distinct coordinate permutations (image in M24): {len(pis)}; sign kernel order: {len(elements) // len(pis)}")
    cycle_types = Counter(tuple(sorted(len(cyc) for cyc in
                                       __import__('sympy').combinatorics.Permutation(list(pi)).cyclic_form))
                          for pi in pis)
    print(f"cycle types of the M24 image (non-trivial cycles): {dict(cycle_types)}")

    # orbits on S
    Skey = {tuple(int(x) for x in r): i for i, r in enumerate(S)}
    perms_S = []
    for pi, c in elements:
        img = monomial_apply(pi, c, S)
        perms_S.append([Skey[tuple(int(x) for x in r)] for r in img])
    orb_S = orbits_of(n, perms_S)
    print(f"orbits on S: {len(orb_S)}, sizes {dict(sorted(Counter(len(o) for o in orb_S).items()))}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "stabiliser_monomials.txt"), "w") as f:
        f.write(f"# monomial stabiliser of {label} in 2^12:M24 — all {len(elements)} elements\n")
        f.write("# format: 24 images of coordinates 0..23, then the sign mask (Golay codeword, decimal)\n")
        f.write(f"# structure: {info}\n")
        for pi, c in elements:
            f.write(" ".join(str(v) for v in pi) + f" {c}\n")
    # generators: greedy (add elements while the group grows)
    from sympy.combinatorics import Permutation, PermutationGroup

    gsel = []
    H = None
    for pi, c in sorted(elements, key=lambda e: -Permutation(signed_perm(*e)).order()):
        p = Permutation(signed_perm(pi, c))
        if H is None or not H.contains(p):
            gsel.append((pi, c))
            H = PermutationGroup([Permutation(signed_perm(*e)) for e in gsel])
            if int(H.order()) == len(elements):
                break
    with open(os.path.join(args.out, "stabiliser_generators.txt"), "w") as f:
        f.write(f"# generators of the monomial stabiliser of {label} (order {len(elements)})\n")
        f.write("# format: 24 images of coordinates 0..23, then the sign mask (Golay codeword, decimal)\n")
        for pi, c in gsel:
            f.write(" ".join(str(v) for v in pi) + f" {c}\n")
    print(f"generators written: {len(gsel)}")

    orb_C_sizes = None
    if not args.no_orbits and len(elements) > 1:
        C = leech_min_vectors().astype(np.int64)
        C8 = C.astype(np.int8)
        keys = np.ascontiguousarray(C8).view(np.dtype((np.void, 24))).ravel()
        order = np.argsort(keys)
        sk = keys[order]
        perms_C = []
        for pi, c in gsel:
            perms_C.append(index_perm_of(C, sk, order, monomial_apply(pi, c, C)))
        orb_C = orbits_of(len(C), perms_C)
        orb_C_sizes = Counter(len(o) for o in orb_C)
        print(f"orbits on C: {len(orb_C)}, sizes {dict(sorted(orb_C_sizes.items()))}")
        # S must be a union of orbits
        Sidx = set(index_perm_of(C, sk, order, S).tolist())
        union_ok = all(set(o) <= Sidx or not (set(o) & Sidx) for o in orb_C)
        print(f"S is a union of stabiliser orbits on C: {union_ok}")
        assert union_ok

    auts, n_classes = gram_automorphisms(S)
    neg_map = np.array([Skey[tuple(int(-x) for x in r)] for r in S])
    has_neg = any(np.array_equal(neg_map, f) for f in auts)
    print(f"|Aut(Gram(S))| = {len(auts)}; x->-x among them: {has_neg}")
    ext = extend_to_co0(S, auts)
    co0_order = len(ext)
    from sympy.combinatorics import Permutation, PermutationGroup

    Gs = PermutationGroup([Permutation(f.tolist()) for f, _ in ext]) if ext else None
    co0_info = {}
    if Gs is not None:
        assert int(Gs.order()) == co0_order, "extending automorphisms do not form a group?"
        co0_info = {
            "order": co0_order,
            "abelian": bool(Gs.is_abelian),
            "element_orders": dict(sorted(Counter(int(Permutation(f.tolist()).order()) for f, _ in ext).items())),
            "centre_order": int(Gs.center().order()),
            "derived_order": int(Gs.derived_subgroup().order()),
            "orbits_on_S": sorted(Counter(len(o) for o in Gs.orbits()).items()),
        }
    print(f"full Co_0 stabiliser: |Stab_Co0(S)| = {co0_order}; structure {co0_info}")
    # export: generators (greedy) as Aut files (den 8 + 24 rows), all elements listed in one file
    for old in [f for f in os.listdir(args.out) if f.startswith("co0_stab_")]:
        os.remove(os.path.join(args.out, old))
    gsel = []
    Hc = None
    for f, num in sorted(ext, key=lambda e: -int(Permutation(e[0].tolist()).order())):
        pf = Permutation(f.tolist())
        if Hc is None or not Hc.contains(pf):
            gsel.append((f, num))
            Hc = PermutationGroup([Permutation(e[0].tolist()) for e in gsel])
            if int(Hc.order()) == co0_order:
                break
    for k, (f, num) in enumerate(gsel):
        with open(os.path.join(args.out, f"co0_stab_gen{k}.txt"), "w") as fh:
            fh.write(f"# generator {k} of Stab_Co0({label}) (order {co0_order}); element order {int(Permutation(f.tolist()).order())}\n")
            fh.write("den 8\n")
            for row in num:
                fh.write(" ".join(str(int(v)) for v in row) + "\n")
    with open(os.path.join(args.out, "co0_stabiliser_elements.txt"), "w") as fh:
        fh.write(f"# all {co0_order} elements of Stab_Co0({label}); each block: element index, then 24 rows of numerators, den 8\n")
        for k, (f, num) in enumerate(ext):
            fh.write(f"# element {k}, order {int(Permutation(f.tolist()).order())}\n")
            for row in num:
                fh.write(" ".join(str(int(v)) for v in row) + "\n")
    print(f"Co_0 stabiliser generators written: {len(gsel)}")

    print(f"RESULT ok=1 set={label!r} size={n} stab_order={len(elements)} m24_image={len(pis)} "
          f"sign_kernel={len(elements) // len(pis)} abelian={int(info['abelian'])} "
          f"orbits_on_S={len(orb_S)} orbits_on_C={sum(orb_C_sizes.values()) if orb_C_sizes else -1} "
          f"gram_aut={len(auts)} gram_classes={n_classes} co0_stab_order={co0_order} seconds={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

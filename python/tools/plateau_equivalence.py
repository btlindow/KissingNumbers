#!/usr/bin/env python3
"""T3.4b — are the 64 plateau 496s Co_0-equivalent?  (exact)

    .venv/bin/python python/tools/plateau_equivalence.py [--atoms runs/ls_search/plateau_atoms_496.json]
        [--out runs/plateau_equiv] [--auts-out data/group/plateau_auts] [--stab 15,16,48] [--jobs 12]
        [--quick] [--no-orbits] [--rep-pairs] [--no-auts-out]

T3.2b found six commuting, pairwise disjoint, negation-closed plateau moves on the
record 496 (atoms 0..3: (12,12); atoms 4,5: (80,80)); the 2^6 = 64 unions give 64
distinct independent 496-sets S_T (T a subset of the atoms, S_0 = the 496 itself).
This tool decides, exactly, for every T whether S_T = g S for some g in Co_0:

1. Gram-graph isomorphism.  A g in Co_0 with g S = S_T induces a bijection
   f : S -> S_T preserving all inner products (an isomorphism of the complete
   graphs with edge colours <x,y> in {-32,-8,0,8}).  Such f are enumerated
   exactly by colour refinement (1-WL) on the DISJOINT UNION of the two coloured
   graphs (so the class labels are shared) plus individualisation-refinement;
   a class whose two sides have different sizes prunes.  (S_T = S gives
   Aut(Gram(S)), T3.4's 64.)
2. Extension.  S spans R^24, so each f extends to at most one orthogonal map
   g_f (least squares on all 496 rows); 8 g_f is rounded to integers (8 Z^24 is a
   sublattice in the sqrt8 scaling, so 8 g is integral for g in Co_0), g_f S = S_T
   is re-verified in exact integers, and g_f is applied to all 196560 minimal
   vectors: g_f in Co_0 iff the images are minimal vectors and the map is a
   bijection (same test as T3.4 / group.m24.verify_aut).
   S_T is Co_0-equivalent to S iff some isomorphism extends; the extending ones
   then form a coset of Stab_Co0(S) (so their number is 8 or 0).
3. Pre-filters, used only where they are sound.  (a) the 1-WL invariant of each
   Gram graph on its own -- necessary for isometry, hence for Co_0-equivalence;
   (b) the Co_0-INVARIANT fingerprint of T3.1b, extended: histogram of
   tight[v] = #{s in S : <v,s> = 16} over v outside S, histogram of the full
   profile (#{<v,s>=16}, #{<v,s>=8}), Gram histogram, Gram histogram of the
   minimum-tightness vertices.  The SHAPE counts are invariants of the monomial
   subgroup only (xi mixes shapes, T4.1) and are therefore reported but NEVER
   used to certify Co_0-inequivalence.
4. Orbit structure: union-find over the 64 sets.  Co_0-equivalence forces an
   equal fingerprint, so it is enough to test every pair inside one fingerprint
   class; the class count that comes out is then complete.  (An optional cheap
   path first asks whether the isometries in K = {T : S_T isometric to S} are
   realised by orthogonal maps that translate the whole 6-cube -- true for
   T ^ 15, false for T ^ 22 and T ^ 35, so the exhaustive path is taken.)
   Stab_Co0(S_T) is recomputed from scratch for a few T (--stab).
5. Positive controls: g S for g in {xi, xi^2, gamma, xi.gamma, -I} must come out
   EQUIVALENT with exactly 8 extending isomorphisms, or the run aborts.
6. The "64 = 64" coincidence: does Aut(Gram(S)) (order 64, 8 in Co_0) know the
   atoms?  Orbits / block structure of Aut(Gram(S)) on the atom partition, and
   whether the 56 non-Co_0 Gram automorphisms (orthogonal maps g_a with
   g_a S = S but g_a Lambda != Lambda) map any S_T into the minimal vectors.

The isometry classes across fingerprint classes are finished off by the companion
script runs/plateau_equiv/isometry_closure.py.

Uses only kiss_ref (Golay/Leech) + group.m24 + numpy/sympy; independent of the C++.
Writes runs/plateau_equiv/{log.txt, results.json}.  If a plateau move HAD turned out
to be realised by a Co_0 element, the explicit matrices would be written to
runs/plateau_equiv/g_T<T>.txt and data/group/plateau_auts/g_atom<k>.txt; none is, so
neither file is produced.

RESULT (2026-08-27): 0 of the 63 non-trivial S_T is a Co_0-image of S, and the 64 sets
are pairwise Co_0-inequivalent -- 64 classes.  They form 8 isometry classes of 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from kiss_ref.leech import leech_min_vectors  # noqa: E402

LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Leech:
    """The 196560 minimal vectors, with a hashed index (row -> position)."""

    def __init__(self, data_dir):
        p = os.path.join(data_dir, "leech_min.i8")
        if os.path.exists(p):
            C8 = np.fromfile(p, dtype=np.int8).reshape(-1, 24)
            assert np.array_equal(C8, leech_min_vectors()), "data/leech_min.i8 is not in the reference order"
        else:
            C8 = leech_min_vectors()
        self.C8 = np.ascontiguousarray(C8)
        self.C = self.C8.astype(np.int64)
        self.keys = self.C8.view(np.dtype((np.void, 24))).ravel()
        self.order = np.argsort(self.keys)
        self.sk = self.keys[self.order]
        neg = self.index_of(-self.C, strict=True)
        self.neg = neg

    def index_of(self, rows, strict=True):
        """Positions in C of integer rows; -1 where the row is not a minimal vector."""
        rows = np.asarray(rows, dtype=np.int64)
        small = np.all(np.abs(rows) <= 127, axis=1)
        r8 = np.ascontiguousarray(np.clip(rows, -127, 127).astype(np.int8))
        k = r8.view(np.dtype((np.void, 24))).ravel()
        pos = np.minimum(np.searchsorted(self.sk, k), len(self.sk) - 1)
        hit = (self.sk[pos] == k) & small
        idx = np.where(hit, self.order[pos], -1)
        if strict:
            assert np.all(hit), "row is not a minimal vector"
        return idx

    def verify_aut(self, num, den=8):
        """(number of minimal vectors mapped to minimal vectors, bijective?) for the map num/den."""
        M = np.asarray(num, dtype=np.int64)
        img = self.C @ M.T
        ok_int = np.all(img % den == 0, axis=1)
        idx = self.index_of(img // den, strict=False)
        idx = np.where(ok_int, idx, -1)
        n_in = int((idx >= 0).sum())
        bij = n_in == len(self.C) and len(np.unique(idx)) == len(self.C)
        return n_in, bij


def read_set(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if s:
                v = [int(t) for t in s.split()]
                assert len(v) == 24
                rows.append(v)
    return np.array(rows, dtype=np.int64)


def read_aut(path):
    num, den = [], None
    with open(path) as f:
        for line in f:
            s = line.split("#", 1)[0].split()
            if not s:
                continue
            if s[0] == "den":
                den = int(s[1])
            else:
                num.append([int(t) for t in s])
    return np.array(num, dtype=np.int64), den


def aut_hash(num):
    """sha256 of the canonical text of 8g (24 rows of 24 integers)."""
    txt = "\n".join(" ".join(str(int(v)) for v in row) for row in num) + "\n"
    return hashlib.sha256(txt.encode()).hexdigest()


def write_aut(path, num, header_lines):
    with open(path, "w") as f:
        for h in header_lines:
            f.write("# " + h + "\n")
        f.write("den 8\n")
        for row in num:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


# ---------------------------------------------------------------------------
# Gram-graph isomorphisms (joint colour refinement + individualisation)
# ---------------------------------------------------------------------------
class GramIso:
    """All isomorphisms f : A -> B of the Gram-coloured complete graphs (f[i] = image of A_i in B)."""

    def __init__(self, A, B):
        self.n = n = len(A)
        assert len(B) == n
        GA = A @ A.T
        GB = B @ B.T
        vals = sorted(set(GA.ravel().tolist()) | set(GB.ravel().tolist()))
        v = np.array(vals)
        K = len(vals)
        E = np.full((2 * n, 2 * n), K, dtype=np.int64)  # cross colour K
        E[:n, :n] = np.searchsorted(v, GA)
        E[n:, n:] = np.searchsorted(v, GB)
        self.E = E
        self.EA, self.EB = E[:n, :n], E[n:, n:]
        self.offdiag = ~np.eye(2 * n, dtype=bool)
        self.side = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
        self.n_refine = 0

    def refine(self, col):
        col = np.asarray(col, dtype=np.int64)
        E, offdiag = self.E, self.offdiag
        while True:
            self.n_refine += 1
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

    def balanced(self, col):
        """Every class has as many A-vertices as B-vertices."""
        cnt = np.bincount(col * 2 + self.side, minlength=2 * (int(col.max()) + 1)).reshape(-1, 2)
        return bool(np.all(cnt[:, 0] == cnt[:, 1]))

    def run(self, limit=None, verbose=False):
        n = self.n
        base = self.refine(np.zeros(2 * n, dtype=np.int64))
        self.base = base
        self.n_classes = len(np.unique(base))
        self.class_sizes_A = dict(sorted(Counter(np.bincount(base[:n]).tolist()).items()))
        out = []
        nodes = [0]
        if not self.balanced(base):
            if verbose:
                log(f"    refinement unbalanced at the root ({self.n_classes} classes): no isomorphism")
            return out, nodes[0]

        def search(col):
            nodes[0] += 1
            if limit is not None and len(out) >= limit:
                return
            cnt = np.bincount(col)
            cands = [c for c in range(len(cnt)) if cnt[c] > 2]
            if not cands:
                # every class = {one A vertex, one B vertex}
                posB = np.full(len(cnt), -1, dtype=np.int64)
                posB[col[n:]] = np.arange(n)
                f = posB[col[:n]]
                if np.array_equal(self.EA, self.EB[np.ix_(f, f)]):
                    out.append(f)
                return
            c = min(cands, key=lambda k: (cnt[k], k))
            i = int(np.where(col[:n] == c)[0][0])
            m = int(col.max()) + 1
            for j in np.where(col[n:] == c)[0]:
                a = col.copy()
                a[i] = m
                a[n + int(j)] = m
                ra = self.refine(a)
                if not self.balanced(ra):
                    continue
                search(ra)
                if limit is not None and len(out) >= limit:
                    return

        search(base)
        return out, nodes[0]


def gram_isomorphisms(A, B, limit=None, verbose=False):
    gi = GramIso(A, B)
    isos, nodes = gi.run(limit=limit, verbose=verbose)
    return isos, gi


# ---------------------------------------------------------------------------
# extension of an isomorphism to the orthogonal map, and the Co_0 test
# ---------------------------------------------------------------------------
def orthogonal_extension(A, B, f):
    """8 g (integer 24x24) with g A_i = B_f(i) for all i, or None if 8 g is not integral."""
    X = A.astype(np.float64)
    Y = B[f].astype(np.float64)
    gT, *_ = np.linalg.lstsq(X, Y, rcond=None)
    G8 = 8 * gT.T
    num = np.rint(G8)
    if not np.allclose(G8, num, atol=1e-6):
        return None
    num = num.astype(np.int64)
    img = A @ num.T
    if np.any(img % 8 != 0) or not np.array_equal(img // 8, B[f]):
        return None
    # orthogonality: (8g)(8g)^T = 64 I
    if not np.array_equal(num @ num.T, 64 * np.eye(24, dtype=np.int64)):
        return None
    return num


def test_extension(L, A, B, f):
    """-> (num or None, status) where status in {'not_integral', 'not_leech', 'co0'}."""
    num = orthogonal_extension(A, B, f)
    if num is None:
        return None, "not_integral"
    n_in, bij = L.verify_aut(num, 8)
    if n_in == len(L.C) and bij:
        return num, "co0"
    return num, f"not_leech({n_in})"


def equivalence(L, A, B, limit_iso=None, verbose=False):
    """Are A and B Co_0-equivalent?  Enumerates the Gram isomorphisms and tests each."""
    t0 = time.time()
    isos, gi = gram_isomorphisms(A, B, limit=limit_iso, verbose=verbose)
    t_iso = time.time() - t0
    stats = Counter()
    ext = []
    for f in isos:
        num, st = test_extension(L, A, B, f)
        stats[st.split("(")[0]] += 1
        if st == "co0":
            ext.append((f, num))
    first = ext[0] if ext else None
    res = {
        "n_iso": len(isos),
        "gram_classes": gi.n_classes,
        "class_sizes": gi.class_sizes_A,
        "n_ext": stats["co0"],
        "ext_stats": dict(stats),
        "t_iso": round(t_iso, 2),
        "t_total": round(time.time() - t0, 2),
    }
    return res, first, isos, ext


# ---------------------------------------------------------------------------
# a cheap isomorphism invariant of ONE Gram-coloured graph (stable 1-WL colouring)
# ---------------------------------------------------------------------------
WL_VALS = np.array([-32, -8, 0, 8])


def wl_invariant(V):
    """(#stable classes, sorted class sizes, sha256 of the WL quotient); equal for isomorphic Gram graphs."""
    G = V @ V.T
    n = len(G)
    E = np.searchsorted(WL_VALS, G)
    offdiag = ~np.eye(n, dtype=bool)
    col = np.zeros(n, dtype=np.int64)
    while True:
        ncol = int(col.max()) + 1
        M = np.where(offdiag, E * (ncol + 1) + col[None, :], -1)
        M.sort(axis=1)
        sig = np.concatenate([col[:, None], M], axis=1)
        _, new = np.unique(sig, axis=0, return_inverse=True)
        new = new.ravel()
        if len(np.unique(new)) == ncol:
            break
        col = new
    nc = int(col.max()) + 1
    sizes = np.bincount(col, minlength=nc)
    quot = [[int(sizes[c])] + [np.bincount(E[int(np.where(col == c)[0][0])][col == d], minlength=len(WL_VALS) + 1).tolist()
                               for d in range(nc)] for c in range(nc)]
    return nc, tuple(sorted(sizes.tolist())), hashlib.sha256(json.dumps(quot).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# T3.1b's Co_0-INVARIANT fingerprint of a set (a necessary condition for Co_0-equivalence
# that is NOT an isometry invariant: it looks at the set inside the Leech lattice)
# ---------------------------------------------------------------------------
def co0_fingerprint(L, idx):
    """Co_0-invariant data of a set S of minimal vectors, plus (separately) monomial-only data.

    Co_0-invariant part ("hash"): every entry is defined purely from inner products between
    Leech minimal vectors, so g in Co_0 carries it unchanged --
      tight   : histogram of tight[v] = #{s in S : <v,s> = 16} over the v outside S
                (T3.1b's invariant),
      prof    : histogram of the full inner-product profile (#{<v,s> = 16}, #{<v,s> = 8}) over
                the v outside S -- a strict refinement of `tight` (S is antipodal, so the
                counts at -16, -8 mirror those at 16, 8 and the profile is determined by the pair),
      gram    : Gram histogram of S,
      mingram : Gram histogram of the minimum-tightness vertices, min_tight, n_min_tight.
    The SHAPE counts (octad / (-+3,+-1^23) / (+-4,+-4)) are invariants of the monomial subgroup
    2^12:M24 ONLY -- Conway's xi mixes the three shapes (T4.1) -- so they are reported as
    "hash_mono" and must never be used to certify Co_0-INequivalence."""
    n = len(idx)
    V = L.C8[idx].astype(np.float32)
    Cf = L.C8.astype(np.float32)
    tight = np.empty(len(Cf), dtype=np.int32)
    n8 = np.empty(len(Cf), dtype=np.int32)
    step = 16384
    for a in range(0, len(Cf), step):
        P = np.rint(Cf[a:a + step] @ V.T)
        tight[a:a + step] = (P == 16).sum(axis=1)
        n8[a:a + step] = (P == 8).sum(axis=1)
    inS = np.zeros(len(Cf), dtype=bool)
    inS[idx] = True
    out = tight[~inS]
    tmin = int(out.min())
    mv = np.where((~inS) & (tight == tmin))[0]
    W = L.C[mv]
    M = W @ W.T
    G = L.C[idx] @ L.C[idx].T
    iu = np.triu_indices(n, 1)
    im = np.triu_indices(len(mv), 1)
    nz = (L.C8[idx] != 0).sum(axis=1)
    fp = {
        "tight": dict(sorted(Counter(out.tolist()).items())),
        "prof": dict(sorted(Counter(zip(out.tolist(), n8[~inS].tolist())).items())),
        "gram": dict(sorted(Counter(G[iu].tolist()).items())),
        "min_tight": tmin,
        "n_min_tight": int(len(mv)),
        "mingram": dict(sorted(Counter(M[im].tolist()).items())),
    }
    fp = {k: (v if k != "prof" else {f"{a},{b}": c for (a, b), c in v.items()}) for k, v in fp.items()}
    fp["hash"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    fp["shapes"] = [int((nz == 8).sum()), int((nz == 24).sum()), int((nz == 2).sum())]
    fp["hash_mono"] = hashlib.sha256((fp["hash"] + json.dumps(fp["shapes"])).encode()).hexdigest()[:16]
    return fp


def fp_text(fp):
    return ("tight=" + " ".join(f"{k}:{v}" for k, v in fp["tight"].items())
            + "|gram=" + ",".join(f"{k}:{v}" for k, v in fp["gram"].items())
            + f"|mingram({fp['min_tight']})=" + ",".join(f"{k}:{v}" for k, v in fp["mingram"].items())
            + f"|profiles={len(fp['prof'])}"
            + "|shapes=" + "/".join(str(x) for x in fp["shapes"]))


# ---------------------------------------------------------------------------
# parallel pairwise jobs (fork: the workers inherit _W, nothing is pickled in)
# ---------------------------------------------------------------------------
_W = {}


def pmap(fn, jobs, nproc):
    jobs = list(jobs)
    if nproc <= 1 or len(jobs) <= 1:
        for j in jobs:
            yield fn(j)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(nproc, len(jobs))) as pool:
        for r in pool.imap(fn, jobs):
            yield r


def _job_fp(T):
    return T, co0_fingerprint(_W["L"], _W["sets"][T])


def _job_isomap(T):
    """Every orthogonal map g of R^24 with g S = S_T (8g integral), as nested lists."""
    L, sets = _W["L"], _W["sets"]
    A, B = L.C[sets[0]], L.C[sets[T]]
    isos, gi = gram_isomorphisms(A, B)
    out = []
    for f in isos:
        num = orthogonal_extension(A, B, f)
        if num is not None:
            out.append(num.tolist())
    return T, out, len(isos)


def _job_equiv(job):
    """Full Co_0-equivalence test S_a -> S_b; returns a JSON-able record."""
    a, b = job
    L, sets, S_idx = _W["L"], _W["sets"], _W["S_idx"]
    ia, ib = sets[a], sets[b]
    A, B = L.C[ia], L.C[ib]
    res, first, isos, ext = equivalence(L, A, B)
    out = {"a": a, "b": b, "overlap": int(np.isin(ib, ia).sum()), "equivalent": first is not None, **res}
    if first is not None:
        f, num = first
        out["g_num"] = num.tolist()
        out["g_hash"] = aut_hash(num)
        out["g_order"] = mat_order(num)
        out["g_trace8"] = int(np.trace(num))
        out["g_fixes"] = int((B[f] == A).all(axis=1).sum())
        out["g_maps_A_into_S"] = int(np.isin(ib[f], S_idx).sum())
        common = np.isin(ia, ib)
        out["g_fixes_common_pointwise"] = bool(np.all(ib[f][common] == ia[common]))
        rem = ~common
        out["g_maps_removed_onto_added"] = bool(set(ib[f][rem].tolist()) == set(ib[~np.isin(ib, ia)].tolist()))
        out["ext_hashes"] = sorted(aut_hash(num2) for _, num2 in ext)
    return out


# ---------------------------------------------------------------------------
# group helpers
# ---------------------------------------------------------------------------
def perm_group_info(perms):
    from sympy.combinatorics import Permutation, PermutationGroup

    ps = [Permutation([int(x) for x in p]) for p in perms]
    G = PermutationGroup(ps)
    return G, {
        "order": int(G.order()),
        "abelian": bool(G.is_abelian),
        "element_orders": dict(sorted(Counter(int(p.order()) for p in ps).items())),
        "exponent": int(G.exponent()) if hasattr(G, "exponent") else None,
        "centre_order": int(G.center().order()),
        "derived_order": int(G.derived_subgroup().order()),
        "orbit_sizes": dict(sorted(Counter(len(o) for o in G.orbits()).items())),
    }


def mat_order(num, den=8, cap=200):
    I = np.eye(24, dtype=np.int64) * den
    P = num.copy()
    for k in range(1, cap + 1):
        if np.array_equal(P, I):
            return k
        P = (P @ num) // den
    return -1


# ---------------------------------------------------------------------------
def main(argv=None):
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default=os.path.join(ROOT, "data", "S496.txt"))
    ap.add_argument("--atoms", default=None,
                    help="plateau atoms JSON (default: data/plateau_atoms_496.json, "
                         "falling back to runs/ls_search/plateau_atoms_496.json)")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "plateau_equiv"))
    ap.add_argument("--auts-out", default=os.path.join(ROOT, "data", "group", "plateau_auts"))
    ap.add_argument("--stab", default="1,48,63", help="masks T whose full Stab_Co0(S_T) is computed (comma list)")
    ap.add_argument("--quick", action="store_true", help="only the 6 generators + a few products")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1), help="worker processes for the pairwise tests")
    ap.add_argument("--no-orbits", action="store_true", help="skip the pairwise orbit-structure phase")
    ap.add_argument("--rep-pairs", action="store_true",
                    help="also run all same-WL pairs of class representatives (only needed to pin down the "
                         "isometry classes when the coset structure could not be verified; O(#reps^2) searches)")
    ap.add_argument("--no-auts-out", action="store_true")
    args = ap.parse_args(argv)
    if args.atoms is None:
        # the committed copy first; the original T3.2b run artefact as fallback
        args.atoms = os.path.join(ROOT, "data", "plateau_atoms_496.json")
        fallback = os.path.join(ROOT, "runs", "ls_search", "plateau_atoms_496.json")
        if not os.path.exists(args.atoms) and os.path.exists(fallback):
            args.atoms = fallback

    os.makedirs(args.out, exist_ok=True)
    LOG = open(os.path.join(args.out, "log.txt"), "w")
    t_start = time.time()
    L = Leech(args.data)
    log(f"Leech minimal vectors: {len(L.C)}")

    S_rows = read_set(args.set)
    S_idx = np.sort(L.index_of(S_rows))
    n = len(S_idx)
    assert n == 496 and len(set(S_idx.tolist())) == n
    with open(args.atoms) as f:
        atoms = json.load(f)["atoms"]
    log(f"S = {os.path.relpath(args.set, ROOT)}, |S| = {n}; atoms: {[a['size'] for a in atoms]}")
    R = []
    A = []
    for k, a in enumerate(atoms):
        r = S_idx[np.array(a["remove_S_pos"])]
        assert np.array_equal(r, np.array(a["remove_vertices"])), f"atom {k}: positions disagree with vertices"
        R.append(np.array(a["remove_vertices"], dtype=np.int64))
        A.append(np.array(a["add_vertices"], dtype=np.int64))
        assert set(L.neg[R[k]].tolist()) == set(R[k].tolist()) and set(L.neg[A[k]].tolist()) == set(A[k].tolist())
    for k in range(len(atoms)):
        for l in range(k + 1, len(atoms)):
            assert not (set(R[k]) & set(R[l])) and not (set(A[k]) & set(A[l]))
    nat = len(atoms)
    NT = 1 << nat

    def set_T(T):
        keep = np.ones(n, dtype=bool)
        add = []
        for k in range(nat):
            if (T >> k) & 1:
                keep &= ~np.isin(S_idx, R[k])
                add.append(A[k])
        idx = np.sort(np.concatenate([S_idx[keep]] + add)) if add else S_idx.copy()
        return idx

    sets = [set_T(T) for T in range(NT)]
    setkeys = {tuple(s.tolist()): T for T, s in enumerate(sets)}
    assert len(setkeys) == NT
    # exact independence of all 64 (max off-diagonal inner product 8) and Gram histogram
    ref_hist = None
    for T, idx in enumerate(sets):
        V = L.C[idx]
        G = V @ V.T
        assert len(idx) == n and np.all(np.diag(G) == 32)
        off = G[~np.eye(n, dtype=bool)]
        assert off.max() <= 8, f"S_{T} not independent"
        h = tuple(sorted(Counter(off.tolist()).items()))
        if ref_hist is None:
            ref_hist = h
        assert h == ref_hist, f"S_{T}: Gram histogram differs"
    log(f"all {NT} sets are independent 496-sets with Gram histogram {dict(ref_hist)}")
    S = L.C[S_idx]
    _W["L"], _W["sets"], _W["S_idx"] = L, sets, S_idx

    # ---- cheap isomorphism invariant: 1-WL on each Gram graph separately -------------------
    log("\n== 1-WL invariant of the 64 Gram graphs (a necessary condition for isometry)")
    inv = {}
    ranks = set()
    for T in range(NT):
        V = L.C[sets[T]]
        inv[T] = wl_invariant(V)
        ranks.add(int(np.linalg.matrix_rank(V)))
    buckets = {}
    for T in range(NT):
        buckets.setdefault(inv[T], []).append(T)
    log(f"  every S_T spans R^24: {ranks == {24}} (ranks {sorted(ranks)})")
    log(f"  {len(buckets)} distinct WL invariants among the {NT} sets:")
    for iv, ts in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        pop4 = sorted({bin(T & 15).count("1") % 2 for T in ts})
        log(f"    classes {iv[0]}, sizes {dict(sorted(Counter(iv[1]).items()))}, quotient {iv[2]}: "
            f"{len(ts)} sets, parity of |T n {{0,1,2,3}}| = {pop4}; T = {ts[:12]}{'...' if len(ts) > 12 else ''}")

    # ---- T3.1b's Co_0-INVARIANT fingerprint (tightness histogram inside the Leech lattice) ----
    log("\n== Co_0-invariant fingerprints of the 64 sets (T3.1b's invariant)")
    fps = {}
    t_fp = time.time()
    for T, fp in pmap(_job_fp, list(range(NT)), args.jobs):
        fps[T] = fp
    fp_classes = {}
    for T in range(NT):
        fp_classes.setdefault(fps[T]["hash"], []).append(T)
    mono_classes = {}
    for T in range(NT):
        mono_classes.setdefault(fps[T]["hash_mono"], []).append(T)
    log(f"  computed in {time.time() - t_fp:.1f}s; {len(fp_classes)} distinct Co_0-invariant fingerprints among the "
        f"{NT} sets => at least {len(fp_classes)} Co_0-inequivalence classes")
    log(f"  refined by the (monomial-only!) shape counts: {len(mono_classes)} classes, sizes "
        f"{sorted((len(v) for v in mono_classes.values()), reverse=True)} -- a lower bound for the MONOMIAL group "
        f"2^12:M24 only, NOT for Co_0 (Conway's xi mixes shapes, T4.1)")
    log(f"  shape counts occurring: {sorted({tuple(fps[T]['shapes']) for T in range(NT)})}")
    const = {k: (len({json.dumps(fps[T][k], sort_keys=True) for T in range(NT)}) == 1)
             for k in ("gram", "mingram", "shapes", "min_tight", "n_min_tight", "tight", "prof")}
    log(f"  constant across all 64: {const}")
    fp_order = sorted(fp_classes, key=lambda h: (-len(fp_classes[h]), min(fp_classes[h])))
    fp_name = {h: i for i, h in enumerate(fp_order)}
    for h in fp_order:
        ts = fp_classes[h]
        log(f"    fp#{fp_name[h]} {h}: {len(ts):2d} sets, T = {ts}; shape counts "
            f"{sorted({tuple(fps[T]['shapes']) for T in ts})}")
        log(f"      {fp_text(fps[ts[0]])}")

    # ---- cross-check: T3.1b's exhaustive k<=12 plateau walk (independent C++ tool) ------------
    ref_dir = os.path.join(ROOT, "runs", "plateau", "s496_k12", "sets")
    xcheck = {}
    if os.path.isdir(ref_dir):
        log("\n== cross-check with T3.1b (runs/plateau/s496_k12)")
        for fn in sorted(os.listdir(ref_dir)):
            idx = np.sort(L.index_of(read_set(os.path.join(ref_dir, fn))))
            T = setkeys.get(tuple(idx.tolist()), -1)
            xcheck[fn] = T
            log(f"  {fn}: |set| {len(idx)}, overlap with S {int(np.isin(idx, S_idx).sum())} -> "
                + (f"S_T with T = {T} (atoms {[k for k in range(nat) if (T >> k) & 1]}), fp#{fp_name[fps[T]['hash']]}"
                   if T >= 0 else "NOT one of the 64 plateau sets (!)"))
        assert all(T >= 0 for T in xcheck.values()), "a T3.1b representative is not among the 64 sets"
        sub = {fps[T]["hash"] for T in range(16)}
        log(f"  fingerprints among the 16 sets T ⊆ {{atoms 0,1,2,3}} (T3.1b's 4-cube): {len(sub)}, "
            f"sizes {sorted((sum(1 for T in range(16) if fps[T]['hash'] == h) for h in sub), reverse=True)} "
            f"(T3.1b: 5 classes, 4+4+4+2+2)")
        # numeric comparison of the tightness histograms with T3.1b's summary.txt
        sm = os.path.join(ROOT, "runs", "plateau", "s496_k12", "summary.txt")
        if os.path.exists(sm):
            ref = set()
            for line in open(sm):
                if "tight=" in line and "|gram=" in line:
                    ref.add(line.split("tight=", 1)[1].split("|gram=", 1)[0].strip())
            mine = {" ".join(f"{k}:{v}" for k, v in fps[T]["tight"].items()) for T in range(16)}
            log(f"  tightness histograms of the 4-cube: {len(mine)} mine vs {len(ref)} in T3.1b's summary.txt; "
                f"identical sets of histograms: {mine == ref}")
            assert mine == ref, "tightness histograms disagree with T3.1b"
        # every 4-cube class must be a union of complementary pairs W <-> W^c (T3.1b)
        compl_ok = all(fps[T]["hash"] == fps[T ^ 15]["hash"] for T in range(16))
        log(f"  every T ⊆ {{0,1,2,3}} has the same fingerprint as its complement T^15: {compl_ok}")
        log(f"  and for the full 6-cube, fp(T) == fp(T ^ 63): {all(fps[T]['hash'] == fps[T ^ 63]['hash'] for T in range(NT))}; "
            f"fp(T) == fp(T ^ 15): {all(fps[T]['hash'] == fps[T ^ 15]['hash'] for T in range(NT))}; "
            f"fp(T) == fp(T ^ 48): {all(fps[T]['hash'] == fps[T ^ 48]['hash'] for T in range(NT))}")

    # ---- positive control: the pipeline must recognise g S for known g in Co_0 ----------------
    log("\n== positive control: g S for known elements g of Co_0")
    ctrl = []
    xi_path = os.path.join(args.data, "group", "xi.txt")
    if os.path.exists(xi_path):
        xnum, xden = read_aut(xi_path)
        g8 = xnum * (8 // xden)
        ctrl += [("xi", g8), ("xi^2", (g8 @ g8) // 8)]
    gp = os.path.join(args.data, "group", "m24_generators.txt")
    if os.path.exists(gp):
        perms = [[int(t) for t in ln.split("#")[0].split()] for ln in open(gp) if ln.split("#")[0].split()]
        P = np.zeros((24, 24), dtype=np.int64)
        for i, j in enumerate(perms[1]):
            P[j, i] = 8
        ctrl.append(("gamma (M24 permutation matrix)", P))
        if os.path.exists(xi_path):
            ctrl.append(("xi.gamma", (g8 @ P) // 8))
    ctrl.append(("-I", -8 * np.eye(24, dtype=np.int64)))
    control_ok = True
    for name, g8 in ctrl:
        n_in, bij = L.verify_aut(g8, 8)
        assert n_in == len(L.C) and bij, f"control {name} is not in Co_0"
        img = L.C[S_idx] @ g8.T
        assert np.all(img % 8 == 0)
        gidx = np.sort(L.index_of(img // 8))
        res, first, isos, ext = equivalence(L, S, L.C[gidx])
        ok = first is not None and res["n_ext"] == 8
        control_ok &= ok
        log(f"  g = {name}: |gS n S| = {int(np.isin(gidx, S_idx).sum())}, fingerprint equal to S: "
            f"{co0_fingerprint(L, gidx)['hash'] == fps[0]['hash']}, isomorphisms {res['n_iso']}, "
            f"extending to Co_0 {res['n_ext']} -> {'EQUIVALENT (control passed)' if ok else 'CONTROL FAILED'}")
    assert control_ok, "positive control failed: the search misses genuine Co_0-equivalences"

    # ---- 0. Aut(Gram(S)) and its Co_0 part (T3.4 re-derived) ---------------------------------
    log("\n== Aut(Gram(S)) (T = 0)")
    res0, first0, auts, stab = equivalence(L, S, S, verbose=True)
    log(f"  {res0}")
    assert res0["n_iso"] == 64 and res0["n_ext"] == 8, res0
    G_aut, info_aut = perm_group_info(auts)
    log(f"  Aut(Gram(S)) structure: {info_aut}")
    G_stab, info_stab = perm_group_info([f for f, _ in stab])
    log(f"  Stab_Co0(S) structure: {info_stab}")
    stab_hashes = sorted(aut_hash(num) for _, num in stab)

    # atom partition of S: position -> atom id (or -1)
    atom_of = np.full(n, -1, dtype=np.int64)
    for k in range(nat):
        atom_of[np.isin(S_idx, R[k])] = k
    # 4. the coincidence: does Aut(Gram(S)) respect the atoms?
    log("\n== Aut(Gram(S)) versus the atom partition of S")
    pres_all = 0
    pres_part = 0
    atom_perms = []
    for f in auts:
        img_atom = atom_of[f]
        # does f map each atom onto an atom (as a set)?
        ok = True
        perm = []
        for k in range(nat):
            imgs = set(img_atom[atom_of == k].tolist())
            if len(imgs) == 1 and next(iter(imgs)) >= 0:
                perm.append(next(iter(imgs)))
            else:
                ok = False
                perm.append(None)
        pres_all += ok
        pres_part += (np.all(img_atom[atom_of >= 0] >= 0))
        atom_perms.append(tuple(perm))
    log(f"  automorphisms mapping every atom R_k onto some atom: {pres_all} of {len(auts)}; "
        f"mapping the union of atoms to itself: {pres_part}")
    log(f"  induced permutations of the atoms: {dict(Counter(atom_perms))}")
    # is the union of atoms / each atom a union of Aut(Gram(S)) orbits?  and of Stab orbits?
    for name, perms in [("Aut(Gram(S))", auts), ("Stab_Co0(S)", [f for f, _ in stab])]:
        Gp, _ = perm_group_info(perms)
        orbs = [sorted(int(x) for x in o) for o in Gp.orbits()]
        atom_union = set(np.where(atom_of >= 0)[0].tolist())
        u = all(set(o) <= atom_union or not (set(o) & atom_union) for o in orbs)
        per = [all(set(o) <= set(np.where(atom_of == k)[0].tolist()) or not (set(o) & set(np.where(atom_of == k)[0].tolist()))
                   for o in orbs) for k in range(nat)]
        log(f"  {name}: {len(orbs)} orbits on S, sizes {dict(sorted(Counter(len(o) for o in orbs).items()))}; "
            f"union of atoms is a union of orbits: {u}; each atom is: {per}")
    # fixed points of the stabiliser elements versus the atoms
    for f, num in stab:
        fix = np.where(f == np.arange(n))[0]
        tr = int(np.trace(num)) // 8
        if len(fix) not in (0, n):
            by_atom = Counter(atom_of[fix].tolist())
            log(f"  stabiliser element trace {tr:+d}: fixes {len(fix)} of S; by atom (-1 = outside atoms): {dict(sorted(by_atom.items()))}")
    # the 56 non-Co_0 Gram automorphisms: do their orthogonal maps send any S_T into the minimal vectors?
    non = [(f, orthogonal_extension(S, S, f)) for f in auts]
    n_nonint = sum(1 for _, num in non if num is None)
    log(f"  Gram automorphisms whose orthogonal map has 8g non-integral: {n_nonint} of {len(auts)}")
    hits = Counter()
    for f, num in non:
        if num is None:
            continue
        n_in, bij = L.verify_aut(num, 8)
        if n_in == len(L.C):
            continue
        for T in range(1, NT):
            img = L.C[sets[T]] @ num.T
            if np.all(img % 8 == 0):
                idx = L.index_of(img // 8, strict=False)
                if np.all(idx >= 0):
                    hits[(T, setkeys.get(tuple(sorted(idx.tolist())), -1))] += 1
    log(f"  non-Co_0 Gram automorphisms g_a with g_a S_T inside the minimal vectors (T != 0): {dict(hits) if hits else 'none'}")

    # ---- 1./2. equivalence of every S_T with S ----------------------------------------------------
    log("\n== Co_0-equivalence of S_T with S (T = bitmask over atoms 0..5)")
    masks = list(range(1, NT))
    if args.quick:
        masks = [1, 2, 4, 8, 16, 32, 3, 48, 15, 63]
    results = {0: {"T": 0, "atoms": [], "overlap": n, "n_iso": 64, "n_ext": 8, "equivalent": True,
                   "gram_classes": res0["gram_classes"], "isometric": True, "fp": fps[0]["hash"],
                   "fp_class": fp_name[fps[0]["hash"]], "fp_equal_to_S": True}}
    gT = {0: (np.arange(n), 8 * np.eye(24, dtype=np.int64))}
    done_pairs = set()
    for out in pmap(_job_equiv, [(0, T) for T in masks], args.jobs):
        T = out["b"]
        done_pairs.add(frozenset((0, T)))
        eq = out["equivalent"]
        rec = {k: v for k, v in out.items() if k not in ("a", "b", "g_num", "ext_hashes")}
        rec["T"] = T
        rec["atoms"] = [k for k in range(nat) if (T >> k) & 1]
        rec["isometric"] = out["n_iso"] > 0
        rec["fp"] = fps[T]["hash"]
        rec["fp_class"] = fp_name[fps[T]["hash"]]
        rec["fp_equal_to_S"] = bool(fps[T]["hash"] == fps[0]["hash"])
        if eq:
            num = np.array(out["g_num"], dtype=np.int64)
            gT[T] = (None, num)
            rec["coset_size"] = len(out["ext_hashes"])
            rec["coset_is_g_Stab"] = bool(out["ext_hashes"] == sorted(aut_hash((num @ s) // 8) for _, s in stab))
        results[T] = rec
        log(f"  T={T:2d} atoms={rec['atoms']} overlap={rec['overlap']:3d}: isomorphisms {out['n_iso']} "
            f"(classes {out['gram_classes']}, {out['t_iso']}s), extending to Co_0: {out['n_ext']} {out['ext_stats']} -> "
            f"{'EQUIVALENT' if eq else ('isometric but NOT Co_0-equivalent' if out['n_iso'] else 'NOT equivalent (not even isometric)')}"
            + (f"; g: order {rec['g_order']}, trace {rec['g_trace8']}/8, fixes {rec['g_fixes']} of S, "
               f"S->S {rec['g_maps_A_into_S']}, coset = g·Stab {rec['coset_is_g_Stab']}, hash {rec['g_hash'][:16]}"
               if eq else ""))
    n_isom = sum(1 for T, r in results.items() if isinstance(T, int) and T and r["isometric"])
    log(f"  summary: {sum(1 for T, r in results.items() if isinstance(T, int) and T and r['equivalent'])} of {len(masks)} "
        f"non-trivial S_T are Co_0-equivalent to S; {n_isom} are isometric to S (Gram graphs isomorphic)")

    # ---- 3. orbit structure ---------------------------------------------------------------------
    log("\n== orbit structure")
    parent = list(range(NT))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    iso_parent = list(range(NT))

    def ifind(x):
        while iso_parent[x] != x:
            iso_parent[x] = iso_parent[iso_parent[x]]
            x = iso_parent[x]
        return x

    def iunion(a, b):
        ra, rb = ifind(a), ifind(b)
        if ra != rb:
            iso_parent[max(ra, rb)] = min(ra, rb)

    tested = sorted(T for T in results if isinstance(T, int))
    tset = set(tested)
    for T in tested:
        if results[T]["equivalent"]:
            union(0, T)
        if results[T]["isometric"]:
            iunion(0, T)
    pair_tests = []
    contradictions = []

    def absorb(out):
        a, b = out["a"], out["b"]
        done_pairs.add(frozenset((a, b)))
        pair_tests.append({k: v for k, v in out.items()
                           if k in ("a", "b", "n_iso", "n_ext", "gram_classes", "t_total", "equivalent", "g_hash")})
        same_fp = fps[a]["hash"] == fps[b]["hash"]
        log(f"    S_{a} vs S_{b} (fp#{fp_name[fps[a]['hash']]}/fp#{fp_name[fps[b]['hash']]}, WL "
            f"{'=' if inv[a] == inv[b] else '!='}): isomorphisms {out['n_iso']}, extending {out['n_ext']} -> "
            f"{'EQUIVALENT' if out['equivalent'] else ('ISOMETRIC but not Co_0-equivalent' if out['n_iso'] else 'not even isometric')}")
        if out["n_iso"]:
            iunion(a, b)
        if out["equivalent"]:
            union(a, b)
            if not same_fp:
                contradictions.append((a, b))
                log(f"    *** CONTRADICTION: S_{a} and S_{b} have different Co_0-invariant fingerprints "
                    f"but a Co_0 element maps one onto the other ***")

    # ---- step 0: the isometry structure, constructively ------------------------------------------
    # phase 4 gives K = {T : S_T is ISOMETRIC to S}.  If K is a subgroup of (Z/2)^6 and one orthogonal
    # map g_k realising S -> S_k also carries S_T to S_{T^k} for EVERY T and every generator k, then
    # every coset T^K is one isometry class, and sets in different cosets can only fail to be isometric
    # -- which one representative pair per coset pair then decides.  This replaces O(64^2) searches by
    # O(#cosets^2).  If the verification fails anywhere the code falls back to the exhaustive scan.
    coset_of = None
    iso_gens = []
    if not args.no_orbits and len(tested) == NT:
        K = sorted(T for T in tested if results[T]["isometric"])
        sub = all((a ^ b) in K for a in K for b in K) and 0 in K
        log(f"\n  step 0: K = {{T : S_T isometric to S}} = {K}; closed under XOR (a subgroup of (Z/2)^6): {sub}")
        if sub and len(K) > 1:
            gens = []
            span = {0}
            for k in K:
                if k not in span:
                    gens.append(k)
                    span |= {x ^ k for x in span}
            log(f"  independent generators of K: {gens} (|K| = {len(K)} = 2^{len(gens)})")
            maps = {}
            for T, nums, n_iso in pmap(_job_isomap, gens, args.jobs):
                maps[T] = [np.array(x, dtype=np.int64) for x in nums]
                log(f"    S -> S_{T}: {n_iso} Gram isomorphisms, {len(maps[T])} orthogonal maps g with g S = S_{T} "
                    f"(8g integral, (8g)(8g)^T = 64 I; none of them is in Co_0)")

            def translates(g, k):
                """Does g carry S_T onto S_(T^k) for EVERY T?"""
                for T in range(NT):
                    img = L.C[sets[T]] @ g.T
                    if not np.all(img % 8 == 0):
                        return False
                    X = np.ascontiguousarray(img // 8)
                    Y = np.ascontiguousarray(L.C[sets[T ^ k]])
                    if not np.array_equal(X[np.lexsort(X.T[::-1])], Y[np.lexsort(Y.T[::-1])]):
                        return False
                return True

            chosen = {}
            for k in gens:
                hits = [g for g in maps.get(k, []) if translates(g, k)]
                log(f"    of the {len(maps.get(k, []))} orthogonal maps S -> S_{k}, {len(hits)} carry S_T onto "
                    f"S_(T^{k}) for all {NT} T")
                if hits:
                    chosen[k] = hits[0]
            ok = len(chosen) == len(gens)
            if not ok:
                log("    the plateau moves in K are NOT realised by a single orthogonal map acting on the whole "
                    "6-cube -- falling back to the exhaustive pairwise scan inside each fingerprint class")
            if ok:
                iso_gens = gens
                Kset = set(K)
                coset_of = {T: min(T ^ k for k in Kset) for T in range(NT)}
                ncos = len(set(coset_of.values()))
                log(f"  VERIFIED: each generator g_k maps S_T onto S_(T^k) for all {NT} T. Hence every coset "
                    f"T^K is contained in one isometry class: {ncos} cosets of size {len(K)}.")
                for T in range(NT):
                    if coset_of[T] != T:
                        iunion(coset_of[T], T)

    if args.no_orbits:
        log("  (skipped: --no-orbits)")
    else:
        log(f"  step 1: the {len(fp_classes)} Co_0-invariant fingerprint classes are pairwise inequivalent by construction"
            + ("; and Co_0-equivalence implies isometry, so only pairs inside one coset of K need testing" if coset_of else ""))
        log(f"  step 2: greedy union-find inside each fingerprint class ({args.jobs} workers)")
        for h in fp_order:
            todo = [T for T in fp_classes[h] if T in tset]
            if len(todo) < 2:
                log(f"  -- fp#{fp_name[h]}: {len(todo)} tested set, nothing to compare")
                continue
            log(f"  -- fp#{fp_name[h]}: {len(todo)} tested sets {todo}")
            anchors = []
            while todo:
                a = todo.pop(0)
                if find(a) != a:
                    continue
                anchors.append(a)
                jobs = [(a, T) for T in todo if find(T) == T and frozenset((a, T)) not in done_pairs
                        and (coset_of is None or coset_of[a] == coset_of[T])]
                for out in pmap(_job_equiv, jobs, args.jobs):
                    absorb(out)
                todo = [T for T in todo if find(T) == T]
            log(f"  -- fp#{fp_name[h]}: {len(anchors)} Co_0-class representatives {anchors}")
        # step 3: all remaining pairs of class representatives with the same 1-WL invariant.
        # (different WL invariant => not isometric => not Co_0-equivalent, no test needed.)
        reps = sorted({find(T) for T in tested})
        if coset_of is not None:
            creps = sorted(set(coset_of.values()))
            jobs = [(a, b) for i, a in enumerate(creps) for b in creps[i + 1:]
                    if frozenset((a, b)) not in done_pairs and inv[a] == inv[b]]
            log(f"  step 3: {len(jobs)} tests, one per pair of the {len(creps)} cosets {creps} "
                f"(isometry is constant on cosets, so one representative pair decides each; pairs with "
                f"different 1-WL invariant are skipped as provably non-isometric)")
        elif args.rep_pairs:
            jobs = [(a, b) for i, a in enumerate(reps) for b in reps[i + 1:]
                    if frozenset((a, b)) not in done_pairs and inv[a] == inv[b]]
            log(f"  step 3: {len(jobs)} independent cross-checks among the {len(reps)} class representatives "
                f"(pairs with different 1-WL invariant are skipped: provably not isometric)")
        else:
            jobs = []
            log("  step 3: skipped (--rep-pairs not given): the Co_0-class count above is already complete "
                "(Co_0-equivalence needs an equal fingerprint, and every same-fingerprint pair was tested); "
                "only the isometry classes across fingerprint classes are left undetermined")
        for out in pmap(_job_equiv, jobs, args.jobs):
            absorb(out)
        log(f"  contradictions with the Co_0-invariant fingerprints: {len(contradictions)} (must be 0)")
    classes = {}
    for T in tested:
        classes.setdefault(find(T), []).append(T)
    iso_classes = {}
    for T in tested:
        iso_classes.setdefault(ifind(T), []).append(T)
    log(f"  Co_0-inequivalent classes among the {len(tested)} tested sets: {len(classes)}; "
        f"sizes {sorted((len(v) for v in classes.values()), reverse=True)}")
    log(f"  isometry (Gram-graph isomorphism) classes among the tested sets, as far as the tests above "
        f"determine them: {len(iso_classes)}; sizes {sorted((len(v) for v in iso_classes.values()), reverse=True)}")
    for r, v in sorted(iso_classes.items()):
        sub = sorted({find(T) for T in v})
        log(f"    isometry class rep S_{r} ({len(v)} sets): splits into {len(sub)} Co_0-classes (reps {sub[:12]}{'...' if len(sub) > 12 else ''})")

    # the action of the atom elements g_k on the 64 sets
    action = {}
    if all(k in gT for k in [1 << j for j in range(nat)]):
        log("  action of the atom elements g_k (k = 0..5) and the stabiliser on the 64 sets:")
        gens = [(f"g{k}", gT[1 << k][1]) for k in range(nat)] + [(f"s{j}", s) for j, (_, s) in enumerate(stab)]
        perms64 = []
        for name, num in gens:
            p = []
            outside = 0
            for T in range(NT):
                img = L.C[sets[T]] @ num.T
                assert np.all(img % 8 == 0)
                idx = np.sort(L.index_of(img // 8))
                p.append(setkeys.get(tuple(idx.tolist()), -1))
                outside += p[-1] < 0
            action[name] = p
            if outside == 0:
                perms64.append(p)
            log(f"    {name}: maps the 64 sets to plateau sets: {NT - outside} of {NT}"
                + (f"; as a permutation of T: {[(T, p[T]) for T in range(NT) if p[T] != T][:8]}..." if outside == 0 else
                   f"; e.g. g·S_T outside the plateau for T = {[T for T in range(NT) if p[T] < 0][:8]}"))
        if len(perms64) == len(gens):
            Gp, info = perm_group_info(perms64)
            log(f"    group generated on the 64 sets: {info}")
            log(f"    orbit of S_0 under it: size {len([o for o in Gp.orbits() if 0 in o][0])}")
    results["action"] = action

    # ---- Stab_Co0(S_T) for a few T ------------------------------------------------------------------
    log("\n== Stab_Co0(S_T) for selected T (full Aut(Gram(S_T)) enumeration, as in T3.4)")
    stab_checks = {}
    for T in [int(t) for t in args.stab.split(",") if t.strip()]:
        B = L.C[sets[T]]
        res, first, isosB, extB = equivalence(L, B, B)
        Gq, infoq = perm_group_info([f for f, _ in extB])
        hT = sorted(aut_hash(num) for _, num in extB)
        conj = None
        if T in gT:
            g = gT[T][1]
            ginv = g.T  # 8 g^{-1} = (8 g)^T
            conj = sorted(aut_hash((g @ s @ ginv) // 64) for _, s in stab)
        stab_checks[T] = {"aut_gram": res["n_iso"], "gram_classes": res["gram_classes"], "stab_order": res["n_ext"],
                          "structure": infoq, "equals_conjugate": (hT == conj) if conj is not None else None,
                          "stab_hashes": hT, "same_hashes_as_stab_S": bool(hT == stab_hashes)}
        log(f"  T={T}: |Aut(Gram(S_T))| = {res['n_iso']} ({res['gram_classes']} WL classes), |Stab_Co0(S_T)| = {res['n_ext']}, "
            f"structure {infoq}; equals g_T Stab(S) g_T^-1: {stab_checks[T]['equals_conjugate']}; "
            f"identical to Stab_Co0(S) as a set of matrices: {stab_checks[T]['same_hashes_as_stab_S']}")
    results["stab_checks"] = stab_checks

    # ---- outputs ---------------------------------------------------------------------------------
    for T, (f, num) in gT.items():
        if T == 0:
            continue
        write_aut(os.path.join(args.out, f"g_T{T:02d}.txt"), num,
                  [f"T3.4b: element of Co_0 mapping the 496 (data/S496.txt) onto the plateau set S_T, T = {T} (atoms {results[T]['atoms']})",
                   f"order {results[T]['g_order']}, trace {results[T]['g_trace8']}/8, sha256(8g) {results[T]['g_hash']}",
                   "Format: 'den D' then 24 rows of 24 integer numerators; (g v)[r] = sum_c num[r][c] v[c] / D."])
    if not args.no_auts_out and all((1 << k) in gT for k in range(nat)):
        os.makedirs(args.auts_out, exist_ok=True)
        for k in range(nat):
            T = 1 << k
            f, num = gT[T]
            rec = results[T]
            write_aut(os.path.join(args.auts_out, f"g_atom{k}.txt"), num,
                      [f"T3.4b: Co_0 element g_{k} mapping the record 496 (data/S496.txt) onto its plateau image under atom {k}",
                       f"(the ({len(R[k])},{len(A[k])}) move of runs/ls_search/plateau_atoms_496.json; |g S ∩ S| = {rec['overlap']}).",
                       f"element order {rec['g_order']}, trace {rec['g_trace8']}/8, fixes {rec['g_fixes_in_S']} vectors of S; sha256(8g) = {rec['g_hash']}",
                       "Verified: 8g integral, (8g)(8g)^T = 64 I, maps all 196560 minimal vectors bijectively onto themselves,",
                       "and g S = S_T exactly (python/tools/plateau_equivalence.py).  One representative of the coset g Stab_Co0(S) (8 elements).",
                       "Format: 'den D' then 24 rows of 24 integer numerators; (g v)[r] = sum_c num[r][c] v[c] / D."])
        log(f"\nwritten: {nat} atom elements to {os.path.relpath(args.auts_out, ROOT)}/g_atom<k>.txt")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"aut_gram_S": info_aut, "stab_S": info_stab, "stab_S_hashes": stab_hashes,
                   "wl_invariants": {str(T): [inv[T][0], list(inv[T][1]), inv[T][2]] for T in range(NT)},
                   "fingerprints": {str(T): fps[T] for T in range(NT)},
                   "fingerprint_classes": {str(fp_name[h]): fp_classes[h] for h in fp_order},
                   "monomial_classes": {k: v for k, v in mono_classes.items()},
                   "fingerprint_text": {str(fp_name[h]): fp_text(fps[fp_classes[h][0]]) for h in fp_order},
                   "t31b_crosscheck": xcheck,
                   "iso_classes": {str(k): v for k, v in iso_classes.items()},
                   "isometry_subgroup_K": (sorted(T for T in range(NT) if coset_of[T] == coset_of[0]) if coset_of else None),
                   "isometry_generators": iso_gens,
                   "per_T": {str(T): r for T, r in results.items() if isinstance(T, int)},
                   "classes": {str(k): v for k, v in classes.items()}, "pair_tests": pair_tests,
                   "action": action, "stab_checks": {str(k): v for k, v in stab_checks.items()}}, f, indent=1, default=str)
    n_eq = sum(1 for T, r in results.items() if isinstance(T, int) and r["equivalent"])
    log(f"\nRESULT ok=1 sets={len(tested)} equivalent_to_S={n_eq} classes={len(classes)} "
        f"iso_classes={len(iso_classes)} aut_gram=64 stab=8 "
        f"seconds={time.time() - t_start:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

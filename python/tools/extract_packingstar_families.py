#!/usr/bin/env python3
"""T4.2 — extract PackingStar's disjoint families (dims 25–31) into data/families/ (SCHEMA.md).

PackingStar publishes no per-dimension S_i files; the S_i, T_i and extra spheres are implicit
in the unit-vector configurations <n>D_<count>_coordinates.npy (float64, (count, 24+d)).
Rows decompose exactly (docs/reports/T1.3-fetch.md §2.1):

    equatorial  (x/√32, 0)                 x ∈ C \ ∪S_i          → x = head·√32
    lifted      (x/(4√3), y/√3)            x ∈ S_i, y ∈ T_i      → x = head·4√3, y = tail·√3
    extra       (0, y')                                          → y' = tail

This tool (1) recovers the integer x (√8 scaling, norm 32), groups lifted rows by y into S_i and
the y's by S_i into T_i; (2) maps the float R^d configuration {y} onto the exact integer root-system
model of SCHEMA.md §2 (A2, D3, D4, D5, E6, E7) by an isometry found from a basis (backtracking on
the Gram matrix, then linear extension, checked on every row); (3) pushes the extra spheres through
the same isometry and rationalises them as (a + b√M)/D with integers a, b — exactness is then
*verified* (norms, all angle conditions of SCHEMA.md §3 in integer arithmetic); (4) converts every
S_i into our coordinates with data/external/coordinate_map.json and verifies it independently
(rows in C by the README §1.2 membership test, norm 32, distinct, Gram ≤ 8) and the family's
pairwise disjointness; (5) writes data/families/dim<n>/{family.json, S_xx.txt}, committing each
distinct 496-set once (nested families reference ../dim31/S_xx.txt).

Usage:
    extract_packingstar_families.py [--dims 25,26,...] [--out data/families] [--dry-run]

Uses only numpy and python/kiss_ref (the independent Python reference), never the C++ outputs.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from kiss_ref.leech import is_lattice_vector_rows, leech_min_vectors  # noqa: E402

PS_CFG = os.path.join(ROOT, "data", "external", "PackingStar", "25D-31D", "25D-31D_new_bounds_configurations")
MAP_JSON = os.path.join(ROOT, "data", "external", "coordinate_map.json")
KISSING = {1: 2, 2: 6, 3: 12, 4: 24, 5: 40, 6: 72, 7: 126}
CONFIG_NAME = {1: "A1", 2: "A2", 3: "A3", 4: "D4", 5: "D5", 6: "E6", 7: "E7"}
N_LEECH = 196560


# ----------------------------------------------------------------------------
# exact models of the K(d) kissing configurations (SCHEMA.md §2 conventions)
# ----------------------------------------------------------------------------
def _lex_sorted(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.int64)
    return np.ascontiguousarray(V[np.lexsort(V.T[::-1])])


def _d_roots(n: int) -> np.ndarray:
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * n
                v[i], v[j] = si, sj
                rows.append(v)
    return np.array(rows, dtype=np.int64)


def _e8_roots() -> np.ndarray:
    rows = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = [0] * 8
                v[i], v[j] = si, sj
                rows.append(v)
    for s in itertools.product((1, -1), repeat=8):
        if s.count(-1) % 2 == 0:
            rows.append(list(s))
    return np.array(rows, dtype=np.int64)


def model(d: int) -> tuple[np.ndarray, int]:
    """(integer vectors (K(d), m), norm2) — see SCHEMA.md §2."""
    if d == 1:
        V, n2 = np.array([[1], [-1]], dtype=np.int64), 1
    elif d == 2:
        V, n2 = np.array([p for p in set(itertools.permutations((1, -1, 0)))], dtype=np.int64), 2
    elif d in (3, 4, 5):
        V, n2 = _d_roots(d), 2
    elif d == 7:
        R = _e8_roots()
        V, n2 = R[R @ np.array([2, -2, 0, 0, 0, 0, 0, 0]) == 0], 8
    elif d == 6:
        R = _e8_roots()
        V, n2 = R[(R @ np.array([2, -2, 0, 0, 0, 0, 0, 0]) == 0) & (R @ np.array([0, 2, -2, 0, 0, 0, 0, 0]) == 0)], 8
    else:
        raise ValueError(d)
    V = _lex_sorted(V)
    assert V.shape[0] == KISSING[d] and np.all((V * V).sum(1) == n2)
    return V, n2


# ----------------------------------------------------------------------------
# exact sign of A + B*sqrt(M)
# ----------------------------------------------------------------------------
def sign_sqrt(A: int, B: int, M: int) -> int:
    A, B, M = int(A), int(B), int(M)
    if B == 0:
        return (A > 0) - (A < 0)
    r = B * B * M
    if B > 0:
        if A >= 0:
            return 1
        return (r > A * A) - (r < A * A)  # A<0: sign = sign(r - A^2)
    # B < 0
    if A <= 0:
        return -1
    return (A * A > r) - (A * A < r)


# ----------------------------------------------------------------------------
# decomposition of a PackingStar configuration
# ----------------------------------------------------------------------------
def decompose(path: str, d: int):
    a = np.load(path)
    n, dim = a.shape
    assert dim == 24 + d, (dim, d)
    head, tail = a[:, :24], a[:, 24:]
    hn, tn = (head ** 2).sum(1), (tail ** 2).sum(1)
    eq = tn < 1e-9
    ex = hn < 1e-9
    li = ~eq & ~ex
    xe = head[eq] * math.sqrt(32)
    xl = head[li] * 4 * math.sqrt(3)
    res = max(np.abs(xe - np.rint(xe)).max(), np.abs(xl - np.rint(xl)).max())
    assert res < 1e-9, res
    xe = np.rint(xe).astype(np.int64)
    xl = np.rint(xl).astype(np.int64)
    yl = tail[li] * math.sqrt(3)
    hn_l, tn_l = (head[li] ** 2).sum(1), (tail[li] ** 2).sum(1)
    assert np.abs(hn_l - 2 / 3).max() < 1e-9 and np.abs(tn_l - 1 / 3).max() < 1e-9
    # group lifted rows by y (first appearance order)
    ykeys = {}
    y_of_row = np.empty(len(yl), dtype=np.int64)
    for r, y in enumerate(yl):
        k = tuple(np.round(y, 8))
        if k not in ykeys:
            ykeys[k] = len(ykeys)
        y_of_row[r] = ykeys[k]
    Y = np.array([np.array(k) for k in ykeys])          # distinct T vectors (unit), first-appearance order
    sets_by_y = collections.defaultdict(list)
    for r in range(len(xl)):
        sets_by_y[int(y_of_row[r])].append(tuple(int(v) for v in xl[r]))
    groups_by_set = collections.OrderedDict()             # frozenset(S) -> [y indices]
    first_row = {}
    for yi in range(len(Y)):
        key = frozenset(sets_by_y[yi])
        assert len(key) == len(sets_by_y[yi]), "duplicate x under one y"
        groups_by_set.setdefault(key, []).append(yi)
        first_row.setdefault(key, min(r for r in range(len(xl)) if y_of_row[r] == yi))
    groups = sorted(groups_by_set.items(), key=lambda kv: first_row[kv[0]])
    E = tail[ex]
    return {"n": n, "eq": xe, "groups": groups, "Y": Y, "E": E, "d": d}


# ----------------------------------------------------------------------------
# isometry float configuration -> integer model
# ----------------------------------------------------------------------------
def find_isometry(Y: np.ndarray, V: np.ndarray, n2: int):
    """Return (sigma, Ybasis_inv, Vbasis): Y[i] ↦ V[sigma[i]]; the linear map is
    y ↦ (y @ Ybasis_inv) @ Vbasis. Gram matrices are matched exactly (rounded)."""
    K, d = Y.shape
    assert V.shape[0] == K
    Gy = np.rint(Y @ Y.T * n2).astype(np.int64)      # target integer Gram
    assert np.abs(Y @ Y.T * n2 - Gy).max() < 1e-6
    Gv = V @ V.T
    # basis rows of Y (greedy by rank)
    basis = []
    for i in range(K):
        cand = basis + [i]
        if np.linalg.matrix_rank(Y[cand], tol=1e-8) == len(cand):
            basis.append(i)
        if len(basis) == d:
            break
    assert len(basis) == d
    # backtracking: assign model vectors to basis rows matching pairwise Gram, then extend
    # linearly and accept only if every row lands on a model vector (a Gram-matched basis may
    # generate a proper sublattice, in which case the extension is another copy of the system)
    lut = {tuple(int(x) for x in v): i for i, v in enumerate(V)}
    Yb_inv = np.linalg.inv(Y[basis])
    coeff = Y @ Yb_inv                                # coeff @ Y[basis] == Y
    assign = [-1] * d
    result = {}
    tried = [0]

    def extend() -> bool:
        Vb = V[assign].astype(np.float64)
        img = coeff @ Vb
        imgi = np.rint(img).astype(np.int64)
        tried[0] += 1
        if np.abs(img - imgi).max() > 1e-6:
            return False
        sigma = [lut.get(tuple(int(x) for x in r), -1) for r in imgi]
        if min(sigma) < 0 or len(set(sigma)) != K:
            return False
        result["sigma"], result["Vb"] = np.array(sigma), Vb
        return True

    def bt(k: int) -> bool:
        if k == d:
            return extend()
        for v in range(K):
            if v in assign[:k]:
                continue
            if all(Gv[v, assign[j]] == Gy[basis[k], basis[j]] for j in range(k)):
                assign[k] = v
                if bt(k + 1):
                    return True
        assign[k] = -1
        return False

    if not bt(0):
        raise RuntimeError("no isometry onto the model (configuration is not the expected root system?)")
    sigma, Vb = result["sigma"], result["Vb"]
    return np.array(sigma), Yb_inv, Vb


def rationalise(X: np.ndarray, n2: int):
    """X float (n, m): find M, D and integer a, b with X = (a + b√M)/D exactly (verified below)."""
    for M in (1, 2, 3, 5, 6, 7):
        sM = math.sqrt(M)
        for D in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
            A = np.zeros(X.shape, dtype=np.int64)
            B = np.zeros(X.shape, dtype=np.int64)
            ok = True
            for idx, x in np.ndenumerate(X):
                xd = x * D
                found = False
                for b in range(0, 33):
                    for bb in ((b, -b) if b else (0,)):
                        aa = round(xd - bb * sM)
                        if abs(aa + bb * sM - xd) < 1e-7:
                            A[idx], B[idx] = aa, bb
                            found = True
                            break
                    if found:
                        break
                if not found:
                    ok = False
                    break
            if not ok:
                continue
            if M == 1:
                A, B = A + B, np.zeros_like(B)
            # exact norm check: |a|^2 + M|b|^2 = D^2 n2 and <a,b> = 0
            na = (A * A).sum(1) + M * (B * B).sum(1)
            ab = (A * B).sum(1)
            if np.all(na == D * D * n2) and np.all(ab == 0):
                return M, D, A, B
    raise RuntimeError("could not rationalise the extra spheres")


# ----------------------------------------------------------------------------
# exact checks of the R^d half (SCHEMA.md §3 items 2–4)
# ----------------------------------------------------------------------------
def check_rd(V: np.ndarray, n2: int, groups: list[list[int]], M: int, D: int, A: np.ndarray, B: np.ndarray) -> dict:
    K = len(V)
    G = V @ V.T
    assert np.all(np.diag(G) == n2)
    off = ~np.eye(K, dtype=bool)
    assert np.all(2 * G[off] <= n2), "T: cos > 1/2"
    assert not np.any(G[off] == n2), "T: repeated direction"
    used = set()
    for g in groups:
        assert len(g) in (2, 3)
        assert not (set(g) & used)
        used.update(g)
        for i, j in itertools.combinations(g, 2):
            assert 2 * G[i, j] <= -n2, f"group {g}: cos > -1/2 inside a group"
        if len(g) == 3:
            for i, j in itertools.combinations(g, 2):
                assert 2 * G[i, j] == -n2
    ne = len(A)
    stats = {"T_pairs_touching": int((2 * G[off] == n2).sum() // 2), "extra": ne}
    if ne:
        # norms
        assert np.all((A * A).sum(1) + M * (B * B).sum(1) == D * D * n2) and np.all((A * B).sum(1) == 0)
        # extra-extra
        P = A @ A.T + M * (B @ B.T)
        Q = A @ B.T + B @ A.T
        touch = 0
        for i in range(ne):
            for j in range(i + 1, ne):
                s = sign_sqrt(2 * P[i, j] - n2 * D * D, 2 * Q[i, j], M)
                assert s <= 0, f"extra {i},{j}: cos > 1/2"
                touch += s == 0
                assert not (Q[i, j] == 0 and P[i, j] == n2 * D * D), "extra: repeated direction"
        stats["extra_pairs_touching"] = touch
        # extra-T
        p = A @ V.T
        q = B @ V.T
        touch = 0
        for i in range(ne):
            for t in range(K):
                pp, qq = int(p[i, t]), int(q[i, t])
                if sign_sqrt(2 * pp, 2 * qq, M) <= 0:
                    continue
                s = sign_sqrt(4 * pp * pp + 4 * M * qq * qq - 3 * n2 * n2 * D * D, 8 * pp * qq, M)
                assert s <= 0, f"extra {i} vs T {t}: cos > sqrt3/2"
                touch += s == 0
        stats["extra_T_touching"] = touch
    return stats


# ----------------------------------------------------------------------------
# Leech side
# ----------------------------------------------------------------------------
def apply_map(rows: np.ndarray, perm: list[int]) -> np.ndarray:
    out = np.zeros_like(rows)
    out[:, np.asarray(perm)] = rows
    return out


def verify_set(S: np.ndarray, Cset: set) -> dict:
    n = len(S)
    keys = set(map(tuple, S.tolist()))
    assert len(keys) == n, "duplicate rows"
    assert np.all((S * S).sum(1) == 32), "norm != 32"
    assert all(k in Cset for k in keys), "row not in C"
    assert int(is_lattice_vector_rows(S).sum()) == n, "membership test failed"
    G = S @ S.T
    off = G[~np.eye(n, dtype=bool)]
    assert off.max() <= 8, "Gram > 8"
    hist = {int(v): int(c) for v, c in zip(*np.unique(off, return_counts=True))}
    return {"size": n, "gram_hist": hist, "antipodal": all(tuple(-x for x in k) in keys for k in keys)}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_set_file(path: str, S: np.ndarray, header: list[str]) -> None:
    with open(path, "w") as f:
        for h in header:
            f.write(f"# {h}\n")
        for r in S:
            f.write(" ".join(str(int(v)) for v in r) + "\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="31,26,28,27,29,30,25")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "families"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    with open(MAP_JSON) as f:
        cmap = json.load(f)
    perm = cmap["perm"]
    assert all(s == 1 for s in cmap["signs"])
    Cset = set(map(tuple, leech_min_vectors().astype(np.int64).tolist()))
    print(f"our C: {len(Cset)} vectors; map perm={perm}")
    files = {int(os.path.basename(p).split("D_")[0]): os.path.join(PS_CFG, p) for p in os.listdir(PS_CFG) if p.endswith(".npy")}
    dims = [int(x) for x in args.dims.split(",")]
    written: dict[frozenset, str] = {}      # frozenset(rows in our coords) -> relative path (from data/families)
    all_ok = True
    summary = []
    for n in dims:
        d = n - 24
        path = files[n]
        dec = decompose(path, d)
        groups = dec["groups"]
        k = len(groups)
        V, n2 = model(d)
        if d == 1:
            Y = dec["Y"]
            sigma = np.array([0 if y[0] > 0 else 1 for y in Y])
            M, D, A, B = 1, 1, np.zeros((0, 1), dtype=np.int64), np.zeros((0, 1), dtype=np.int64)
        else:
            sigma, Yb_inv, Vb = find_isometry(dec["Y"], V, n2)
            E = dec["E"]
            Eimg = (E @ Yb_inv) @ Vb
            M, D, A, B = rationalise(Eimg, n2)
        T_groups = [[int(sigma[yi]) for yi in ys] for _, ys in groups]
        stats = check_rd(V, n2, T_groups, M, D, A, B)
        # Leech side: convert, verify, disjointness, equatorial complement
        S_ours = []
        S_stats = []
        for Sset, _ in groups:
            S = apply_map(np.array(sorted(Sset), dtype=np.int64), perm)
            S = _lex_sorted(S)
            S_stats.append(verify_set(S, Cset))
            S_ours.append(S)
        keysets = [frozenset(map(tuple, S.tolist())) for S in S_ours]
        union = set().union(*keysets)
        assert len(union) == sum(len(s) for s in keysets), "S_i not pairwise disjoint"
        eq = set(map(tuple, apply_map(dec["eq"], perm).tolist()))
        assert eq == Cset - union, "equatorial rows != C \\ union S_i"
        weights = [len(g) - 1 for g in T_groups]
        lifted = sum(w * len(S) for w, S in zip(weights, S_ours))
        count = len(A) + N_LEECH + lifted
        assert count == dec["n"], (count, dec["n"])
        assert len(A) == (0 if d == 1 else KISSING[d])
        pair_kinds = collections.Counter("antipodal" if np.array_equal(V[g[0]], -V[g[1]]) else "cos=-1/2" for g in T_groups if len(g) == 2)
        print(f"dim {n} (d={d}): {os.path.basename(path)} rows={dec['n']} sets={k} sizes={sorted(collections.Counter(len(S) for S in S_ours).items())} pairs={dict(pair_kinds)} "
              f"groups={dict(collections.Counter(len(g) for g in T_groups))} T={CONFIG_NAME[d]} ambient={V.shape[1]} norm2={n2} "
              f"extra={len(A)} sqrt={M} den={D} count={count} touching(T-T,E-E,E-T)="
              f"{stats['T_pairs_touching']},{stats.get('extra_pairs_touching', 0)},{stats.get('extra_T_touching', 0)}")
        # write
        out_dir = os.path.join(args.out, f"dim{n}")
        rel_dir = f"dim{n}"
        sets_json = []
        src_rel = os.path.relpath(path, ROOT)
        src_sha = sha256_file(path)
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
        for i, (S, ks, st) in enumerate(zip(S_ours, keysets, S_stats), start=1):
            if ks in written:
                rel = os.path.relpath(written[ks], rel_dir)
                fpath = os.path.join(args.out, written[ks])
            else:
                fname = f"S_{i:02d}.txt"
                fpath = os.path.join(out_dir, fname)
                header = [
                    f"S_{i} of PackingStar's {n}D family (set {i} of {k}), |S|={len(S)}, in our coordinates",
                    f"source: {src_rel} (sha256 {src_sha}); lifted rows with T-vector(s) {T_groups[i-1]} of {CONFIG_NAME[d]}",
                    f"map: ours[perm[i]] = theirs[i], perm = {perm} (data/external/coordinate_map.json)",
                    f"written by python/tools/extract_packingstar_families.py on {_dt.date.today().isoformat()}; rows sorted lexicographically",
                    "sqrt8-integer scaling (squared norm 32); independent set iff all off-diagonal Gram entries <= 8",
                ]
                if not args.dry_run:
                    write_set_file(fpath, S, header)
                written[ks] = f"{rel_dir}/{fname}"
                rel = fname
            entry = {"file": rel, "size": int(len(S))}
            if not args.dry_run:
                entry["sha256"] = sha256_file(fpath)
            sets_json.append(entry)
        fam = {
            "schema_version": 1,
            "dim": n,
            "d": d,
            "coordinates": "leech-sqrt8-integer",
            "sets": sets_json,
            "T": {"config": CONFIG_NAME[d], "K": int(len(V)), "ambient": int(V.shape[1]), "norm2": int(n2),
                  "vectors": V.tolist(), "groups": T_groups},
            "extra": {"count": int(len(A)), "sqrt": int(M), "den": int(D), "a": A.tolist(), "b": B.tolist()},
            "count": int(count),
            "count_terms": {"extra": int(len(A)), "leech": N_LEECH, "lifted": int(lifted)},
            "count_formula": f"{len(A)} + {N_LEECH} + sum_i (|T_i| - 1) * |S_i|  (weights {weights})",
            "provenance": {
                "source": src_rel, "source_sha256": src_sha,
                "repo": "https://github.com/CDM1619/PackingStar @ " + cmap["repo_commits"]["PackingStar"],
                "coordinate_map": "data/external/coordinate_map.json (perm only, no signs)",
                "tool": "python/tools/extract_packingstar_families.py", "date": _dt.date.today().isoformat(),
                "notes": "S_i = lifted rows grouped by their R^d vector; T = float configuration mapped onto the "
                         "integer model by an isometry fixed on a basis; extras rationalised in Q(sqrt(M)) and "
                         "verified exactly; sets in first-appearance order of the upstream rows; "
                         "shared sets are referenced (nested families, SCHEMA.md §4)",
                "set_gram_hist": [st["gram_hist"] for st in S_stats][0],
                "sets_antipodal": all(st["antipodal"] for st in S_stats),
            },
        }
        if not args.dry_run:
            with open(os.path.join(out_dir, "family.json"), "w") as f:
                json.dump(fam, f, indent=None, separators=(",", ":"))
                f.write("\n")
        summary.append((n, k, count, len(A), M, D))
    print(f"distinct 496-sets written: {len(written)}")
    print("RESULT ok=%d dims=%s counts=%s sets=%s distinct=%d time_s=%.1f" % (
        int(all_ok), ",".join(str(s[0]) for s in summary), ",".join(str(s[2]) for s in summary),
        ",".join(str(s[1]) for s in summary), len(written), time.time() - t0))


if __name__ == "__main__":
    main()

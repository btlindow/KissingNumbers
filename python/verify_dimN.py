#!/usr/bin/env python3
"""Verify a disjoint family S_1..S_k with its R^d data as a kissing configuration in R^{24+d} (PLAN T4.3).

    python/verify_dimN.py data/families/dim31 [--strict] [--skip-full-equatorial] [--skip-float]
                          [--tile 512 1024] [--workers 8]

Input: a family directory (or a family.json path) in the schema of
data/families/SCHEMA.md (T4.2). The reader is tolerant: "sets" entries may be
plain file names; the "T"/"extra" blocks may be plain vector lists, may carry
"scale"/"den" (all entries divided by it) and "sqrt" M (entries given as
[a, b] pairs meaning a + b sqrt M, or as the schema's "a"/"b" matrices), and
entries may be integers, small dyadic floats, or strings such as "-1/2",
"sqrt(2)/2", "(2+sqrt(2))/4". Everything in R^d is handled EXACTLY in the
number field Q(sqrt p_1, ...) (kiss_ref.exact); the Leech half in int64 /
exact float32 GEMM.

The configuration (README 1.3, norm-4 scaling; x a minimal vector of squared
norm 32 in the sqrt8-integer coordinates, so x/sqrt8 has norm 4; y in T_i and
y' extra are unit vectors u(.) of the R^d data):

    equatorial   (x/sqrt8, 0)                                x in C \\ (S_1 u ... u S_k)
    lifted       (sqrt(2/3) x/sqrt8, sqrt(4/3) u(y))          x in S_i, y in T_i
    extra        (0, 2 u(y'))                                 y' extra

count = #extra + 196560 + sum_i (|T_i| - 1) |S_i|   (the README 1.3 form: K(d) + K(24) + ...).

(a) EXACT casework. ip = <x,x'> (integer, in {-32,...,32} for minimal vectors),
    p = <y,y'>, Na = |y|^2, Nb = |y'|^2 in the R^d field; cos = p/sqrt(Na Nb).
    Every pair type reduces to one of these integer / field inequalities:

      eq-eq            ip/8 <= 2                              <=> ip <= 16      (distinct minimal vectors)
      eq-lift          sqrt(2/3) ip/8 <= 2                    <=> ip <= 8 sqrt6 = 19.6; ip <= 16 suffices,
                                                                  i.e. the equatorial set is exactly C \\ U
      lift-lift, same S_i, same x, y != y'
                       (2/3) 4 + (4/3) cos <= 2               <=> cos <= -1/2   (within-group condition)
                                                                  <=> p < 0 and 4 p^2 >= Na Nb
      lift-lift, same S_i, x != x', y = y'
                       ip/12 + 4/3 <= 2                       <=> ip <= 8       (S_i independent)
      lift-lift, same S_i, x != x', y != y'
                       ip/12 + (4/3) cos <= 2, cos <= -1/2    <=  ip <= 32, always true
      lift-lift, different S_i, S_j (x != x' by disjointness)
                       ip/12 + (4/3) cos <= 2, cos <= 1/2     <=  ip <= 16     (distinct minimal vectors;
                                                                  tight: ip = 16 and cos = 1/2 give exactly 2)
      extra-eq         0 <= 2                                     always
      extra-lift       2 sqrt(4/3) cos <= 2                   <=> cos <= sqrt3/2 <=> p <= 0 or 4 p^2 <= 3 Na Nb
      extra-extra      4 cos <= 2                             <=> cos <= 1/2   <=> p <= 0 or 4 p^2 <= Na Nb

    The R^d conditions are decided exactly in the field; ip <= 16 for all
    distinct pairs of C is verified by a full float32-GEMM pass over C (exact:
    integer inputs |.| <= 4, partial sums <= 384, see verify_dim25.py) unless
    --skip-full-equatorial, and the maxima of ip over eq-lift and over
    cross-set lifted pairs are computed explicitly as well. Pair counts per
    type are summed and checked against C(count, 2).

(b) FLOAT. Explicit float64 coordinates in R^{24+d} (the R^d data embedded in
    exactly d coordinates by an orthonormal embedding of its span, exact rank
    <= d checked in the field first), tiled GEMM as in verify_dim25.py, max
    off-diagonal inner product <= 2 + 1e-9 and the class of the pair attaining it.

Prints `RESULT ok=1 dim=... count=... count_exact=... count_float=... max_offdiag=...`,
exit 0 on success, 1 otherwise (the first failing check is named).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # see verify_dim25.py

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiss_ref.exact import (Q, cos_le_half, cos_le_neg_half, cos_le_sqrt3_half, dot, is_antipodal,  # noqa: E402
                            is_same_direction, norm2, parse_entry, parse_vector, rank)
from kiss_ref.leech import DIM, N_MIN  # noqa: E402
from kiss_ref.small_kissing import K as KISSING  # noqa: E402
from kiss_ref.small_kissing import orthonormal_embedding  # noqa: E402
from verify_S import check_set, leech_index, quote, read_set  # noqa: E402
from verify_dim25 import default_workers, exact_full_pass, pairwise_offdiag_max  # noqa: E402

SQRT8 = math.sqrt(8.0)
LIFT_X = math.sqrt(2.0 / 3.0)
LIFT_Y = math.sqrt(4.0 / 3.0)
PAIR_CLASSES = ("eq-eq", "eq-lift", "lift-lift-same-x", "lift-lift-same-set", "lift-lift-cross-set",
                "extra-eq", "extra-lift", "extra-extra")


class FamilyError(ValueError):
    """A family that fails a check; str(e) names the check."""


# ---------------------------------------------------------------------------
# reading family.json (schema v1 of data/families/SCHEMA.md, tolerant superset)
# ---------------------------------------------------------------------------
class Family:
    def __init__(self) -> None:
        self.dir = ""
        self.raw: dict = {}
        self.dim = 0
        self.d = 0
        self.set_files: list[str] = []
        self.set_meta: list[dict] = []
        self.T: list[list[Q]] = []
        self.groups: list[list[int]] = []
        self.extras: list[list[Q]] = []
        self.T_meta: dict = {}
        self.extra_meta: dict = {}
        self.count_claimed: int | None = None
        self.warnings: list[str] = []


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def _depth(x) -> int:
    k = 0
    while isinstance(x, (list, tuple)) and len(x):
        x = x[0]
        k += 1
    return k


def parse_vector_block(block, what: str) -> tuple[list[list[Q]], dict]:
    """A "T"/"extra" block -> (rows of Q, meta). Accepts the schema's forms and plain lists."""
    meta: dict = {}
    if block is None:
        return [], meta
    if isinstance(block, (list, tuple)):
        block = {"vectors": list(block)}
    if not isinstance(block, dict):
        raise FamilyError(f"{what}: block must be an object or a list")
    meta = {k: v for k, v in block.items() if k not in ("vectors", "rows", "a", "b", "groups")}
    radical = _get(block, "sqrt", "radical", "M")
    radical = int(radical) if radical is not None else None
    scale = _get(block, "scale", "den", "denominator", "D")
    scale_q = parse_entry(scale) if scale is not None else None
    if scale_q is not None and scale_q.is_zero():
        raise FamilyError(f"{what}: zero scale")
    rows: list[list[Q]] = []
    if "a" in block or "b" in block:
        a = block.get("a", [])
        b = block.get("b", [])
        if not a and b:
            a = [[0] * len(r) for r in b]
        if not b and a:
            b = [[0] * len(r) for r in a]
        if len(a) != len(b):
            raise FamilyError(f"{what}: 'a' has {len(a)} rows, 'b' has {len(b)}")
        M = radical if radical is not None else 1
        sq = Q.sqrt(M)
        for ra, rb in zip(a, b):
            if len(ra) != len(rb):
                raise FamilyError(f"{what}: an 'a' row and its 'b' row have different lengths")
            rows.append(parse_vector([parse_entry(x) + parse_entry(y) * sq for x, y in zip(ra, rb)],
                                     scale=scale_q))
    else:
        vectors = _get(block, "vectors", "rows", default=[])
        dep = _depth(vectors)
        pairs = dep == 3 and radical is not None and all(len(e) == 2 for v in vectors for e in v)
        if dep not in (0, 2) and not pairs:
            raise FamilyError(f"{what}: 'vectors' must be a list of vectors (entries: numbers, strings, "
                              f"or [a, b] pairs with 'sqrt')")
        for v in vectors:
            rows.append(parse_vector(v, radical=radical, scale=scale_q))
    if rows:
        m = len(rows[0])
        if m == 0 or any(len(r) != m for r in rows):
            raise FamilyError(f"{what}: rows must all have the same positive length")
    claimed = _get(block, "count", "K")
    if claimed is not None and int(claimed) != len(rows):
        raise FamilyError(f"{what}: claims {claimed} rows but has {len(rows)}")
    return rows, meta


def load_family(path: str) -> Family:
    fam = Family()
    if os.path.isdir(path):
        fam.dir = path
        jpath = os.path.join(path, "family.json")
    else:
        fam.dir = os.path.dirname(os.path.abspath(path))
        jpath = path
    with open(jpath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise FamilyError("family.json: top level must be an object")
    fam.raw = raw
    dim = _get(raw, "dim", "n")
    d = raw.get("d")
    if dim is None and d is None:
        raise FamilyError("family.json: needs 'dim' (= 24 + d) or 'd'")
    dim = int(dim) if dim is not None else 24 + int(d)
    d = int(d) if d is not None else dim - 24
    if dim != 24 + d:
        raise FamilyError(f"family.json: dim {dim} != 24 + d {d}")
    if d < 1:
        raise FamilyError(f"family.json: d = {d} < 1")
    fam.dim, fam.d = dim, d

    sets = raw.get("sets")
    if not isinstance(sets, list) or not sets:
        raise FamilyError("family.json: 'sets' must be a non-empty list")
    per_set_groups: list = []
    for k, s in enumerate(sets):
        if isinstance(s, str):
            s = {"file": s}
        if not isinstance(s, dict) or not _get(s, "file", "path"):
            raise FamilyError(f"family.json: sets[{k}] has no 'file'")
        fname = _get(s, "file", "path")
        fam.set_files.append(fname if os.path.isabs(fname) else os.path.normpath(os.path.join(fam.dir, fname)))
        fam.set_meta.append(s)
        per_set_groups.append(_get(s, "T", "group", "T_index", "triangle", "pair"))

    T_block = _get(raw, "T", "t", "T_vectors")
    groups = None
    if isinstance(T_block, dict):
        groups = T_block.get("groups")
    elif isinstance(T_block, (list, tuple)) and _depth(T_block) == 3:
        # list of groups, each a list of vectors
        vecs, groups, seen = [], [], {}
        for g in T_block:
            gi = []
            for v in g:
                key = json.dumps(v)
                if key not in seen:
                    seen[key] = len(vecs)
                    vecs.append(v)
                gi.append(seen[key])
            groups.append(gi)
        T_block = {"vectors": vecs}
    fam.T, fam.T_meta = parse_vector_block(T_block, "T")
    if groups is None:
        if all(g is not None for g in per_set_groups):
            if all(_depth(g) == 1 for g in per_set_groups):
                groups = [list(map(int, g)) for g in per_set_groups]
            else:  # per-set vectors
                vecs = [json.dumps(v) for v in (T_block.get("vectors", []) if isinstance(T_block, dict) else [])]
                groups = []
                for g in per_set_groups:
                    gi = []
                    for v in g:
                        key = json.dumps(v)
                        if key not in vecs:
                            vecs.append(key)
                            fam.T.append(parse_vector(v))
                        gi.append(vecs.index(key))
                    groups.append(gi)
        else:
            raise FamilyError("family.json: no T groups (T.groups or a per-set 'T'/'group' entry)")
    try:
        fam.groups = [[int(i) for i in g] for g in groups]
    except (TypeError, ValueError):
        raise FamilyError("family.json: T.groups must be lists of integers") from None

    fam.extras, fam.extra_meta = parse_vector_block(_get(raw, "extra", "extras"), "extra")
    if fam.extras and fam.T and len(fam.extras[0]) != len(fam.T[0]):
        raise FamilyError(f"extra rows have {len(fam.extras[0])} coordinates, T rows {len(fam.T[0])}")
    c = raw.get("count")
    fam.count_claimed = int(c) if c is not None else None
    return fam


# ---------------------------------------------------------------------------
# exact float32 GEMM maxima over big integer matrices (see verify_dim25.exact_full_pass)
# ---------------------------------------------------------------------------
def cross_max(A: np.ndarray, B: np.ndarray, rb: int = 1024, cb: int = 32768, workers: int | None = None) -> tuple[int, tuple[int, int]]:
    """max of A B^T over all entries (int rows with |entries| <= 4 -> exact in float32)."""
    Af = np.ascontiguousarray(A, dtype=np.float32)
    BT = np.ascontiguousarray(np.asarray(B, dtype=np.float32).T)
    lock = threading.Lock()
    state = {"max": -np.inf, "pair": (-1, -1)}

    def task(i0: int) -> None:
        Ai = Af[i0:i0 + rb]
        best, pair = -np.inf, (-1, -1)
        for j0 in range(0, BT.shape[1], cb):
            G = Ai @ BT[:, j0:j0 + cb]
            m = float(G.max())
            if m > best:
                k = int(np.argmax(G))
                best, pair = m, (i0 + k // G.shape[1], j0 + k % G.shape[1])
        with lock:
            if best > state["max"]:
                state["max"], state["pair"] = best, pair

    with ThreadPoolExecutor(max_workers=workers or default_workers()) as ex:
        list(ex.map(task, range(0, len(Af), rb)))
    return int(state["max"]), state["pair"]


def union_gram_maxima(U: np.ndarray, sid: np.ndarray, rb: int = 2048, cb: int = 4096,
                      workers: int | None = None) -> dict:
    """Max off-diagonal of U U^T split into same-set and cross-set pairs (sid = set id per row)."""
    Uf = np.ascontiguousarray(U, dtype=np.float32)
    UT = np.ascontiguousarray(Uf.T)
    lock = threading.Lock()
    state = {"same": -np.inf, "same_pair": (-1, -1), "cross": -np.inf, "cross_pair": (-1, -1)}
    n = len(Uf)

    def task(i0: int) -> None:
        ni = min(rb, n - i0)
        Ai = Uf[i0:i0 + ni]
        loc = {"same": -np.inf, "same_pair": (-1, -1), "cross": -np.inf, "cross_pair": (-1, -1)}
        for j0 in range(i0, n, cb):
            nj = min(cb, n - j0)
            G = Ai @ UT[:, j0:j0 + nj]
            if not np.all(np.isin(G, (-32.0, -16.0, -8.0, 0.0, 8.0, 16.0, 32.0))):
                raise AssertionError("inner product outside the minimal-vector classes")
            same = sid[i0:i0 + ni, None] == sid[None, j0:j0 + nj]
            off = j0 - i0
            if off < ni:
                np.fill_diagonal(G[off:, :], -np.inf)  # entries (i0+off+k, j0+k)
            Gs = np.where(same, G, -np.inf)
            Gc = np.where(same, -np.inf, G)
            for key, M in (("same", Gs), ("cross", Gc)):
                m = float(M.max())
                if m > loc[key]:
                    k = int(np.argmax(M))
                    loc[key], loc[key + "_pair"] = m, (i0 + k // nj, j0 + k % nj)
        with lock:
            for key in ("same", "cross"):
                if loc[key] > state[key]:
                    state[key], state[key + "_pair"] = loc[key], loc[key + "_pair"]

    with ThreadPoolExecutor(max_workers=workers or default_workers()) as ex:
        list(ex.map(task, range(0, n, rb)))
    return state


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cosf(p: Q, Na: Q, Nb: Q) -> float:
    return float(p) / math.sqrt(float(Na) * float(Nb))


def _deg(c: float) -> str:
    return f"{math.degrees(math.acos(max(-1.0, min(1.0, c)))):.2f} deg"


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------
def verify_family(fam: Family, full_pass: bool = True, do_float: bool = True, rb: int = 512, cb: int = 1024,
                  workers: int | None = None, strict: bool = False, log=print) -> dict:
    """Run every check. Returns a dict with ok / message / count / count_exact / count_float / details."""
    t_start = time.time()
    out: dict = {"ok": False, "message": "", "dim": fam.dim, "d": fam.d, "count": None, "count_exact": None,
                 "count_float": None, "warnings": list(fam.warnings), "pairs": {}}
    d = fam.d
    index = leech_index()
    C = index.C
    workers = workers or default_workers()

    def fail(msg: str) -> dict:
        out["message"] = msg
        log(f"FAIL        : {msg}")
        return out

    # ---- A. the sets ------------------------------------------------------
    t0 = time.time()
    idx_list: list[np.ndarray] = []
    for k, path in enumerate(fam.set_files):
        meta = fam.set_meta[k]
        try:
            S, lines = read_set(path)
        except (OSError, ValueError) as e:
            return fail(f"set {k + 1}: parse error: {e}")
        res = check_set(S, lines, index)
        if not res["ok"]:
            return fail(f"set {k + 1} ({os.path.basename(path)}): {res['message']}")
        if meta.get("size") is not None and int(meta["size"]) != res["size"]:
            return fail(f"set {k + 1}: file has {res['size']} rows, family.json claims size {meta['size']}")
        if meta.get("sha256"):
            h = sha256_file(path)
            if h != meta["sha256"]:
                msg = f"set {k + 1}: sha256 {h[:16]}... != claimed {str(meta['sha256'])[:16]}..."
                if strict:
                    return fail(msg)
                out["warnings"].append(msg)
        idx_list.append(np.asarray(res["idx"], dtype=np.int64))
    sizes = [len(i) for i in idx_list]
    all_idx = np.concatenate(idx_list)
    sid = np.concatenate([np.full(len(i), k, dtype=np.int32) for k, i in enumerate(idx_list)])
    order = np.argsort(all_idx, kind="stable")
    dup = np.flatnonzero(all_idx[order][1:] == all_idx[order][:-1])
    if len(dup):
        a, b = int(order[dup[0]]), int(order[dup[0] + 1])
        return fail(f"sets {sid[a] + 1} and {sid[b] + 1} overlap: they share the vector "
                    f"({' '.join(str(int(v)) for v in C[all_idx[a]])}) [{len(dup)} shared vector(s)]")
    in_union = np.zeros(N_MIN, dtype=bool)
    in_union[all_idx] = True
    eq_idx = np.flatnonzero(~in_union)
    n_eq = len(eq_idx)
    n_union = len(all_idx)
    if n_eq + n_union != N_MIN:
        return fail("equatorial set is not exactly C \\ U (internal)")
    log(f"sets        : k={len(sizes)} sizes={sizes} union={n_union} pairwise disjoint, each independent; "
        f"equatorial = C \\ U = {n_eq}  ({time.time() - t0:.1f} s)")

    # ---- B. the R^d kissing configuration T and its groups --------------
    t0 = time.time()
    T = fam.T
    if not T:
        return fail("T: no vectors")
    m = len(T[0])
    NT = [norm2(t) for t in T]
    for i, N in enumerate(NT):
        if N.is_zero():
            return fail(f"T: row {i} is the zero vector")
    nT = len(T)
    G_T = [[None] * nT for _ in range(nT)]
    max_cos_T = -2.0
    for i in range(nT):
        for j in range(i + 1, nT):
            p = dot(T[i], T[j])
            G_T[i][j] = G_T[j][i] = p
            if is_same_direction(p, NT[i], NT[j]):
                return fail(f"T: rows {i} and {j} have the same direction")
            if not cos_le_half(p, NT[i], NT[j]):
                c = _cosf(p, NT[i], NT[j])
                return fail(f"T: rows {i} and {j} have cos = {c:.6f} > 1/2 ({_deg(c)}); ip {p}, norms {NT[i]}, {NT[j]}")
            max_cos_T = max(max_cos_T, _cosf(p, NT[i], NT[j]))
    groups = fam.groups
    if len(groups) != len(sizes):
        return fail(f"groups: {len(groups)} T groups for {len(sizes)} sets (need one group per S_i)")
    used: dict[int, int] = {}
    for gi, g in enumerate(groups):
        if len(g) not in (2, 3):
            return fail(f"group {gi + 1}: size {len(g)} (need an antipodal pair or an equilateral triangle)")
        for i in g:
            if not 0 <= i < nT:
                return fail(f"group {gi + 1}: T index {i} out of range")
            if i in used:
                return fail(f"group {gi + 1}: T vector {i} already used by group {used[i] + 1}")
            used[i] = gi
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                i, j = g[a], g[b]
                p = G_T[i][j]
                if not cos_le_neg_half(p, NT[i], NT[j]):
                    c = _cosf(p, NT[i], NT[j])
                    return fail(f"group {gi + 1} (for S_{gi + 1}): T vectors {i} and {j} have cos = {c:.6f} > -1/2 "
                                f"({_deg(c)}); a group needs pairwise cos <= -1/2")
        if len(g) == 2 and not is_antipodal(G_T[g[0]][g[1]], NT[g[0]], NT[g[1]]):
            # cos = -1/2 exactly (two vertices of a triangle): allowed by the template and by
            # SCHEMA.md 3.3 (PackingStar's 29D family uses such pairs); noted, never a failure.
            c = _cosf(G_T[g[0]][g[1]], NT[g[0]], NT[g[1]])
            out["warnings"].append(f"group {gi + 1}: pair {g} is not antipodal (cos = {c:.6f}, {_deg(c)}; two "
                                   f"vertices of a triangle is allowed by the template and SCHEMA.md 3.3)")
    unused = [i for i in range(nT) if i not in used]
    if unused:
        out["warnings"].append(f"T: {len(unused)} vector(s) not in any group {unused[:10]}{'...' if len(unused) > 10 else ''}")
    norm_set = sorted({str(N) for N in NT})
    if strict and fam.T_meta.get("norm2") is not None and norm_set != [str(Q(int(fam.T_meta["norm2"])))]:
        return fail(f"T: norms {norm_set} != claimed norm2 {fam.T_meta['norm2']}")
    if strict and fam.T_meta.get("K") is not None and int(fam.T_meta["K"]) != nT:
        return fail(f"T: K claimed {fam.T_meta['K']} != {nT} rows")
    sizes_T = [len(g) for g in groups]
    log(f"T           : {nT} vectors in Z^{m}-ambient field, norms {norm_set}, pairwise cos <= 1/2 (max {max_cos_T:.6f}); "
        f"groups: {sum(1 for g in groups if len(g) == 3)} triangles + {sum(1 for g in groups if len(g) == 2)} pairs, "
        f"disjoint, within-group cos <= -1/2  ({time.time() - t0:.1f} s)")

    # ---- C. extra spheres ----------------------------------------------
    t0 = time.time()
    E = fam.extras
    NE = [norm2(e) for e in E]
    for i, N in enumerate(NE):
        if N.is_zero():
            return fail(f"extra: row {i} is the zero vector")
    max_cos_EE, max_cos_ET = -2.0, -2.0
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            p = dot(E[i], E[j])
            if is_same_direction(p, NE[i], NE[j]):
                return fail(f"extra: rows {i} and {j} have the same direction")
            if not cos_le_half(p, NE[i], NE[j]):
                c = _cosf(p, NE[i], NE[j])
                return fail(f"extra: rows {i} and {j} have cos = {c:.6f} > 1/2 ({_deg(c)})")
            max_cos_EE = max(max_cos_EE, _cosf(p, NE[i], NE[j]))
        for j in range(nT):
            p = dot(E[i], T[j])
            if not cos_le_sqrt3_half(p, NE[i], NT[j]):
                c = _cosf(p, NE[i], NT[j])
                return fail(f"extra {i} is within 30 degrees of T vector {j}: cos = {c:.6f} > sqrt3/2 ({_deg(c)}); "
                            f"ip {p}, norms {NE[i]}, {NT[j]}")
            max_cos_ET = max(max_cos_ET, _cosf(p, NE[i], NT[j]))
    if strict and E and fam.extra_meta.get("count") is not None and int(fam.extra_meta["count"]) != len(E):
        return fail("extra: count mismatch")
    if strict and E and sorted({str(N) for N in NE}) != norm_set:
        return fail(f"extra: norms {sorted({str(N) for N in NE})} != T norms {norm_set} (SCHEMA.md 4)")
    n_ex = len(E)
    kd = KISSING.get(d)
    log(f"extra       : {n_ex} vectors (K({d}) = {kd}), pairwise cos <= 1/2 (max {max_cos_EE:.6f}), "
        f"cos to every T vector <= sqrt3/2 (max {max_cos_ET:.6f})  ({time.time() - t0:.1f} s)")
    if kd is not None and n_ex != kd:
        out["warnings"].append(f"extra: {n_ex} extra spheres, the record form uses K({d}) = {kd}")

    # ---- D. rank of the R^d data --------------------------------------
    r = rank(T + E)
    if r > d:
        return fail(f"T and extra vectors span a {r}-dimensional space > d = {d}")
    log(f"rank        : span(T u extra) has exact rank {r} <= d = {d}")

    # ---- E. the count -----------------------------------------------------
    n_lift = sum(s * t for s, t in zip(sizes, sizes_T))
    lifted_terms = sum((t - 1) * s for s, t in zip(sizes, sizes_T))
    count = n_ex + N_MIN + lifted_terms
    formula = f"{n_ex} + {N_MIN} + " + " + ".join(f"{t - 1}*{s}" for s, t in zip(sizes, sizes_T))
    out["count"] = count
    out["formula"] = f"{n_ex} + 196560 + sum_i (|T_i|-1)|S_i| = {n_ex} + 196560 + {lifted_terms}"
    log(f"count       : #extra + 196560 + sum_i (|T_i|-1)|S_i| = {n_ex} + 196560 + {lifted_terms} = {count}"
        f"  (rows: equatorial {n_eq} + lifted {n_lift} + extra {n_ex})")
    if fam.count_claimed is not None and fam.count_claimed != count:
        return fail(f"count: family.json claims {fam.count_claimed}, the data give {count}")
    ct = fam.raw.get("count_terms")
    if isinstance(ct, dict):
        claimed = (ct.get("extra"), ct.get("leech"), ct.get("lifted"))
        if any(v is not None for v in claimed) and claimed != (n_ex, N_MIN, lifted_terms):
            return fail(f"count_terms: claims {claimed}, data give {(n_ex, N_MIN, lifted_terms)}")

    # ---- F. exact casework on the assembled configuration ---------------
    t0 = time.time()
    pairs: dict[str, int] = {}
    pairs["eq-eq"] = n_eq * (n_eq - 1) // 2
    pairs["eq-lift"] = n_eq * n_lift
    pairs["lift-lift-same-x"] = sum(s * (t * (t - 1) // 2) for s, t in zip(sizes, sizes_T))
    pairs["lift-lift-same-set"] = sum((s * (s - 1) // 2) * t * t for s, t in zip(sizes, sizes_T))
    pairs["lift-lift-cross-set"] = (n_lift * n_lift - sum((s * t) ** 2 for s, t in zip(sizes, sizes_T))) // 2
    pairs["extra-eq"] = n_ex * n_eq
    pairs["extra-lift"] = n_ex * n_lift
    pairs["extra-extra"] = n_ex * (n_ex - 1) // 2
    out["pairs"] = pairs
    if sum(pairs.values()) != count * (count - 1) // 2:
        return fail(f"pair bookkeeping: {sum(pairs.values())} != C({count}, 2)")

    U = C[all_idx].astype(np.int64)
    # eq-lift: max ip over U x C[eq]; need ip <= 8 sqrt6 = 19.6, i.e. ip <= 16 (32 would mean overlap)
    max_el, pel = cross_max(U, C[eq_idx], workers=workers)
    if max_el > 16:
        return fail(f"eq-lift: ip {max_el} > 16 (a lifted vector coincides with an equatorial one?)")
    # lift-lift, x != x': same set needs ip <= 8 (checked per set), cross set needs ip <= 16
    ug = union_gram_maxima(U, sid, workers=workers)
    max_same = int(ug["same"]) if np.isfinite(ug["same"]) else None
    max_cross = int(ug["cross"]) if np.isfinite(ug["cross"]) else None
    if max_same is not None and max_same > 8:
        return fail(f"lift-lift same set: ip {max_same} > 8")
    if max_cross is not None and max_cross > 16:
        return fail(f"lift-lift cross set: ip {max_cross} > 16")
    log(f"exact       : eq-lift max ip = {max_el} (needs <= 16 < 8 sqrt6); lift-lift max ip same set = {max_same} "
        f"(needs <= 8), cross set = {max_cross} (needs <= 16; 16 with cos(y,y') = 1/2 is exactly 2)")
    # eq-eq (and the cross-set bound again): ip <= 16 for all distinct pairs of C
    if full_pass:
        fp = exact_full_pass(C, rb=rb, cb=cb, workers=workers)
        if fp["max"] > 16 or not fp["classes_ok"]:
            return fail(f"eq-eq: full pass found ip {fp['max']} > 16 or a bad class")
        log(f"exact       : eq-eq full pass over all {fp['pairs']} pairs of C: max off-diagonal ip = {int(fp['max'])}, "
            f"min = {int(fp['min'])}, values seen {fp['classes_seen']} ({fp['seconds']:.1f} s)")
    else:
        log("exact       : eq-eq: ip <= 16 for distinct minimal vectors taken from the Leech minimal norm "
            "(|x-x'|^2 = 64 - 2 ip >= 32), full pass skipped (--skip-full-equatorial)")
    log("exact       : lift-lift same x: within-group cos <= -1/2 (B); extra-lift: cos <= sqrt3/2 (C); "
        "extra-extra: cos <= 1/2 (C); extra-eq: 0")
    log("exact       : pairs " + " ".join(f"{k}={v}" for k, v in pairs.items()) +
        f" total={sum(pairs.values())}=C({count},2)")
    out["count_exact"] = count
    out["exact_s"] = time.time() - t0
    out["max_ip_eq_lift"], out["max_ip_same_set"], out["max_ip_cross_set"] = max_el, max_same, max_cross
    log(f"exact       : count = {count}  ({out['exact_s']:.1f} s)")

    # ---- G. float pass ----------------------------------------------------
    if not do_float:
        out.update(ok=True, message="ok", count_float=None, seconds=time.time() - t_start)
        return out
    t0 = time.time()
    X, meta = assemble_coordinates(C, idx_list, T, groups, E, d, eq_idx=eq_idx)
    norms = (X * X).sum(axis=1)
    if not np.all(np.abs(norms - 4.0) <= 1e-9):
        return fail(f"float: a vector has norm {norms[np.argmax(np.abs(norms - 4.0))]!r} != 4")
    st = pairwise_offdiag_max(X, rb=rb, cb=cb, workers=workers)
    best, pair = st["max"], st["pair"]
    cls = pair_class(pair[0], pair[1], meta)
    out["max_offdiag"], out["max_pair"], out["max_class"], out["min_offdiag"] = best, pair, cls, st["min"]
    out["float_s"] = time.time() - t0
    if best > 2.0 + 1e-9:
        return fail(f"float: max off-diagonal inner product {best!r} > 2 at pair {pair} class={cls}")
    log(f"float       : {len(X)} vectors in R^{24 + d}; max off-diagonal inner product = {best:.12f} (tol 2+1e-9) "
        f"at pair {pair} class={cls}; min {st['min']:.6f}  ({out['float_s']:.1f} s, {st['tiles']} tiles)")
    out["count_float"] = len(X)
    if len(X) != count:
        return fail(f"float: {len(X)} rows != count {count}")
    out.update(ok=True, message="ok", seconds=time.time() - t_start)
    return out


def assemble_coordinates(C: np.ndarray, idx_list: list[np.ndarray], T: list[list[Q]], groups: list[list[int]],
                         E: list[list[Q]], d: int, eq_idx: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Float64 rows of the R^{24+d} configuration and a row -> (kind, set, T index) map."""
    n = len(C)
    if eq_idx is None:
        in_union = np.zeros(n, dtype=bool)
        for i in idx_list:
            in_union[i] = True
        eq_idx = np.flatnonzero(~in_union)
    rows_d = np.array([[float(x) for x in r] for r in T + E], dtype=np.float64)
    Y = orthonormal_embedding(rows_d, d)
    Y /= np.sqrt((Y * Y).sum(axis=1))[:, None]  # unit vectors
    YT, YE = Y[:len(T)], Y[len(T):]
    n_eq = len(eq_idx)
    n_lift = sum(len(i) * len(g) for i, g in zip(idx_list, groups))
    n_ex = len(E)
    X = np.zeros((n_eq + n_lift + n_ex, DIM + d), dtype=np.float64)
    kind = np.zeros(len(X), dtype=np.int8)
    setid = np.full(len(X), -1, dtype=np.int32)
    tid = np.full(len(X), -1, dtype=np.int32)
    xid = np.full(len(X), -1, dtype=np.int64)
    X[:n_eq, :DIM] = C[eq_idx].astype(np.float64) / SQRT8
    xid[:n_eq] = eq_idx
    r = n_eq
    for k, (idx, g) in enumerate(zip(idx_list, groups)):
        lifted = C[idx].astype(np.float64) / SQRT8 * LIFT_X
        for t in g:
            X[r:r + len(idx), :DIM] = lifted
            X[r:r + len(idx), DIM:] = LIFT_Y * YT[t]
            kind[r:r + len(idx)] = 1
            setid[r:r + len(idx)] = k
            tid[r:r + len(idx)] = t
            xid[r:r + len(idx)] = idx
            r += len(idx)
    if n_ex:
        X[r:, DIM:] = 2.0 * YE
        kind[r:] = 2
        tid[r:] = np.arange(n_ex)
    return X, {"kind": kind, "set": setid, "t": tid, "x": xid, "n_eq": n_eq, "n_lift": n_lift, "n_ex": n_ex}


def pair_class(i: int, j: int, meta: dict) -> str:
    ki, kj = int(meta["kind"][i]), int(meta["kind"][j])
    a, b = sorted((ki, kj))
    if (a, b) == (0, 0):
        return "eq-eq"
    if (a, b) == (0, 1):
        return "eq-lift"
    if (a, b) == (1, 1):
        if meta["set"][i] != meta["set"][j]:
            return "lift-lift-cross-set"
        return "lift-lift-same-x" if meta["x"][i] == meta["x"][j] else "lift-lift-same-set"
    if (a, b) == (0, 2):
        return "extra-eq"
    if (a, b) == (1, 2):
        return "extra-lift"
    return "extra-extra"


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a dim-(24+d) family directory (data/families/SCHEMA.md).")
    ap.add_argument("family", help="family directory (with family.json) or a family.json path")
    ap.add_argument("--tile", type=int, nargs=2, default=(512, 1024), metavar=("ROWS", "COLS"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-full-equatorial", action="store_true",
                    help="rely on the Leech minimal-norm argument for eq-eq instead of the full pass")
    ap.add_argument("--skip-float", action="store_true", help="skip method (b)")
    ap.add_argument("--strict", action="store_true",
                    help="also enforce SCHEMA.md's exact forms (sha256, sizes, antipodal pairs, extra norms)")
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        fam = load_family(args.family)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"family error: {e}")
        print(f"RESULT ok=0 reason={quote('family: ' + str(e))}")
        return 1
    print(f"family      : {args.family}  dim={fam.dim} d={fam.d} sets={len(fam.set_files)} T={len(fam.T)} "
          f"groups={len(fam.groups)} extra={len(fam.extras)} claimed_count={fam.count_claimed}")
    print(f"config      : cpus={os.cpu_count()} workers={args.workers or default_workers()} tile={tuple(args.tile)} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    rb, cb = args.tile
    try:
        res = verify_family(fam, full_pass=not args.skip_full_equatorial, do_float=not args.skip_float,
                            rb=rb, cb=cb, workers=args.workers, strict=args.strict)
    except FamilyError as e:
        res = {"ok": False, "message": str(e), "dim": fam.dim, "count": None}
        print(f"FAIL        : {e}")
    for w in res.get("warnings", []):
        print(f"warning     : {w}")
    if not res["ok"]:
        print(f"RESULT ok=0 dim={fam.dim} d={fam.d} count={res.get('count')} reason={quote(res['message'])} "
              f"s={time.time() - t0:.1f}")
        return 1
    cf = res["count_float"] if res["count_float"] is not None else "skipped"
    mo = f"{res['max_offdiag']:.12f}" if "max_offdiag" in res else "skipped"
    print(f"RESULT ok=1 dim={fam.dim} d={fam.d} sets={len(fam.set_files)} count={res['count']} "
          f"count_exact={res['count_exact']} count_float={cf} max_offdiag={mo} "
          f"max_class={res.get('max_class', '-')} extra={len(fam.extras)} K={KISSING.get(fam.d)} "
          f"exact_s={res.get('exact_s', 0):.1f} float_s={res.get('float_s', 0):.1f} s={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

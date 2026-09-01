"""Explicit integer-scaled kissing configurations in R^d, d = 2..7 (docs/design.md T4.3).

The R^d half of the CJKT / PackingStar template (README section 1.3): a kissing
configuration of K(d) unit vectors (pairwise cos <= 1/2), partitioned into
"groups" T_i (antipodal pairs, cos = -1, or equilateral triangles, cos = -1/2),
plus K(d) "extra" directions y' with cos <= 1/2 among themselves and
cos <= sqrt(3)/2 (angle >= 30 degrees) to every T-vector.

Everything here is EXACT: every configuration is stored as an integer matrix
whose rows all have the same squared norm ``norm2``; the ambient dimension m
may exceed d (the rows span a d-dimensional subspace), which is harmless for
all the angle conditions (they only use inner products) and is undone by
``embed_in_Rd`` / ``orthonormal_embedding`` for the float half of the verifiers.

    d  K(d)  model                              ambient  norm2
    2    6   A2 roots  (1,-1,0) perms, sum 0       3       2
    3   12   A3 = D3 roots  (+-1,+-1,0) perms      3       2
    4   24   D4 roots                              4       2
    5   40   D5 roots                              5       2
    6   72   E8 roots orthogonal to an A2 pair     8       8
    7  126   E8 roots orthogonal to one root       8       8

(E8 roots are taken in the standard 8-coordinate form with the (+-1/2)^8
vectors scaled by 2: (+-2,+-2,0^6) and (+-1)^8 with an even number of minus
signs, norm 8.)

Angle predicates on integer vectors a, b with p = <a,b>, Na = |a|^2, Nb = |b|^2
(norms may differ; every comparison is a sign-aware square comparison):

    cos <= 1/2         <=>  p <= 0  or  4 p^2 <= Na Nb
    cos <= -1/2        <=>  p <  0  and 4 p^2 >= Na Nb
    cos <= sqrt(3)/2   <=>  p <= 0  or  4 p^2 <= 3 Na Nb
    cos == -1          <=>  p <  0  and   p^2 == Na Nb
    cos == +1          <=>  p >  0  and   p^2 == Na Nb

Also provided: all "triangles" (triples with pairwise cos = -1/2), an exact
DFS search for a partition into a requested number of disjoint triangles with
the leftover matched into antipodal pairs, a search for extra directions, and
an isometry finder that maps a float copy of the configuration (e.g.
PackingStar's) onto the integer model.

Pure numpy; no dependency on anything else in the repository.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

__all__ = [
    "K", "AMBIENT", "NAMES", "config", "gram", "orthonormal_embedding", "embed_in_Rd",
    "cmp_sqrt", "cos_le_half", "cos_le_neg_half", "cos_le_sqrt3_half", "is_antipodal", "is_same_direction",
    "check_kissing", "antipodal_pairs", "triangles", "triangles_of", "triangle_partition", "max_triangle_partition",
    "extra_sphere_candidates", "lattice_extra_spheres", "rational_rotation", "rotated_extra_spheres",
    "rotated_extra_spheres_exact", "check_extra_ab", "extra_spheres", "check_extra", "isometry_to_model",
    "schema_T_block", "schema_extra_block",
]

K = {1: 2, 2: 6, 3: 12, 4: 24, 5: 40, 6: 72, 7: 126}
AMBIENT = {2: 3, 3: 3, 4: 4, 5: 5, 6: 8, 7: 8}
NAMES = {2: "A2", 3: "A3=D3", 4: "D4", 5: "D5", 6: "E6", 7: "E7"}

# Fixed roots defining the E7 / E6 subsystems of E8 (see config()).
_E8_R1 = np.array([2, -2, 0, 0, 0, 0, 0, 0], dtype=np.int64)
_E8_R2 = np.array([0, 2, -2, 0, 0, 0, 0, 0], dtype=np.int64)


# ---------------------------------------------------------------------------
# the configurations
# ---------------------------------------------------------------------------
def _lex_sorted(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.int64)
    order = np.lexsort(V.T[::-1])  # coordinate 0 most significant
    return np.ascontiguousarray(V[order])


def _d_roots(n: int) -> np.ndarray:
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * n
                v[i], v[j] = si, sj
                rows.append(v)
    return np.array(rows, dtype=np.int64)


@lru_cache(maxsize=None)
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
    R = np.array(rows, dtype=np.int64)
    assert R.shape == (240, 8) and np.all((R * R).sum(axis=1) == 8)
    return R


@lru_cache(maxsize=None)
def _config_cached(d: int) -> tuple[np.ndarray, int]:
    if d == 2:
        V = np.array([[1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1]], dtype=np.int64)
        norm2 = 2
    elif d in (3, 4, 5):
        V, norm2 = _d_roots(d), 2
    elif d == 7:
        R = _e8_roots()
        V, norm2 = R[R @ _E8_R1 == 0], 8
    elif d == 6:
        R = _e8_roots()
        V, norm2 = R[(R @ _E8_R1 == 0) & (R @ _E8_R2 == 0)], 8
    else:
        raise ValueError(f"no configuration for d={d} (need 2 <= d <= 7)")
    V = _lex_sorted(V)
    V.setflags(write=False)
    assert V.shape == (K[d], AMBIENT[d]) and np.all((V * V).sum(axis=1) == norm2)
    return V, norm2


def config(d: int) -> tuple[np.ndarray, int]:
    """(vectors: int64 array (K(d), m), norm2): the K(d)-point kissing configuration, exact."""
    V, norm2 = _config_cached(d)
    return V.copy(), norm2


def gram(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.int64)
    return V @ V.T


# ---------------------------------------------------------------------------
# exact angle predicates (scalars or numpy arrays; all integer arithmetic)
# ---------------------------------------------------------------------------
def cmp_sqrt(a, b, M) -> int:
    """Sign of a - b*sqrt(M) for Python integers a, b and integer M >= 0 (exact)."""
    a, b, M = int(a), int(b), int(M)
    if M < 0:
        raise ValueError("M must be >= 0")
    r = b * b * M  # (b sqrt M)^2
    if b >= 0:
        if a <= 0:
            return 0 if (a == 0 and r == 0) else -1
        return (a * a > r) - (a * a < r)
    # b < 0: b sqrt M <= 0
    if a >= 0:
        return 0 if (a == 0 and r == 0) else 1
    return (r > a * a) - (r < a * a)  # both negative: a <= b sqrt M  <=>  a^2 >= r


def _ints(p, Na, Nb):
    """Cast to int64, or to Python ints (object dtype) when 4 p^2 or 3 Na Nb could overflow int64."""
    p, Na, Nb = np.asarray(p), np.asarray(Na), np.asarray(Nb)
    big = any(a.dtype == object for a in (p, Na, Nb))
    if not big:
        p, Na, Nb = p.astype(np.int64), Na.astype(np.int64), Nb.astype(np.int64)
        lim = 2**31 - 1  # then 4 p^2 < 2^64 / 4 and 3 Na Nb < 3 * 2^62 ... keep a safe margin
        if (p.size and np.abs(p).max() > lim // 2) or (Na.size and Nb.size and
                                                        int(Na.max()) * int(Nb.max()) > 2**60):
            big = True
    if big:
        p, Na, Nb = p.astype(object), Na.astype(object), Nb.astype(object)
    return p, Na, Nb


def cos_le_half(p, Na, Nb):
    """cos(a,b) <= 1/2 with p = <a,b>, Na = |a|^2, Nb = |b|^2 (arrays ok)."""
    p, Na, Nb = _ints(p, Na, Nb)
    return np.asarray((p <= 0) | (4 * p * p <= Na * Nb), dtype=bool)


def cos_le_neg_half(p, Na, Nb):
    p, Na, Nb = _ints(p, Na, Nb)
    return np.asarray((p < 0) & (4 * p * p >= Na * Nb), dtype=bool)


def cos_le_sqrt3_half(p, Na, Nb):
    p, Na, Nb = _ints(p, Na, Nb)
    return np.asarray((p <= 0) | (4 * p * p <= 3 * Na * Nb), dtype=bool)


def is_antipodal(p, Na, Nb):
    p, Na, Nb = _ints(p, Na, Nb)
    return np.asarray((p < 0) & (p * p == Na * Nb), dtype=bool)


def is_same_direction(p, Na, Nb):
    p, Na, Nb = _ints(p, Na, Nb)
    return np.asarray((p > 0) & (p * p == Na * Nb), dtype=bool)


def check_kissing(V: np.ndarray) -> dict:
    """Pairwise cos <= 1/2 and distinct directions, for integer rows of any norms.

    Returns {"ok", "n", "max_cos", "touching_pairs", "message"}; on failure the
    first offending pair (i, j) is in the message.
    """
    V = np.asarray(V, dtype=np.int64)
    n = len(V)
    N = (V * V).sum(axis=1)
    if np.any(N == 0):
        return {"ok": False, "n": n, "message": f"zero vector at row {int(np.flatnonzero(N == 0)[0])}"}
    G = V @ V.T
    Na, Nb = N[:, None], N[None, :]
    off = ~np.eye(n, dtype=bool)
    bad = off & ~cos_le_half(G, Na, Nb)
    if bad.any():
        i, j = (int(v) for v in np.argwhere(bad)[0])
        return {"ok": False, "n": n, "message": f"rows {i},{j} have cos > 1/2 (ip {int(G[i, j])}, "
                                                 f"norms {int(N[i])},{int(N[j])})"}
    same = off & is_same_direction(G, Na, Nb)
    if same.any():
        i, j = (int(v) for v in np.argwhere(same)[0])
        return {"ok": False, "n": n, "message": f"rows {i},{j} have the same direction"}
    touching = off & (4 * G * G == Na * Nb) & (G > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = G / np.sqrt(Na * Nb).astype(np.float64)
    np.fill_diagonal(cosines, -np.inf)
    return {"ok": True, "n": n, "max_cos": float(cosines.max()) if n > 1 else None,
            "touching_pairs": int(touching.sum() // 2), "message": "ok"}


# ---------------------------------------------------------------------------
# float embedding of the span in exactly d coordinates
# ---------------------------------------------------------------------------
def orthonormal_embedding(V: np.ndarray, d: int, tol: float = 1e-9) -> np.ndarray:
    """Float64 (n, d) coordinates with the same Gram matrix as the integer rows V.

    V may be given in any ambient dimension m >= its rank; the rank must be
    <= d (raises otherwise), and the coordinates are padded with zeros if the
    rank is < d. Gram equality is verified to ``tol`` (relative to norm2).
    """
    V = np.asarray(V, dtype=np.float64)
    n, m = V.shape
    if n == 0:
        return np.zeros((0, d))
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    rank = int((s > 1e-9 * max(1.0, s[0])).sum())
    if rank > d:
        raise ValueError(f"vectors span a {rank}-dimensional space, more than d={d}")
    X = np.zeros((n, d))
    X[:, :rank] = U[:, :rank] * s[:rank]
    err = np.abs(X @ X.T - V @ V.T).max()
    scale = max(1.0, float(np.abs(V @ V.T).max()))
    if err > tol * scale:
        raise ArithmeticError(f"embedding Gram error {err}")
    return X


def embed_in_Rd(d: int, unit: bool = False) -> np.ndarray:
    """config(d) as float64 coordinates in exactly d dimensions (same Gram, or unit rows)."""
    V, norm2 = config(d)
    X = orthonormal_embedding(V, d)
    return X / math.sqrt(norm2) if unit else X


# ---------------------------------------------------------------------------
# antipodal pairs and triangles
# ---------------------------------------------------------------------------
def antipodal_pairs(V: np.ndarray) -> dict[int, int]:
    """{i: j} with V[j] = -V[i] (as directions), for every i that has an antipode."""
    V = np.asarray(V, dtype=np.int64)
    N = (V * V).sum(axis=1)
    G = V @ V.T
    anti = is_antipodal(G, N[:, None], N[None, :])
    out = {}
    for i in range(len(V)):
        js = np.flatnonzero(anti[i])
        if len(js):
            out[i] = int(js[0])
    return out


def triangles_of(V: np.ndarray) -> list[tuple[int, int, int]]:
    """All index triples with pairwise cos exactly -1/2 (coplanar equilateral, sum 0 for equal norms)."""
    V = np.asarray(V, dtype=np.int64)
    N = (V * V).sum(axis=1)
    G = V @ V.T
    neg = (G < 0) & (4 * G * G == N[:, None] * N[None, :])  # cos == -1/2 exactly
    n = len(V)
    out = []
    for a in range(n):
        bs = np.flatnonzero(neg[a])
        for b in bs:
            if b <= a:
                continue
            for c in np.flatnonzero(neg[a] & neg[b]):
                if c > b:
                    out.append((a, int(b), int(c)))
    return out


def triangles(d: int) -> list[tuple[int, int, int]]:
    return triangles_of(config(d)[0])


def triangle_partition(d: int, want: int, require_pairs: bool = True, node_limit: int = 5_000_000,
                       V: np.ndarray | None = None) -> dict | None:
    """Exact DFS: ``want`` pairwise disjoint triangles in config(d) (or in V).

    Points not covered by a chosen triangle are "leftover"; with
    ``require_pairs`` the leftover set must be a union of antipodal pairs
    (so every S_i gets a group of weight >= 1). Returns None if no such
    packing exists (exhaustive search) or if ``node_limit`` DFS nodes were
    exceeded (then ``triangle_partition.last_status == "limit"``), else
      {"triangles": [(a,b,c), ...], "pairs": [(i,j), ...], "leftover": [...],
       "nodes": int, "weight": 2*#triangles + #pairs}
    """
    V = config(d)[0] if V is None else np.asarray(V, dtype=np.int64)
    n = len(V)
    tris = triangles_of(V)
    anti = antipodal_pairs(V)
    tri_bits = [(1 << a) | (1 << b) | (1 << c) for a, b, c in tris]
    by_point: list[list[int]] = [[] for _ in range(n)]
    for k, (a, b, c) in enumerate(tris):
        by_point[a].append(k)
        by_point[b].append(k)
        by_point[c].append(k)
    max_skips = n - 3 * want
    if want < 0 or max_skips < 0:
        triangle_partition.last_status = "infeasible"
        return None
    state = {"nodes": 0}
    chosen: list[int] = []
    full = (1 << n) - 1

    class Limit(Exception):
        pass

    def dfs(covered: int, skipped: int, nskip: int) -> bool:
        state["nodes"] += 1
        if state["nodes"] > node_limit:
            raise Limit
        if len(chosen) == want:
            return True
        free = full & ~covered & ~skipped
        if free == 0:
            return False
        # bound: remaining free points must host the remaining triangles
        if bin(free).count("1") < 3 * (want - len(chosen)):
            return False
        p = (free & -free).bit_length() - 1  # lowest free point
        for k in by_point[p]:
            tb = tri_bits[k]
            if tb & (covered | skipped):
                continue
            if require_pairs:
                # a point whose antipode was skipped must itself be skipped
                bad = False
                for q in (tris[k]):
                    if q in anti and (skipped >> anti[q]) & 1:
                        bad = True
                        break
                if bad:
                    continue
            chosen.append(k)
            if dfs(covered | tb, skipped, nskip):
                return True
            chosen.pop()
        # skip p (leave it for the pairs)
        if nskip < max_skips:
            if require_pairs:
                q = anti.get(p)
                if q is None or (covered >> q) & 1:
                    return False  # p could never be paired
                sk = skipped | (1 << p) | (1 << q)
                add = 2 if not (skipped >> q) & 1 else 1
                if nskip + add > max_skips:
                    return False
                return dfs(covered, sk, nskip + add)
            return dfs(covered, skipped | (1 << p), nskip + 1)
        return False

    try:
        found = dfs(0, 0, 0)
    except Limit:
        triangle_partition.last_status = "limit"
        return None
    triangle_partition.last_status = "done"
    if not found:
        return None
    covered = 0
    for k in chosen:
        covered |= tri_bits[k]
    leftover = [i for i in range(n) if not (covered >> i) & 1]
    pairs, used = [], set()
    for i in leftover:
        if i in used:
            continue
        j = anti.get(i)
        if j is not None and j in leftover and j not in used:
            pairs.append((i, j))
            used.update((i, j))
    singles = [i for i in leftover if i not in used]
    return {"triangles": [tris[k] for k in chosen], "pairs": pairs, "leftover": leftover, "singles": singles,
            "nodes": state["nodes"], "weight": 2 * len(chosen) + len(pairs)}


triangle_partition.last_status = "none"


def max_triangle_partition(d: int, node_limit: int = 5_000_000, V: np.ndarray | None = None) -> dict:
    """Largest number of disjoint triangles found (exhaustive from K/3 downwards, subject to node_limit)."""
    n = K[d] if V is None else len(V)
    log = []
    for want in range(n // 3, -1, -1):
        res = triangle_partition(d, want, require_pairs=True, node_limit=node_limit, V=V)
        log.append((want, triangle_partition.last_status, res["nodes"] if res else None))
        if res is not None:
            res["want"] = want
            res["log"] = log
            res["exhaustive"] = all(st == "done" for _, st, _ in log)
            return res
    return {"triangles": [], "pairs": [], "want": 0, "log": log, "exhaustive": False}


# ---------------------------------------------------------------------------
# extra spheres (directions at >= 30 degrees from every T-vector, <= 60 degrees apart)
# ---------------------------------------------------------------------------
def _span_constraints(d: int) -> np.ndarray:
    """Rows c with <candidate, c> = 0 forcing candidates into the span of config(d)."""
    if d == 2:
        return np.array([[1, 1, 1]], dtype=np.int64)
    if d == 7:
        return _E8_R1[None, :]
    if d == 6:
        return np.stack([_E8_R1, _E8_R2])
    return np.zeros((0, AMBIENT[d]), dtype=np.int64)


def _span_grid(d: int, max_entry: int) -> np.ndarray:
    """All nonzero integer vectors of the span of config(d) with entries in [-max_entry, max_entry].

    The span constraints of the E8-ambient models are x0 = x1 (E7) and
    x0 = x1 = x2 (E6), of the A2 model x0 + x1 + x2 = 0; the grid is
    enumerated over the free coordinates only (7^7 = 823543 rows for E7 with
    max_entry 3 instead of 7^8).
    """
    vals = np.arange(-max_entry, max_entry + 1, dtype=np.int64)
    m = AMBIENT[d]
    if d in (6, 7):
        k = 3 if d == 6 else 2
        free = m - (k - 1)
        grids = np.meshgrid(*([vals] * free), indexing="ij")
        F = np.stack([g.ravel() for g in grids], axis=1)
        C = np.concatenate([np.repeat(F[:, :1], k, axis=1), F[:, 1:]], axis=1)
    elif d == 2:
        grids = np.meshgrid(vals, vals, indexing="ij")
        F = np.stack([g.ravel() for g in grids], axis=1)
        C = np.concatenate([F, -F.sum(axis=1, keepdims=True)], axis=1)
        C = C[np.abs(C[:, 2]) <= max_entry]
    else:
        grids = np.meshgrid(*([vals] * m), indexing="ij")
        C = np.stack([g.ravel() for g in grids], axis=1)
    C = C[np.any(C != 0, axis=1)]
    cons = _span_constraints(d)
    if len(cons):
        assert not np.any(C @ cons.T)
    return C


def extra_sphere_candidates(d: int, max_entry: int = 2, T: np.ndarray | None = None,
                            max_candidates: int | None = None, seed: int = 0) -> np.ndarray:
    """Primitive integer vectors of the span (entries in [-max_entry, max_entry]) at >= 30 deg from every T-vector.

    With ``max_candidates`` the list is cut to the vectors of smallest norm
    (ties broken at random with ``seed``), which keeps the boolean
    compatibility matrix of the clique search small.
    """
    T = config(d)[0] if T is None else np.asarray(T, dtype=np.int64)
    C = _span_grid(d, max_entry)
    g = np.gcd.reduce(np.abs(C), axis=1)
    C = C[g == 1]
    Nc = (C * C).sum(axis=1)
    Nt = (T * T).sum(axis=1)
    ok = np.ones(len(C), dtype=bool)
    for b0 in range(0, len(C), 65536):  # bounded memory for the 823k-row E7 grid
        P = C[b0:b0 + 65536] @ T.T
        ok[b0:b0 + 65536] = np.all(cos_le_sqrt3_half(P, Nc[b0:b0 + 65536, None], Nt[None, :]), axis=1)
    C, Nc = C[ok], Nc[ok]
    if max_candidates is not None and len(C) > max_candidates:
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(len(C)), Nc))
        C = C[np.sort(order[:max_candidates])]
    return C


def _compat_matrix(C: np.ndarray) -> np.ndarray:
    N = (C * C).sum(axis=1)
    G = C @ C.T
    A = cos_le_half(G, N[:, None], N[None, :])
    np.fill_diagonal(A, False)
    return A


def _greedy_clique(A: np.ndarray, order: np.ndarray) -> list[int]:
    chosen: list[int] = []
    for v in order:
        if all(A[v, u] for u in chosen):
            chosen.append(int(v))
    return chosen


def _clique_local_search(A: np.ndarray, start: list[int], rng: np.random.Generator, iters: int = 20000,
                         target: int | None = None) -> list[int]:
    """Simple (1,2)-swap / plateau local search for a larger clique (ARW-lite).

    A is the boolean compatibility matrix (never copied to a wider dtype:
    for 20k candidates an int64 copy would be 3 GB).
    """
    n = len(A)
    inC = np.zeros(n, dtype=bool)
    inC[start] = True
    # miss[v] = number of clique members NOT adjacent to v
    miss = np.zeros(n, dtype=np.int32)
    for u in np.flatnonzero(inC):
        miss[~A[u]] += 1
    miss[inC] = 0
    best = list(np.flatnonzero(inC))

    def recompute(u: int) -> None:
        miss[u] = int(inC.sum() - np.count_nonzero(A[u] & inC))

    for _ in range(iters):
        size = int(inC.sum())
        if target is not None and size >= target:
            break
        free = np.flatnonzero(~inC & (miss == 0))
        if len(free):
            v = int(rng.choice(free))
            inC[v] = True
            miss[~A[v] & ~inC] += 1
            miss[v] = 0
            if inC.sum() > len(best):
                best = list(np.flatnonzero(inC))
            continue
        ones = np.flatnonzero(~inC & (miss == 1))
        if len(ones) == 0:
            # perturb: drop a random member
            members = np.flatnonzero(inC)
            u = int(rng.choice(members))
            inC[u] = False
            miss[~A[u] & ~inC] -= 1
            recompute(u)
            continue
        v = int(rng.choice(ones))
        # the unique member not adjacent to v
        u = int(np.flatnonzero(inC & ~A[v])[0])
        inC[u] = False
        miss[~A[u] & ~inC] -= 1
        recompute(u)
        inC[v] = True
        miss[~A[v] & ~inC] += 1
        miss[v] = 0
        if inC.sum() > len(best):
            best = list(np.flatnonzero(inC))
    return best


def lattice_extra_spheres(d: int, max_entry: int = 2, T: np.ndarray | None = None, target: int | None = None,
                          seed: int = 0, restarts: int = 20, iters: int = 20000,
                          max_candidates: int | None = 12000) -> dict:
    """Strategy 1: integer grid candidates -> compatibility graph -> clique.

    Returns {"vectors": int array (n, m), "n": n, "target": K(d), "candidates": #, "reached": bool}.
    The result is exact (every returned vector satisfies both conditions by
    integer arithmetic); only the *number* found is heuristic. The vectors
    need not share a norm.
    """
    T = config(d)[0] if T is None else np.asarray(T, dtype=np.int64)
    target = K[d] if target is None else target
    C = extra_sphere_candidates(d, max_entry=max_entry, T=T, max_candidates=max_candidates, seed=seed)
    if len(C) == 0:
        return {"vectors": C, "n": 0, "target": target, "candidates": 0, "reached": False, "method": "lattice"}
    A = _compat_matrix(C)
    rng = np.random.default_rng(seed)
    deg = A.sum(axis=1)
    best: list[int] = []
    for r in range(restarts):
        order = np.argsort(-deg, kind="stable") if r == 0 else rng.permutation(len(C))
        cl = _greedy_clique(A, order)
        cl = _clique_local_search(A, cl, rng, iters=iters, target=target)
        if len(cl) > len(best):
            best = cl
        if len(best) >= target:
            break
    E = C[sorted(best)]
    return {"vectors": E, "n": len(E), "target": target, "candidates": int(len(C)), "reached": len(E) >= target,
            "method": "lattice"}


def _span_basis(d: int) -> np.ndarray:
    """Integer basis (d rows) of the span of config(d) in its ambient coordinates."""
    if d in (3, 4, 5):
        return np.eye(d, dtype=np.int64)
    if d == 2:
        return np.array([[1, -1, 0], [0, 1, -1]], dtype=np.int64)
    if d == 7:
        B = [[1, 1, 0, 0, 0, 0, 0, 0]] + [[0, 0] + [1 if j == i else 0 for j in range(6)] for i in range(6)]
        return np.array(B, dtype=np.int64)
    if d == 6:
        B = [[1, 1, 1, 0, 0, 0, 0, 0]] + [[0, 0, 0] + [1 if j == i else 0 for j in range(5)] for i in range(5)]
        return np.array(B, dtype=np.int64)
    raise ValueError(d)


def rational_rotation(d: int, rng: np.random.Generator, denominators=(2, 3), density: float = 0.5):
    """A random rational orthogonal matrix of the ambient space that fixes the orthogonal complement of the span.

    Cayley transform Q = (I - A)(I + A)^-1 of a rational skew matrix
    A = sum c_kl (b_k b_l^T - b_l b_k^T) over pairs of span basis vectors with
    c_kl in {0, +-1/den}. Returns (Qnum: int (m, m), den: int) with Q = Qnum / den.
    Exact (sympy rationals).
    """
    import sympy

    B = _span_basis(d)
    m = B.shape[1]
    A = sympy.zeros(m, m)
    for k in range(d):
        for l in range(k + 1, d):
            if rng.random() < density:
                c = sympy.Rational(int(rng.choice([-1, 1])), int(rng.choice(denominators)))
                bk, bl = sympy.Matrix(B[k].tolist()), sympy.Matrix(B[l].tolist())
                A += c * (bk * bl.T - bl * bk.T)
    Qm = (sympy.eye(m) - A) * (sympy.eye(m) + A).inv()
    den = sympy.ilcm(*[sympy.fraction(x)[1] for x in Qm])
    Qnum = np.array([[int(x * den) for x in row] for row in Qm.tolist()], dtype=object)
    return Qnum, int(den)


def rotated_extra_spheres(d: int, T: np.ndarray | None = None, seed: int = 0, tries: int = 200,
                          denominators=(2, 3), density: float = 0.5) -> dict:
    """Strategy 2: a rational rotation of config(d) itself, at >= 30 degrees from every T-vector.

    Pairwise cos <= 1/2 among the extras holds automatically (they are an
    isometric copy of a kissing configuration), so only the extra-T condition
    is searched for, by random rational rotations (Cayley transforms). The
    returned integer vectors are den * (rotated config) and all share the norm
    norm2 * den^2; ``den`` is returned so that vectors / den has norm norm2.
    """
    V, norm2 = config(d)
    T = V if T is None else np.asarray(T, dtype=np.int64)
    Nt = (T * T).sum(axis=1)
    rng = np.random.default_rng(seed)
    Vo = V.astype(object)
    for t in range(tries):
        Qnum, den = rational_rotation(d, rng, denominators=denominators, density=density)
        E = Vo @ Qnum.T  # object ints; rows have norm norm2 * den^2
        Ne = np.array([int(sum(int(x) * int(x) for x in row)) for row in E], dtype=object)
        assert all(int(v) == norm2 * den * den for v in Ne)
        P = E @ T.astype(object).T
        ok = cos_le_sqrt3_half(P, Ne[:, None], Nt[None, :])
        if bool(np.all(ok)):
            Ei = np.array([[int(x) for x in row] for row in E], dtype=np.int64) if den * max(abs(int(x)) for x in Qnum.ravel()) < 2**20 else E
            return {"vectors": Ei, "n": len(E), "target": K[d], "candidates": None, "reached": True,
                    "method": "rotation", "den": den, "tries": t + 1}
    return {"vectors": np.zeros((0, V.shape[1]), dtype=np.int64), "n": 0, "target": K[d], "candidates": None,
            "reached": False, "method": "rotation", "den": None, "tries": tries}


def _sign_ab(A, B, M: int) -> np.ndarray:
    """Elementwise sign of A + B sqrt(M) for integer arrays A, B (M >= 1 square-free; exact)."""
    A = np.asarray(A, dtype=object)
    B = np.asarray(B, dtype=object)
    sA = np.sign(A.astype(np.int64)) if A.size and np.abs(A).max() < 2**62 else np.array([(x > 0) - (x < 0) for x in A.ravel()], dtype=np.int64).reshape(A.shape)
    sB = np.sign(B.astype(np.int64)) if B.size and np.abs(B).max() < 2**62 else np.array([(x > 0) - (x < 0) for x in B.ravel()], dtype=np.int64).reshape(B.shape)
    out = np.where(sA == sB, sA, 0)
    out = np.where(sB == 0, sA, out)
    out = np.where(sA == 0, sB, out)
    mixed = (sA != 0) & (sB != 0) & (sA != sB)
    if mixed.any():
        a2 = A * A
        b2 = B * B * M
        out = np.where(mixed & (a2 > b2), sA, out)
        out = np.where(mixed & (a2 < b2), sB, out)
        out = np.where(mixed & (a2 == b2), 0, out)
    return out.astype(np.int64)


def _plane_list(d: int, M: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Orthogonal integer pairs (a, b) spanning planes inside span(config(d)) with sqrt(|a|^2 |b|^2) in Q(sqrt M)."""
    from .exact import squarefree_split

    m = AMBIENT[d]
    e = np.eye(m, dtype=np.int64)
    if d == 2:
        base = [(np.array([1, -1, 0]), np.array([1, 1, -2]))]
    else:
        free = {3: range(3), 4: range(4), 5: range(5), 6: range(3, 8), 7: range(2, 8)}[d]
        head = {6: e[0] + e[1] + e[2], 7: e[0] + e[1]}.get(d)
        base = []
        for i in free:
            for j in free:
                if i < j:
                    base.append((e[i], e[j]))
                    base.append((e[i] + e[j], e[i] - e[j]))
                    for k in free:
                        if k not in (i, j):
                            base.append((e[i] + e[j], e[k]))
            if head is not None:
                base.append((head, e[i]))
        if d in (4, 5, 6, 7):
            for i in free:
                for j in free:
                    for k in free:
                        if i < j < k:
                            for l in free:
                                if l not in (i, j, k):
                                    base.append((e[i] + e[j] + e[k], e[l]))
    out = []
    for a, b in base:
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        assert int(a @ b) == 0
        cons = _span_constraints(d)
        if len(cons) and (np.any(cons @ a) or np.any(cons @ b)):
            continue
        _, r = squarefree_split(int(a @ a) * int(b @ b))
        if r in (1, M):
            out.append((a, b))
    return out


def _plane_rotation_exact(a: np.ndarray, b: np.ndarray, c, s) -> list[list]:
    """R = I + (c-1)(uu^T + vv^T) + s(vu^T - uv^T), u = a/|a|, v = b/|b|, exact (entries Q)."""
    from .exact import Q

    m = len(a)
    Na, Nb = int(a @ a), int(b @ b)
    root = Q.sqrt(Na * Nb)
    inv_root = root.inv()
    R = [[Q(1 if i == j else 0) for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(m):
            proj = Q(int(a[i]) * int(a[j])) * Q(Fraction(1, Na)) + \
                Q(int(b[i]) * int(b[j])) * Q(Fraction(1, Nb))
            skew = Q(int(b[i]) * int(a[j]) - int(a[i]) * int(b[j])) * inv_root
            R[i][j] = R[i][j] + (c - Q(1)) * proj + s * skew
    return R


def _exact_to_ab(rows: list[list], M: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Rows of Q with radicals in {1, M} -> integer (A, B, D) with rows = (A + B sqrt M) / D."""
    from fractions import Fraction

    D = 1
    for r in rows:
        for x in r:
            for rad, v in x.c.items():
                if rad not in (1, M):
                    raise ValueError(f"radical {rad} outside Q(sqrt {M})")
                D = D * v.denominator // math.gcd(D, v.denominator)
    A = np.array([[int(Fraction(x.c.get(1, 0)) * D) for x in r] for r in rows], dtype=np.int64)
    B = np.array([[int(Fraction(x.c.get(M, 0)) * D) for x in r] for r in rows], dtype=np.int64)
    return A, B, D


def check_extra_ab(A: np.ndarray, B: np.ndarray, M: int, D: int, T: np.ndarray, N: int) -> dict:
    """SCHEMA.md 3.4 exactly, vectorised: extras y' = (A + B sqrt M)/D of norm N against T rows of norm N."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)
    n = len(A)
    if n == 0:
        return {"ok": True, "n": 0, "message": "ok"}
    # norms: |a|^2 + M |b|^2 = D^2 N and <a,b> = 0
    if np.any((A * A).sum(1) + M * (B * B).sum(1) != D * D * N) or np.any((A * B).sum(1) != 0):
        return {"ok": False, "n": n, "message": "extra rows do not have norm N"}
    # extra-T: p = (P + Q sqrt M)/D; cos <= sqrt3/2 <=> 2p <= 0 or (4P^2 + 4MQ^2 - 3N^2D^2) + 8PQ sqrt M <= 0
    P, Qm = A @ T.T, B @ T.T
    nonpos = _sign_ab(P, Qm, M) <= 0
    sq = _sign_ab(4 * P * P + 4 * M * Qm * Qm - 3 * N * N * D * D, 8 * P * Qm, M) <= 0
    bad = ~(nonpos | sq)
    if bad.any():
        i, j = (int(v) for v in np.argwhere(bad)[0])
        return {"ok": False, "n": n, "message": f"extra {i} within 30 degrees of T vector {j}"}
    # extra-extra: p = (P + Q sqrt M)/D^2; cos <= 1/2 <=> (2P - N D^2) + 2Q sqrt M <= 0; distinct directions
    P = A @ A.T + M * (B @ B.T)
    Qm = A @ B.T + B @ A.T
    off = ~np.eye(n, dtype=bool)
    le = _sign_ab(2 * P - N * D * D, 2 * Qm, M) <= 0
    if (off & ~le).any():
        i, j = (int(v) for v in np.argwhere(off & ~le)[0])
        return {"ok": False, "n": n, "message": f"extras {i} and {j} have cos > 1/2"}
    same = off & (P == N * D * D) & (Qm == 0)
    if same.any():
        i, j = (int(v) for v in np.argwhere(same)[0])
        return {"ok": False, "n": n, "message": f"extras {i} and {j} coincide"}
    return {"ok": True, "n": n, "message": "ok"}


def rotated_extra_spheres_exact(d: int, M: int | None = None, max_len: int = 3, tries: int = 3000,
                                seed: int = 0, T: np.ndarray | None = None) -> dict:
    """Strategy 3: config(d) rotated by products of plane rotations with angles 45 deg (M = 2) or 30/60 deg (M = 3).

    The rotated copy is exact over Q(sqrt M) (schema form (a + b sqrt M)/D)
    and is a kissing configuration by construction; the search is over
    sequences of <= max_len plane rotations (planes from _plane_list, signs
    random) for one whose 126 (etc.) images are all >= 30 degrees from T,
    decided exactly by check_extra_ab. Returns the schema "extra" fields plus
    "n", "reached", "rotations" (the sequence found), "tries".
    """
    from .exact import Q

    V, N = config(d)
    T = V if T is None else np.asarray(T, dtype=np.int64)
    if M is None:
        M = 2 if d % 2 else 3
    if M == 2:
        angles = [(Q.sqrt(2) * Q(Fraction(1, 2)), Q.sqrt(2) * Q(Fraction(1, 2)), "45")]
    else:
        half = Q(Fraction(1, 2))
        angles = [(Q.sqrt(3) * half, half, "30"), (half, Q.sqrt(3) * half, "60")]
    planes = _plane_list(d, M)
    rng = np.random.default_rng(seed)
    m = V.shape[1]
    Vq = [[Q(int(x)) for x in row] for row in V]
    seen: set = set()
    for t in range(tries):
        L = int(rng.integers(1, max_len + 1))
        seq = []
        for _ in range(L):
            k = int(rng.integers(len(planes)))
            ai = int(rng.integers(len(angles)))
            sgn = int(rng.choice([-1, 1]))
            seq.append((k, ai, sgn))
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        # R = R_L ... R_1
        R = [[Q(1 if i == j else 0) for j in range(m)] for i in range(m)]
        for k, ai, sgn in seq:
            a, b = planes[k]
            c, s, _ = angles[ai]
            Rk = _plane_rotation_exact(a, b, c, s * Q(sgn))
            R = [[sum((Rk[i][l] * R[l][j] for l in range(m)), Q()) for j in range(m)] for i in range(m)]
        E = [[sum((Vq[r][l] * R[i][l] for l in range(m)), Q()) for i in range(m)] for r in range(len(V))]
        try:
            A, B, D = _exact_to_ab(E, M)
        except ValueError:
            continue
        chk = check_extra_ab(A, B, M, D, T, N)
        if chk["ok"]:
            desc = [(planes[k][0].tolist(), planes[k][1].tolist(), ("-" if sgn < 0 else "+") + angles[ai][2])
                    for k, ai, sgn in seq]
            return {"n": len(V), "target": K[d], "reached": True, "method": "exact-rotation", "count": len(V),
                    "sqrt": M, "den": D, "a": A.tolist(), "b": B.tolist(), "rotations": desc, "tries": t + 1,
                    "vectors_exact": E}
    return {"n": 0, "target": K[d], "reached": False, "method": "exact-rotation", "count": 0, "sqrt": M,
            "den": 1, "a": [], "b": [], "rotations": [], "tries": tries}


def extra_spheres(d: int, max_entry: int = 2, T: np.ndarray | None = None, target: int | None = None,
                  seed: int = 0, restarts: int = 20, iters: int = 20000, tries: int = 200,
                  max_candidates: int | None = 12000, exact_tries: int = 3000) -> dict:
    """Extra directions for the record template (README 1.3): cos <= 1/2 among themselves, >= 30 degrees from T.

    Strategies, in order, stopping at the first that reaches K(d):
    1. integer grid candidates + clique search (``lattice_extra_spheres``; integer vectors of any norms);
    2. exact plane rotations of the configuration over Q(sqrt 2) / Q(sqrt 3)
       (``rotated_extra_spheres_exact``; schema (a + b sqrt M)/D form);
    3. random rational Cayley rotations (``rotated_extra_spheres``).
    Returns the best result; "method" says which. Every returned configuration
    is exact (integer or Q(sqrt M) square comparisons); only the number found
    is heuristic.
    """
    target = K[d] if target is None else target
    res = lattice_extra_spheres(d, max_entry=max_entry, T=T, target=target, seed=seed, restarts=restarts,
                                iters=iters, max_candidates=max_candidates)
    if res["reached"]:
        return res
    ex = rotated_extra_spheres_exact(d, T=T, seed=seed, tries=exact_tries)
    if ex["reached"]:
        return ex
    rot = rotated_extra_spheres(d, T=T, seed=seed, tries=tries)
    return rot if rot["n"] > res["n"] else res


# ---------------------------------------------------------------------------
# data/families/SCHEMA.md blocks (T4.2's schema v1)
# ---------------------------------------------------------------------------
def schema_T_block(d: int, partition: dict | None = None) -> dict:
    """The "T" object of family.json for config(d): vectors, norm2, groups (triangles then pairs)."""
    V, norm2 = config(d)
    if partition is None:
        partition = max_triangle_partition(d)
    groups = [list(t) for t in partition["triangles"]] + [list(p) for p in partition["pairs"]]
    return {"config": NAMES[d], "K": int(len(V)), "ambient": int(V.shape[1]), "norm2": int(norm2),
            "vectors": V.tolist(), "groups": groups}


def schema_extra_block(E: np.ndarray, norm2: int) -> dict:
    """The "extra" object of family.json: rows y' = (a + b sqrt M) / D with |y'|^2 = norm2.

    E: integer rows all of one squared norm Ne. Then y' = e * sqrt(norm2 / Ne):
    with norm2 * Ne = s^2 r (r square-free) this is e * s * sqrt(r) / Ne, i.e.
    M = r, D = Ne, b = s e, a = 0 (and if r = 1: M = 1, a = s e, b = 0),
    reduced by the gcd of all entries and D.
    """
    from .exact import squarefree_split

    E = np.asarray(E, dtype=np.int64)
    if len(E) == 0:
        return {"count": 0, "sqrt": 1, "den": 1, "a": [], "b": []}
    Ne = {int(v) for v in (E * E).sum(axis=1)}
    if len(Ne) != 1:
        raise ValueError(f"extra rows have different norms {sorted(Ne)}; the schema needs a common norm")
    Ne = Ne.pop()
    s, r = squarefree_split(int(norm2) * Ne)
    D = Ne
    coef = E * s
    g = math.gcd(int(np.gcd.reduce(np.abs(coef).ravel())), D)
    coef, D = coef // g, D // g
    zeros = np.zeros_like(coef)
    a, b = (coef, zeros) if r == 1 else (zeros, coef)
    return {"count": int(len(E)), "sqrt": int(r), "den": int(D), "a": a.tolist(), "b": b.tolist()}


def check_extra(E: np.ndarray, T: np.ndarray) -> dict:
    """Extra directions: pairwise cos <= 1/2 (and distinct), cos <= sqrt3/2 to every T row. Exact."""
    E = np.asarray(E, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)
    res = check_kissing(E) if len(E) else {"ok": True, "n": 0, "message": "ok"}
    if not res["ok"]:
        return {"ok": False, "message": "extra-extra: " + res["message"]}
    if len(E) and len(T):
        Ne = (E * E).sum(axis=1)
        Nt = (T * T).sum(axis=1)
        P = E @ T.T
        bad = ~cos_le_sqrt3_half(P, Ne[:, None], Nt[None, :])
        if bad.any():
            i, j = (int(v) for v in np.argwhere(bad)[0])
            return {"ok": False, "message": f"extra {i} is within 30 degrees of T-vector {j} "
                                            f"(ip {int(P[i, j])}, norms {int(Ne[i])},{int(Nt[j])})"}
        with np.errstate(divide="ignore", invalid="ignore"):
            max_cos_T = float((P / np.sqrt(Ne[:, None] * Nt[None, :])).max())
    else:
        max_cos_T = None
    return {"ok": True, "message": "ok", "n": len(E), "max_cos_to_T": max_cos_T,
            "max_cos_pairwise": res.get("max_cos")}


# ---------------------------------------------------------------------------
# mapping a float copy of the configuration onto the integer model
# ---------------------------------------------------------------------------
def isometry_to_model(Vf: np.ndarray, d: int, tol: float = 1e-6, node_limit: int = 2_000_000) -> np.ndarray | None:
    """Find perm with config(d)[perm[k]] the image of Vf[k] under some isometry.

    Vf: float (K(d), d') rows, all of equal norm, expected to be a copy of the
    K(d) configuration (any scaling). Backtracking over the images of a basis
    of d rows, pruned by exact Gram agreement; the remaining images are forced
    by linearity and checked. Returns None if no isometry exists.
    """
    Vf = np.asarray(Vf, dtype=np.float64)
    W, norm2 = config(d)
    n = len(W)
    if len(Vf) != n:
        return None
    nf = (Vf * Vf).sum(axis=1)
    if np.abs(nf - nf[0]).max() > tol * nf[0]:
        return None
    Gf = np.rint(Vf @ Vf.T / nf[0] * norm2).astype(np.int64)  # Gram in the model's scaling
    if np.abs(Vf @ Vf.T / nf[0] * norm2 - Gf).max() > 1e-6 * norm2:
        return None
    GW = W @ W.T
    # a basis: greedy choice of d linearly independent rows of Vf
    basis: list[int] = []
    for i in range(n):
        if np.linalg.matrix_rank(Vf[basis + [i]], tol=1e-8) == len(basis) + 1:
            basis.append(i)
        if len(basis) == d:
            break
    if len(basis) < d:
        return None
    state = {"nodes": 0}
    img: list[int] = []

    def finish() -> np.ndarray | None:
        # linear map A with A Vf[basis[k]] = W[img[k]]; images of all rows
        B = Vf[basis]                       # (d, d')
        Wb = W[img].astype(np.float64)      # (d, m)
        # solve for A (m x d'): Wb = B A^T  ->  A^T = lstsq(B, Wb)
        AT, *_ = np.linalg.lstsq(B, Wb, rcond=None)
        Y = Vf @ AT                          # (n, m)
        Yi = np.rint(Y).astype(np.int64)
        if np.abs(Y - Yi).max() > 1e-6:
            return None
        lookup = {row.tobytes(): i for i, row in enumerate(W)}
        perm = np.array([lookup.get(row.tobytes(), -1) for row in Yi], dtype=np.int64)
        if np.any(perm < 0) or len(set(perm.tolist())) != n:
            return None
        if not np.array_equal(GW[np.ix_(perm, perm)], Gf):
            return None
        return perm

    def bt(k: int) -> np.ndarray | None:
        state["nodes"] += 1
        if state["nodes"] > node_limit:
            return None
        if k == d:
            return finish()
        i = basis[k]
        for cand in range(n):
            if cand in img:
                continue
            if all(GW[cand, img[t]] == Gf[i, basis[t]] for t in range(k)):
                img.append(cand)
                r = bt(k + 1)
                if r is not None:
                    return r
                img.pop()
        return None

    return bt(0)

#!/usr/bin/env python3
"""Build and verify the explicit R^25 kissing configuration of README section 1.3
(d = 1, T = {+1, -1}) for a certified Leech subset S (PLAN T1.4).

    python/verify_dim25.py S.txt [--tile 512 1024] [--workers 8] [--skip-full-equatorial]

Norm-4 scaling (integer coordinates divided by sqrt 8). The configuration:

  equatorial  (x, 0)                          for x in C \\ S          196560 - |S| vectors
  lifted      (x * sqrt(2/3), +- sqrt(4/3))   for x in S              2 |S| vectors

all of norm 4; it is a kissing configuration iff every pairwise inner product
is <= 2, giving K(25) >= 196560 + |S|. The count is verified two ways and both
numbers are printed:

(a) EXACT integer casework. With ip = <x, x'> the sqrt-8-integer inner product
    of two minimal vectors (norm-4 inner product ip / 8):
      equatorial-equatorial   ip/8 <= 2                    <=>  ip <= 16
      equatorial-lifted       sqrt(2/3) ip/8 <= 2          <=>  ip <= 8 sqrt 6 ~ 19.6, so ip <= 16 suffices
      lifted-lifted, x != x'  (2/3) ip/8 + (4/3) y y' <= 2 <=>  ip + 16 y y' <= 24:
                              same sign  =>  ip <= 8  (S independent),
                              opposite   =>  ip <= 40, always true (ip <= 32)
      lifted-lifted, x = x', y != y'   (2/3) 4 - (4/3) = 4/3 <= 2, always true.
    ip <= 16 for two DISTINCT minimal vectors is the statement that the minimal
    norm of the Leech lattice is 32: x - x' is a nonzero lattice vector, so
    64 - 2 ip = |x - x'|^2 >= 32. The script does not rely on that argument
    alone: by default it makes a full pass over all C(196560, 2) pairs (exact
    integer arithmetic carried in float32 GEMM, see exact_full_pass) and checks
    every off-diagonal entry is <= 16; --skip-full-equatorial replaces the pass
    by the argument (and says so). The equatorial-lifted and lifted-lifted
    cases are always checked by explicit int64 inner products; in particular
    the equatorial set is verified to be exactly C \\ S (an x that is both
    lifted and equatorial would give ip = 32 > 19.6 and fail).

(b) FLOAT. Explicit float64 coordinates of all 196560 + |S| vectors in R^25,
    blocked matrix products (upper-triangular 512 x 1024 tiles, one Python
    thread per row block with single-threaded BLAS), max off-diagonal inner
    product <= 2 + 1e-9, and the class of the pair attaining the maximum.

Independent of the C++ code: only kiss_ref and verify_S (numpy) are used.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# OpenBLAS does not parallelise these K = 25 products (measured: ~20 GFLOPS at
# any thread count), so the passes below run one single-threaded GEMM per
# Python thread instead. Must be set before numpy is first imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiss_ref.leech import DIM, N_MIN  # noqa: E402
from verify_S import check_set, leech_index, quote, read_set  # noqa: E402

SQRT8 = math.sqrt(8.0)
LIFT_X = math.sqrt(2.0 / 3.0)  # multiplies the (already norm-4 scaled) x
LIFT_Y = math.sqrt(4.0 / 3.0)  # 25th coordinate, times +-1

PAIR_CLASSES = ("eq-eq", "eq-lift", "lift-lift-same", "lift-lift-opp", "lift-lift-same-x")


# ---------------------------------------------------------------------------
# (a) exact integer casework
# ---------------------------------------------------------------------------
def default_workers() -> int:
    """Physical-core guess: SMT siblings only slow the bandwidth-bound tiles down."""
    return max(1, (os.cpu_count() or 2) // 2)


def pairwise_offdiag_max(A: np.ndarray, rb: int = 512, cb: int = 1024, workers: int | None = None,
                         expect_diag: float | None = None, sample_classes: bool = False) -> dict:
    """Max (and min) of (A A^T)[i, j] over all i != j, exactly the values GEMM produces.

    Upper-triangular tiles of rb x cb entries (the pair (i, j) with j < i is
    covered by the tile of row block(j)), one thread pool task per row block,
    each task running single-threaded BLAS on its own tile buffer. Returns the
    max with the pair attaining it, the min, and (optionally) the set of
    distinct values seen in the first row of every tile.
    """
    n = len(A)
    AT = np.ascontiguousarray(A.T)
    lock = threading.Lock()
    state = {"max": -np.inf, "pair": (-1, -1), "min": np.inf, "classes": set(), "tiles": 0}
    neg_inf = -np.inf

    def task(i0: int) -> None:
        ni = min(rb, n - i0)
        buf = np.empty((ni, cb), dtype=A.dtype)
        Ai = A[i0:i0 + ni]
        best, pair, vmin, classes, tiles = neg_inf, (-1, -1), np.inf, set(), 0
        for j0 in range(i0, n, cb):
            nj = min(cb, n - j0)
            G = buf[:, :nj]
            np.matmul(Ai, AT[:, j0:j0 + nj], out=G)
            off = j0 - i0
            if off < ni:  # tile contains diagonal entries (i0 + off + k, j0 + k)
                d = G[off:ni, :nj]
                if expect_diag is not None and not np.all(np.diagonal(d) == expect_diag):
                    raise AssertionError(f"diagonal entry != {expect_diag} in rows {i0}..{i0 + ni}")
                vmin = min(vmin, float(G.min()))  # the diagonal never attains the min
                np.fill_diagonal(d, neg_inf)
            else:
                vmin = min(vmin, float(G.min()))
            if sample_classes:
                classes |= set(G[0].tolist())
            m = float(G.max())
            if m > best:
                k = int(np.argmax(G))
                best, pair = m, (i0 + k // nj, j0 + k % nj)
            tiles += 1
        with lock:
            if best > state["max"]:
                state["max"], state["pair"] = best, pair
            state["min"] = min(state["min"], vmin)
            state["classes"] |= classes
            state["tiles"] += tiles

    with ThreadPoolExecutor(max_workers=workers or default_workers()) as ex:
        list(ex.map(task, range(0, n, rb)))  # list() re-raises worker exceptions
    state["classes"].discard(neg_inf)
    if expect_diag is not None:
        state["classes"].discard(expect_diag)
    return state


def exact_full_pass(C: np.ndarray, rb: int = 512, cb: int = 1024, workers: int | None = None) -> dict:
    """Max/min of <x, x'> over ALL pairs x != x' in C, exactly.

    The products are computed by float32 GEMM. Every input is an integer with
    |value| <= 4, every product <= 16 and every partial sum <= 24 * 16 = 384,
    so every intermediate is an integer far below 2^24 and float32 arithmetic
    (mul/add/FMA, in any association order) is exact. The diagonal is checked
    to be exactly 32 before it is masked out, and the first row of every tile
    is checked to lie in the six off-diagonal inner-product classes.
    """
    t0 = time.time()
    n = len(C)
    Cf = np.ascontiguousarray(C, dtype=np.float32)
    st = pairwise_offdiag_max(Cf, rb=rb, cb=cb, workers=workers, expect_diag=32.0, sample_classes=True)
    allowed = {-32.0, -16.0, -8.0, 0.0, 8.0, 16.0}
    return {"max": st["max"], "min": st["min"], "pairs": n * (n - 1) // 2,
            "classes_ok": st["classes"] <= allowed, "classes_seen": sorted(int(c) for c in st["classes"]),
            "tiles": st["tiles"], "seconds": time.time() - t0}


def exact_count(C: np.ndarray, S: np.ndarray, S_idx: np.ndarray, full_pass: bool = True,
                rb: int = 512, cb: int = 1024, workers: int | None = None) -> dict:
    """The exact casework of the docstring. Returns ok/count/pair counts/details."""
    t0 = time.time()
    n = len(C)
    m = len(S)
    S = np.asarray(S, dtype=np.int64)
    S_idx = np.asarray(S_idx, dtype=np.int64)
    out: dict = {"ok": False, "size": m, "count": None, "pairs": {}, "notes": []}

    # Structural facts: S is a set of distinct members, equatorial = C \ S.
    if len(np.unique(S_idx)) != m or np.any(S_idx < 0) or np.any(S_idx >= n):
        out["message"] = "S indices are not distinct valid canonical indices"
        return out
    if not np.array_equal(C[S_idx].astype(np.int64), S):
        out["message"] = "S rows do not match C[S_idx]"
        return out
    in_s = np.zeros(n, dtype=bool)
    in_s[S_idx] = True
    eq_idx = np.flatnonzero(~in_s)
    n_eq = len(eq_idx)
    if n_eq + m != n or np.any(in_s[eq_idx]):
        out["message"] = "equatorial set is not exactly C \\ S"
        return out
    out["n_equatorial"] = int(n_eq)
    out["n_lifted"] = int(2 * m)

    # lifted-lifted (x != x'): int64 Gram of S.
    G = S @ S.T
    if not np.array_equal(np.diagonal(G), np.full(m, 32)):
        out["message"] = "a lifted vector does not have norm 32"
        return out
    off = G[~np.eye(m, dtype=bool)] if m > 1 else np.zeros(0, dtype=np.int64)
    same_ok = bool(np.all(off <= 8))      # ip + 16 <= 24
    opp_ok = bool(np.all(off <= 40))      # ip - 16 <= 24 (always, but checked)
    out["pairs"]["lift-lift-same"] = m * (m - 1)          # unordered: 2 * C(m,2) (signs ++ and --)
    out["pairs"]["lift-lift-opp"] = m * (m - 1)           # unordered: (x,+),(x',-) over ordered (x,x'), x != x'
    out["pairs"]["lift-lift-same-x"] = m                  # (x,+),(x,-): value 4/3
    out["max_ip_lifted"] = int(off.max()) if len(off) else None
    if not same_ok:
        i, j = np.argwhere((G > 8) & ~np.eye(m, dtype=bool))[0]
        out["message"] = f"lifted-lifted same sign: rows {i},{j} of S have ip {int(G[i, j])} > 8"
        return out
    if not opp_ok:  # unreachable for minimal vectors, kept for completeness
        out["message"] = "lifted-lifted opposite sign: ip > 40"
        return out

    # equatorial-lifted: int64 products S x C[eq]; need ip <= 16 (< 8 sqrt 6).
    Ceq = C[eq_idx].astype(np.int64)
    max_el = -10**9
    for b0 in range(0, m, 64):
        P = S[b0:b0 + 64] @ Ceq.T
        max_el = max(max_el, int(P.max()))
    out["max_ip_eq_lift"] = max_el
    out["pairs"]["eq-lift"] = 2 * m * n_eq
    if max_el > 16:
        out["message"] = f"equatorial-lifted: ip {max_el} > 16 (a lifted vector coincides with an equatorial one?)"
        return out
    if not (max_el * LIFT_X / 8.0 <= 2.0):  # the actual inequality, for the record
        out["message"] = "equatorial-lifted bound violated"
        return out

    # equatorial-equatorial: need ip <= 16 for distinct minimal vectors.
    out["pairs"]["eq-eq"] = n_eq * (n_eq - 1) // 2
    if full_pass:
        fp = exact_full_pass(C, rb=rb, cb=cb, workers=workers)
        out["full_pass"] = fp
        if fp["max"] > 16 or not fp["classes_ok"]:
            out["message"] = f"equatorial-equatorial: full pass found ip {fp['max']} > 16 or a bad class"
            return out
        out["notes"].append(
            f"eq-eq: full exact pass over all {fp['pairs']} pairs of C: max off-diagonal ip = {int(fp['max'])}, "
            f"min = {int(fp['min'])}, values seen {fp['classes_seen']} ({fp['seconds']:.1f} s)")
    else:
        out["notes"].append("eq-eq: ip <= 16 for distinct minimal vectors taken from the Leech minimal norm "
                            "(|x-x'|^2 = 64 - 2 ip >= 32), full pass skipped (--skip-full-equatorial)")

    total_pairs = sum(out["pairs"].values())
    count = n_eq + 2 * m
    if total_pairs != count * (count - 1) // 2:
        out["message"] = f"pair count {total_pairs} != C({count},2)"
        return out
    out.update(ok=True, count=count, message="ok", seconds=time.time() - t0)
    return out


# ---------------------------------------------------------------------------
# (b) explicit float64 coordinates
# ---------------------------------------------------------------------------
def build_coordinates(C: np.ndarray, S_idx: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Float64 (196560 + |S|, 25) array: equatorial rows first, then S with +, then S with -."""
    n = len(C)
    m = len(S_idx)
    in_s = np.zeros(n, dtype=bool)
    in_s[S_idx] = True
    eq = C[~in_s].astype(np.float64) / SQRT8
    lifted = C[S_idx].astype(np.float64) / SQRT8 * LIFT_X
    X = np.zeros((n - m + 2 * m, DIM + 1), dtype=np.float64)
    n_eq = n - m
    X[:n_eq, :DIM] = eq
    X[n_eq:n_eq + m, :DIM] = lifted
    X[n_eq:n_eq + m, DIM] = LIFT_Y
    X[n_eq + m:, :DIM] = lifted
    X[n_eq + m:, DIM] = -LIFT_Y
    return X, n_eq, m


def pair_class(i: int, j: int, n_eq: int, m: int) -> str:
    def kind(k):
        if k < n_eq:
            return ("eq", None, None)
        k -= n_eq
        return ("lift", k % m, +1 if k < m else -1)

    a, b = kind(i), kind(j)
    if a[0] == "eq" and b[0] == "eq":
        return "eq-eq"
    if a[0] == "eq" or b[0] == "eq":
        return "eq-lift"
    if a[1] == b[1]:
        return "lift-lift-same-x"
    return "lift-lift-same" if a[2] == b[2] else "lift-lift-opp"


def float_count(C: np.ndarray, S_idx: np.ndarray, rb: int = 512, cb: int = 1024,
                workers: int | None = None, tol: float = 1e-9) -> dict:
    t0 = time.time()
    X, n_eq, m = build_coordinates(C, S_idx)
    total = len(X)
    norms = (X * X).sum(axis=1)
    if not np.all(np.abs(norms - 4.0) <= 1e-9):
        return {"ok": False, "message": "a vector does not have norm 4", "count": None}
    st = pairwise_offdiag_max(X, rb=rb, cb=cb, workers=workers)
    best, best_pair = st["max"], st["pair"]
    ok = best <= 2.0 + tol
    return {"ok": ok, "count": total if ok else None, "max_offdiag": best, "min_offdiag": st["min"],
            "pair": best_pair, "class": pair_class(best_pair[0], best_pair[1], n_eq, m),
            "n_equatorial": n_eq, "n_lifted": 2 * m, "seconds": time.time() - t0, "tiles": st["tiles"],
            "message": "ok" if ok else f"max off-diagonal {best!r} > 2 at {best_pair}"}


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the R^25 configuration of README 1.3 for S.")
    ap.add_argument("set", help="S.txt")
    ap.add_argument("--tile", type=int, nargs=2, default=(512, 1024), metavar=("ROWS", "COLS"),
                    help="tile shape for both full passes (default 512 1024)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Python threads running single-threaded GEMMs (default: cpu_count/2)")
    ap.add_argument("--skip-full-equatorial", action="store_true",
                    help="rely on the Leech minimal-norm argument for eq-eq instead of the full pass")
    ap.add_argument("--skip-float", action="store_true", help="skip method (b)")
    args = ap.parse_args(argv)
    t0 = time.time()

    try:
        S, lines = read_set(args.set)
    except (OSError, ValueError) as e:
        print(f"parse error: {e}")
        print(f"RESULT ok=0 size=0 reason={quote('parse: ' + str(e))}")
        return 1
    index = leech_index()
    C = index.C
    res = check_set(S, lines, index)
    print(f"file        : {args.set}")
    print(f"S check     : ok={int(res['ok'])} size={res['size']} {res['message']}")
    if not res["ok"]:
        print(f"RESULT ok=0 size={res['size']} reason={quote(res['message'])}")
        return 1
    m = res["size"]
    print(f"config      : equatorial={N_MIN - m} (C \\ S)  lifted={2 * m} (S x {{+1,-1}})  "
          f"total={N_MIN + m}  cpus={os.cpu_count()} workers={args.workers or default_workers()} "
          f"tile={tuple(args.tile)} OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")

    # (a)
    rb, cb = args.tile
    workers = args.workers or default_workers()
    ex = exact_count(C, S, res["idx"], full_pass=not args.skip_full_equatorial, rb=rb, cb=cb, workers=workers)
    for note in ex.get("notes", []):
        print(f"exact       : {note}")
    if not ex["ok"]:
        print(f"exact       : FAIL {ex['message']}")
        print(f"RESULT ok=0 size={m} count_exact=0 reason={quote(ex['message'])}")
        return 1
    print(f"exact       : lifted-lifted max ip = {ex['max_ip_lifted']} (same sign needs <= 8); "
          f"equatorial-lifted max ip = {ex['max_ip_eq_lift']} (needs <= 16 < 8 sqrt 6)")
    print("exact       : pairs " + " ".join(f"{k}={v}" for k, v in ex["pairs"].items()) +
          f" total={sum(ex['pairs'].values())}=C({ex['count']},2)")
    print(f"exact       : count = {ex['count']}  ({ex['seconds']:.1f} s)")

    # (b)
    if args.skip_float:
        print(f"RESULT ok=1 size={m} count_exact={ex['count']} count_float=skipped s={time.time() - t0:.1f}")
        return 0
    fl = float_count(C, res["idx"], rb=rb, cb=cb, workers=workers)
    if not fl["ok"]:
        print(f"float       : FAIL {fl['message']}")
        print(f"RESULT ok=0 size={m} count_exact={ex['count']} count_float=0 reason={quote(fl['message'])}")
        return 1
    print(f"float       : max off-diagonal inner product = {fl['max_offdiag']:.12f} (tol 2+1e-9) "
          f"at pair {fl['pair']} class={fl['class']}")
    print(f"float       : count = {fl['count']}  ({fl['seconds']:.1f} s, {fl['tiles']} tiles, "
          f"min off-diagonal {fl['min_offdiag']:.6f})")
    ok = ex["count"] == fl["count"] == N_MIN + m
    print(f"RESULT ok={int(ok)} size={m} count_exact={ex['count']} count_float={fl['count']} "
          f"max_offdiag_float={fl['max_offdiag']:.12f} max_class={fl['class']} "
          f"exact_s={ex['seconds']:.1f} float_s={fl['seconds']:.1f} s={time.time() - t0:.1f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

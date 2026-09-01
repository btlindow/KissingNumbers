#!/usr/bin/env python3
"""Independent certificate verifier for a Leech-subset S (README section 5, PLAN T1.4).

    python/verify_S.py S.txt            -> prints `RESULT ok=1 size=<n> ...`, exit 0
                                           or `RESULT ok=0 size=<n> reason="..."`, exit 1

Independence (PLAN section 2.4): this script uses only ``kiss_ref`` (the pure
numpy re-implementation of the Golay code and the 196560 minimal vectors). It
never reads data/leech_min.* or any C++ output; the set C is regenerated here.

Checks, in this order (the first failing check is reported with the file line
numbers of the offending row(s) / pair and the inner product):

1. every row has squared norm 32 (sqrt-8 integer scaling);
2. every row is one of the 196560 minimal vectors (looked up in the regenerated
   C, and cross-checked against the arithmetic membership test of README 1.2);
3. all rows are distinct;
4. every off-diagonal entry of the integer Gram matrix S S^T is <= 8
   (no pair at 60 degrees, where <x,y> = 16).

On success it also reports whether S is closed under negation and the Gram
histogram over unordered pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiss_ref.leech import DIM, N_MIN, is_lattice_vector_rows, leech_min_vectors  # noqa: E402

IP_CLASSES = (-32, -16, -8, 0, 8, 16, 32)


# ---------------------------------------------------------------------------
# Reading the set file (same grammar as PLAN 2.2 / kiss::read_set)
# ---------------------------------------------------------------------------
def read_set(path: str) -> tuple[np.ndarray, list[int]]:
    """Parse a set file -> (int64 array (n, 24), 1-based file line of each row).

    '#' starts a comment (whole line or trailing); blank lines are skipped.
    Raises ValueError with ``file:line:`` on a row that is not 24 integers.
    """
    rows: list[list[int]] = []
    lines: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            toks = text.split()
            if len(toks) != DIM:
                raise ValueError(f"{path}:{lineno}: expected {DIM} integers, got {len(toks)}")
            try:
                vals = [int(t) for t in toks]
            except ValueError:
                raise ValueError(f"{path}:{lineno}: non-integer token in {text!r}") from None
            rows.append(vals)
            lines.append(lineno)
    S = np.array(rows, dtype=np.int64).reshape(-1, DIM)
    return S, lines


# ---------------------------------------------------------------------------
# Lookup of rows in the regenerated C
# ---------------------------------------------------------------------------
class LeechIndex:
    """The regenerated minimal vectors plus an exact row -> index dictionary."""

    def __init__(self) -> None:
        self.C = leech_min_vectors()  # int8 (196560, 24), canonical order
        assert self.C.shape == (N_MIN, DIM)
        self.lookup = {row.tobytes(): i for i, row in enumerate(self.C)}
        assert len(self.lookup) == N_MIN

    def indices(self, S: np.ndarray) -> np.ndarray:
        """Canonical index of each row of S, -1 if the row is not in C."""
        S = np.asarray(S, dtype=np.int64)
        out = np.full(len(S), -1, dtype=np.int64)
        for k, row in enumerate(S):
            if row.min() < -128 or row.max() > 127:
                continue
            out[k] = self.lookup.get(row.astype(np.int8).tobytes(), -1)
        return out


@lru_cache(maxsize=None)
def leech_index() -> LeechIndex:
    return LeechIndex()


def vec_str(row) -> str:
    return "(" + " ".join(str(int(v)) for v in row) + ")"


# ---------------------------------------------------------------------------
# The certificate check
# ---------------------------------------------------------------------------
def check_set(S: np.ndarray, lines: list[int] | None = None, index: LeechIndex | None = None) -> dict:
    """Run the four checks. Returns a dict with at least ``ok``, ``size``, ``message``.

    On success also: ``idx`` (canonical indices, int64), ``antipodal`` (bool),
    ``gram`` (dict inner product -> number of unordered pairs), ``max_offdiag``.
    """
    S = np.asarray(S, dtype=np.int64).reshape(-1, DIM)
    n = len(S)
    lines = list(lines) if lines is not None else []

    def name(k: int) -> str:
        if k < len(lines):
            return f"line {lines[k]} (row {k})"
        return f"row {k}"

    res: dict = {"ok": False, "size": n, "message": ""}
    if n == 0:
        res.update(ok=True, message="ok", idx=np.zeros(0, dtype=np.int64), antipodal=True, gram={}, max_offdiag=-32)
        return res

    # 1. norms
    norms = (S * S).sum(axis=1)
    bad = np.flatnonzero(norms != 32)
    if len(bad):
        k = int(bad[0])
        res["message"] = (f"norm: {name(k)} has squared norm {int(norms[k])} != 32: {vec_str(S[k])} "
                          f"[{len(bad)} offending row(s)]")
        return res

    # 2. membership in C (dictionary lookup) + arithmetic cross-check
    index = index or leech_index()
    idx = index.indices(S)
    arith = is_lattice_vector_rows(S)  # lattice vector of norm 32 <=> minimal vector
    if not np.array_equal(idx >= 0, arith):
        k = int(np.flatnonzero((idx >= 0) != arith)[0])
        res["message"] = (f"internal: membership lookup and arithmetic test disagree at {name(k)}: "
                          f"{vec_str(S[k])}")
        return res
    bad = np.flatnonzero(idx < 0)
    if len(bad):
        k = int(bad[0])
        res["message"] = (f"membership: {name(k)} is not a Leech minimal vector: {vec_str(S[k])} "
                          f"[{len(bad)} offending row(s)]")
        return res

    # 3. distinct
    order = np.argsort(idx, kind="stable")
    dup = np.flatnonzero(idx[order][1:] == idx[order][:-1])
    if len(dup):
        a, b = sorted((int(order[dup[0]]), int(order[dup[0] + 1])))
        res["message"] = (f"distinct: {name(a)} and {name(b)} are the same vector: {vec_str(S[a])} "
                          f"[{len(dup)} duplicate row(s)]")
        return res

    # 4. Gram off-diagonal <= 8 (exact int64)
    G = S @ S.T
    assert np.array_equal(np.diagonal(G), np.full(n, 32))
    upper = np.triu(G, 1)  # strictly upper triangle, zeros elsewhere
    offenders = np.argwhere(upper > 8)  # row-major order = lexicographic (i, j)
    if len(offenders):
        i, j = (int(v) for v in offenders[0])
        ip = int(G[i, j])
        deg = " (60 degrees)" if ip == 16 else ""
        res["message"] = (f"gram: {name(i)} and {name(j)} have inner product {ip} > 8{deg}: "
                          f"{vec_str(S[i])} . {vec_str(S[j])} [{len(offenders)} offending pair(s)]")
        return res

    iu = np.triu_indices(n, 1)
    vals, counts = np.unique(G[iu], return_counts=True)
    gram = {int(v): int(c) for v, c in zip(vals, counts)}
    members = set(map(bytes, S.astype(np.int8)))
    antipodal = all((-row).astype(np.int8).tobytes() in members for row in S)
    res.update(ok=True, message="ok", idx=idx, antipodal=antipodal, gram=gram,
               max_offdiag=int(max(gram)) if gram else -32)
    return res


def gram_str(gram: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(gram.items()))


def quote(s: str) -> str:
    return '"' + s.replace('"', "'") + '"'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", help="S.txt (24 integers per line, '#' comments)")
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        S, lines = read_set(args.set)
    except (OSError, ValueError) as e:
        print(f"parse error: {e}")
        print(f"RESULT ok=0 size=0 file={quote(args.set)} reason={quote('parse: ' + str(e))}")
        return 1
    index = leech_index()
    t_gen = time.time() - t0
    res = check_set(S, lines, index)
    print(f"file      : {args.set}")
    print(f"C source  : kiss_ref.leech.leech_min_vectors() ({len(index.C)} vectors, {t_gen:.2f} s)")
    print(f"rows      : {res['size']}")
    if not res["ok"]:
        print(f"FAIL      : {res['message']}")
        print(f"RESULT ok=0 size={res['size']} file={quote(args.set)} reason={quote(res['message'])}")
        return 1
    print(f"antipodal : {int(res['antipodal'])}")
    print(f"gram      : {gram_str(res['gram'])} (unordered pairs by inner product)")
    print(f"RESULT ok=1 size={res['size']} antipodal={int(res['antipodal'])} "
          f"max_offdiag={res['max_offdiag']} gram={gram_str(res['gram'])} "
          f"file={quote(args.set)} s={time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

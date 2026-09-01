#!/usr/bin/env python3
"""A1 — standalone checker for the plateau-496 family (data/S496_family/).

    .venv/bin/python python/tools/check_64.py [data/S496_family] [--expect 64]

Dependencies: numpy + python/kiss_ref only (the pure-Python Golay/Leech
reference); it never reads data/leech_min.* or anything the C++ produced, and
it is independent of python/tools/materialize_64.py (integer matmul here,
float GEMM there).

For every S_*.txt in the directory it verifies

  1. exactly 496 rows, each of squared norm 32;
  2. every row is one of the 196560 Leech minimal vectors (dictionary lookup in
     the regenerated set, cross-checked against the arithmetic membership test);
  3. all rows distinct;
  4. every off-diagonal inner product <= 8 (no pair at 16: independence);
  5. S closed under negation (antipodal);

then computes the full tightness vector tight[v] = #{s in S : <v,s> = 16} over
ALL 196560 vertices (exact int32 matmul) and reports

  - the number of FREE vertices (tight = 0 outside S) — a free vertex means S
    extends to 497, i.e. a new record: reported loudly, run fails with exit 2;
  - the minimum tightness outside S and the counts of tightness-1/2/3 vertices
    (a nonzero count would mean a (k,k+1)-swap with k <= 3 exists);
  - the full tightness histogram.

One line per file, a summary, and `RESULT ok=<0|1> ...`.  Exit 0 iff every file
passes and no free vertex exists anywhere.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref.leech import DIM, N_MIN, is_lattice_vector_rows, leech_min_vectors  # noqa: E402


def read_rows(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            toks = text.split()
            if len(toks) != DIM:
                raise ValueError(f"{path}:{lineno}: expected {DIM} integers, got {len(toks)}")
            rows.append([int(t) for t in toks])
    return np.array(rows, dtype=np.int64).reshape(-1, DIM)


def hist_str(h) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(h.items()))


def check_file(path: str, C: np.ndarray, lut: dict) -> dict:
    r = {"file": path, "ok": False, "free": -1, "min_tight": -1, "t123": (-1, -1, -1)}
    S = read_rows(path)
    if len(S) != 496:
        r["msg"] = f"count: {len(S)} rows != 496"
        return r
    norms = (S * S).sum(axis=1)
    if not np.all(norms == 32):
        r["msg"] = f"norm: row {int(np.flatnonzero(norms != 32)[0])} has norm != 32"
        return r
    idx = np.array([lut.get(row.astype(np.int8).tobytes(), -1) for row in S], dtype=np.int64)
    arith = is_lattice_vector_rows(S)
    if not np.array_equal(idx >= 0, arith):
        r["msg"] = "internal: lookup and arithmetic membership tests disagree"
        return r
    if np.any(idx < 0):
        r["msg"] = f"membership: row {int(np.flatnonzero(idx < 0)[0])} is not a Leech minimal vector"
        return r
    if len(set(idx.tolist())) != 496:
        r["msg"] = "distinct: duplicate rows"
        return r
    G = S @ S.T  # exact int64
    off = G[np.triu_indices(496, 1)]
    if off.max() > 8:
        bad = int((off > 8).sum())
        r["msg"] = f"independence: {bad} pair(s) with inner product > 8 (max {int(off.max())})"
        return r
    gram = Counter(off.tolist())
    members = set(map(bytes, S.astype(np.int8)))
    if not all((-row).astype(np.int8).tobytes() in members for row in S):
        r["msg"] = "antipodal: S is not closed under negation"
        return r

    # full tightness vector over all 196560 vertices (exact integer matmul)
    Ci = C.astype(np.int32)
    Si = S.astype(np.int32)
    tight = np.empty(N_MIN, dtype=np.int32)
    for a in range(0, N_MIN, 32768):
        tight[a:a + 32768] = (Ci[a:a + 32768] @ Si.T == 16).sum(axis=1, dtype=np.int32)
    member = np.zeros(N_MIN, dtype=bool)
    member[idx] = True
    if int(tight[member].max()) != 0:
        r["msg"] = "internal: a member has nonzero tightness (contradicts independence)"
        return r
    t_out = tight[~member]
    free = int((t_out == 0).sum())
    r.update(ok=True, msg="ok", free=free, min_tight=int(t_out.min()),
             t123=tuple(int((t_out == k).sum()) for k in (1, 2, 3)),
             hist=Counter(t_out[t_out > 0].tolist()), gram=gram,
             free_vertices=np.flatnonzero((tight == 0) & ~member).tolist() if free else [])
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", nargs="?", default="data/S496_family")
    ap.add_argument("--expect", type=int, default=64, help="expected number of S_*.txt files")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "S_*.txt")))
    print(f"dir       : {args.dir} ({len(files)} set files, expected {args.expect})")
    C = leech_min_vectors()
    lut = {row.tobytes(): i for i, row in enumerate(C)}
    print(f"C source  : kiss_ref.leech.leech_min_vectors() ({len(C)} vectors, regenerated)")

    n_ok = free_total = 0
    fails, hists, mints = [], set(), Counter()
    record_files = []
    for path in files:
        try:
            r = check_file(path, C, lut)
        except (OSError, ValueError) as e:
            r = {"file": path, "ok": False, "msg": f"parse: {e}", "free": -1,
                 "min_tight": -1, "t123": (-1, -1, -1)}
        base = os.path.basename(path)
        if not r["ok"]:
            fails.append(base)
            print(f"{base}: FAIL {r['msg']}")
            continue
        n_ok += 1
        free_total += r["free"]
        hists.add(hist_str(r["hist"]))
        mints[r["min_tight"]] += 1
        t1, t2, t3 = r["t123"]
        print(f"{base}: ok=1 size=496 antipodal=1 gram={hist_str(r['gram'])} "
              f"free={r['free']} min_tight={r['min_tight']} t1={t1} t2={t2} t3={t3} "
              f"tight_hist={hist_str(r['hist'])}")
        if r["free"]:
            record_files.append(base)
            print(f"*** {base}: {r['free']} FREE VERTEX/VERTICES {r['free_vertices']} — "
                  f"THIS SET EXTENDS TO {496 + r['free'] if r['free'] else 497}: "
                  f"POTENTIAL NEW RECORD, INVESTIGATE IMMEDIATELY ***")

    ok = (not fails) and len(files) == args.expect and free_total == 0
    print(f"summary   : {n_ok}/{len(files)} files pass; free vertices total {free_total}; "
          f"min-tightness values {sorted(mints.items())}; distinct tightness histograms {len(hists)}")
    print(f"RESULT ok={int(ok)} files={len(files)} pass={n_ok} fail={len(fails)} "
          f"free_total={free_total} min_tight={'/'.join(str(k) for k in sorted(mints))} "
          f"distinct_hists={len(hists)}"
          + (f" record_candidates={','.join(record_files)}" if record_files else "")
          + (f" failed={','.join(fails)}" if fails else ""))
    if record_files:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

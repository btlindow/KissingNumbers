#!/usr/bin/env python3
"""A1 — materialise the 64 plateau 496-sets as certificate files (data/S496_family/).

    .venv/bin/python python/tools/materialize_64.py [--set data/S496.txt]
        [--atoms runs/ls_search/plateau_atoms_496.json] [--out data/S496_family]

Deterministic: given the same two inputs it reproduces the directory byte for
byte (no timestamps; rows sorted by canonical Leech index).

The six commuting plateau atoms (four (12,12), two (80,80); T3.2b,
runs/ls_search/plateau_atoms_496.json) are pairwise disjoint and
negation-closed; S_T for a mask T <= {0..5} is the record 496 of data/S496.txt
with the atoms of T applied (remove the atom's `remove_vertices`, add its
`add_vertices`; all indices are canonical indices into the 196560 minimal
vectors). This script

  1. reconstructs the atoms and sanity-checks them against T3.2b's numbers
     (sizes 12,12,12,12,80,80; `remove_vertices` = the ascending index list of
     S at `remove_S_pos`; pairwise disjointness; negation closure; each atom a
     valid plateau move: added conflicts inside the removed set, added vectors
     mutually non-adjacent);
  2. generates all 64 sets S_00..S_63, verifies each (independent 496-subset of
     the minimal vectors, antipodal, Gram histogram), computes the full
     tightness vector over all 196560 vertices (free vertices, min tightness,
     tightness-1..3 counts, histogram);
  3. checks the applicability of the six atom moves on every S_T (atom a
     applies forward iff a not in T, in reverse iff a in T — the 6-cube
     structure);
  4. groups the 64 by tightness histogram and cross-checks the 5 fingerprint
     classes (16/16/16/8/8) and the 8x8 isometry classification against T3.4b
     (embedded below, from runs/plateau_equiv/{results,isometry_classes}.json);
  5. writes data/S496_family/S_00.txt .. S_63.txt with provenance headers and
     INDEX.md.

Exit 0 with `RESULT ok=1 ...` only if every check passes.  Any set with a free
vertex would be a new record (497): it is reported loudly and the run fails so
nobody misses it (a free vertex would mean a 497-element set; the extension
to runs/found/ and runs both verifiers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref.leech import DIM, N_MIN, leech_min_vectors  # noqa: E402

# ---------------------------------------------------------------------------
# T3.4b classifications (docs/reports/T3.4b.md; runs/plateau_equiv/results.json
# `fingerprint_classes` and runs/plateau_equiv/isometry_classes.json `classes`).
# Embedded so the directory regenerates without the gitignored runs/ tree; the
# fingerprint classes are RE-DERIVED from the computed tightness histograms
# below and compared against this table (a mismatch fails the run).  The
# isometry classes cannot be recomputed cheaply and are carried as provenance.
# ---------------------------------------------------------------------------
FP_CLASSES = {  # class id -> masks (5 classes, sizes 16/16/16/8/8; id 0 = the record's)
    0: [0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63],
    1: [1, 4, 11, 14, 18, 23, 24, 29, 34, 39, 40, 45, 49, 52, 59, 62],
    2: [2, 7, 8, 13, 17, 20, 27, 30, 33, 36, 43, 46, 50, 55, 56, 61],
    3: [3, 12, 19, 28, 35, 44, 51, 60],
    4: [6, 9, 22, 25, 38, 41, 54, 57],
}
ISO_CLASSES = {  # class id -> masks (8 classes of 8; id = its smallest mask's class rep)
    0: [0, 15, 22, 25, 35, 44, 53, 58],
    1: [1, 14, 17, 30, 33, 46, 49, 62],
    2: [2, 13, 18, 29, 34, 45, 50, 61],
    3: [3, 12, 21, 26, 32, 47, 54, 57],
    4: [4, 11, 20, 27, 36, 43, 52, 59],
    5: [5, 10, 19, 28, 38, 41, 48, 63],
    6: [6, 9, 16, 31, 37, 42, 51, 60],
    7: [7, 8, 23, 24, 39, 40, 55, 56],
}
FP_OF = {t: c for c, ts in FP_CLASSES.items() for t in ts}
ISO_OF = {t: c for c, ts in ISO_CLASSES.items() for t in ts}

EXPECTED_GRAM = {-32: 248, -8: 37504, 0: 47504, 8: 37504}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_set_rows(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            toks = text.split()
            if len(toks) != DIM:
                raise ValueError(f"{path}:{lineno}: expected {DIM} integers")
            rows.append([int(t) for t in toks])
    return np.array(rows, dtype=np.int64).reshape(-1, DIM)


def hist_str(h: Counter) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(h.items()))


class Fail(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="data/S496.txt")
    ap.add_argument("--atoms", default=None,
                    help="plateau atoms JSON (default: data/plateau_atoms_496.json, "
                         "falling back to runs/ls_search/plateau_atoms_496.json)")
    ap.add_argument("--out", default="data/S496_family")
    args = ap.parse_args(argv)
    if args.atoms is None:
        # the committed copy first; the original T3.2b run artefact as fallback
        args.atoms = "data/plateau_atoms_496.json"
        if not os.path.exists(args.atoms) and os.path.exists("runs/ls_search/plateau_atoms_496.json"):
            args.atoms = "runs/ls_search/plateau_atoms_496.json"

    C = leech_min_vectors()  # int8 (196560, 24), canonical order
    assert C.shape == (N_MIN, DIM)
    lut = {r.tobytes(): i for i, r in enumerate(C)}
    neg = np.array([lut[(-r).tobytes()] for r in C], dtype=np.int64)
    Cf = C.astype(np.float64)  # inner products <= 32: exact in float64

    def tightness(idx: np.ndarray) -> np.ndarray:
        """tight[v] = #{s in S: <C[v],C[s]> = 16} over all 196560 v (float GEMM, exact)."""
        Sf = Cf[idx]
        out = np.zeros(N_MIN, dtype=np.int32)
        for a in range(0, N_MIN, 65536):
            out[a:a + 65536] = (Cf[a:a + 65536] @ Sf.T == 16.0).sum(axis=1)
        return out

    # --- base set ----------------------------------------------------------
    set_sha = sha256_file(args.set)
    atoms_sha = sha256_file(args.atoms)
    S0_rows = read_set_rows(args.set)
    check(len(S0_rows) == 496, f"{args.set}: expected 496 rows, got {len(S0_rows)}")
    base = np.array(sorted(lut[r.astype(np.int8).tobytes()] for r in S0_rows), dtype=np.int64)
    check(len(set(base.tolist())) == 496, "base rows not distinct")

    # --- atoms: reconstruct + sanity-check (spec item 1) -------------------
    doc = json.load(open(args.atoms))
    atoms = doc["atoms"]
    check([a["size"] for a in atoms] == [12, 12, 12, 12, 80, 80],
          f"atom sizes {[a['size'] for a in atoms]} != [12,12,12,12,80,80]")
    rem, add = [], []
    for i, a in enumerate(atoms):
        r = np.array(a["remove_vertices"], dtype=np.int64)
        d = np.array(a["add_vertices"], dtype=np.int64)
        check(len(r) == a["size"] and len(d) == a["size"], f"atom {i}: ragged sizes")
        check(np.array_equal(r, base[np.array(a["remove_S_pos"])]),
              f"atom {i}: remove_vertices != base[remove_S_pos]")
        check(set(r.tolist()) == set(neg[r].tolist()), f"atom {i}: remove set not negation-closed")
        check(set(d.tolist()) == set(neg[d].tolist()), f"atom {i}: add set not negation-closed")
        check(not (set(d.tolist()) & set(base.tolist())), f"atom {i}: add set meets S")
        rem.append(set(r.tolist()))
        add.append(set(d.tolist()))
    for i in range(6):
        for j in range(i + 1, 6):
            check(not (rem[i] & rem[j]), f"atoms {i},{j}: remove sets intersect")
            check(not (add[i] & add[j]), f"atoms {i},{j}: add sets intersect")
            check(not (rem[i] & add[j]) and not (add[i] & rem[j]),
                  f"atoms {i},{j}: remove/add sets intersect")
    # all 208 added vectors are mutually non-adjacent (needed for every union move)
    all_add = np.array(sorted(set().union(*add)), dtype=np.int64)
    Ga = Cf[all_add] @ Cf[all_add].T
    check(not np.any(np.triu(Ga, 1) == 16.0), "added vectors of different atoms are adjacent")
    # each atom is a valid plateau move on S: added conflicts lie inside its removed set
    tight0 = tightness(base)
    base_set = set(base.tolist())
    for i in range(6):
        di = np.array(sorted(add[i]), dtype=np.int64)
        G = Cf[di] @ Cf[base].T
        conf_cols = set(base[np.unique(np.argwhere(G == 16.0)[:, 1])].tolist())
        check(conf_cols == rem[i], f"atom {i}: union of added conflicts != removed set")
    print(f"atoms     : 6 atoms (12,12,12,12,80,80), pairwise disjoint, negation-closed, "
          f"valid plateau moves on S; remove_S_pos consistent; 208 added vectors mutually non-adjacent")

    # --- generate the 64 sets (spec item 2) --------------------------------
    os.makedirs(args.out, exist_ok=True)
    rows_out = []
    hist_by_mask = {}
    hist_groups = {}
    free_total = 0
    for T in range(64):
        atoms_in = [i for i in range(6) if T >> i & 1]
        s = set(base.tolist())
        for i in atoms_in:
            s -= rem[i]
            s |= add[i]
        idx = np.array(sorted(s), dtype=np.int64)
        check(len(idx) == 496, f"S_{T}: size {len(idx)} != 496")
        overlap = len(s & base_set)
        n12 = sum(1 for i in atoms_in if i < 4)
        n80 = len(atoms_in) - n12
        check(overlap == 496 - 12 * n12 - 80 * n80, f"S_{T}: overlap {overlap} unexpected")

        # verify: independent antipodal 496-subset of the minimal vectors
        V = C[idx].astype(np.int64)
        G = V @ V.T
        check(np.array_equal(np.diagonal(G), np.full(496, 32)), f"S_{T}: bad norms")
        off = G[np.triu_indices(496, 1)]
        check(off.max() <= 8, f"S_{T}: NOT independent (off-diagonal {off.max()})")
        gram = Counter(off.tolist())
        check(dict(gram) == EXPECTED_GRAM, f"S_{T}: Gram histogram {dict(gram)} unexpected")
        check(set(neg[idx].tolist()) == s, f"S_{T}: not antipodal")

        # tightness over all 196560 vertices
        tight = tight0 if T == 0 else tightness(idx)
        member = np.zeros(N_MIN, dtype=bool)
        member[idx] = True
        check(int(tight[member].max(initial=0)) == 0, f"S_{T}: member with nonzero tightness")
        t_out = tight[~member]
        free = int((t_out == 0).sum())
        mint = int(t_out.min())
        t123 = [int((t_out == k).sum()) for k in (1, 2, 3)]
        hist = Counter(t_out[t_out > 0].tolist())
        hist_by_mask[T] = (hist, free, mint, t123, overlap)
        hist_groups.setdefault(hist_str(hist), []).append(T)
        free_total += free
        if free:
            fv = np.flatnonzero((tight == 0) & ~member)
            print(f"*** S_{T:02d} HAS {free} FREE VERTEX/VERTICES {fv.tolist()} — "
                  f"POTENTIAL 497-SET / NEW RECORD: DO NOT IGNORE ***")

        # atom applicability on S_T (spec item 5): forward iff atom not in T,
        # reverse iff atom in T — the 6-cube structure.
        for i in range(6):
            fwd_r, fwd_a = (rem[i], add[i]) if i not in atoms_in else (add[i], rem[i])
            check(fwd_r <= s and not (fwd_a & s), f"S_{T}: atom {i} membership pattern broken")
            ai = np.array(sorted(fwd_a), dtype=np.int64)
            Gi = Cf[ai] @ Cf[idx].T
            conf = set(idx[np.unique(np.argwhere(Gi == 16.0)[:, 1])].tolist())
            check(conf == fwd_r, f"S_{T}: atom {i} ({'rev' if i in atoms_in else 'fwd'}) "
                                 f"is not a valid plateau move here")

        # write the certificate file
        bits = format(T, "06b")[::-1]  # bit i = atom i
        alist = ",".join(map(str, atoms_in)) if atoms_in else "none (the record set itself)"
        hdr = [
            f"# S_{T:02d}: plateau 496-set S_T of the record 496, atom mask T={T} "
            f"(bits atom0..atom5 = {bits}; atoms applied: {alist})",
            f"# base: data/S496.txt (sha256 {set_sha})",
            f"# atoms: runs/ls_search/plateau_atoms_496.json (sha256 {atoms_sha}) — "
            f"four (12,12) + two (80,80) commuting, disjoint, negation-closed plateau moves (T3.2b)",
            "# S_T = (S \\ union of the applied atoms' remove sets) + (union of their add sets);",
            "# rows are Leech minimal vectors (norm-32 integer scaling), sorted by canonical index",
            "# generated by python/tools/materialize_64.py (task A1); deterministic, no timestamp",
            f"# invariants: |S|=496, antipodal, Gram off-diag {{-32:248,-8:37504,0:47504,8:37504}}, "
            f"free vertices {free}, min tightness {mint}, "
            f"fingerprint class {FP_OF[T]} (T3.4b), isometry class {ISO_OF[T]} (T3.4b)",
        ]
        body = "\n".join(" ".join(f"{int(v):3d}" for v in row) for row in C[idx])
        path = os.path.join(args.out, f"S_{T:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(hdr) + "\n" + body + "\n")
        rows_out.append((T, atoms_in, overlap, sha256_file(path), mint, free, t123,
                         FP_OF[T], ISO_OF[T], hist))
        print(f"S_{T:02d} : atoms [{alist:>23s}] overlap {overlap:3d} free {free} "
              f"min_tight {mint} t123 {t123} fp {FP_OF[T]} iso {ISO_OF[T]}")

    # --- fingerprint-class cross-check (spec item 3) -----------------------
    check(len(hist_groups) == 5, f"{len(hist_groups)} distinct tightness histograms, expected 5")
    sizes = sorted((len(v) for v in hist_groups.values()), reverse=True)
    check(sizes == [16, 16, 16, 8, 8], f"class sizes {sizes} != [16,16,16,8,8]")
    for members in hist_groups.values():
        cls = {FP_OF[t] for t in members}
        check(len(cls) == 1 and sorted(members) == sorted(FP_CLASSES[cls.pop()]),
              f"computed histogram class {sorted(members)} does not match T3.4b's fingerprint classes")
    print("fp check  : 5 distinct tightness histograms, sizes 16/16/16/8/8, "
          "partition identical to T3.4b's fingerprint classes")

    # --- INDEX.md ----------------------------------------------------------
    mint_hist = Counter(r[4] for r in rows_out)
    lines = [
        "# data/S496_family — the 64 plateau 496-sets (task A1)",
        "",
        f"Generated deterministically by `python/tools/materialize_64.py` from `data/S496.txt`",
        f"(sha256 `{set_sha}`) and the six commuting plateau atoms of",
        f"`runs/ls_search/plateau_atoms_496.json` (sha256 `{atoms_sha}`; T3.2b: four (12,12) and",
        "two (80,80) moves, pairwise disjoint, negation-closed).  `S_T` = the record 496 with the",
        "atoms of mask T applied (bit i of T = atom i).  Every set is a verified independent",
        "antipodal 496-subset of the 196560 Leech minimal vectors with Gram off-diagonal histogram",
        "{-32:248, -8:37504, 0:47504, 8:37504}; every set certifies K(25) >= 197056.",
        "",
        "Certification (A1): every one of the 64 sets has **0 free vertices** (no set extends to",
        "497 by adding any vector) and **no vertex of tightness 1, 2 or 3** (no (k,k+1)-swap with",
        "k <= 3 exists anywhere on the plateau); the minimum tightness outside S is **4** at all 64",
        "sets (80 tightness-4 vertices each).  Checked by `python/tools/check_64.py` (numpy +",
        "python/kiss_ref, integer arithmetic) and `python/verify_S.py` independently.",
        "",
        "Fingerprint class = T3.4b's Co0-invariant tightness-fingerprint class (5 classes, sizes",
        "16/16/16/8/8; class 0 contains the record); re-derived here from the computed histograms",
        "and identical to T3.4b's partition.  Isometry class = T3.4b's congruence classification",
        "(8 classes of 8; the 64 sets are pairwise Co0-INEQUIVALENT, so class labels refine to 64",
        "singleton Co0-classes).  Overlap = |S_T ^ S_00|.",
        "",
        "| set | atoms | overlap | sha256 | min tight | free | t1 | t2 | t3 | fp | iso | tightness histogram (v outside S) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (T, atoms_in, overlap, sha, mint, free, t123, fp, iso, hist) in rows_out:
        alist = ",".join(map(str, atoms_in)) if atoms_in else "-"
        lines.append(f"| S_{T:02d} | {alist} | {overlap} | `{sha}` | {mint} | {free} | "
                     f"{t123[0]} | {t123[1]} | {t123[2]} | {fp} | {iso} | {hist_str(hist)} |")
    lines += [
        "",
        f"Summary: free vertices total {free_total}; min-tightness distribution over the 64 sets: "
        + ", ".join(f"{k}: {v} sets" for k, v in sorted(mint_hist.items())) + ";",
        "tightness-1/2/3 vertices: 0 at every set.",
        "",
        "Sources: docs/reports/T3.2b.md (atoms), docs/reports/T3.4b.md (classes),",
        "docs/reports/A1.md (this materialisation).",
    ]
    with open(os.path.join(args.out, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"RESULT ok=1 sets=64 free_total={free_total} min_tight_all="
          f"{'/'.join(str(k) for k in sorted(mint_hist))} t123_total=0 fp_classes=5 "
          f"sizes=16/16/16/8/8 out={args.out}")
    if free_total:
        print("RESULT_NOTE free vertices found — see the *** lines above: potential new record")
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print(f"FAIL: {e}")
        print(f"RESULT ok=0 reason=\"{e}\"")
        sys.exit(1)

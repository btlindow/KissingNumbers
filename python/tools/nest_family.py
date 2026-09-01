#!/usr/bin/env python3
"""T4.2 — derive the nested families dim<n> (n = 26..30) from a 42-set dim31 family directory.

A chain family S_i = g^{i-1} S (tools/disjoint_family --method chain) is pairwise disjoint, so any
k of its sets form a family for the dimension whose record uses k sets (README §1.3: 2, 5, 8, 14, 24).
This tool writes data/families/<out>/dim<n>/family.json referencing ../dim31/S_01.txt .. S_<k>.txt
(SCHEMA.md §4 nesting, exactly as PackingStar's families are stored) with the T/extra blocks copied
verbatim from the template family of that dimension (data/families/dim<n>/family.json by default:
the record's T configuration, groups, and extra spheres), and the count recomputed from the sizes.

Usage:
    nest_family.py --dim31 data/families/ours/dim31 [--templates data/families] [--dims 26,27,28,29,30]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

K_FOR_DIM = {26: 2, 27: 5, 28: 8, 29: 14, 30: 24}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim31", required=True, help="directory with family.json and S_01..S_42.txt")
    ap.add_argument("--templates", default=os.path.join("data", "families"))
    ap.add_argument("--dims", default="26,27,28,29,30")
    args = ap.parse_args()
    d31 = os.path.abspath(args.dim31)
    with open(os.path.join(d31, "family.json")) as f:
        fam31 = json.load(f)
    sets31 = fam31["sets"]
    parent = os.path.dirname(d31)
    for n in (int(x) for x in args.dims.split(",")):
        k = K_FOR_DIM[n]
        if len(sets31) < k:
            print(f"dim{n}: need {k} sets, dim31 has {len(sets31)}", file=sys.stderr)
            return 1
        with open(os.path.join(args.templates, f"dim{n}", "family.json")) as f:
            tpl = json.load(f)
        groups = tpl["T"]["groups"]
        if len(groups) != k:
            print(f"dim{n}: template has {len(groups)} groups, expected {k}", file=sys.stderr)
            return 1
        out_dir = os.path.join(parent, f"dim{n}")
        os.makedirs(out_dir, exist_ok=True)
        sets = []
        for s in sets31[:k]:
            src = os.path.join(d31, s["file"])
            rel = os.path.relpath(src, out_dir).replace(os.sep, "/")
            sets.append({"file": rel, "size": s["size"], "sha256": sha256_file(src)})
        lifted = sum((len(g) - 1) * s["size"] for g, s in zip(groups, sets))
        n_extra = tpl["extra"]["count"]
        fam = {
            "schema_version": 1,
            "dim": n,
            "d": n - 24,
            "coordinates": "leech-sqrt8-integer",
            "sets": sets,
            "T": tpl["T"],
            "extra": tpl["extra"],
            "count": n_extra + 196560 + lifted,
            "count_terms": {"extra": n_extra, "leech": 196560, "lifted": lifted},
            "count_formula": f"{n_extra} + 196560 + sum_i (|T_i| - 1) * |S_i|",
            "provenance": {
                "tool": "python/tools/nest_family.py",
                "note": f"S_1..S_{k} of the dim31 chain family {os.path.relpath(d31, parent)} "
                        f"(S_i = g^(i-1) S, pairwise disjoint, see its family.json / g.txt); "
                        f"T and extra copied from {os.path.join(args.templates, f'dim{n}', 'family.json')}",
                "dim31_provenance": fam31.get("provenance"),
                "date": _dt.date.today().isoformat(),
            },
        }
        with open(os.path.join(out_dir, "family.json"), "w") as f:
            json.dump(fam, f, indent=1)
            f.write("\n")
        print(f"dim{n}: {k} sets -> {out_dir}/family.json, count {fam['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

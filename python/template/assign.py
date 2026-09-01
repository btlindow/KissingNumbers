"""Joint assignment of the S_i to the T groups, and rewriting a family with the optimal partition (T5.1 items 4-5).

count = #extra + 196560 + sum_i (|T_i| - 1)·|S_i|   (README 1.3, SCHEMA.md 5)

Given sizes |S_i| and groups T_i, the count is maximised by giving the largest S_i to the heaviest
groups (weight |T_i| - 1 in {1, 2}): sort both and pair them (assign_sets_to_groups; the rearrangement
inequality makes this optimal and it is the ONLY freedom once sizes and groups are fixed). Since every
record set has |S_i| = 496 the assignment is irrelevant today; the levers are the number of triangles
(triangles.py: the counting bound 2·floor(K/3) + [K mod 3 = 2] is attained for every d once the
d = 3 partition uses 4 triangles) and the number of extras (extra_spheres.py: <= K(d)).

    python/template/assign.py data/families/ours/dim27 runs/template/dim27_ours --optimal
    python/template/assign.py data/families/dim27      runs/template/dim27_record --optimal

writes <out>/family.json + copied S_xx.txt with the optimal partition of config(d) (max_weight_partition),
the largest sets assigned to the triangles, the extras copied from the base family (the extra-sphere
condition depends on T as a set only, not on its partition), and the count recomputed. Verify with
python/verify_dimN.py <out>.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiss_ref import small_kissing as sk  # noqa: E402
from template.triangles import max_weight_partition  # noqa: E402

LEECH = 196560

__all__ = ["family_count", "assign_sets_to_groups", "load_raw", "rewrite_family"]


def family_count(sizes: list[int], groups: list[list[int]], n_extra: int) -> int:
    if len(sizes) != len(groups):
        raise ValueError(f"{len(sizes)} sets but {len(groups)} groups")
    return n_extra + LEECH + sum((len(g) - 1) * s for s, g in zip(sizes, groups))


def assign_sets_to_groups(sizes: list[int], groups: list[list[int]]) -> dict:
    """Largest sets to heaviest groups. Returns {"order": set index for group j, "count": ..., "groups": groups}.
    Uses the first len(groups) sets if there are more sets than groups (the largest ones)."""
    if len(sizes) < len(groups):
        raise ValueError(f"need {len(groups)} sets, have {len(sizes)}")
    by_size = sorted(range(len(sizes)), key=lambda i: -sizes[i])[:len(groups)]
    by_weight = sorted(range(len(groups)), key=lambda j: -len(groups[j]))
    order = [0] * len(groups)
    for si, gj in zip(by_size, by_weight):
        order[gj] = si
    chosen_sizes = [sizes[i] for i in order]
    return {"order": order, "sizes": chosen_sizes, "groups": groups,
            "count_without_extra": LEECH + sum((len(g) - 1) * s for s, g in zip(chosen_sizes, groups))}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(family_dir: str) -> tuple[dict, list[str]]:
    """family.json dict and the absolute paths of its set files."""
    with open(os.path.join(family_dir, "family.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    files = [os.path.normpath(os.path.join(family_dir, s["file"])) for s in raw["sets"]]
    return raw, files


def optimal_groups(d: int) -> list[list[int]]:
    """Optimal partition of config(d): triangles first, then pairs (indices into config(d) rows)."""
    V, _ = sk.config(d)
    part = max_weight_partition(V)
    assert part["proven"]
    return [list(t) for t in part["triangles"]] + [list(p) for p in part["pairs"]]


def rewrite_family(base_dir: str, out_dir: str, groups: list[list[int]] | None = None, note: str = "") -> dict:
    """Write out_dir/family.json (+ copied S files) from base_dir with the given groups (indices into
    config(d)); groups=None keeps the base T block and only re-orders sets by size. Returns the new json."""
    raw, files = load_raw(base_dir)
    d = int(raw["d"])
    sizes = [int(s["size"]) for s in raw["sets"]]
    if groups is None:
        T_block = raw["T"]
        groups = T_block["groups"]
    else:
        T_block = sk.schema_T_block(d, {"triangles": [g for g in groups if len(g) == 3],
                                         "pairs": [g for g in groups if len(g) == 2]})
    asg = assign_sets_to_groups(sizes, groups)
    os.makedirs(out_dir, exist_ok=True)
    sets = []
    for j, si in enumerate(asg["order"]):
        name = f"S_{j + 1:02d}.txt"
        dst = os.path.join(out_dir, name)
        shutil.copyfile(files[si], dst)
        sets.append({"file": name, "size": sizes[si], "sha256": sha256_file(dst),
                     "from": os.path.relpath(files[si], out_dir)})
    n_extra = int(raw["extra"]["count"])
    count = family_count([s["size"] for s in sets], groups, n_extra)
    weights = [len(g) - 1 for g in groups]
    new = {
        "schema_version": raw.get("schema_version", 1), "dim": 24 + d, "d": d,
        "coordinates": raw.get("coordinates", "leech-sqrt8-integer"),
        "sets": sets, "T": T_block, "extra": raw["extra"], "count": count,
        "count_terms": {"extra": n_extra, "leech": LEECH, "lifted": count - n_extra - LEECH},
        "count_formula": f"{n_extra} + {LEECH} + sum_i (|T_i| - 1) * |S_i|  (weights {weights})",
        "provenance": {"tool": "python/template/assign.py (T5.1)", "base": os.path.relpath(base_dir, out_dir),
                       "base_count": raw.get("count"), "date": _dt.date.today().isoformat(),
                       "note": note or "sets copied from the base family (largest sets assigned to the triangles); "
                                       "T = config(d) with the max-weight partition of template/triangles.py; "
                                       "extras copied verbatim from the base (they depend on T as a set only)"},
    }
    with open(os.path.join(out_dir, "family.json"), "w", encoding="utf-8") as f:
        json.dump(new, f, indent=1)
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="base family directory")
    ap.add_argument("out", nargs="?", help="output directory (omit: just print the count analysis)")
    ap.add_argument("--optimal", action="store_true", help="use the optimal partition of config(d)")
    a = ap.parse_args(argv)
    raw, _ = load_raw(a.base)
    d = int(raw["d"])
    sizes = [int(s["size"]) for s in raw["sets"]]
    groups = raw["T"]["groups"]
    n_extra = int(raw["extra"]["count"])
    cur = family_count(sizes, groups, n_extra)
    print(f"base {a.base}: d={d} sets={len(sizes)} sizes={sorted(set(sizes))} groups={len(groups)} "
          f"weights={[len(g) - 1 for g in groups]} extra={n_extra} count={cur} (claimed {raw.get('count')})")
    opt = optimal_groups(d)
    asg = assign_sets_to_groups(sizes, opt)
    best = asg["count_without_extra"] + n_extra
    print(f"optimal partition of config({d}): {sum(len(g) == 3 for g in opt)} triangles + {sum(len(g) == 2 for g in opt)} pairs, "
          f"sets used {len(opt)}, count {best} ({best - cur:+d})")
    if a.out:
        new = rewrite_family(a.base, a.out, opt if a.optimal else None)
        print(f"wrote {a.out}/family.json count={new['count']} sets={len(new['sets'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

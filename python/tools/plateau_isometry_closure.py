#!/usr/bin/env python3
"""T3.4b follow-up: pin down the ISOMETRY classes of the 64 plateau sets exactly.

The main run (plateau_equivalence.py) tests every pair inside one Co_0-fingerprint class
and every pair (S_0, S_T); that leaves the cross-fingerprint pairs undecided, so it can
only report a REFINEMENT of the true isometry partition.  Isometry is an equivalence
relation, so it is enough to test one representative pair per pair of observed classes,
and only when the two have the same 1-WL invariant (otherwise they are provably not
isometric).  Uses the same exact machinery; no Leech test is needed here.

    .venv/bin/python python/tools/plateau_isometry_closure.py [--jobs 12]
        [--results runs/plateau_equiv/results.json] [--atoms data/plateau_atoms_496.json]
        [--out runs/plateau_equiv/isometry_classes.json]

Requires `results.json` from a full (not --quick) plateau_equivalence.py run.
Reported at T3.4b: 8 isometry classes of size 8 (73 cross-class tests, ~4 min).

This file was originally written and run as `runs/plateau_equiv/isometry_closure.py`
(gitignored, with absolute paths hard-coded); it is committed here unchanged in logic
so that a fresh clone can reproduce the isometry half of the T3.4b claim.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python", "tools"))
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np  # noqa: E402
import plateau_equivalence as PE  # noqa: E402


def job(pair):
    a, b = pair
    L, sets = PE._W["L"], PE._W["sets"]
    isos, gi = PE.gram_isomorphisms(L.C[sets[a]], L.C[sets[b]], limit=1)
    return a, b, len(isos)


def default_atoms() -> str:
    """The committed copy first, the original T3.2b run artefact as fallback."""
    p = os.path.join(ROOT, "data", "plateau_atoms_496.json")
    fallback = os.path.join(ROOT, "runs", "ls_search", "plateau_atoms_496.json")
    if not os.path.exists(p) and os.path.exists(fallback):
        return fallback
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--results", default=os.path.join(ROOT, "runs", "plateau_equiv", "results.json"))
    ap.add_argument("--atoms", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "plateau_equiv",
                                                  "isometry_classes.json"))
    args = ap.parse_args()
    if args.atoms is None:
        args.atoms = default_atoms()
    if not os.path.exists(args.results):
        sys.exit(f"{args.results} missing — run python/tools/plateau_equivalence.py "
                 f"(full, ~45 min) first")

    R = json.load(open(args.results))
    obs = {int(k): sorted(int(x) for x in v) for k, v in R["iso_classes"].items()}
    wl = {int(k): R["wl_invariants"][k][2] for k in R["wl_invariants"]}
    L = PE.Leech(os.path.join(ROOT, "data"))
    S_idx = np.sort(L.index_of(PE.read_set(os.path.join(ROOT, "data", "S496.txt"))))
    atoms = json.load(open(args.atoms))["atoms"]
    Rm = [np.array(a["remove_vertices"]) for a in atoms]
    Ad = [np.array(a["add_vertices"]) for a in atoms]

    def set_T(T):
        keep = np.ones(len(S_idx), bool)
        add = []
        for k in range(6):
            if (T >> k) & 1:
                keep &= ~np.isin(S_idx, Rm[k])
                add.append(Ad[k])
        return np.sort(np.concatenate([S_idx[keep]] + add)) if add else S_idx.copy()

    sets = [set_T(T) for T in range(64)]
    PE._W["L"], PE._W["sets"], PE._W["S_idx"] = L, sets, S_idx

    reps = sorted(obs)
    parent = {r: r for r in reps}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    jobs = [(a, b) for i, a in enumerate(reps) for b in reps[i + 1:] if wl[a] == wl[b]]
    print(f"observed classes: {len(reps)} {reps}")
    print(f"cross-class tests with equal 1-WL invariant: {len(jobs)} "
          f"(the other {len(reps)*(len(reps)-1)//2 - len(jobs)} pairs differ in 1-WL, "
          f"hence are not isometric)")
    t0 = time.time()
    out = []
    for a, b, n in PE.pmap(job, jobs, args.jobs):
        out.append({"a": a, "b": b, "n_iso": n})
        print(f"  S_{a} vs S_{b}: {'ISOMETRIC' if n else 'not isometric'}", flush=True)
        if n:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    merged = {}
    for r in reps:
        merged.setdefault(find(r), []).extend(obs[r])
    for k in merged:
        merged[k] = sorted(merged[k])
    print(f"\nISOMETRY classes among the 64: {len(merged)}; sizes "
          f"{sorted((len(v) for v in merged.values()), reverse=True)}")
    for k, v in sorted(merged.items()):
        xors = sorted({v[0] ^ t for t in v})
        print(f"  class rep S_{k} ({len(v)} sets): {v}   [T ^ rep = {xors}]")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"classes": {str(k): v for k, v in merged.items()}, "tests": out,
               "seconds": round(time.time() - t0, 1)},
              open(args.out, "w"), indent=1)
    print(f"\nRESULT ok=1 iso_classes={len(merged)} tests={len(jobs)} "
          f"seconds={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()

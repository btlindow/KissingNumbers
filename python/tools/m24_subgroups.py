#!/usr/bin/env python3
"""Export small subgroups of M24 (sympy) as monomial lists for tools/orbit_mis (T3.4 item 3).

    python python/tools/m24_subgroups.py [--out runs/orbits/subgroups] [--slow]

Writes one file per subgroup, format = the monomial list read by
kiss::load_monomial_list: 24 coordinate images followed by the sign mask (0
here — these are pure coordinate permutations). Subgroups: Sylow p-subgroups
(p = 2, 3, 5, 7, 11, 23), pointwise stabilisers of 1..5 points (M23, M22,
M21 = PSL(3,4), 2^4:A5, order 48), the derived subgroup and centre of the
Sylow 2-subgroup, the Borel subgroup 23:11 and the Sylow 2-subgroup of
PSL(2,23), and (with --slow, centralizer() takes minutes) centralisers of
elements of order 2, 3, 5 and 7. Every group order is printed and written to
the file header so tools/orbit_mis can cross-check it by closure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from sympy.combinatorics import Permutation, PermutationGroup  # noqa: E402

from group.m24 import M24_ORDER, m24_generators, preserves_code  # noqa: E402


def reduce_generators(G, tries=50):
    """A small generating set (random elements) of G with the same order."""
    order = int(G.order())
    if len(G.generators) <= 4:
        return list(G.generators)
    best = None
    for k in (1, 2, 3, 4, 6):
        for _ in range(tries):
            gens = [G.random() for _ in range(k)]
            if int(PermutationGroup(gens).order()) == order:
                return gens
    return best if best is not None else list(G.generators)


def write_group(out_dir, name, G, note=""):
    gens = [list(p.array_form) + list(range(len(p.array_form), 24)) for p in reduce_generators(G)]
    gens = [g for g in gens if g != list(range(24))]
    for g in gens:
        assert preserves_code(g), name
    order = int(G.order())
    path = os.path.join(out_dir, f"{name}.txt")
    with open(path, "w") as f:
        f.write(f"# {name}: subgroup of M24 of order {order} ({len(gens)} generators){' — ' + note if note else ''}\n")
        f.write("# format: 24 images of coordinates 0..23, then the sign mask (0 = pure permutation)\n")
        for g in gens:
            f.write(" ".join(str(v) for v in g) + " 0\n")
    print(f"{name:14s} order {order:>10d} gens {len(gens)}  -> {path}")
    return order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "runs", "orbits", "subgroups"))
    ap.add_argument("--slow", action="store_true", help="also compute centralisers (minutes)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    alpha, gamma, delta = (Permutation(list(g)) for g in m24_generators())
    G = PermutationGroup([alpha, gamma, delta])
    assert int(G.order()) == M24_ORDER
    orders = {}
    for p in (2, 3, 5, 7, 11, 23):
        P = G.sylow_subgroup(p)
        orders[f"Syl{p}"] = write_group(args.out, f"Syl{p}", P, f"Sylow {p}-subgroup")
        if p == 2:
            orders["Syl2_derived"] = write_group(args.out, "Syl2_derived", P.derived_subgroup(), "derived subgroup of Syl2")
            orders["Syl2_centre"] = write_group(args.out, "Syl2_centre", P.center(), "centre of Syl2")
    H = G
    names = ["M23", "M22", "M21", "M20", "M19"]
    for k, nm in enumerate(names):
        H = H.stabilizer(23 - k)
        orders[nm] = write_group(args.out, nm, H, f"pointwise stabiliser of {k + 1} points")
    psl = PermutationGroup([alpha, gamma])
    assert int(psl.order()) == 6072
    orders["PSL_Syl2"] = write_group(args.out, "PSL_Syl2", psl.sylow_subgroup(2), "Sylow 2-subgroup of PSL(2,23)")
    orders["PSL_Syl3"] = write_group(args.out, "PSL_Syl3", psl.sylow_subgroup(3), "Sylow 3-subgroup of PSL(2,23)")
    if args.slow:
        # representatives of small element orders from random elements
        import random

        rnd = random.Random(1)
        reps = {}
        while len(reps) < 4:
            g = G.random(af=False)
            o = int(g.order())
            if o in (2, 3, 5, 7) and o not in reps:
                reps[o] = g
            rnd.random()
        for o, g in sorted(reps.items()):
            C = G.centralizer(g)
            orders[f"Cent{o}"] = write_group(args.out, f"Cent{o}", C, f"centraliser of an element of order {o}")
    print(f"RESULT ok=1 groups={len(orders)} out={args.out} seconds={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""B1 — independent checker for `data/scheme/clique_g.json` (omega(G) = 24).

Until now the check for this claim lived only as a pasted snippet in
docs/reports/B1.md, so a fresh clone had no committed way to re-verify it.
This script is that way.  It re-derives everything from committed files and
trusts nothing from the generating tool (`tools/clique_g`):

  1. the 24 stored indices really name vectors of `data/leech_min.i8`, and the
     `clique_vectors` copy stored in the JSON agrees with them entry by entry;
  2. their Gram matrix is exactly 16*(I + J) in integer arithmetic, so the 24
     vectors are pairwise at 60 degrees (inner product +16) and are a basis of
     R^24 -- hence omega(G) >= 24;
  3. the upper bound omega(G) <= 24 needs no computation: k pairwise-16
     norm-32 vectors have Gram 16*(I_k + J_k), which is positive definite and
     therefore of rank k, and the vectors live in R^24, so k <= 24.  We check
     the stored `omega` and `upper_bound_argument` fields are consistent with
     that and (independently) that 16*(I+J) is positive definite by an exact
     integer LDL^T / leading-principal-minor test on the stored size.
  4. the maximal-clique census through the base edge: the recorded sizes, once
     the two base-edge vertices are added back, are exactly {8, 12, 15, 17,
     23, 24}, the counts sum to 5028032232, and the maximum agrees with omega.

Not re-run here: the two Bron-Kerbosch enumerations that produced the census
counts (about 9 minutes via `build/release/tools/clique_g`); items 1-3 are
re-computed from scratch, item 4 is a consistency check of the stored census.

Run (from the repo root, after `tools/gen_leech data/`):
    .venv/bin/python python/tools/check_clique_g.py
    .venv/bin/python python/tools/check_clique_g.py --json data/scheme/clique_g.json \
        --vectors data/leech_min.i8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DIM = 24
NORM = 32          # our integer scaling <v, v> = 32
CONFLICT = 16      # inner product of a 60-degree pair
EXPECTED_SIZES = [8, 12, 15, 17, 23, 24]
EXPECTED_CENSUS_TOTAL = 5028032232


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise SystemExit(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def positive_definite_exact(M: np.ndarray) -> bool:
    """Exact rational leading-principal-minor test (Sylvester's criterion)."""
    n = M.shape[0]
    A = [[Fraction(int(x)) for x in row] for row in M]
    for i in range(n):
        if A[i][i] <= 0:
            return False
        piv = A[i][i]
        for j in range(i + 1, n):
            if A[j][i] == 0:
                continue
            f = A[j][i] / piv
            for k in range(i, n):
                A[j][k] -= f * A[i][k]
    return all(A[i][i] > 0 for i in range(n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=os.path.join(ROOT, "data", "scheme", "clique_g.json"))
    ap.add_argument("--vectors", default=os.path.join(ROOT, "data", "leech_min.i8"))
    args = ap.parse_args(argv)

    with open(args.json) as f:
        d = json.load(f)

    if not os.path.exists(args.vectors):
        fail(f"{args.vectors} missing — run `build/release/tools/gen_leech data/` first")
    C = np.fromfile(args.vectors, dtype=np.int8).reshape(-1, DIM).astype(np.int64)
    check(C.shape == (196560, DIM), f"unexpected vector file shape {C.shape}")

    # -- 1. the stored indices and the stored vectors agree ------------------
    K = list(d["clique"])
    check(len(K) == len(set(K)) == 24, f"clique field is not 24 distinct indices ({len(K)})")
    check(all(0 <= i < C.shape[0] for i in K), "clique index out of range")
    V = C[K]
    stored = np.array(d["clique_vectors"], dtype=np.int64)
    check(stored.shape == (24, DIM), f"clique_vectors shape {stored.shape}")
    check(np.array_equal(V, stored), "clique_vectors disagree with leech_min.i8 at those indices")
    check(np.all((V * V).sum(axis=1) == NORM), "a clique vector does not have norm 32")

    # -- 2. Gram = 16 (I + J) ------------------------------------------------
    G = V @ V.T
    target = CONFLICT * (np.eye(24, dtype=np.int64) + 1)
    check(np.array_equal(G, target), "Gram is not 16 (I + J)")
    rank = np.linalg.matrix_rank(G.astype(np.float64))
    check(rank == 24, f"Gram rank {rank} != 24")

    # -- 3. the <= 24 argument ----------------------------------------------
    check(int(d["omega"]) == 24, f"stored omega = {d['omega']} != 24")
    check(positive_definite_exact(target), "16 (I + J) is not positive definite (exact test)")
    # 16(I_k + J_k) is PD for every k, so k pairwise-16 vectors are independent
    # in R^24: k <= 24.  Sanity-check the same at k = 25 to show it is the
    # ambient dimension, not the matrix, that caps k.
    check(positive_definite_exact(CONFLICT * (np.eye(25, dtype=np.int64) + 1)),
          "16 (I_25 + J_25) unexpectedly not PD")

    # -- 4. the census through the base edge --------------------------------
    edge = d["base_edge"]
    check(len(edge) == 2 and C[edge[0]] @ C[edge[1]] == CONFLICT,
          "base_edge is not an edge of the conflict graph")
    census = {int(k) + 2: int(v) for k, v in d["maximal_cliques_by_size"].items()}
    sizes = sorted(census)
    check(sizes == EXPECTED_SIZES, f"maximal clique sizes {sizes} != {EXPECTED_SIZES}")
    total = sum(census.values())
    check(total == EXPECTED_CENSUS_TOTAL,
          f"census total {total} != {EXPECTED_CENSUS_TOTAL}")
    check(max(sizes) == int(d["omega"]), "largest census clique disagrees with omega")
    check(bool(d.get("census_complete")), "census_complete is not true")
    check(int(d["max_clique_in_common_neighbourhood"]) + 2 == 24,
          "max clique in N(edge) + 2 != 24")

    print(f"clique     : 24 indices of {os.path.relpath(args.vectors, ROOT)}, "
          f"norms all {NORM}, Gram = 16(I+J), rank 24")
    print(f"upper bound: {d['upper_bound_argument']}")
    print(f"census     : through edge {tuple(edge)} "
          f"({d['common_neighbours']} common neighbours, "
          f"{d['common_neighbourhood_regular_degree']}-regular); "
          + ", ".join(f"size {s}: {census[s]}" for s in sizes))
    print(f"RESULT ok=1 omega=24 clique_gram=16(I+J) rank=24 "
          f"maximal_sizes={','.join(map(str, sizes))} census_total={total} "
          f"census_complete=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

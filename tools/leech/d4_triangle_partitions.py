#!/usr/bin/env python3
"""Does the 24-cell (the D_4 root system) admit a partition into 8 zero-sum triples?

Everything below is decided in exact arithmetic: the vectors are integer tuples, all inner
products are Python ints, and every cosine is a ``fractions.Fraction``. No floating point is
consulted for any check (the only float in the file is the solver tolerance inside
``python/template/triangles.py``, which is used purely as a *cross-check* of results that this
script has already established exactly).

Template recap (docs/reports/05-lifting-template-and-families.md, docs/note/kissing27.tex):
a kissing configuration T in R^d is split into groups T_i with pairwise cos <= -1/2 inside each
group; the lifted family has weight w = sum_i (|T_i| - 1), and Proposition 5 (prop:weight) gives
w <= floor(2K(d)/3). For d = 3, T = cuboctahedron, K = 12, four triangles, w = 8.  For d = 4, T = the 24-cell =
D_4 roots, K = 24, floor(2*24/3) = 16: the question settled here is whether eight disjoint
equilateral triangles (equivalently eight zero-sum triples) exist, and how many such
partitions there are.

Run:  PYTHONPATH=python .venv/bin/python tools/leech/d4_triangle_partitions.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import itertools
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

FAILURES: list[str] = []


def result(name: str, ok: bool | None = None, **kv) -> None:
    """Print one ``RESULT <name> key=value ...`` line; ok=False records a failure."""
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    tag = "" if ok is None else f" ok={int(bool(ok))}"
    print(f"RESULT {name}{tag}" + (f" {parts}" if parts else ""), flush=True)
    if ok is False:
        FAILURES.append(name)


def check(name: str, ok: bool, **kv) -> bool:
    result(name, ok, **kv)
    return ok


# ---------------------------------------------------------------------------
# 1. the 24 roots of D_4
# ---------------------------------------------------------------------------
def d4_roots() -> list[tuple[int, ...]]:
    """All (+-1, +-1, 0, 0) and permutations: 24 integer vectors of squared norm 2."""
    rows = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0, 0, 0, 0]
                v[i], v[j] = si, sj
                rows.append(tuple(v))
    return sorted(rows)


def ip(a, b) -> int:
    return sum(x * y for x, y in zip(a, b))


def cos_frac(a, b, na: int, nb: int) -> Fraction:
    """Exact cosine for two integer vectors of squared norms na, nb -- here na = nb = 2, so the
    cosine <a,b>/2 is rational and no square root is ever needed."""
    assert na == nb == 2
    return Fraction(ip(a, b), 2)


def step1(V):
    n = len(V)
    ok = check("step1.count", n == 24, n=n, expected=24)
    ok &= check("step1.distinct", len(set(V)) == 24, distinct=len(set(V)))
    norms = {ip(v, v) for v in V}
    ok &= check("step1.norm2", norms == {2}, norms=sorted(norms))
    ips = sorted({ip(V[i], V[j]) for i in range(n) for j in range(n)})
    ok &= check("step1.inner_products", set(ips) <= {-2, -1, 0, 1, 2}, values=ips)
    coss = sorted({cos_frac(V[i], V[j], 2, 2) for i in range(n) for j in range(n)})
    want = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    ok &= check("step1.cosines", coss == want, values="{" + ",".join(str(c) for c in coss) + "}")
    tot = tuple(sum(v[k] for v in V) for k in range(4))
    ok &= check("step1.sums_to_zero", tot == (0, 0, 0, 0), sum=str(tot).replace(" ", ""))
    # degree profile: how many roots sit at each inner product from a fixed root
    prof = {}
    for p in (-2, -1, 0, 1, 2):
        counts = {sum(1 for b in V if ip(a, b) == p) for a in V}
        prof[p] = sorted(counts)
    result("step1.degree_profile", None,
           **{f"ip{p}": (prof[p][0] if len(prof[p]) == 1 else prof[p]) for p in (-2, -1, 0, 1, 2)})
    return ok


# ---------------------------------------------------------------------------
# 2. triples at pairwise cosine -1/2, and the norm identity
# ---------------------------------------------------------------------------
def step2(V):
    n = len(V)
    neg_half = [[cos_frac(V[i], V[j], 2, 2) == Fraction(-1, 2) for j in range(n)] for i in range(n)]
    tris, zero_sum, nonzero_sum = [], 0, 0
    for a, b, c in itertools.combinations(range(n), 3):
        if neg_half[a][b] and neg_half[a][c] and neg_half[b][c]:
            tris.append((a, b, c))
            s = tuple(V[a][k] + V[b][k] + V[c][k] for k in range(4))
            if s == (0, 0, 0, 0):
                zero_sum += 1
            else:
                nonzero_sum += 1
    ok = check("step2.triples_pairwise_cos_neg_half", len(tris) == 32, count=len(tris))
    ok &= check("step2.all_zero_sum", nonzero_sum == 0, zero_sum=zero_sum, nonzero_sum=nonzero_sum)

    # The identity |a+b+c|^2 = 3*2 + 2*(sum of pairwise inner products), verified term by term
    # on every one of the 32 triples, in integers.
    bad = 0
    for a, b, c in tris:
        s = tuple(V[a][k] + V[b][k] + V[c][k] for k in range(4))
        lhs = ip(s, s)
        pw = ip(V[a], V[b]) + ip(V[a], V[c]) + ip(V[b], V[c])
        rhs = 3 * 2 + 2 * pw
        if lhs != rhs or pw != -3 or lhs != 0:
            bad += 1
    ok &= check("step2.norm_identity", bad == 0, checked=len(tris), violations=bad,
                identity="|a+b+c|^2 = 3*2 + 2*(-1-1-1) = 6 - 6 = 0")

    # Converse direction: among ALL C(24,3) triples, "pairwise ip = -1" <=> "zero-sum".
    zs_all = [t for t in itertools.combinations(range(n), 3)
              if tuple(V[t[0]][k] + V[t[1]][k] + V[t[2]][k] for k in range(4)) == (0, 0, 0, 0)]
    ok &= check("step2.coincide", set(zs_all) == set(tris),
                zero_sum_triples=len(zs_all), cos_neg_half_triples=len(tris),
                equal=int(set(zs_all) == set(tris)))
    result("step2.why", None,
           forward="pairwise ip=-1 => |a+b+c|^2 = 6 + 2(-3) = 0 => a+b+c=0",
           backward="a+b+c=0 => 0 = 6 + 2*sum_pw => sum_pw=-3; each ip>=-2 and ip=-2 forces "
                    "b=-a, then c=0 impossible, so ip in {-1,0,1,2}; three ips summing to -3 with "
                    "each >=-1 forces all three = -1")
    # make the "backward" argument itself a checked fact, exhaustively
    bad2 = 0
    for a, b, c in zs_all:
        pw = [ip(V[a], V[b]), ip(V[a], V[c]), ip(V[b], V[c])]
        if sorted(pw) != [-1, -1, -1]:
            bad2 += 1
    ok &= check("step2.backward_exhaustive", bad2 == 0, zero_sum_triples=len(zs_all), violations=bad2)
    return ok, tris


# ---------------------------------------------------------------------------
# 3. no four roots pairwise at cos <= -1/2
# ---------------------------------------------------------------------------
def step3(V):
    n = len(V)
    le = [[cos_frac(V[i], V[j], 2, 2) <= Fraction(-1, 2) for j in range(n)] for i in range(n)]
    quads = [q for q in itertools.combinations(range(n), 4)
             if all(le[x][y] for x, y in itertools.combinations(q, 2))]
    ok = check("step3.no_four_pairwise_le_neg_half", len(quads) == 0, quadruples=len(quads),
               searched=len(list(itertools.combinations(range(n), 4))))
    # the a priori proof, restated with the exact bound for norm-2 vectors:
    # |a+b+c+d|^2 = 4*2 + 2*(6 ips each <= -1) <= 8 - 12 = -4 < 0.
    result("step3.proof", None, identity="|a+b+c+d|^2 = 4*2 + 2*sum_of_6_ips <= 8 - 12 = -4 < 0",
           conclusion="group size <= 3")
    # largest clique in the cos<=-1/2 graph, exhaustively
    best = 0
    for k in (1, 2, 3, 4):
        found = any(all(le[x][y] for x, y in itertools.combinations(q, 2))
                    for q in itertools.combinations(range(n), k))
        if found:
            best = k
    ok &= check("step3.max_group_size", best == 3, max_clique_cos_le_neg_half=best)
    return ok


# ---------------------------------------------------------------------------
# 4. exact cover: maximum packing, perfect partitions, and their count
# ---------------------------------------------------------------------------
def max_triangle_packing_exact(n: int, tris) -> tuple[int, list]:
    """Exhaustive branch-and-bound for the maximum number of pairwise disjoint triangles."""
    bits = [(1 << a) | (1 << b) | (1 << c) for a, b, c in tris]
    by_point: list[list[int]] = [[] for _ in range(n)]
    for k, (a, b, c) in enumerate(tris):
        by_point[a].append(k)
        by_point[b].append(k)
        by_point[c].append(k)
    full = (1 << n) - 1
    best = [0, []]

    def dfs(used: int, chosen: list[int]) -> None:
        free = full & ~used
        nfree = bin(free).count("1")
        if len(chosen) + nfree // 3 <= best[0]:
            return
        if free == 0:
            if len(chosen) > best[0]:
                best[0], best[1] = len(chosen), list(chosen)
            return
        p = (free & -free).bit_length() - 1
        # branch on the lowest free point: either it is in a chosen triangle, or it is uncovered
        for k in by_point[p]:
            if bits[k] & used:
                continue
            chosen.append(k)
            dfs(used | bits[k], chosen)
            chosen.pop()
        if len(chosen) > best[0]:
            best[0], best[1] = len(chosen), list(chosen)
        dfs(used | (1 << p), chosen)  # leave p uncovered

    dfs(0, [])
    return best[0], [tris[k] for k in best[1]]


def count_perfect_partitions(n: int, tris) -> tuple[int, list]:
    """Exact cover: count ALL partitions of the n points into disjoint triangles (unordered sets
    of triangles). Returns (count, one witness)."""
    bits = [(1 << a) | (1 << b) | (1 << c) for a, b, c in tris]
    by_point: list[list[int]] = [[] for _ in range(n)]
    for k, (a, b, c) in enumerate(tris):
        by_point[a].append(k)
        by_point[b].append(k)
        by_point[c].append(k)
    full = (1 << n) - 1
    total = [0]
    witness: list[list[int]] = []

    def dfs(used: int, chosen: list[int]) -> None:
        if used == full:
            total[0] += 1
            if not witness:
                witness.append(list(chosen))
            return
        free = full & ~used
        p = (free & -free).bit_length() - 1  # lowest uncovered point must be covered now
        for k in by_point[p]:
            if bits[k] & used:
                continue
            chosen.append(k)
            dfs(used | bits[k], chosen)
            chosen.pop()

    dfs(0, [])
    return total[0], ([tris[k] for k in witness[0]] if witness else [])


def step4(V, tris):
    n = len(V)
    pack, packing = max_triangle_packing_exact(n, tris)
    ok = check("step4.max_disjoint_triangles", pack == 8, value=pack, upper_bound_floor_K_over_3=n // 3,
               method="exhaustive branch-and-bound (exact)")
    npart, witness = count_perfect_partitions(n, tris)
    ok &= check("step4.perfect_partition_exists", npart > 0, partitions=npart)
    ok &= check("step4.perfect_partition_count", npart == 40, count=npart,
                note="unordered sets of 8 triangles covering all 24 roots")
    # validate the witness completely and independently
    cov = sorted(i for t in witness for i in t)
    good = (len(witness) == 8 and cov == list(range(24)))
    for t in witness:
        s = tuple(sum(V[i][k] for i in t) for k in range(4))
        pw = [ip(V[t[0]], V[t[1]]), ip(V[t[0]], V[t[2]]), ip(V[t[1]], V[t[2]])]
        good &= (s == (0, 0, 0, 0)) and sorted(pw) == [-1, -1, -1]
    ok &= check("step4.witness_valid", good, groups=len(witness), covered=len(cov),
                all_zero_sum=1, all_pairwise_cos=str(Fraction(-1, 2)))
    for j, t in enumerate(witness, 1):
        result(f"step4.partition.T{j:02d}", None, idx=",".join(str(i) for i in t),
               vectors="|".join("(" + ",".join(str(x) for x in V[i]) + ")" for i in t),
               sum="(0,0,0,0)")

    # independent recount by raw brute force over all C(32,8) subsets of triangles
    bits = [sum(1 << i for i in t) for t in tris]
    full = (1 << n) - 1
    brute = 0
    for comb in itertools.combinations(range(len(tris)), 8):
        m = 0
        for k in comb:
            if m & bits[k]:
                break
            m |= bits[k]
        else:
            if m == full:
                brute += 1
    ok &= check("step4.brute_force_recount", brute == npart, brute_force=brute, dfs=npart,
                subsets_examined=len(list(itertools.combinations(range(len(tris)), 8))))

    # orbit decomposition of the 40 partitions under the 384 signed permutations of the model
    # (a subgroup of Aut(D_4); triality is not represented in these coordinates)
    pos = {v: i for i, v in enumerate(V)}
    auts = []
    for p in itertools.permutations(range(4)):
        for s in itertools.product((1, -1), repeat=4):
            auts.append([pos[tuple(s[k] * V[i][p[k]] for k in range(4))] for i in range(n)])
    allparts = _all_perfect_partitions(n, tris)
    canon = [frozenset(frozenset(t) for t in P) for P in allparts]
    seen, orbits = set(), []
    for x in canon:
        if x in seen:
            continue
        orb = {frozenset(frozenset(a[i] for i in t) for t in x) for a in auts}
        orbits.append(len(orb))
        seen |= orb
    ok &= check("step4.orbits", sum(orbits) == npart, group_order=len(auts),
                orbit_sizes=",".join(str(o) for o in sorted(orbits)), total=sum(orbits))
    return ok, packing, witness, npart


def _all_perfect_partitions(n: int, tris) -> list[list[tuple[int, int, int]]]:
    bits = [(1 << a) | (1 << b) | (1 << c) for a, b, c in tris]
    by_point: list[list[int]] = [[] for _ in range(n)]
    for k, (a, b, c) in enumerate(tris):
        by_point[a].append(k)
        by_point[b].append(k)
        by_point[c].append(k)
    full = (1 << n) - 1
    out: list[list[tuple[int, int, int]]] = []

    def dfs(used: int, chosen: list[int]) -> None:
        if used == full:
            out.append([tris[k] for k in chosen])
            return
        free = full & ~used
        p = (free & -free).bit_length() - 1
        for k in by_point[p]:
            if bits[k] & used:
                continue
            chosen.append(k)
            dfs(used | bits[k], chosen)
            chosen.pop()

    dfs(0, [])
    return out


# ---------------------------------------------------------------------------
# 5. maximum weight w = sum_i (|T_i| - 1)
# ---------------------------------------------------------------------------
def max_weight_exact(V, tris):
    """Exhaustive branch-and-bound for max sum (|T_i| - 1) over disjoint groups: triangles
    (weight 2), admissible pairs with cos <= -1/2 (weight 1), singletons (weight 0)."""
    n = len(V)
    le = [[cos_frac(V[i], V[j], 2, 2) <= Fraction(-1, 2) for j in range(n)] for i in range(n)]
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2) if le[i][j]]
    blocks = [(t, 2) for t in tris] + [(p, 1) for p in pairs]
    bits = [(sum(1 << i for i in b), w) for b, w in blocks]
    by_point: list[list[int]] = [[] for _ in range(n)]
    for k, (b, _w) in enumerate(blocks):
        for i in b:
            by_point[i].append(k)
    full = (1 << n) - 1
    best = [0, []]

    def dfs(used: int, wt: int, chosen: list[int]) -> None:
        free = full & ~used
        nfree = bin(free).count("1")
        if wt + 2 * (nfree // 3) + (1 if nfree % 3 == 2 else 0) <= best[0]:
            return  # Proposition 5 counting bound applied to the free points
        if free == 0:
            if wt > best[0]:
                best[0], best[1] = wt, list(chosen)
            return
        p = (free & -free).bit_length() - 1
        for k in by_point[p]:
            bm, w = bits[k]
            if bm & used:
                continue
            chosen.append(k)
            dfs(used | bm, wt + w, chosen)
            chosen.pop()
        if wt > best[0]:
            best[0], best[1] = wt, list(chosen)
        dfs(used | (1 << p), wt, chosen)  # p is a singleton group (weight 0)

    dfs(0, 0, [])
    return best[0], [blocks[k][0] for k in best[1]], len(pairs)


def step5(V, tris):
    w, groups, npairs = max_weight_exact(V, tris)
    bound = (2 * 24) // 3
    ok = check("step5.max_weight", w == 16, weight=w, prop5_bound_floor_2K_over_3=bound,
               attained=int(w == bound), method="exhaustive branch-and-bound (exact)")
    ok &= check("step5.optimum_is_all_triangles", all(len(g) == 3 for g in groups) and len(groups) == 8,
                triangles=sum(1 for g in groups if len(g) == 3),
                pairs=sum(1 for g in groups if len(g) == 2),
                admissible_pairs_total=npairs)
    result("step5.count", None, dim=28, formula="24 + 196560 + 16*|S|",
           with_S_496=24 + 196560 + 16 * 496)
    return ok


# ---------------------------------------------------------------------------
# 6. cross-check against python/template/triangles.py and kiss_ref.small_kissing
# ---------------------------------------------------------------------------
def step6(V, tris, npart):
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import numpy as np
    from kiss_ref import small_kissing as sk
    from template import triangles as tri_mod

    W, norm2 = sk.config(4)
    ok = check("step6.config_matches", set(map(tuple, W.tolist())) == set(V) and norm2 == 2,
               n=len(W), norm2=norm2, same_set=int(set(map(tuple, W.tolist())) == set(V)))

    # index translation from our lexicographic ordering to the repo's config(4) ordering
    idx = {tuple(r): i for i, r in enumerate(W.tolist())}
    ours_in_theirs = {tuple(sorted(idx[V[i]] for i in t)) for t in tris}

    theirs = {tuple(sorted(t)) for t in sk.triangles_of(W)}
    ok &= check("step6.triangles_of_agree", theirs == ours_in_theirs,
                repo_triangles=len(theirs), ours=len(ours_in_theirs), equal=int(theirs == ours_in_theirs))

    pack = tri_mod.max_triangle_packing(W)
    ok &= check("step6.max_triangle_packing", pack["value"] == 8 and pack["proven"],
                value=pack["value"], proven=int(bool(pack["proven"])), leftover=len(pack["leftover"]),
                n_triangles=pack["n_triangles_total"])

    part = tri_mod.max_weight_partition(W)
    ok &= check("step6.max_weight_partition", part["value"] == 16 and part["proven"],
                value=part["value"], proven=int(bool(part["proven"])),
                triangles=len(part["triangles"]), pairs=len(part["pairs"]),
                leftover=len(part["leftover"]))

    wub = tri_mod.weight_upper_bound(24)
    ok &= check("step6.weight_upper_bound", wub == 16, value=wub)

    zs = tri_mod.zero_sum_bound(W)
    ok &= check("step6.zero_sum_bound", zs["bound"] == 8 and zs["sums_to_zero"],
                bound=zs["bound"], sums_to_zero=int(zs["sums_to_zero"]), reason=zs["reason"].split(":")[0])

    tp = sk.triangle_partition(4, 8, require_pairs=False)
    ok &= check("step6.sk_triangle_partition_8", tp is not None and len(tp["triangles"]) == 8,
                found=int(tp is not None), triangles=(len(tp["triangles"]) if tp else 0),
                leftover=(len(tp["leftover"]) if tp else -1), nodes=(tp["nodes"] if tp else -1))
    tp9 = sk.triangle_partition(4, 9, require_pairs=False)
    ok &= check("step6.sk_triangle_partition_9_impossible", tp9 is None,
                found=int(tp9 is not None), status=sk.triangle_partition.last_status,
                note="9 disjoint triangles need 27 > 24 points")

    # the committed dim-28 record family: does it already use 8 triangles / weight 16?
    try:
        rec = tri_mod.record_partition(4)
        ok &= check("step6.record_dim28", rec["weight"] == 16 and rec["n_triangles"] == 8,
                    triangles=rec["n_triangles"], pairs=rec["n_pairs"], weight=rec["weight"],
                    count=rec["count"], extra=rec["extra"], sizes=sorted(set(rec["sizes"])))
        rec_set = {tuple(sorted(g)) for g in rec["groups"]}
        ok &= check("step6.record_partition_is_one_of_ours", rec_set <= {tuple(sorted(t)) for t in theirs} and
                    sorted(i for g in rec_set for i in g) == list(range(24)),
                    perfect=int(sorted(i for g in rec_set for i in g) == list(range(24))),
                    among_the_counted=npart)
    except Exception as e:  # pragma: no cover -- only if data/families/dim28 is absent
        result("step6.record_dim28", False, error=type(e).__name__, msg=str(e)[:80])
        FAILURES.append("step6.record_dim28")
        ok = False
    del np
    return ok


# ---------------------------------------------------------------------------
# 7. Proposition 6 (prop:zerosum)
# ---------------------------------------------------------------------------
def step7(V):
    n = len(V)
    tot = tuple(sum(v[k] for v in V) for k in range(4))
    sums_to_zero = tot == (0, 0, 0, 0)
    r = n % 3
    forbids = sums_to_zero and r == 1
    ok = check("step7.prop6_applies", sums_to_zero, sums_to_zero=int(sums_to_zero), K=n, K_mod_3=r)
    ok &= check("step7.prop6_forbids_perfect_partition", forbids is False,
                forbids=int(forbids), K_mod_3=r,
                reason="Prop 6 forbids only when |T| = 1 mod 3; 24 = 0 mod 3, so it PERMITS "
                       "a perfect partition (and its first clause forces every triangle to be zero-sum)")
    # contrast with d = 5 (D_5, K = 40 = 1 mod 3), where Prop 6 does bite
    result("step7.contrast_d5", None, K=40, K_mod_3=40 % 3, triangles_at_most=40 // 3 - 1,
           note="Prop 6 removes the 13th triangle for D_5")
    return ok


# ---------------------------------------------------------------------------
def main() -> int:
    V = d4_roots()
    ok = step1(V)
    ok2, tris = step2(V)
    ok &= ok2
    ok &= step3(V)
    ok3, packing, witness, npart = step4(V, tris)
    ok &= ok3
    ok &= step5(V, tris)
    ok &= step6(V, tris, npart)
    ok &= step7(V)

    result("answer", ok,
           question="does the 24-cell admit a partition into 8 zero-sum triples",
           answer="YES", perfect_partitions=npart, max_weight=16,
           prop5_bound=16, dim28_count_with_S496=24 + 196560 + 16 * 496)
    if FAILURES:
        print(f"RESULT overall ok=0 failures={len(FAILURES)} names={','.join(FAILURES)}")
        return 1
    print("RESULT overall ok=1 failures=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

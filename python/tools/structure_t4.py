"""A4b — the 80 tightness-4 vertices as a channel/rook object, the (80,80)
plateau atoms in channel terms, S ∪ T4 = 576, census-wide certification, and
the A4-type decomposition of B1's 24-clique.

Deterministic, exact integer arithmetic for every claim (float64 GEMMs are
used only for inner products of integer vectors bounded far below 2^53, hence
exact).  One `RESULT section=...` line per verified fact; interpretation and
VERIFIED/CONJECTURED labels live in docs/reports/A4b.md.

Inputs
    data/S496.txt                            the record set (sqrt8 scaling, norm 32)
    runs/orbits/stab496/co0_stabiliser_elements.txt   Stab_Co0(S) (re-verified here)
    data/plateau_atoms_496.json              the six plateau atoms (T3.2b/A1; the
                                             original run artefact
                                             runs/ls_search/plateau_atoms_496.json
                                             is used as a fallback)
    runs/a4/channel_table.json               A4's 15-channel usage table (cross-checked,
                                             then recomputed from scratch here)
    data/scheme/clique_g.json                B1's 24-clique of the conflict graph
    runs/gpu_mis/*/sets/S_*.txt              the GPU census set dumps (--census)

Sections
    U0  setup: S, V, stabiliser re-verified, Turyn blocks, Lambda-basis
    U1  the 80 tightness-4 vertices: duad census, 20 orbits, channel/cell table
    U2  the six plateau atoms in channel terms (what each move does to the
        45-cell rook model); the frame swap as a channel exchange
    U3  S ∪ T4 = 576 = 24^2: Gram, tight-frame identity, channel table of the
        144 orbits, conflict components in cell terms
    U4  (--census) certification of the 4160-census: cluster the GPU dumps
        into seed families, certify one representative per family A1-style
    U5  B1's 24-clique decomposed on the Turyn frame (monads/duads/triads)

Run:  .venv/bin/python python/tools/structure_t4.py [--census] [--standins N]
Writes runs/a4b/summary.json plus per-section artefacts in runs/a4b/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiss_ref.leech import leech_min_vectors  # noqa: E402
from structure_496 import (  # noqa: E402
    apair_key, build_basis, coords_of, gf2_rank, gf2_span, lattice_hnf,
    load_S, load_stab_elements, smith_invariants,
)

RUNS = os.path.join(ROOT, "runs", "a4b")
os.makedirs(RUNS, exist_ok=True)

SUMMARY: dict = {}
PW2 = (1 << np.arange(24, dtype=np.int64))


def result(section: str, **kv):
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"RESULT section={section} {parts}")
    SUMMARY.setdefault(section, []).append(
        {k: (v if isinstance(v, (int, str, bool)) else str(v)) for k, v in kv.items()})


# --------------------------------------------------------------------- setup

def setup():
    """S, V, the three trace+8 involutions (same deterministic order as A4's
    S1: t1 fixes 208 of S, t2/t3 fix 144, tie-broken by matrix bytes), and the
    exact Lambda basis.  Every stabiliser property is re-verified."""
    t0 = time.time()
    S = load_S()
    V = np.asarray(leech_min_vectors(), dtype=np.int64)
    vindex = {bytes(r): i for i, r in enumerate(V.astype(np.int8))}
    sset = {bytes(r.astype(np.int8)) for r in S}

    mats = load_stab_elements()
    Vsorted = np.unique(V, axis=0)
    for M in mats:
        assert np.array_equal(M @ M.T, 64 * np.eye(24, dtype=np.int64))
        img = V @ M.T
        assert np.all(img % 8 == 0)
        assert np.array_equal(np.unique(img // 8, axis=0), Vsorted)
        simg = S @ M.T
        assert np.all(simg % 8 == 0)
        assert all(bytes(r.astype(np.int8)) in sset for r in simg // 8)
    t8 = [M for M in mats if np.trace(M) == 64]
    assert len(t8) == 3

    def fixed_count(M):
        return int(np.sum((S @ M.T) // 8 == S) // 24) if False else \
            int(np.sum(np.all((S @ M.T) // 8 == S, axis=1)))

    t8.sort(key=lambda M: (-fixed_count(M), M.tobytes()))
    assert [fixed_count(M) for M in t8] == [208, 144, 144]
    B, IB8 = build_basis(V)
    result("U0", stab_reverified=True, fixed_on_S="208,144,144",
           seconds=round(time.time() - t0, 1))
    return S, V, vindex, sset, t8, B, IB8


def block_patterns(X: np.ndarray, ts) -> np.ndarray:
    """(n1, n2, n3) block-norm patterns, n_i = |component in B_i|^2."""
    Xf = X.astype(np.float64)
    pats = []
    for M in ts:
        img = (Xf @ M.T.astype(np.float64)) / 8.0
        d = np.einsum("ij,ij->i", Xf, img).astype(np.int64)
        assert np.all((32 - d) % 2 == 0)
        pats.append((32 - d) // 2)
    tup = np.stack(pats, axis=1)
    assert np.all(tup.sum(axis=1) == 32) and np.all(tup >= 0)
    return tup


def duad_orbit_data(v: np.ndarray, ts, IB8):
    """For a duad v: (type, orbit {±v, ±y}, channel mask, rook rows).

    type  = the unique i with t_i v = v (the support pair is the other two
            blocks); y = t_j v for either j != i (both give the same orbit).
    sums  = v+y, v-y: one norm-64 vector in each support block; both are
            congruent mod 2Lambda; their common class is the channel mask.
    rows  = {block: apair_key(sum vector in that block)} — the rook row/col.
    """
    imgs = [(v @ M.T) // 8 for M in ts]
    fixers = [i for i in range(3) if np.array_equal(imgs[i], v)]
    assert len(fixers) == 1, "not a duad"
    t = fixers[0]
    j = [x for x in range(3) if x != t][0]
    y = imgs[j]
    wp, wm = v + y, v - y
    assert int(wp @ wp) == 64 and int(wm @ wm) == 64
    cw = coords_of(wp.reshape(1, -1), IB8)[0] % 2
    cw2 = coords_of(wm.reshape(1, -1), IB8)[0] % 2
    assert np.array_equal(cw, cw2)
    mask = int((cw.astype(np.int64) * PW2).sum())
    rows = {}
    for w in (wp, wm):
        negs = [i for i in range(3) if np.array_equal((w @ ts[i].T) // 8, -w)]
        assert len(negs) == 1
        rows[negs[0]] = apair_key(w)
    assert set(rows) == {x for x in range(3) if x != t}
    orbit = {apair_key(v), apair_key(y)}
    return t, orbit, mask, rows


def orbit_decompose(X: np.ndarray, ts, IB8):
    """Decompose a negation-closed duad set into stabiliser orbits {±x, ±y}.
    Returns list of (type, orbit_keys, mask, rows) — one entry per orbit —
    and asserts the set is exactly the union of its orbits (closure)."""
    keys = {apair_key(v) for v in X}
    assert 2 * len(keys) == len(X), "set not negation-closed"
    seen = set()
    orbits = []
    for v in X:
        k = apair_key(v)
        if k in seen:
            continue
        t, orb, mask, rows = duad_orbit_data(v, ts, IB8)
        assert orb <= keys, "orbit leaves the set: not stabiliser-closed"
        seen |= orb
        orbits.append((t, orb, mask, rows))
    assert len(seen) == len(keys)
    return orbits


def tightness(V: np.ndarray, S: np.ndarray) -> np.ndarray:
    """tight[v] = #{s in S : <v,s> = 16}, exact (float64 GEMM on ints < 2^9)."""
    Vf = V.astype(np.float64)
    Sf = S.astype(np.float64)
    out = np.zeros(len(V), dtype=np.int64)
    step = 16384
    for lo in range(0, len(V), step):
        D = Vf[lo:lo + step] @ Sf.T
        out[lo:lo + step] = (D == 16).sum(axis=1)
    return out


CHANNEL_NAMES = {}   # mask -> short name c0..c14, filled in U1


def chname(mask: int) -> str:
    return CHANNEL_NAMES.get(mask, f"m{mask}")


# --------------------------------------------------------------------- U1

def u1_t4(S, V, sset, ts, IB8):
    t0 = time.time()
    tight = tightness(V, S)
    smask = np.fromiter((bytes(r) in sset for r in V.astype(np.int8)),
                        dtype=bool, count=len(V))
    t4 = V[(tight == 4) & ~smask]
    assert len(t4) == 80
    # mutual non-adjacency + Gram histogram (T3.1 recheck)
    G = t4 @ t4.T
    off = G[~np.eye(80, dtype=bool)]
    ghist = Counter(int(x) for x in off)
    assert dict(ghist) == {-32: 80, -8: 1280, 0: 3680, 8: 1280}, dict(ghist)
    assert 16 not in ghist and -16 not in ghist
    pats = block_patterns(t4, ts)
    pu, pc = np.unique(pats, axis=0, return_counts=True)
    phist = {tuple(int(x) for x in u): int(c) for u, c in zip(pu, pc)}
    result("U1a", t4_count=80, antipodal_pairs=40, mutually_nonadjacent=True,
           gram_pairs=str({k: v // 2 for k, v in sorted(ghist.items())}),
           duad_patterns=str({str(k): v for k, v in phist.items()}))

    orbits = orbit_decompose(t4, ts, IB8)
    assert len(orbits) == 20
    tcnt = Counter(o[0] for o in orbits)
    # channel/cell table
    cells = Counter((t, m) for (t, orb, m, rows) in orbits)
    per_channel = Counter(m for (t, orb, m, rows) in orbits)
    result("U1b", orbits=20, orbit_sizes="all 4",
           orbits_by_pair=str(dict(sorted(tcnt.items()))),
           distinct_channels=len(per_channel),
           per_channel_orbits=str({m: c for m, c in sorted(per_channel.items())}),
           cells=str({f"pair{t}/{m}": c for (t, m), c in sorted(cells.items())}))
    result("U1", seconds=round(time.time() - t0, 1))
    return t4, orbits, tight, smask


# --------------------------------------------------------------------- U2

def plateau_atoms_path() -> str:
    """The six plateau atoms: the committed copy first, the original T3.2b run
    artefact (gitignored `runs/`) as a fallback."""
    p = os.path.join(ROOT, "data", "plateau_atoms_496.json")
    fallback = os.path.join(ROOT, "runs", "ls_search", "plateau_atoms_496.json")
    if not os.path.exists(p) and os.path.exists(fallback):
        return fallback
    return p


def u2_atoms(S, V, sset, ts, IB8, t4, t4_orbits, s_orbits):
    t0 = time.time()
    with open(plateau_atoms_path()) as f:
        atoms = json.load(f)["atoms"]
    assert [a["size"] for a in atoms] == [12, 12, 12, 12, 80, 80]
    t4keys = {k for (_, orb, _, _) in t4_orbits for k in orb}
    s_by_key = {}
    for oi, (t, orb, m, rows) in enumerate(s_orbits):
        for k in orb:
            s_by_key[k] = oi

    def cellstr(orbs):
        c = Counter((t, m) for (t, _, m, _) in orbs)
        return str({f"pair{t}/{chname(m)}": v for (t, m), v in sorted(c.items())})

    atom_data = []
    for ai, atom in enumerate(atoms):
        rem = V[atom["remove_vertices"]]
        add = V[atom["add_vertices"]]
        assert all(bytes(r.astype(np.int8)) in sset for r in rem)
        assert not any(bytes(r.astype(np.int8)) in sset for r in add)
        ro = orbit_decompose(rem, ts, IB8)
        ao = orbit_decompose(add, ts, IB8)
        n_t4 = sum(1 for (_, orb, _, _) in ao if orb <= t4keys)
        # physical row bookkeeping (channel, block, row): the rows the removal
        # vacates vs the rows the adds occupy
        rows_rem = sorted((m, b, r) for (t, _, m, rows) in ro for b, r in rows.items())
        rows_add = sorted((m, b, r) for (t, _, m, rows) in ao for b, r in rows.items())
        # alternating cycle structure: bipartite graph removed-orbits vs
        # added-orbits, edge = shared physical row; 2-regular => cycles
        nodes = [("R", i) for i in range(len(ro))] + [("A", i) for i in range(len(ao))]
        rowmap: dict = {}
        for i, (t, _, m, rows) in enumerate(ro):
            for b, r in rows.items():
                rowmap.setdefault((m, b, r), []).append(("R", i))
        for i, (t, _, m, rows) in enumerate(ao):
            for b, r in rows.items():
                rowmap.setdefault((m, b, r), []).append(("A", i))
        deg_hist = Counter(len(v) for v in rowmap.values())
        deg_ok = all(len(v) == 2 and {x[0] for x in v} == {"R", "A"} for v in rowmap.values())
        cyc_lens = ["n/a"]
        if deg_ok:
            adjn = {n: [] for n in nodes}
            for v in rowmap.values():
                a, b2 = v
                adjn[a].append(b2)
                adjn[b2].append(a)
            cyc_lens = []
            seen = set()
            for n in nodes:
                if n in seen:
                    continue
                ln, cur, prev = 0, n, None
                while cur not in seen:
                    seen.add(cur)
                    ln += 1
                    nxt = [x for x in adjn[cur] if x != prev]
                    prev, cur = cur, (nxt[0] if nxt else adjn[cur][0])
                cyc_lens.append(ln)
            cyc_lens = sorted(cyc_lens, reverse=True)
        atom_data.append((ro, ao))
        result("U2a", atom=ai, size=atom["size"], remove_orbits=len(ro), add_orbits=len(ao),
               add_orbits_that_are_t4=n_t4,
               remove_cells=cellstr(ro), add_cells=cellstr(ao),
               same_physical_rows=bool(rows_rem == rows_add),
               rows_removed=len(set(rows_rem)), rows_added=len(set(rows_add)),
               rows_common=len(set(rows_rem) & set(rows_add)),
               row_degree_hist=str(dict(sorted(deg_hist.items()))),
               every_row_one_removed_one_added=deg_ok,
               alternating_cycle_lengths=str(cyc_lens))

    # each atom removes ALL of S's orbits in its channels, adds a full rewiring
    s_orbit_ids = {id(o): oi for oi, o in enumerate(s_orbits)}
    for ai, (ro, ao) in enumerate(atom_data):
        chs = {m for (_, _, m, _) in ro}
        s_in_chs = {frozenset(orb) for (t, orb, m, rows) in s_orbits if m in chs}
        rem_orbs = {frozenset(orb) for (t, orb, m, rows) in ro}
        result("U2a2", atom=ai, channels=str(sorted(chname(m) for m in chs)),
               removes_all_S_orbits_of_its_channels=bool(rem_orbs == s_in_chs),
               s_orbits_in_channels=len(s_in_chs))

    # the four (12,12) atoms jointly = the frame swap of T3.3
    ro4 = [o for ai in range(4) for o in atom_data[ai][0]]
    ao4 = [o for ai in range(4) for o in atom_data[ai][1]]
    rem_ch = {m for (_, _, m, _) in ro4}
    add_ch = {m for (_, _, m, _) in ao4}
    # both must be a single frame channel and a perfect K888 matching there
    rows_rem = Counter((b) for (_, _, _, rows) in ro4 for b in rows)
    rows_add = Counter((b) for (_, _, _, rows) in ao4 for b in rows)
    rowset_rem = {(b, r) for (_, _, _, rows) in ro4 for b, r in rows.items()}
    rowset_add = {(b, r) for (_, _, _, rows) in ao4 for b, r in rows.items()}
    result("U2b", frame_swap_remove_channels=str(sorted(chname(m) for m in rem_ch)),
           frame_swap_add_channels=str(sorted(chname(m) for m in add_ch)),
           same_single_channel=bool(rem_ch == add_ch and len(rem_ch) == 1),
           remove_is_perfect_matching=bool(dict(rows_rem) == {0: 8, 1: 8, 2: 8}
                                           and len(rowset_rem) == 24),
           add_is_perfect_matching=bool(dict(rows_add) == {0: 8, 1: 8, 2: 8}
                                        and len(rowset_add) == 24),
           add_rows_equal_remove_rows=bool(rowset_rem == rowset_add))

    # joint check for the four (12,12) atoms: together they remove ALL of S's
    # cF2 orbits and add exactly the 12-orbit t4 matching of cF2
    rem4 = {frozenset(orb) for ai in range(4) for (t, orb, m, rows) in atom_data[ai][0]}
    add4 = {frozenset(orb) for ai in range(4) for (t, orb, m, rows) in atom_data[ai][1]}
    s_cf2 = {frozenset(orb) for (t, orb, m, rows) in s_orbits if chname(m) == "cF2"}
    t4_cf2 = {frozenset(orb) for (t, orb, m, rows) in t4_orbits if chname(m) == "cF2"}
    result("U2b2", four_12atoms_remove_eq_S_cF2=bool(rem4 == s_cf2),
           four_12atoms_add_eq_t4_cF2=bool(add4 == t4_cf2))

    # identity: the 80 t4 vertices = adds of atoms 0-3 (48) + the tightness-4
    # quarter of each (80,80) atom's adds (16 + 16)
    all_add_orbs = {frozenset(orb) for ai in range(6) for (t, orb, m, rows) in atom_data[ai][1]}
    t4_orbs = {frozenset(orb) for (t, orb, m, rows) in t4_orbits}
    result("U2b3", t4_orbits_subset_of_atom_adds=bool(t4_orbs <= all_add_orbs),
           t4_orbits_in_adds=len(t4_orbs & all_add_orbs), atom_add_orbits=len(all_add_orbs))

    # PG(3,2): the involution sigma: c -> c XOR cF2 and the channel-group structure
    mF2 = next(m for m in CHANNEL_NAMES if CHANNEL_NAMES[m] == "cF2")
    mF0 = next(m for m in CHANNEL_NAMES if CHANNEL_NAMES[m] == "cF0")
    mF1 = next(m for m in CHANNEL_NAMES if CHANNEL_NAMES[m] == "cF1")
    m04 = next(m for m in CHANNEL_NAMES if CHANNEL_NAMES[m] == "c04")
    m07 = next(m for m in CHANNEL_NAMES if CHANNEL_NAMES[m] == "c07")
    mE = next(m for m in CHANNEL_NAMES if CHANNEL_NAMES[m] == "cE")
    span = set()
    for a in (0, mF0):
        for b in (0, mF1):
            for c in (0, mF2):
                span.add(a ^ b ^ c)
    result("U2d", sigma_pairs_moved_by_plateau="{0,cF2},{cF0,c07},{cF1,c04}",
           cF0_xor_c07_eq_cF2=bool(mF0 ^ m07 == mF2),
           cF1_xor_c04_eq_cF2=bool(mF1 ^ m04 == mF2),
           cF0_xor_cF1_eq_cE=bool(mF0 ^ mF1 == mE),
           frame_span_dim=3, frame_span_size=len(span),
           frame_span_names=str(sorted(chname(m) for m in span if m)),
           empty_channel_in_frame_span=bool(mE in span))

    # the two (80,80) atoms, orbit-level anatomy
    for ai in (4, 5):
        ro, ao = atom_data[ai]
        pr = Counter((t, chname(m)) for (t, _, m, _) in ro)
        pa = Counter((t, chname(m)) for (t, _, m, _) in ao)
        # per-cell delta of the S usage table
        delta = Counter()
        for (t, _, m, _) in ao:
            delta[(t, chname(m))] += 1
        for (t, _, m, _) in ro:
            delta[(t, chname(m))] -= 1
        delta = {f"pair{t}/{c}": v for (t, c), v in sorted(delta.items()) if v}
        result("U2c", atom=ai, remove_cells=str({f"pair{t}/{c}": v for (t, c), v in sorted(pr.items())}),
               add_cells=str({f"pair{t}/{c}": v for (t, c), v in sorted(pa.items())}),
               cell_usage_delta=str(delta))
    result("U2", seconds=round(time.time() - t0, 1))
    return atom_data


# --------------------------------------------------------------------- U3

def u3_union(S, V, ts, IB8, t4, t4_orbits, s_orbits, T_channels):
    t0 = time.time()
    U = np.concatenate([S, t4], axis=0)
    assert U.shape == (576, 24)
    # Gram histogram, exact
    G = U @ U.T
    assert np.all(np.diag(G) == 32)
    off = G[~np.eye(576, dtype=bool)]
    vals, counts = np.unique(off, return_counts=True)
    ghist = {int(v): int(c) for v, c in zip(vals, counts)}
    # tight-frame test: sum of v v^T over the union
    P = U.T @ U
    frame_const = 576 * 32 // 24
    is_tight_frame = bool(np.array_equal(P, frame_const * np.eye(24, dtype=np.int64)))
    PS = S.T @ S
    s_tight = bool(np.array_equal(PS, (496 * 32 // 24 if (496 * 32) % 24 == 0 else -1)
                                  * np.eye(24, dtype=np.int64))) if (496 * 32) % 24 == 0 else False
    P4 = t4.T @ t4
    t4_diag = sorted(set(int(x) for x in np.diag(P4)))
    result("U3a", union_size=576, note="576=24^2", gram_offdiag=str(ghist),
           conflicts_pm16=ghist.get(16, 0) + ghist.get(-16, 0),
           sum_vvT_eq_768I=is_tight_frame, frame_const=frame_const,
           S_alone_tight_frame=s_tight, S_sum_vvT_diag=str(sorted(set(int(x) for x in np.diag(PS)))[:4]),
           t4_sum_vvT_diag=str(t4_diag))

    # channel table of the 144 orbits
    all_orbits = s_orbits + t4_orbits
    assert len(all_orbits) == 144
    table = {m: [0, 0, 0] for m in T_channels}
    for (t, _, m, _) in all_orbits:
        table[m][t] += 1
    per_channel = {chname(m): sum(v) for m, v in sorted(table.items())}
    result("U3b", union_orbits=144,
           union_cell_table=str({chname(m): v for m, v in sorted(table.items())}),
           per_channel_totals=str(per_channel),
           channel_total_hist=str(dict(sorted(Counter(per_channel.values()).items()))))

    # conflict structure: edges only between t4 and S (both sides independent)
    Sf, Tf = S.astype(np.float64), t4.astype(np.float64)
    D = (Tf @ Sf.T == 16)
    conf_counts = D.sum(axis=1)
    assert np.all(conf_counts == 4)
    # orbit-level conflict graph: t4 orbit x conflicts S orbit y iff any pair at dot 16
    s_of = {}
    for oi, (t, orb, m, rows) in enumerate(s_orbits):
        for k in orb:
            s_of[k] = oi
    t4_conf = []  # per t4 orbit: set of S orbit ids
    for (t, orb, m, rows) in t4_orbits:
        smem = set()
        for k in orb:
            v = np.array(k, dtype=np.int64)
            hits = np.where((Sf @ v.astype(np.float64)) == 16)[0]
            for h in hits:
                smem.add(s_of[apair_key(S[h])])
        t4_conf.append(frozenset(smem))
    conf_sizes = Counter(len(c) for c in t4_conf)
    # do conflicts follow rook rows? t4 orbit conflicts exactly the S orbits
    # sharing a rook row (same block, same +-pair of norm-64 sum vectors) in the
    # same channel
    row_index = {}
    for oi, (t, orb, m, rows) in enumerate(s_orbits):
        for b, r in rows.items():
            row_index.setdefault((m, b, r), set()).add(oi)
    rook_ok = True
    for (t, orb, m, rows), conf in zip(t4_orbits, t4_conf):
        pred = set()
        for b, r in rows.items():
            pred |= row_index.get((m, b, r), set())
        if pred != set(conf):
            rook_ok = False
    result("U3c", t4_vertex_conf=4, t4_orbit_conflict_S_orbits=str(dict(sorted(conf_sizes.items()))),
           conflicts_are_exactly_shared_rook_rows_same_channel=rook_ok)

    # components of the bipartite conflict graph on orbits
    parent = list(range(144))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ti, conf in enumerate(t4_conf):
        for so in conf:
            ra, rb = find(124 + ti), find(so)
            if ra != rb:
                parent[ra] = rb
    comps = {}
    for i in range(144):
        comps.setdefault(find(i), []).append(i)
    active = [c for c in comps.values() if len(c) > 1]
    comp_profile = Counter()
    comp_desc = []
    for c in sorted(active, key=len, reverse=True):
        ns = sum(1 for i in c if i < 124)
        nt = len(c) - ns
        chs = {chname(all_orbits[i][2]) for i in c}
        comp_profile[(ns, nt)] += 1
        comp_desc.append((ns, nt, sorted(chs)))
    result("U3d", conflict_components=len(active),
           isolated_S_orbits=144 - sum(len(c) for c in active) - 0,
           component_profile_Sorbits_t4orbits=str(dict(sorted(comp_profile.items()))),
           components=str(comp_desc))
    result("U3", seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------- U4

# the five Co0-invariant tightness histograms of the plateau (T3.4b / A1 / T3.2c §8.2)
FP_CLASSES = [
    {4: 80, 6: 640, 7: 256, 8: 2704, 9: 8064, 10: 31424, 11: 52672, 12: 49552, 13: 31360,
     14: 13440, 15: 2560, 16: 1480, 17: 256, 18: 704, 19: 64, 20: 528, 22: 128, 24: 152},
    {4: 80, 6: 672, 7: 128, 8: 2800, 9: 8320, 10: 31312, 11: 51584, 12: 51616, 13: 29312,
     14: 14912, 15: 1792, 16: 1864, 18: 848, 20: 544, 22: 128, 24: 152},
    {4: 80, 6: 672, 7: 128, 8: 2800, 9: 8320, 10: 31056, 11: 52864, 12: 49056, 13: 31872,
     14: 13632, 15: 2048, 16: 1864, 18: 848, 20: 544, 22: 128, 24: 152},
    {4: 80, 6: 640, 7: 256, 8: 2576, 9: 8448, 10: 31424, 11: 52032, 12: 49680, 13: 31744,
     14: 13440, 15: 2432, 16: 1480, 17: 256, 18: 704, 19: 64, 20: 528, 22: 128, 24: 152},
    {4: 80, 6: 640, 7: 256, 8: 2320, 9: 9728, 10: 28864, 11: 54848, 12: 47888, 13: 32000,
     14: 13952, 15: 2176, 16: 1480, 17: 256, 18: 704, 19: 64, 20: 528, 22: 128, 24: 152},
]
GRAM_FULL = {-32: 496, -8: 75008, 0: 95008, 8: 75008}


def read_set_file(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                rows.append([int(v) for v in line.split()])
    return np.array(rows, dtype=np.int64)


def certify_set(X: np.ndarray, V, vindex) -> dict:
    """A1-style certification, exact: returns dict of invariants, raises on
    any structural failure."""
    assert X.shape == (496, 24)
    idx = []
    for r in X:
        k = bytes(r.astype(np.int8))
        assert k in vindex, "row is not a Leech minimal vector"
        idx.append(vindex[k])
    assert len(set(idx)) == 496, "duplicate rows"
    xkeys = {bytes(r.astype(np.int8)) for r in X}
    assert all(bytes((-r).astype(np.int8)) in xkeys for r in X), "not antipodal"
    G = X @ X.T
    assert np.all(np.diag(G) == 32)
    off = G[~np.eye(496, dtype=bool)]
    vals, counts = np.unique(off, return_counts=True)
    ghist = {int(v): int(c) for v, c in zip(vals, counts)}
    assert 16 not in ghist and -16 not in ghist, "not independent"
    tight = tightness(V, X)
    inside = np.zeros(len(V), dtype=bool)
    inside[idx] = True
    th = tight[~inside]
    hist = dict(sorted(Counter(th.tolist()).items()))
    free = int((th == 0).sum())
    t123 = int(((th >= 1) & (th <= 3)).sum())
    fp = FP_CLASSES.index(hist) if hist in FP_CLASSES else -1
    return {"indices": tuple(sorted(idx)), "gram": ghist, "gram_ok": ghist == GRAM_FULL,
            "free": free, "t123": t123, "min_tight": int(th.min()), "hist": hist, "fp": fp}


def u4_census(S, V, vindex, sset, ts, IB8, standins: int):
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "gpu_mis", "prod_*", "sets", "S_*.txt")))
    result("U4a", dump_files=len(files))
    sets = {}
    for fn in files:
        X = read_set_file(fn)
        key = X[np.lexsort(X.T[::-1])].tobytes()
        sets.setdefault(key, (X, fn))
    distinct = list(sets.values())
    result("U4b", distinct_sets=len(distinct))

    # bit-pack for fast pairwise overlaps
    n = len(distinct)
    bits = np.zeros((n, len(V)), dtype=bool)
    for i, (X, _) in enumerate(distinct):
        for r in X:
            bits[i, vindex[bytes(r.astype(np.int8))]] = True
    packed = np.packbits(bits, axis=1)
    # overlap matrix in chunks
    fam = list(range(n))

    def find(x):
        while fam[x] != x:
            fam[x] = fam[fam[x]]
            x = fam[x]
        return x

    overlaps_within, overlaps_across = [], []
    POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
    for i in range(n):
        ov = POP[np.bitwise_and(packed[i][None, :], packed[i + 1:])].sum(axis=1)
        for j0, o in enumerate(ov):
            j = i + 1 + j0
            if o >= 200:
                overlaps_within.append(int(o))
                ra, rb = find(i), find(j)
                if ra != rb:
                    fam[ra] = rb
            else:
                overlaps_across.append(int(o))
    fams: dict[int, list[int]] = {}
    for i in range(n):
        fams.setdefault(find(i), []).append(i)
    families = sorted(fams.values(), key=lambda f: distinct[f[0]][1])
    gap_ok = (min(overlaps_within, default=496) >= 288
              and max(overlaps_across, default=0) <= 100)
    result("U4c", families=len(families),
           family_sizes=str(dict(sorted(Counter(len(f) for f in families).items()))),
           within_family_overlap_range=f"[{min(overlaps_within, default='-')},{max(overlaps_within, default='-')}]",
           across_family_overlap_max=max(overlaps_across, default=0),
           bimodal_gap_clean=gap_ok)

    # which family is the record's? (contains data/S496.txt or overlaps it >= 288)
    s_bits = np.zeros(len(V), dtype=bool)
    for r in S:
        s_bits[vindex[bytes(r.astype(np.int8))]] = True
    s_packed = np.packbits(s_bits)
    id_fam = [fi for fi, f in enumerate(families)
              if int(POP[np.bitwise_and(packed[f[0]][None, :], s_packed[None, :])].sum()) >= 288]
    result("U4d", record_family_found=len(id_fam) == 1, record_family_index=str(id_fam))

    # certify EVERY distinct dumped set (exact); tightness class census per family
    os.makedirs(os.path.join(RUNS, "census_reps"), exist_ok=True)
    rep_files = []
    fam_stats = []
    bad = 0
    for fi, f in enumerate(families):
        classes = Counter()
        gram_ok = True
        free_total = 0
        t123_total = 0
        min_ts = set()
        for k, i in enumerate(f):
            X, fn = distinct[i]
            c = certify_set(X, V, vindex)
            classes[c["fp"]] += 1
            gram_ok &= c["gram_ok"]
            free_total += c["free"]
            t123_total += c["t123"]
            min_ts.add(c["min_tight"])
            if c["fp"] < 0 or not c["gram_ok"] or c["free"] or c["t123"]:
                bad += 1
            if k == 0:
                rep_path = os.path.join(RUNS, "census_reps", f"R_{fi:02d}.txt")
                with open(rep_path, "w") as fh:
                    fh.write(f"# A4b census family {fi} representative; source {fn}\n")
                    for r in X:
                        fh.write(" ".join(str(int(v)) for v in r) + "\n")
                rep_files.append(rep_path)
        fam_stats.append((fi, len(f), dict(sorted(classes.items())), gram_ok,
                          free_total, t123_total, sorted(min_ts)))
    for fi, sz, cls, gok, ft, t123, mts in fam_stats:
        result("U4e", family=fi, members=sz, fp_classes=str(cls), gram_all_record=gok,
               free_total=ft, t123_total=t123, min_tightness=str(mts))
    all_classes = Counter()
    for _, _, cls, _, _, _, _ in fam_stats:
        for k, v in cls.items():
            all_classes[k] += v
    result("U4f", certified_sets=n, failures=bad,
           fp_class_totals=str(dict(sorted(all_classes.items()))),
           all_in_5_known_classes=bool(-1 not in all_classes),
           all_gram_record=bool(all(g for _, _, _, g, _, _, _ in fam_stats)),
           free_total=0 if not bad else "see U4e", families_covered=len(families))

    # stand-in families: fresh random Co0 images of S (genuinely equivalent
    # seeds, statistically equivalent to the gpu_mis seeds which were NOT
    # logged as group elements)
    if standins > 0:
        aut_dir = os.path.join(RUNS, "auts")
        os.makedirs(aut_dir, exist_ok=True)
        need = [f for f in range(standins)
                if not os.path.exists(os.path.join(aut_dir, f"aut_{f:04d}.u32"))]
        if need:
            subprocess.run([os.path.join(ROOT, "build", "release", "tools", "random_aut"),
                            "--seed", "20260830", "--count", str(standins), "--out", aut_dir + "/"],
                           check=True, cwd=ROOT)
        s_idx = np.array(sorted(vindex[bytes(r.astype(np.int8))] for r in S))
        stand_classes = Counter()
        sbad = 0
        cross_max = 0
        for k in range(standins):
            perm = np.fromfile(os.path.join(aut_dir, f"aut_{k:04d}.u32"), dtype=np.uint32)
            assert len(perm) == len(V) and len(np.unique(perm)) == len(V)
            X = V[perm[s_idx]]
            c = certify_set(X, V, vindex)
            stand_classes[c["fp"]] += 1
            if c["fp"] < 0 or not c["gram_ok"] or c["free"] or c["t123"]:
                sbad += 1
            xb = np.zeros(len(V), dtype=bool)
            xb[perm[s_idx]] = True
            xp = np.packbits(xb)
            ov = POP[np.bitwise_and(packed, xp[None, :])].sum(axis=1)
            cross_max = max(cross_max, int(ov.max()))
            with open(os.path.join(RUNS, "census_reps", f"STANDIN_{k:02d}.txt"), "w") as fh:
                fh.write(f"# A4b stand-in family {k}: random Co0 image of S (aut_{k:04d}.u32)\n")
                for r in X:
                    fh.write(" ".join(str(int(v)) for v in r) + "\n")
        result("U4g", standin_families=standins, failures=sbad,
               fp_classes=str(dict(sorted(stand_classes.items()))),
               note="stand-ins are fresh random Co0 images (the gpu_mis runs did not log "
                    "their 64 group elements); statistically equivalent seeds, SAID SO",
               max_overlap_with_dumped_families=cross_max)

    # independent verifier pass on the representatives
    vfail = 0
    for rp in rep_files:
        r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"),
                            os.path.join(ROOT, "python", "verify_S.py"), rp],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and "ok=1" in r.stdout and "size=496" in r.stdout
        if not ok:
            vfail += 1
            print(f"VERIFY-FAIL {rp}: {r.stdout.strip()} {r.stderr.strip()}")
    result("U4h", verify_S_reps=len(rep_files), verify_S_failures=vfail)
    with open(os.path.join(RUNS, "census.json"), "w") as f:
        json.dump({"families": [{"index": fi, "members": sz, "classes": {str(a): b for a, b in cls.items()},
                                 "gram_all_record": gok, "free_total": ft, "t123_total": t123,
                                 "min_tightness": mts}
                                for fi, sz, cls, gok, ft, t123, mts in fam_stats]}, f, indent=1)
    result("U4", seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------- U5

def u5_clique(ts):
    t0 = time.time()
    with open(os.path.join(ROOT, "data", "scheme", "clique_g.json")) as f:
        cj = json.load(f)
    K = np.array(cj["clique_vectors"],
                 dtype=np.int64)
    assert K.shape == (24, 24)
    G = K @ K.T
    assert np.array_equal(G, 16 * (np.eye(24, dtype=np.int64) + 1))
    pats = block_patterns(K, ts)
    pu, pc = np.unique(pats, axis=0, return_counts=True)
    phist = {tuple(int(x) for x in u): int(c) for u, c in zip(pu, pc)}
    n_monad = sum(c for p, c in phist.items() if 32 in p)
    n_duad = sum(c for p, c in phist.items() if sorted(p) == [0, 16, 16])
    n_triad = 24 - n_monad - n_duad
    ssum = K.sum(axis=0)
    result("U5", clique_verified="Gram=16(I+J)", monads=n_monad, duads=n_duad, triads=n_triad,
           patterns=str({str(k): v for k, v in sorted(phist.items())}),
           seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true", help="run U4 (census certification)")
    ap.add_argument("--standins", type=int, default=0,
                    help="add N fresh random-Co0-image stand-in families to U4")
    args = ap.parse_args()

    t_start = time.time()
    S, V, vindex, sset, ts, B, IB8 = setup()

    # channel legend: A4's 15 channels; frame channels and the empty channel
    # get mnemonic names (F0/F1/F2 match A4's frame indices, E = globally empty)
    with open(os.path.join(ROOT, "runs", "a4", "channel_table.json")) as f:
        a4_table = {int(k): v for k, v in json.load(f).items()}
    assert len(a4_table) == 15
    special = {4388112: "cF0", 2490447: "cF1", 1689745: "cF2", 6616415: "cE"}
    plain = [m for m in sorted(a4_table) if m not in special]
    for i, m in enumerate(plain):
        CHANNEL_NAMES[m] = f"c{i:02d}"
    CHANNEL_NAMES.update(special)
    T_channels = sorted(a4_table)
    print("# channel legend:", {chname(m): m for m in T_channels})

    # S's own 124 orbits, recomputed here, cross-checked against A4's table
    s_orbits = orbit_decompose(S, ts, IB8)
    assert len(s_orbits) == 124
    s_table = {m: [0, 0, 0] for m in T_channels}
    for (t, _, m, _) in s_orbits:
        s_table[m][t] += 1
    assert s_table == a4_table, "recomputed S cell table differs from runs/a4/channel_table.json"
    result("U0b", s_orbits=124, s_cell_table_matches_A4=True)

    t4, t4_orbits, tight, smask = u1_t4(S, V, sset, ts, IB8)
    u2_atoms(S, V, sset, ts, IB8, t4, t4_orbits, s_orbits)
    u3_union(S, V, ts, IB8, t4, t4_orbits, s_orbits, T_channels)
    u5_clique(ts)
    if args.census:
        u4_census(S, V, vindex, sset, ts, IB8, args.standins)

    result("A4b", ok=1, seconds=round(time.time() - t_start, 1))
    with open(os.path.join(RUNS, "summary.json"), "w") as f:
        json.dump(SUMMARY, f, indent=1)


if __name__ == "__main__":
    main()

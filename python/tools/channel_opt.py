"""A2' — exact optimum over Turyn-symmetric (2^3-invariant) 60-degree-free sets.

Builds the EXACT orbit-level conflict model on the usable Stab_Co0(S496)-orbits
(the channel / rook structure of A4, re-derived from scratch with assertions),
then solves the maximum-weight independent-set problem exactly.

Background (docs/reports/A4.md, T3.4 section 5): Stab_Co0(S496) = blockwise
{+-1}^3 on R^24 = B1+B2+B3, Lambda cap B_i = sqrt2E8.  Its orbits on the
196560 minimal vectors: 23040 of size 8 (all self-conflicting), 2880 of size 4
("duad" orbits {+-x, +-y}, x perp y) and 360 of size 2 ("monad" orbits {+-m},
m a sqrt2E8 minimal vector of one block).  A 2^3-invariant 60-degree-free set
is exactly an independent union of usable orbits, weight = orbit size.

Orbit-level conflict = ANY cross dot +16 between members.  Because every orbit
is closed under negation, <u, v> = -16 with u in O1, v in O2 gives
<u, -v> = +16 with -v in O2; so at the orbit level a +-16 relation between two
orbits is the same thing as a +16 relation, and "conflict iff some cross dot
has absolute value 16" is EXACTLY the independence condition (the vector-level
constraint forbids only +16).

Subcommands
    build        build + verify the model, write runs/a2/model.npz + census
    census       print the structural census from the cached model
    structure    verify the line factorisation of ALL conflicts; runs/a2/lines.npz
    cliques      structural + greedy edge clique cover -> runs/a2/cliques.npz
    lp           LP relaxation over the clique cover (HiGHS)
    milp         exact MILP (scipy HiGHS) over the clique cover
    sat          SAT decision "weight >= BOUND?" via python-sat cadical195,
                 with DRAT proof on UNSAT (the checkable artifact)
    maxsat       RC2 MaxSAT (documented dead end: too slow on this instance)
    cpsat        OR-Tools CP-SAT exact optimum (the solver that closes it)
    pairs        exact per-pair / two-pair / duad-only / monad-only optima
    perfect      A4 conjecture: which channel subsets can be simultaneously
                 perfect (12 orbits each)?
    selftest     soundness test of the weighted cardinality CNF encoding

Run:  .venv/bin/python python/tools/channel_opt.py build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "python", "tools"))

RUNS = os.path.join(ROOT, "runs", "a2")
os.makedirs(RUNS, exist_ok=True)
MODEL = os.path.join(RUNS, "model.npz")


def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------------------ build

def build(args):
    t_start = time.time()
    import structure_496 as s496
    from kiss_ref.leech import leech_min_vectors

    S = s496.load_S()
    V = np.asarray(leech_min_vectors(), dtype=np.int64)
    n_V = len(V)
    assert n_V == 196560
    vindex = {bytes(r): i for i, r in enumerate(V.astype(np.int8))}

    # --- stabiliser elements, ordered as in A4 (t1 fixes 208 of S; t2, t3 fix 144)
    mats = s496.load_stab_elements()
    for M in mats:
        assert np.array_equal(M @ M.T, 64 * np.eye(24, dtype=np.int64))
    t8 = [M for M in mats if np.trace(M) == 64]
    assert len(t8) == 3

    def fixed_count(M):
        return int(np.sum(np.all((S @ M.T) // 8 == S, axis=1)))

    t8.sort(key=lambda M: (-fixed_count(M), M.tobytes()))
    assert [fixed_count(M) for M in t8] == [208, 144, 144]
    ts = t8

    # --- permutations of V induced by t1, t2, t3 and by -I
    perms = []
    for M in ts:
        img = (V @ M.T)
        assert np.all(img % 8 == 0)
        img //= 8
        p = np.fromiter((vindex[bytes(r)] for r in img.astype(np.int8)),
                        dtype=np.int64, count=n_V)
        perms.append(p)
    neg = np.fromiter((vindex[bytes(r)] for r in (-V).astype(np.int8)),
                      dtype=np.int64, count=n_V)
    log(f"# permutations built ({time.time()-t_start:.1f}s)")

    # --- orbits of the order-8 group (union-find)
    parent = np.arange(n_V, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in perms + [neg]:
        for i in range(n_V):
            ri, rj = find(i), find(int(p[i]))
            if ri != rj:
                parent[ri] = rj
    roots = np.fromiter((find(i) for i in range(n_V)), dtype=np.int64, count=n_V)
    orbit_members: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        orbit_members.setdefault(int(r), []).append(i)
    size_hist = Counter(len(m) for m in orbit_members.values())
    assert dict(size_hist) == {8: 23040, 4: 2880, 2: 360}, dict(size_hist)
    log(f"# orbit sizes {dict(size_hist)} ({time.time()-t_start:.1f}s)")

    # --- self-conflict check: an orbit is usable iff no internal dot == 16.
    # (vector constraint forbids dot +16 only; orbits are negation-closed)
    n_self8 = 0
    usable = []
    for r, mem in sorted(orbit_members.items(), key=lambda kv: min(kv[1])):
        W = V[mem]
        G = W @ W.T
        internal16 = bool(np.any(G == 16))
        if len(mem) == 8:
            assert internal16, "size-8 orbit without self-conflict!"
            n_self8 += 1
        else:
            assert not internal16, "small orbit self-conflicts!"
            usable.append(mem)
    assert n_self8 == 23040 and len(usable) == 3240
    n_orb = len(usable)

    # --- block classification machinery
    B, IB8 = s496.build_basis(V)
    pw2 = (1 << np.arange(24, dtype=np.int64))

    def mod2_mask(vecs) -> np.ndarray:
        C = s496.coords_of(np.asarray(vecs, dtype=np.int64), IB8) % 2
        return (C * pw2).sum(axis=1)

    def block_of(w) -> int:
        """The unique i with t_i w = -w (w supported on block B_{i+1})."""
        negs = [i for i, M in enumerate(ts) if np.array_equal((w @ M.T) // 8, -w)]
        assert len(negs) == 1
        return negs[0]

    # frame lines per (block, channel): enumerate norm-64 vectors of each
    # block lattice, classify mod 2Lambda -> 135 frames of 8 +-pairs per block
    Ps = []
    I24 = np.eye(24, dtype=np.int64)
    for M in ts:
        img = B @ M.T
        assert np.all(img % 8 == 0)
        Ps.append(s496.coords_of(img // 8, IB8))
    blocks = []
    for P in Ps:
        K = s496.left_kernel(P + I24)
        Kb = np.array(K, dtype=np.int64) @ B
        assert Kb.shape[0] == 8
        blocks.append(Kb)
    line_index: dict[tuple[int, int], dict[tuple, int]] = {}
    block_channels: list[set[int]] = []
    for b, Kb in enumerate(blocks):
        vecs = s496.short_vectors_list(Kb @ Kb.T, 64)
        v64 = np.array([c for nrm, c in vecs if nrm == 64]) @ Kb
        assert len(v64) == 2160
        masks = mod2_mask(v64)
        by_class: dict[int, set] = {}
        for m, v in zip(masks, v64):
            by_class.setdefault(int(m), set()).add(s496.apair_key(v))
        assert len(by_class) == 135
        assert all(len(lines) == 8 for lines in by_class.values())
        for m, lines in by_class.items():
            line_index[(b, m)] = {k: i for i, k in enumerate(sorted(lines))}
        block_channels.append(set(by_class))
    shared01 = block_channels[0] & block_channels[1]
    shared02 = block_channels[0] & block_channels[2]
    shared12 = block_channels[1] & block_channels[2]
    assert shared01 == shared02 == shared12 and len(shared01) == 15
    T_channels = sorted(shared01)
    # the 15 channels are the nonzero elements of a 4-dim GF(2) subspace
    assert s496.gf2_rank(T_channels) == 4
    Tset = set(T_channels) | {0}
    assert all((a ^ b) in Tset for a in Tset for b in Tset)
    chan_id = {c: i for i, c in enumerate(T_channels)}
    log(f"# 3 blocks x 135 frames; 15 shared channels = PG(3,2) "
        f"({time.time()-t_start:.1f}s)")

    # --- orbit metadata
    # kind 0 = duad, 1 = monad; for duads: pair p (0:{B2,B3},1:{B1,B3},2:{B1,B2}),
    # channel id 0..14, row = line id in lower block, col = line id in upper block.
    kind = np.zeros(n_orb, dtype=np.int64)
    weight = np.zeros(n_orb, dtype=np.int64)
    pair = np.full(n_orb, -1, dtype=np.int64)     # duads: pair index; monads: -1
    mblock = np.full(n_orb, -1, dtype=np.int64)   # monads: block index
    chan = np.full(n_orb, -1, dtype=np.int64)
    row = np.full(n_orb, -1, dtype=np.int64)      # line id in lower-numbered block
    col = np.full(n_orb, -1, dtype=np.int64)      # line id in higher-numbered block
    # class representatives for the conflict computation (1 or 2 per orbit)
    class_vecs = []
    class_orbit = []
    # global line key (chan, block, line) used by each duad, for cross-pair checks
    lines_used: list[list[tuple[int, int, int]]] = []
    # monad line data: which frame lines conflict with a monad (filled later)

    PAIR_BLOCKS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    for oi, mem in enumerate(usable):
        W = V[mem]
        weight[oi] = len(mem)
        if len(mem) == 2:
            kind[oi] = 1
            m = W[0]
            b = block_of(m)
            mblock[oi] = b
            class_vecs.append(m)
            class_orbit.append(oi)
            lines_used.append([])
            continue
        # duad: pick x and y != +-x
        x = W[0]
        y = next(w for w in W[1:] if not np.array_equal(w, -x))
        assert int(x @ y) == 0
        u, w2 = x + y, x - y
        assert int(u @ u) == 64 and int(w2 @ w2) == 64
        bu, bw = block_of(u), block_of(w2)
        assert bu != bw
        p = next(i for i in range(3) if i not in (bu, bw))
        pair[oi] = p
        mu = int(mod2_mask(u.reshape(1, -1))[0])
        mw = int(mod2_mask(w2.reshape(1, -1))[0])
        assert mu == mw and mu in chan_id, "duad channel not among the 15!"
        chan[oi] = chan_id[mu]
        j, k = PAIR_BLOCKS[p]
        lo, hi = (u, w2) if bu == j else (w2, u)
        r = line_index[(j, mu)][s496.apair_key(lo)]
        c = line_index[(k, mu)][s496.apair_key(hi)]
        row[oi], col[oi] = r, c
        class_vecs.append(x)
        class_orbit.append(oi)
        class_vecs.append(y)
        class_orbit.append(oi)
        lines_used.append([(chan_id[mu], j, r), (chan_id[mu], k, c)])

    n_duad = int(np.sum(kind == 0))
    n_monad = int(np.sum(kind == 1))
    assert (n_duad, n_monad) == (2880, 360)
    assert 4 * n_duad + 2 * n_monad == 12240  # all duads + monads, A4 S5b
    assert dict(Counter(int(p) for p in pair[kind == 0])) == {0: 960, 1: 960, 2: 960}
    assert dict(Counter(int(b) for b in mblock[kind == 1])) == {0: 120, 1: 120, 2: 120}
    # every (pair, channel) cell is a FULL 8x8 rook grid
    cells = Counter((int(p), int(c)) for p, c, k in zip(pair, chan, kind) if k == 0)
    assert len(cells) == 45 and set(cells.values()) == {64}
    for p in range(3):
        for c in range(15):
            sel = (pair == p) & (chan == c)
            rc = set(zip(row[sel].tolist(), col[sel].tolist()))
            assert len(rc) == 64  # all 64 (row, col) present exactly once
    log(f"# metadata: 2880 duads = 3 pairs x 15 channels x 64 rook cells; "
        f"360 monads = 3 blocks x 120 ({time.time()-t_start:.1f}s)")

    # --- conflict graph on antipodal-class representatives
    R = np.array(class_vecs, dtype=np.int64)
    corb = np.array(class_orbit, dtype=np.int64)
    n_cls = len(R)
    assert n_cls == 2 * 2880 + 360 == 6120
    A = np.zeros((n_orb, n_orb), dtype=bool)
    step = 512
    for lo in range(0, n_cls, step):
        D = R[lo:lo + step] @ R.T
        ii, jj = np.nonzero(np.abs(D) == 16)
        A[corb[lo + ii], corb[jj]] = True
    A |= A.T
    assert not A.diagonal().any()  # usable orbits never self-conflict
    n_edges = int(np.triu(A, 1).sum())
    assert n_edges == 456480, n_edges  # T3.4's reduced-graph edge count
    log(f"# conflict graph: 3240 orbits, {n_edges} edges == T3.4 "
        f"({time.time()-t_start:.1f}s)")

    # --- reference set S -> orbit indices
    orbit_of_vec = {}
    for oi, mem in enumerate(usable):
        for i in mem:
            orbit_of_vec[i] = oi
    s_orbits = sorted({orbit_of_vec[vindex[bytes(r.astype(np.int8))]] for r in S})
    assert len(s_orbits) == 124
    s_idx = np.array(s_orbits, dtype=np.int64)
    assert int(weight[s_idx].sum()) == 496
    assert not A[np.ix_(s_idx, s_idx)].any()  # S is independent at orbit level
    # channel table of S matches runs/a4/channel_table.json
    tab = {c: [0, 0, 0] for c in range(15)}
    for oi in s_orbits:
        tab[int(chan[oi])][int(pair[oi])] += 1
    with open(os.path.join(ROOT, "runs", "a4", "channel_table.json")) as f:
        a4tab = json.load(f)
    a4tab = {chan_id[int(k)]: v for k, v in a4tab.items()}
    assert tab == a4tab, "S channel usage disagrees with A4"
    log(f"# reference S: 124 orbits, weight 496, independent, "
        f"channel table == A4 ({time.time()-t_start:.1f}s)")

    # --- structural census of the edges
    # categories: mm (monad-monad), md (monad-duad), rook (same pair+channel),
    # xchan (same pair, different channel), xpair_same_chan, xpair_xchan
    cat = Counter()
    Ei, Ej = np.nonzero(np.triu(A, 1))
    line_sets = [set(l) for l in lines_used]
    rook_ok = True
    xpair_line_ok = True
    for a, b in zip(Ei.tolist(), Ej.tolist()):
        ka, kb = kind[a], kind[b]
        if ka and kb:
            cat["mm"] += 1
            assert mblock[a] == mblock[b]  # monads conflict only within a block
        elif ka or kb:
            cat["md"] += 1
            d = a if kb else b
            m = b if kb else a
            assert mblock[m] in PAIR_BLOCKS[int(pair[d])]
        elif pair[a] == pair[b]:
            if chan[a] == chan[b]:
                cat["rook"] += 1
                if not (row[a] == row[b] or col[a] == col[b]):
                    rook_ok = False
            else:
                cat["xchan"] += 1
        else:
            if chan[a] == chan[b]:
                cat["xpair_same_chan"] += 1
                if not (line_sets[a] & line_sets[b]):
                    xpair_line_ok = False
            else:
                cat["xpair_xchan"] += 1
    assert cat["xpair_xchan"] == 0, "cross-pair cross-channel conflicts exist!"
    assert rook_ok, "same-cell conflict without shared row/col!"
    assert xpair_line_ok, "cross-pair same-channel conflict without shared line!"
    # converse checks: every shared row/col IS a conflict; every shared line
    # across pairs IS a conflict
    for p in range(3):
        for c in range(0, 15, 7):  # spot-check 3 channels per pair
            sel = np.flatnonzero((pair == p) & (chan == c))
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    a, b = sel[i], sel[j]
                    share = (row[a] == row[b]) or (col[a] == col[b])
                    assert bool(A[a, b]) == bool(share)
    by_line: dict[tuple, list[int]] = {}
    for oi, ls in enumerate(lines_used):
        for l in ls:
            by_line.setdefault(l, []).append(oi)
    assert len(by_line) == 15 * 3 * 8  # 360 (chan, block, line) triples
    assert all(len(v) == 16 for v in by_line.values())  # 8 orbits x 2 pairs
    for l, ois in list(by_line.items())[::37]:
        sub = A[np.ix_(ois, ois)]
        assert sub.sum() == len(ois) * (len(ois) - 1)  # a 16-clique
    log(f"# census: {dict(cat)} (sum {sum(cat.values())}) "
        f"({time.time()-t_start:.1f}s)")

    # monad-duad structure: a monad conflicts with duads through frame lines?
    # For monad m in block b and duad with line (c, b, l): conflict iff
    # |m . u| = 32 for u the line's vector. Census how many (channel, line)
    # per monad, and verify conflicts factor through lines.
    md_lines_ok = True
    monad_line_conf: dict[int, set] = {}
    for oi in np.flatnonzero(kind == 1):
        confs = np.flatnonzero(A[oi] & (kind == 0))
        lset = set()
        for d in confs:
            ls = [l for l in lines_used[d] if l[1] == mblock[oi]]
            if len(ls) != 1:
                md_lines_ok = False
            lset.update(ls)
        # verify: EVERY duad on a conflicting line conflicts
        for l in lset:
            if not all(A[oi, d] for d in by_line[l]
                       if pair[d] >= 0):
                md_lines_ok = False
        monad_line_conf[int(oi)] = lset
    assert md_lines_ok, "monad-duad conflicts do not factor through lines!"
    md_hist = Counter(len(v) for v in monad_line_conf.values())
    per_chan_hist = Counter()
    for oi, lset in monad_line_conf.items():
        cc = Counter(l[0] for l in lset)
        per_chan_hist.update(Counter(cc.values()))
        assert len(cc) <= 15
    log(f"# monad->conflicting (chan,line) count hist {dict(md_hist)}; "
        f"lines-per-channel hist {dict(per_chan_hist)}")

    # per-pair cross-channel degree structure
    xchan_deg = Counter()
    for p in range(3):
        sel = np.flatnonzero((pair == p))
        for c in range(15):
            si = sel[chan[sel] == c]
            for c2 in range(c + 1, 15):
                sj = sel[chan[sel] == c2]
                e = int(A[np.ix_(si, sj)].sum())
                xchan_deg[(p, c, c2)] = e
    xc_hist = Counter(xchan_deg.values())
    log(f"# cross-channel same-pair edge counts per (pair, chan<chan') "
        f"hist {dict(sorted(xc_hist.items()))}")

    census = {
        "orbits": n_orb, "duads": n_duad, "monads": n_monad,
        "edges": n_edges, "edge_census": dict(cat),
        "reference_orbits": len(s_orbits), "reference_weight": 496,
        "monad_line_conflict_hist": {str(k): v for k, v in sorted(md_hist.items())},
        "xchan_block_edge_hist": {str(k): v for k, v in sorted(xc_hist.items())},
        "channels": T_channels,
    }
    with open(os.path.join(RUNS, "census.json"), "w") as f:
        json.dump(census, f, indent=1)

    # pack edges + metadata
    np.savez_compressed(
        MODEL,
        edges_i=Ei.astype(np.int32), edges_j=Ej.astype(np.int32),
        kind=kind, weight=weight, pair=pair, mblock=mblock,
        chan=chan, row=row, col=col,
        ref=s_idx,
        monad_lines=np.array(
            [[oi, l[0], l[1], l[2]] for oi, ls in monad_line_conf.items()
             for l in sorted(ls)], dtype=np.int32),
        orbit_min_vec=np.array([min(m) for m in usable], dtype=np.int64),
        members=np.array([m + [-1] * (4 - len(m)) for m in usable],
                         dtype=np.int64),
    )
    log(f"RESULT section=build orbits={n_orb} edges={n_edges} "
        f"census={dict(cat)} ref=496 ok=1 seconds={time.time()-t_start:.1f}")


def load_model():
    d = np.load(MODEL)
    n = len(d["kind"])
    A = np.zeros((n, n), dtype=bool)
    A[d["edges_i"], d["edges_j"]] = True
    A |= A.T
    return d, A


LINES = os.path.join(RUNS, "lines.npz")
PAIR_BLOCKS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


# ------------------------------------------------------------------ structure

def structure(args):
    """Verify that the WHOLE conflict graph factors through block-level line
    data, and save that data (runs/a2/lines.npz):

      * duad orbit <-> (pair p, chan c, row r, col s): its two "lines" are
        (c, j, r) and (c, k, s) with (j, k) = PAIR_BLOCKS[p]; the line vector
        u(c,b,l) is a norm-64 vector of block b (+- ambiguous, fine for dots);
      * same channel:      conflict <=> shared line                  (16-cliques)
      * different channel, same pair:
                           conflict <=> |u.u'| = 32 AND |w.w'| = 32  (both blocks)
      * different channel, different pair: never a conflict
      * monad-duad:        conflict <=> |m.u| = 32 for the duad's line u in the
                           monad's block
      * monad-monad:       conflict <=> same block and |m.m'| = 16.

    Derivation of the duad-duad rule (proved here by exhaustive check): with
    x=(u+w)/2, y=(u-w)/2, cross dots are (+-u.u' +- w.w')/4, and block dots lie
    in {0,+-16,+-32,+-64}; |a+-b| = 64 forces (|a|,|b|) in {(64,0),(0,64),
    (32,32)}, i.e. shared line (same channel) or double-32 (cross channel).
    """
    t0 = time.time()
    from kiss_ref.leech import leech_min_vectors

    d, A = load_model()
    V = np.asarray(leech_min_vectors(), dtype=np.int64)
    kind, pair, chan = d["kind"], d["pair"], d["chan"]
    row, col, mblock = d["row"], d["col"], d["mblock"]
    members = d["members"]
    n = len(kind)

    # reconstruct u (line vector, lower block) and w (upper block) per duad;
    # blockwise components via the duad's own members: x, y with u = x+y, w = x-y
    # and u sits in the lower-numbered block of the pair.  Which of x+y, x-y is
    # "lower" was fixed at build time through row/col; recompute via block test:
    # t_i negates block i; we do not need t_i here -- instead use the fact that
    # for duads of pair p the two components live in blocks PAIR_BLOCKS[p], and
    # identify them by matching against the line tables we build as we go.
    duads = np.flatnonzero(kind == 0)
    monads = np.flatnonzero(kind == 1)
    X = V[members[duads, 0]]
    # y = the member that is neither x nor -x
    Y = np.empty_like(X)
    for t, oi in enumerate(duads):
        W = V[members[oi, :4]]
        Y[t] = next(w for w in W[1:] if not np.array_equal(w, -W[0]))
    U, W2 = X + Y, X - Y
    assert np.all((U * U).sum(1) == 64) and np.all((W2 * W2).sum(1) == 64)

    # line vector table linevec[c, b, l]: identify each duad's block components
    # by the stabiliser (t_i negates exactly block B_{i+1}), same ordering as
    # build (t1 fixes 208 of S, t2/t3 fix 144)
    import structure_496 as s496
    S = s496.load_S()
    mats = s496.load_stab_elements()
    t8 = [M for M in mats if np.trace(M) == 64]
    assert len(t8) == 3

    def fixed_count(M):
        return int(np.sum(np.all((S @ M.T) // 8 == S, axis=1)))

    t8.sort(key=lambda M: (-fixed_count(M), M.tobytes()))
    assert [fixed_count(M) for M in t8] == [208, 144, 144]

    def block_of_rows(W):
        """block index per row of W (each supported on exactly one block)."""
        out = np.full(len(W), -1, dtype=np.int64)
        for i, M in enumerate(t8):
            neg = np.all((W @ M.T) // 8 == -W, axis=1)
            assert not np.any(neg & (out >= 0))
            out[neg] = i
        assert np.all(out >= 0)
        return out

    bU, bW = block_of_rows(U), block_of_rows(W2)
    linevec = np.zeros((15, 3, 8, 24), dtype=np.int64)
    filled = np.zeros((15, 3, 8), dtype=bool)
    for t, oi in enumerate(duads):
        p, c, r, s = int(pair[oi]), int(chan[oi]), int(row[oi]), int(col[oi])
        j, k = PAIR_BLOCKS[p]
        assert {int(bU[t]), int(bW[t])} == {j, k}
        u_t, w_t = (U[t], W2[t]) if bU[t] == j else (W2[t], U[t])
        for b, l, v in ((j, r, u_t), (k, s, w_t)):
            if filled[c, b, l]:
                assert np.array_equal(linevec[c, b, l], v) \
                    or np.array_equal(linevec[c, b, l], -v)
            else:
                linevec[c, b, l] = v
                filled[c, b, l] = True
    assert filled.all()
    # frames: within (c, b) the 8 lines are pairwise orthogonal, norm 64
    LV = linevec.reshape(15 * 3 * 8, 24)
    G = LV @ LV.T
    for c in range(15):
        for b in range(3):
            i0 = (c * 3 + b) * 8
            g = G[i0:i0 + 8, i0:i0 + 8]
            assert np.array_equal(np.diag(g), np.full(8, 64))
            assert not np.any(g[~np.eye(8, dtype=bool)])
    log(f"# line table: 15 channels x 3 blocks x 8 lines, frames verified "
        f"({time.time()-t0:.1f}s)")

    # every duad's (row, col) points to +-(its own u, w) exactly
    for t, oi in enumerate(duads):
        p, c, r, s = int(pair[oi]), int(chan[oi]), int(row[oi]), int(col[oi])
        j, k = PAIR_BLOCKS[p]
        u_t, w_t = (U[t], W2[t]) if bU[t] == j else (W2[t], U[t])
        assert abs(int(linevec[c, j, r] @ u_t)) == 64
        assert abs(int(linevec[c, k, s] @ w_t)) == 64

    # block-level 45-degree graphs H[b]: |u.u'| == 32 between channel lines
    dot_hist = Counter()
    H = np.zeros((3, 120, 120), dtype=bool)   # index (c, l) -> 8*c + l
    for b in range(3):
        Lb = linevec[:, b].reshape(120, 24)
        Gb = Lb @ Lb.T
        H[b] = np.abs(Gb) == 32
        offd = np.abs(Gb[~np.eye(120, dtype=bool)])
        dot_hist.update(offd.tolist())
        # same-channel off-diagonal dots are 0; cross-channel: exactly 4 lines
        # at |dot|=32 per (line, other channel)
        for c in range(15):
            for c2 in range(15):
                blkdeg = H[b, 8 * c:8 * c + 8, 8 * c2:8 * c2 + 8].sum(axis=1)
                assert np.all(blkdeg == (0 if c == c2 else 4))
    assert set(dot_hist) <= {0, 16, 32}, dict(dot_hist)
    log(f"# H_b: cross-channel line dots hist {dict(sorted(dot_hist.items()))}; "
        f"each line: 4 at |32| per foreign channel ({time.time()-t0:.1f}s)")

    # ---- factorisation check, all duad pairs at once, per pair p
    lid = np.full(n, -1, dtype=np.int64)   # not used; per-duad global line ids
    rowg = 8 * chan[duads] + row[duads]    # line id within lower block (0..119)
    colg = 8 * chan[duads] + col[duads]
    dpair = pair[duads]
    idx_of = {int(oi): t for t, oi in enumerate(duads)}
    Add = A[np.ix_(duads, duads)]
    pred = np.zeros_like(Add)
    for p in range(3):
        sel = np.flatnonzero(dpair == p)
        j, k = PAIR_BLOCKS[p]
        r_, c_ = rowg[sel], colg[sel]
        same_row = r_[:, None] == r_[None, :]
        same_col = c_[:, None] == c_[None, :]
        hj = H[j][np.ix_(r_, r_)]
        hk = H[k][np.ix_(c_, c_)]
        pp = same_row | same_col | (hj & hk)
        np.fill_diagonal(pp, False)
        pred[np.ix_(sel, sel)] = pp
        # cross-pair with p' > p: shared line in the common block only
        for p2 in range(p + 1, 3):
            sel2 = np.flatnonzero(dpair == p2)
            j2, k2 = PAIR_BLOCKS[p2]
            common = ({j, k} & {j2, k2}).pop()
            a1 = r_ if j == common else c_
            a2 = (rowg[sel2] if j2 == common else colg[sel2])
            pp = a1[:, None] == a2[None, :]
            pred[np.ix_(sel, sel2)] = pp
            pred[np.ix_(sel2, sel)] = pp.T
    assert np.array_equal(Add, pred), "duad-duad factorisation FAILED"
    log(f"# duad-duad conflicts == line factorisation, all {Add.sum()//2} edges "
        f"({time.time()-t0:.1f}s)")

    # ---- monads: m.linevec and m.m'
    Mv = V[members[monads, 0]]
    MG = Mv @ LV.T                      # (360, 360 lines)
    Mconf = np.abs(MG) == 32
    # monad-duad: conflict <=> monad's block hosts one of the duad's lines with
    # |dot| = 32
    Amd = A[np.ix_(monads, duads)]
    pmd = np.zeros_like(Amd)
    lin_g = np.zeros((3, len(duads)), dtype=np.int64)  # global line id per block
    for t in range(len(duads)):
        p = int(dpair[t])
        j, k = PAIR_BLOCKS[p]
        lin_g[j, t] = rowg[t]
        lin_g[k, t] = colg[t]
        lin_g[3 - j - k, t] = -1
    for mi in range(len(monads)):
        b = int(mblock[monads[mi]])
        gl = lin_g[b]
        usable_l = gl >= 0
        conf_lines = Mconf[mi].reshape(15 * 3, 8).reshape(15, 3, 8)[:, b].reshape(120)
        pmd[mi, usable_l] = conf_lines[gl[usable_l]]
    assert np.array_equal(Amd, pmd), "monad-duad factorisation FAILED"
    # per monad: 14 conflicting lines, 2 per channel in 7 channels
    per_m = Mconf.reshape(360, 15, 3, 8)
    for mi in range(len(monads)):
        b = int(mblock[monads[mi]])
        cnt = per_m[mi, :, b, :].sum(axis=1)
        assert per_m[mi, :, [x for x in range(3) if x != b], :].sum() == 0
        assert sorted(cnt.tolist()) == [0] * 8 + [2] * 7
    # monad-monad
    Amm = A[np.ix_(monads, monads)]
    GM = Mv @ Mv.T
    pmm = (np.abs(GM) == 16)
    same_b = mblock[monads][:, None] == mblock[monads][None, :]
    assert not np.any(pmm & ~same_b)
    np.fill_diagonal(pmm, False)
    assert np.array_equal(Amm, pmm), "monad-monad factorisation FAILED"
    log(f"# monad conflicts factor: m-d via |m.u|=32 (14 lines: 2 in each of 7 "
        f"channels), m-m via |m.m'|=16 in-block ({time.time()-t0:.1f}s)")

    np.savez_compressed(
        LINES,
        linevec=linevec, H=H,
        monad_vec=Mv, monad_line_conf=Mconf,
        duad_rowg=rowg, duad_colg=colg, duad_pair=dpair, duad_orbit=duads,
        monad_orbit=monads,
    )
    log(f"RESULT section=structure factorisation=verified "
        f"line_dot_hist={dict(sorted(dot_hist.items()))} ok=1 "
        f"seconds={time.time()-t0:.1f}")


# ------------------------------------------------------------------ cliques

CLIQUES = os.path.join(RUNS, "cliques.npz")


def _grow_clique(A, seed, uncovered):
    """Greedily extend `seed` (list of vertices, pairwise adjacent) to a
    maximal clique, preferring vertices that cover many uncovered edges."""
    cl = list(seed)
    cand = A[cl[0]].copy()
    for v in cl[1:]:
        cand &= A[v]
    cand[cl] = False
    while cand.any():
        cs = np.flatnonzero(cand)
        gain = uncovered[np.ix_(cs, cl)].sum(axis=1)
        v = int(cs[np.argmax(gain)])
        cl.append(v)
        cand &= A[v]
        cand[v] = False
    return cl


def build_cliques(d, A):
    """Structural + greedy edge clique cover.  Returns list of index arrays;
    every conflict edge is inside at least one clique (asserted)."""
    t0 = time.time()
    kind, pair, chan = d["kind"], d["pair"], d["chan"]
    row, col, mblock = d["row"], d["col"], d["mblock"]
    n = len(kind)
    duads = np.flatnonzero(kind == 0)
    monads = np.flatnonzero(kind == 1)
    uncovered = A.copy()
    cliques = []

    def add(cl):
        cl = np.asarray(sorted(cl), dtype=np.int32)
        sub = A[np.ix_(cl, cl)]
        assert sub.sum() == len(cl) * (len(cl) - 1), "not a clique!"
        cliques.append(cl)
        uncovered[np.ix_(cl, cl)] = False

    # 1. line cliques: 16 duads per line + monad extensions (each monad on the
    # line is adjacent to all 16 duads; partition those monads into mm-cliques)
    by_line: dict[tuple[int, int, int], list[int]] = {}
    for oi in duads:
        p, c, r, s = int(pair[oi]), int(chan[oi]), int(row[oi]), int(col[oi])
        j, k = PAIR_BLOCKS[p]
        by_line.setdefault((c, j, r), []).append(int(oi))
        by_line.setdefault((c, k, s), []).append(int(oi))
    assert len(by_line) == 360 and all(len(v) == 16 for v in by_line.values())
    n_line_cliques = 0
    for l, ois in sorted(by_line.items()):
        base = list(ois)
        # monads adjacent to every duad of the line
        mm = [int(m) for m in monads if A[m, base].all()]
        rest = set(mm)
        first = True
        while rest or first:
            if not rest:
                add(base)
                break
            grp = [rest.pop()]
            for m in sorted(rest):
                if all(A[m, g] for g in grp):
                    grp.append(m)
                    rest.discard(m)
            add(base + grp)
            n_line_cliques += 1
            first = False
        n_line_cliques += 0
    log(f"# clique cover: {len(cliques)} line(+monad) cliques "
        f"({time.time()-t0:.1f}s)")
    n_after_lines = len(cliques)

    # 2. monad-monad leftovers per block
    for b in range(3):
        mb = monads[mblock[monads] == b]
        while True:
            sub = uncovered[np.ix_(mb, mb)]
            ii, jj = np.nonzero(np.triu(sub, 1))
            if len(ii) == 0:
                break
            add(_grow_clique(A, [int(mb[ii[0]]), int(mb[jj[0]])], uncovered))
    n_after_mm = len(cliques)

    # 3. everything else (cross-channel duad edges + stragglers): greedy
    while True:
        ii, jj = np.nonzero(np.triu(uncovered, 1))
        if len(ii) == 0:
            break
        # process a batch to avoid rescanning the whole matrix every time
        for a, b in zip(ii.tolist()[:4000], jj.tolist()[:4000]):
            if uncovered[a, b]:
                add(_grow_clique(A, [a, b], uncovered))
    sizes = Counter(len(c) for c in cliques)
    log(f"# clique cover: {len(cliques)} cliques total "
        f"(lines {n_after_lines}, +mm {n_after_mm - n_after_lines}, "
        f"+greedy {len(cliques) - n_after_mm}); size hist "
        f"{dict(sorted(sizes.items()))} ({time.time()-t0:.1f}s)")
    return cliques


def cliques_cmd(args):
    d, A = load_model()
    cls = build_cliques(d, A)
    flat = np.concatenate(cls).astype(np.int32)
    lens = np.array([len(c) for c in cls], dtype=np.int32)
    np.savez_compressed(CLIQUES, flat=flat, lens=lens)
    log(f"RESULT section=cliques n={len(cls)} "
        f"max_size={int(lens.max())} ok=1")


def load_cliques():
    z = np.load(CLIQUES)
    flat, lens = z["flat"], z["lens"]
    out, pos = [], 0
    for L in lens:
        out.append(flat[pos:pos + L])
        pos += L
    return out


# ------------------------------------------------------------------ lp / milp

def _milp_solve(d, A, cliques, integrality, time_limit, extra_rows=None,
                fixed_zero=None):
    from scipy import sparse
    from scipy.optimize import milp, LinearConstraint, Bounds

    n = len(d["kind"])
    w = d["weight"].astype(float)
    rows, cols_, vals = [], [], []
    r = 0
    for cl in cliques:
        rows.extend([r] * len(cl))
        cols_.extend(cl.tolist())
        vals.extend([1.0] * len(cl))
        r += 1
    # monads-per-block <= 8 (valid: independent monads in a block are pairwise
    # orthogonal E8 root lines; a frame has 8)
    kind, mblock = d["kind"], d["mblock"]
    monads = np.flatnonzero(kind == 1)
    block_rows = []
    for b in range(3):
        mb = monads[mblock[monads] == b]
        rows.extend([r] * len(mb))
        cols_.extend(mb.tolist())
        vals.extend([1.0] * len(mb))
        block_rows.append(r)
        r += 1
    ub = np.ones(r)
    ub[block_rows] = 8.0
    if extra_rows:
        for idxs, bound in extra_rows:
            rows.extend([r] * len(idxs))
            cols_.extend(list(idxs))
            vals.extend([1.0] * len(idxs))
            ub = np.append(ub, bound)
            r += 1
    Asp = sparse.csr_matrix((vals, (rows, cols_)), shape=(r, n))
    lb_v = np.zeros(n)
    ub_v = np.ones(n)
    if fixed_zero is not None:
        ub_v[fixed_zero] = 0.0
    con = LinearConstraint(Asp, -np.inf, ub)
    res = milp(c=-w, constraints=[con], integrality=np.full(n, integrality),
               bounds=Bounds(lb_v, ub_v),
               options={"time_limit": time_limit, "disp": True})
    return res


def lp_cmd(args):
    from scipy import sparse
    from scipy.optimize import linprog
    t0 = time.time()
    d, A = load_model()
    cls = load_cliques()
    n = len(d["kind"])
    w = d["weight"].astype(float)
    rows, cols_, vals = [], [], []
    for r, cl in enumerate(cls):
        rows.extend([r] * len(cl))
        cols_.extend(cl.tolist())
        vals.extend([1.0] * len(cl))
    Asp = sparse.csr_matrix((vals, (rows, cols_)), shape=(len(cls), n))
    res = linprog(-w, A_ub=Asp, b_ub=np.ones(len(cls)), bounds=(0, 1),
                  method="highs-ipm")
    val = -res.fun if res.fun is not None else None
    log(f"RESULT section=lp bound={val} status={res.status} "
        f"cliques={len(cls)} seconds={time.time()-t0:.1f}")


def milp_cmd(args):
    t0 = time.time()
    d, A = load_model()
    cls = load_cliques()
    res = _milp_solve(d, A, cls, integrality=1, time_limit=args.time)
    gap = getattr(res, "mip_gap", None)
    bound = getattr(res, "mip_dual_bound", None)
    val = -res.fun if res.fun is not None else None
    log(f"RESULT section=milp primal={val} "
        f"dual_bound={-bound if bound is not None else None} "
        f"gap={gap} status={res.status} message={res.message!r} "
        f"seconds={time.time()-t0:.1f}")


# ------------------------------------------------------------------ SAT

def _card_atleast(cnf_clauses, lits, k, top):
    """Append clauses enforcing sum(lits) >= k.  Returns new top var id."""
    from pysat.card import CardEnc, EncType
    k, top = int(k), int(top)
    lits = [int(x) for x in lits]
    if k <= 0:
        return top
    if k > len(lits):
        aux = top + 1
        cnf_clauses.append([aux])
        cnf_clauses.append([-aux])
        return aux
    enc = CardEnc.atleast(lits=lits, bound=k, top_id=top,
                          encoding=EncType.sortnetwrk)
    cnf_clauses.extend(enc.clauses)
    return max(top, enc.nv)


def _card_atmost(cnf_clauses, lits, k, top):
    """Append clauses enforcing sum(lits) <= k.  Returns new top var id."""
    from pysat.card import CardEnc, EncType
    k, top = int(k), int(top)
    lits = [int(x) for x in lits]
    if k >= len(lits):
        return top
    enc = CardEnc.atmost(lits=lits, bound=k, top_id=top,
                         encoding=EncType.sortnetwrk)
    cnf_clauses.extend(enc.clauses)
    return max(top, enc.nv)


def build_cnf_weight_ge(d, A, target_weight, use_cuts=False):
    """CNF satisfiable iff a 2^3-invariant 60-degree-free orbit union of
    weight >= target_weight exists.  Vars 1..n = orbits.  Weighted count via
    duplicated duad vars (d1 <-> d <-> d2), monads counted once:
    4*duads + 2*monads >= target  <=>  2*duads + monads >= ceil(target/2).

    With use_cuts=True, ENTAILED constraints are additionally encoded (each is
    a theorem about the model proved by an exact subproblem solve recorded in
    runs/a2/{pairs,chancuts}.json + the monad-frame bound): channel <= 12,
    pair-line <= 16, pair-plane <= 32, pair <= 64 duads, monads/block <= 8.
    The resulting UNSAT is then a theorem MODULO those lemmas."""
    n = len(d["kind"])
    kind = d["kind"]
    clauses = []
    Ei, Ej = np.nonzero(np.triu(A, 1))
    for a, b in zip((Ei + 1).tolist(), (Ej + 1).tolist()):
        clauses.append([-a, -b])
    top = n
    lits = []
    for oi in range(n):
        v = oi + 1
        if kind[oi] == 0:
            top += 1
            clauses.append([-v, top])
            clauses.append([-top, v])
            lits.extend([v, top])
        else:
            lits.append(v)
    k = (int(target_weight) + 1) // 2
    top = _card_atleast(clauses, lits, k, top)
    n_cut_constraints = 0
    if use_cuts:
        pair_v, chan_v, mblock_v = d["pair"], d["chan"], d["mblock"]
        with open(os.path.join(RUNS, "chancuts.json")) as f:
            cc = json.load(f)
        for e in cc["lines"] + cc["planes"]:
            sel = np.flatnonzero((kind == 0) & (pair_v == e["pair"])
                                 & np.isin(chan_v, e["channels"]))
            top = _card_atmost(clauses, (sel + 1).tolist(), e["max"], top)
            n_cut_constraints += 1
        with open(os.path.join(RUNS, "pairs.json")) as f:
            pj = json.load(f)
        for p in range(3):
            key = f"pair{p}_duads_only"
            if pj.get(key, {}).get("status") == "OPTIMAL":
                sel = np.flatnonzero((kind == 0) & (pair_v == p))
                top = _card_atmost(clauses, (sel + 1).tolist(),
                                   pj[key]["opt_weight"] // 4, top)
                n_cut_constraints += 1
        for c in range(15):
            sel = np.flatnonzero((kind == 0) & (chan_v == c))
            top = _card_atmost(clauses, (sel + 1).tolist(), 12, top)
            n_cut_constraints += 1
        for b in range(3):
            sel = np.flatnonzero((kind == 1) & (mblock_v == b))
            top = _card_atmost(clauses, (sel + 1).tolist(), 8, top)
            n_cut_constraints += 1
    meta = {"n_orbits": n, "edge_clauses": len(Ei), "k": k,
            "target_weight": int(target_weight), "n_vars": top,
            "n_clauses": len(clauses), "cut_constraints": n_cut_constraints}
    return clauses, top, meta


def sat_cmd(args):
    """Decision: does a 2^3-invariant 60-free set of weight >= target exist?"""
    from pysat.solvers import Solver
    t0 = time.time()
    d, A = load_model()
    clauses, top, meta = build_cnf_weight_ge(d, A, args.target,
                                             use_cuts=args.cuts)
    log(f"# CNF: {meta}")
    suffix = "_cuts" if args.cuts else ""
    dimacs = os.path.join(RUNS, f"sat_ge{args.target}{suffix}.cnf")
    with open(dimacs, "w") as f:
        f.write(f"p cnf {top} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
    log(f"# wrote {dimacs} ({time.time()-t0:.1f}s)")
    if args.write_only:
        log(f"RESULT section=sat target={args.target} wrote={dimacs} "
            f"write_only=1 seconds={time.time()-t0:.1f}")
        return
    log("# solving with cadical195")
    with Solver(name="cadical195", bootstrap_with=clauses,
                with_proof=True) as s:
        sat = s.solve()
        if sat:
            model = s.get_model()
            chosen = [v - 1 for v in model[:meta["n_orbits"]] if v > 0]
            wsum = int(d["weight"][chosen].sum())
            out = os.path.join(RUNS, f"sat_ge{args.target}_witness.json")
            with open(out, "w") as f:
                json.dump({"orbits": chosen, "weight": wsum}, f)
            log(f"RESULT section=sat target={args.target} answer=SAT "
                f"weight={wsum} witness={out} seconds={time.time()-t0:.1f}")
        else:
            proof = s.get_proof()
            pf = os.path.join(RUNS, f"sat_ge{args.target}.drat")
            with open(pf, "w") as f:
                f.write("\n".join(proof) + "\n")
            log(f"RESULT section=sat target={args.target} answer=UNSAT "
                f"proof={pf} proof_lines={len(proof)} "
                f"seconds={time.time()-t0:.1f}")


def checkcnf_cmd(args):
    """Re-generate the decision CNF from the (re-verified) model and check it
    is identical to the stored artifact runs/a2/sat_ge<target>.cnf.  A stranger
    verifies the UNSAT theorem by:
      1. .venv/bin/python python/tools/channel_opt.py build      (model asserts)
      2. .venv/bin/python python/tools/channel_opt.py selftest   (encoding test)
      3. .venv/bin/python python/tools/channel_opt.py checkcnf --target 497
      4. drat-trim runs/a2/sat_ge497.cnf runs/a2/sat_ge497.drat  -> s VERIFIED
    """
    t0 = time.time()
    d, A = load_model()
    clauses, top, meta = build_cnf_weight_ge(d, A, args.target)
    path = os.path.join(RUNS, f"sat_ge{args.target}.cnf")
    with open(path) as f:
        header = f.readline().split()
        assert header[:2] == ["p", "cnf"]
        assert int(header[2]) == top and int(header[3]) == len(clauses), \
            (header, top, len(clauses))
        for i, line in enumerate(f):
            want = clauses[i]
            got = [int(x) for x in line.split()[:-1]]
            assert got == want, (i, got, want)
        assert i + 1 == len(clauses)
    log(f"RESULT section=checkcnf target={args.target} file={path} "
        f"identical=1 n_vars={top} n_clauses={len(clauses)} "
        f"seconds={time.time()-t0:.1f}")


def selftest_cmd(args):
    """Soundness test of the weighted-cardinality CNF scheme against brute
    force on random weighted graphs (weights in {2,4}), exactly the encoding
    used by `sat`: duplicated duad vars + sortnetwrk atleast."""
    from pysat.solvers import Solver
    rng = np.random.default_rng(20260830)
    n_tests = 0
    for trial in range(40):
        n = int(rng.integers(4, 11))
        A = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    A[i, j] = A[j, i] = True
        w = rng.choice([2, 4], size=n)
        best = 0
        for mask in range(1 << n):
            idx = [i for i in range(n) if mask >> i & 1]
            if not A[np.ix_(idx, idx)].any():
                best = max(best, int(w[idx].sum()))
        for target in range(1, best + 7):
            clauses = [[-i - 1, -j - 1] for i in range(n)
                       for j in range(i + 1, n) if A[i, j]]
            top, lits = n, []
            for i in range(n):
                v = i + 1
                if w[i] == 4:
                    top += 1
                    clauses.append([-v, top])
                    clauses.append([-top, v])
                    lits.extend([v, top])
                else:
                    lits.append(v)
            k = -(-target // 2)
            top = _card_atleast(clauses, lits, k, top)
            with Solver(name="cadical195", bootstrap_with=clauses) as s:
                sat = s.solve()
                if sat:
                    model = s.get_model()
                    chosen = [v - 1 for v in model[:n] if v > 0]
                    assert not A[np.ix_(chosen, chosen)].any()
                    assert int(w[chosen].sum()) >= 2 * k
            assert sat == (2 * k <= best), (trial, target, k, best, sat)
            n_tests += 1
    log(f"RESULT section=selftest tests={n_tests} ok=1")


def maxsat_cmd(args):
    """Exact max-weight independent orbit union via core-guided MaxSAT (RC2).
    Hard clauses: conflict edges (+ optionally entailed cuts).  Soft: each
    orbit var with its weight.  On termination RC2 has PROVED optimality."""
    from pysat.formula import WCNF
    from pysat.examples.rc2 import RC2Stratified
    t0 = time.time()
    d, A = load_model()
    n = len(d["kind"])
    weight = d["weight"]
    wcnf = WCNF()
    Ei, Ej = np.nonzero(np.triu(A, 1))
    for a, b in zip((Ei + 1).tolist(), (Ej + 1).tolist()):
        wcnf.append([-a, -b])
    for oi in range(n):
        wcnf.append([oi + 1], weight=int(weight[oi]))
    total = int(weight.sum())
    log(f"# WCNF: {n} soft, {len(Ei)} hard, total weight {total}")
    with RC2Stratified(wcnf, solver=args.solver, adapt=True, exhaust=True,
                       minz=True, trim=5) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    chosen = [v - 1 for v in model[:n] if v > 0]
    wsum = int(weight[chosen].sum())
    assert wsum == total - cost
    assert not A[np.ix_(chosen, chosen)].any()
    out = os.path.join(RUNS, "maxsat_optimum.json")
    with open(out, "w") as f:
        json.dump({"opt_weight": wsum, "orbits": sorted(chosen)}, f)
    log(f"RESULT section=maxsat OPT={wsum} orbits={len(chosen)} "
        f"witness={out} seconds={time.time()-t0:.1f}")


def cpsat_cmd(args):
    """Exact optimum via OR-Tools CP-SAT (clause learning + LP + symmetry).
    Model: bool per orbit, AtMostOne per cover clique (covers every conflict
    edge), monads-per-block <= 8 (entailed), objective = sum of weights.
    Warm start with the reference 496."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    d, A = load_model()
    cls = load_cliques()
    n = len(d["kind"])
    weight, kind, mblock = d["weight"], d["kind"], d["mblock"]
    # confirm the cover covers every edge
    cov = np.zeros((n, n), dtype=bool)
    for cl in cls:
        cov[np.ix_(cl, cl)] = True
    assert not np.any(A & ~cov), "clique cover incomplete!"
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(n)]
    if getattr(args, "duads_only", False):
        for i in np.flatnonzero(kind == 1):
            m.Add(x[int(i)] == 0)
    for cl in cls:
        m.AddAtMostOne([x[int(i)] for i in cl])
    monads = np.flatnonzero(kind == 1)
    for b in range(3):
        mb = monads[mblock[monads] == b]
        m.Add(sum(x[int(i)] for i in mb) <= 8)
    if args.cuts:
        # entailed cuts: exact subproblem optima proven by `pairs` (CP-SAT
        # OPTIMAL results recorded in runs/a2/pairs.json).  Each cut bounds the
        # weight of the model restricted to a vertex subset by that subset's
        # proven optimum -- valid for any independent set.
        with open(os.path.join(RUNS, "pairs.json")) as f:
            pj = json.load(f)
        pair_v, mblock_v = d["pair"], d["mblock"]

        def wsum(sel):
            return sum(int(weight[int(i)]) * x[int(i)] for i in sel)

        n_cuts = 0
        for p in range(3):
            key = f"pair{p}_duads_only"
            if pj.get(key, {}).get("status") == "OPTIMAL":
                sel = np.flatnonzero((kind == 0) & (pair_v == p))
                m.Add(wsum(sel) <= pj[key]["opt_weight"])
                n_cuts += 1
            key = f"pair{p}_with_monads"
            if pj.get(key, {}).get("status") == "OPTIMAL":
                j, k = PAIR_BLOCKS[p]
                sel = np.concatenate([
                    np.flatnonzero((kind == 0) & (pair_v == p)),
                    np.flatnonzero((kind == 1) & np.isin(mblock_v, [j, k]))])
                m.Add(wsum(sel) <= pj[key]["opt_weight"])
                n_cuts += 1
        for p in range(3):
            for q in range(p + 1, 3):
                key = f"pairs{p}{q}_duads_only"
                if pj.get(key, {}).get("status") == "OPTIMAL":
                    sel = np.flatnonzero((kind == 0) & np.isin(pair_v, [p, q]))
                    m.Add(wsum(sel) <= pj[key]["opt_weight"])
                    n_cuts += 1
        for key, sel in (("duads_only_global", np.flatnonzero(kind == 0)),
                         ("monads_only_global", np.flatnonzero(kind == 1))):
            if pj.get(key, {}).get("status") == "OPTIMAL":
                m.Add(wsum(sel) <= pj[key]["opt_weight"])
                n_cuts += 1
        log(f"# added {n_cuts} entailed subproblem cuts from pairs.json")
        ccp = os.path.join(RUNS, "chancuts.json")
        if os.path.exists(ccp):
            with open(ccp) as f:
                cc = json.load(f)
            chan_v = d["chan"]
            n2 = 0
            for e in cc["lines"] + cc["planes"]:
                sel = np.flatnonzero(
                    (kind == 0) & (pair_v == e["pair"])
                    & np.isin(chan_v, e["channels"]))
                m.Add(sum(x[int(i)] for i in sel) <= e["max"])
                n2 += 1
            log(f"# added {n2} proven channel-subset cuts from chancuts.json")
    m.Maximize(sum(int(weight[i]) * x[i] for i in range(n)))
    ref = set(int(i) for i in d["ref"])
    for i in range(n):
        m.AddHint(x[i], 1 if i in ref else 0)
    sv = cp_model.CpSolver()
    sv.parameters.num_workers = args.workers
    sv.parameters.max_time_in_seconds = args.time
    sv.parameters.log_search_progress = True
    sv.parameters.symmetry_level = 4
    status = sv.Solve(m)
    sname = sv.StatusName(status)
    best = int(sv.ObjectiveValue()) if status in (cp_model.OPTIMAL,
                                                  cp_model.FEASIBLE) else None
    bound = sv.BestObjectiveBound()
    if best is not None:
        chosen = [i for i in range(n) if sv.Value(x[i])]
        assert not A[np.ix_(chosen, chosen)].any()
        assert int(weight[chosen].sum()) == best
        tag = "_duads" if getattr(args, "duads_only", False) else ""
        with open(os.path.join(RUNS, f"cpsat_best{tag}.json"), "w") as f:
            json.dump({"weight": best, "orbits": chosen, "status": sname,
                       "bound": bound}, f)
    log(f"RESULT section=cpsat duads_only={getattr(args, 'duads_only', False)} "
        f"status={sname} best={best} "
        f"upper_bound={bound} seconds={time.time()-t0:.1f}")


def scip_cmd(args):
    """Exact optimum via SCIP (pyscipopt): clique-cover formulation + entailed
    cuts; SCIP's symmetry machinery (orbital fixing) is the point."""
    from pyscipopt import Model, quicksum
    t0 = time.time()
    d, A = load_model()
    cls = load_cliques()
    n = len(d["kind"])
    weight, kind, mblock, pair_v, chan_v = (d["weight"], d["kind"],
                                            d["mblock"], d["pair"], d["chan"])
    mod = Model("a2")
    x = [mod.addVar(vtype="B", name=f"x{i}") for i in range(n)]
    for cl in cls:
        mod.addCons(quicksum(x[int(i)] for i in cl) <= 1)
    monads = np.flatnonzero(kind == 1)
    for b in range(3):
        mb = monads[mblock[monads] == b]
        mod.addCons(quicksum(x[int(i)] for i in mb) <= 8)
    ccp = os.path.join(RUNS, "chancuts.json")
    if os.path.exists(ccp):
        with open(ccp) as f:
            cc = json.load(f)
        for e in cc["lines"] + cc["planes"]:
            sel = np.flatnonzero((kind == 0) & (pair_v == e["pair"])
                                 & np.isin(chan_v, e["channels"]))
            mod.addCons(quicksum(x[int(i)] for i in sel) <= e["max"])
    pjp = os.path.join(RUNS, "pairs.json")
    if os.path.exists(pjp):
        with open(pjp) as f:
            pj = json.load(f)
        for p in range(3):
            key = f"pair{p}_duads_only"
            if pj.get(key, {}).get("status") == "OPTIMAL":
                sel = np.flatnonzero((kind == 0) & (pair_v == p))
                mod.addCons(quicksum(x[int(i)] for i in sel)
                            <= pj[key]["opt_weight"] // 4)
    mod.setObjective(quicksum(float(weight[i]) * x[i] for i in range(n)),
                     "maximize")
    sol = mod.createPartialSol()
    for i in d["ref"]:
        mod.setSolVal(sol, x[int(i)], 1.0)
    mod.addSol(sol)
    mod.setParam("limits/time", args.time)
    mod.optimize()
    status = mod.getStatus()
    best = mod.getObjVal() if mod.getNSols() > 0 else None
    dualb = mod.getDualbound()
    log(f"RESULT section=scip status={status} primal={best} "
        f"dual_bound={dualb} nodes={mod.getNNodes()} "
        f"seconds={time.time()-t0:.1f}")


# ------------------------------------------------------------------ subproblems

def _cpsat_mis(A_sub, weights, time_limit=600.0, workers=4, hint=None,
               channel_of=None, exact12_channels=None, log_progress=False):
    """Exact max-weight IS on a subgraph via CP-SAT.  Returns
    (status_name, best, upper_bound, chosen).  If exact12_channels is given
    (with channel_of), each of those channels must contribute exactly 12."""
    from ortools.sat.python import cp_model
    n = len(weights)
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(n)]
    Ei, Ej = np.nonzero(np.triu(A_sub, 1))
    for a, b in zip(Ei.tolist(), Ej.tolist()):
        m.AddAtMostOne([x[a], x[b]])
    if exact12_channels is not None:
        for c in exact12_channels:
            idx = [i for i in range(n) if channel_of[i] == c]
            m.Add(sum(x[i] for i in idx) == 12)
    m.Maximize(sum(int(weights[i]) * x[i] for i in range(n)))
    if hint is not None:
        for i in range(n):
            m.AddHint(x[i], 1 if i in hint else 0)
    sv = cp_model.CpSolver()
    sv.parameters.num_workers = workers
    sv.parameters.max_time_in_seconds = time_limit
    sv.parameters.log_search_progress = log_progress
    sv.parameters.symmetry_level = 4
    status = sv.Solve(m)
    sname = sv.StatusName(status)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    best = int(sv.ObjectiveValue()) if ok else None
    chosen = [i for i in range(n) if sv.Value(x[i])] if ok else []
    if chosen:
        assert not A_sub[np.ix_(chosen, chosen)].any()
    return sname, best, sv.BestObjectiveBound(), chosen


def _mis_exact_sat(A_sub, weights, lb, log_prefix="", solver="cadical195"):
    """Exact max weight independent set by SAT ascent: try weight >= lb+step
    until UNSAT.  weights must all be equal (uniform) or in {2,4}.
    Returns (opt_weight, best_set, proved)."""
    from pysat.solvers import Solver
    n = len(weights)
    uniform = len(set(int(x) for x in weights)) == 1
    Ei, Ej = np.nonzero(np.triu(A_sub, 1))
    base = [[-int(a) - 1, -int(b) - 1] for a, b in zip(Ei, Ej)]
    best_w, best_set = 0, []
    target = int(lb)
    while True:
        clauses = list(base)
        top = n
        if uniform:
            wunit = int(weights[0])
            k = (target + wunit - 1) // wunit + (
                0 if target % wunit == 0 else 0)
            k = -(-target // wunit)
            lits = list(range(1, n + 1))
        else:
            lits = []
            for oi in range(n):
                v = oi + 1
                if int(weights[oi]) == 4:
                    top += 1
                    clauses.append([-v, top])
                    clauses.append([-top, v])
                    lits.extend([v, top])
                else:
                    lits.append(v)
            k = -(-target // 2)
        top = _card_atleast(clauses, lits, k, top)
        with Solver(name=solver, bootstrap_with=clauses) as s:
            if s.solve():
                model = s.get_model()
                chosen = [v - 1 for v in model[:n] if v > 0]
                w = int(sum(int(weights[c]) for c in chosen))
                assert w >= target
                best_w, best_set = w, chosen
                target = w + (min(int(x) for x in weights))
                log(f"# {log_prefix} SAT at >= {w}; trying {target}")
            else:
                log(f"# {log_prefix} UNSAT at >= {target}; optimum = {best_w}")
                return best_w, best_set, True


def pairs_cmd(args):
    """Exact optima of structured subproblems (CP-SAT):
    - each single pair (960 duads, duad-only)
    - each single pair + the monads of its two blocks
    - each two-pair union (1920 duads, duad-only)
    - duad-only global (2880) and monad-only global (360)
    """
    t0 = time.time()
    d, A = load_model()
    kind, pair, mblock, weight = d["kind"], d["pair"], d["mblock"], d["weight"]
    out = {}

    def solve(name, sel, tl):
        sname, best, bound, chosen = _cpsat_mis(
            A[np.ix_(sel, sel)], weight[sel], time_limit=tl, workers=8)
        out[name] = {"status": sname, "opt_weight": best,
                     "upper_bound": bound, "orbits": len(chosen)}
        log(f"# {name}: {sname} best={best} bound={bound} "
            f"orbits={len(chosen)} ({time.time()-t0:.1f}s)")

    for p in range(3):
        sel = np.flatnonzero((kind == 0) & (pair == p))
        solve(f"pair{p}_duads_only", sel, args.time)
        j, k = PAIR_BLOCKS[p]
        selm = np.flatnonzero((kind == 1) & np.isin(mblock, [j, k]))
        solve(f"pair{p}_with_monads", np.concatenate([sel, selm]), args.time)
    for p in range(3):
        for q in range(p + 1, 3):
            sel = np.flatnonzero((kind == 0) & np.isin(pair, [p, q]))
            solve(f"pairs{p}{q}_duads_only", sel, args.time)
    solve("duads_only_global", np.flatnonzero(kind == 0), args.time)
    solve("monads_only_global", np.flatnonzero(kind == 1), args.time)
    with open(os.path.join(RUNS, "pairs.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"RESULT section=pairs {json.dumps(out)} seconds={time.time()-t0:.1f}")


def chancuts_cmd(args):
    """Prove per-(pair, channel-subset) caps that become entailed cuts:
    for every pair p and every PG(3,2) line {c, c', c xor c'} the duad max is
    16 (= 8 x max cap 2); for every plane (7 channels) it is 32 (= 8 x 4).
    Every individual cap is PROVED by its own CP-SAT solve; results in
    runs/a2/chancuts.json."""
    from itertools import combinations
    t0 = time.time()
    d, A = load_model()
    kind, pair, chan = d["kind"], d["pair"], d["chan"]
    with open(os.path.join(RUNS, "census.json")) as f:
        ch = json.load(f)["channels"]
    id_of = {c: i for i, c in enumerate(ch)}
    lines_pg = []
    for a, b in combinations(range(15), 2):
        cxor = ch[a] ^ ch[b]
        cid = id_of[cxor]
        if a < b < cid:
            lines_pg.append((a, b, cid))
    assert len(lines_pg) == 35
    planes_pg = []
    for h in range(1, 16):  # hyperplanes of GF(2)^4 as functionals... instead:
        pass
    # planes = 2-dim subspace complements? A plane of PG(3,2) = 7 points closed
    # under xor (a 3-dim GF(2)-subspace minus 0).
    from itertools import combinations as comb
    seen = set()
    for a, b, c in comb(range(15), 3):
        s = {ch[a], ch[b], ch[c]}
        if (ch[a] ^ ch[b]) in s or len(s) < 3:
            continue
        span = {0, ch[a], ch[b], ch[c], ch[a] ^ ch[b], ch[a] ^ ch[c],
                ch[b] ^ ch[c], ch[a] ^ ch[b] ^ ch[c]}
        key = frozenset(span)
        if key in seen:
            continue
        seen.add(key)
        planes_pg.append(sorted(id_of[m] for m in span if m))
    assert len(planes_pg) == 15 and all(len(p) == 7 for p in planes_pg)
    out = {"lines": [], "planes": []}
    for p in range(3):
        for L in lines_pg:
            sel = np.flatnonzero((kind == 0) & (pair == p)
                                 & np.isin(chan, list(L)))
            sname, best, bound, _ = _cpsat_mis(
                A[np.ix_(sel, sel)], np.ones(len(sel), dtype=np.int64),
                time_limit=120, workers=2)
            assert sname == "OPTIMAL" and best == 16, (p, L, sname, best)
            out["lines"].append({"pair": p, "channels": list(L), "max": best})
        log(f"# pair {p}: all 35 line caps = 16 PROVED "
            f"({time.time()-t0:.1f}s)")
        for P in planes_pg:
            sel = np.flatnonzero((kind == 0) & (pair == p)
                                 & np.isin(chan, P))
            sname, best, bound, _ = _cpsat_mis(
                A[np.ix_(sel, sel)], np.ones(len(sel), dtype=np.int64),
                time_limit=600, workers=3)
            assert sname == "OPTIMAL", (p, P, sname)
            out["planes"].append({"pair": p, "channels": P, "max": best})
        pm = sorted({e["max"] for e in out["planes"] if e["pair"] == p})
        log(f"# pair {p}: 15 plane caps PROVED, values {pm} "
            f"({time.time()-t0:.1f}s)")
    with open(os.path.join(RUNS, "chancuts.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"RESULT section=chancuts lines={len(out['lines'])} "
        f"planes={len(out['planes'])} "
        f"line_max_values={sorted({e['max'] for e in out['lines']})} "
        f"plane_max_values={sorted({e['max'] for e in out['planes']})} "
        f"seconds={time.time()-t0:.1f}")


def perfect_cmd(args):
    """A4 conjecture: which channel subsets can be simultaneously perfect
    (12 orbits = a full K888 matching in each)?  Exact, via CP-SAT on the duad
    subgraph of the chosen channels with per-channel count = 12."""
    from itertools import combinations
    t0 = time.time()
    d, A = load_model()
    kind, chan = d["kind"], d["chan"]
    duads = np.flatnonzero(kind == 0)

    def feasible(chans):
        sel = duads[np.isin(chan[duads], list(chans))]
        ch_of = chan[sel]
        sname, best, bound, chosen = _cpsat_mis(
            A[np.ix_(sel, sel)], np.ones(len(sel), dtype=np.int64),
            time_limit=args.time, workers=4,
            channel_of=ch_of, exact12_channels=list(chans))
        if sname not in ("OPTIMAL", "INFEASIBLE"):
            return sname  # timeout: report the raw status, never guess
        return sname == "OPTIMAL"

    # the 3 frame channels of S
    ref = d["ref"]
    ref_perfect = sorted({int(c) for c in chan[ref]
                          if (chan[ref] == c).sum() == 12})
    log(f"# S's perfect channels: {ref_perfect}")

    # k-subsets of the 15 channels classified up to the automorphism group of
    # the conflict graph, whose induced action on channels is verified
    # (`perfect --check-aut`, and in the report) to be the full GL(4,2).
    # Orbit invariant used: multiset over 3-subsets of "is a PG(3,2) line",
    # refined by the GF(2)-rank of the subset -- for k <= 5 this separates the
    # GL(4,2) orbits (checked by explicit orbit enumeration below).
    with open(os.path.join(ROOT, "runs", "a2", "census.json")) as f:
        ch = json.load(f)["channels"]

    def orbits_of_ksubsets(k, perms):
        seen = {}
        reps = []
        for sub in combinations(range(15), k):
            key = frozenset(sub)
            if key in seen:
                continue
            orb = set()
            frontier = [tuple(sorted(sub))]
            orb.add(frontier[0])
            while frontier:
                s = frontier.pop()
                for gperm in perms:
                    t = tuple(sorted(gperm[i] for i in s))
                    if t not in orb:
                        orb.add(t)
                        frontier.append(t)
            for t in orb:
                seen[frozenset(t)] = len(reps)
            reps.append((tuple(sorted(sub)), len(orb)))
        return reps

    # generators of the induced channel action, recomputed from the graph
    import pynauty
    n = len(kind)
    adjd = {i: np.flatnonzero(A[i]).tolist() for i in range(n)}
    duadset = set(np.flatnonzero(kind == 0).tolist())
    monset = set(np.flatnonzero(kind == 1).tolist())
    gg = pynauty.Graph(n, directed=False, adjacency_dict=adjd,
                       vertex_coloring=[duadset, monset])
    gens, s1, s2, _, _ = pynauty.autgrp(gg)
    dsel = np.flatnonzero(kind == 0)
    chan_gens = []
    for gp in gens:
        gp = np.array(gp)
        cp = np.full(15, -1)
        for c, c2 in zip(chan[dsel], chan[gp[dsel]]):
            if cp[c] == -1:
                cp[c] = c2
            else:
                assert cp[c] == c2, "generator does not respect channels"
        chan_gens.append(cp)
    aut_order = float(s1) * 10 ** s2
    log(f"# Aut(graph) order {s1}e{s2}; induced channel action generated")

    out = {"S_perfect_channels": ref_perfect, "aut_order": aut_order,
           "by_k": {}}
    max_k = 0
    for k in range(args.kmin, 16):
        reps = orbits_of_ksubsets(k, chan_gens)
        assert sum(cnt for _, cnt in reps) == \
            len(list(combinations(range(15), k)))
        entry = []
        any_feas = False
        for rep, cnt in reps:
            feas = feasible(rep)
            entry.append({"rep": list(rep), "orbit_size": cnt,
                          "perfect_feasible": feas})
            any_feas = any_feas or (feas is True)
            log(f"# k={k} rep={rep} orbit={cnt} perfect_feasible={feas} "
                f"({time.time()-t0:.1f}s)")
        out["by_k"][str(k)] = entry
        if not any_feas:
            all_infeasible = all(e["perfect_feasible"] is False
                                 for e in entry)
            if all_infeasible:
                log(f"# k={k}: NO {k}-subset of channels can be "
                    f"simultaneously perfect -- PROVED (all orbit reps "
                    f"INFEASIBLE; supersets inherit infeasibility)")
            else:
                log(f"# k={k}: no rep PROVED feasible, but "
                    f"{sum(1 for e in entry if e['perfect_feasible'] not in (True, False))} "
                    f"rep(s) timed out UNKNOWN -- k={k} is UNDETERMINED, "
                    f"max_simultaneously_perfect={k-1} is only a LOWER bound")
            max_k = k - 1
            out["k_break_proved"] = all_infeasible
            break
        max_k = k
    out["max_simultaneously_perfect"] = max_k
    with open(os.path.join(RUNS, "perfect.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"RESULT section=perfect max_simultaneously_perfect={max_k} "
        f"S_perfect={ref_perfect} seconds={time.time()-t0:.1f}")


# ------------------------------------------------------ A2 finish: decomposition

def _pg_lines_planes():
    """PG(3,2) lines (35 triples) and planes (15 seven-sets) as channel ids."""
    from itertools import combinations
    with open(os.path.join(RUNS, "census.json")) as f:
        ch = json.load(f)["channels"]
    id_of = {c: i for i, c in enumerate(ch)}
    lines_pg = []
    for a, b in combinations(range(15), 2):
        cid = id_of[ch[a] ^ ch[b]]
        if a < b < cid:
            lines_pg.append((a, b, cid))
    assert len(lines_pg) == 35
    seen, planes_pg = set(), []
    for a, b, c in combinations(range(15), 3):
        s = {ch[a], ch[b], ch[c]}
        if (ch[a] ^ ch[b]) in s or len(s) < 3:
            continue
        span = {ch[a], ch[b], ch[c], ch[a] ^ ch[b], ch[a] ^ ch[c],
                ch[b] ^ ch[c], ch[a] ^ ch[b] ^ ch[c]}
        key = frozenset(span)
        if key not in seen:
            seen.add(key)
            planes_pg.append(tuple(sorted(id_of[m] for m in span)))
    assert len(planes_pg) == 15
    return lines_pg, planes_pg


def _monad_plane_assignment(d, planes_pg):
    """VERIFIED fact: each monad's 14 conflict lines lie in 7 channels that
    form a PG(3,2) plane (2 lines per channel, all in the monad's own block);
    each plane receives exactly 24 monads (8 per block).  Returns
    monad_orbit_index -> plane index."""
    ml = d["monad_lines"]
    bym = {}
    for row in ml:
        bym.setdefault(int(row[0]), set()).add(int(row[1]))
    plane_ix = {frozenset(P): i for i, P in enumerate(planes_pg)}
    out = {}
    for m, chans in bym.items():
        assert frozenset(chans) in plane_ix, "monad channels are not a plane"
        out[m] = plane_ix[frozenset(chans)]
    from collections import Counter
    assert set(Counter(out.values()).values()) == {24}
    return out


def profilecaps_cmd(args):
    """g(k) := exact max #duads in ONE pair with every per-channel count <= k
    (k = 1..8; g(8) = 64 is the proven pair cap).  Valid profile cuts follow:
    for every pair p and every k, sum_c min(m_{p,c}, k) <= g(k), because
    truncating a feasible pair-solution to k per channel stays feasible.
    Each g(k) is PROVED by its own CP-SAT solve (pair 0; all pairs equivalent
    under the verified S3 pair symmetry of Aut)."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    d, A = load_model()
    kind, pair, chan = d["kind"], d["pair"], d["chan"]
    sel = np.flatnonzero((kind == 0) & (pair == 0))
    A_sub = A[np.ix_(sel, sel)]
    ch_of = chan[sel]
    out = {}
    for k in range(1, 8):
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f"x{i}") for i in range(len(sel))]
        Ei, Ej = np.nonzero(np.triu(A_sub, 1))
        for a, b in zip(Ei.tolist(), Ej.tolist()):
            m.AddAtMostOne([x[a], x[b]])
        for c in range(15):
            idx = np.flatnonzero(ch_of == c)
            m.Add(sum(x[int(i)] for i in idx) <= k)
        m.Maximize(sum(x))
        sv = cp_model.CpSolver()
        sv.parameters.num_workers = args.workers
        sv.parameters.max_time_in_seconds = args.time
        sv.parameters.symmetry_level = 4
        status = sv.Solve(m)
        sname = sv.StatusName(status)
        best = int(sv.ObjectiveValue()) if sname in ("OPTIMAL", "FEASIBLE") \
            else None
        out[str(k)] = {"status": sname, "best": best,
                       "bound": sv.BestObjectiveBound()}
        log(f"# g({k}) [pair 0, per-channel <= {k}]: {sname} best={best} "
            f"bound={sv.BestObjectiveBound()} ({time.time()-t0:.1f}s)")
    out["8"] = {"status": "OPTIMAL", "best": 64, "bound": 64.0,
                "note": "= proven pair cap (pairs.json)"}
    with open(os.path.join(RUNS, "profilecaps.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"RESULT section=profilecaps "
        f"g={[out[str(k)]['best'] for k in range(1, 9)]} "
        f"statuses={[out[str(k)]['status'] for k in range(1, 9)]} "
        f"seconds={time.time()-t0:.1f}")


def _sub_cuts_pairs2(d, sel, clauses, top):
    """Entailed cuts for the two-pair (0,1) duad instance, all previously
    PROVED: per-(pair,channel) <= 8 [rook], per-channel across both pairs <= 8
    [distinct lines of the shared block's 8-frame, structure section],
    per-pair PG-line <= 16 and plane <= 32 [chancuts.json], per-pair <= 64
    [pairs.json].  Literals are positions in sel (1-based)."""
    pair_v, chan_v = d["pair"][sel], d["chan"][sel]
    lines_pg, planes_pg = _pg_lines_planes()
    ncuts = 0
    pos = np.arange(1, len(sel) + 1)
    for c in range(15):
        idx = pos[chan_v == c]
        top = _card_atmost(clauses, idx.tolist(), 8, top)
        ncuts += 1
        for p in (0, 1):
            idx2 = pos[(chan_v == c) & (pair_v == p)]
            top = _card_atmost(clauses, idx2.tolist(), 8, top)
            ncuts += 1
    for p in (0, 1):
        for L in lines_pg:
            idx = pos[(pair_v == p) & np.isin(chan_v, list(L))]
            top = _card_atmost(clauses, idx.tolist(), 16, top)
            ncuts += 1
        for P in planes_pg:
            idx = pos[(pair_v == p) & np.isin(chan_v, list(P))]
            top = _card_atmost(clauses, idx.tolist(), 32, top)
            ncuts += 1
        idx = pos[pair_v == p]
        top = _card_atmost(clauses, idx.tolist(), 64, top)
        ncuts += 1
    return top, ncuts


def subcnf_cmd(args):
    """Write decision CNFs for the decomposition subproblems (kissat food).

    --kind pairs2 --target K : duads of pairs {0,1} (1920 vars), independence
        + proven entailed cuts; SAT iff two-pair duad count >= K orbits.
        UNSAT at K = 97 proves the two-pair optimum is 96 orbits (= 384).
    --kind plane --target K : duads whose channel lies in plane 0 of PG(3,2)
        (1344 vars, all 3 pairs) + cuts; SAT iff plane duad count >= K.
        (All 15 planes are Aut-equivalent: induced channel action = GL(4,2).)
    --kind perfect9 --rep R : duads on the R-th GL(4,2)-orbit representative
        9-subset of channels (from perfect.json), each channel forced to
        EXACTLY 12 orbits.  UNSAT for all 5 reps proves that no 9 channels
        can be simultaneously perfect."""
    t0 = time.time()
    d, A = load_model()
    kind, pair, chan = d["kind"], d["pair"], d["chan"]
    if args.kind == "pairs2":
        sel = np.flatnonzero((kind == 0) & np.isin(pair, [0, 1]))
        name = f"sub_pairs2_ge{args.target}"
    elif args.kind == "plane":
        _, planes_pg = _pg_lines_planes()
        sel = np.flatnonzero((kind == 0) & np.isin(chan, planes_pg[0]))
        name = f"sub_plane_ge{args.target}"
    elif args.kind == "perfect9":
        with open(os.path.join(RUNS, "perfect.json")) as f:
            reps = [e["rep"] for e in json.load(f)["by_k"]["9"]]
        chans = reps[args.rep]
        sel = np.flatnonzero((kind == 0) & np.isin(chan, chans))
        name = f"sub_perfect9_rep{args.rep}"
    elif args.kind == "gk":
        sel = np.flatnonzero((kind == 0) & (pair == 0))
        name = f"sub_g{args.k}_ge{args.target}"
    elif args.kind == "pairs2plane":
        _, planes_pg = _pg_lines_planes()
        sel = np.flatnonzero((kind == 0) & np.isin(pair, [0, 1])
                             & np.isin(chan, planes_pg[0]))
        name = f"sub_pairs2plane_ge{args.target}"
    else:
        raise SystemExit(f"unknown --kind {args.kind}")
    A_sub = A[np.ix_(sel, sel)]
    clauses = []
    Ei, Ej = np.nonzero(np.triu(A_sub, 1))
    for a, b in zip((Ei + 1).tolist(), (Ej + 1).tolist()):
        clauses.append([-a, -b])
    top = len(sel)
    pos = np.arange(1, len(sel) + 1)
    if args.kind == "pairs2":
        top, ncuts = _sub_cuts_pairs2(d, sel, clauses, top)
        top = _card_atleast(clauses, pos.tolist(), args.target, top)
    elif args.kind == "plane":
        chan_v, pair_v = chan[sel], pair[sel]
        ncuts = 0
        for c in sorted(set(chan_v.tolist())):
            idx = pos[chan_v == c]
            top = _card_atmost(clauses, idx.tolist(), 12, top)  # channel<=12
            ncuts += 1
            for p in range(3):
                for q in range(p + 1, 3):
                    idx2 = pos[(chan_v == c) & np.isin(pair_v, [p, q])]
                    top = _card_atmost(clauses, idx2.tolist(), 8, top)
                    ncuts += 1  # shared-block frame
        for p in range(3):
            idx = pos[pair_v == p]
            top = _card_atmost(clauses, idx.tolist(), 32, top)  # plane cap
            ncuts += 1
        top = _card_atleast(clauses, pos.tolist(), args.target, top)
    elif args.kind == "gk":
        chan_v = chan[sel]
        ncuts = 0
        for c in range(15):
            idx = pos[chan_v == c].tolist()
            top = _card_atmost(clauses, idx, args.k, top)
            ncuts += 1
        top = _card_atleast(clauses, pos.tolist(), args.target, top)
    elif args.kind == "pairs2plane":
        chan_v, pair_v = chan[sel], pair[sel]
        ncuts = 0
        for c in sorted(set(chan_v.tolist())):
            idx = pos[chan_v == c].tolist()
            top = _card_atmost(clauses, idx, 8, top)  # shared-block frame
            ncuts += 1
        for p in (0, 1):
            idx = pos[pair_v == p].tolist()
            top = _card_atmost(clauses, idx, 32, top)  # per-pair plane cap
            ncuts += 1
        top = _card_atleast(clauses, pos.tolist(), args.target, top)
    else:  # perfect9
        chan_v = chan[sel]
        ncuts = 0
        for c in sorted(set(chan_v.tolist())):
            idx = pos[chan_v == c].tolist()
            top = _card_atmost(clauses, idx, 12, top)
            top = _card_atleast(clauses, idx, 12, top)
            ncuts += 2
    path = os.path.join(RUNS, f"{name}.cnf")
    with open(path, "w") as f:
        f.write(f"p cnf {top} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
    log(f"RESULT section=subcnf kind={args.kind} n={len(sel)} vars={top} "
        f"clauses={len(clauses)} cuts={ncuts} wrote={path} "
        f"seconds={time.time()-t0:.1f}")


def plane_cmd(args):
    """CP-SAT on the plane subproblem: duads whose channel lies in plane 0
    (1344) plus, with --monads, the 24 monads assigned to that plane (the
    only monads whose duad conflicts touch these channels).  Objective =
    4*duads + 2*monads (true weight) or #duads with --orbits."""
    t0 = time.time()
    d, A = load_model()
    kind, chan = d["kind"], d["chan"]
    _, planes_pg = _pg_lines_planes()
    P0 = planes_pg[0]
    sel_d = np.flatnonzero((kind == 0) & np.isin(chan, P0))
    sel = sel_d
    if args.monads:
        m2p = _monad_plane_assignment(d, planes_pg)
        sel_m = np.array(sorted(m for m, pl in m2p.items() if pl == 0))
        sel = np.concatenate([sel_d, sel_m])
    w = np.where(kind[sel] == 0, 4, 2).astype(np.int64)
    if args.orbits:
        w = np.ones(len(sel), dtype=np.int64)
    # warm start from the reference S restricted to the plane
    refset = set(int(i) for i in d["ref"])
    hint = set(i for i, o in enumerate(sel) if int(o) in refset)
    sname, best, bound, chosen = _cpsat_mis(
        A[np.ix_(sel, sel)], w, time_limit=args.time, workers=args.workers,
        hint=hint, log_progress=True)
    log(f"RESULT section=plane monads={bool(args.monads)} "
        f"orbits_obj={bool(args.orbits)} status={sname} best={best} "
        f"upper_bound={bound} chosen={len(chosen)} "
        f"seconds={time.time()-t0:.1f}")


def monadkill_cmd(args):
    """Exact monad->line kill function: u(k) := minimum, over independent
    k-subsets of one block's 120 monad orbits, of the number of distinct
    block lines they conflict with (each monad conflicts with exactly 14).
    Since a selected monad excludes every duad using one of its conflict
    lines, and the duads touching block b use pairwise distinct b-lines,
    (#duads touching b) <= 120 - u(#monads in b) is a PROVED cut for every
    block.  Each u(k) is a CP-SAT MINIMISATION run to optimality."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    d, _ = load_model()
    l = np.load(LINES)
    mlc = l["monad_line_conf"]
    kind = d["kind"]
    mon_idx = np.flatnonzero(kind == 1)
    mb = d["mblock"][mon_idx]
    ei, ej = d["edges_i"], d["edges_j"]
    mm = (kind[ei] == 1) & (kind[ej] == 1)
    pos_of = {int(o): i for i, o in enumerate(mon_idx)}
    mons = np.flatnonzero(mb == 0)  # block 0 (blocks equivalent under Aut)
    lset = sorted(set(np.flatnonzero(mlc[mons].any(0)).tolist()))
    lix = {li: i for i, li in enumerate(lset)}
    assert len(lset) == 120
    edges = [(pos_of[int(a)], pos_of[int(b)]) for a, b in
             zip(ei[mm], ej[mm]) if pos_of[int(a)] in set(mons.tolist())
             and pos_of[int(b)] in set(mons.tolist())]
    out = {}
    for k in range(1, 9):
        m = cp_model.CpModel()
        x = {int(i): m.NewBoolVar(f"x{i}") for i in mons.tolist()}
        y = [m.NewBoolVar(f"y{j}") for j in range(120)]
        for a, b in edges:
            m.AddAtMostOne([x[a], x[b]])
        for i in mons.tolist():
            for li in np.flatnonzero(mlc[i]).tolist():
                m.AddImplication(x[i], y[lix[li]])
        m.Add(sum(x.values()) == k)
        m.Minimize(sum(y))
        sv = cp_model.CpSolver()
        sv.parameters.num_workers = args.workers
        sv.parameters.max_time_in_seconds = args.time
        status = sv.Solve(m)
        sname = sv.StatusName(status)
        best = int(sv.ObjectiveValue()) if sname in ("OPTIMAL", "FEASIBLE") \
            else None
        out[str(k)] = {"status": sname, "min_killed": best,
                       "bound": sv.BestObjectiveBound()}
        log(f"# u({k}) = {best} [{sname}, lb={sv.BestObjectiveBound()}] "
            f"({time.time()-t0:.1f}s)")
    with open(os.path.join(RUNS, "monadkill.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"RESULT section=monadkill "
        f"u={[out[str(k)]['min_killed'] for k in range(1, 9)]} "
        f"statuses={[out[str(k)]['status'] for k in range(1, 9)]} "
        f"seconds={time.time()-t0:.1f}")


def subopt_cmd(args):
    """Generic exact duad subproblem: restrict to --pairs (comma list) and a
    channel subset (--channels line0 | plane0 | affine0 | all), maximise the
    orbit count by CP-SAT, append the outcome to runs/a2/subopt.json.  Any
    OPTIMAL value V yields the valid cut: for every Aut-image of the shape
    (all pair-pairs x all planes/lines, by the verified S3 x GL(4,2)
    transitivity), the corresponding profile sum is <= V."""
    t0 = time.time()
    d, A = load_model()
    kind, pair, chan = d["kind"], d["pair"], d["chan"]
    lines_pg, planes_pg = _pg_lines_planes()
    prs = [int(x) for x in args.pairs.split(",")]
    if args.channels == "line0":
        chans = list(lines_pg[0])
    elif args.channels == "plane0":
        chans = list(planes_pg[0])
    elif args.channels == "affine0":
        chans = sorted(set(range(15)) - set(planes_pg[0]))
    elif args.channels == "all":
        chans = list(range(15))
    else:
        raise SystemExit(f"unknown --channels {args.channels}")
    sel = np.flatnonzero((kind == 0) & np.isin(pair, prs)
                         & np.isin(chan, chans))
    refset = set(int(i) for i in d["ref"])
    hint = set(i for i, o in enumerate(sel) if int(o) in refset)
    sname, best, bound, chosen = _cpsat_mis(
        A[np.ix_(sel, sel)], np.ones(len(sel), dtype=np.int64),
        time_limit=args.time, workers=args.workers, hint=hint,
        log_progress=True)
    key = f"pairs{args.pairs}_{args.channels}"
    ledger = os.path.join(RUNS, "subopt.json")
    j = json.load(open(ledger)) if os.path.exists(ledger) else {}
    j[key] = {"status": sname, "best": best, "bound": bound,
              "n": len(sel), "seconds": round(time.time() - t0, 1)}
    with open(ledger, "w") as f:
        json.dump(j, f, indent=1)
    log(f"RESULT section=subopt key={key} status={sname} best={best} "
        f"bound={bound} n={len(sel)} seconds={time.time()-t0:.1f}")


def combine_cmd(args):
    """The certified profile bound: maximise 4*duads + 2*monads over
    channel-occupancy profiles m[p][c] (p = pair, c = channel) and per-block
    monad counts, subject to EVERY constraint proved so far.  Each ingredient
    is only used if its artefact records a PROVED status, so the resulting
    optimum is a valid upper bound for the full problem: any independent
    orbit set induces a feasible profile.  Ingredients:

      [structural, VERIFIED]  m[p][c] <= 8 (rook rows/cols);
        m[p][c] + m[q][c] <= 8 per channel and pair-pair (the shared block's
        8-frame: same-channel duads of both pairs use distinct lines of it);
        duads touching block b <= 120 - u(monads in b) (kill function).
      [chancuts.json, PROVED] per-pair PG-line <= 16, plane <= 32.
      [pairs.json, PROVED] per-pair <= 64; pair + its blocks' monads <= 288
        (weight); monads per block <= 8.
      [profilecaps.json, PROVED entries] sum_c min(m[p][c], k) <= g(k).
      [--pairs2 P] two-pair orbit cap (pass only if PROVED, e.g. kissat
        UNSAT of sub_pairs2_ge{P+1}).
      [--plane Q] global plane orbit cap (pass only if PROVED).
      [--perfect8] at most 8 channels at occupancy 12 (pass only if all five
        sub_perfect9 instances are UNSAT)."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    lines_pg, planes_pg = _pg_lines_planes()
    m = cp_model.CpModel()
    # unary/order encoding: z[p][c][k] = [m_{p,c} >= k+1], k = 0..7, so that
    # m_{p,c} = sum_k z and sum_c min(m_{p,c}, K) = sum_c sum_{k<K} z -- every
    # cut is linear in z.
    z = [[[m.NewBoolVar(f"z{p}_{c}_{k}") for k in range(8)]
          for c in range(15)] for p in range(3)]
    for p in range(3):
        for c in range(15):
            for k in range(7):
                m.AddImplication(z[p][c][k + 1], z[p][c][k])
    mv = [[sum(z[p][c]) for c in range(15)] for p in range(3)]
    mon = [m.NewIntVar(0, 8, f"mon{b}") for b in range(3)]
    used = []
    # per-channel shared-frame caps
    for c in range(15):
        for p in range(3):
            for q in range(p + 1, 3):
                m.Add(mv[p][c] + mv[q][c] <= 8)
    used.append("channel pairwise <= 8 (structural)")
    # per-pair line/plane/total caps
    for p in range(3):
        for L in lines_pg:
            m.Add(sum(mv[p][c] for c in L) <= 16)
        for P in planes_pg:
            m.Add(sum(mv[p][c] for c in P) <= 32)
        m.Add(sum(mv[p]) <= 64)
    used.append("pair line<=16 plane<=32 total<=64 (chancuts+pairs PROVED)")
    # pair + its two blocks' monads <= 288 weight
    for p in range(3):
        j, k = PAIR_BLOCKS[p]
        m.Add(4 * sum(mv[p]) + 2 * (mon[j] + mon[k]) <= 288)
    used.append("pair+monads <= 288 (pairs PROVED)")
    # g(k) truncation cuts
    pcp = os.path.join(RUNS, "profilecaps.json")
    gvals = {}
    if os.path.exists(pcp):
        with open(pcp) as f:
            g = json.load(f)
        # a timed-out run's best_bound is still a PROVED upper bound on
        # g(k); monotonise since g is nondecreasing in k
        ub = {}
        for k in range(8, 0, -1):
            e = g.get(str(k), {})
            if not e:
                continue
            b = int(e["bound"])
            if k < 8 and (k + 1) in ub:
                b = min(b, ub[k + 1][0])
            tag = "PROVED optimal" if e.get("status") == "OPTIMAL" \
                else "PROVED dual bound (timeout)"
            ub[k] = (b, tag)
        for k in range(1, 8):
            if k in ub and ub[k][0] < 15 * k and ub[k][0] < 64:
                gvals[k] = ub[k]
    if args.hypo_g:
        for tok in args.hypo_g.split(","):
            k, v = tok.split(":")
            gvals[int(k)] = (int(v), "caller-asserted PROOF")
    if gvals:
        for k, (val, tag) in sorted(gvals.items()):
            for p in range(3):
                m.Add(sum(z[p][c][j] for c in range(15)
                          for j in range(k)) <= val)
            used.append(f"g({k}) <= {val} ({tag})")
    # kill function
    mkp = os.path.join(RUNS, "monadkill.json")
    if os.path.exists(mkp):
        with open(mkp) as f:
            u = json.load(f)
        if all(u[str(k)]["status"] == "OPTIMAL" for k in range(1, 9)):
            utab = [0] + [u[str(k)]["min_killed"] for k in range(1, 9)]
            for b in range(3):
                kv = m.NewIntVar(0, 56, f"kill{b}")
                m.AddElement(mon[b], utab, kv)
                touching = sum(sum(mv[p]) for p in range(3)
                               if b in PAIR_BLOCKS[p])
                m.Add(touching <= 120 - kv)
            used.append("block kill: duads_b <= 120 - u(mon_b) (PROVED)")
    if args.pairs2 is not None:
        for p in range(3):
            for q in range(p + 1, 3):
                m.Add(sum(mv[p]) + sum(mv[q]) <= args.pairs2)
        used.append(f"two-pair <= {args.pairs2} orbits (caller-asserted PROOF)")
    if args.plane is not None:
        for P in planes_pg:
            m.Add(sum(mv[p][c] for p in range(3) for c in P) <= args.plane)
        used.append(f"plane <= {args.plane} orbits (caller-asserted PROOF)")
    if args.pairs2plane is not None:
        for P in planes_pg:
            for p in range(3):
                for q in range(p + 1, 3):
                    m.Add(sum(mv[p][c] + mv[q][c] for c in P)
                          <= args.pairs2plane)
        used.append(f"two-pair plane <= {args.pairs2plane} orbits "
                    f"(caller-asserted PROOF)")
    if args.affine2 is not None:
        affs = [sorted(set(range(15)) - set(P)) for P in planes_pg]
        for Aset in affs:
            for p in range(3):
                for q in range(p + 1, 3):
                    m.Add(sum(mv[p][c] + mv[q][c] for c in Aset)
                          <= args.affine2)
        used.append(f"two-pair affine <= {args.affine2} orbits "
                    f"(caller-asserted PROOF)")
    if args.perfect8:
        inds = []
        for c in range(15):
            zz = m.NewBoolVar(f"perf{c}")
            tot = sum(mv[p][c] for p in range(3))
            m.Add(tot == 12).OnlyEnforceIf(zz)
            m.Add(tot <= 11).OnlyEnforceIf(zz.Not())
            inds.append(zz)
        m.Add(sum(inds) <= 8)
        used.append("at most 8 perfect channels (caller-asserted PROOF)")
    m.Maximize(4 * sum(sum(mv[p]) for p in range(3)) + 2 * sum(mon))
    sv = cp_model.CpSolver()
    sv.parameters.num_workers = args.workers
    sv.parameters.max_time_in_seconds = args.time
    status = sv.Solve(m)
    sname = sv.StatusName(status)
    best = int(sv.ObjectiveValue())
    ub = sv.BestObjectiveBound()
    prof = [[sv.Value(mv[p][c]) for c in range(15)] for p in range(3)]
    for line in used:
        log(f"#   ingredient: {line}")
    log(f"# maximising profile: pair sums = "
        f"{[sum(r) for r in prof]}, monads = {[sv.Value(x) for x in mon]}")
    log(f"# NOTE: the CERTIFIED upper bound for the original problem is "
        f"best_bound={ub} (equals the incumbent only when OPTIMAL)")
    log(f"RESULT section=combine status={sname} profile_incumbent={best} "
        f"certified_bound={ub} ingredients={len(used)} "
        f"seconds={time.time()-t0:.1f}")


# ------------------------------------------------------------------ census print

def census(args):
    with open(os.path.join(RUNS, "census.json")) as f:
        print(json.dumps(json.load(f), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("census")
    sub.add_parser("structure")
    sub.add_parser("cliques")
    sub.add_parser("lp")
    p_milp = sub.add_parser("milp")
    p_milp.add_argument("--time", type=float, default=7200.0)
    p_sat = sub.add_parser("sat")
    p_sat.add_argument("--target", type=int, default=497)
    p_sat.add_argument("--cuts", action="store_true")
    p_sat.add_argument("--write-only", action="store_true")
    p_pairs = sub.add_parser("pairs")
    p_pairs.add_argument("--time", type=float, default=900.0)
    p_perf = sub.add_parser("perfect")
    p_perf.add_argument("--time", type=float, default=600.0)
    p_perf.add_argument("--kmin", type=int, default=2)
    sub.add_parser("chancuts")
    p_scip = sub.add_parser("scip")
    p_scip.add_argument("--time", type=float, default=7200.0)
    p_ms = sub.add_parser("maxsat")
    p_ms.add_argument("--solver", default="cadical195")
    sub.add_parser("selftest")
    p_cc = sub.add_parser("checkcnf")
    p_cc.add_argument("--target", type=int, default=497)
    p_cp = sub.add_parser("cpsat")
    p_cp.add_argument("--time", type=float, default=7200.0)
    p_cp.add_argument("--workers", type=int, default=10)
    p_cp.add_argument("--cuts", action="store_true")
    p_cp.add_argument("--duads-only", action="store_true")
    p_pc = sub.add_parser("profilecaps")
    p_pc.add_argument("--time", type=float, default=1200.0)
    p_pc.add_argument("--workers", type=int, default=8)
    p_sc = sub.add_parser("subcnf")
    p_sc.add_argument("--kind", required=True,
                      choices=["pairs2", "plane", "perfect9", "gk", "pairs2plane"])
    p_sc.add_argument("--target", type=int, default=97)
    p_sc.add_argument("--rep", type=int, default=0)
    p_sc.add_argument("--k", type=int, default=2)
    p_pl = sub.add_parser("plane")
    p_pl.add_argument("--time", type=float, default=3600.0)
    p_pl.add_argument("--workers", type=int, default=8)
    p_pl.add_argument("--monads", action="store_true")
    p_pl.add_argument("--orbits", action="store_true")
    p_mk = sub.add_parser("monadkill")
    p_mk.add_argument("--time", type=float, default=600.0)
    p_mk.add_argument("--workers", type=int, default=4)
    p_cb = sub.add_parser("combine")
    p_cb.add_argument("--time", type=float, default=300.0)
    p_cb.add_argument("--workers", type=int, default=4)
    p_cb.add_argument("--pairs2", type=int, default=None)
    p_cb.add_argument("--plane", type=int, default=None)
    p_cb.add_argument("--pairs2plane", type=int, default=None)
    p_cb.add_argument("--affine2", type=int, default=None)
    p_cb.add_argument("--hypo-g", default=None)
    p_cb.add_argument("--perfect8", action="store_true")
    p_so = sub.add_parser("subopt")
    p_so.add_argument("--pairs", default="0,1")
    p_so.add_argument("--channels", default="plane0")
    p_so.add_argument("--time", type=float, default=1800.0)
    p_so.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"build": build, "census": census, "structure": structure,
     "cliques": cliques_cmd, "lp": lp_cmd, "milp": milp_cmd,
     "sat": sat_cmd, "pairs": pairs_cmd, "perfect": perfect_cmd,
     "maxsat": maxsat_cmd, "selftest": selftest_cmd,
     "checkcnf": checkcnf_cmd, "cpsat": cpsat_cmd,
     "chancuts": chancuts_cmd, "scip": scip_cmd,
     "profilecaps": profilecaps_cmd, "subcnf": subcnf_cmd,
     "plane": plane_cmd, "monadkill": monadkill_cmd,
     "combine": combine_cmd, "subopt": subopt_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()

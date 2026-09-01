#!/usr/bin/env python3
"""T1.3 (conversion half): map external Leech-subset data into our coordinates.

Both external repositories (PackingStar, Kallal-Kan-Wang) use the sqrt8-integer
scaling of README section 1.2 but a different labelling of the 24 coordinates:
their 759 octads share no octad with our cyclic Golay code under identity
coordinates (docs/reports/T1.3-fetch.md section 4).  Any two Steiner systems
S(5,8,24) are isomorphic, so a coordinate permutation pi carrying their octad
system onto ours exists; this tool finds one by backtracking, checks that it
maps their whole 196560-vector set onto ours (falling back to a Golay-codeword
sign pattern if it does not), records the monomial map in a JSON file and
applies it to any external set.

Map convention (data/external/coordinate_map.json):

    ours[perm[i]] = signs[i] * theirs[i]        for i = 0..23

Subcommands
-----------
  find-map   [--ref FILE] [--out JSON]           find and verify the map
  convert    (--npy F | --txt F) [--map JSON] [--out OUT] [--label TEXT] [--sort]
                                                  apply the map; S text format to
                                                  stdout or --out
  verify     FILE...                              independent check: rows in C,
                                                  distinct, norm 32, Gram <= 8,
                                                  antipodal closure
  crosses    FILE [--budget SEC]                  cross structure: maximal cliques
                                                  of the orthogonality graph on
                                                  antipodal classes, exact cover
                                                  by 8-cliques + a 24-clique
  all        [--budget SEC]                       the T1.3 deliverables: find-map,
                                                  convert 496 and 488, verify, crosses

Reuse for T4.2 (the 59 KKW sets, the 52 distinct PackingStar S_i):

    convert_external.py convert --txt data/external/Kissing-Numbers/S_7.txt --out runs/x/S_07.txt

The PackingStar S_i for n = 26..31 are implicit in the unit-vector configurations;
extract them with the decomposition in python/tools/inspect_external.py
(decompose_config), save each as .npy or text, and run `convert` on it.

Uses only numpy and the independent Python reference python/kiss_ref (never the
C++ generator's output), see docs/design.md section 2.4.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from kiss_ref.golay import golay_codewords, octads  # noqa: E402
from kiss_ref.leech import is_lattice_vector_rows, leech_min_vectors  # noqa: E402

EXT = os.path.join(ROOT, "data", "external")
PS_DIR = os.path.join(EXT, "PackingStar")
KKW_DIR = os.path.join(EXT, "Kissing-Numbers")
DEFAULT_REF = os.path.join(PS_DIR, "25D-31D", "Si_configurations", "24D_196560_coordinates.npy")
DEFAULT_MAP = os.path.join(EXT, "coordinate_map.json")
S496_SRC = os.path.join(PS_DIR, "25D-31D", "Si_configurations", "24D_496_Si.npy")
S488_SRC = os.path.join(KKW_DIR, "S_1.txt")
DIM = 24
N_MIN = 196560


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo_dir: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def load_external(path: str, kind: str | None = None) -> np.ndarray:
    """Load an external vector list (.npy of ints/integral floats, or text) as int64 (n, 24)."""
    if kind is None:
        kind = "npy" if path.endswith(".npy") else "txt"
    if kind == "npy":
        a = np.load(path)
        if not np.issubdtype(a.dtype, np.integer):
            r = np.rint(a)
            if np.abs(a - r).max() != 0:
                raise SystemExit(f"{path}: non-integral entries (max residual {np.abs(a - r).max():.3g})")
            a = r
        a = a.astype(np.int64)
    else:
        a = read_set(path)
    if a.ndim != 2 or a.shape[1] != DIM:
        raise SystemExit(f"{path}: expected (n, 24), got {a.shape}")
    return a


def read_set(path: str) -> np.ndarray:
    """S text format (PLAN 2.2): 24 ints per line, '#' comments, blank lines ignored."""
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            s = line.split("#", 1)[0].strip()
            if not s:
                continue
            vals = s.split()
            if len(vals) != DIM:
                raise SystemExit(f"{path}:{ln}: expected 24 values, got {len(vals)}")
            rows.append([int(v) for v in vals])
    return np.array(rows, dtype=np.int64).reshape(-1, DIM)


def write_set(path_or_none, rows: np.ndarray, header_lines: list[str]) -> None:
    out = sys.stdout if path_or_none is None else open(path_or_none, "w")
    try:
        for h in header_lines:
            out.write(f"# {h}\n")
        for r in rows:
            out.write(" ".join(str(int(v)) for v in r) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


def row_set(a: np.ndarray) -> set[tuple[int, ...]]:
    return set(map(tuple, a.tolist()))


def our_c_set() -> set[tuple[int, ...]]:
    return row_set(leech_min_vectors().astype(np.int64))


def octad_masks(vectors: np.ndarray) -> list[int]:
    """Supports of the (+-2^8, 0^16)-shape rows as sorted 24-bit masks."""
    oc = vectors[np.abs(vectors).max(axis=1) == 2]
    masks = ((oc != 0).astype(np.int64) << np.arange(DIM)).sum(axis=1)
    return sorted(set(int(m) for m in masks.tolist()))


def apply_map(vectors: np.ndarray, perm: list[int], signs: list[int]) -> np.ndarray:
    """ours[perm[i]] = signs[i] * theirs[i]."""
    v = np.asarray(vectors, dtype=np.int64)
    out = np.zeros_like(v)
    out[:, np.asarray(perm)] = v * np.asarray(signs, dtype=np.int64)[None, :]
    return out


# ----------------------------------------------------------------------------
# find-map
# ----------------------------------------------------------------------------
def find_octad_permutation(their_octads: list[int], our_octads: list[int]) -> tuple[list[int], int]:
    """Backtracking: perm[their coordinate] = our coordinate mapping octads onto octads.

    Prune: any 5 points of S(5,8,24) lie in a unique octad, so as soon as 5 points
    of one of their octads are assigned, the unique our-octad through their images
    must contain the images of all other assigned points of that octad.
    Returns (perm, nodes visited).
    """
    if len(their_octads) != 759 or len(our_octads) != 759:
        raise SystemExit(f"need 759 octads on both sides, got {len(their_octads)} / {len(our_octads)}")
    five: dict[frozenset[int], int] = {}
    for o in our_octads:
        pts = [i for i in range(DIM) if o >> i & 1]
        for c in itertools.combinations(pts, 5):
            five[frozenset(c)] = o
    their_pts = [[i for i in range(DIM) if o >> i & 1] for o in their_octads]
    oct_by_pt = [[k for k, pts in enumerate(their_pts) if i in pts] for i in range(DIM)]

    # assignment order: greedily pick the point lying on the most octads that already
    # have >= 4 assigned points, so the 5-point constraint kicks in as early as possible
    order: list[int] = []
    assigned: set[int] = set()
    while len(order) < DIM:
        best = None
        for i in range(DIM):
            if i in assigned:
                continue
            score = sum(1 for pts in their_pts if i in pts and len(assigned.intersection(pts)) >= 4)
            if best is None or score > best[0]:
                best = (score, i)
        order.append(best[1])
        assigned.add(best[1])

    perm = [-1] * DIM
    used = [False] * DIM
    nodes = 0

    def consistent(i: int) -> bool:
        for k in oct_by_pt[i]:
            img = [perm[p] for p in their_pts[k] if perm[p] >= 0]
            if len(img) >= 5:
                o = five[frozenset(img[:5])]
                if any(not (o >> q & 1) for q in img[5:]):
                    return False
        return True

    def bt(d: int) -> bool:
        nonlocal nodes
        if d == DIM:
            return True
        i = order[d]
        for q in range(DIM):
            if used[q]:
                continue
            perm[i] = q
            used[q] = True
            nodes += 1
            if consistent(i) and bt(d + 1):
                return True
            perm[i] = -1
            used[q] = False
        return False

    if not bt(0):
        raise SystemExit("no coordinate permutation maps their octads onto ours (impossible for two S(5,8,24))")
    img = sorted({sum(1 << perm[p] for p in pts) for pts in their_pts})
    assert img == sorted(our_octads), "octad image check failed"
    return perm, nodes


def cmd_find_map(args) -> dict:
    t0 = time.time()
    ref = load_external(args.ref)
    if len(ref) != N_MIN:
        raise SystemExit(f"{args.ref}: expected {N_MIN} rows, got {len(ref)}")
    their = octad_masks(ref)
    ours = octads()
    common = len(set(their) & set(ours))
    print(f"their octads: {len(their)}, ours: {len(ours)}, common under identity: {common}")
    perm, nodes = find_octad_permutation(their, ours)
    print(f"permutation found: perm={perm} (backtracking nodes={nodes})")

    C = our_c_set()
    signs = [1] * DIM
    mapped = apply_map(ref, perm, signs)
    ok = row_set(mapped) == C
    print(f"perm alone maps their 196560-set onto ours: {ok}")
    sign_note = "none needed"
    if not ok:
        # A sign change on coordinate set D preserves the (+-2^8) shape vectors iff
        # |D ∩ octad| is even for every octad, i.e. D is in the dual of the octad span
        # = the Golay code itself (self-dual). So only Golay-codeword sign patterns can
        # work; try all 4096 (on our side of the permutation).
        found = False
        for c in golay_codewords():
            s = [-1 if (c >> perm[i]) & 1 else 1 for i in range(DIM)]
            if row_set(apply_map(ref, perm, s)) == C:
                signs, found = s, True
                break
        if not found:
            raise SystemExit("no monomial map (perm + Golay sign pattern) carries their set onto ours; "
                             "a non-monomial Leech automorphism would be needed (T4.1)")
        mapped = apply_map(ref, perm, signs)
        sign_note = f"Golay-codeword pattern, negated coordinates {[i for i in range(DIM) if signs[i] < 0]}"
        print(f"sign pattern needed: {sign_note}")
    # bijection: 196560 distinct images, all in C, |C| = 196560
    n_img = len(row_set(mapped))
    assert n_img == N_MIN and row_set(mapped) <= C
    # independent membership test on the mapped rows
    members = int(is_lattice_vector_rows(mapped).sum())
    norms = np.unique((mapped * mapped).sum(axis=1)).tolist()
    print(f"mapped rows: distinct={n_img}, in C={n_img}, pass membership test={members}, norms={norms}")
    rec = {
        "convention": "ours[perm[i]] = signs[i] * theirs[i]  (i = their coordinate, 0..23)",
        "perm": perm,
        "signs": signs,
        "inverse_perm": [perm.index(j) for j in range(DIM)],
        "found_by": "backtracking on the octad systems (python/tools/convert_external.py find-map)",
        "backtracking_nodes": nodes,
        "reference_set": {"path": os.path.relpath(args.ref, ROOT), "sha256": sha256_file(args.ref),
                          "rows": int(len(ref))},
        "repo_commits": {"PackingStar": git_head(PS_DIR), "Kissing-Numbers": git_head(KKW_DIR)},
        "our_golay": "cyclic (23,12,7) code g(x)=x^11+x^10+x^6+x^5+x^4+x^2+1 + parity bit 23 (python/kiss_ref/golay.py)",
        "verification": {
            "their_octads_common_with_ours_under_identity": common,
            "perm_maps_their_octads_onto_ours": True,
            "sign_pattern": sign_note,
            "mapped_set_equals_our_C": True,
            "mapped_rows_distinct": n_img,
            "mapped_rows_pass_membership_test": members,
        },
        "date": _dt.date.today().isoformat(),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"wrote {os.path.relpath(args.out, ROOT)}")
    print(f"RESULT ok=1 perm={','.join(map(str, perm))} negated={sum(1 for s in signs if s < 0)} "
          f"nodes={nodes} time_s={time.time() - t0:.1f}")
    return rec


# ----------------------------------------------------------------------------
# convert
# ----------------------------------------------------------------------------
def load_map(path: str) -> dict:
    with open(path) as f:
        m = json.load(f)
    perm, signs = m["perm"], m["signs"]
    assert sorted(perm) == list(range(DIM)) and all(s in (1, -1) for s in signs) and len(signs) == DIM
    return m


def convert_rows(src: np.ndarray, m: dict, sort: bool) -> np.ndarray:
    out = apply_map(src, m["perm"], m["signs"])
    if sort:
        out = out[np.lexsort(out.T[::-1])]
    return out


def provenance_header(src_path: str, m: dict, n: int, label: str | None, sort: bool) -> list[str]:
    rel = os.path.relpath(src_path, ROOT)
    repo = "PackingStar" if os.path.abspath(src_path).startswith(PS_DIR) else \
           "Kissing-Numbers" if os.path.abspath(src_path).startswith(KKW_DIR) else None
    lines = []
    if label:
        lines.append(label)
    lines.append(f"source: {rel} (sha256 {sha256_file(src_path)})")
    if repo:
        url = {"PackingStar": "https://github.com/CDM1619/PackingStar",
               "Kissing-Numbers": "https://github.com/kenzkallal/Kissing-Numbers"}[repo]
        lines.append(f"repo: {url} @ {m.get('repo_commits', {}).get(repo, git_head(os.path.join(EXT, repo)))}")
    neg = [i for i, s in enumerate(m["signs"]) if s < 0]
    lines.append(f"map: ours[perm[i]] = signs[i]*theirs[i]; perm = {m['perm']}; "
                 f"negated coordinates = {neg if neg else 'none'} (data/external/coordinate_map.json)")
    lines.append(f"converted by python/tools/convert_external.py on {_dt.date.today().isoformat()}; "
                 f"rows in {'canonical sorted' if sort else 'source'} order")
    lines.append(f"{n} rows, 24 ints each, sqrt8-integer scaling (squared norm 32); independent set iff all "
                 f"off-diagonal Gram entries <= 8")
    return lines


def cmd_convert(args) -> None:
    src_path = args.npy or args.txt
    src = load_external(src_path, "npy" if args.npy else "txt")
    m = load_map(args.map)
    out = convert_rows(src, m, args.sort)
    write_set(args.out, out, provenance_header(src_path, m, len(out), args.label, args.sort))
    if args.out:
        print(f"wrote {os.path.relpath(args.out, ROOT)} ({len(out)} rows)")


# ----------------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------------
def verify_rows(name: str, S: np.ndarray, C: set[tuple[int, ...]] | None = None) -> dict:
    C = our_c_set() if C is None else C
    S = np.asarray(S, dtype=np.int64)
    keys = row_set(S)
    n = len(S)
    distinct = len(keys) == n
    norms = (S * S).sum(axis=1)
    norm_ok = bool(np.all(norms == 32))
    in_c = sum(1 for k in keys if k in C)
    member = int(is_lattice_vector_rows(S).sum())
    G = S @ S.T
    off = G[~np.eye(n, dtype=bool)]
    hist = {int(v): int(c) for v, c in zip(*np.unique(off, return_counts=True))}
    max_off = int(off.max()) if n > 1 else None
    antipodal = all(tuple(-v for v in k) in keys for k in keys)
    ok = distinct and norm_ok and in_c == n and member == n and (max_off is None or max_off <= 8)
    print(f"{name}: size={n} distinct={distinct} norm32={norm_ok} in_C={in_c}/{n} membership_test={member}/{n} "
          f"max_offdiag={max_off} gram_hist={hist} antipodal_closed={antipodal}")
    print(f"RESULT ok={int(ok)} size={n} file={name}")
    return {"size": n, "ok": ok, "distinct": distinct, "in_C": in_c, "max_offdiag": max_off,
            "gram_hist": hist, "antipodal": antipodal}


def cmd_verify(args) -> None:
    C = our_c_set()
    for p in args.files:
        verify_rows(os.path.relpath(p, ROOT), read_set(p), C)


# ----------------------------------------------------------------------------
# crosses
# ----------------------------------------------------------------------------
def _bits(m: int):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


def _popcount(m: int) -> int:
    return bin(m).count("1")


def antipodal_classes(S: np.ndarray) -> np.ndarray:
    reps: dict[tuple[int, ...], int] = {}
    R = []
    for r in S:
        t = tuple(int(v) for v in r)
        k = max(t, tuple(-v for v in t))
        if k not in reps:
            reps[k] = len(R)
            R.append(k)
    return np.array(R, dtype=np.int64)


def maximal_cliques(adj: list[int], n: int) -> list[int]:
    """Bron-Kerbosch with pivoting on int bitmasks; returns clique masks."""
    out: list[int] = []
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * n + 100))

    def bk(r: int, p: int, x: int) -> None:
        if p == 0 and x == 0:
            out.append(r)
            return
        u = max(_bits(p | x), key=lambda v: _popcount(adj[v] & p))
        for v in _bits(p & ~adj[u]):
            bk(r | (1 << v), p & adj[v], x & adj[v])
            p &= ~(1 << v)
            x |= 1 << v

    bk(0, (1 << n) - 1, 0)
    return out


def count_k_cliques(adj: list[int], P: int, k: int) -> int:
    if k == 0:
        return 1
    tot = 0
    for v in _bits(P):
        P &= ~(1 << v)
        tot += count_k_cliques(adj, P & adj[v], k - 1)
    return tot


def k_cliques_in(adj: list[int], P: int, k: int):
    """Yield masks of k-cliques inside P, in increasing lexicographic order."""
    if k == 0:
        yield 0
        return
    while P:
        b = P & -P
        u = b.bit_length() - 1
        P ^= b
        for rest in k_cliques_in(adj, P & adj[u], k - 1):
            yield rest | b


class _Timeout(Exception):
    pass


def exact_cover(adj: list[int], U: int, frames: list[int], use_frames: bool, deadline: float,
                stats: dict):
    """Partition U into 8-cliques (and, if use_frames, 24-cliques from `frames`).

    Branch on the uncovered vertex with the fewest uncovered neighbours; prune when
    some uncovered vertex has < 7 uncovered neighbours. Yields lists of clique masks.
    """
    parts: list[int] = []

    def rec(U: int):
        stats["nodes"] += 1
        if U == 0:
            yield list(parts)
            return
        if time.time() > deadline:
            raise _Timeout
        best = None
        for v in _bits(U):
            d = _popcount(adj[v] & U)
            if d < 7:
                return
            if best is None or d < best[0]:
                best = (d, v)
        v = best[1]
        if use_frames:
            for c in frames:
                if (c >> v) & 1 and (c & U) == c:
                    parts.append(c)
                    yield from rec(U & ~c)
                    parts.pop()
        for rest in k_cliques_in(adj, adj[v] & U, 7):
            c = rest | (1 << v)
            parts.append(c)
            yield from rec(U & ~c)
            parts.pop()

    yield from rec(U)


def span_count(Cint: np.ndarray, X: np.ndarray) -> int:
    """Number of minimal vectors in the span of mutually orthogonal norm-32 rows X."""
    P = Cint @ X.T
    return int(np.sum((P * P).sum(axis=1) == 1024))


def cmd_crosses(args) -> dict:
    t0 = time.time()
    S = read_set(args.file) if not args.file.endswith(".npy") else load_external(args.file)
    name = os.path.relpath(args.file, ROOT)
    R = antipodal_classes(S)
    n = len(R)
    G = R @ R.T
    A = (G == 0)
    np.fill_diagonal(A, False)
    adj = [int(sum(1 << j for j in np.nonzero(A[i])[0].tolist())) for i in range(n)]
    FULL = (1 << n) - 1
    deg = collections.Counter(int(x) for x in A.sum(axis=1))
    print(f"{name}: {len(S)} vectors, {n} antipodal classes; orthogonality graph degree distribution "
          f"{dict(sorted(deg.items()))}")
    Ai = A.astype(np.int32)
    transitive = not np.any(((Ai @ Ai) > 0) & ~A & ~np.eye(n, dtype=bool))
    print(f"  relation 'orthogonal' transitive on classes: {transitive}")

    cl = maximal_cliques(adj, n)
    sizes = collections.Counter(_popcount(c) for c in cl)
    omega = max(sizes)
    frames = [c for c in cl if _popcount(c) == 24]
    print(f"  maximal cliques: {len(cl)}, size distribution {dict(sorted(sizes.items()))}, clique number {omega}")
    fr_disjoint = all((frames[i] & frames[j]) == 0 for i in range(len(frames)) for j in range(i + 1, len(frames)))
    print(f"  24-cliques (full frames): {len(frames)}, pairwise disjoint: {fr_disjoint}")
    for k, c in enumerate(frames):
        print(f"    frame #{k}: classes {sorted(_bits(c))}")
    n8 = count_k_cliques(adj, FULL, 8)
    print(f"  8-cliques (all, not only maximal): {n8}")

    Cint = leech_min_vectors().astype(np.int64)
    result = {"file": name, "vectors": int(len(S)), "classes": n, "degree_distribution": dict(sorted(deg.items())),
              "transitive": bool(transitive), "maximal_cliques": len(cl),
              "maximal_clique_sizes": {int(k): int(v) for k, v in sorted(sizes.items())},
              "clique_number": omega, "frames": [sorted(_bits(c)) for c in frames],
              "frames_disjoint": fr_disjoint, "eight_cliques": n8, "partitions": {}}
    if n % 8 != 0:
        print(f"  {n} classes is not a multiple of 8: no partition into 8- and 24-cliques exists")
        result["partitions"]["note"] = f"{n} classes, not a multiple of 8"
    else:
        tasks = []
        for k, c in enumerate(frames):
            tasks.append((f"frame #{k} + {(n - 24) // 8} octuples", FULL & ~c, False))
        if len(frames) > 1:
            allf = 0
            for c in frames:
                allf |= c
            tasks.append((f"all {len(frames)} frames + {(n - 24 * len(frames)) // 8} octuples", FULL & ~allf, False))
        tasks.append((f"{n // 8} octuples, no frame", FULL, False))
        for label, U0, use_frames in tasks:
            stats = {"nodes": 0}
            t1 = time.time()
            status, sol = "none", None
            try:
                for s in exact_cover(adj, U0, frames, use_frames, time.time() + args.budget, stats):
                    sol, status = s, "found"
                    break
            except _Timeout:
                status = "timeout"
            el = time.time() - t1
            entry = {"status": status, "nodes": stats["nodes"], "seconds": round(el, 2)}
            if sol is not None:
                # verify the partition explicitly
                cover = 0
                for c in sol:
                    assert (cover & c) == 0
                    cover |= c
                    vs = sorted(_bits(c))
                    assert all(A[a, b] for a, b in itertools.combinations(vs, 2))
                assert cover == U0
                spans = sorted(collections.Counter(span_count(Cint, R[sorted(_bits(c))]) for c in sol).items())
                entry["octuples"] = [sorted(_bits(c)) for c in sol]
                entry["span_counts"] = {int(k): int(v) for k, v in spans}
                print(f"  partition {label}: FOUND (nodes {stats['nodes']}, {el:.1f}s); |C ∩ span| per octuple: "
                      f"{dict(spans)}")
            else:
                print(f"  partition {label}: {'NO PARTITION (exhaustive)' if status == 'none' else 'unresolved (timeout)'} "
                      f"(nodes {stats['nodes']}, {el:.1f}s)")
            result["partitions"][label] = entry
    if frames:
        print(f"  |C ∩ span| for frames: {[span_count(Cint, R[sorted(_bits(c))]) for c in frames]} (expect 196560)")
    print(f"RESULT file={name} classes={n} maximal_cliques={len(cl)} omega={omega} frames={len(frames)} "
          f"eight_cliques={n8} partition_1frame_28oct="
          f"{'yes' if any(v.get('status') == 'found' for k, v in result['partitions'].items() if k.startswith('frame #')) else 'no'} "
          f"time_s={time.time() - t0:.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return result


# ----------------------------------------------------------------------------
# all
# ----------------------------------------------------------------------------
def cmd_all(args) -> None:
    ns = argparse.Namespace(ref=DEFAULT_REF, out=DEFAULT_MAP)
    cmd_find_map(ns)
    m = load_map(DEFAULT_MAP)
    jobs = [
        (S496_SRC, "npy", os.path.join(ROOT, "data", "S496.txt"),
         "S496: the 496-vector independent set of Ma et al. 2025 (PackingStar, arXiv:2511.13391), in our coordinates"),
        (S488_SRC, "txt", os.path.join(ROOT, "data", "S488.txt"),
         "S488: the 488-vector independent set of Kallal-Kan-Wang (arXiv:1608.07270, S_1.txt), in our coordinates"),
    ]
    C = our_c_set()
    for src_path, kind, out_path, label in jobs:
        src = load_external(src_path, kind)
        out = convert_rows(src, m, False)
        write_set(out_path, out, provenance_header(src_path, m, len(out), label, False))
        print(f"wrote {os.path.relpath(out_path, ROOT)} ({len(out)} rows)")
        back = read_set(out_path)
        assert np.array_equal(back, out)
        verify_rows(os.path.relpath(out_path, ROOT), back, C)
    for out_path in (os.path.join(ROOT, "data", "S496.txt"), os.path.join(ROOT, "data", "S488.txt")):
        cmd_crosses(argparse.Namespace(file=out_path, budget=args.budget, json=None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find-map", help="find the monomial map from the external coordinates to ours")
    p.add_argument("--ref", default=DEFAULT_REF, help="external 196560-vector list (.npy or text)")
    p.add_argument("--out", default=DEFAULT_MAP, help="JSON output (use '' to skip writing)")
    p = sub.add_parser("convert", help="apply the map to an external set")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npy")
    g.add_argument("--txt")
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--out", default=None, help="output S text file (default: stdout)")
    p.add_argument("--label", default=None, help="first header comment line")
    p.add_argument("--sort", action="store_true", help="write rows in canonical sorted order")
    p = sub.add_parser("verify", help="independent Python verification of S text files")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("crosses", help="cross structure analysis of a set")
    p.add_argument("file")
    p.add_argument("--budget", type=float, default=120.0, help="seconds per exact-cover search")
    p.add_argument("--json", default=None, help="write the analysis as JSON")
    p = sub.add_parser("all", help="produce the T1.3 deliverables")
    p.add_argument("--budget", type=float, default=120.0)
    args = ap.parse_args()
    {"find-map": cmd_find_map, "convert": cmd_convert, "verify": cmd_verify,
     "crosses": cmd_crosses, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()

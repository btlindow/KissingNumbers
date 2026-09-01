"""Tests for python/verify_dimN.py (docs/design.md T4.3).

A synthetic dim-26 (d = 2) family is built from data/S496.txt and its image
under an M24 generator (a coordinate permutation preserving the Golay code and
hence C; T4.1 found the alpha and gamma images overlap the 496 in 0 vectors):
count 6 + 196560 + 2*2*496 = 198550. Corrupted variants must be rejected.
The families written by T4.2 under data/families/ are checked (exact part)
when present. Fixture tests skip when data/S496.txt is absent.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import verify_dimN  # noqa: E402
from kiss_ref import small_kissing as sk  # noqa: E402
from kiss_ref.exact import Q  # noqa: E402
from verify_S import check_set, leech_index, read_set  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
VERIFY = os.path.join(REPO, "python", "verify_dimN.py")
S496 = os.path.join(REPO, "data", "S496.txt")
M24 = os.path.join(REPO, "data", "group", "m24_generators.txt")
FAMILIES = os.path.join(REPO, "data", "families")
README_COUNTS = {26: 198550, 27: 200044, 28: 204520, 29: 209496, 30: 220440, 31: 238350}


def result_line(out: str) -> dict:
    line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")][-1]
    return dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))


def m24_generators() -> list[list[int]]:
    gens = []
    with open(M24) as f:
        for line in f:
            text = line.split("#", 1)[0].strip()
            if text:
                gens.append([int(t) for t in text.split()])
    return gens


def permute_coordinates(S: np.ndarray, g: list[int]) -> np.ndarray:
    """y[g[i]] = x[i]."""
    Y = np.zeros_like(S)
    Y[:, g] = S
    return Y


def write_set(path: str, S: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# synthetic test set\n")
        for row in S:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


@pytest.fixture(scope="module")
def two_sets():
    """(S496 rows, disjoint image rows) or skip."""
    if not os.path.exists(S496) or not os.path.exists(M24):
        pytest.skip("data/S496.txt or data/group/m24_generators.txt missing")
    S, _ = read_set(S496)
    index = leech_index()
    idx = index.indices(S)
    assert np.all(idx >= 0)
    for g in m24_generators():
        S2 = permute_coordinates(S, g)
        idx2 = index.indices(S2)
        assert np.all(idx2 >= 0), "an M24 generator did not preserve C"
        if len(np.intersect1d(idx, idx2)) == 0:
            assert check_set(S2)["ok"]
            return S, S2
    pytest.fail("no generator image disjoint from the 496")


@pytest.fixture(scope="module")
def family_dir(tmp_path_factory, two_sets):
    """A schema-v1 family directory for d = 2 with A2 triangles and the six (2,-1,-1)-type extras."""
    S, S2 = two_sets
    d = tmp_path_factory.mktemp("fam26")
    write_set(os.path.join(d, "S_01.txt"), S)
    write_set(os.path.join(d, "S_02.txt"), S2)
    Tb = sk.schema_T_block(2)
    Eb = sk.schema_extra_block(sk.lattice_extra_spheres(2)["vectors"], 2)
    fam = {"schema_version": 1, "dim": 26, "d": 2, "coordinates": "leech-sqrt8-integer",
           "sets": [{"file": "S_01.txt", "size": 496, "sha256": verify_dimN.sha256_file(os.path.join(d, "S_01.txt"))},
                    {"file": "S_02.txt", "size": 496}],
           "T": Tb, "extra": Eb, "count": 198550,
           "count_terms": {"extra": 6, "leech": 196560, "lifted": 1984}}
    with open(os.path.join(d, "family.json"), "w") as f:
        json.dump(fam, f)
    return str(d)


def load_json(family_dir: str) -> dict:
    with open(os.path.join(family_dir, "family.json")) as f:
        return json.load(f)


def variant(tmp_path, family_dir: str, mutate) -> str:
    """Copy the family directory's json (sets referenced by absolute path), mutate it, return the new dir."""
    j = load_json(family_dir)
    for s in j["sets"]:
        if isinstance(s, dict):
            s["file"] = os.path.join(family_dir, s["file"])
    j = mutate(j) or j
    with open(os.path.join(tmp_path, "family.json"), "w") as f:
        json.dump(j, f)
    return str(tmp_path)


def run_exact(path: str, strict: bool = False) -> dict:
    try:
        fam = verify_dimN.load_family(path)
    except verify_dimN.FamilyError as e:  # reader-level rejection (the CLI prints it as RESULT ok=0)
        return {"ok": False, "message": str(e), "count": None, "warnings": []}
    return verify_dimN.verify_family(fam, full_pass=False, do_float=False, strict=strict, log=lambda *_: None)


# ---------------------------------------------------------------------------
# the good family
# ---------------------------------------------------------------------------
def test_synthetic_family_exact(family_dir):
    """Exact casework in-process (Leech minimal-norm argument for eq-eq); count 198550."""
    fam = verify_dimN.load_family(family_dir)
    assert fam.dim == 26 and fam.d == 2 and len(fam.set_files) == 2 and len(fam.T) == 6 and len(fam.extras) == 6
    res = verify_dimN.verify_family(fam, full_pass=False, do_float=False, strict=True, log=lambda *_: None)
    assert res["ok"], res["message"]
    assert res["count"] == res["count_exact"] == 198550 == 6 + 196560 + 2 * 2 * 496
    assert res["max_ip_eq_lift"] == 16 and res["max_ip_same_set"] == 8 and res["max_ip_cross_set"] == 16
    assert sum(res["pairs"].values()) == 198550 * 198549 // 2
    assert res["pairs"]["lift-lift-same-x"] == 2 * 496 * 3 and res["pairs"]["extra-extra"] == 15
    assert not res["warnings"]


def test_cli_full_pass_and_float(family_dir):
    """The whole verifier as a stranger would run it (subprocess: single-threaded BLAS per worker)."""
    out = subprocess.run([PY, VERIFY, family_dir, "--strict"], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout + out.stderr
    r = result_line(out.stdout)
    assert r["ok"] == "1" and r["count"] == r["count_exact"] == r["count_float"] == "198550"
    assert abs(float(r["max_offdiag"]) - 2) < 1e-9 and r["max_class"] == "lift-lift-same-x"
    assert "full pass" in out.stdout and r["extra"] == "6" and r["K"] == "6"


def test_assembled_coordinates(family_dir):
    fam = verify_dimN.load_family(family_dir)
    index = leech_index()
    idx_list = [index.indices(read_set(p)[0]) for p in fam.set_files]
    X, meta = verify_dimN.assemble_coordinates(index.C, idx_list, fam.T, fam.groups, fam.extras, 2)
    assert X.shape == (198550, 26)
    assert np.abs((X * X).sum(axis=1) - 4).max() < 1e-9
    n_eq = meta["n_eq"]
    assert n_eq == 196560 - 992 and meta["n_lift"] == 2976 and meta["n_ex"] == 6
    # two lifts of the same x with different y in a triangle: 8/3 - 2/3 = 2 exactly
    assert abs(X[n_eq] @ X[n_eq + 496] - 2.0) < 1e-12
    assert verify_dimN.pair_class(n_eq, n_eq + 496, meta) == "lift-lift-same-x"
    assert verify_dimN.pair_class(0, n_eq, meta) == "eq-lift"
    assert verify_dimN.pair_class(n_eq, n_eq + 3 * 496, meta) == "lift-lift-cross-set"
    assert verify_dimN.pair_class(len(X) - 1, len(X) - 2, meta) == "extra-extra"
    assert verify_dimN.pair_class(len(X) - 1, n_eq, meta) == "extra-lift"
    # extras are orthogonal to the Leech part and have norm 4
    assert np.all(X[-6:, :24] == 0)


# ---------------------------------------------------------------------------
# tolerant reader (PLAN draft forms)
# ---------------------------------------------------------------------------
def test_reader_plan_draft_forms(tmp_path, family_dir):
    def mutate(j):
        Tb, Eb = j["T"], j["extra"]
        # sets as plain file names, T as a list of groups of vectors (strings with fractions),
        # extras as vectors with a scale and an explicit sqrt: y' = b sqrt3 / 3
        V = Tb["vectors"]
        groups = [[[str(x) + "/2" for x in (2 * np.array(V[i])).tolist()] for i in g] for g in Tb["groups"]]
        extras = {"sqrt": 3, "scale": "3", "vectors": [[[0, x] for x in row] for row in Eb["b"]]}
        return {"dim": 26, "sets": [s["file"] for s in j["sets"]], "T": groups, "extra": extras}
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert res["ok"], res["message"]
    assert res["count"] == 198550


def test_reader_per_set_groups_and_string_entries(tmp_path, family_dir):
    def mutate(j):
        Tb = j["T"]
        sets = [{"file": s["file"], "group": g} for s, g in zip(j["sets"], Tb["groups"])]
        vecs = [[f"{x}" for x in row] for row in Tb["vectors"]]
        return {"d": 2, "sets": sets, "T": {"vectors": vecs}, "extra": j["extra"]}
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert res["ok"], res["message"]


def test_reader_no_extras_is_fine(tmp_path, family_dir):
    def mutate(j):
        j.pop("extra")
        j["count"] = 198544
        j.pop("count_terms")
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert res["ok"] and res["count"] == 198544
    assert any("record form uses K(2) = 6" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# corrupted inputs
# ---------------------------------------------------------------------------
def test_overlapping_sets_rejected(tmp_path, family_dir):
    def mutate(j):
        j["sets"][1] = dict(j["sets"][0])  # S_2 := S_1
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "overlap" in res["message"] and "496 shared" in res["message"]


def test_partially_overlapping_sets_rejected(tmp_path, family_dir, two_sets):
    S, S2 = two_sets
    # S[0] plus every row of S2 compatible with it (ip <= 8): an independent set sharing exactly one
    # vector with S_1 (replacing a single row of S2 by S[0] would fail as *dependent* first: S[0]
    # is at 60 degrees from a dozen rows of S2)
    keep = S2[(S2 @ S[0]) <= 8]
    assert len(keep) >= 400
    S3 = np.concatenate([keep, S[:1]])
    write_set(os.path.join(tmp_path, "S_03.txt"), S3)

    def mutate(j):
        j["sets"][1] = {"file": os.path.join(tmp_path, "S_03.txt")}
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "sets 1 and 2 overlap" in res["message"] and "[1 shared" in res["message"]


def test_dependent_set_rejected(tmp_path, family_dir, two_sets):
    S, S2 = two_sets
    C = leech_index().C.astype(np.int64)
    nb = C[np.flatnonzero(C @ S2[0] == 16)[0]]
    write_set(os.path.join(tmp_path, "S_bad.txt"), np.concatenate([S2[:-1], nb[None, :]]))

    def mutate(j):
        j["sets"][1] = {"file": os.path.join(tmp_path, "S_bad.txt")}
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and res["message"].startswith("set 2") and "inner product 16" in res["message"]


def test_T_group_with_60_degree_pair_rejected(tmp_path, family_dir):
    def mutate(j):
        V = np.array(j["T"]["vectors"])
        G = V @ V.T
        i, k = 0, int(np.flatnonzero(G[0] == 1)[0])  # cos = 1/2: 60 degrees
        other = [t for t in range(6) if t not in (i, k)]
        j["T"]["groups"] = [[i, k, other[0]], other[1:]]
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "group 1" in res["message"] and "60.00 deg" in res["message"]


def test_T_vector_used_twice_rejected(tmp_path, family_dir):
    def mutate(j):
        g = j["T"]["groups"]
        g[1][0] = g[0][0]
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "already used" in res["message"]


def test_extra_at_20_degrees_rejected(tmp_path, family_dir):
    def mutate(j):
        # (3,-2,-1) is in the A2 plane and at 19.1 degrees from (1,-1,0): cos = 5/sqrt(28)
        j["extra"] = {"vectors": [[3, -2, -1]]}
        j["count"] = 196560 + 1984 + 1
        j["count_terms"] = {"extra": 1, "leech": 196560, "lifted": 1984}
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "within 30 degrees" in res["message"] and "19.1" in res["message"]


def test_extras_too_close_to_each_other_rejected(tmp_path, family_dir):
    def mutate(j):
        j["extra"] = {"vectors": [[2, -1, -1], [3, -1, -2]]}
        j["count"] = 196560 + 1984 + 2
        j.pop("count_terms")
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and res["message"].startswith("extra: rows 0 and 1 have cos")


def test_wrong_count_claim_rejected(tmp_path, family_dir):
    res = run_exact(variant(tmp_path, family_dir, lambda j: j.update(count=198551)))
    assert not res["ok"] and "claims 198551" in res["message"]


def test_missing_group_rejected(tmp_path, family_dir):
    def mutate(j):
        j["T"]["groups"].pop()
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "1 T groups for 2 sets" in res["message"]


def test_rank_too_large_rejected(tmp_path, family_dir):
    def mutate(j):
        j["T"]["vectors"].append([1, 1, 1])  # orthogonal to the A2 plane (cos 0 with every T vector), rank 3 > d = 2
        j["T"]["K"] = 7
    res = run_exact(variant(tmp_path, family_dir, mutate))
    assert not res["ok"] and "3-dimensional space > d = 2" in res["message"]
    # without the K update the reader itself rejects the block
    res = run_exact(variant(tmp_path, family_dir, lambda j: j["T"]["vectors"].append([1, 1, 1])))
    assert not res["ok"] and "claims 6 rows but has 7" in res["message"]


def test_bad_json_and_missing_file(tmp_path, family_dir):
    with open(os.path.join(tmp_path, "family.json"), "w") as f:
        f.write("{not json")
    out = subprocess.run([PY, VERIFY, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 1 and result_line(out.stdout)["ok"] == "0"
    d2 = variant(tmp_path, family_dir, lambda j: j["sets"].append({"file": "nope.txt"}))
    res = run_exact(d2)
    assert not res["ok"] and "parse error" in res["message"]


def test_strict_mode_sha256_and_non_antipodal_pair(tmp_path, family_dir):
    def mutate(j):
        j["sets"][0]["sha256"] = "0" * 64
    path = variant(tmp_path, family_dir, mutate)
    assert run_exact(path)["ok"]  # warning only
    res = run_exact(path, strict=True)
    assert not res["ok"] and "sha256" in res["message"]


# ---------------------------------------------------------------------------
# T4.2's families (exact part only; the full runs are in docs/reports/T4.3.md)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", sorted(README_COUNTS))
def test_packingstar_family_exact(n):
    path = os.path.join(FAMILIES, f"dim{n}")
    if not os.path.exists(os.path.join(path, "family.json")):
        pytest.skip(f"{path} not present")
    fam = verify_dimN.load_family(path)
    assert fam.dim == n and len(fam.extras) == sk.K[n - 24]
    res = verify_dimN.verify_family(fam, full_pass=False, do_float=False, strict=True, log=lambda *_: None)
    assert res["ok"], res["message"]
    assert res["count"] == README_COUNTS[n]
    assert res["max_ip_cross_set"] == 16 and res["max_ip_same_set"] == 8 and res["max_ip_eq_lift"] == 16

"""Tests for python/verify_S.py and python/verify_dim25.py (docs/design.md T1.4).

Everything is computed from the independent numpy generator. Fixture tests
(data/S496.txt, data/S488.txt) and the C++/Python cross-checks
(build/release/tools/verify_s) are skipped when the files are absent.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import verify_S  # noqa: E402
import verify_dim25  # noqa: E402
from kiss_ref.leech import DIM, N_MIN, leech_min_vectors  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
VERIFY_S = os.path.join(REPO, "python", "verify_S.py")
VERIFY_DIM25 = os.path.join(REPO, "python", "verify_dim25.py")
CPP_TOOL = os.path.join(REPO, "build", "release", "tools", "verify_s")
FIXTURES = {"S496.txt": 496, "S488.txt": 488}


def fixture(name: str) -> str:
    p = os.path.join(REPO, "data", name)
    if not os.path.exists(p):
        pytest.skip(f"fixture {p} not present")
    return p


@pytest.fixture(scope="module")
def C() -> np.ndarray:
    return leech_min_vectors()


def greedy_mis(C: np.ndarray, seed: int) -> np.ndarray:
    """Greedy random maximal independent set (canonical indices)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(C))
    blocked = np.zeros(len(C), dtype=bool)
    Ci = C.astype(np.int32)
    S = []
    for v in perm:
        if blocked[v]:
            continue
        S.append(int(v))
        blocked |= (Ci @ Ci[v]) == 16
    return np.array(S, dtype=np.int64)


@pytest.fixture(scope="module")
def greedy(C) -> tuple[np.ndarray, np.ndarray]:
    idx = greedy_mis(C, 20260825)
    return idx, C[idx].astype(np.int64)


def write_set(path: str, S: np.ndarray, header: str = "") -> str:
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n\n")
        for row in S:
            f.write(" ".join(str(int(v)) for v in row) + "\n")
    return path


def neighbour60(C: np.ndarray, x: np.ndarray) -> np.ndarray:
    ips = C.astype(np.int64) @ np.asarray(x, dtype=np.int64)
    j = int(np.flatnonzero(ips == 16)[0])
    return C[j].astype(np.int64)


def result_line(out: str) -> dict:
    line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")][-1]
    return dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))


# ---------------------------------------------------------------------------
# in-process checks on the greedy set and corruptions
# ---------------------------------------------------------------------------
def test_greedy_set_verifies(C, greedy):
    idx, S = greedy
    assert 150 < len(S) < 400
    res = verify_S.check_set(S)
    assert res["ok"] and res["size"] == len(S) and res["message"] == "ok"
    assert np.array_equal(res["idx"], idx)
    assert res["max_offdiag"] <= 8
    assert set(res["gram"]) <= {-32, -16, -8, 0, 8}
    # maximal: no vertex outside S is free (tightness 0)
    tight = ((C.astype(np.int64) @ S.T) == 16).sum(axis=1)
    in_s = np.zeros(N_MIN, dtype=bool)
    in_s[idx] = True
    assert not np.any((tight == 0) & ~in_s)
    assert np.all(tight[idx] == 0)


def test_empty_set():
    res = verify_S.check_set(np.zeros((0, DIM), dtype=np.int64))
    assert res["ok"] and res["size"] == 0


def test_corrupt_60_degree(C, greedy):
    _, S = greedy
    B = S.copy()
    B[1] = neighbour60(C, S[0])
    res = verify_S.check_set(B)
    assert not res["ok"] and res["size"] == len(S)
    assert "gram" in res["message"] and "row 0" in res["message"] and "row 1" in res["message"]
    assert "inner product 16" in res["message"]
    # with line numbers
    lines = [10 + 2 * k for k in range(len(B))]
    res = verify_S.check_set(B, lines)
    assert "line 10 (row 0)" in res["message"] and "line 12 (row 1)" in res["message"]


def test_corrupt_duplicate(greedy):
    _, S = greedy
    B = S.copy()
    B[5] = B[3]
    res = verify_S.check_set(B)
    assert not res["ok"] and "distinct" in res["message"]
    assert "row 3" in res["message"] and "row 5" in res["message"]


def test_corrupt_norm(greedy):
    _, S = greedy
    B = S.copy()
    B[2] = 0
    B[2, 0] = 8  # lattice vector of norm 64
    res = verify_S.check_set(B)
    assert not res["ok"] and res["message"].startswith("norm") and "row 2" in res["message"] and "64" in res["message"]
    B[2, 0] = 200  # outside int8 as well
    res = verify_S.check_set(B)
    assert not res["ok"] and res["message"].startswith("norm")


def test_corrupt_not_in_C(C, greedy):
    _, S = greedy
    # octad vector with one +-2 moved off the octad: norm 32, not a lattice vector
    k = int(np.flatnonzero((np.abs(C) == 2).sum(axis=1) == 8)[0])
    w = C[k].astype(np.int64).copy()
    frm = int(np.flatnonzero(w != 0)[0])
    to = int(np.flatnonzero(w == 0)[0])
    w[to], w[frm] = w[frm], 0
    assert (w * w).sum() == 32
    B = S.copy()
    B[4] = w
    res = verify_S.check_set(B)
    assert not res["ok"] and res["message"].startswith("membership") and "row 4" in res["message"]
    # a norm-32 vector with the wrong sign parity on an octad is also rejected
    w2 = C[k].astype(np.int64).copy()
    w2[frm] = -w2[frm]
    B[4] = w2
    res = verify_S.check_set(B)
    assert not res["ok"] and res["message"].startswith("membership")


def test_read_set_grammar(tmp_path, greedy):
    _, S = greedy
    p = tmp_path / "s.txt"
    with open(p, "w") as f:
        f.write("# header\n\n")
        f.write("\t".join(str(int(v)) for v in S[0]) + "   # trailing\r\n")
        f.write("   \n")
        f.write(" ".join(str(int(v)) for v in S[1]) + "\n")
    rows, lines = verify_S.read_set(str(p))
    assert rows.shape == (2, DIM) and lines == [3, 5]
    assert np.array_equal(rows, S[:2])
    for bad in ("1 2 3\n", "x " + "0 " * 23 + "\n", "1.5 " + "0 " * 23 + "\n"):
        with open(p, "w") as f:
            f.write(bad)
        with pytest.raises(ValueError):
            verify_S.read_set(str(p))


# ---------------------------------------------------------------------------
# the scripts end to end
# ---------------------------------------------------------------------------
def run_py(script: str, *args: str) -> tuple[int, str]:
    r = subprocess.run([PY, script, *args], capture_output=True, text=True, cwd=REPO)
    return r.returncode, r.stdout + r.stderr


def test_verify_S_script(tmp_path, C, greedy):
    _, S = greedy
    good = write_set(str(tmp_path / "good.txt"), S, header="greedy set")
    rc, out = run_py(VERIFY_S, good)
    res = result_line(out)
    assert rc == 0 and res["ok"] == "1" and int(res["size"]) == len(S), out
    B = S.copy()
    B[1] = neighbour60(C, S[0])
    bad = write_set(str(tmp_path / "bad.txt"), B, header="corrupted")
    rc, out = run_py(VERIFY_S, bad)
    res = result_line(out)
    assert rc == 1 and res["ok"] == "0" and int(res["size"]) == len(S)
    assert "line 3 (row 0)" in out and "line 4 (row 1)" in out and "inner product 16" in out
    rc, out = run_py(VERIFY_S, str(tmp_path / "missing.txt"))
    assert rc == 1 and result_line(out)["ok"] == "0"


def test_exact_casework_in_process(C, greedy):
    idx, S = greedy
    ex = verify_dim25.exact_count(C, S, idx, full_pass=False)
    assert ex["ok"] and ex["count"] == N_MIN + len(S)
    assert ex["max_ip_lifted"] <= 8 and ex["max_ip_eq_lift"] == 16
    assert sum(ex["pairs"].values()) == ex["count"] * (ex["count"] - 1) // 2
    # a corrupted S (60-degree pair) fails the lifted-lifted same-sign case
    B = S.copy()
    B[1] = neighbour60(C, S[0])
    bidx = verify_S.leech_index().indices(B)
    ex = verify_dim25.exact_count(C, B, bidx, full_pass=False)
    assert not ex["ok"] and "same sign" in ex["message"]
    # an x both lifted and equatorial (wrong index list) is caught structurally
    ex = verify_dim25.exact_count(C, S, np.concatenate([idx[:-1], idx[:1]]), full_pass=False)
    assert not ex["ok"]


def test_float_coordinates_small(C, greedy):
    idx, S = greedy
    X, n_eq, m = verify_dim25.build_coordinates(C, idx)
    assert X.shape == (N_MIN + m, DIM + 1) and n_eq == N_MIN - m
    assert np.allclose((X * X).sum(axis=1), 4.0)
    # lifted pairs of the same x: inner product 4/3; the classes of a few pairs
    lifted = X[n_eq:]
    assert np.allclose((lifted[:m] * lifted[m:]).sum(axis=1), 4.0 / 3.0)
    assert verify_dim25.pair_class(0, 1, n_eq, m) == "eq-eq"
    assert verify_dim25.pair_class(0, n_eq, n_eq, m) == "eq-lift"
    assert verify_dim25.pair_class(n_eq, n_eq + m, n_eq, m) == "lift-lift-same-x"
    assert verify_dim25.pair_class(n_eq, n_eq + 1, n_eq, m) == "lift-lift-same"
    assert verify_dim25.pair_class(n_eq, n_eq + m + 1, n_eq, m) == "lift-lift-opp"


# ---------------------------------------------------------------------------
# fixtures: the 496 and the 488 (skipped if absent)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,size", sorted(FIXTURES.items()))
def test_fixture_python(name, size):
    path = fixture(name)
    S, lines = verify_S.read_set(path)
    res = verify_S.check_set(S, lines)
    assert res["ok"], res["message"]
    assert res["size"] == size and res["antipodal"]
    rc, out = run_py(VERIFY_S, path)
    r = result_line(out)
    assert rc == 0 and r["ok"] == "1" and int(r["size"]) == size, out


@pytest.mark.parametrize("name,size", sorted(FIXTURES.items()))
def test_fixture_cpp_vs_python(name, size, tmp_path, C):
    path = fixture(name)
    if not os.path.exists(CPP_TOOL):
        pytest.skip(f"{CPP_TOOL} not built")
    _, out_py = run_py(VERIFY_S, path)
    r_cpp = subprocess.run([CPP_TOOL, path, "--data", "data"], capture_output=True, text=True, cwd=REPO)
    r1, r2 = result_line(out_py), result_line(r_cpp.stdout)
    assert r_cpp.returncode == 0 and r1["ok"] == "1" and r2["ok"] == "1"
    assert int(r1["size"]) == int(r2["size"]) == size
    assert r1["antipodal"] == r2["antipodal"] == "1"
    assert r1["gram"] == r2["gram"]
    # 60-degree corruption rejected by both, naming the same pair (lines 1 and 2 are the data rows)
    S, lines = verify_S.read_set(path)
    B = S.copy()
    B[1] = neighbour60(C, S[0])
    bad = write_set(str(tmp_path / "bad.txt"), B)
    rc, out_py = run_py(VERIFY_S, bad)
    r_cpp = subprocess.run([CPP_TOOL, bad, "--data", "data"], capture_output=True, text=True, cwd=REPO)
    assert rc == 1 and r_cpp.returncode == 1
    assert result_line(out_py)["ok"] == "0" and result_line(r_cpp.stdout)["ok"] == "0"
    for out in (out_py, r_cpp.stdout):
        assert "line 1 (row 0)" in out and "line 2 (row 1)" in out and "inner product 16" in out


def test_verify_dim25_s496():
    path = fixture("S496.txt")
    rc, out = run_py(VERIFY_DIM25, path)
    r = result_line(out)
    assert rc == 0 and r["ok"] == "1", out
    assert int(r["count_exact"]) == 197056 and int(r["count_float"]) == 197056
    assert float(r["max_offdiag_float"]) <= 2 + 1e-9

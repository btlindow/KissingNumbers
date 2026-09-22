"""Falsification suite for verify2627_layers_independent.py: every corrupted two-layer artefact
below must be REJECTED, and the uncorrupted one accepted. Seven corruptions per dimension, five
of them in the second layer.

    PYTHONPATH=python python tools/kravatskiy/falsify_layers.py <dim26-27 package> [workdir]

The package must hold heads{n}_Y.npy, heads{n}_side.npy, heads{n}_layer2_u.npy and
heads{n}_layer2_line.npy for n = 26 and 27. Verifier runs go four at a time.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kiss_ref.leech import leech_min_vectors

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = {26: 24, 27: 32}          # only used to PICK a corruption; the verifier derives its own bars


def load(pkg, n):
    d = os.path.join(pkg, "data")
    return (np.load(os.path.join(d, f"heads{n}_Y.npy")).astype(np.int64),
            np.load(os.path.join(d, f"heads{n}_side.npy")).astype(np.int64),
            np.load(os.path.join(d, f"heads{n}_layer2_u.npy")).astype(np.int64),
            np.load(os.path.join(d, f"heads{n}_layer2_line.npy")).astype(np.int64))


def cases(n, Y, side, U2, line, Z):
    own = np.full(len(Y), -1, np.int64)
    for s in range(0, len(Y), 64):
        hit = (Y[s:s + 64] @ Z.T) > 48
        own[s:s + 64] = np.where(hit.sum(1) == 1, hit.argmax(1), -1)
    owners1 = set(own[own >= 0].tolist())
    used = {u.tobytes() for u in U2}
    out = []

    # 1. a second-layer owner replaced by a minimal vector above the bar against some first-layer head
    M = Z @ Y[0]
    z = next(int(i) for i in np.nonzero(M > BAR[n])[0] if i not in owners1 and Z[i].tobytes() not in used)
    U = U2.copy(); U[0] = Z[z]
    out.append(("second-layer owner too close to a first-layer head", Y, side, U, line))
    # 2. two second-layer heads with <u,u'> = 16 forced onto one line
    G = U2 @ U2.T
    i, j = next((a, b) for a, b in zip(*np.nonzero(G == 16)) if line[a] != line[b])
    L = line.copy(); L[j] = L[i]
    out.append(("two second-layer heads at <u,u'> = 16 put on one line", Y, side, U2, L))
    # 3. a second-layer owner that is already a first-layer owner
    U = U2.copy(); U[1] = Z[next(iter(owners1))]
    out.append(("second-layer owner equal to a first-layer owner", Y, side, U, line))
    # 4. a duplicated second-layer owner
    U = U2.copy(); U[2] = U2[3]
    out.append(("second-layer owner duplicated", Y, side, U, line))
    # 5. a second-layer 'owner' of norm 32 that is not a minimal vector of the lattice: (5,2,1,1,1,0,...)
    U = U2.copy(); U[4] = 0; U[4, :5] = (5, 2, 1, 1, 1)
    assert int(U[4] @ U[4]) == 32
    out.append(("second-layer owner of norm 32 that is not a minimal vector", Y, side, U, line))
    # 6. a line index out of range
    L = line.copy(); L[5] = 3
    out.append(("second-layer line index out of range", Y, side, U2, L))
    # 7. the first layer is still guarded: a class head moved to another triangle
    S = side.copy(); c = int(np.nonzero(own >= 0)[0][0]); S[c] = (S[c] + 1) % (side.max() + 1)
    out.append(("first-layer head moved to another triangle", Y, S, U2, line))
    return out


def run(args):
    name, n, d, total, want_pass = args
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))
    p = subprocess.run([sys.executable, os.path.join(HERE, "verify2627_layers_independent.py"), d, str(n), str(total)],
                       capture_output=True, text=True, env=env)
    passed = p.returncode == 0 and "ALL CHECKS PASS" in p.stdout
    return name, n, passed == want_pass, passed


def main(pkg, work=None):
    work = work or tempfile.mkdtemp(prefix="falsify_layers_")
    Z = leech_min_vectors().astype(np.int64)
    jobs = []
    for n in (26, 27):
        Y, side, U2, line = load(pkg, n)
        own_n = 0
        for s in range(0, len(Y), 64):
            own_n += int((((Y[s:s + 64] @ Z.T) > 48).sum(1) == 1).sum())
        total = 196560 - own_n - len(U2) + 3 * len(Y) + 2 * len(U2) + (6 if n == 26 else 12)
        allc = [("the uncorrupted artefact", Y, side, U2, line)] + cases(n, Y, side, U2, line, Z)
        for t, (name, y, s_, u, l) in enumerate(allc):
            d = os.path.join(work, f"dim{n}_case{t}", "data"); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"heads{n}_Y.npy"), y); np.save(os.path.join(d, f"heads{n}_side.npy"), s_)
            np.save(os.path.join(d, f"heads{n}_layer2_u.npy"), u); np.save(os.path.join(d, f"heads{n}_layer2_line.npy"), l)
            jobs.append((name, n, os.path.dirname(d), total, t == 0))
    bad = 0
    with ThreadPoolExecutor(4) as ex:
        for name, n, good, passed in ex.map(run, jobs):
            verdict = "accepts" if passed else "rejects"
            print(f"  {'ok  ' if good else 'FAIL'} dim {n}: {verdict} {name}", flush=True)
            bad += 0 if good else 1
    shutil.rmtree(work, ignore_errors=True)
    print("ALL LAYER FALSIFICATION TESTS PASS" if bad == 0 else f"{bad} TEST(S) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))

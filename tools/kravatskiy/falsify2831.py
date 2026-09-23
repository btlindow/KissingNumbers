"""Falsification suite for verify2831_independent.py: every corrupted artefact below must be
REJECTED and the uncorrupted one accepted.

    PYTHONPATH=python python tools/kravatskiy/falsify2831.py <package> <n> <expected total>
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kiss_ref.leech import leech_min_vectors

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    name, d, n, tot, want = args
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))
    p = subprocess.run([sys.executable, os.path.join(HERE, "verify2831_independent.py"), d, str(n), str(tot)],
                       capture_output=True, text=True, env=env)
    passed = p.returncode == 0 and "ALL CHECKS PASS" in p.stdout
    return name, passed == want, passed


def main(pkg, n, tot):
    work = tempfile.mkdtemp(prefix="falsify2831_")
    Z = leech_min_vectors().astype(np.int64)
    src = os.path.join(pkg, "data")
    O = np.load(os.path.join(src, f"owners{n}.npy")).astype(np.int64)
    B = np.load(os.path.join(src, f"bounds{n}.npy")).astype(np.int64)
    G = json.load(open(os.path.join(src, f"geom{n}.json")))
    key = {Z[i].astype(np.int8).tobytes() for i in range(len(Z))}

    cases = [("the uncorrupted artefact", O, B, G, True)]

    # 1. an owner that is not a lattice vector, but has the right norm
    U = O.copy(); U[0] = 0; U[0, :5] = (5, 2, 1, 1, 1)
    assert int(U[0] @ U[0]) == 32 and U[0].astype(np.int8).tobytes() not in key
    cases.append(("an owner of norm 32 that is not a minimal vector", U, B, G, False))

    # 2. a duplicated owner line
    U = O.copy(); U[1] = O[2]
    cases.append(("a duplicated owner line", U, B, G, False))

    # 3. an owner replaced by one at 60 degrees to another in its own group (<u,u'> = 16)
    g0 = O[B[0]:B[1]]
    ip = Z @ g0[0]
    cand = [i for i in np.nonzero(ip == 16)[0] if Z[i].astype(np.int8).tobytes() not in
            {O[j].astype(np.int8).tobytes() for j in range(len(O))}]
    U = O.copy(); U[B[0] + 1] = Z[cand[0]]
    cases.append(("an owner at 60 degrees to another in its own class", U, B, G, False))

    # 4. a whole line moved into another group
    U = O.copy(); U[[B[1] - 1, B[1]]] = U[[B[1], B[1] - 1]]
    cases.append(("two owners swapped across a group boundary", U, B, G, False))

    # 5. a direction perturbed so its group is no longer at 120 degrees
    G2 = json.loads(json.dumps(G))
    i = G2["groups"][0][0]
    G2["directions"][i] = G["directions"][G["groups"][1][0]]
    cases.append(("a direction replaced so its group is not pairwise at 120 degrees", O, B, G2, False))

    # 6. an axis direction scaled off the unit sphere
    G3 = json.loads(json.dumps(G))
    G3["axis"][0] = [[2 * c for c in comp] for comp in G["axis"][0]]
    cases.append(("an axis direction of the wrong norm", O, B, G3, False))

    # 7. the bounds altered so the groups do not partition the owners
    B2 = B.copy(); B2[1] = B2[1] + 1
    cases.append(("bounds that do not partition the owner list", O, B2, G, False))

    jobs = []
    for t, (name, U, BB, GG, want) in enumerate(cases):
        d = os.path.join(work, f"case{t}", "data")
        os.makedirs(d)
        for f in os.listdir(src):
            if f.endswith(".npy") and not f.startswith(("owners", "bounds")):
                shutil.copy2(os.path.join(src, f), os.path.join(d, f))
        np.save(os.path.join(d, f"owners{n}.npy"), U.astype(np.int8))
        np.save(os.path.join(d, f"bounds{n}.npy"), BB.astype(np.int32))
        json.dump(GG, open(os.path.join(d, f"geom{n}.json"), "w"))
        jobs.append((name, os.path.dirname(d), n, tot, want))

    bad = 0
    with ThreadPoolExecutor(3) as ex:
        for name, good, passed in ex.map(run, jobs):
            print(f"  {'ok  ' if good else 'FAIL'} dim {n}: {'accepts' if passed else 'rejects'} {name}", flush=True)
            bad += 0 if good else 1
    shutil.rmtree(work, ignore_errors=True)
    print("ALL 28-31 FALSIFICATION TESTS PASS" if bad == 0 else f"{bad} TEST(S) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))

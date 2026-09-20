"""Falsifiability tests for the two independent verifiers.

A verifier that passes everything proves nothing. This copies each package to a
temporary directory, corrupts it in a way that must break the claim, and
requires the verifier to exit non-zero.

Usage:
    PYTHONPATH=python python tools/kravatskiy/falsify.py <dim25-pkg> <dim2627-pkg>
"""

import io
import os
import pickle
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(mod, *args):
    """Run a verifier, swallowing its output; return its exit status."""
    import importlib
    m = importlib.import_module(mod)
    m.FAIL = []
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = m.main(*args)
    except Exception:
        return 1
    return rc


def copy(pkg):
    d = tempfile.mkdtemp()
    dst = os.path.join(d, os.path.basename(pkg.rstrip("\\/")))
    shutil.copytree(pkg, dst)
    return dst


def expect_fail(label, rc):
    ok = rc != 0
    print(f"  {'ok  ' if ok else 'FAIL'}  rejects {label}")
    return ok


def main(p25, p2627):
    import verify25_independent as v25
    import verify2627_independent as v2627
    good = True

    print("falsify.py --- the verifiers must reject corrupted artefacts\n")
    print("dimension 25")
    print(f"  (control: the untouched package verifies)      rc={run('verify25_independent', p25)}")

    # 1. move one head off its sphere
    d = copy(p25)
    f = os.path.join(d, "data", "heads_X.npy")
    X = np.load(f); X[0] *= 1.01; np.save(f, X)
    good &= expect_fail("a head displaced off |x|^2 = 3", run("verify25_independent", d))

    # 2. un-remove an owner by duplicating another owner (owners no longer distinct)
    d = copy(p25)
    f = os.path.join(d, "data", "heads_U.npy")
    U = np.load(f); U[5] = U[6]; np.save(f, U)
    good &= expect_fail("two heads claiming the same owner", run("verify25_independent", d))

    # 3. replace an owner by a different minimal vector, so its head now
    #    conflicts with a retained equator point
    d = copy(p25)
    f = os.path.join(d, "data", "heads_U.npy")
    U = np.load(f); U[7] = -U[7]; np.save(f, U)
    good &= expect_fail("an owner replaced by its negative", run("verify25_independent", d))

    # 4. corrupt an interior head so it leaves the sphere
    d = copy(p25)
    f = os.path.join(d, "data", "heads_exact.pkl")
    with open(f, "rb") as fh:
        pk = pickle.load(fh)
    oi, coords = pk["rat"][0]
    pk["rat"][0] = (oi, [c * 2 for c in coords])
    with open(f, "wb") as fh:
        pickle.dump(pk, fh)
    good &= expect_fail("an interior head scaled off the sphere", run("verify25_independent", d))

    # 5. move the extra equator point
    d = copy(p25)
    f = os.path.join(d, "data", "extra_P.npy")
    P = np.load(f); P[0] = -P[0]; np.save(f, P)
    good &= expect_fail("the extra point negated", run("verify25_independent", d))

    for n in (26, 27):
        print(f"\ndimension {n}")
        print(f"  (control: the untouched package verifies)      "
              f"rc={run('verify2627_independent', p2627, n)}")

        # 6. move one head to another triangle
        d = copy(p2627)
        f = os.path.join(d, "data", f"heads{n}_side.npy")
        s = np.load(f); s[0] = (s[0] + 1) % (2 if n == 26 else 4); np.save(f, s)
        good &= expect_fail("a head moved to another triangle",
                            run("verify2627_independent", d, n))

        # 7. duplicate a head onto two triangles
        d = copy(p2627)
        fy = os.path.join(d, "data", f"heads{n}_Y.npy")
        fs = os.path.join(d, "data", f"heads{n}_side.npy")
        Y = np.load(fy); s = np.load(fs)
        Y[1] = Y[0]; s[1] = (s[0] + 1) % (2 if n == 26 else 4)
        np.save(fy, Y); np.save(fs, s)
        good &= expect_fail("a head duplicated onto two triangles",
                            run("verify2627_independent", d, n))

        # 8. displace a head by a minimal vector
        d = copy(p2627)
        fy = os.path.join(d, "data", f"heads{n}_Y.npy")
        Y = np.load(fy); Y[2] = Y[2] + Y[3] // 3; np.save(fy, Y)
        good &= expect_fail("a head displaced", run("verify2627_independent", d, n))

    print()
    print("ALL FALSIFICATION TESTS PASS" if good else "SOME CORRUPTIONS WERE NOT CAUGHT")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

#!/usr/bin/env python3
"""TASK 6 / PART 1: does any other coset carry a larger FORCED class?

GENERAL FORM.  Take a centre c in R^24 and let the class be the lattice points on a sphere
    P(c, rho) = { x - c : x in Lambda, |x - c|^2 = rho }.
For x, y on that sphere,
    <x - c, y - c> = <x,y> - <x,c> - <y,c> + |c|^2
and |x - c|^2 = |y - c|^2 = rho gives <x,c> = (|x|^2 + |c|^2 - rho)/2, so

    <x - c, y - c>  =  rho - |x - y|^2 / 2                                       (*)

-- independent of c.  Hence the normalised cosine is  1 - |x-y|^2/(2 rho), and since
|x - y|^2 >= 4 for distinct lattice points,

    the class is FORCED (every pair at cosine <= 1/4)  <=>  rho <= 8/3,

with equality exactly when some pair has |x - y|^2 = 4.  So rho = 8/3 is the
LARGEST radius at which no choice has to be made.  For rho > 8/3 the class stops being the
whole sphere and becomes a packing problem again.

TWO CONSEQUENCES.
1.  A forced class, normalised, is a spherical code in R^24 whose inner products lie in
        { 1 - k/(2 rho) : k = 4, 6, 8, ... , k <= 4 rho }.
    At rho = 8/3 that is {1/4, -1/8, -1/2, -7/8}, a four-element set, so the
    Delsarte-Goethals-Seidel LP applies and bounds EVERY forced class at once.
2.  Only the centre is left to choose.  With c = (k/q) v the sphere condition is
    |q x - k v|^2 = q^2 rho, an integer, so the enumeration is the same Fincke-Pohst run
    that confirmed 553.

Scaling below: sqrt-8 integers, so rho_our = 8 rho_std and 8/3 -> 64/3.
"""
from __future__ import annotations
import os, sys, json
from fractions import Fraction as F
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "python", "kiss_ref")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, HERE)
from kiss_ref.leech import leech_min_vectors                       # noqa: E402
from fincke_pohst import lll                                       # noqa: E402
from fp_fast import enumerate_ball                                 # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools", "structure"))
from r_subspace_vs_channels import modular_hnf                     # noqa: E402
from bounds.lp_delsarte import solve_lp, rationalise, verify_certificate  # noqa: E402

N = 24
DET = 1 << 36


def lp_bound_for(rho_std):
    """Delsarte LP for the inner-product set forced by a sphere of radius^2 rho_std."""
    ks = [k for k in range(4, 4 * 24) if k % 2 == 0 and F(k) <= 4 * rho_std]
    A = sorted({F(1) - F(k, 1) / (2 * rho_std) for k in ks})
    A = [a for a in A if a < 1]
    best = None
    for deg in (4, 6, 8, 10, 12, 16, 20):
        res = solve_lp(A, deg, N)
        if res.status != 0:
            continue
        cert = rationalise(res)
        b = verify_certificate(cert.f, cert.A, cert.n)
        if best is None or b < best[0]:
            best = (b, deg, cert)
    return A, best


def main():
    C = leech_min_vectors().astype(np.int64)
    rng = np.random.default_rng(17)

    # ---- the LP bound on every forced class ---------------------------------
    A, bb = lp_bound_for(F(8, 3))
    print(f"RESULT P1-LP a forced class normalises to a spherical code in R^24 with inner "
          f"products in {[str(a) for a in A]}")
    print(f"RESULT P1-LP EXACT Delsarte bound on ANY forced coset class: {bb[0]} "
          f"(degree {bb[1]}); the coset at rho = 8/3 achieves 553")

    # ---- candidate centres --------------------------------------------------
    # v of assorted norms, built as sums of two minimal vectors:
    # |x + y|^2 = 8 + 2<x,y>, <x,y> in {-4,-2,-1,0,1,2,4} -> norms 0,4,6,8,10,12,16
    G = C @ C[0]
    vs = {}
    for c, norm in ((-16, 4), (-8, 6), (0, 8), (8, 10), (16, 12)):
        j = int(np.nonzero(G == c)[0][0])
        w = C[0] + C[j]
        assert int(w @ w) == 8 * norm, (norm, int(w @ w))
        vs[norm] = w
    vs[16] = 2 * C[0]
    # norm 24 matters: alpha^2 m = 8/3 has rational solutions only for m = 6 (alpha = 2/3),
    # m = 24 (alpha = 1/3), m = 54 (2/9), m = 96 (1/6), ...  -- these are the centres for
    # which the origin itself lies on the sphere, as it does in the 553 case.
    j = int(np.nonzero(G == 16)[0][0])
    w24 = C[0] + C[j] + C[0] + C[int(np.nonzero(G == 16)[0][1])]
    if int(w24 @ w24) == 8 * 24:
        vs[24] = w24
    else:
        for jj in np.nonzero(G == 16)[0][:200]:
            for kk in np.nonzero(G == 16)[0][:200]:
                cand = C[0] + C[int(jj)] + C[int(kk)]
                if int(cand @ cand) == 8 * 24:
                    vs[24] = cand
                    break
            if 24 in vs:
                break
    print(f"RESULT P1 base vectors built, norms {sorted(vs)} "
          f"(squared norms {[int(vs[m] @ vs[m]) for m in sorted(vs)]} in sqrt-8 units)")

    rows = [[DET if i == j else 0 for j in range(N)] for i in range(N)]
    rows += C[rng.choice(len(C), 400, replace=False)].tolist()
    B = lll(modular_hnf(rows))          # LLL-reduce ONCE; scaling preserves reducedness
    print(f"RESULT P1 LLL-reduced basis of L: shortest row norm^2 = "
          f"{min(sum(t*t for t in r) for r in B)} (the minimum of L is 32)", flush=True)

    print()
    print(f"{'norm m':>6} {'alpha':>7} {'|c|^2':>7}  distance^2 histogram inside the "
          f"forced ball (rho <= 8/3), best forced class")
    best = (0, "none")
    for m in sorted(vs):
        v = vs[m]
        for q in (2, 3, 4, 6):
            Bq = [[q * t for t in r] for r in B]
            for k in range(1, 2 * q + 1):
                if np.gcd(k, q) != 1:
                    continue
                alpha = F(k, q)
                R2 = (q * q * 64) // 3            # floor(q^2 * rho_our), rho_our = 64/3
                s = (k * v).tolist()
                pts, dd = enumerate_ball(Bq, s, R2)
                d2 = Counter(dd)
                if not d2:
                    continue
                bestrho = max(d2.items(), key=lambda kv: kv[1])
                c2 = F(int(alpha ** 2) if False else 0)
                cn = alpha * alpha * F(m)
                top = sorted(d2.items())[:6]
                print(f"{m:>6} {str(alpha):>7} {str(cn):>7}  {top}"
                      f"{' ...' if len(d2) > 6 else ''}   best {bestrho[1]} at "
                      f"|qx-kv|^2 = {bestrho[0]}")
                if bestrho[1] > best[0]:
                    best = (bestrho[1], f"m = {m}, alpha = {alpha}")
    print()
    print(f"RESULT P1 largest forced class found: {best[0]} ({best[1]}); "
          f"beats 553: {best[0] > 553}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

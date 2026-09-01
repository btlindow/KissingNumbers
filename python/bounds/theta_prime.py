"""Cross-checks of the T2.2 scheme LP against spectral bounds on the conflict
graph G (vertices = Leech minimal vectors, x ~ y iff <x,y> = 16, i.e. cos 1/2):

  * Hoffman ratio bound   N (-lambda_min) / (k - lambda_min), k = 4600;
  * Lovasz theta, symmetry-reduced to the scheme ("theta_sym"): the same LP
    as the scheme LP but WITHOUT the sign constraints a_i >= 0 on the free
    classes (only a_6 = 1, a_16 = 0 and a Q >= 0 remain);
  * Schrijver theta' (= the scheme LP, a_i >= 0 kept).

Why these are what they are.
  theta(G) = max { 1^T B 1 : B psd, tr B = 1, B_xy = 0 for xy in E }.
  If B is feasible then so is its average over any group of automorphisms
  of G.  Averaging over Co_0 (which preserves inner products, hence every
  class) gives a matrix in the centraliser algebra of Co_0 on C.  Under the
  assumption that the seven classes are exactly the Co_0-orbitals (Co_2 is
  transitive on each class), that algebra is the Bose-Mesner algebra, so
  B = (1/N) sum_i b_i A_i with b_6 = 1/N (trace), b_5 = 0 (edges), and B psd
  <=> (b Q)_r >= 0 for all r.  With a_i = N b_i:  1^T B 1 = sum_i a_i v_i N /
  N ... more simply, the reduced program is  max sum_i a_i  s.t. a_6 = 1,
  a_5 = 0, a Q >= 0  — the scheme LP without a >= 0.  Schrijver's theta'
  adds B >= 0 entrywise, i.e. a_i >= 0: exactly the Delsarte scheme LP.
  (Without the orbital assumption the reduced programs are still valid upper
  bounds on alpha(G): the inner distribution of any independent set is
  feasible for them.  Schrijver 1979 shows theta' <= Delsarte LP in general,
  with equality for Schurian schemes such as this one.)

  Hoffman.  From a Q >= 0 alone, writing w = a Q: w_0 = sum_i a_i = |S|,
  sum_r w_r = N a_6 = N, sum_r w_r lambda_r = N a_5 = 0 (lambda_r = P_{r5}
  the eigenvalues of A_5).  Hence |S| k = -sum_{r>=1} w_r lambda_r
  <= -lambda_min (N - |S|), i.e. |S| <= N(-lambda_min)/(k - lambda_min).
  This uses no sign constraint on a, so theta_sym <= Hoffman, and
  theta' <= theta_sym.  The dual certificate of the LP that realises the
  Hoffman value is beta_r = (lambda_r - lambda_min)/(k - lambda_min) for
  r >= 1, beta_0 = 0 (checked here exactly).

CLI:  .venv/bin/python python/bounds/theta_prime.py [--json PATH]
      RESULT hoffman=... theta_sym=... theta_prime=... lp=... all_equal=...
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bounds.lp_scheme import (  # type: ignore[no-redef]
        Result,
        SchemeLP,
        certificate_text,
        fmt_frac,
        make_lp,
        solve_scheme_lp,
    )
    from bounds.scheme import (  # type: ignore[no-redef]
        CLASS_DOTS,
        CONFLICT,
        DEFAULT_JSON,
        Eigen,
        eigenmatrices,
        hoffman_bound,
        load_scheme,
        to_fractions,
    )
else:
    from .lp_scheme import Result, SchemeLP, certificate_text, fmt_frac, make_lp, solve_scheme_lp
    from .scheme import (
        CLASS_DOTS,
        CONFLICT,
        DEFAULT_JSON,
        Eigen,
        eigenmatrices,
        hoffman_bound,
        load_scheme,
        to_fractions,
    )


def conflict_eigenvalues(eig: Eigen, conflict: int = CONFLICT) -> list[Fraction]:
    P = to_fractions(eig.P)
    return [P[r][conflict] for r in range(len(P))]


def hoffman_certificate(eig: Eigen, conflict: int = CONFLICT) -> list[Fraction]:
    """beta_r = (lambda_r - lambda_min)/(k - lambda_min) for r >= 1, beta_0 = 0."""
    lam = conflict_eigenvalues(eig, conflict)
    k, lmin = lam[0], min(lam[1:])
    return [Fraction(0)] + [(l - lmin) / (k - lmin) for l in lam[1:]]


def theta_sym(eig: Eigen) -> Result:
    return solve_scheme_lp(make_lp(eig, (CONFLICT,), None, nonneg=False, name="theta_sym"))


def theta_prime(eig: Eigen) -> Result:
    return solve_scheme_lp(make_lp(eig, (CONFLICT,), None, nonneg=True, name="theta_prime"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args(argv)
    s = load_scheme(args.json)
    eig = eigenmatrices(s)
    lam = conflict_eigenvalues(eig, s.conflict)
    print("conflict graph G: N =", s.N, " valency k =", lam[0])
    print("spectrum (eigenvalue^multiplicity):", " ".join(f"{fmt_frac(l)}^{m}" for l, m in zip(lam, eig.m)))
    hb = hoffman_bound(eig, s.conflict)
    lmin = min(lam[1:])
    print(f"Hoffman ratio bound N(-lmin)/(k-lmin) = {s.N}*{-lmin}/({lam[0]}+{-lmin}) = {hb} = {float(hb):.9f}")

    tp = theta_prime(eig)
    ts = theta_sym(eig)
    print("\n== theta' (Schrijver) = scheme LP with a >= 0")
    print(certificate_text(tp))
    print("\n== theta (Lovasz, symmetry-reduced) = scheme LP WITHOUT a_i >= 0 on the free classes")
    print(certificate_text(ts))

    beta_h = hoffman_certificate(eig, s.conflict)
    lp = tp.lp
    proved = lp.verify_certificate(beta_h)
    print("\n== Hoffman's certificate as a dual solution of the LP: beta_r = (lambda_r - lambda_min)/(k - lambda_min), r >= 1")
    print("   beta =", ", ".join(fmt_frac(b) for b in beta_h))
    print(f"   proves |S| <= {proved}   (equals the LP optimum: {proved == tp.exact_value}; "
          f"equals the exact dual found by enumeration: {beta_h == tp.beta})")

    print("\nOrdering: alpha(G) <= theta' <= theta_sym <= Hoffman  ->  "
          f"{fmt_frac(tp.exact_value)} <= {fmt_frac(ts.exact_value)} <= {fmt_frac(hb)}")
    all_equal = tp.exact_value == ts.exact_value == hb
    if all_equal:
        print("All three coincide: dropping a >= 0 gains nothing (theta = theta'), and the LP cannot beat the\n"
              "ratio bound because its optimal dual IS the Hoffman certificate (F(i) = -1 on every allowed class).")
    else:
        print("They differ: see the certificates above for which constraints are active.")
    print(f"\nRESULT hoffman={fmt_frac(hb)} theta_sym={fmt_frac(ts.exact_value)} theta_prime={fmt_frac(tp.exact_value)} "
          f"lp={fmt_frac(tp.exact_value)} hoffman_cert_ok={int(proved == tp.exact_value)} all_equal={int(all_equal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

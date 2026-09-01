"""Three-point (Schrijver / Terwilliger) semidefinite upper bound on |S| for the
Leech minimal-vector scheme (docs/design.md T2.3, docs/reports/T2.3.md).

Setting
-------
V = the N = 196560 minimal vectors of the Leech lattice, Gamma = Co_0 acting on
V, transitively, with 7 orbitals = the inner-product classes
CLASS_DOTS = (-32,-16,-8,0,8,16,32) (valencies 1, 4600, 47104, 93150, 47104,
4600, 1).  S subset V is admissible iff no two of its members have inner
product 16 (class CONFLICT = 5); |S| = 496 is the record and T2.2 proved every
two-point method gives 9360/11 -> 850.

The relaxation
--------------
`orbitals.py` supplies the D = 148 orbitals u of H = Stab(x_0) on V x V, i.e.
the Gamma-orbits O_u of ordered triples (x_0, y, z), with

    size_u   = #{(y,z) : (x_0,y,z) in O_u}    so |O_u| = N * size_u,
    c[u,s,t] = #{w : (y,w) in s, (w,z) in t}  for (y,z) in u,

the structure constants of A = span{B_u} = the centraliser algebra of H
(B_u = the 0/1 matrix of the orbital u).  A is closed under transposition, so
it is a semisimple matrix *-algebra.

For S subset V put lambda_u = #{(a,b,c) in S^3 : (a,b,c) in O_u} and

    x_u = lambda_u / (|S| * size_u).                                       (1)

Three facts, each proved in `_derivation()` below and re-checked numerically in
the tests:

 (P1)  M1 := sum_u x_u B_u  is positive semidefinite.
       Indeed (N/|S|)^{-1} M1 = (1/|Gamma|) sum_{g in Gamma} [x_0 in gS] *
       chi_{gS} chi_{gS}^T, a nonnegative combination of rank-one PSD matrices,
       whose (y,z) entry is lambda_u/|O_u| for u = orbit of (x_0,y,z).
 (P2)  M2 := sum_u (x_{diag(k(u))} - x_u) B_u  is positive semidefinite, by the
       same average with [x_0 in gS] replaced by [x_0 not in gS]
       (diag(k) = the orbital of (y_k, y_k), so x_{diag(k)} = pi_k/(|S| v_k)
       with pi_k = #{(a,b) in S^2 : class(a,b) = k}).
 (P3)  0 <= x_u <= 1, x_{diag(IDENTITY)} = 1, and sum_k v_k x_{diag(k)} = |S|.

So the following is a valid relaxation ("forbidden" = the set of classes that
may not occur inside S; for our problem forbidden = {CONFLICT}):

    maximise    sum_k v_k * x_{diag(k)}
    subject to  x_{diag(IDENTITY)} = 1
                x_u = x_{u^T} = x_{swap(u)}                (S_3 symmetry, (4))
                x_u = 0     whenever a class of u is forbidden
                0 <= x_u <= 1
                M2's coefficients  y_u = x_{diag(k(u))} - x_u >= 0
                M1 >= 0, M2 >= 0.

Making "M >= 0" finite
---------------------
A is a semisimple *-algebra and V is a faithful A-module containing every
irreducible A-module (double centraliser), so for M in A

        M >= 0 as an N x N matrix   <=>   every irreducible representation of A
                                          maps M to a PSD matrix
                                    <=>   the left regular representation of A
                                          maps M to a PSD matrix

(the regular representation contains every irreducible with multiplicity equal
to its dimension).  In the basis {B_u} the left regular representation is
L(M)[w,t] = sum_u x_u c[w,u,t]; it is self-adjoint for the trace form
<B_s, B_t> = delta_{st} size_s, so with Delta = diag(size)

        G(x) := Delta * L(x),   G(x)[w,t] = size_w * sum_u x_u c[w,u,t]      (2)

is a **symmetric integer-coefficient** D x D matrix (symmetric whenever
x_u = x_{u^T}) and

        M(x) >= 0  (N x N)   <=>   G(x) >= 0  (148 x 148).                   (3)

G(x) = Delta^{1/2} Z(x) Delta^{1/2} with Z(x) = Delta^{1/2} L(x) Delta^{-1/2}
the representation in the orthonormal basis B_u/sqrt(size_u); Z is what the
numerical solver sees (it is well scaled: Z(1) has spectrum {N, 0}), G is what
the exact certificate uses (integer coefficients).

Rigorous certificate
--------------------
For PSD rational Y1, Y2, nonnegative rationals mu (multiplier of y >= 0) and
rho (multiplier of x <= 1), and h defined by

    h_q = c_q + <Y1, G_q> + <Y2, Gy_q> + (mu^T Cy)_q - rho_q                 (5)

(G_q, Gy_q = the coefficient matrices of block 1 / block 2 for the S_3-orbit q,
Cy = the matrix of y_u as a function of x), every admissible S satisfies

    |S| = c^T x
        <= c^T x + <Y1,G(x)> + <Y2,G(y)> + mu^T (Cy x) + sum_{q free} rho_q (1-x_q)
         = sum_q h_q x_q + sum_{q free} rho_q
        <= h_e + sum_{q free} rho_q + sum_{q free, q != e} max(0, h_q)       (6)

using x_e = 1, x_q = 0 on forbidden orbits and 0 <= x_q <= 1 from (P3).  Every
term of (6) is an exact rational, so the bound is rigorous whatever the quality
of the numerical dual: an inaccurate dual makes the bound weaker, never wrong.
Y1, Y2 are made exactly PSD by construction (a nonnegative rational combination
of rank-one terms rounded from the solver's eigendecomposition), so no PSD test
of a rounded matrix is ever needed.

`verify_certificate` re-checks a saved certificate from scratch: it rebuilds
G_q and Gy_q from data/scheme/orbitals.json and recomputes (5) and (6) in exact
rational arithmetic, using nothing from the solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from .orbitals import (
    CLASS_DOTS,
    CONFLICT,
    IDENTITY,
    NCLASS,
    Orbitals,
    OrbitalError,
    load_orbitals,
    set_point,
    triple_orbits,
)

__all__ = [
    "Sdp3",
    "terwilliger_dimension",
    "class_cells",
    "verify_certificate",
    "Sdp3Result",
    "psd_exact",
    "load_orbitals",
]


# --------------------------------------------------------------------------- exact PSD


def psd_exact(M: Sequence[Sequence[Fraction]]) -> tuple[bool, int]:
    """Exact PSD test of a symmetric rational matrix by symmetric Gaussian
    elimination with diagonal pivoting.  Returns (is_psd, rank).

    A symmetric matrix is PSD iff the elimination never meets a negative
    diagonal and every zero diagonal has a zero row (for PSD M, M[k][k] = 0
    forces M[k][j] = 0)."""
    n = len(M)
    A = [[Fraction(v) for v in row] for row in M]
    for i in range(n):
        for j in range(i):
            if A[i][j] != A[j][i]:
                raise ValueError(f"matrix is not symmetric at ({i},{j})")
    rank = 0
    active = list(range(n))
    while active:
        # pivot on the largest diagonal entry (keeps the numerators small)
        k = max(active, key=lambda a: A[a][a])
        d = A[k][k]
        if d < 0:
            return False, rank
        if d == 0:
            for j in active:
                if A[k][j] != 0:
                    return False, rank
            active.remove(k)
            continue
        active.remove(k)
        rank += 1
        col = {i: A[i][k] for i in active}
        for i in active:
            ci = col[i]
            if ci == 0:
                continue
            f = ci / d
            Ai, Ak = A[i], A[k]
            for j in active:
                if col[j]:
                    Ai[j] -= f * Ak[j]
    return True, rank


def _to_fraction_matrix(M: np.ndarray) -> list[list[Fraction]]:
    return [[Fraction(v) for v in row] for row in M]


# --------------------------------------------------------------------------- the SDP


@dataclass
class Sdp3Result:
    forbidden: tuple[int, ...]
    status: str
    value: float | None                     # numerical optimum of the relaxation
    x: np.ndarray | None                    # optimal x, one entry per S_3 orbit
    bound: Fraction | None = None           # rigorous rounded bound (exact rational)
    bound_floor: int | None = None          # floor(bound) = the reportable |S| <= ...
    cert_violation: Fraction | None = None  # sum of max(0, h_q) over q != e
    cert_ranks: tuple[int, int] | None = None
    solver: str = ""
    solve_seconds: float = 0.0


class Sdp3:
    """Builder / solver / certifier for the three-point relaxation."""

    def __init__(self, O: Orbitals | None = None):
        self.O = O if O is not None else load_orbitals()
        O = self.O
        D = self.D = O.D
        self.N = O.N
        self.v = np.array(O.valencies, dtype=np.int64)
        self.size = np.array([u.size for u in O.orbitals], dtype=object)
        self.ii = np.array([u.i for u in O.orbitals])
        self.jj = np.array([u.j for u in O.orbitals])
        self.kk = np.array([u.k for u in O.orbitals])
        self.dg = np.array([O.diag(k).id for k in range(NCLASS)])

        tv, members = triple_orbits(O)
        self.tv = np.array(tv)
        self.members = members
        self.Q = Q = len(members)
        self.e = int(self.tv[O.diag(IDENTITY).id])

        # objective: |S| = sum_k v_k x_{diag(k)}
        self.obj = np.zeros(Q, dtype=np.int64)
        for k in range(NCLASS):
            self.obj[self.tv[self.dg[k]]] += int(self.v[k])

        # y_u = x_{diag(k(u))} - x_u  as a D x Q integer matrix
        Cy = np.zeros((D, Q), dtype=np.int64)
        for u in range(D):
            Cy[u, self.tv[self.dg[self.kk[u]]]] += 1
            Cy[u, self.tv[u]] -= 1
        self.Cy = Cy

        # --- float, well-scaled versions Z_q, Zy_q for the solver -----------
        c = O.c                                  # [u,s,t] int64
        L = np.transpose(c, (1, 0, 2))           # L[u,w,t] = c[w,u,t]
        self.sqsize = np.sqrt(np.array([float(s) for s in self.size]))
        scal = self.sqsize[:, None] / self.sqsize[None, :]
        Lf = L.astype(float)
        self.Z = np.array([scal * Lf[ms].sum(axis=0) for ms in members])
        self.Zy = np.einsum("uq,uwt->qwt", Cy.astype(float), scal * Lf)
        # diagonal equilibration (a congruence: PSD-equivalent, better conditioned)
        R = np.abs(self.Z).sum(axis=(0, 2)) + np.abs(self.Zy).sum(axis=(0, 2))
        d = 1.0 / np.sqrt(np.maximum(R, 1e-300))
        self.d = d / d.max()
        self.Zp = self.d[None, :, None] * self.Z * self.d[None, None, :]
        self.Zyp = self.d[None, :, None] * self.Zy * self.d[None, None, :]
        self._G = None
        self._Gy = None

    # --- exact integer coefficient matrices G_q, Gy_q of (2) ----------------
    # G(x)[w,t] = size_w * sum_u x_u c[w,u,t];  x_u = sum_q [u in q] X_q.
    # Built lazily (Python big integers): only the exact checks need them.

    def _build_G(self) -> None:
        O, D, Q = self.O, self.D, self.Q
        Lo = np.transpose(O.c, (1, 0, 2)).astype(object)
        sw = self.size[:, None]
        G = np.empty((Q, D, D), dtype=object)
        for q, ms in enumerate(self.members):
            G[q] = sw * Lo[ms].sum(axis=0)
        Gy = np.empty((Q, D, D), dtype=object)
        CyT = self.Cy.astype(object)
        for q in range(Q):
            acc = np.zeros((D, D), dtype=object)
            for u in range(D):
                if CyT[u, q]:
                    acc = acc + CyT[u, q] * Lo[u]
            Gy[q] = sw * acc
        for q in range(Q):
            if not np.array_equal(G[q], G[q].T) or not np.array_equal(Gy[q], Gy[q].T):
                raise OrbitalError(f"coefficient matrix {q} is not symmetric — "
                                   "the S_3 reduction is wrong")
        self._G, self._Gy = G, Gy

    @property
    def G(self) -> np.ndarray:
        if self._G is None:
            self._build_G()
        return self._G

    @property
    def Gy(self) -> np.ndarray:
        if self._Gy is None:
            self._build_G()
        return self._Gy

    # ---------------------------------------------------------------- helpers

    def forbidden_orbits(self, forbidden: Iterable[int]) -> list[int]:
        """S_3 orbits q that must have x_q = 0 because one of the three
        pairwise classes of the triple is forbidden."""
        forb = tuple(forbidden)
        bad = np.zeros(self.D, dtype=bool)
        for f in forb:
            bad |= (self.ii == f) | (self.jj == f) | (self.kk == f)
        return sorted({int(self.tv[u]) for u in range(self.D) if bad[u]})

    def free_orbits(self, forbidden: Iterable[int]) -> list[int]:
        z = set(self.forbidden_orbits(forbidden))
        return [q for q in range(self.Q) if q not in z]

    def x_of_set(self, name: str) -> tuple[int, np.ndarray]:
        """(|S|, x) for a set recorded in orbitals.json, as exact Fractions,
        one entry per S_3 orbit; raises if the per-orbital values disagree."""
        n, xu = set_point(self.O, name)
        xq = [None] * self.Q
        for u in range(self.D):
            q = int(self.tv[u])
            if xq[q] is None:
                xq[q] = xu[u]
            elif xq[q] != xu[u]:
                raise OrbitalError(f"{name}: x is not constant on S_3 orbit {q}")
        return n, np.array(xq, dtype=object)

    # ---------------------------------------------------------------- checking

    def check_point(self, x: np.ndarray, forbidden: Iterable[int], exact_psd: bool = True) -> dict:
        """Check every constraint of the relaxation at the point x (Fractions).
        Returns a dict of results; never raises for a violated constraint."""
        Q = self.Q
        x = np.array([Fraction(t) for t in x], dtype=object)
        y = self.Cy.astype(object) @ x
        z = self.forbidden_orbits(forbidden)
        out = {
            "objective": sum(int(self.obj[q]) * x[q] for q in range(Q)),
            "x_e": x[self.e],
            "nonneg": all(t >= 0 for t in x),
            "at_most_one": all(t <= 1 for t in x),
            "forbidden_zero": all(x[q] == 0 for q in z),
            "y_nonneg": all(t >= 0 for t in y),
        }
        if exact_psd:
            G1 = _to_fraction_matrix(np.tensordot(x, self.G, axes=(0, 0)))
            G2 = _to_fraction_matrix(np.tensordot(x, self.Gy, axes=(0, 0)))
            ok1, r1 = psd_exact(G1)
            ok2, r2 = psd_exact(G2)
            out.update(psd_block1=ok1, rank_block1=r1, psd_block2=ok2, rank_block2=r2)
        else:
            xf = np.array([float(t) for t in x])
            for tag, B in (("block1", self.Z), ("block2", self.Zy)):
                Mf = np.tensordot(xf, B, axes=(0, 0))
                Mf = (Mf + Mf.T) / 2
                w = np.linalg.eigvalsh(Mf)
                scale = max(abs(w).max(), 1.0)
                out[f"psd_{tag}"] = bool(w.min() > -1e-9 * scale)
                out[f"min_eig_{tag}"] = float(w.min())
                out[f"scale_{tag}"] = float(scale)
        out["feasible"] = all(
            bool(out[k]) for k in ("nonneg", "at_most_one", "forbidden_zero", "y_nonneg",
                                   "psd_block1", "psd_block2")
        ) and out["x_e"] == 1
        return out

    # ---------------------------------------------------------------- solving

    def solve(self, forbidden: Iterable[int] = (CONFLICT,), solver: str = "CLARABEL",
              verbose: bool = False, **kw) -> Sdp3Result:
        """Solve the relaxation numerically.  The PSD blocks are stated in the
        diagonally equilibrated basis Zp = d Z d (a congruence, so PSD-
        equivalent); d is folded back into the certificate."""
        import time

        import cvxpy as cp

        forb = tuple(forbidden)
        Q = self.Q
        z = self.forbidden_orbits(forb)
        x = cp.Variable(Q, name="x")
        c_nn = x >= 0
        c_ub = x <= 1
        c_e = x[self.e] == 1
        cons = [c_nn, c_ub, c_e]
        if z:
            cons.append(x[z] == 0)
        c_p1 = sum(x[q] * self.Zp[q] for q in range(Q)) >> 0
        c_p2 = sum(x[q] * self.Zyp[q] for q in range(Q)) >> 0
        c_y = self.Cy @ x >= 0
        cons += [c_p1, c_p2, c_y]
        scale = float(self.N)
        prob = cp.Problem(cp.Maximize((self.obj @ x) / scale), cons)
        t0 = time.time()
        prob.solve(solver=solver, verbose=verbose, **kw)
        dt = time.time() - t0
        val = None if prob.value is None else float(prob.value) * scale

        def dv(c):
            try:
                return None if c.dual_value is None else np.array(c.dual_value, dtype=float) * scale
            except Exception:
                return None

        res = Sdp3Result(forbidden=forb, status=str(prob.status), value=val,
                         x=None if x.value is None else np.array(x.value),
                         solver=solver, solve_seconds=dt)
        res._duals = {"Y1": dv(c_p1), "Y2": dv(c_p2), "mu": dv(c_y), "rho": dv(c_ub)}  # type: ignore[attr-defined]
        return res

    # ------------------------------------------------------------ certificate

    def _rational_psd(self, Y: np.ndarray, bits: int, rel_drop: float
                      ) -> tuple[np.ndarray, int, list]:
        """Round a numerical dual block Y (in the equilibrated Zp basis) to an
        exactly PSD rational matrix Yhat = Yint / 2**den_exp expressed in the
        G basis, so that <Yhat, G(x)> ~ <Y, Zp(x)> while <Yhat, G(x)> >= 0
        holds EXACTLY for every G(x) >= 0.

        Zp(x) = D Z(x) D and Z(x) = Delta^{-1/2} G(x) Delta^{-1/2}, so
        <Y, Zp(x)> = sum_i lam_i p_i^T G(x) p_i with p_i = Delta^{-1/2} D q_i
        (q_i, lam_i = the eigenpairs of Y).  Yhat = sum_i lam_i p_i p_i^T is a
        nonnegative combination of rank-one terms: PSD by construction, whatever
        the rounding.  Returns (Yint, den_exp, [(m_i, P_i)]) with
        Yint = sum_i m_i P_i P_i^T and Yhat = Yint / 2**den_exp."""
        Y = (Y + Y.T) / 2
        w, V = np.linalg.eigh(Y)
        wmax = max(float(w.max()), 0.0)
        keep = [i for i in range(len(w)) if w[i] > rel_drop * max(wmax, 1e-300)]
        D = self.D
        if not keep:
            return np.zeros((D, D), dtype=object), 0, []
        te = bits - int(math.floor(math.log2(wmax))) - 1
        terms = []                                   # (M_i, P_i, den_i)
        for i in keep:
            m = int(round(float(w[i]) * 2.0 ** te)) if te >= 0 else int(round(float(w[i]) / 2.0 ** (-te)))
            if m <= 0:
                continue
            p = (self.d * V[:, i]) / self.sqsize
            pmax = float(np.abs(p).max())
            if pmax == 0:
                continue
            se = bits - int(math.floor(math.log2(pmax))) - 1
            P = np.rint(p * 2.0 ** se).astype(np.int64).astype(object)
            terms.append((m, P, te + 2 * se))
        if not terms:
            return np.zeros((D, D), dtype=object), 0, []
        T = max(t[2] for t in terms)
        Yint = np.zeros((D, D), dtype=object)
        scaled = []
        for m, P, den in terms:
            ms = m << (T - den)
            Yint = Yint + ms * np.outer(P, P)
            scaled.append((ms, P))
        return Yint, T, scaled

    @staticmethod
    def _contract(Yint: np.ndarray, den_exp: int, Gq: np.ndarray) -> Fraction:
        return Fraction(int((Yint * Gq).sum()), 1 << den_exp)

    def certify(self, res: Sdp3Result, bits: int = 52, rel_drop: float = 1e-9,
                verbose: bool = False) -> Sdp3Result:
        """Turn the numerical dual of `res` into an exact rational certificate.
        Validity does not depend on the numerics: Y1, Y2 are PSD and mu, rho
        nonnegative by construction, and the bound (6) is computed exactly."""
        duals = getattr(res, "_duals", None)
        if duals is None or duals.get("Y1") is None:
            raise OrbitalError("no dual solution available (solve() first)")
        free = self.free_orbits(res.forbidden)
        ranks = []
        psd_terms = []
        a = [Fraction(0)] * self.Q
        for key, Gs in (("Y1", self.G), ("Y2", self.Gy)):
            Y = duals[key]
            wY = np.linalg.eigvalsh((Y + Y.T) / 2)
            if -wY.min() > wY.max():                    # cvxpy sign convention
                Y = -Y
            Yint, den, terms = self._rational_psd(Y, bits, rel_drop)
            ranks.append(len(terms))
            psd_terms.append((den, terms))
            for q in range(self.Q):
                a[q] += self._contract(Yint, den, Gs[q])

        den = 1 << bits
        mu_raw = duals.get("mu")
        rho_raw = duals.get("rho")
        mu_raw = np.zeros(self.D) if mu_raw is None else np.asarray(mu_raw, dtype=float).ravel()
        rho_raw = np.zeros(self.Q) if rho_raw is None else np.asarray(rho_raw, dtype=float).ravel()
        best = None
        for smu in (1.0, -1.0):
            mu = [Fraction(int(round(max(0.0, smu * t) * den)), den) for t in mu_raw]
            muCy = [sum(mu[u] * int(self.Cy[u, q]) for u in range(self.D) if self.Cy[u, q])
                    for q in range(self.Q)]
            h = [Fraction(int(self.obj[q])) + a[q] + muCy[q] for q in range(self.Q)]
            for srho in (1.0, -1.0):
                rho = [Fraction(int(round(max(0.0, srho * t) * den)), den) for t in rho_raw]
                g = [h[q] - rho[q] for q in range(self.Q)]
                viol = sum((g[q] for q in free if q != self.e and g[q] > 0), Fraction(0))
                bound = g[self.e] + sum(rho[q] for q in free) + viol
                if best is None or bound < best[0]:
                    best = (bound, viol, g[self.e], g, rho, mu)
        bound, viol, he, g, rho, mu = best
        res.bound = bound
        res.bound_floor = math.floor(bound)
        res.cert_violation = viol
        res.cert_ranks = (ranks[0], ranks[1])
        res._cert = {                                   # type: ignore[attr-defined]
            "forbidden": list(res.forbidden),
            "free": self.free_orbits(res.forbidden),
            "e": self.e,
            "h": g,                 # h_q - rho_q, exact Fractions
            "rho": rho,             # >= 0, exact Fractions
            "mu": mu,               # >= 0, exact Fractions
            "a": list(a),           # <Y1,G_q> + <Y2,Gy_q>, exact Fractions
            "psd_terms": psd_terms,  # per block: (den_exp, [(m_i, P_i)])
            "bound": bound,
        }
        if verbose:
            print(f"  certificate: h_e = {float(he):.9f}, rounding slack = {float(viol):.3e}, "
                  f"bound = {float(bound):.9f} -> |S| <= {res.bound_floor}")
        return res

    @staticmethod
    def certificate(res: Sdp3Result) -> dict:
        c = getattr(res, "_cert", None)
        if c is None:
            raise OrbitalError("no certificate (certify() first)")
        return c

    def save_certificate(self, res: Sdp3Result, path: str) -> str:
        """Write the exact certificate so a stranger can re-check the bound:
        the rank-one decompositions of Y1, Y2 (integer vectors P_i, integer
        weights m_i, a common power-of-two denominator), mu, rho and the
        resulting h."""
        import json

        c = self.certificate(res)

        def fr(x):
            x = Fraction(x)
            return [int(x.numerator), int(x.denominator)]

        blocks = []
        for den, terms in c["psd_terms"]:
            blocks.append({"den_exp": int(den),
                           "terms": [{"m": int(m), "p": [int(t) for t in P]} for m, P in terms]})
        doc = {
            "task": "T2.3",
            "description": ("Exact dual certificate of the three-point SDP bound on |S| for the "
                            "Leech minimal-vector scheme. bound = h[e] + sum_{q free} rho[q] + "
                            "sum_{q free, q != e} max(0, h[q]); h[q] = obj[q] + <Y1,G_q> + "
                            "<Y2,Gy_q> + (mu^T Cy)_q - rho[q]; Y_b = sum_i (m_i / 2^den_exp) "
                            "p_i p_i^T with p_i integer vectors, so Y_b >= 0 by construction."),
            "N": self.N, "D": self.D, "vars": self.Q, "e": self.e,
            "forbidden_classes": [int(f) for f in res.forbidden],
            "forbidden_dots": [int(CLASS_DOTS[f]) for f in res.forbidden],
            "free_orbits": [int(q) for q in c["free"]],
            "objective": [int(t) for t in self.obj],
            "orbit_members": [[int(u) for u in ms] for ms in self.members],
            "Cy": [[int(t) for t in row] for row in self.Cy],
            "blocks": blocks,
            "mu": [fr(t) for t in c["mu"]],
            "rho": [fr(t) for t in c["rho"]],
            "h": [fr(t) for t in c["h"]],
            "bound": fr(c["bound"]),
            "bound_float": float(c["bound"]),
            "bound_floor": int(math.floor(c["bound"])),
            "solver": res.solver, "solver_status": res.status, "solver_value": res.value,
        }
        with open(path, "w") as f:
            json.dump(doc, f)
        return path


# ------------------------------------------------------ independent re-check


def verify_certificate(path: str, orbitals_path: str | None = None) -> dict:
    """Re-check a certificate written by Sdp3.save_certificate from scratch:
    rebuild the coefficient matrices from orbitals.json, recompute
    h_q = obj_q + <Y1,G_q> + <Y2,Gy_q> + (mu^T Cy)_q - rho_q in exact rational
    arithmetic and re-derive the bound.  Uses nothing from the solver.

    Returns a dict with the recomputed bound and the comparison against the
    stored one; raises OrbitalError on any inconsistency."""
    import json

    with open(path) as f:
        doc = json.load(f)
    O = load_orbitals(orbitals_path) if orbitals_path else load_orbitals()
    D, Q = O.D, len(doc["orbit_members"])
    if D != int(doc["D"]) or O.N != int(doc["N"]):
        raise OrbitalError("certificate was made for a different scheme")
    size = [u.size for u in O.orbitals]
    kk = [u.k for u in O.orbitals]
    ii = [u.i for u in O.orbitals]
    jj = [u.j for u in O.orbitals]
    members = [[int(u) for u in ms] for ms in doc["orbit_members"]]
    tv = [-1] * D
    for q, ms in enumerate(members):
        for u in ms:
            tv[u] = q
    if min(tv) < 0:
        raise OrbitalError("orbit_members does not cover all orbitals")
    Cy = np.array(doc["Cy"], dtype=np.int64)
    # rebuild Cy from scratch and compare
    dg = {}
    for k in range(NCLASS):
        dg[k] = O.diag(k).id
    Cy2 = np.zeros((D, Q), dtype=np.int64)
    for u in range(D):
        Cy2[u, tv[dg[kk[u]]]] += 1
        Cy2[u, tv[u]] -= 1
    if not np.array_equal(Cy, Cy2):
        raise OrbitalError("Cy in the certificate does not match orbitals.json")
    obj = np.zeros(Q, dtype=np.int64)
    for k in range(NCLASS):
        obj[tv[dg[k]]] += O.valencies[k]
    if [int(t) for t in obj] != [int(t) for t in doc["objective"]]:
        raise OrbitalError("objective in the certificate does not match orbitals.json")
    # forbidden orbits
    forb = [int(f) for f in doc["forbidden_classes"]]
    bad = set()
    for u in range(D):
        if ii[u] in forb or jj[u] in forb or kk[u] in forb:
            bad.add(tv[u])
    free = sorted(set(range(Q)) - bad)
    if free != [int(q) for q in doc["free_orbits"]]:
        raise OrbitalError("free orbit list does not match the forbidden classes")
    # coefficient matrices
    Lo = np.transpose(O.c, (1, 0, 2)).astype(object)
    sw = np.array(size, dtype=object)[:, None]
    G = [sw * Lo[ms].sum(axis=0) for ms in members]
    Gy = []
    for q in range(Q):
        acc = np.zeros((D, D), dtype=object)
        for u in range(D):
            if Cy[u, q]:
                acc = acc + int(Cy[u, q]) * Lo[u]
        Gy.append(sw * acc)
    # the two PSD multipliers, as integer matrices over 2**den_exp
    a = [Fraction(0)] * Q
    for blk, Gs in zip(doc["blocks"], (G, Gy)):
        den = int(blk["den_exp"])
        Yint = np.zeros((D, D), dtype=object)
        for term in blk["terms"]:
            m = int(term["m"])
            if m < 0:
                raise OrbitalError("negative weight in a PSD block: not a PSD multiplier")
            P = np.array([int(t) for t in term["p"]], dtype=object)
            Yint = Yint + m * np.outer(P, P)
        for q in range(Q):
            a[q] += Fraction(int((Yint * Gs[q]).sum()), 1 << den)
    mu = [Fraction(n, d) for n, d in doc["mu"]]
    rho = [Fraction(n, d) for n, d in doc["rho"]]
    if any(t < 0 for t in mu) or any(t < 0 for t in rho):
        raise OrbitalError("negative mu / rho: not a valid multiplier")
    h = []
    for q in range(Q):
        val = Fraction(int(obj[q])) + a[q] - rho[q]
        val += sum(mu[u] * int(Cy[u, q]) for u in range(D) if Cy[u, q])
        h.append(val)
    stored = [Fraction(n, d) for n, d in doc["h"]]
    if h != stored:
        raise OrbitalError("recomputed h differs from the stored h")
    e = int(doc["e"])
    bound = h[e] + sum(rho[q] for q in free) \
        + sum(h[q] for q in free if q != e and h[q] > 0)
    if bound != Fraction(*doc["bound"]):
        raise OrbitalError("recomputed bound differs from the stored bound")
    return {"bound": bound, "bound_floor": math.floor(bound), "D": D, "vars": Q,
            "forbidden_dots": doc["forbidden_dots"], "matches_stored": True}


# ------------------------------------------------------- the Terwilliger algebra


def terwilliger_dimension(O: Orbitals, prime: int = (1 << 61) - 1) -> int:
    """dim T(x), the Terwilliger algebra of the scheme at x = the subalgebra of
    the D-dimensional centraliser algebra A = span{B_u} generated by the dual
    idempotents E_i* (= B_{diag(i)}, the diagonal of the class-i sphere) and the
    adjacency matrices A_k (= sum of the orbitals u with k(u) = k).

    Computed by closing the generated subspace under left and right
    multiplication by the generators, in the structure constants c, over
    GF(prime) (a large prime).  Rank over GF(p) <= rank over Q <= D, so a
    computed dimension of D proves dim_Q T(x) = D."""
    D = O.D
    c = O.c
    gens = []
    for i in range(NCLASS):
        g = np.zeros(D, dtype=np.int64)
        g[O.diag(i).id] = 1
        gens.append(g)
    for k in range(NCLASS):
        g = np.zeros(D, dtype=np.int64)
        for u in O.orbitals:
            if u.k == k:
                g[u.id] = 1
        gens.append(g)
    # right/left multiplication matrices: (a g)_w = sum_s a_s R[w,s], (g a)_w = sum_t L[w,t] a_t
    R = [((c * g[None, None, :]).sum(axis=2) % prime).astype(object) for g in gens]
    Lm = [((c * g[None, :, None]).sum(axis=1) % prime).astype(object) for g in gens]
    basis: list[np.ndarray] = []
    piv: list[int] = []

    def add(vec):
        v = np.array([int(t) % prime for t in vec], dtype=object)
        for p_, row in zip(piv, basis):
            if v[p_]:
                v = (v - v[p_] * row) % prime
        nz = [j for j in range(D) if v[j]]
        if not nz:
            return None
        j = nz[0]
        v = (v * pow(int(v[j]), prime - 2, prime)) % prime
        basis.append(v)
        piv.append(j)
        return v

    elems = []
    for g in gens:
        if add(g) is not None:
            elems.append(np.array([int(t) for t in g], dtype=object))
    i = 0
    while i < len(elems):
        a = elems[i]
        for Rg, Lg in zip(R, Lm):
            for prod in ((Rg @ a) % prime, (Lg @ a) % prime):
                if add(prod) is not None:
                    elems.append(prod)
        i += 1
    return len(basis)


def class_cells(O: Orbitals) -> int:
    """Number of nonempty class-triples (i,j,k) = dim span{E_i* A_k E_j*}."""
    return len({(u.i, u.j, u.k) for u in O.orbitals})


# --------------------------------------------------------------------------- CLI


def _fmt(x) -> str:
    return "None" if x is None else (f"{float(x):.6f}" if not isinstance(x, int) else str(x))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--bits", type=int, default=52)
    ap.add_argument("--no-exact-psd", action="store_true",
                    help="check recorded sets with float eigenvalues only (fast)")
    ap.add_argument("--save-certificate", default=None,
                    help="write the exact dual certificate of the |S| bound to this JSON file")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    S = Sdp3()
    O = S.O
    print(f"orbitals of Stab(x) on ordered pairs: D = {S.D}; class-cells (i,j,k): "
          f"{class_cells(O)}; dim T(x) = {terwilliger_dimension(O)}")
    print(f"S_3 orbits of triples = SDP variables: {S.Q}; identity orbit e = {S.e}; "
          f"two {S.D} x {S.D} PSD blocks (left regular representation of the "
          f"centraliser algebra of Stab(x))")

    ok = True

    # --- the decisive test: real sets must be feasible ----------------------
    for name in sorted(O.sets):
        n, xq = S.x_of_set(name)
        r = S.check_point(xq, [CONFLICT], exact_psd=not a.no_exact_psd)
        print(f"feasibility of {name} (|S| = {n}): objective = {r['objective']}, x_e = {r['x_e']}, "
              f"x >= 0 {r['nonneg']}, x <= 1 {r['at_most_one']}, forbidden zero "
              f"{r['forbidden_zero']}, y >= 0 {r['y_nonneg']}, PSD1 {r['psd_block1']}, "
              f"PSD2 {r['psd_block2']} -> FEASIBLE {r['feasible']}")
        ok = ok and r["feasible"] and r["objective"] == n

    # --- the whole scheme with nothing forbidden ----------------------------
    ones = np.array([Fraction(1)] * S.Q, dtype=object)
    r = S.check_point(ones, [], exact_psd=not a.no_exact_psd)
    print(f"feasibility of S = C (x == 1, nothing forbidden): objective = {r['objective']} "
          f"(= N: {r['objective'] == S.N}), PSD1 {r['psd_block1']}, PSD2 {r['psd_block2']}")
    ok = ok and r["objective"] == S.N and r["psd_block1"] and r["psd_block2"]

    # --- the three bounds ---------------------------------------------------
    results = {}
    for tag, forb in (("nothing", ()), ("{16}", (CONFLICT,)), ("{16,8}", (CONFLICT, 4))):
        res = S.solve(forb, solver=a.solver, verbose=a.verbose)
        S.certify(res, bits=a.bits, verbose=a.verbose)
        results[tag] = res
        print(f"forbid {tag:7s}: relaxation optimum {_fmt(res.value)} (status {res.status}, "
              f"{res.solve_seconds:.1f} s)  ->  RIGOROUS |S| <= {res.bound_floor}  "
              f"[exact rational bound {float(res.bound):.6f}, dual ranks {res.cert_ranks}, "
              f"rounding slack {float(res.cert_violation):.3e}]")

    b_none = results["nothing"].bound_floor
    b_16 = results["{16}"].bound_floor
    b_168 = results["{16,8}"].bound_floor
    sane = (b_none >= S.N and 496 <= b_16 <= 850 and b_168 == 48)
    ok = ok and sane
    print(f"sanity: nothing forbidden {b_none} >= N = {S.N}; forbid 16 -> {b_16} in [496, 850]; "
          f"forbid 16 and 8 -> {b_168} (orthoplex bound 48)")
    if a.save_certificate:
        S.save_certificate(results["{16}"], a.save_certificate)
        print(f"certificate written to {a.save_certificate}")
    print(f"RESULT ok={1 if ok else 0} D={S.D} vars={S.Q} bound16={b_16} "
          f"bound_none={b_none} bound168={b_168}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

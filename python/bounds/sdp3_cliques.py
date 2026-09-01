"""Clique cuts for the three-point SDP on the Leech minimal-vector scheme
(task B1; docs/reports/B1.md).  Builds on bounds.sdp3_scheme (T2.3).

The clique number of the conflict graph
---------------------------------------
G = (V, E): V = the N = 196560 minimal vectors, uv in E iff <u,v> = 16.
tools/clique_g proves omega(G) = 24:

 * upper bound: k vectors of norm 32 with pairwise inner product 16 have Gram
   16(I_k + J_k), which is positive definite (eigenvalues 16(k+1) and 16), so
   the vectors are linearly independent in R^24 and k <= 24;
 * lower bound: K24 below is an explicit 24-clique (verified in exact integer
   arithmetic here at import time and by tools/clique_g / test_clique_g).

Lifted clique inequalities (the derivation)
-------------------------------------------
Let S be admissible (independent in G), Gamma the verified transitive group of
T2.2/T2.3, and x_u = lambda_u / (|S| size_u) the SDP variables of T2.3, where
lambda_u = #{(p,q,r) in S^3 : (p,q,r) in O_u} and O_u is the Gamma-orbit of
ordered triples through orbital u.  Fix ANY pair (a, b) in V^2 with
i = class(a,b) != CONFLICT, and ANY clique K of G.  For every g in Gamma the
set gS is again independent, so it meets the clique gK... more directly:

    [a in gS][b in gS] * sum_{v in K} [v in gS]  <=  [a in gS][b in gS]   (*)

because |K ∩ gS| <= 1.  Averaging (*) over g in Gamma, each term with
u = u(a,b,v) = the orbital of the ordered triple (a,b,v) gives

    avg_g [a in gS][b in gS][v in gS] = lambda_u / |O_u| = (|S|/N) x_u ,
    avg_g [a in gS][b in gS]          = pi_i / (N v_i)   = (|S|/N) x_diag(i),

(pi_i = the number of class-i pairs of S; both are the (P1) computation of
sdp3_scheme).  Dividing by |S|/N:

    CUT1(a,b,K):   sum_{v in K} x_{u(a,b,v)}  <=  x_diag(i).            (C1)

The same average of  [a in gS](1 - [b in gS]) sum_{v in K} [v in gS]
                                          <= [a in gS](1 - [b in gS]) gives

    CUT2(a,b,K):   sum_{v in K} (x_diag(j_v) - x_{u(a,b,v)})
                       <= 1 - x_diag(i),   j_v = class(a,v).            (C2)

Both are exact linear inequalities on the T2.3 variables, valid for every
admissible S — the derivation never uses the relaxation.  a = b (i = the
identity class) is allowed: (C1) becomes sum_v x_diag(class(a,v)) <= 1.
Triples with a forbidden pairwise class keep their coefficients: they sit on
variables that are exactly 0 for admissible S (and fixed to 0 in the SDP).

Identifying u(a,b,v):  the 148 orbitals are separated by their four-point
histograms (T2.3 section 1), and the class triple (i,j,k) determines the
orbital except in the single split cell (3,3,3) (orthogonal triples,
43164 = 924 + 42240).  There the histogram entry H[5,5,5] = #{w : w is a
common G-neighbour of a, b, v} is 2 for the small orbital (id 73) and 0 for
the big one (id 74); w must be one of the p^3_{55} = 44 common G-neighbours
of (a,b), so the test costs 44 dot products.  `selfcheck_classifier`
re-derives the exact split 924 + 42240 from scratch for a random orthogonal
pair, and the GPU-verified histograms guarantee no third value of H[5,5,5]
exists.

Certificate with cuts
---------------------
With rows A x <= t (integer A, t in {0,1}) and multipliers nu >= 0, (5)/(6)
of sdp3_scheme extend to

    h_q   = obj_q + <Y1,G_q> + <Y2,Gy_q> + (mu^T Cy)_q - (nu^T A)_q - rho_q
    bound = h_e + sum_{q free} rho_q + nu^T t + sum_{q free, q != e} max(0,h_q)

and remain exact rationals.  `verify_cuts_certificate` re-checks a saved
certificate from scratch, INCLUDING re-deriving every cut row from its
recorded instance (a, b, K) with the classifier above and re-verifying that K
is a clique — nothing from the solver or from this module's earlier run is
trusted.

Result (see docs/reports/B1.md): the baseline optimum already satisfies every
sampled cut, the strongest ones with equality to solver precision, so the
rigorous bound stays |S| <= 837.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from .orbitals import CLASS_DOTS, CONFLICT, IDENTITY, NCLASS, OrbitalError
from .sdp3_scheme import Sdp3, Sdp3Result

__all__ = [
    "K24",
    "load_vectors",
    "CliqueCutter",
    "Sdp3Cuts",
    "verify_cuts_certificate",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VECTORS = os.path.join(_HERE, "..", "..", "data", "leech_min.i8")

# The explicit maximum clique of G found by tools/clique_g (canonical vertex
# indices of data/leech_min.i8).  Verified by verify_k24() before use.
K24 = (0, 1, 2386, 2398, 2399, 2403, 2404, 2406, 2407, 2410, 2411, 2414,
       2417, 2418, 2419, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 65390)


def load_vectors(path: str | os.PathLike = DEFAULT_VECTORS) -> np.ndarray:
    C = np.fromfile(path, dtype=np.int8).astype(np.int32).reshape(-1, 24)
    if C.shape[0] != 196560:
        raise OrbitalError(f"{path}: expected 196560 vectors, got {C.shape[0]}")
    return C


def class_of(dots: np.ndarray) -> np.ndarray:
    """Canonical class index 0..6 for inner products in {-32,-16,-8,0,8,16,32}."""
    t = (np.asarray(dots) + 32) >> 3
    return t - (t > 1) - (t > 7)


def verify_k24(C: np.ndarray, clique: Sequence[int] = K24) -> None:
    """Exact integer verification that `clique` is a clique of G (pairwise
    inner products 16, norms 32).  Raises OrbitalError otherwise."""
    V = C[list(clique)].astype(np.int64)
    Gr = V @ V.T
    n = len(clique)
    if not np.all(np.diag(Gr) == 32):
        raise OrbitalError("clique member with norm != 32")
    off = Gr[~np.eye(n, dtype=bool)]
    if not np.all(off == 16):
        raise OrbitalError("clique pair with inner product != 16")


# --------------------------------------------------------------------------- cuts


@dataclass
class Cut:
    kind: str                 # "C1" or "C2"
    a: int
    b: int
    K: tuple[int, ...]
    row: np.ndarray           # int64, length Q; constraint row . x <= rhs
    rhs: int                  # 0 for C1, 1 for C2


class CliqueCutter:
    """Builds valid cut rows (C1)/(C2) for concrete instances (a, b, K)."""

    def __init__(self, S: Sdp3, C: np.ndarray):
        self.S = S
        self.C = C
        O = S.O
        self.cell: dict[tuple[int, int, int], list[int]] = {}
        for u in O.orbitals:
            self.cell.setdefault((u.i, u.j, u.k), []).append(u.id)
        # the only split cell is (3,3,3): orbitals (small, big) by orbit size
        split = sorted(self.cell[(3, 3, 3)], key=lambda u: O.orbitals[u].osize)
        if len(split) != 2 or [O.orbitals[u].osize for u in split] != [924, 42240]:
            raise OrbitalError("unexpected split of the (3,3,3) cell")
        for ijk, us in self.cell.items():
            if len(us) > 1 and ijk != (3, 3, 3):
                raise OrbitalError(f"unexpected split cell {ijk}")
        self.small333, self.big333 = split
        self.qdiag = [int(S.tv[S.dg[k]]) for k in range(NCLASS)]

    # -- orbital of the ordered triple (a, b, v) ----------------------------
    def orbital(self, i: int, j: int, k: int, a: int, b: int, v: int,
                conf_ab: np.ndarray | None) -> int:
        us = self.cell[(i, j, k)]
        if len(us) == 1:
            return us[0]
        # (3,3,3): count common G-neighbours of {a,b,v} among the 44 common
        # G-neighbours of (a,b); the histogram value is 2 (small) or 0 (big)
        n5 = int(np.count_nonzero((self.C[conf_ab] @ self.C[v]) == 16))
        if n5 == 2:
            return self.small333
        if n5 == 0:
            return self.big333
        raise OrbitalError(f"(3,3,3) triple with {n5} common conflict neighbours")

    def _conf_ab(self, a: int, b: int) -> np.ndarray:
        da = self.C @ self.C[a]
        db = self.C @ self.C[b]
        return np.where((da == 16) & (db == 16))[0]

    def cuts(self, a: int, b: int, K: Sequence[int], kinds: Iterable[str] = ("C1",),
             ) -> list[Cut]:
        S, C = self.S, self.C
        i = int(class_of(int(C[a] @ C[b])))
        if i == CONFLICT:
            raise OrbitalError("base pair (a,b) is an edge of G: the cut is vacuous")
        K = tuple(int(v) for v in K)
        verify_k24(C, K)  # K must be a clique (any size)
        need333 = i == 3
        conf_ab = self._conf_ab(a, b) if need333 else None
        if conf_ab is not None and i == 3 and len(conf_ab) != 44:
            raise OrbitalError("p^3_55 != 44 for the base pair")
        r1 = np.zeros(S.Q, dtype=np.int64)
        r2 = np.zeros(S.Q, dtype=np.int64)
        for v in K:
            j = int(class_of(int(C[a] @ C[v])))
            k = int(class_of(int(C[b] @ C[v])))
            u = self.orbital(i, j, k, a, b, v, conf_ab)
            q = int(S.tv[u])
            r1[q] += 1
            r2[self.qdiag[j]] += 1
            r2[q] -= 1
        r1[self.qdiag[i]] -= 1
        r2[self.qdiag[i]] += 1
        out = []
        if "C1" in kinds:
            out.append(Cut("C1", a, b, K, r1, 0))
        if "C2" in kinds:
            out.append(Cut("C2", a, b, K, r2, 1))
        return out

    # -- sampling -----------------------------------------------------------
    def sample_pool(self, rng: np.random.Generator, cliques: Sequence[Sequence[int]],
                    pairs_per_class: int = 25,
                    classes: Sequence[int] = (0, 1, 2, 3, 4, 6),
                    kinds: Iterable[str] = ("C1",)) -> list[Cut]:
        """Random base pairs of each allowed class x the given cliques,
        deduplicated by (row, rhs)."""
        C = self.C
        N = C.shape[0]
        seen: set = set()
        pool: list[Cut] = []
        for k in classes:
            for _ in range(pairs_per_class):
                a = int(rng.integers(N))
                if k == IDENTITY:
                    b = a
                else:
                    cand = np.where(class_of(C @ C[a]) == k)[0]
                    b = int(rng.choice(cand))
                for K in cliques:
                    for cut in self.cuts(a, b, K, kinds):
                        key = (cut.rhs, tuple(cut.row.tolist()))
                        if key not in seen:
                            seen.add(key)
                            pool.append(cut)
        return pool

    # -- the mandatory validity gate ---------------------------------------
    def check_sets_exact(self, cuts: Sequence[Cut]) -> dict[str, dict[str, int]]:
        """Every recorded set (the 496 and the 488) must satisfy every cut in
        exact rational arithmetic.  Raises OrbitalError on any violation.
        Returns {set: {'eq': #tight, 'lt': #strict}}."""
        S = self.S
        out = {}
        for name in sorted(S.O.sets):
            n, xq = S.x_of_set(name)
            eq = lt = 0
            for cut in cuts:
                val = sum(Fraction(int(cut.row[q])) * xq[q]
                          for q in range(S.Q) if cut.row[q])
                if val > cut.rhs:
                    raise OrbitalError(
                        f"cut {cut.kind}(a={cut.a}, b={cut.b}, |K|={len(cut.K)}) is "
                        f"VIOLATED by {name}: {val} > {cut.rhs} — the cut is wrong")
                if val == cut.rhs:
                    eq += 1
                else:
                    lt += 1
            out[name] = {"eq": eq, "lt": lt}
        return out

    def selfcheck_classifier(self, rng: np.random.Generator) -> None:
        """Re-derive the exact (3,3,3) split 43164 = 924 + 42240 for a random
        orthogonal pair with the 44-neighbour rule.  Raises on mismatch."""
        C = self.C
        a = int(rng.integers(C.shape[0]))
        da = C @ C[a]
        b = int(rng.choice(np.where(class_of(da) == 3)[0]))
        db = C @ C[b]
        conf_ab = np.where((da == 16) & (db == 16))[0]
        if len(conf_ab) != 44:
            raise OrbitalError(f"p^3_55 = {len(conf_ab)} != 44")
        vs = np.where((class_of(da) == 3) & (class_of(db) == 3))[0]
        if len(vs) != 43164:
            raise OrbitalError(f"p^3_33 = {len(vs)} != 43164")
        n5 = ((C[conf_ab] @ C[vs].T) == 16).sum(axis=0)
        cnt = {int(t): int((n5 == t).sum()) for t in np.unique(n5)}
        if cnt != {0: 42240, 2: 924}:
            raise OrbitalError(f"(3,3,3) split by common conflict neighbours: {cnt}")


# --------------------------------------------------------------------------- SDP with cuts


class Sdp3Cuts(Sdp3):
    """T2.3's relaxation plus linear cut rows A x <= t, with the extended
    exact certificate."""

    def solve_with_cuts(self, cuts: Sequence[Cut], forbidden: Iterable[int] = (CONFLICT,),
                        solver: str = "CLARABEL", verbose: bool = False, **kw) -> Sdp3Result:
        import time

        import cvxpy as cp

        forb = tuple(forbidden)
        Q = self.Q
        z = self.forbidden_orbits(forb)
        x = cp.Variable(Q, name="x")
        c_nn = x >= 0
        c_ub = x <= 1
        cons = [c_nn, c_ub, x[self.e] == 1]
        if z:
            cons.append(x[z] == 0)
        c_p1 = sum(x[q] * self.Zp[q] for q in range(Q)) >> 0
        c_p2 = sum(x[q] * self.Zyp[q] for q in range(Q)) >> 0
        c_y = self.Cy @ x >= 0
        cons += [c_p1, c_p2, c_y]
        c_cut = None
        if cuts:
            A = np.stack([c.row for c in cuts]).astype(float)
            t = np.array([c.rhs for c in cuts], dtype=float)
            c_cut = A @ x <= t
            cons.append(c_cut)
        scale = float(self.N)
        prob = cp.Problem(cp.Maximize((self.obj @ x) / scale), cons)
        t0 = time.time()
        prob.solve(solver=solver, verbose=verbose, **kw)
        dt = time.time() - t0
        val = None if prob.value is None else float(prob.value) * scale

        def dv(c):
            try:
                return None if c is None or c.dual_value is None else \
                    np.array(c.dual_value, dtype=float) * scale
            except Exception:
                return None

        res = Sdp3Result(forbidden=forb, status=str(prob.status), value=val,
                         x=None if x.value is None else np.array(x.value),
                         solver=solver, solve_seconds=dt)
        res._duals = {"Y1": dv(c_p1), "Y2": dv(c_p2), "mu": dv(c_y), "rho": dv(c_ub),
                      "nu": dv(c_cut)}  # type: ignore[attr-defined]
        res._cuts = list(cuts)          # type: ignore[attr-defined]
        return res

    def certify_with_cuts(self, res: Sdp3Result, bits: int = 52, rel_drop: float = 1e-9,
                          verbose: bool = False) -> Sdp3Result:
        """Extended exact certificate:
        h_q = obj_q + <Y1,G_q> + <Y2,Gy_q> + (mu^T Cy)_q - (nu^T A)_q - rho_q,
        bound = h_e + sum_{q free} rho_q + nu^T t + sum_{q free != e} max(0, h_q)."""
        duals = getattr(res, "_duals", None)
        cuts: list[Cut] = getattr(res, "_cuts", [])
        if duals is None or duals.get("Y1") is None:
            raise OrbitalError("no dual solution available (solve_with_cuts() first)")
        free = self.free_orbits(res.forbidden)
        ranks = []
        psd_terms = []
        a = [Fraction(0)] * self.Q
        for key, Gs in (("Y1", self.G), ("Y2", self.Gy)):
            Y = duals[key]
            wY = np.linalg.eigvalsh((Y + Y.T) / 2)
            if -wY.min() > wY.max():
                Y = -Y
            Yint, den, terms = self._rational_psd(Y, bits, rel_drop)
            ranks.append(len(terms))
            psd_terms.append((den, terms))
            for q in range(self.Q):
                a[q] += self._contract(Yint, den, Gs[q])

        den = 1 << bits
        mu_raw = duals.get("mu")
        rho_raw = duals.get("rho")
        nu_raw = duals.get("nu")
        mu_raw = np.zeros(self.D) if mu_raw is None else np.asarray(mu_raw, dtype=float).ravel()
        rho_raw = np.zeros(self.Q) if rho_raw is None else np.asarray(rho_raw, dtype=float).ravel()
        nu_raw = np.zeros(len(cuts)) if nu_raw is None else np.asarray(nu_raw, dtype=float).ravel()
        best = None
        for smu in (1.0, -1.0):
            mu = [Fraction(int(round(max(0.0, smu * t) * den)), den) for t in mu_raw]
            muCy = [sum(mu[u] * int(self.Cy[u, q]) for u in range(self.D) if self.Cy[u, q])
                    for q in range(self.Q)]
            for snu in ((1.0, -1.0) if len(cuts) else (1.0,)):
                nu = [Fraction(int(round(max(0.0, snu * t) * den)), den) for t in nu_raw]
                nuA = [Fraction(0)] * self.Q
                nu_t = Fraction(0)
                for ci, cut in enumerate(cuts):
                    if nu[ci] == 0:
                        continue
                    nu_t += nu[ci] * cut.rhs
                    for q in range(self.Q):
                        if cut.row[q]:
                            nuA[q] += nu[ci] * int(cut.row[q])
                h = [Fraction(int(self.obj[q])) + a[q] + muCy[q] - nuA[q]
                     for q in range(self.Q)]
                for srho in (1.0, -1.0):
                    rho = [Fraction(int(round(max(0.0, srho * t) * den)), den) for t in rho_raw]
                    g = [h[q] - rho[q] for q in range(self.Q)]
                    viol = sum((g[q] for q in free if q != self.e and g[q] > 0), Fraction(0))
                    bound = g[self.e] + sum(rho[q] for q in free) + nu_t + viol
                    if best is None or bound < best[0]:
                        best = (bound, viol, g[self.e], g, rho, mu, nu, nu_t)
        bound, viol, he, g, rho, mu, nu, nu_t = best
        res.bound = bound
        res.bound_floor = math.floor(bound)
        res.cert_violation = viol
        res.cert_ranks = (ranks[0], ranks[1])
        res._cert = {  # type: ignore[attr-defined]
            "forbidden": list(res.forbidden), "free": free, "e": self.e,
            "h": g, "rho": rho, "mu": mu, "nu": nu, "nu_t": nu_t, "a": list(a),
            "psd_terms": psd_terms, "bound": bound,
        }
        if verbose:
            print(f"  certificate: h_e = {float(he):.9f}, rounding slack = {float(viol):.3e}, "
                  f"nu^T t = {float(nu_t):.3e}, bound = {float(bound):.9f} -> "
                  f"|S| <= {res.bound_floor}")
        return res

    def save_cuts_certificate(self, res: Sdp3Result, path: str) -> str:
        """Extended certificate JSON: everything of T2.3's format plus the cut
        instances (kind, a, b, K), their integer rows / rhs and nu."""
        import json

        c = self.certificate(res)
        cuts: list[Cut] = getattr(res, "_cuts", [])

        def fr(x):
            x = Fraction(x)
            return [int(x.numerator), int(x.denominator)]

        blocks = []
        for dn, terms in c["psd_terms"]:
            blocks.append({"den_exp": int(dn),
                           "terms": [{"m": int(m), "p": [int(t) for t in P]} for m, P in terms]})
        doc = {
            "task": "B1",
            "description": (
                "Exact dual certificate of the three-point SDP bound with clique cuts. "
                "bound = h[e] + sum_{q free} rho[q] + nu^T t + sum_{q free, q != e} "
                "max(0, h[q]); h[q] = obj[q] + <Y1,G_q> + <Y2,Gy_q> + (mu^T Cy)_q - "
                "(nu^T A)_q - rho[q]. Cut rows A (rhs t) are the lifted clique "
                "inequalities (C1)/(C2) of bounds.sdp3_cliques; each instance "
                "(kind, a, b, K) is recorded so the row can be re-derived from the "
                "vectors and orbitals.json alone."),
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
            "nu": [fr(t) for t in c["nu"]],
            "cuts": [{"kind": cut.kind, "a": int(cut.a), "b": int(cut.b),
                      "K": [int(v) for v in cut.K],
                      "row": [int(t) for t in cut.row], "rhs": int(cut.rhs)}
                     for cut in cuts],
            "h": [fr(t) for t in c["h"]],
            "bound": fr(c["bound"]),
            "bound_float": float(c["bound"]),
            "bound_floor": int(math.floor(c["bound"])),
            "solver": res.solver, "solver_status": res.status, "solver_value": res.value,
        }
        with open(path, "w") as f:
            json.dump(doc, f)
        return path


def verify_cuts_certificate(path: str, orbitals_path: str | None = None,
                            vectors_path: str | os.PathLike = DEFAULT_VECTORS) -> dict:
    """Independent re-check of a certificate written by save_cuts_certificate.
    Rebuilds G_q / Gy_q from orbitals.json, re-derives every cut row from its
    recorded instance (a, b, K) — re-verifying that K is a clique and
    re-classifying every triple — and recomputes h and the bound in exact
    rational arithmetic.  Uses nothing from the solver."""
    import json

    from .orbitals import load_orbitals

    with open(path) as f:
        doc = json.load(f)
    O = load_orbitals(orbitals_path) if orbitals_path else load_orbitals()
    S = Sdp3(O)
    if S.D != int(doc["D"]) or S.N != int(doc["N"]) or S.Q != int(doc["vars"]):
        raise OrbitalError("certificate was made for a different scheme")
    if [[int(u) for u in ms] for ms in doc["orbit_members"]] != \
            [[int(u) for u in ms] for ms in S.members]:
        raise OrbitalError("orbit_members do not match orbitals.json")
    if [int(t) for t in doc["objective"]] != [int(t) for t in S.obj]:
        raise OrbitalError("objective does not match orbitals.json")
    if not np.array_equal(np.array(doc["Cy"], dtype=np.int64), S.Cy):
        raise OrbitalError("Cy does not match orbitals.json")
    forb = tuple(int(f) for f in doc["forbidden_classes"])
    free = S.free_orbits(forb)
    if free != [int(q) for q in doc["free_orbits"]]:
        raise OrbitalError("free orbit list does not match the forbidden classes")
    # cut rows re-derived from scratch
    C = load_vectors(vectors_path)
    cutter = CliqueCutter(S, C)
    cuts = []
    for cd in doc["cuts"]:
        rebuilt = cutter.cuts(int(cd["a"]), int(cd["b"]), [int(v) for v in cd["K"]],
                              kinds=(cd["kind"],))[0]
        if [int(t) for t in rebuilt.row] != [int(t) for t in cd["row"]] or \
                rebuilt.rhs != int(cd["rhs"]):
            raise OrbitalError(f"cut row for instance {cd['kind']}({cd['a']},{cd['b']}) "
                               "does not re-derive — certificate is inconsistent")
        cuts.append(rebuilt)
    # multipliers
    a = [Fraction(0)] * S.Q
    for blk, Gs in zip(doc["blocks"], (S.G, S.Gy)):
        dn = int(blk["den_exp"])
        Yint = np.zeros((S.D, S.D), dtype=object)
        for term in blk["terms"]:
            m = int(term["m"])
            if m < 0:
                raise OrbitalError("negative weight in a PSD block")
            P = np.array([int(t) for t in term["p"]], dtype=object)
            Yint = Yint + m * np.outer(P, P)
        for q in range(S.Q):
            a[q] += Fraction(int((Yint * Gs[q]).sum()), 1 << dn)
    mu = [Fraction(n, d) for n, d in doc["mu"]]
    rho = [Fraction(n, d) for n, d in doc["rho"]]
    nu = [Fraction(n, d) for n, d in doc["nu"]]
    if any(t < 0 for t in mu) or any(t < 0 for t in rho) or any(t < 0 for t in nu):
        raise OrbitalError("negative multiplier")
    if len(nu) != len(cuts):
        raise OrbitalError("nu length != number of cuts")
    h = []
    for q in range(S.Q):
        val = Fraction(int(S.obj[q])) + a[q] - rho[q]
        val += sum(mu[u] * int(S.Cy[u, q]) for u in range(S.D) if S.Cy[u, q])
        val -= sum(nu[ci] * int(cut.row[q]) for ci, cut in enumerate(cuts) if cut.row[q])
        h.append(val)
    if h != [Fraction(n, d) for n, d in doc["h"]]:
        raise OrbitalError("recomputed h differs from the stored h")
    e = int(doc["e"])
    nu_t = sum((nu[ci] * cut.rhs for ci, cut in enumerate(cuts)), Fraction(0))
    bound = h[e] + sum(rho[q] for q in free) + nu_t \
        + sum(h[q] for q in free if q != e and h[q] > 0)
    if bound != Fraction(*doc["bound"]):
        raise OrbitalError("recomputed bound differs from the stored bound")
    return {"bound": bound, "bound_floor": math.floor(bound), "cuts": len(cuts),
            "forbidden_dots": doc["forbidden_dots"], "matches_stored": True}


# --------------------------------------------------------------------------- CLI


def _table_row(tag: str, res: Sdp3Result, ncuts: int) -> str:
    return (f"{tag:34s} cuts {ncuts:4d}  float {res.value:14.6f}  "
            f"exact {float(res.bound):14.6f}  floor {res.bound_floor:6d}  "
            f"status {res.status}  slack {float(res.cert_violation):.3e}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vectors", default=DEFAULT_VECTORS)
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--bits", type=int, default=52)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pairs-per-class", type=int, default=25)
    ap.add_argument("--triangles", type=int, default=4,
                    help="random 3-subsets of K24 per base pair")
    ap.add_argument("--big-pool", type=int, default=0,
                    help="extra instances for the violation scan only")
    ap.add_argument("--save-certificate", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    C = load_vectors(args.vectors)
    verify_k24(C)
    print(f"omega(G) = 24: explicit 24-clique verified (pairwise inner products 16); "
          f"<= 24 by the Gram rank argument (tools/clique_g)")

    S = Sdp3Cuts()
    cutter = CliqueCutter(S, C)
    cutter.selfcheck_classifier(rng)
    print("classifier self-check: (3,3,3) split re-derived as 924 + 42240 exactly")

    tri = [tuple(int(K24[t]) for t in rng.choice(24, 3, replace=False))
           for _ in range(args.triangles)]
    pool_tri = cutter.sample_pool(rng, tri, args.pairs_per_class, kinds=("C1", "C2"))
    pool_omega = cutter.sample_pool(rng, [K24], args.pairs_per_class, kinds=("C1", "C2"))
    print(f"cut pool: {len(pool_tri)} triangle cuts, {len(pool_omega)} omega-clique cuts "
          f"(deduplicated, kinds C1+C2)")

    # ---- MANDATORY validity gate: the 496 and 488 stay exactly feasible ----
    st = cutter.check_sets_exact(pool_tri + pool_omega)
    for name, r in sorted(st.items()):
        print(f"exact feasibility of {name} under all {r['eq'] + r['lt']} cuts: "
              f"OK ({r['eq']} tight, {r['lt']} strict)")

    # ---- bound table -------------------------------------------------------
    runs = [
        ("baseline (T2.3, no cuts)", []),
        ("+ triangle cuts", pool_tri),
        ("+ omega-clique cuts", pool_omega),
        ("+ triangle + omega cuts", pool_tri + pool_omega),
    ]
    results = {}
    print("\nnote: entrywise nonnegativity of both blocks (theta-plus route (b)) is "
          "already in the baseline: M1 >= 0 entrywise <=> x >= 0, M2 >= 0 entrywise "
          "<=> y >= 0, both imposed by T2.3.\n")
    ok = True
    for tag, cuts in runs:
        res = S.solve_with_cuts(cuts, (CONFLICT,), solver=args.solver, verbose=args.verbose)
        S.certify_with_cuts(res, bits=args.bits, verbose=args.verbose)
        results[tag] = res
        print(_table_row(tag, res, len(cuts)))
        ok = ok and res.bound_floor is not None and 496 <= res.bound_floor <= 850

    # ---- violation scan at the optima -------------------------------------
    xs = {tag: results[tag].x for tag in results}
    scan = pool_tri + pool_omega
    if args.big_pool:
        extra_tri = [tuple(int(K24[t]) for t in rng.choice(24, 3, replace=False))
                     for _ in range(8)]
        scan = scan + cutter.sample_pool(rng, extra_tri + [K24], args.big_pool,
                                         kinds=("C1", "C2"))
    A = np.stack([c.row for c in scan]).astype(float)
    t = np.array([c.rhs for c in scan], dtype=float)
    for tag in ("baseline (T2.3, no cuts)", "+ triangle + omega cuts"):
        v = A @ xs[tag] - t
        print(f"violation scan at '{tag}' optimum over {len(scan)} cuts: "
              f"max {v.max():.3e} (<= 0 means no cut separates), "
              f"#within 1e-7 of tight: {int((v > -1e-7).sum())}")

    # ---- sanity runs -------------------------------------------------------
    res_none = S.solve_with_cuts([], (), solver=args.solver)
    S.certify_with_cuts(res_none, bits=args.bits)
    res_168 = S.solve_with_cuts(pool_omega, (CONFLICT, 4), solver=args.solver)
    S.certify_with_cuts(res_168, bits=args.bits)
    print(f"sanity: forbid nothing (no cuts: clique cuts need independence) -> "
          f"{res_none.bound_floor} >= N = {S.N}: {res_none.bound_floor >= S.N}; "
          f"forbid {{16,8}} with omega cuts -> {res_168.bound_floor} (orthoplex 48)")
    ok = ok and res_none.bound_floor >= S.N and res_168.bound_floor == 48

    if args.save_certificate:
        best_tag = min(results, key=lambda tg: results[tg].bound)
        S.save_cuts_certificate(results[best_tag], args.save_certificate)
        print(f"certificate of '{best_tag}' written to {args.save_certificate}")
        v = verify_cuts_certificate(args.save_certificate, vectors_path=args.vectors)
        print(f"independent re-check: bound_floor = {v['bound_floor']}, "
              f"matches_stored = {v['matches_stored']}")
        ok = ok and v["matches_stored"]

    b0 = results["baseline (T2.3, no cuts)"].bound_floor
    b3 = results["+ triangle + omega cuts"].bound_floor
    print(f"RESULT ok={1 if ok else 0} omega=24 baseline={b0} with_cuts={b3} "
          f"none={res_none.bound_floor} forbid168={res_168.bound_floor}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

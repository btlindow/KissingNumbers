"""Association scheme of the Leech minimal vectors: intersection matrices and
exact eigenmatrices P, Q (docs/design.md T2.2, README section 3 W1 item 2).

Input: data/scheme/intersection_numbers.json written by tools/scheme_numbers
(GPU brute force, x = C[0], all y).  Classes are indexed c = 0..6 in the
canonical order of PLAN section 2.2:

    c      : 0    1    2   3  4   5   6
    <x,y>  : -32  -16  -8  0  8   16  32        (sqrt(8) scaling)
    cos    : -1  -1/2 -1/4 0 1/4 1/2  1

Class 6 (cos 1) is the IDENTITY relation R_6 = {(x,x)}; class 0 is the antipode
y = -x; class 5 (cos 1/2) is the conflict graph G.  p[k][i][j] = p^k_{ij} =
#{z : (x,z) in R_i, (z,y) in R_j} for (x,y) in R_k.

Conventions fixed here (and used by lp_scheme.py / theta_prime.py):

* Intersection matrix L_i (7x7):  (L_i)_{jk} = p^k_{ij}   (row j, column k).
  With A_i the adjacency matrices, A_i A_j = sum_k p^k_{ij} A_k, and the
  Bose-Mesner algebra is commutative, so the L_i commute and
  L_i L_j = sum_k p^k_{ij} L_k (checked).
* First eigenmatrix P (7x7): rows = eigenspaces (primitive idempotents E_r),
  columns = classes;  P_{r i} = eigenvalue of A_i on E_r.  Row 0 is the
  trivial eigenspace (all-ones), P_{0 i} = v_i; column 6 (identity class) is
  all ones.  The remaining rows are ordered by DECREASING eigenvalue of the
  conflict graph A_5 (ties by increasing multiplicity).
  Rows of P are exactly the right eigenvectors of every L_i normalised so
  that the identity-class coordinate is 1:  from A_i A_j = sum_k p^k_{ij} A_k
  applied to E_r one gets P_{ri} P_{rj} = sum_k p^k_{ij} P_{rk}, i.e.
  L_i u = P_{ri} u for u = (P_{r0}, ..., P_{r6}).
* Multiplicities m_r = dim E_r = N / sum_i P_{ri}^2 / v_i.
* Second (dual) eigenmatrix Q = N P^{-1};  equivalently Q_{i r} = m_r P_{ri}/v_i.
  E_r = (1/N) sum_i Q_{ir} A_i.  Column 0 of Q is all ones (Q_{i0} = 1), row 6
  of Q is (m_0, ..., m_6).
  For a subset S with inner distribution a_i = |{(x,y) in S^2 : (x,y) in R_i}|/|S|,
  (a Q)_r = (|S|/N) * (eigenvalue-weighted) = (N/|S|) * 1_S^T E_r 1_S >= 0
  because E_r is positive semidefinite -- the Delsarte inequalities.

All arithmetic is exact (sympy Rational / Python int).

CLI:  .venv/bin/python python/bounds/scheme.py  [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Sequence

import sympy as sp

NCLASS = 7
IDENTITY = 6      # class index of cos = 1 (dot 32)
CONFLICT = 5      # class index of cos = 1/2 (dot 16): the conflict graph
ANTIPODE = 0      # class index of cos = -1
CLASS_DOTS = (-32, -16, -8, 0, 8, 16, 32)
CLASS_COS = tuple(Fraction(d, 32) for d in CLASS_DOTS)

DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "scheme", "intersection_numbers.json"
)


class SchemeError(ValueError):
    pass


@dataclass
class Scheme:
    N: int
    classes: tuple[int, ...]           # inner products (sqrt(8) scaling)
    valencies: tuple[int, ...]         # v_i
    p: tuple[tuple[tuple[int, ...], ...], ...]   # p[k][i][j]
    identity: int = IDENTITY
    conflict: int = CONFLICT
    source: dict | None = None

    @property
    def d(self) -> int:
        return len(self.classes) - 1


def load_scheme(path: str | os.PathLike = DEFAULT_JSON) -> Scheme:
    with open(path) as f:
        data = json.load(f)
    if not data.get("scheme", False):
        raise SchemeError(
            f"{path}: the relations were NOT found to be an association scheme "
            f"(distinct matrices per class {data.get('distinct_matrices_per_class')}); "
            "the LP must be run on the coherent configuration instead — see docs/reports/T2.2.md"
        )
    classes = tuple(int(c) for c in data["classes"])
    if classes != CLASS_DOTS:
        raise SchemeError(f"unexpected class order {classes}")
    p = tuple(tuple(tuple(int(x) for x in row) for row in mat) for mat in data["p"])
    s = Scheme(
        N=int(data["N"]),
        classes=classes,
        valencies=tuple(int(v) for v in data["valencies"]),
        p=p,
        identity=int(data.get("identity_class", IDENTITY)),
        conflict=int(data.get("conflict_class", CONFLICT)),
        source=data,
    )
    check_identities(s)
    return s


# --------------------------------------------------------------------------- identities


def check_identities(s: Scheme) -> None:
    """Re-check the axioms of a symmetric association scheme on the numbers
    (independently of the C++ tool).  Raises SchemeError with the first failure."""
    n, v, p, e = s.d + 1, s.valencies, s.p, s.identity
    if len(v) != n or len(p) != n or any(len(m) != n or any(len(r) != n for r in m) for m in p):
        raise SchemeError("shape")
    if sum(v) != s.N:
        raise SchemeError(f"valencies sum to {sum(v)} != N = {s.N}")
    if v[e] != 1:
        raise SchemeError("identity class must have valency 1")
    for k, i, j in product(range(n), repeat=3):
        if p[k][i][j] < 0:
            raise SchemeError(f"negative p^{k}_{{{i}{j}}}")
        if p[k][i][j] != p[k][j][i]:
            raise SchemeError(f"not symmetric: p^{k}_{{{i}{j}}} = {p[k][i][j]} != p^{k}_{{{j}{i}}} = {p[k][j][i]}")
        if v[k] * p[k][i][j] != v[i] * p[i][k][j]:
            raise SchemeError(f"v_k p^k_ij != v_i p^i_kj at k={k} i={i} j={j}")
    for k, i in product(range(n), repeat=2):
        if sum(p[k][i]) != v[i]:
            raise SchemeError(f"row sum p^{k}_{{{i}.}} = {sum(p[k][i])} != v_{i} = {v[i]}")
        if p[e][i][k] != (v[i] if i == k else 0):
            raise SchemeError(f"p^e_{{{i}{k}}} != delta v_i")
        if p[k][i][e] != (1 if i == k else 0):
            raise SchemeError(f"p^{k}_{{{i}e}} != delta_ik")


# --------------------------------------------------------------------------- matrices


def intersection_matrices(s: Scheme) -> list[sp.Matrix]:
    """L_i with (L_i)_{jk} = p^k_{ij}."""
    n = s.d + 1
    return [sp.Matrix(n, n, lambda j, k, i=i: sp.Integer(s.p[k][i][j])) for i in range(n)]


def check_bose_mesner(s: Scheme, L: Sequence[sp.Matrix] | None = None) -> None:
    """L_i L_j == sum_k p^k_{ij} L_k for all i, j (implies commutativity)."""
    L = intersection_matrices(s) if L is None else L
    n = s.d + 1
    for i in range(n):
        for j in range(n):
            rhs = sp.zeros(n, n)
            for k in range(n):
                if s.p[k][i][j]:
                    rhs += s.p[k][i][j] * L[k]
            if L[i] * L[j] != rhs:
                raise SchemeError(f"L_{i} L_{j} != sum_k p^k_{{ij}} L_k")
            if L[i] * L[j] != L[j] * L[i]:
                raise SchemeError(f"L_{i} and L_{j} do not commute")


@dataclass
class Eigen:
    P: sp.Matrix            # 7x7, rows = eigenspaces, columns = classes
    Q: sp.Matrix            # 7x7, Q = N P^{-1}
    m: tuple[int, ...]      # multiplicities (row r of P)
    N: int
    valencies: tuple[int, ...]

    def conflict_spectrum(self, conflict: int = CONFLICT) -> list[tuple[sp.Rational, int]]:
        """[(eigenvalue of A_conflict, multiplicity)] in row order of P."""
        return [(self.P[r, conflict], self.m[r]) for r in range(self.P.rows)]


def _common_eigenvectors(L: Sequence[sp.Matrix], identity: int) -> list[tuple[list[sp.Rational], sp.Matrix]]:
    """Find the n one-dimensional common eigenspaces of the commuting L_i.
    Method: take generic integer combinations M = sum c_i L_i until M has n
    distinct eigenvalues; its eigenvectors are then the common eigenvectors.
    Each is normalised so that coordinate `identity` equals 1 (it cannot be 0:
    the identity-class coordinate of a row of P is P_{r,identity} = 1).
    Returns [(eigenvalues of L_0..L_{n-1} on the vector, vector)]."""
    n = L[0].rows
    tried = [
        [1, 0, 0, 0, 0, 0, 0][:n],
        [0, 1, 0, 0, 0, 0, 0][:n],
        [1, 3, 9, 27, 81, 243, 729][:n],
        [5, -2, 7, 1, -3, 11, 13][:n],
        [17, 2, -5, 23, 4, -1, 6][:n],
    ]
    for c in tried:
        M = sp.zeros(n, n)
        for ci, Li in zip(c, L):
            if ci:
                M += ci * Li
        ev = M.eigenvects()
        if len(ev) != n or any(mult != 1 for _, mult, _ in ev):
            continue
        out = []
        for lam, _, vecs in ev:
            if not lam.is_rational:
                raise SchemeError(f"irrational eigenvalue {lam}: the scheme is not rational")
            u = vecs[0]
            if u[identity] == 0:
                raise SchemeError("common eigenvector with zero identity coordinate")
            u = u / u[identity]
            u = sp.Matrix([sp.nsimplify(x) for x in u])
            lams = []
            for Li in L:
                w = Li * u
                # eigenvalue = w[identity] / u[identity] = w[identity]; verify on all coordinates
                lam_i = w[identity]
                if w != lam_i * u:
                    raise SchemeError("vector is not a common eigenvector")
                lams.append(sp.Rational(lam_i))
            out.append((lams, u))
        return out
    raise SchemeError("no generic combination of the L_i had simple spectrum")


def eigenmatrices(s: Scheme) -> Eigen:
    """Exact first and second eigenmatrices (see module docstring for the
    conventions).  Raises SchemeError if any structural check fails."""
    n = s.d + 1
    L = intersection_matrices(s)
    check_bose_mesner(s, L)
    rows = _common_eigenvectors(L, s.identity)
    # Each row of P is the eigenvalue tuple; multiplicity from the standard formula.
    prow = []
    for lams, u in rows:
        # lams[i] is the eigenvalue of L_i, i.e. P_{r i}; must equal u[i] (u is a row of P).
        for i in range(n):
            if lams[i] != u[i]:
                raise SchemeError("eigenvalue/eigenvector inconsistency")
        denom = sum(sp.Rational(lams[i]) ** 2 / s.valencies[i] for i in range(n))
        m = sp.Rational(s.N) / denom
        if not m.is_integer or m <= 0:
            raise SchemeError(f"multiplicity {m} is not a positive integer")
        prow.append((lams, int(m)))
    # Order: trivial row first (eigenvalues == valencies), then decreasing conflict eigenvalue.
    triv = [r for r in prow if all(r[0][i] == s.valencies[i] for i in range(n))]
    if len(triv) != 1:
        raise SchemeError("trivial eigenspace not found exactly once")
    rest = sorted((r for r in prow if r is not triv[0]), key=lambda r: (-r[0][s.conflict], r[1]))
    ordered = triv + rest
    P = sp.Matrix([[sp.Rational(x) for x in lams] for lams, _ in ordered])
    m = tuple(mm for _, mm in ordered)
    if sum(m) != s.N:
        raise SchemeError(f"multiplicities sum to {sum(m)} != N")
    if m[0] != 1:
        raise SchemeError("trivial eigenspace must have multiplicity 1")
    Q = s.N * P.inv()
    # Cross-check Q_{ir} = m_r P_{ri} / v_i and the orthogonality relations.
    for i in range(n):
        for r in range(n):
            if Q[i, r] != sp.Rational(m[r] * P[r, i], s.valencies[i]):
                raise SchemeError("Q != m_r P_{ri}/v_i")
    if P * Q != s.N * sp.eye(n) or Q * P != s.N * sp.eye(n):
        raise SchemeError("P Q != N I")
    for i in range(n):
        if i != s.identity and sum(m[r] * P[r, i] for r in range(n)) != 0:
            raise SchemeError(f"trace of A_{i} is not zero")
    if any(P[r, s.identity] != 1 for r in range(n)) or any(Q[i, 0] != 1 for i in range(n)):
        raise SchemeError("P column identity / Q column 0 not all ones")
    return Eigen(P=P, Q=Q, m=m, N=s.N, valencies=s.valencies)


def hoffman_bound(eig: Eigen, conflict: int = CONFLICT) -> Fraction:
    """Hoffman ratio bound N (-lambda_min) / (k - lambda_min) for the regular
    graph A_conflict of valency k (exact)."""
    k = eig.P[0, conflict]
    lmin = min(eig.P[r, conflict] for r in range(1, eig.P.rows))
    val = sp.Rational(eig.N) * (-lmin) / (k - lmin)
    return Fraction(int(val.p), int(val.q))


def to_fractions(M: sp.Matrix) -> list[list[Fraction]]:
    return [[Fraction(int(sp.Rational(M[i, j]).p), int(sp.Rational(M[i, j]).q)) for j in range(M.cols)] for i in range(M.rows)]


# --------------------------------------------------------------------------- CLI


def format_matrix(M: sp.Matrix, row_labels: Sequence[str], col_labels: Sequence[str], width: int = 12) -> str:
    lines = [" " * 10 + "".join(f"{c:>{width}s}" for c in col_labels)]
    for r in range(M.rows):
        lines.append(f"{row_labels[r]:>10s}" + "".join(f"{str(M[r, c]):>{width}s}" for c in range(M.cols)))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args(argv)
    s = load_scheme(args.json)
    print(f"scheme: N = {s.N}, classes (dot) = {s.classes}, valencies = {s.valencies}")
    print(f"scheme property (from json): {s.source.get('scheme')}, y checked: {s.source.get('y_checked')}, x = {s.source.get('x')}")
    print("identities: OK (symmetric, v_k p^k_ij = v_i p^i_kj, row sums, identity class)")
    L = intersection_matrices(s)
    check_bose_mesner(s, L)
    print("Bose-Mesner: L_i L_j = sum_k p^k_ij L_k for all i,j (commutative): OK")
    for i in range(s.d + 1):
        print(f"L_{i} (dot {s.classes[i]:>3d}), (L_i)_jk = p^k_ij:")
        for j in range(s.d + 1):
            print("   ", " ".join(f"{int(L[i][j, k]):>6d}" for k in range(s.d + 1)))
    eig = eigenmatrices(s)
    cols = [f"A({d})" for d in s.classes]
    rows = [f"E{r} m={m}" for r, m in enumerate(eig.m)]
    print("\nfirst eigenmatrix P (rows: eigenspaces E_r with multiplicity; columns: classes by inner product):")
    print(format_matrix(eig.P, rows, cols))
    print("\nsecond eigenmatrix Q = N P^{-1} (rows: classes; columns: eigenspaces):")
    print(format_matrix(eig.Q, [f"a({d})" for d in s.classes], [f"E{r}" for r in range(len(eig.m))]))
    print(f"\nmultiplicities m = {eig.m}, sum = {sum(eig.m)} = N: {sum(eig.m) == s.N}")
    print(f"P Q = N I: {eig.P * eig.Q == s.N * sp.eye(s.d + 1)}")
    spec = eig.conflict_spectrum(s.conflict)
    print("conflict graph (class 16, valency 4600) spectrum (eigenvalue^multiplicity):",
          " ".join(f"{lam}^{m}" for lam, m in spec))
    print(f"Hoffman bound N(-lmin)/(k-lmin) = {hoffman_bound(eig, s.conflict)} = {float(hoffman_bound(eig, s.conflict)):.6f}")
    print(f"RESULT ok=1 scheme=1 N={s.N} PQ_eq_NI=1 multiplicities={','.join(map(str, eig.m))} "
          f"conflict_eigenvalues={','.join(str(l) for l, _ in spec)} hoffman={hoffman_bound(eig, s.conflict)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

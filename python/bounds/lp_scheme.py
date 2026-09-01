"""Association-scheme (Delsarte) LP bound for subsets of the Leech minimal
vectors that avoid given inner-product classes (docs/design.md T2.2, README section 3
W1 item 2).  This is the bound that decides gate G1.

Setting.  C = the 196560 minimal vectors, 7 inner-product classes
c = 0..6 <-> <x,y> in {-32,-16,-8,0,8,16,32} (sqrt(8) scaling), which form a
symmetric association scheme (verified by tools/scheme_numbers, exact
eigenmatrices P, Q from scheme.py).  For a subset S the inner distribution is

    a_i = #{(x,y) in S^2 : <x,y> in class i} / |S|,   so  sum_i a_i = |S|,

a_identity = 1, a_i >= 0, and Delsarte's inequalities (a Q)_r >= 0 for every
eigenspace r (because (a Q)_r = (N/|S|) 1_S^T E_r 1_S with E_r positive
semidefinite).  A set avoiding the conflict class (cos 1/2, dot 16) has
a_5 = 0.  Hence

    |S| <= LP := max sum_i a_i  s.t.  a_6 = 1, a_5 = 0, a >= 0, a Q >= 0.

Certificate (weak duality, hand-checkable).  Let e be the identity class,
Fx the set of classes with prescribed values (a_e = 1, forbidden classes 0,
and a_0 = 1 in the antipodal variant) and Fr the remaining "free" classes.
Put q_r := sum_{i in Fx} Q_{ir} a_i  (for Fx = {e}: q_r = Q_{er} = m_r, the
multiplicities).  If beta_0..beta_6 >= 0 satisfy

    sum_r beta_r Q_{ir} <= -1        for every free class i,

then for every feasible a:

    sum_{i in Fr} a_i <= - sum_{i in Fr} a_i sum_r beta_r Q_{ir}
                      = - sum_r beta_r [ (a Q)_r - q_r ]
                      <= sum_r beta_r q_r,

so |S| = sum_i a_i <= sum_{i in Fx} a_i + sum_r beta_r q_r.  For the main LP:
|S| <= 1 + sum_r beta_r m_r.  In Delsarte's language F(i) := 1 + sum_r
beta_r Q_{ir} is a positive-definite function on the scheme (a nonnegative
combination of the columns of Q, i.e. of the E_r), F <= 0 on the allowed
non-identity classes, and the bound is F(e) = 1 + sum_r beta_r m_r.

Two independent solvers.
  * float: scipy HiGHS (finds the value and, via its marginals, a candidate
    dual);
  * exact: enumeration of all basic solutions with Fractions (at most 6 free
    variables), for the primal AND for the dual; the two exact optima must
    coincide (strong duality), and every certificate is re-verified from
    scratch by `verify_certificate` before it is reported.

CLI:  .venv/bin/python python/bounds/lp_scheme.py [--json PATH]
      prints every variant with its exact certificate and ends with
      RESULT bound_exact=... bound_floor=... .
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bounds.scheme import (  # type: ignore[no-redef]
        ANTIPODE,
        CLASS_DOTS,
        CONFLICT,
        DEFAULT_JSON,
        IDENTITY,
        Eigen,
        eigenmatrices,
        load_scheme,
        to_fractions,
    )
else:
    from .scheme import (
        ANTIPODE,
        CLASS_DOTS,
        CONFLICT,
        DEFAULT_JSON,
        IDENTITY,
        Eigen,
        eigenmatrices,
        load_scheme,
        to_fractions,
    )

Frac = Fraction
Row = list[Fraction]


class LPError(ValueError):
    pass


# --------------------------------------------------------------------------- exact linear algebra


def solve_square(A: Sequence[Row], b: Row) -> Row | None:
    """Solve A x = b exactly (Gauss-Jordan over Fractions); None if singular."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        inv = 1 / M[col][col]
        M[col] = [x * inv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


@dataclass
class ExactLP:
    """max c.x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq   (all Fractions).
    Solved by enumerating basic solutions: every subset of n constraints that
    contains all equality rows and has a nonsingular matrix.  Correct whenever
    the feasible region is nonempty and pointed and the objective is bounded
    (then an optimal vertex exists); raises LPError otherwise."""

    c: Row
    A_ub: list[Row] = field(default_factory=list)
    b_ub: Row = field(default_factory=list)
    A_eq: list[Row] = field(default_factory=list)
    b_eq: Row = field(default_factory=list)

    def feasible(self, x: Row, tol: Fraction = Fraction(0)) -> bool:
        for row, rhs in zip(self.A_ub, self.b_ub):
            if sum(r * xi for r, xi in zip(row, x)) > rhs + tol:
                return False
        for row, rhs in zip(self.A_eq, self.b_eq):
            if sum(r * xi for r, xi in zip(row, x)) != rhs:
                return False
        return True

    def objective(self, x: Row) -> Fraction:
        return sum(ci * xi for ci, xi in zip(self.c, x))

    def solve(self) -> tuple[Fraction, Row, list[tuple[Fraction, Row]]]:
        """Returns (optimum, an optimal vertex, all feasible vertices with their objective)."""
        n = len(self.c)
        n_eq = len(self.A_eq)
        if n_eq > n:
            raise LPError("more equalities than variables")
        vertices: list[tuple[Fraction, Row]] = []
        seen: set[tuple[Fraction, ...]] = set()
        for rows in combinations(range(len(self.A_ub)), n - n_eq):
            A = [self.A_eq[i] for i in range(n_eq)] + [self.A_ub[i] for i in rows]
            b = [self.b_eq[i] for i in range(n_eq)] + [self.b_ub[i] for i in rows]
            x = solve_square(A, b)
            if x is None or not self.feasible(x):
                continue
            key = tuple(x)
            if key in seen:
                continue
            seen.add(key)
            vertices.append((self.objective(x), x))
        if not vertices:
            raise LPError("no feasible vertex (infeasible, or not pointed)")
        best = max(vertices, key=lambda t: t[0])
        return best[0], best[1], vertices


# --------------------------------------------------------------------------- the scheme LP


@dataclass
class SchemeLP:
    """The LP  max sum_i a_i  s.t.  a_i = fixed[i] (i in fixed),  a_i >= 0
    (i free, if nonneg),  (a Q)_r >= 0 for all r."""

    Q: list[Row]                    # Q[i][r]
    m: tuple[int, ...]              # multiplicities (row `identity` of Q)
    N: int
    fixed: dict[int, Fraction]      # class -> prescribed value (identity -> 1, forbidden -> 0, ...)
    nonneg: bool = True
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.Q)

    @property
    def free(self) -> list[int]:
        return [i for i in range(self.n) if i not in self.fixed]

    @property
    def q(self) -> Row:
        """q_r = sum_{i fixed} Q_{ir} a_i."""
        return [sum(self.Q[i][r] * v for i, v in self.fixed.items()) for r in range(self.n)]

    @property
    def const(self) -> Fraction:
        return sum(self.fixed.values(), Fraction(0))

    def full_vector(self, a_free: Row) -> Row:
        a = [Fraction(0)] * self.n
        for i, v in self.fixed.items():
            a[i] = Fraction(v)
        for i, v in zip(self.free, a_free):
            a[i] = Fraction(v)
        return a

    def aQ(self, a: Row) -> Row:
        return [sum(a[i] * self.Q[i][r] for i in range(self.n)) for r in range(self.n)]

    # -- primal ---------------------------------------------------------------
    def primal_exact(self) -> ExactLP:
        fr, q = self.free, self.q
        A_ub: list[Row] = []
        b_ub: Row = []
        for r in range(self.n):                       # -(sum_{i free} Q_ir a_i) <= q_r
            A_ub.append([-self.Q[i][r] for i in fr])
            b_ub.append(q[r])
        if self.nonneg:
            for k in range(len(fr)):                  # -a_i <= 0
                A_ub.append([Fraction(-1 if j == k else 0) for j in range(len(fr))])
                b_ub.append(Fraction(0))
        return ExactLP(c=[Fraction(1)] * len(fr), A_ub=A_ub, b_ub=b_ub)

    def check_primal(self, a: Row) -> None:
        """Every hypothesis of the primal, exactly.  Raises LPError."""
        if len(a) != self.n:
            raise LPError("a has the wrong length")
        for i, v in self.fixed.items():
            if a[i] != v:
                raise LPError(f"a_{i} = {a[i]} != prescribed {v}")
        if self.nonneg and any(x < 0 for x in a):
            raise LPError("a has a negative entry")
        for r, s in enumerate(self.aQ(a)):
            if s < 0:
                raise LPError(f"(aQ)_{r} = {s} < 0")

    # -- dual -----------------------------------------------------------------
    def dual_exact(self) -> ExactLP:
        """min sum_r beta_r q_r  s.t.  sum_r beta_r Q_{ir} <= -1 (i free; '=' if
        the primal has no sign constraints), beta >= 0.  Stored as a max of the
        negated objective."""
        fr, q = self.free, self.q
        n = self.n
        A_ub: list[Row] = []
        b_ub: Row = []
        A_eq: list[Row] = []
        b_eq: Row = []
        for i in fr:
            row = [self.Q[i][r] for r in range(n)]
            if self.nonneg:
                A_ub.append(row)
                b_ub.append(Fraction(-1))
            else:
                A_eq.append(row)
                b_eq.append(Fraction(-1))
        for r in range(n):                            # -beta_r <= 0
            A_ub.append([Fraction(-1 if s == r else 0) for s in range(n)])
            b_ub.append(Fraction(0))
        return ExactLP(c=[-qr for qr in q], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    def verify_certificate(self, beta: Row) -> Fraction:
        """Independent re-check of the dual certificate; returns the bound
        const + sum_r beta_r q_r it proves.  Raises LPError if beta is not a
        valid certificate.  This is the function a sceptic should read."""
        n = self.n
        if len(beta) != n:
            raise LPError("beta has the wrong length")
        if any(b < 0 for b in beta):
            raise LPError("beta has a negative entry")
        for i in self.free:
            s = sum(beta[r] * self.Q[i][r] for r in range(n))
            if self.nonneg and s > -1:
                raise LPError(f"sum_r beta_r Q_{{{i}r}} = {s} > -1 on free class {i}")
            if not self.nonneg and s != -1:
                raise LPError(f"sum_r beta_r Q_{{{i}r}} = {s} != -1 on free class {i}")
        return self.const + sum(b * qr for b, qr in zip(beta, self.q))

    # -- float ----------------------------------------------------------------
    def solve_highs(self) -> tuple[float, np.ndarray, np.ndarray]:
        """HiGHS: returns (value, a_free, dual multipliers beta >= 0 of the (aQ)_r >= 0 rows)."""
        fr, q = self.free, self.q
        A_ub = np.array([[-float(self.Q[i][r]) for i in fr] for r in range(self.n)])
        b_ub = np.array([float(x) for x in q])
        bounds = [(0.0, None) if self.nonneg else (None, None)] * len(fr)
        res = linprog(c=-np.ones(len(fr)), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            raise LPError(f"HiGHS status {res.status}: {res.message}")
        beta = -np.asarray(res.ineqlin.marginals)   # marginals of a min problem are <= 0
        return float(self.const) - float(res.fun), np.asarray(res.x), beta


@dataclass
class Result:
    name: str
    lp: SchemeLP
    highs_value: float
    exact_value: Fraction
    a_star: Row                 # full inner distribution (all 7 classes)
    beta: Row                   # exact dual certificate (from the exact dual)
    beta_highs: Row | None      # rationalised HiGHS marginals if they verify, else None
    dual_value: Fraction
    n_primal_vertices: int
    n_optimal_vertices: int

    @property
    def floor(self) -> int:
        return int(self.exact_value)   # exact_value >= 0 here


def rationalise(xs: Iterable[float], max_den: int = 10**6, eps: float = 1e-9) -> Row:
    out = []
    for x in xs:
        if abs(x) < eps:
            out.append(Fraction(0))
        else:
            out.append(Fraction(x).limit_denominator(max_den))
    return out


def solve_scheme_lp(lp: SchemeLP) -> Result:
    hv, a_free_f, beta_f = lp.solve_highs()
    # exact primal
    pval, a_free, verts = lp.primal_exact().solve()
    a_star = lp.full_vector(a_free)
    lp.check_primal(a_star)
    exact_value = lp.const + pval
    n_opt = sum(1 for v, _ in verts if v == pval)
    # exact dual
    dneg, beta, _ = lp.dual_exact().solve()
    dual_value = lp.const - dneg
    proved = lp.verify_certificate(beta)
    if proved != dual_value or proved != exact_value:
        raise LPError(f"strong duality failed: primal {exact_value}, dual {dual_value}, certificate {proved}")
    if abs(hv - float(exact_value)) > 1e-6 * max(1.0, abs(hv)):
        raise LPError(f"HiGHS {hv} disagrees with exact {exact_value}")
    # rationalised HiGHS duals as a second, independent certificate (may fail
    # to verify if HiGHS returned a degenerate/rounded dual; that is not an error)
    beta_highs: Row | None = None
    try:
        cand = rationalise(beta_f)
        if lp.verify_certificate(cand) == exact_value:
            beta_highs = cand
    except LPError:
        beta_highs = None
    return Result(lp.name, lp, hv, exact_value, a_star, beta, beta_highs, dual_value, len(verts), n_opt)


# --------------------------------------------------------------------------- variants


def make_lp(eig: Eigen, forbidden: Iterable[int], fixed_extra: Mapping[int, int] | None = None,
            nonneg: bool = True, name: str = "", identity: int = IDENTITY) -> SchemeLP:
    Q = to_fractions(eig.Q)
    fixed: dict[int, Fraction] = {identity: Fraction(1)}
    for i in forbidden:
        fixed[i] = Fraction(0)
    for i, v in (fixed_extra or {}).items():
        if i in fixed and fixed[i] != v:
            raise LPError(f"class {i} both forbidden and prescribed")
        fixed[i] = Fraction(v)
    return SchemeLP(Q=Q, m=eig.m, N=eig.N, fixed=fixed, nonneg=nonneg, name=name)


VARIANTS: dict[str, dict] = {
    # name: (forbidden classes, extra fixed values, description)
    "main":        dict(forbidden=(CONFLICT,), fixed_extra=None,
                        desc="S independent in the conflict graph: a_16 = 0 (the T2.2 bound)"),
    "antipodal":   dict(forbidden=(CONFLICT,), fixed_extra={ANTIPODE: 1},
                        desc="additionally S = -S: a_-32 = 1"),
    "forbid_16_8": dict(forbidden=(CONFLICT, 4), fixed_extra=None,
                        desc="cos <= 0 only (README section 8 item 2): a_16 = a_8 = 0; T2.1 gives 48"),
    "none":        dict(forbidden=(), fixed_extra=None,
                        desc="nothing forbidden: must return N = 196560"),
    "only_pm32":   dict(forbidden=(1, 2, 3, 4, 5), fixed_extra=None,
                        desc="only +-32 allowed: must return 2"),
}


def run_variants(eig: Eigen, names: Sequence[str] | None = None) -> dict[str, Result]:
    out = {}
    for name in names or VARIANTS:
        v = VARIANTS[name]
        out[name] = solve_scheme_lp(make_lp(eig, v["forbidden"], v["fixed_extra"], name=name))
    return out


# --------------------------------------------------------------------------- reporting


def fmt_frac(x: Fraction) -> str:
    return str(x) if x.denominator != 1 else str(x.numerator)


def certificate_text(res: Result) -> str:
    lp = res.lp
    n = lp.n
    lines = []
    lines.append(f"  fixed: " + ", ".join(f"a({CLASS_DOTS[i]}) = {fmt_frac(v)}" for i, v in sorted(lp.fixed.items())))
    lines.append(f"  free classes: " + ", ".join(f"a({CLASS_DOTS[i]})" for i in lp.free)
                 + ("" if lp.nonneg else "   (NO sign constraints on the free a_i)"))
    lines.append(f"  HiGHS (float): {res.highs_value:.9f}")
    lines.append(f"  exact optimum: {fmt_frac(res.exact_value)} = {float(res.exact_value):.9f}   floor {res.floor}"
                 f"   [{res.n_primal_vertices} feasible vertices, {res.n_optimal_vertices} optimal]")
    lines.append("  a* (inner distribution attaining it): "
                 + ", ".join(f"a({CLASS_DOTS[i]}) = {fmt_frac(res.a_star[i])}" for i in range(n)))
    aq = lp.aQ(res.a_star)
    lines.append("  (a* Q)_r = " + ", ".join(fmt_frac(x) for x in aq) + "   (all >= 0; zeros = active Delsarte inequalities)")
    lines.append("  dual certificate beta_r (r = eigenspace E_r, multiplicity m_r):")
    lines.append("      r    m_r        beta_r          beta_r * q_r")
    q = lp.q
    for r in range(n):
        lines.append(f"      {r}  {lp.m[r]:>6d}   {fmt_frac(res.beta[r]):>14s}   {fmt_frac(res.beta[r] * q[r]):>14s}")
    lines.append(f"  q_r = sum_(fixed i) Q_ir a_i = " + ", ".join(fmt_frac(x) for x in q))
    lines.append(f"  bound = sum_(fixed) a_i + sum_r beta_r q_r = {fmt_frac(lp.const)} + "
                 f"{fmt_frac(sum(b * qr for b, qr in zip(res.beta, q)))} = {fmt_frac(res.dual_value)}")
    lines.append("  check on every free class i:  F(i) := sum_r beta_r Q_ir  " + ("<= -1" if lp.nonneg else "= -1"))
    for i in lp.free:
        s = sum(res.beta[r] * lp.Q[i][r] for r in range(n))
        lines.append(f"      class a({CLASS_DOTS[i]:>3d}):  F = {fmt_frac(s):>10s}   {'ok' if (s <= -1 if lp.nonneg else s == -1) else 'VIOLATED'}")
    if res.beta_highs is not None:
        same = "identical to the exact dual" if res.beta_highs == res.beta else "a different optimal dual"
        lines.append("  rationalised HiGHS marginals also verify: beta = ("
                     + ", ".join(fmt_frac(x) for x in res.beta_highs) + f") -> {same}")
    else:
        lines.append("  rationalised HiGHS marginals do not verify exactly (degenerate dual); exact dual above stands")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args(argv)
    s = load_scheme(args.json)
    eig = eigenmatrices(s)
    print(f"scheme: N = {s.N}, classes = {s.classes}, valencies = {s.valencies}, multiplicities m = {eig.m}")
    print("Q (rows: classes a(dot); columns: eigenspaces E_0..E_6), Q_ir = m_r P_ri / v_i:")
    Qf = to_fractions(eig.Q)
    for i in range(s.d + 1):
        print(f"    a({CLASS_DOTS[i]:>3d}): " + " ".join(f"{fmt_frac(x):>10s}" for x in Qf[i]))
    results = run_variants(eig)
    for name, res in results.items():
        print(f"\n== {name}: {VARIANTS[name]['desc']}")
        print(certificate_text(res))
    main_res = results["main"]
    print()
    print("RESULT " + " ".join([
        f"bound_exact={fmt_frac(main_res.exact_value)}",
        f"bound_floor={main_res.floor}",
        f"highs={main_res.highs_value:.6f}",
        f"antipodal={fmt_frac(results['antipodal'].exact_value)}",
        f"antipodal_floor={results['antipodal'].floor}",
        f"forbid_16_8={fmt_frac(results['forbid_16_8'].exact_value)}",
        f"none={fmt_frac(results['none'].exact_value)}",
        f"only_pm32={fmt_frac(results['only_pm32'].exact_value)}",
    ]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

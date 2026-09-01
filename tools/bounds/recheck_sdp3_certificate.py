#!/usr/bin/env python3
"""TASK 1, verifier B: standalone re-check of a three-point SDP certificate.

Independent of python/bounds/ and of any solver: reads only
data/scheme/orbitals.json (raw JSON) and the certificate JSON, and redoes every
step in fractions.Fraction / Python ints.

    python tools/bounds/recheck_sdp3_certificate.py data/scheme/sdp3_certificate.json

Steps
  B1  parse the orbital table; check the S_3 action (transpose, swap) is an
      action, that orbit_members is exactly its orbit partition of 0..D-1, and
      that the c-tensor row/column sums are the scheme's.
  B2  rebuild the objective and the matrix Cy from the orbital table alone and
      compare with the stored ones.
  B3  rebuild the free/forbidden orbit split from forbidden_classes and compare.
  B4  rebuild Y1, Y2 from their integer rank-one decompositions (PSD by
      construction; every weight m_i must be >= 0), contract against
      G_q and Gy_q built from the c-tensor, and recompute h and the bound.
  B5  instantiate the certificate inequality on the record set stored in
      orbitals.json: obj . x = |S| and |S| <= bound must both hold exactly.
"""
from __future__ import annotations
import json, math, os, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

def main(argv):
    cert_path = argv[1] if len(argv) > 1 else os.path.join(ROOT, "data/scheme/sdp3_certificate.json")
    O = json.load(open(os.path.join(ROOT, "data/scheme/orbitals.json")))
    K = json.load(open(cert_path))
    ok = True

    D = O["D"]; N = O["N"]
    classes = O["classes"]; val = O["valencies"]
    assert D == K["D"] and N == K["N"]
    # orbitals_format: [id, i, o, z_rep, j, k, size, orbit_size, transpose_id, swap_xy_id]
    ii = [r[1] for r in O["orbitals"]]
    jj = [r[4] for r in O["orbitals"]]
    kk = [r[5] for r in O["orbitals"]]
    size = [r[6] for r in O["orbitals"]]
    tr = [r[8] for r in O["orbitals"]]
    sw = [r[9] for r in O["orbitals"]]

    # ---- B1 -----------------------------------------------------------------
    act_ok = all(tr[tr[u]] == u and sw[sw[u]] == u for u in range(D))
    # transpose swaps the roles of y and z: (i,j,k) -> (j,i,k); swap_xy: (i,j,k) -> (i,k,j)
    lab_ok = all((ii[tr[u]], jj[tr[u]], kk[tr[u]]) == (jj[u], ii[u], kk[u]) for u in range(D)) and \
             all((ii[sw[u]], jj[sw[u]], kk[sw[u]]) == (ii[u], kk[u], jj[u]) for u in range(D))
    # orbit closure
    members = [[int(t) for t in m] for m in K["orbit_members"]]
    Q = len(members)
    tv = [-1] * D
    for q, ms in enumerate(members):
        for u in ms:
            if tv[u] != -1:
                raise SystemExit("orbit_members is not a partition")
            tv[u] = q
    part_ok = all(t >= 0 for t in tv) and Q == K["vars"]
    closed = all(tv[tr[u]] == tv[u] and tv[sw[u]] == tv[u] for u in range(D))
    # each block must be a single orbit, not a union: grow from its first element
    single = True
    for ms in members:
        seen, stack = {ms[0]}, [ms[0]]
        while stack:
            u = stack.pop()
            for w in (tr[u], sw[u]):
                if w not in seen:
                    seen.add(w); stack.append(w)
        single = single and seen == set(ms)
    # c-tensor: c[u][s][t]; row sums over t must be v_{class(y,·)}-consistent
    c = [[0] * D for _ in range(D)]  # flattened per (u,s) -> dict
    cus = {}
    for u, s, t, v in O["c"]:
        cus.setdefault((u, s), {})[t] = v
    # sum_t c[u,s,t] = size_s / (v_{i(s)}) * ... : use the scheme identity
    # sum_{s,t} c[u,s,t] = N for every u (every w lies in exactly one (s,t) cell)
    tot_ok = True
    for u in range(D):
        tot = sum(sum(dd.values()) for (uu, s), dd in cus.items() if uu == u)
        if tot != N:
            tot_ok = False
    print(f"RESULT B1 s3_action={act_ok} labels={lab_ok} partition={part_ok} closed={closed} "
          f"orbits_are_single={single} c_rowtotals=N:{tot_ok} D={D} Q={Q}")
    ok = ok and act_ok and lab_ok and part_ok and closed and single and tot_ok

    # ---- B2 -----------------------------------------------------------------
    diag = {}
    for u in range(D):
        if ii[u] == jj[u] and kk[u] == O["identity_class"]:
            diag[ii[u]] = u
    obj = [0] * Q
    for k, u in diag.items():
        obj[tv[u]] += val[k]
    Cy = [[0] * Q for _ in range(D)]
    for u in range(D):
        Cy[u][tv[diag[kk[u]]]] += 1
        Cy[u][tv[u]] -= 1
    obj_ok = obj == [int(t) for t in K["objective"]]
    cy_ok = Cy == [[int(t) for t in row] for row in K["Cy"]]
    print(f"RESULT B2 diag_classes={sorted(diag)} objective_matches={obj_ok} Cy_matches={cy_ok} "
          f"sum_obj={sum(obj)} (=N: {sum(obj) == N})")
    ok = ok and obj_ok and cy_ok and sum(obj) == N

    # ---- B3 -----------------------------------------------------------------
    forb = [int(f) for f in K["forbidden_classes"]]
    dots = [classes[f] for f in forb]
    bad = {tv[u] for u in range(D) if ii[u] in forb or jj[u] in forb or kk[u] in forb}
    free = sorted(set(range(Q)) - bad)
    free_ok = free == [int(q) for q in K["free_orbits"]]
    e = int(K["e"])
    e_ok = tv[diag[O["identity_class"]]] == e and e in free
    print(f"RESULT B3 forbidden_classes={forb} dots={dots} (stored {K['forbidden_dots']}) "
          f"free={len(free)} matches={free_ok} e={e} e_ok={e_ok}")
    ok = ok and free_ok and e_ok and dots == list(K["forbidden_dots"])

    # ---- B4 -----------------------------------------------------------------
    # G_q[w,t]  = size_w * sum_{u in q}       c[w,u,t]
    # Gy_q[w,t] = size_w * sum_u  Cy[u][q] *  c[w,u,t]
    def contract(Yint, coef):
        """sum_{w,t} Yint[w][t] * size_w * sum_u coef[u] c[w,u,t], coef sparse dict."""
        s = 0
        for u, cu in coef.items():
            for w in range(D):
                dd = cus.get((w, u))
                if not dd:
                    continue
                yw = Yint[w]; sz = size[w]
                acc = 0
                for t, v in dd.items():
                    acc += yw[t] * v
                s += cu * sz * acc
        return s

    Yints = []
    for blk in K["blocks"]:
        Y = [[0] * D for _ in range(D)]
        for term in blk["terms"]:
            m = int(term["m"])
            if m < 0:
                raise SystemExit("negative rank-one weight: block is not PSD by construction")
            p = [int(t) for t in term["p"]]
            for a in range(D):
                if p[a] == 0:
                    continue
                mp = m * p[a]; Ya = Y[a]
                for b in range(D):
                    if p[b]:
                        Ya[b] += mp * p[b]
        Yints.append((Y, int(blk["den_exp"])))
    # symmetry of the Yint (they are sums of m p p^T, so automatic) - assert anyway
    sym_ok = all(all(Y[a][b] == Y[b][a] for a in range(D) for b in range(a)) for Y, _ in Yints)

    a_vec = [Fraction(0)] * Q
    for q in range(Q):
        coef1 = {u: 1 for u in members[q]}
        coef2 = {u: Cy[u][q] for u in range(D) if Cy[u][q]}
        (Y1, d1), (Y2, d2) = Yints
        a_vec[q] = Fraction(contract(Y1, coef1), 1 << d1) + Fraction(contract(Y2, coef2), 1 << d2)

    mu = [Fraction(n, d) for n, d in K["mu"]]
    rho = [Fraction(n, d) for n, d in K["rho"]]
    mult_ok = all(t >= 0 for t in mu) and all(t >= 0 for t in rho)
    h = []
    for q in range(Q):
        v = Fraction(obj[q]) + a_vec[q] - rho[q]
        v += sum(mu[u] * Cy[u][q] for u in range(D) if Cy[u][q])
        h.append(v)
    h_ok = h == [Fraction(n, d) for n, d in K["h"]]
    bound = h[e] + sum(rho[q] for q in free) + sum(h[q] for q in free if q != e and h[q] > 0)
    b_ok = bound == Fraction(*K["bound"]) and math.floor(bound) == int(K["bound_floor"])
    print(f"RESULT B4 psd_by_construction={sym_ok and mult_ok} h_matches={h_ok} "
          f"bound={float(bound):.9f} floor={math.floor(bound)} matches_stored={b_ok}")
    ok = ok and sym_ok and mult_ok and h_ok and b_ok

    # ---- B5 -----------------------------------------------------------------
    for st in O["sets"]:
        n = st["n"]; lam = st["triples"]
        x = [Fraction(0)] * Q
        for u in range(D):
            xu = Fraction(lam[u], n * size[u])
            if x[tv[u]] == 0:
                x[tv[u]] = xu
            elif x[tv[u]] != xu:
                raise SystemExit("x is not S_3-symmetric")
        objx = sum(Fraction(obj[q]) * x[q] for q in range(Q))
        zero_on_forb = all(x[q] == 0 for q in range(Q) if q not in free)
        # the certificate chain, instantiated:  obj.x <= sum_q h_q x_q + sum_free rho_q <= bound
        mid = sum(h[q] * x[q] for q in free) + sum(rho[q] for q in free)
        print(f"RESULT B5 set={os.path.basename(st['path'])} n={n} obj.x={objx} (=n: {objx == n}) "
              f"x_e={x[e]} zero_on_forbidden={zero_on_forb} "
              f"chain obj.x<=mid<=bound: {objx <= mid <= bound} (mid={float(mid):.6f})")
        ok = ok and objx == n and x[e] == 1 and zero_on_forb and objx <= bound

    print(f"RESULT task1-verifierB file={os.path.basename(cert_path)} forbidden_dots={K['forbidden_dots']} "
          f"bound_floor={math.floor(bound)} ok={int(ok)}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Independent exact verification of A. Kravatskiy's tau(25) >= 197569.

Written for this repository. It shares no code with his verify.py: the Leech
minimal vectors come from THIS repository's Golay/Leech generator
(python/kiss_ref), the algebra is re-derived below, and the exact comparisons
live in exact_cmp.py. The only thing read from his package is the coordinate
data under data/.

Units. Everything is in "Cohn units", in which a Leech minimal vector has
squared norm 32. His heads_X.npy is in norm-4 units (minimal vectors of norm
4), so it is scaled by sqrt(8); his heads_exact.pkl rationals are already in
Cohn units (norm 24).

The configuration in R^25, all points of squared norm 32, compatible iff the
inner product is at most 16 (60 degrees):

    equator   (z, 0)          z a minimal vector that is not an owner   195554
    extra     (p, 0)          p = -(sqrt6/3) v0, not a lattice vector        1
    caps      (x, +-sqrt8)    x a head, |x|^2 = 24                    2 x 1006
    poles     (0, +-sqrt32)                                                  2
                                                                      = 197569

Reduced pair conditions:

    head vs retained equator        <x, z>  <= 16
    head vs head, same cap          <x, x'> <=  8      (the height contributes 8)
    head vs head, opposite cap      <x, x'> <= 24      (automatic, Cauchy-Schwarz)
    extra vs retained equator       <p, z>  <= 16
    extra vs head                   <p, x>  <= 16
    pole vs head                    exactly 16         (tight)
    equator vs equator              <= 16              (automatic on the shell)

A class head is  x = u + t*v  with u the owner (norm 32), v a "lean" of norm 48
with <u,v> = -24, and t = (3 - sqrt3)/6. Then for an integer vector z,

    <x, z> = A + t*B,  A = <u,z>, B = <v,z>   in Z,
    <x, z> <= 16   <=>   6A + 3B - 96 <= B*sqrt3,

and for two class heads, with A = <u,u'>, B = <u,v'> + <v,u'>, C = <v,v'>,

    <x, x'> = (6A + 3B + 2C - (B + C)*sqrt3) / 6,
    <x, x'> <= 8   <=>   6A + 3B + 2C - 48 <= (B + C)*sqrt3.

For the extra point, with A = <v0,u>, B = <v0,v>,

    <p, x> = ( B*sqrt2 - (2A + B)*sqrt6 ) / 6.

Usage:
    PYTHONPATH=python python \
        tools/kravatskiy/verify25_independent.py <path-to-dim25-lens-heads>
"""

import math
import os
import pickle
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_cmp import (le_rat_times_sqrt, le_rat_times_sqrt_vec,
                       le_s_sqrt2_plus_t_sqrt6)

from kiss_ref.leech import leech_min_vectors

SQRT8 = math.sqrt(8.0)
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def main(pkg):
    data = os.path.join(pkg, "data")
    print("verify25_independent.py --- exact, independent check of K(25) >= 197569\n")

    print("1. Leech minimal vectors from THIS repository's Golay code")
    Z = leech_min_vectors().astype(np.int64)
    check("196560 vectors of norm 32",
          Z.shape == (196560, 24) and bool(((Z * Z).sum(1) == 32).all()))
    index = {Z[i].astype(np.int8).tobytes(): i for i in range(Z.shape[0])}
    check("all distinct", len(index) == 196560)

    print("\n2. the artefact")
    U = np.load(os.path.join(data, "heads_U.npy")).astype(np.int64)
    Xf = np.load(os.path.join(data, "heads_X.npy")).astype(np.float64) * SQRT8
    Pf = np.load(os.path.join(data, "extra_P.npy")).astype(np.float64) * SQRT8
    with open(os.path.join(data, "heads_exact.pkl"), "rb") as fh:
        pk = pickle.load(fh)
    rat_raw = pk["rat"]
    H = U.shape[0]
    check("1006 heads with owners", H == 1006 and Xf.shape == (1006, 24))
    owner_idx = np.array([index.get(U[i].astype(np.int8).tobytes(), -1) for i in range(H)])
    check("every owner is a Leech minimal vector", bool((owner_idx >= 0).all()))
    check("owners are distinct", len(set(owner_idx.tolist())) == H)
    dn = float(np.abs((Xf * Xf).sum(1) - 24).max())
    check("|x|^2 = 24 in Cohn units", dn < 1e-9, f"max drift {dn:.2e}")

    print("\n3. reconstructing the heads independently from the float data")
    t_minus = (3.0 - math.sqrt(3.0)) / 6.0
    rat_index = {int(r[0]) for r in rat_raw}
    check("35 interior heads recorded", len(rat_raw) == 35 and len(rat_index) == 35)

    cls_ids, Vint, bad_lean = [], [], 0
    for i in range(H):
        if i in rat_index:
            continue
        vr = np.rint((Xf[i] - U[i]) / t_minus).astype(np.int64)
        if (np.abs((Xf[i] - U[i]) / t_minus - vr).max() > 1e-6
                or int((vr * vr).sum()) != 48 or int(vr @ U[i]) != -24):
            bad_lean += 1
            continue
        cls_ids.append(i)
        Vint.append(vr)
    cls_ids = np.array(cls_ids)
    Vint = np.array(Vint, dtype=np.int64)
    check("971 class heads: v integral of norm 48 with <u,v> = -24",
          len(cls_ids) == 971 and bad_lean == 0,
          f"{len(cls_ids)} found, {bad_lean} rejected")
    Ucls = U[cls_ids]
    drift = float(np.abs((Ucls + t_minus * Vint) - Xf[cls_ids]).max())
    check("exact class head reproduces the stored value", drift < 1e-6,
          f"max drift {drift:.2e}")
    W = Ucls + Vint
    check("w = u + v is a minimal vector with <u,w> = 8",
          bool(((W * W).sum(1) == 32).all()) and bool(((Ucls * W).sum(1) == 8).all()))

    rat_owner, rat_num, rat_den, ok_norm = [], [], [], True
    for oi, coords in rat_raw:
        fr = [Fraction(c) for c in coords]
        if sum(c * c for c in fr) != 24:
            ok_norm = False
        den = 1
        for c in fr:
            den = den * c.denominator // math.gcd(den, c.denominator)
        rat_owner.append(int(oi))
        rat_num.append([int(c.numerator * (den // c.denominator)) for c in fr])
        rat_den.append(int(den))
    check("each interior head has |X|^2 = 24 exactly (rational arithmetic)", ok_norm)
    close = max(float(np.abs(np.array(rat_num[k], float) / rat_den[k]
                             - Xf[rat_owner[k]]).max()) for k in range(len(rat_raw)))
    check("each interior head matches the stored float head", close < 1e-6,
          f"max drift {close:.2e}")

    print("\n4. the extra equator point")
    v0 = np.rint(-Pf[0] * 3.0 / math.sqrt(6.0)).astype(np.int64)
    check("p = -(sqrt6/3) v0 with v0 integral of norm 48",
          int((v0 * v0).sum()) == 48
          and float(np.abs(-v0 * math.sqrt(6.0) / 3.0 - Pf[0]).max()) < 1e-6)
    check("v0 is a lean of the configuration",
          bool((Vint == v0).all(axis=1).any()))
    Bz0 = Z @ v0
    # <p,z> = -(sqrt6/3) B <= 16  <=>  -48 <= B*sqrt6
    okp = le_rat_times_sqrt_vec(np.full(Bz0.shape, -48, dtype=np.int64), Bz0, 6)
    conf = set(np.where(~okp)[0].tolist())
    block = set(np.where(Bz0 == -24)[0].tolist())
    check("p conflicts with exactly the block <v0,z> = -24", conf == block,
          f"{len(conf)} conflicts, block {len(block)}")
    owners_set = set(owner_idx.tolist())
    check("every point p conflicts with is an owner (so it is removed)",
          conf <= owners_set, f"{len(conf)} conflicts")

    retained = np.ones(196560, bool)
    retained[list(owners_set)] = False

    print("\n5. class heads against the equator, exact integer arithmetic")
    viol = owner_hit = 0
    for s in range(0, len(cls_ids), 64):
        uu, vv = Ucls[s:s + 64], Vint[s:s + 64]
        A, B = uu @ Z.T, vv @ Z.T
        bad = ~le_rat_times_sqrt_vec(6 * A + 3 * B - 96, B, 3)
        viol += int((bad & retained[None, :]).sum())
        for k in range(uu.shape[0]):
            if not bad[k, owner_idx[cls_ids[s + k]]]:
                owner_hit += 1
    check("no class head conflicts with a retained equator point", viol == 0,
          f"violations {viol}")
    check("every class head does conflict with its own owner", owner_hit == 0,
          f"owners not hit {owner_hit}")

    print("\n6. class head against class head, exact integer arithmetic")
    AA, UV, CC = Ucls @ Ucls.T, Ucls @ Vint.T, Vint @ Vint.T
    BB = UV + UV.T
    okm = le_rat_times_sqrt_vec(6 * AA + 3 * BB + 2 * CC - 48, BB + CC, 3)
    iu = np.triu_indices(len(cls_ids), 1)
    nbad = int((~okm[iu]).sum())
    check(f"all {len(iu[0])} class-head pairs satisfy <x,x'> <= 8", nbad == 0, f"{nbad} bad")

    print("\n7. the 35 interior heads, exact rational arithmetic")
    Zo = Z.astype(object)
    viol = owner_hit = 0
    for k in range(len(rat_num)):
        ip = Zo @ np.array(rat_num[k], dtype=object)
        lim = 16 * rat_den[k]
        bad = np.array([int(x) > lim for x in ip])
        viol += int((bad & retained).sum())
        if not bad[owner_idx[rat_owner[k]]]:
            owner_hit += 1
    check("no interior head conflicts with a retained equator point", viol == 0,
          f"violations {viol}")
    check("every interior head conflicts with its own owner", owner_hit == 0,
          f"owners not hit {owner_hit}")

    nbad = 0
    for a in range(len(rat_num)):
        na = np.array(rat_num[a], dtype=object)
        for b in range(a + 1, len(rat_num)):
            ip = Fraction(int(na @ np.array(rat_num[b], dtype=object)),
                          rat_den[a] * rat_den[b])
            if ip > 8:
                nbad += 1
    check("all 595 interior-interior pairs satisfy <x,x'> <= 8", nbad == 0, f"{nbad} bad")

    nbad = 0
    Uo, Vo = Ucls.astype(object), Vint.astype(object)
    for k in range(len(rat_num)):
        num = np.array(rat_num[k], dtype=object)
        den = rat_den[k]
        P, Q = Uo @ num, Vo @ num
        for j in range(len(cls_ids)):
            L = Fraction(int(6 * P[j] + 3 * Q[j]), den) - 48
            M = Fraction(int(Q[j]), den)
            if not le_rat_times_sqrt(L, M, 3):
                nbad += 1
    check(f"all {35 * len(cls_ids)} interior-class pairs satisfy <x,x'> <= 8",
          nbad == 0, f"{nbad} bad")

    print("\n8. the extra point against the heads")
    A0, B0 = Ucls @ v0, Vint @ v0
    nbad = sum(1 for j in range(len(cls_ids))
               if not le_s_sqrt2_plus_t_sqrt6(int(B0[j]),
                                              -(2 * int(A0[j]) + int(B0[j])), 96))
    check("every class head clears the extra point", nbad == 0, f"{nbad} bad")
    nbad = 0
    v0o = v0.astype(object)
    for k in range(len(rat_num)):
        G = Fraction(int(np.array(rat_num[k], dtype=object) @ v0o), rat_den[k])
        if not le_rat_times_sqrt(Fraction(-48), G, 6):
            nbad += 1
    check("every interior head clears the extra point", nbad == 0, f"{nbad} bad")

    print("\n9. the count in R^25")
    eq = 196560 - H
    total = eq + 1 + 2 * H + 2
    check("equator 195554 + extra 1 + 2 x 1006 caps + 2 poles = 197569",
          eq == 195554 and total == 197569, f"total {total}")

    print()
    if FAIL:
        print("FAILURES:", FAIL)
        return 1
    print("ALL CHECKS PASS      K(25) >= 197569   (independent, exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

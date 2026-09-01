"""TASK 1: three-point Terwilliger SDP with BOTH +-16 forbidden (the LINE problem),
alongside a re-verification of the stored +16-only certificate."""
import json, math, sys
from fractions import Fraction
import numpy as np
from bounds.sdp3_scheme import Sdp3, verify_certificate
from bounds.scheme import CLASS_DOTS, CONFLICT

MINUS16 = 1
assert CLASS_DOTS[MINUS16] == -16 and CLASS_DOTS[CONFLICT] == 16

print("RESULT stored-837-certificate:", end=" ", flush=True)
v = verify_certificate("data/scheme/sdp3_certificate.json")
print({k: (str(v[k]) if k == "bound" else v[k]) for k in ("bound_floor","D","vars","forbidden_dots","matches_stored")})
print("   exact bound =", float(v["bound"]))

S = Sdp3()
print(f"D={S.D} Q={S.Q} e={S.e}", flush=True)

# --- decisive formulation check: the record 496 (which IS +-16-free) must be feasible
for name in sorted(S.O.sets):
    n, xq = S.x_of_set(name)
    r = S.check_point(xq, [MINUS16, CONFLICT], exact_psd=True)
    print(f"RESULT feasibility[{name}] |S|={n} forbid={{-16,+16}}: obj={r['objective']} "
          f"feasible={r['feasible']} forbidden_zero={r['forbidden_zero']}", flush=True)

out = {}
for tag, forb in (("{16}", (CONFLICT,)), ("{-16,16}", (MINUS16, CONFLICT))):
    res = S.solve(forb, solver="CLARABEL")
    S.certify(res, bits=52)
    out[tag] = res
    print(f"RESULT sdp3 forbid={tag}: numeric={res.value!r} status={res.status} "
          f"exact={float(res.bound):.6f} floor={res.bound_floor} ranks={res.cert_ranks} "
          f"slack={float(res.cert_violation):.3e} ({res.solve_seconds:.1f}s)", flush=True)

S.save_certificate(out["{-16,16}"], "data/scheme/sdp3_certificate_pm16.json")
print("saved data/scheme/sdp3_certificate_pm16.json")
w = verify_certificate("data/scheme/sdp3_certificate_pm16.json")
print("RESULT reverify-pm16:", w["bound_floor"], w["forbidden_dots"], w["matches_stored"], float(w["bound"]))

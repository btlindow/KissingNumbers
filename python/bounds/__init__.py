"""bounds: rigorous upper bounds on |S| for the Leech-subset kissing project
(docs/design.md T2.x, README section 3 W1).

Everything here is exact-rational where it matters: floating point is used only
to *find* a candidate certificate (HiGHS), never to *certify* a bound.
"""

from .gegenbauer import (
    gegenbauer_coeffs,
    gegenbauer_eval,
    gegenbauer_table,
    poly_eval,
    to_gegenbauer_basis,
)
from .lp_delsarte import (
    A_ALL_LEECH,
    A_NONPOS,
    A_RESTRICTED,
    Certificate,
    delsarte_bound,
    rationalise,
    solve_lp,
    verify_certificate,
)

__all__ = [
    "gegenbauer_coeffs",
    "gegenbauer_eval",
    "gegenbauer_table",
    "poly_eval",
    "to_gegenbauer_basis",
    "A_ALL_LEECH",
    "A_NONPOS",
    "A_RESTRICTED",
    "Certificate",
    "delsarte_bound",
    "rationalise",
    "solve_lp",
    "verify_certificate",
]

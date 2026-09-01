"""group: M24 / Leech automorphism reference code (docs/design.md T4.1, shared with T3.4).

Independent of src/ (docs/design.md section 2.4): everything here is derived from
kiss_ref.golay / kiss_ref.leech and verified computationally.
"""

from .m24 import (
    M24_ORDER,
    delta,
    find_sextet,
    group_order,
    m24_generators,
    permute_mask,
    preserves_code,
    psl_generators,
    verify_aut,
    xi_numerators,
)

__all__ = [
    "M24_ORDER",
    "delta",
    "find_sextet",
    "group_order",
    "m24_generators",
    "permute_mask",
    "preserves_code",
    "psl_generators",
    "verify_aut",
    "xi_numerators",
]

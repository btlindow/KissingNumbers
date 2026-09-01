"""kiss_ref: independent pure-Python reference implementations for the
Leech-subset kissing-number project.

Nothing in this package may import from ``src/`` or read the C++ generator's
output as ground truth (docs/design.md section 2.4).
"""

from .golay import golay_codewords, is_codeword, octads, weight_distribution

__all__ = ["golay_codewords", "weight_distribution", "is_codeword", "octads"]

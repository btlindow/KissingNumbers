# Working notes

These are the detailed working notes behind the summary paper
[`docs/handover/findings.pdf`](../handover/findings.pdf). The paper states the results and labels
each one (certificate attached / certificate held / search evidence); these notes are the long form
underneath it — the constructions, the exact commands, the `RESULT` lines the runs actually printed,
the certificate paths, the negative results, and the places where a later computation corrected an
earlier conclusion. They are organised by topic rather than chronologically, and each is meant to be
readable on its own. Where a claim rests on search rather than proof, the notes say so; where an
earlier analysis was wrong, the corrected statement and the fact that it was corrected are both
recorded, usually in a "Corrections" section at the end of the document.

| Document | Contents |
|---|---|
| [`01-lattice-and-verification.md`](01-lattice-and-verification.md) | The Golay code and the Leech lattice, the 196,560 minimal vectors and their canonical ordering, the adjacency table, the GPU tightness kernel, the independent verifiers, the external record data and the coordinate map onto it. Also the build and environment pin. |
| [`02-upper-bounds.md`](02-upper-bounds.md) | Upper bounds on the maximum 60°-free subset: the Delsarte LP, the 6-class association scheme and the proof that every two-point method is capped at 9360/11, the three-point Terwilliger SDP giving 837 with an exact rational dual certificate, the clique number ω = 24 and why clique cuts are inert, and the scoping of four-point methods. |
| [`03-search-for-497.md`](03-search-for-497.md) | The search for a 497-element set: exhaustive swap search, the plateau graph, the GPU local-search engine, frame-structured and symmetry/orbit search, the 64 certified maximal 496-sets and their Co₀-inequivalence, and the Turyn-slice bound 496 ≤ opt ≤ 720. The 497 negative is characterised here rather than merely asserted. |
| [`04-structure-of-the-496.md`](04-structure-of-the-496.md) | What the record 496 actually looks like: the Turyn decomposition into three √2·E₈ blocks, monads/duads/triads, the PG(3,2) of 15 channels carrying 8×8 rook grids, the count 496 = 4·(3·12 + 9·8 + 2·8), plateau moves as single channel-pair rewirings, and the refuted structural hypotheses. Carries the notation the other documents lean on. |
| [`05-lifting-template-and-families.md`](05-lifting-template-and-families.md) | The lifting template that turns such a set into kissing configurations in dimensions 25–31: Leech automorphisms, the disjoint families, the dimension-N verifiers, the partition-weight optimality results including the four-triangle partition that gives the dimension-27 improvement, and the sweep showing the template cannot win for n ≥ 32, with the structural reason why. |
| [`06-record-and-priority.md`](06-record-and-priority.md) | The K(27) ≥ 200540 record: what it improves on, which files constitute its certificate and what the verifier checks, the literature and record-table search behind the priority claim and its limits, the independent co-discovery by Alexey Kravatsky, and the verified K(25) ≥ 197058 lowered-lift observation. |

To re-run any of this from a fresh clone — prerequisites, commands, expected output and timings, and
which claims a clone cannot re-check without a long recompute — see
[`REPRODUCE.md`](../../REPRODUCE.md) at the repository root.

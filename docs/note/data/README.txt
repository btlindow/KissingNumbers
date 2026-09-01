Certificate for  K(27) >= 200540
================================

Accompanies the note "An improved kissing number in dimension 27" (Ben Lindow,
benlindow@gmail.com). Everything needed to check the claim is in this directory.

QUICK CHECK (about 2 seconds; needs only Python 3 and NumPy):

    python3 verify.py

    expected final line:
    RESULT ok=1 dim=27 sets=4 weight=8 count=200540

The script is self-contained: it rebuilds the extended binary Golay code and all
196560 minimal vectors of the Leech lattice from scratch, reads the files below,
and checks every condition of the construction in exact integer arithmetic. It
imports nothing beyond the standard library and NumPy, and reads nothing outside
this directory.

Two slower, more exhaustive modes:

    python3 verify.py --full     (~90 s) brute-force maximum inner product over
                                 all 19317818520 pairs of Leech minimal vectors,
                                 instead of using the known inner-product set
    python3 verify.py --float    floating-point pass over explicit coordinates
                                 in R^27; reports the maximum off-diagonal inner
                                 product (exactly 2.000000000000, in the scaling
                                 where all vectors have squared norm 4)

FILES

    S_01.txt .. S_04.txt   the four subsets of the Leech minimal vectors, 496
                           vectors each, one vector per line, 24 integers.
                           Coordinates are scaled by sqrt(8), so each vector has
                           squared norm 32 and inner products are integers; two
                           vectors are at 60 degrees exactly when their inner
                           product is 16. Each S_i is 60-degree-free and the four
                           are pairwise disjoint. These are four of the five
                           496-element sets of Ma et al. (arXiv:2511.13391),
                           re-expressed in the coordinates described in the note.

    T.txt                  the 12 directions of the R^3 kissing configuration
                           (the cuboctahedron, as the A_3 = D_3 root system),
                           the four disjoint triangles partitioning them, the
                           {e_i - e_j} description with the four 3-cycles, and
                           the explicit A_3 -> D_3 isometry.

    extra.txt              the 12 additional spheres lying in the extra 3
                           dimensions, in exact form (a + b*sqrt(2))/2, together
                           with normalised coordinates and the exact maxima of
                           the relevant inner products.

    family.json            all of the above in machine-readable form, with
                           sha256 digests of the four set files.

    verify.py              the checker described above.

WHAT IS CHECKED

    every S_i lies in the Leech minimal vectors, has 496 distinct elements, and
    contains no pair at 60 degrees; the four are pairwise disjoint; the 12
    T-directions have pairwise angle at least 60 degrees and each triangle has
    its three members at pairwise 120 degrees; the 12 extra spheres are at least
    60 degrees apart and at least 30 degrees from every T-direction; and the
    resulting 200540 vectors in R^27 are pairwise at least 60 degrees apart. The
    pair counts by type sum to exactly C(200540,2) = 20108045530.

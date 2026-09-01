"""T5.1 — slack in the CJKT/PackingStar template's R^d half (README 1.3, W4).

triangles.py      exact maximum triangle packing / maximum-weight group partition (ILP, HiGHS via
                  scipy.optimize.milp) of the K(d) kissing configurations, with certificates; the
                  counting bound that no K(d)-point configuration can do better; a continuous search
                  for non-lattice configurations with prescribed triangles.
extra_spheres.py  extra directions (cos <= 1/2 among themselves, >= 30 degrees from T): exact maximum
                  over lattice / rotated candidate pools (ILP), continuous K(d)+1 search, the K(d) bound.
assign.py         family.json -> count; assignment of the larger S_i to the heavier groups; rewrite a
                  family with the optimal partition (this is what produces the improved dim-27 family).
"""

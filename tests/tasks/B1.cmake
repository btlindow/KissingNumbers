# B1 — clique number of the conflict graph G. Fast acceptance subset of the
# clique_g tool (common-neighbour counts, explicit 24-clique verified by exact
# integer arithmetic, B&B exercised on a triangle neighbourhood). The full
# omega(G) = 24 run is `tools/clique_g` without --test (docs/reports/B1.md).
kiss_add_test(NAME test_clique_g
  SOURCES ${CMAKE_SOURCE_DIR}/tools/clique_g.cpp
  ARGS --test
  TIMEOUT 300)

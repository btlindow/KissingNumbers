# T3.4 — orbits / orbit conflict graph / weighted orbit MIS acceptance test
# (CPU only; reads data/adj.u32 via mmap for the orbit-graph part, exit 77 =
# skip when it is absent — the synthetic B&B, orbit-invariant and
# stabiliser-of-the-496 checks always run). Working directory is the source
# root (kiss_add_test) so `data` and `runs/orbits` resolve.
kiss_add_test(NAME test_orbits SOURCES test_orbits.cpp TIMEOUT 900 ARGS data)
set_tests_properties(test_orbits PROPERTIES SKIP_RETURN_CODE ${KISS_SKIP_RC})

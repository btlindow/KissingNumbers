# T3.1 — swap neighbourhoods acceptance test (CPU only; reads data/adj.u32 via mmap).
# Runs from CMAKE_SOURCE_DIR (kiss_add_test) so `data` resolves. Exit 77 = skip
# when data/adj.u32 is absent (synthetic checks still run first).
kiss_add_test(NAME test_swaps SOURCES test_swaps.cpp TIMEOUT 900 ARGS data)
set_tests_properties(test_swaps PROPERTIES SKIP_RETURN_CODE ${KISS_SKIP_RC})

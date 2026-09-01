# T3.1b — plateau-walk acceptance test (CPU only; reads data/adj.u32 via mmap).
# Runs from CMAKE_SOURCE_DIR so `data` resolves. Exit 77 = skip when data/adj.u32
# is absent (the synthetic checks still run first).
kiss_add_test(NAME test_plateau SOURCES test_plateau.cpp TIMEOUT 900 ARGS data)
set_tests_properties(test_plateau PROPERTIES SKIP_RETURN_CODE ${KISS_SKIP_RC})

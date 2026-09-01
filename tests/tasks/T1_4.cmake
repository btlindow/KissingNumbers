# T1.4 — verifiers (verify_independent, tightness_cpu) acceptance test.
# Runs from CMAKE_SOURCE_DIR so it can find data/S496.txt and data/S488.txt
# (both optional: the fixture checks are skipped with a note if absent).
kiss_add_test(NAME test_verify SOURCES test_verify.cpp TIMEOUT 600)

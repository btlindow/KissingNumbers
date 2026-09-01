# T3.3 — frame-structured search acceptance test (CPU only; regenerates C, reads
# data/S496.txt from CMAKE_SOURCE_DIR for the cross-structure and extension checks).
kiss_add_test(NAME test_frames SOURCES test_frames.cpp TIMEOUT 900 ARGS data/S496.txt)

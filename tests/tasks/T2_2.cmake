# T2.2 — pair-class histogram kernel acceptance test (kernel vs CPU double
# loop on a sample of y's, row/column sums on all y, sub-range entry point).
# Working directory is the source root (kiss_add_test), so `data` resolves.
kiss_add_test(NAME test_scheme_gpu SOURCES test_scheme_gpu.cu GPU LIBS kiss_cuda TIMEOUT 600)

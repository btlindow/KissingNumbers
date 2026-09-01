# T2.3 — triple-class histogram kernel acceptance test (kernel vs CPU double
# loop on a sample of z's, sums/marginals on all z, sub-range entry point).
# Working directory is the source root (kiss_add_test), so `data` resolves.
kiss_add_test(NAME test_triple_gpu SOURCES test_triple_gpu.cu GPU LIBS kiss_cuda TIMEOUT 900)

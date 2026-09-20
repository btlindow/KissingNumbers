# T3.2b — per-chain ILS kernel acceptance test (invariants after every launch,
# refill from 460-subsets of the 496, climb from greedy sets, seeds from the 488
# and the 496 in both modes, T3.1's plateau moves, throughput). Needs
# data/adj.u32 (skips otherwise) and ~5.5 GB of free device memory (skips with
# 77 after retrying for a few minutes). verify_independent over 512 chains per
# launch is parallel with OpenMP (the nvcc host pass needs the flag).
kiss_add_test(NAME test_ls_search SOURCES test_ls_search.cu GPU LIBS kiss_cuda TIMEOUT 2400)
if(TARGET test_ls_search)
  kiss_cuda_openmp(test_ls_search)
endif()

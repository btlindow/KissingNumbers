# T3.2a — chain state + incremental move kernels acceptance test (fuzz vs CPU
# model, rescan/drain refill, throughput). Needs data/adj.u32 (skips otherwise)
# and ~4.5 GB of device memory (skips with 77 if < 5.5 GB free after retries).
# The CPU model is parallel over chains with OpenMP (the nvcc host pass needs the flag).
kiss_add_test(NAME test_ls_moves SOURCES test_ls_moves.cu GPU LIBS kiss_cuda TIMEOUT 1800)
if(TARGET test_ls_moves)
  kiss_cuda_openmp(test_ls_moves)
endif()

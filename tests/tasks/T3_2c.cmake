# T3.2c — host driver acceptance test: config/JSON unit tests (CPU), the record
# custody chain (target 490 from 480-subsets of the 496), checkpoint/resume
# determinism and a 3-minute soak with --selfcheck. Runs tools/gpu_mis as a
# subprocess, so it needs data/adj.u32 and ~5.5 GB of free device memory
# (skips with 77 otherwise). The soak length is the --soak-minutes argument.
kiss_add_test(NAME test_gpu_mis SOURCES test_gpu_mis.cpp GPU LIBS kiss_cuda TIMEOUT 1500
              ARGS --soak-minutes 3)
if(TARGET test_gpu_mis)
  add_dependencies(test_gpu_mis gpu_mis verify_s)
endif()

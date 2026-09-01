# T1.5 — GPU fused tightness kernel acceptance test.
# The CPU reference inside the .cu file uses OpenMP; nvcc's host pass needs
# -fopenmp explicitly (OpenMP::OpenMP_CXX only adds it for CXX sources).
kiss_add_test(NAME test_tightness_gpu SOURCES test_tightness_gpu.cu GPU LIBS kiss_cuda TIMEOUT 900)
if(TARGET test_tightness_gpu)
  target_compile_options(test_tightness_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif()

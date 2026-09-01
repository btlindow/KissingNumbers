# ctest targets (T0.1). GPU tests carry the label `gpu` and are skipped
# gracefully (exit code 77 → SKIP_RETURN_CODE) when no device is present.
set(KISS_SKIP_RC 77)

if(KISS_ENABLE_GPU_TESTS)
  add_executable(smoke_gpu smoke_gpu.cu)
  target_link_libraries(smoke_gpu PRIVATE kiss kiss_cuda)
  kiss_apply_flags(smoke_gpu)
  add_test(NAME smoke_gpu COMMAND smoke_gpu)
  set_tests_properties(smoke_gpu PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE ${KISS_SKIP_RC}
    PASS_REGULAR_EXPRESSION "RESULT .*ok=1"
    TIMEOUT 60)
endif()

# T1.1 — Golay code acceptance test (CPU only).
add_executable(test_golay test_golay.cpp)
target_link_libraries(test_golay PRIVATE kiss)
kiss_apply_flags(test_golay)
add_test(NAME test_golay COMMAND test_golay)
set_tests_properties(test_golay PROPERTIES
  PASS_REGULAR_EXPRESSION "RESULT .*ok=1"
  TIMEOUT 120)

# T1.2 — Leech minimal vectors / canonical data / membership test (CPU only).
add_executable(test_leech test_leech.cpp)
target_link_libraries(test_leech PRIVATE kiss)
kiss_apply_flags(test_leech)
add_test(NAME test_leech COMMAND test_leech)
set_tests_properties(test_leech PROPERTIES
  PASS_REGULAR_EXPRESSION "RESULT .*ok=1"
  TIMEOUT 300)

# kiss_info doubles as the CPU-side smoke test of the library/tool wiring.
add_test(NAME kiss_info COMMAND kiss_info)
set_tests_properties(kiss_info PROPERTIES
  PASS_REGULAR_EXPRESSION "RESULT .*ok=1"
  TIMEOUT 30)

if(KISS_SANITIZE)
  # cudart maps a huge virtual range at init; ASan's shadow gap must be unprotected.
  set_tests_properties(kiss_info test_golay test_leech PROPERTIES
    ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0")
  if(KISS_ENABLE_GPU_TESTS)
    set_tests_properties(smoke_gpu PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0")
  endif()
endif()

# ---------------------------------------------------------------------------
# Per-task test registration (Wave 3+). Each task adds its own file
# tests/tasks/<task>.cmake calling kiss_add_test(); no edits to this file needed.
#   kiss_add_test(NAME test_foo SOURCES test_foo.cu [GPU] [LIBS kiss_cuda] [TIMEOUT 300] [ARGS ...])
# GPU tests get the `gpu` label, SKIP_RETURN_CODE 77 and are only built when
# KISS_ENABLE_GPU_TESTS is on. Tests must print a final line `RESULT ... ok=1`.
# ---------------------------------------------------------------------------
function(kiss_add_test)
  cmake_parse_arguments(KT "GPU" "NAME;TIMEOUT" "SOURCES;LIBS;ARGS" ${ARGN})
  if(KT_GPU AND NOT KISS_ENABLE_GPU_TESTS)
    return()
  endif()
  if(NOT KT_TIMEOUT)
    set(KT_TIMEOUT 300)
  endif()
  add_executable(${KT_NAME} ${KT_SOURCES})
  target_link_libraries(${KT_NAME} PRIVATE kiss ${KT_LIBS})
  kiss_apply_flags(${KT_NAME})
  add_test(NAME ${KT_NAME} COMMAND ${KT_NAME} ${KT_ARGS}
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(${KT_NAME} PROPERTIES
    PASS_REGULAR_EXPRESSION "RESULT .*ok=1"
    TIMEOUT ${KT_TIMEOUT})
  if(KT_GPU)
    set_tests_properties(${KT_NAME} PROPERTIES LABELS "gpu" SKIP_RETURN_CODE ${KISS_SKIP_RC})
  endif()
  if(KISS_SANITIZE)
    set_tests_properties(${KT_NAME} PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0")
  endif()
endfunction()

file(GLOB KISS_TASK_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tasks/*.cmake)
foreach(f ${KISS_TASK_TESTS})
  include(${f})
endforeach()

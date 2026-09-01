# Project options and toolchain pin checks (T0.1).
include_guard(GLOBAL)

option(KISS_ENABLE_GPU_TESTS "Register ctest targets that need a CUDA device (label: gpu)" ON)
option(KISS_WERROR            "Treat warnings as errors"                                    ON)
option(KISS_SANITIZE          "Build host code with AddressSanitizer + UBSan"               OFF)

# docs/design.md §1: pin CUDA 12.8. /usr/local/cuda symlinks to 13.0, and 12.8's nvcc
# is on PATH; the presets set CMAKE_CUDA_COMPILER explicitly so the two can
# never disagree. Warn loudly if a different toolkit sneaks in.
set(KISS_EXPECTED_NVCC "/usr/local/cuda-12.8/bin/nvcc" CACHE FILEPATH
    "nvcc the project is pinned to (docs/design.md §1)")
if(NOT CMAKE_CUDA_COMPILER STREQUAL KISS_EXPECTED_NVCC)
  message(WARNING
    "CMAKE_CUDA_COMPILER is '${CMAKE_CUDA_COMPILER}', expected '${KISS_EXPECTED_NVCC}'. "
    "Use one of the presets in CMakePresets.json (cmake --preset release).")
endif()
if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8
   OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  message(WARNING "nvcc ${CMAKE_CUDA_COMPILER_VERSION} found; the project is pinned to 12.8.x")
endif()
if(NOT "86" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'; the target GPU is sm_86 (docs/design.md §1)")
endif()

message(STATUS "CUDA compiler : ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
message(STATUS "CUDA archs    : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ compiler  : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "GPU tests     : ${KISS_ENABLE_GPU_TESTS}   sanitize: ${KISS_SANITIZE}   werror: ${KISS_WERROR}")

# Short git hash of the source tree ("unknown" outside a repo).
function(kiss_git_hash out_var)
  find_package(Git QUIET)
  set(hash "unknown")
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE h OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE rc ERROR_QUIET)
    if(rc EQUAL 0)
      set(hash "${h}")
    endif()
  endif()
  set(${out_var} "${hash}" PARENT_SCOPE)
endfunction()

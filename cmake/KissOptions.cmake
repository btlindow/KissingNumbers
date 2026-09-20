# Project options and toolchain pin checks (T0.1).
include_guard(GLOBAL)

option(KISS_ENABLE_GPU_TESTS "Register ctest targets that need a CUDA device (label: gpu)" ON)
option(KISS_SANITIZE          "Build host code with AddressSanitizer + UBSan"               OFF)

# -Werror is the default where the code was developed (gcc). MSVC's /W4 flags a
# different, larger set — mostly signed/unsigned narrowing inside the STL — so
# warnings are reported there but not fatal. Override with -DKISS_WERROR=ON.
if(MSVC)
  option(KISS_WERROR "Treat warnings as errors" OFF)
else()
  option(KISS_WERROR "Treat warnings as errors" ON)
endif()

# ---------------------------------------------------------------------------
# CUDA toolkit
#
# docs/design.md §1 pins a toolkit per machine, because more than one is
# usually installed and PATH vs. symlink must never disagree. The pin itself
# lives in CMakePresets.json (CMAKE_CUDA_COMPILER); this file only checks that
# whatever CMake resolved is inside the supported range and is what the preset
# asked for.
#
#   Linux box (§1):   CUDA 12.8 at /usr/local/cuda-12.8
#   Windows box (§1): CUDA 13.x under
#                     C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vNN.N
#
# Both are supported; sm_86 is current in each. CUDA 13 dropped Maxwell,
# Pascal and Volta, which this project never targeted.
# ---------------------------------------------------------------------------
set(KISS_CUDA_MIN 12.8 CACHE STRING "Oldest CUDA toolkit the project is tested against")
set(KISS_CUDA_MAX 14.0 CACHE STRING "First CUDA toolkit version NOT tested (exclusive)")
set(KISS_EXPECTED_NVCC "" CACHE FILEPATH
    "nvcc the active preset pins (docs/design.md §1); empty disables the check")

if(KISS_EXPECTED_NVCC AND NOT CMAKE_CUDA_COMPILER STREQUAL KISS_EXPECTED_NVCC)
  message(WARNING
    "CMAKE_CUDA_COMPILER is '${CMAKE_CUDA_COMPILER}', expected '${KISS_EXPECTED_NVCC}'. "
    "Use one of the presets in CMakePresets.json (cmake --preset release).")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS KISS_CUDA_MIN
   OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL KISS_CUDA_MAX)
  message(WARNING "nvcc ${CMAKE_CUDA_COMPILER_VERSION} found; tested range is "
                  "[${KISS_CUDA_MIN}, ${KISS_CUDA_MAX}).")
endif()

# Every machine the project has run on has an sm_86 card (RTX 3070 Laptop, then
# RTX 3080 Ti). The kernels assume Ampere occupancy and __dp4a throughput; they
# are correct elsewhere but untuned, so say so rather than fail.
if(NOT "86" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'; the tuned target is sm_86 (docs/design.md §1)")
endif()

message(STATUS "Host system   : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CUDA compiler : ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
message(STATUS "CUDA archs    : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ compiler  : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
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

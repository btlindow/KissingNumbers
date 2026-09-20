# T0.1 — build system for the Leech-subset kissing-number project.
# See docs/design.md §1 (machine envelope), §2.1 (layout), §2.4 (conventions).
#
# Builds on Linux/gcc and on Windows/MSVC; pick a preset from CMakePresets.json
# rather than configuring by hand, because the CUDA toolkit is pinned there.
cmake_minimum_required(VERSION 3.25)

project(kissing
  VERSION 0.1.0
  DESCRIPTION "Leech-lattice kissing-configuration search (CPU + CUDA)"
  LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(KissOptions)   # options + compiler/toolchain checks
include(KissFlags)     # kiss_apply_flags(<target>) — §2.4 flags per configuration

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit ${KISS_CUDA_MIN} REQUIRED)   # CUDA::cudart, CUDA::curand, ...

# MSVC's own /openmp is OpenMP 2.0 and rejects unsigned loop counters, which
# src/ uses throughout; ask CMake for the LLVM runtime that ships with VS.
# (OpenMP_RUNTIME_MSVC is CMake 3.30+; older CMake on Windows would need the
# loops rewritten, so say why rather than silently falling back.)
if(MSVC)
  if(CMAKE_VERSION VERSION_LESS 3.30)
    message(FATAL_ERROR "CMake >= 3.30 is required on MSVC (OpenMP_RUNTIME_MSVC); found ${CMAKE_VERSION}")
  endif()
  set(OpenMP_RUNTIME_MSVC "llvm")
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Generated version header (git hash logged in every run directory, §2.4).
kiss_git_hash(KISS_GIT_HASH)
configure_file(cmake/version_info.h.in
               "${CMAKE_BINARY_DIR}/generated/kiss/version_info.h" @ONLY)

# ---------------------------------------------------------------------------
# kiss — CPU C++ library (src/*.cpp, headers in include/kiss/)
# ---------------------------------------------------------------------------
file(GLOB KISS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(kiss STATIC ${KISS_SOURCES})
target_include_directories(kiss PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(kiss PUBLIC OpenMP::OpenMP_CXX)
kiss_apply_flags(kiss)

# ---------------------------------------------------------------------------
# kiss_cuda — CUDA library (cuda/*.cu)
# ---------------------------------------------------------------------------
file(GLOB KISS_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")
add_library(kiss_cuda STATIC ${KISS_CUDA_SOURCES})
target_include_directories(kiss_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
target_link_libraries(kiss_cuda PUBLIC kiss CUDA::cudart CUDA::curand)
kiss_apply_flags(kiss_cuda)

# ---------------------------------------------------------------------------
# tools/*.cpp — one executable per file, output in build/<preset>/tools/
# ---------------------------------------------------------------------------
file(GLOB KISS_TOOL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp")
set(KISS_TOOL_TARGETS "")
foreach(src IN LISTS KISS_TOOL_SOURCES)
  get_filename_component(name "${src}" NAME_WE)
  add_executable(${name} "${src}")
  target_link_libraries(${name} PRIVATE kiss kiss_cuda)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  kiss_apply_flags(${name})
  list(APPEND KISS_TOOL_TARGETS ${name})
endforeach()
message(STATUS "kiss tools: ${KISS_TOOL_TARGETS}")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

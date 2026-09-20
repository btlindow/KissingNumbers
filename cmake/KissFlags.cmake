# Per-target compile flags implementing docs/design.md §2.4:
#   host : -O3 -Wall -Wextra -Werror (release)          | -O0 -g (debug)
#   cuda : -O3 -lineinfo --expt-relaxed-constexpr        | -O0 -g -G (debug)
#   sanitize preset: host ASan + UBSan (KISS_SANITIZE=ON)
#
# MSVC equivalents, used when CMAKE_CXX_COMPILER_ID is MSVC:
#   /O2 /W4 (/WX only if KISS_WERROR)   | /Od /Zi
#   /openmp:llvm — the classic /openmp is OpenMP 2.0 and rejects the unsigned
#   loop counters used throughout src/; the LLVM runtime accepts them.
#   /EHsc, /utf-8 (the sources are UTF-8 and full of mathematical symbols),
#   /bigobj (tools/gpu_mis.cpp and tools/frames.cpp exceed the 2^16 section
#   limit), /Zc:__cplusplus (otherwise __cplusplus reports 199711L).
#   The CUDA host pass additionally needs /Zc:preprocessor: CUDA 13's bundled
#   CCCL (<cuda/std/__cccl/preprocessor.h>) refuses to compile under MSVC's
#   traditional preprocessor, and cuda/build_adjacency.cu pulls in CUB.
#   /wd4211 and /wd4125 silence warnings in code nvcc generates and then
#   attributes to our lines with #line: the .stub.c files redeclare extern as
#   static, and the embedded fatbin is a string literal full of octal escapes.
include_guard(GLOBAL)

function(kiss_apply_flags target)
  if(MSVC)
    set(warn /W4)
    if(KISS_WERROR)
      list(APPEND warn /WX)
    endif()
    # nvcc forwards host flags one at a time; commas would be split.
    set(warn_host "")
    foreach(f IN LISTS warn)
      list(APPEND warn_host "-Xcompiler=${f}")
    endforeach()

    target_compile_options(${target} PRIVATE
      # --- C++ ---
      $<$<COMPILE_LANGUAGE:CXX>:${warn};/EHsc;/utf-8;/bigobj;/Zc:__cplusplus;/permissive->
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od;/Zi>
      # --- CUDA ---
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;${warn_host};-Xcompiler=/EHsc;-Xcompiler=/utf-8;-Xcompiler=/bigobj;-Xcompiler=/Zc:preprocessor;-Xcompiler=/wd4211;-Xcompiler=/wd4125>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;-lineinfo>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
    )
    # fopen/gmtime/getenv are "unsafe" to MSVC; the code checks every return.
    target_compile_definitions(${target} PRIVATE
      _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)

    if(KISS_SANITIZE)
      # MSVC ships ASan but no UBSan, and it is incompatible with /RTC and with
      # the /MDd CRT used by the Debug config's default runtime.
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>)
    endif()
    return()
  endif()

  set(cxx_warn -Wall -Wextra)
  if(KISS_WERROR)
    list(APPEND cxx_warn -Werror)
  endif()

  # Host-compiler flags for the CUDA host pass (nvcc -Xcompiler=...).
  string(REPLACE ";" "," cxx_warn_csv "${cxx_warn}")

  target_compile_options(${target} PRIVATE
    # --- C++ ---
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_warn}>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0;-g>
    # --- CUDA ---
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=${cxx_warn_csv}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
  )
  if(KISS_WERROR)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
  endif()

  if(KISS_SANITIZE)
    # nvcc splits -Xcompiler on commas, so each host flag gets its own -Xcompiler.
    set(san -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    set(san_cuda "")
    foreach(f IN LISTS san)
      list(APPEND san_cuda "-Xcompiler=${f}")
    endforeach()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${san}>
      $<$<COMPILE_LANGUAGE:CUDA>:${san_cuda}>)
    target_link_options(${target} PRIVATE ${san})
  endif()
endfunction()

# OpenMP for the *host pass of a .cu file*. OpenMP::OpenMP_CXX guards its flags
# with $<COMPILE_LANGUAGE:CXX>, so a CUDA source that runs a parallel CPU
# reference gets nothing from it and has to ask for the flag itself. The
# spelling is compiler-specific, and on MSVC the wrong one is merely *ignored*
# (`cl : Command line warning D9002`), which silently serialises the reference
# instead of failing the build -- so it lives here and not in three task files.
function(kiss_cuda_openmp target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp:llvm>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    target_link_options(${target} PRIVATE -fopenmp)
  endif()
endfunction()

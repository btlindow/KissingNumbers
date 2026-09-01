# Per-target compile flags implementing docs/design.md §2.4:
#   host : -O3 -Wall -Wextra -Werror (release)          | -O0 -g (debug)
#   cuda : -O3 -lineinfo --expt-relaxed-constexpr        | -O0 -g -G (debug)
#   sanitize preset: host ASan + UBSan (KISS_SANITIZE=ON)
include_guard(GLOBAL)

function(kiss_apply_flags target)
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

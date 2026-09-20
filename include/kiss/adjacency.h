// T1.6 — host-side reader for the conflict-graph adjacency table
// data/adj.u32 (docs/design.md §2.2, §2.3).
//
// File format: raw uint32, N rows × DEG columns, row-major; row i lists the
// indices j with ⟨C[i],C[j]⟩ = 16, sorted ascending. 3,616,704,000 bytes.
// The table is never loaded into RAM (docs/design.md §1: ~7 GB free): it is mapped
// read-only — mmap PROT_READ/MAP_PRIVATE on POSIX, a PAGE_READONLY section on
// Win32 — and pages come from the page cache on demand.
//
// The device upload (`to_device`) is the free function
// kiss::cuda::adjacency_to_device(const Adjacency&) in cuda/adjacency_cuda.h,
// so that libkiss stays a pure CPU library (kiss_cuda links kiss, not the
// other way round).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "kiss/types.h"

namespace kiss {

class Adjacency {
 public:
  static constexpr std::size_t EXPECTED_BYTES =
      static_cast<std::size_t>(N) * static_cast<std::size_t>(DEG) * sizeof(uint32_t);

  // mmap the file; throws std::runtime_error if it cannot be opened, mapped,
  // or its size is not exactly EXPECTED_BYTES.
  explicit Adjacency(const std::filesystem::path& file);
  ~Adjacency();
  Adjacency(const Adjacency&) = delete;
  Adjacency& operator=(const Adjacency&) = delete;
  Adjacency(Adjacency&& o) noexcept;
  Adjacency& operator=(Adjacency&& o) noexcept;

  // DEG sorted entries of row v (v < N; unchecked).
  const uint32_t* row(uint32_t v) const {
    return data_ + static_cast<std::size_t>(v) * static_cast<std::size_t>(DEG);
  }
  // Binary search of v in row(u).
  bool adjacent(uint32_t u, uint32_t v) const;

  std::size_t rows() const { return static_cast<std::size_t>(N); }
  std::size_t cols() const { return static_cast<std::size_t>(DEG); }
  std::size_t bytes() const { return EXPECTED_BYTES; }
  const uint32_t* data() const { return data_; }
  const std::filesystem::path& path() const { return path_; }

  // Hint the kernel that a full sequential pass is coming (madvise); optional,
  // and a no-op on Win32 (see src/adjacency.cpp).
  void advise_sequential() const;

 private:
  void unmap_() noexcept;

  std::filesystem::path path_;
  const uint32_t* data_ = nullptr;
  void* map_ = nullptr;
  std::size_t map_bytes_ = 0;
};

}  // namespace kiss

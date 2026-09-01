// T1.6 — mmap-backed Adjacency (see include/kiss/adjacency.h).
#include "kiss/adjacency.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace kiss {

Adjacency::Adjacency(const std::filesystem::path& file) : path_(file) {
  const int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("Adjacency: cannot open " + file.string() + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const int e = errno;
    ::close(fd);
    throw std::runtime_error("Adjacency: fstat failed on " + file.string() + ": " + std::strerror(e));
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  if (size != EXPECTED_BYTES) {
    ::close(fd);
    throw std::runtime_error("Adjacency: " + file.string() + " has " + std::to_string(size) +
                             " bytes, expected " + std::to_string(EXPECTED_BYTES) +
                             " (= 196560 x 4600 x 4); regenerate with tools/build_adj");
  }
  void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  const int e = errno;
  ::close(fd);  // the mapping keeps its own reference
  if (p == MAP_FAILED)
    throw std::runtime_error("Adjacency: mmap failed on " + file.string() + ": " + std::strerror(e));
  map_ = p;
  map_bytes_ = size;
  data_ = static_cast<const uint32_t*>(p);
}

Adjacency::~Adjacency() {
  if (map_) ::munmap(map_, map_bytes_);
}

Adjacency::Adjacency(Adjacency&& o) noexcept
    : path_(std::move(o.path_)), data_(o.data_), map_(o.map_), map_bytes_(o.map_bytes_) {
  o.data_ = nullptr;
  o.map_ = nullptr;
  o.map_bytes_ = 0;
}

Adjacency& Adjacency::operator=(Adjacency&& o) noexcept {
  if (this != &o) {
    if (map_) ::munmap(map_, map_bytes_);
    path_ = std::move(o.path_);
    data_ = o.data_;
    map_ = o.map_;
    map_bytes_ = o.map_bytes_;
    o.data_ = nullptr;
    o.map_ = nullptr;
    o.map_bytes_ = 0;
  }
  return *this;
}

bool Adjacency::adjacent(uint32_t u, uint32_t v) const {
  const uint32_t* r = row(u);
  return std::binary_search(r, r + DEG, v);
}

void Adjacency::advise_sequential() const {
  if (map_) ::madvise(map_, map_bytes_, MADV_SEQUENTIAL);
}

}  // namespace kiss

// T1.6 — mmap-backed Adjacency (see include/kiss/adjacency.h).
//
// Two backends, one lifetime contract: after the constructor returns, `map_`
// is a read-only view of the whole 3.62 GB file and no descriptor or handle is
// still held open — on POSIX the mapping keeps its own reference to the file,
// and on Win32 a view stays valid after both the section and file handles are
// closed. So the destructor only has to release the view.
#include "kiss/adjacency.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace kiss {

namespace {

#if defined(_WIN32)
// Win32 errors are numeric; FormatMessage gives the same kind of text
// std::strerror does, so failures read alike on both platforms.
std::string win_error(DWORD e) {
  char* msg = nullptr;
  const DWORD n = ::FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, e, 0, reinterpret_cast<char*>(&msg), 0, nullptr);
  std::string s = (n && msg) ? std::string(msg, n) : ("error " + std::to_string(e));
  if (msg) ::LocalFree(msg);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}
#endif

}  // namespace

Adjacency::Adjacency(const std::filesystem::path& file) : path_(file) {
#if defined(_WIN32)
  const HANDLE fh = ::CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fh == INVALID_HANDLE_VALUE)
    throw std::runtime_error("Adjacency: cannot open " + file.string() + ": " +
                             win_error(::GetLastError()));

  LARGE_INTEGER li{};
  if (!::GetFileSizeEx(fh, &li)) {
    const DWORD e = ::GetLastError();
    ::CloseHandle(fh);
    throw std::runtime_error("Adjacency: cannot size " + file.string() + ": " + win_error(e));
  }
  const auto size = static_cast<std::size_t>(li.QuadPart);
  if (size != EXPECTED_BYTES) {
    ::CloseHandle(fh);
    throw std::runtime_error("Adjacency: " + file.string() + " has " + std::to_string(size) +
                             " bytes, expected " + std::to_string(EXPECTED_BYTES) +
                             " (= 196560 x 4600 x 4); regenerate with tools/build_adj");
  }

  const HANDLE mh = ::CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mh) {
    const DWORD e = ::GetLastError();
    ::CloseHandle(fh);
    throw std::runtime_error("Adjacency: CreateFileMapping failed on " + file.string() + ": " +
                             win_error(e));
  }
  void* p = ::MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
  const DWORD e = ::GetLastError();
  ::CloseHandle(mh);  // the view keeps its own reference
  ::CloseHandle(fh);
  if (!p)
    throw std::runtime_error("Adjacency: MapViewOfFile failed on " + file.string() + ": " +
                             win_error(e));
#else
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
#endif
  map_ = p;
  map_bytes_ = size;
  data_ = static_cast<const uint32_t*>(p);
}

// Release the view/mapping. Free function so the destructor and the move
// assignment cannot drift apart.
void Adjacency::unmap_() noexcept {
  if (!map_) return;
#if defined(_WIN32)
  ::UnmapViewOfFile(map_);
#else
  ::munmap(map_, map_bytes_);
#endif
  map_ = nullptr;
  map_bytes_ = 0;
  data_ = nullptr;
}

Adjacency::~Adjacency() { unmap_(); }

Adjacency::Adjacency(Adjacency&& o) noexcept
    : path_(std::move(o.path_)), data_(o.data_), map_(o.map_), map_bytes_(o.map_bytes_) {
  o.data_ = nullptr;
  o.map_ = nullptr;
  o.map_bytes_ = 0;
}

Adjacency& Adjacency::operator=(Adjacency&& o) noexcept {
  if (this != &o) {
    unmap_();
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
#if defined(_WIN32)
  // Win32 has no post-hoc madvise: the access pattern is declared at open time
  // via FILE_FLAG_SEQUENTIAL_SCAN, and we open for random access because
  // `adjacent()` binary-searches single rows. PrefetchVirtualMemory would warm
  // the whole 3.62 GB, which is the opposite of what the hint is for.
  // Deliberately a no-op; correctness does not depend on it.
#else
  if (map_) ::madvise(map_, map_bytes_, MADV_SEQUENTIAL);
#endif
}

}  // namespace kiss

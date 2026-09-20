// T1.2 — I/O for the vector-set text format (docs/design.md §2.2, README §5).
//
// S text format: one vector per line, 24 space-separated integers; lines whose
// first non-blank character is '#' are comments; blank lines are ignored; a
// trailing "# ..." after the 24 values is also accepted as a comment. Order is
// free. Reading does NOT check norms or lattice membership — that is the job
// of the verifiers (T1.4).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "kiss/types.h"

namespace kiss {

// std::fopen on a std::filesystem::path. Needed because path::c_str() is
// `const wchar_t*` on Windows, where the narrow fopen would not compile and a
// path::string() round trip would lose any character outside the active code
// page. Returns nullptr exactly as std::fopen does; callers check and throw.
inline std::FILE* fopen_path(const std::filesystem::path& path, const char* mode) {
#if defined(_WIN32)
  const std::wstring wmode(mode, mode + std::char_traits<char>::length(mode));
  return ::_wfopen(path.c_str(), wmode.c_str());
#else
  return std::fopen(path.c_str(), mode);
#endif
}

// Parse a set file. Throws std::runtime_error naming file and line on a row
// with the wrong number of values, a non-integer token or a value outside
// int8 range [-128,127].
std::vector<Vec> read_set(const std::filesystem::path& path);

// Write a set file. If header_comment is non-empty each of its lines is
// written first prefixed by "# ". Throws std::runtime_error on I/O failure.
void write_set(const std::filesystem::path& path, const std::vector<Vec>& S,
               const std::string& header_comment = "");

// ---- additions beyond §2.3 (documented in docs/reports/T1.2.md) ------------

// Whole-file binary helpers. Throw std::runtime_error on failure.
std::vector<uint8_t> read_binary_file(const std::filesystem::path& path);
void write_binary_file(const std::filesystem::path& path, const void* data, std::size_t bytes);

// SHA-256 (FIPS 180-4) as 64 lowercase hex digits; sha256_file streams the file.
std::string sha256_hex(const void* data, std::size_t bytes);
std::string sha256_file(const std::filesystem::path& path);

}  // namespace kiss

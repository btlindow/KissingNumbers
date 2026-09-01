// T1.2 — set-file text I/O, binary file helpers, SHA-256. See include/kiss/io.h.
#include "kiss/io.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace kiss {

namespace {

[[noreturn]] void fail(const std::filesystem::path& path, std::size_t line, const std::string& what) {
  throw std::runtime_error(path.string() + ":" + std::to_string(line) + ": " + what);
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------
class Sha256 {
 public:
  Sha256() {
    h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  }
  void update(const uint8_t* data, std::size_t len) {
    total_ += len;
    while (len > 0) {
      const std::size_t take = std::min(len, static_cast<std::size_t>(64) - buflen_);
      std::memcpy(buf_ + buflen_, data, take);
      buflen_ += take;
      data += take;
      len -= take;
      if (buflen_ == 64) {
        transform(buf_);
        buflen_ = 0;
      }
    }
  }
  std::string hex_digest() {
    const uint64_t bits = total_ * 8;
    const uint8_t one = 0x80, zero = 0x00;
    update(&one, 1);
    total_ -= 1;
    while (buflen_ != 56) {
      update(&zero, 1);
      total_ -= 1;
    }
    uint8_t lenbe[8];
    for (int i = 0; i < 8; ++i) lenbe[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    update(lenbe, 8);
    static const char* hexd = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t h : h_)
      for (int i = 28; i >= 0; i -= 4) out.push_back(hexd[(h >> i) & 0xF]);
    return out;
  }

 private:
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void transform(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + S1 + ch + K[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }

  std::array<uint32_t, 8> h_{};
  uint8_t buf_[64]{};
  std::size_t buflen_ = 0;
  uint64_t total_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Set text format
// ---------------------------------------------------------------------------
std::vector<Vec> read_set(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("read_set: cannot open " + path.string());
  std::vector<Vec> out;
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    // Strip a comment (first '#' onwards) and trailing CR.
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    const char* p = line.c_str();
    Vec v{};
    int count = 0;
    for (;;) {
      while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
      if (*p == '\0') break;
      if (count == DIM) fail(path, lineno, "more than 24 values");
      char* end = nullptr;
      errno = 0;
      const long val = std::strtol(p, &end, 10);
      if (end == p) fail(path, lineno, std::string("not an integer: '") + p + "'");
      if (*end != '\0' && *end != ' ' && *end != '\t' && *end != '\r')
        fail(path, lineno, std::string("not an integer: '") + p + "'");
      if (errno == ERANGE || val < -128 || val > 127)
        fail(path, lineno, "value " + std::string(p, static_cast<std::size_t>(end - p)) +
                               " outside int8 range");
      v[static_cast<std::size_t>(count++)] = static_cast<int8_t>(val);
      p = end;
    }
    if (count == 0) continue;  // blank or comment-only line
    if (count != DIM)
      fail(path, lineno, "expected 24 values, got " + std::to_string(count));
    out.push_back(v);
  }
  return out;
}

void write_set(const std::filesystem::path& path, const std::vector<Vec>& S,
               const std::string& header_comment) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("write_set: cannot open " + path.string());
  std::string buf;
  if (!header_comment.empty()) {
    std::size_t start = 0;
    while (start <= header_comment.size()) {
      std::size_t nl = header_comment.find('\n', start);
      if (nl == std::string::npos) nl = header_comment.size();
      buf += "# ";
      buf.append(header_comment, start, nl - start);
      buf.push_back('\n');
      start = nl + 1;
    }
  }
  char tmp[8];
  for (const Vec& v : S) {
    for (int k = 0; k < DIM; ++k) {
      const int len = std::snprintf(tmp, sizeof tmp, k ? " %d" : "%d",
                                    static_cast<int>(v[static_cast<std::size_t>(k)]));
      buf.append(tmp, static_cast<std::size_t>(len));
    }
    buf.push_back('\n');
  }
  const bool ok = std::fwrite(buf.data(), 1, buf.size(), f) == buf.size();
  const bool closed = std::fclose(f) == 0;
  if (!ok || !closed) throw std::runtime_error("write_set: write failed for " + path.string());
}

// ---------------------------------------------------------------------------
// Binary helpers and SHA-256 front ends
// ---------------------------------------------------------------------------
std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path.string());
  std::vector<uint8_t> data;
  const std::uintmax_t size = std::filesystem::file_size(path);
  data.resize(static_cast<std::size_t>(size));
  const std::size_t got = size ? std::fread(data.data(), 1, data.size(), f) : 0;
  std::fclose(f);
  if (got != data.size()) throw std::runtime_error("short read on " + path.string());
  return data;
}

void write_binary_file(const std::filesystem::path& path, const void* data, std::size_t bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open " + path.string() + " for writing");
  const bool ok = std::fwrite(data, 1, bytes, f) == bytes;
  const bool closed = std::fclose(f) == 0;
  if (!ok || !closed) throw std::runtime_error("write failed for " + path.string());
}

std::string sha256_hex(const void* data, std::size_t bytes) {
  Sha256 h;
  h.update(static_cast<const uint8_t*>(data), bytes);
  return h.hex_digest();
}

std::string sha256_file(const std::filesystem::path& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("sha256_file: cannot open " + path.string());
  Sha256 h;
  std::vector<uint8_t> buf(1u << 20);
  for (;;) {
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    if (got) h.update(buf.data(), got);
    if (got < buf.size()) break;
  }
  const bool err = std::ferror(f) != 0;
  std::fclose(f);
  if (err) throw std::runtime_error("sha256_file: read error on " + path.string());
  return h.hex_digest();
}

}  // namespace kiss

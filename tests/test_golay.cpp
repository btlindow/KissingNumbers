// T1.1 — acceptance tests for the C++ extended Golay code (include/kiss/golay.h).
//
// Checks (all exhaustive unless stated):
//   1. 4096 words, sorted ascending, distinct, all < 2^24.
//   2. weight distribution == {0:1, 8:759, 12:2576, 16:759, 24:1}.
//   3. parity: bit 23 == parity of bits 0..22 for every word.
//   4. linearity: every XOR of two codewords is a codeword (all 4096^2 ordered
//      pairs via golay_is_codeword), and the GF(2)-span of the 12 basis words
//      x^k·g(x) (k=0..11, parity-extended) is exactly the codeword set.
//   5. zero and all-ones present.
//   6. minimum distance 8 — both as min nonzero weight and as a direct check of
//      all C(4096,2) pairwise Hamming distances.
//   7. octads: 759, weight 8, sorted; every pair of octads meets in 0, 2, 4 or 8
//      coordinates (all 759^2 pairs).
//   8. golay_is_codeword agrees with the sorted list on every 24-bit mask and
//      rejects masks with bits above 23.
//   9. Cross-check against the independent Python implementation: SHA-256 of
//      the sorted masks written as 4096 lines of 6 lowercase hex digits
//      (LF-terminated) must equal the fingerprint from docs/reports/T1.1-python.md.
//
// Prints `RESULT ok=1 ...` on success (docs/design.md §2.4); exit code 0 iff all pass.
#include "kiss/golay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      ++g_failures;                                                 \
      std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                     \
      std::printf("\n");                                            \
    }                                                               \
  } while (0)

// ---------------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4), enough to fingerprint a ~28 KB buffer.
// ---------------------------------------------------------------------------
class Sha256 {
 public:
  Sha256() { reset(); }

  void update(const uint8_t* data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
      buf_[buflen_++] = data[i];
      if (buflen_ == 64) {
        transform(buf_);
        buflen_ = 0;
      }
      ++total_;
    }
  }

  std::string hex_digest() {
    const uint64_t bits = total_ * 8;
    const uint8_t one = 0x80, zero = 0x00;
    update(&one, 1);
    while (buflen_ != 56) update(&zero, 1);
    uint8_t lenbe[8];
    for (int i = 0; i < 8; ++i) lenbe[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    update(lenbe, 8);
    static const char* hexd = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t h : h_) {
      for (int i = 28; i >= 0; i -= 4) out.push_back(hexd[(h >> i) & 0xF]);
    }
    return out;
  }

 private:
  void reset() {
    h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    buflen_ = 0;
    total_ = 0;
  }
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
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }

  std::array<uint32_t, 8> h_;
  uint8_t buf_[64];
  std::size_t buflen_;
  uint64_t total_;
};

std::string sha256_hex(const std::string& s) {
  Sha256 h;
  h.update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  return h.hex_digest();
}

inline int popc(uint32_t x) { return __builtin_popcount(x); }

// Independent GF(2) carry-less multiply for the basis-span check (does not
// call anything in src/golay.cpp beyond the public constant).
uint32_t clmul_ref(uint32_t a, uint32_t b) {
  uint32_t r = 0;
  for (int i = 0; i < 32; ++i)
    if ((b >> i) & 1u) r ^= a << i;
  return r;
}

}  // namespace

int main() {
  using clock = std::chrono::steady_clock;

  // Known-answer test for the embedded SHA-256 so a wrong hash below can be
  // attributed to the code, not to the hasher.
  CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256('') wrong: %s", sha256_hex("").c_str());
  CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "sha256('abc') wrong: %s", sha256_hex("abc").c_str());

  auto t0 = clock::now();
  const std::vector<uint32_t> words = kiss::golay_codewords();
  auto t1 = clock::now();
  const double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // 1. count, sorted, distinct, range
  CHECK(words.size() == 4096, "got %zu words", words.size());
  CHECK(std::is_sorted(words.begin(), words.end()), "not sorted");
  CHECK(std::adjacent_find(words.begin(), words.end()) == words.end(), "duplicate word");
  for (uint32_t w : words) CHECK(w < (1u << 24), "word %06x exceeds 24 bits", w);

  // 2. weight distribution
  const std::array<int, 25> dist = kiss::golay_weight_distribution(words);
  std::array<int, 25> expect{};
  expect[0] = 1; expect[8] = 759; expect[12] = 2576; expect[16] = 759; expect[24] = 1;
  bool dist_ok = (dist == expect);
  CHECK(dist_ok, "weight distribution mismatch");
  std::printf("weight distribution :");
  for (int k = 0; k <= 24; ++k)
    if (dist[k]) std::printf(" %d:%d", k, dist[k]);
  std::printf("\n");

  // 3. parity bit
  for (uint32_t w : words)
    CHECK(((w >> 23) & 1u) == static_cast<uint32_t>(popc(w & 0x7FFFFFu) & 1),
          "parity bit wrong in %06x", w);

  // 5. zero and all-ones
  CHECK(std::binary_search(words.begin(), words.end(), 0u), "zero word missing");
  CHECK(std::binary_search(words.begin(), words.end(), 0xFFFFFFu), "all-ones missing");

  // 8. membership table vs sorted list, exhaustively over all 2^24 masks
  {
    std::size_t idx = 0;
    long mismatches = 0;
    for (uint32_t m = 0; m < (1u << 24); ++m) {
      bool in_list = (idx < words.size() && words[idx] == m);
      if (in_list) ++idx;
      if (kiss::golay_is_codeword(m) != in_list) ++mismatches;
    }
    CHECK(idx == words.size(), "membership sweep consumed %zu of %zu words", idx, words.size());
    CHECK(mismatches == 0, "%ld membership mismatches", mismatches);
    CHECK(!kiss::golay_is_codeword(1u << 24), "bit 24 accepted");
    CHECK(!kiss::golay_is_codeword(0xFFFFFFFFu), "0xffffffff accepted");
  }

  // 4a. XOR closure over all ordered pairs (16.7M lookups)
  {
    long bad = 0;
    for (uint32_t a : words)
      for (uint32_t b : words)
        if (!kiss::golay_is_codeword(a ^ b)) ++bad;
    CHECK(bad == 0, "%ld XOR pairs leave the code", bad);
  }

  // 4b. span of the 12 basis words x^k g(x) (parity-extended) == the code
  {
    std::array<uint32_t, 12> basis{};
    for (int k = 0; k < 12; ++k) {
      uint32_t c = clmul_ref(1u << k, kiss::GOLAY_GENERATOR);
      if (popc(c) & 1) c |= 1u << 23;
      basis[static_cast<std::size_t>(k)] = c;
      CHECK(kiss::golay_is_codeword(c), "basis word x^%d g(x) = %06x not in code", k, c);
    }
    std::vector<uint32_t> span;
    span.reserve(4096);
    for (uint32_t m = 0; m < 4096; ++m) {
      uint32_t c = 0;
      for (int k = 0; k < 12; ++k)
        if ((m >> k) & 1u) c ^= basis[static_cast<std::size_t>(k)];
      span.push_back(c);
    }
    std::sort(span.begin(), span.end());
    CHECK(std::adjacent_find(span.begin(), span.end()) == span.end(),
          "basis words are linearly dependent");
    CHECK(span == words, "span of basis != codeword list");
  }

  // 6. minimum distance 8: min nonzero weight, and all C(4096,2) pairs
  {
    int min_w = 25;
    for (uint32_t w : words)
      if (w && popc(w) < min_w) min_w = popc(w);
    CHECK(min_w == 8, "min nonzero weight %d", min_w);
    int min_d = 25;
    for (std::size_t i = 0; i < words.size(); ++i)
      for (std::size_t j = i + 1; j < words.size(); ++j) {
        int d = popc(words[i] ^ words[j]);
        if (d < min_d) min_d = d;
      }
    CHECK(min_d == 8, "min pairwise distance %d", min_d);
  }

  // 7. octads
  const std::vector<uint32_t> oct = kiss::golay_octads();
  CHECK(oct.size() == 759, "got %zu octads", oct.size());
  CHECK(std::is_sorted(oct.begin(), oct.end()), "octads not sorted");
  for (uint32_t o : oct) {
    CHECK(popc(o) == 8, "octad %06x has weight %d", o, popc(o));
    CHECK(std::binary_search(words.begin(), words.end(), o), "octad %06x not a codeword", o);
    CHECK(kiss::golay_is_codeword(o ^ 0xFFFFFFu), "complement of octad %06x not a codeword", o);
  }
  {
    std::array<long, 9> inter{};  // histogram of |A ∩ B| over all ordered pairs
    long bad = 0;
    for (uint32_t a : oct)
      for (uint32_t b : oct) {
        int k = popc(a & b);
        ++inter[static_cast<std::size_t>(k)];
        if (k != 0 && k != 2 && k != 4 && k != 8) ++bad;
      }
    CHECK(bad == 0, "%ld octad pairs with intersection not in {0,2,4,8}", bad);
    CHECK(inter[8] == 759, "diagonal count %ld", inter[8]);
    std::printf("octad intersections : 0:%ld 2:%ld 4:%ld 8:%ld (ordered pairs)\n",
                inter[0], inter[2], inter[4], inter[8]);
    // Steiner S(5,8,24) counts per octad: 30 disjoint, 448 meet in 2, 280 meet in 4.
    CHECK(inter[0] == 759L * 30 && inter[2] == 759L * 448 && inter[4] == 759L * 280,
          "octad intersection histogram off");
  }

  // 9. fingerprint vs the independent Python implementation
  const char* kExpectedSha =
      "bf7ccc59243c42c70adbc220c5c49d4b55ec66c94e155ad4a9369ed12dfea610";
  std::string text;
  text.reserve(4096 * 7);
  char line[8];
  for (uint32_t w : words) {
    std::snprintf(line, sizeof line, "%06x\n", w);
    text += line;
  }
  const std::string sha = sha256_hex(text);
  CHECK(sha == kExpectedSha, "got %s", sha.c_str());
  std::printf("first masks         : %06x %06x %06x %06x\n", words[0], words[1], words[2], words[3]);
  std::printf("last masks          : %06x %06x\n", words[4094], words[4095]);
  std::printf("sha256(sorted list) : %s (%s)\n", sha.c_str(),
              sha == kExpectedSha ? "matches python" : "MISMATCH");

  // Second call must return the same (cached) list.
  CHECK(kiss::golay_codewords() == words, "cached call differs");

  const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::printf("RESULT ok=%d words=%zu octads=%zu w8=%d w12=%d w16=%d min_d=8 sha_match=%d "
              "gen_ms=%.3f total_ms=%.0f failures=%d\n",
              g_failures == 0 ? 1 : 0, words.size(), oct.size(), dist[8], dist[12], dist[16],
              sha == kExpectedSha ? 1 : 0, gen_ms, total_ms, g_failures);
  return g_failures == 0 ? 0 : 1;
}

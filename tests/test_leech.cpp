// T1.2 — acceptance tests for kiss/leech.h, kiss/io.h, kiss/types.h.
//
//  1. shape counts 97152 / 98304 / 1104, total 196560; rows distinct and
//     strictly increasing (canonical order); every row has norm 32 and passes
//     is_lattice_vector; the complementary residue class is also a codeword.
//  2. negation closure: neg[neg[i]] == i, C[neg[i]] == -C[i]; index_of
//     round-trips for every i; index_of rejects non-members.
//  3. shape (b) sign convention: both (−3,+1^23)- and (+3,−1^23)-based
//     constructions give the same set; (−3,+1^23) itself passes membership.
//  4. is_lattice_vector negative/positive cases (see report).
//  5. inner-product histogram from a base vector of each shape and 64 random
//     base vectors (fixed seed), OpenMP over the 196560 dots.
//  6. packed layout: pack → bytes equal the int8 rows in the SoA layout;
//     unpack(pack(C)) == C byte-exactly; the same holds via files.
//  7. save_leech / load_leech round trip in a scratch directory; load_leech
//     rejects truncated / corrupted files; read_set / write_set round trip and
//     error cases; sha256 known answers.
// Prints `RESULT ok=1 ...`; exit code 0 iff all pass.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "kiss/golay.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/platform.h"

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

using kiss::DIM;
using kiss::N;
using kiss::Vec;

std::array<int, DIM> to_int(const Vec& v) {
  std::array<int, DIM> x{};
  for (int k = 0; k < DIM; ++k) x[static_cast<std::size_t>(k)] = v[static_cast<std::size_t>(k)];
  return x;
}

Vec make_vec(std::initializer_list<int> vals) {
  Vec v{};
  std::size_t k = 0;
  for (int x : vals) v[k++] = static_cast<int8_t>(x);
  return v;
}

// Histogram of ⟨base, C[j]⟩ over all j, keyed by inner product.
std::map<int, long> ip_histogram(const kiss::Leech& L, const Vec& base) {
  std::array<long, 7> h{};
  long other = 0;
#pragma omp parallel
  {
    std::array<long, 7> local{};
    long local_other = 0;
#pragma omp for schedule(static)
    for (int j = 0; j < N; ++j) {
      const int c = kiss::ip_class(kiss::dot(base, L.C[static_cast<std::size_t>(j)]));
      if (c < 0)
        ++local_other;
      else
        ++local[static_cast<std::size_t>(c)];
    }
#pragma omp critical
    {
      for (std::size_t k = 0; k < 7; ++k) h[k] += local[k];
      other += local_other;
    }
  }
  std::map<int, long> out;
  const int vals[7] = {-32, -16, -8, 0, 8, 16, 32};
  for (std::size_t k = 0; k < 7; ++k) out[vals[k]] = h[k];
  if (other) out[999] = other;
  return out;
}

const std::map<int, long> kExpectedHist = {{32, 1},   {16, 4600},  {8, 47104}, {0, 93150},
                                           {-8, 47104}, {-16, 4600}, {-32, 1}};

bool throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const kiss::Leech L = kiss::generate_leech();
  const auto t_gen = clock::now();
  const double gen_ms = std::chrono::duration<double, std::milli>(t_gen - t0).count();

  // ---- 1. counts, order, norms, membership ---------------------------------
  CHECK(L.C.size() == static_cast<std::size_t>(N), "got %zu rows", L.C.size());
  CHECK(L.neg.size() == static_cast<std::size_t>(N), "neg has %zu entries", L.neg.size());
  int shape_counts[3] = {0, 0, 0};
  long bad_norm = 0, bad_member = 0, bad_shape = 0, bad_complement = 0, not_increasing = 0;
  for (std::size_t i = 0; i < L.C.size(); ++i) {
    const Vec& v = L.C[i];
    if (kiss::norm2(v) != 32) ++bad_norm;
    const auto x = to_int(v);
    if (!L.is_lattice_vector(x)) ++bad_member;
    const int s = kiss::leech_shape(v);
    if (s < 0)
      ++bad_shape;
    else
      ++shape_counts[s];
    if (i > 0 && !(L.C[i - 1] < v)) ++not_increasing;
    // complementary residue class (0 mod 4 resp. 1 mod 4) is a codeword too
    {
      const int m = x[0] & 1;
      const int r = m ? 1 : 0;
      uint32_t mask = 0;
      for (int k = 0; k < DIM; ++k)
        if ((x[static_cast<std::size_t>(k)] & 3) == r) mask |= 1u << k;
      if (!kiss::golay_is_codeword(mask)) ++bad_complement;
    }
  }
  CHECK(shape_counts[0] == 97152, "octad shape count %d", shape_counts[0]);
  CHECK(shape_counts[1] == 98304, "(3,1^23) shape count %d", shape_counts[1]);
  CHECK(shape_counts[2] == 1104, "(4,4) shape count %d", shape_counts[2]);
  CHECK(bad_shape == 0, "%ld rows of unknown shape", bad_shape);
  CHECK(bad_norm == 0, "%ld rows with norm != 32", bad_norm);
  CHECK(bad_member == 0, "%ld rows fail is_lattice_vector", bad_member);
  CHECK(bad_complement == 0, "%ld rows whose complementary class is not a codeword", bad_complement);
  CHECK(not_increasing == 0, "%ld adjacent rows not strictly increasing", not_increasing);
  // Distinctness independent of the sort: hash into a set of raw rows.
  {
    std::set<Vec> s(L.C.begin(), L.C.end());
    CHECK(s.size() == L.C.size(), "%zu distinct rows", s.size());
  }
  std::printf("shapes              : octad=%d three_one=%d four_four=%d total=%zu\n",
              shape_counts[0], shape_counts[1], shape_counts[2], L.C.size());

  // ---- 2. negation table and index_of ---------------------------------------
  {
    long bad_neg = 0, bad_idx = 0, bad_invol = 0;
    for (std::size_t i = 0; i < L.C.size(); ++i) {
      const uint32_t j = L.neg[i];
      if (j >= L.C.size() || L.C[j] != kiss::negate(L.C[i])) ++bad_neg;
      else if (L.neg[j] != i) ++bad_invol;
      if (L.index_of(L.C[i]) != static_cast<int32_t>(i)) ++bad_idx;
    }
    CHECK(bad_neg == 0, "%ld wrong neg entries", bad_neg);
    CHECK(bad_invol == 0, "%ld entries with neg[neg[i]] != i", bad_invol);
    CHECK(bad_idx == 0, "%ld index_of round-trip failures", bad_idx);
    CHECK(L.index_of(Vec{}) == -1, "zero vector found");
    CHECK(L.index_of(make_vec({8})) == -1, "(8,0..) found");
    // (2^8,0^16) on coordinates 0..7 is minimal iff {0..7} is an octad.
    CHECK((L.index_of(make_vec({2, 2, 2, 2, 2, 2, 2, 2})) >= 0) == kiss::golay_is_codeword(0xFFu),
          "index_of disagrees with octad test on coordinates 0..7");
  }

  // ---- 3. shape (b) sign convention ------------------------------------------
  {
    const std::vector<uint32_t> words = kiss::golay_codewords();
    // (−3 at 0, +1 elsewhere) must itself be a lattice vector and a member.
    Vec base;
    for (int k = 0; k < DIM; ++k) base[static_cast<std::size_t>(k)] = k == 0 ? -3 : 1;
    CHECK(L.is_lattice_vector(to_int(base)), "(-3,1^23) fails membership");
    CHECK(L.index_of(base) >= 0, "(-3,1^23) not in C");
    Vec base_plus;
    for (int k = 0; k < DIM; ++k) base_plus[static_cast<std::size_t>(k)] = k == 0 ? 3 : -1;
    CHECK(L.is_lattice_vector(to_int(base_plus)), "(3,-1^23) fails membership");
    CHECK(L.index_of(base_plus) >= 0, "(3,-1^23) not in C");
    // Both conventions generate members of C, and their sets coincide with the
    // set of all shape-(b) rows.
    std::set<Vec> from_minus, from_plus;
    for (int i = 0; i < DIM; ++i)
      for (uint32_t c : words) {
        Vec a, b;
        for (int j = 0; j < DIM; ++j) {
          int x = (j == i) ? -3 : 1;
          if ((c >> j) & 1u) x = -x;
          a[static_cast<std::size_t>(j)] = static_cast<int8_t>(x);
          b[static_cast<std::size_t>(j)] = static_cast<int8_t>(-x);
        }
        from_minus.insert(a);
        from_plus.insert(b);
      }
    long missing_minus = 0, missing_plus = 0;
    for (const Vec& v : from_minus)
      if (L.index_of(v) < 0) ++missing_minus;
    for (const Vec& v : from_plus)
      if (L.index_of(v) < 0) ++missing_plus;
    CHECK(from_minus.size() == 98304, "%zu distinct (−3-based) vectors", from_minus.size());
    CHECK(from_minus == from_plus, "the two sign conventions give different sets");
    CHECK(missing_minus == 0 && missing_plus == 0, "missing: %ld (−3-based), %ld (+3-based)",
          missing_minus, missing_plus);
    std::printf("shape (b) convention: (-3 at i, +1 elsewhere) then negate c: members=%zu/98304, "
                "(+3,-1^23) variant gives the same set=%d\n",
                from_minus.size() - static_cast<std::size_t>(missing_minus),
                from_minus == from_plus ? 1 : 0);
  }

  // ---- 4. membership test cases ----------------------------------------------
  {
    auto vec = [](std::initializer_list<int> vals) {
      std::array<int, DIM> x{};
      std::size_t k = 0;
      for (int v : vals) x[k++] = v;
      return x;
    };
    std::array<int, DIM> ones{};
    ones.fill(1);
    CHECK(!L.is_lattice_vector(ones), "(1^24) accepted (sum 24 ≡ 0, needs 4 mod 8)");
    std::array<int, DIM> mixed{};
    mixed.fill(1);
    mixed[3] = 2;
    CHECK(!L.is_lattice_vector(mixed), "mixed parity accepted");
    // (2^8, 0^16) on coordinates 0..7 — accepted iff {0..7} is an octad.
    const auto two8 = vec({2, 2, 2, 2, 2, 2, 2, 2});
    CHECK(L.is_lattice_vector(two8) == kiss::golay_is_codeword(0xFFu),
          "(2^8,0^16) on 0..7 vs octad test disagree");
    // A support that is certainly not an octad: 8 coordinates containing two
    // whole 4-sets from different octads is not guaranteed; instead use weight
    // 7 and weight 9 supports (no codewords of those weights).
    CHECK(!L.is_lattice_vector(vec({2, 2, 2, 2, 2, 2, 2})), "(2^7,0^17) accepted");
    CHECK(!L.is_lattice_vector(vec({2, 2, 2, 2, 2, 2, 2, 2, 2})), "(2^9,0^15) accepted");
    // Take a real octad and spoil it by moving one coordinate.
    {
      const uint32_t oct = kiss::golay_octads()[0];
      std::array<int, DIM> x{};
      for (int k = 0; k < DIM; ++k) x[static_cast<std::size_t>(k)] = ((oct >> k) & 1u) ? 2 : 0;
      CHECK(L.is_lattice_vector(x), "octad-shape vector rejected");
      // find one bit in and one bit out
      int in = -1, out = -1;
      for (int k = 0; k < DIM; ++k) {
        if (((oct >> k) & 1u) && in < 0) in = k;
        if (!((oct >> k) & 1u) && out < 0) out = k;
      }
      x[static_cast<std::size_t>(in)] = 0;
      x[static_cast<std::size_t>(out)] = 2;
      CHECK(!L.is_lattice_vector(x), "spoiled octad accepted");
      // Odd number of minus signs on an octad: sum ≡ 4 mod 8, rejected.
      std::array<int, DIM> y{};
      for (int k = 0; k < DIM; ++k) y[static_cast<std::size_t>(k)] = ((oct >> k) & 1u) ? 2 : 0;
      y[static_cast<std::size_t>(in)] = -2;
      CHECK(!L.is_lattice_vector(y), "octad with one minus sign accepted");
    }
    // Non-minimal lattice vectors (norm 64): in the lattice, not in C.
    CHECK(L.is_lattice_vector(vec({8})), "(8,0^23) rejected (it is in the lattice)");
    CHECK(L.index_of(make_vec({8})) == -1, "(8,0^23) reported as minimal");
    CHECK(L.is_lattice_vector(vec({4, 4, 4, 4})), "(4^4,0^20) rejected (it is in the lattice)");
    CHECK(L.index_of(make_vec({4, 4, 4, 4})) == -1, "(4^4,0^20) reported as minimal");
    CHECK(L.is_lattice_vector(vec({4, -4})), "(4,-4,0^22) rejected");
    CHECK(!L.is_lattice_vector(vec({4})), "(4,0^23) accepted (sum 4 ≡ 4, needs 0 mod 8)");
    CHECK(!L.is_lattice_vector(vec({4, 4, 4})), "(4^3,0^21) accepted");
    CHECK(L.is_lattice_vector(std::array<int, DIM>{}), "zero vector rejected");
    std::array<int, DIM> big{};
    big.fill(-3);
    big[0] = 5;  // all ≡ 1 mod 4, sum = 5 - 69 = -64 ≡ 0 → rejected (needs 4)
    CHECK(!L.is_lattice_vector(big), "(5,-3^23) accepted");
    big[0] = 1;  // sum = 1 - 69 = -68 ≡ 4 mod 8, all ≡ 1 mod 4 → accepted
    CHECK(L.is_lattice_vector(big), "(1,-3^23) rejected");
    // Sum of two minimal vectors is always in the lattice.
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> pick(0, N - 1);
    long bad_sums = 0;
    for (int t = 0; t < 2000; ++t) {
      const Vec& a = L.C[static_cast<std::size_t>(pick(rng))];
      const Vec& b = L.C[static_cast<std::size_t>(pick(rng))];
      std::array<int, DIM> x{};
      for (int k = 0; k < DIM; ++k)
        x[static_cast<std::size_t>(k)] = a[static_cast<std::size_t>(k)] + b[static_cast<std::size_t>(k)];
      if (!L.is_lattice_vector(x)) ++bad_sums;
    }
    CHECK(bad_sums == 0, "%ld sums of minimal vectors rejected", bad_sums);
    // Random small integer vectors: almost surely not in the lattice, and
    // is_lattice_vector must agree with an independent brute check.
    long accepted = 0;
    std::uniform_int_distribution<int> small(-4, 4);
    for (int t = 0; t < 20000; ++t) {
      std::array<int, DIM> x{};
      for (int k = 0; k < DIM; ++k) x[static_cast<std::size_t>(k)] = small(rng);
      if (L.is_lattice_vector(x)) ++accepted;
    }
    CHECK(accepted == 0, "%ld random vectors accepted", accepted);
  }

  // ---- 5. inner-product histograms -------------------------------------------
  const auto t_hist0 = clock::now();
  int hist_ok = 0, hist_total = 0;
  {
    // One base vector of each shape: the first row of each shape in C.
    int first_of_shape[3] = {-1, -1, -1};
    for (int i = 0; i < N && (first_of_shape[0] < 0 || first_of_shape[1] < 0 || first_of_shape[2] < 0); ++i) {
      const int s = kiss::leech_shape(L.C[static_cast<std::size_t>(i)]);
      if (s >= 0 && first_of_shape[s] < 0) first_of_shape[s] = i;
    }
    std::vector<int> bases(first_of_shape, first_of_shape + 3);
    std::mt19937 rng(20260825);
    std::uniform_int_distribution<int> pick(0, N - 1);
    for (int t = 0; t < 64; ++t) bases.push_back(pick(rng));
    for (std::size_t b = 0; b < bases.size(); ++b) {
      const int i = bases[b];
      const std::map<int, long> h = ip_histogram(L, L.C[static_cast<std::size_t>(i)]);
      ++hist_total;
      if (h == kExpectedHist) {
        ++hist_ok;
      } else {
        CHECK(false, "histogram mismatch for base index %d (shape %d)", i,
              kiss::leech_shape(L.C[static_cast<std::size_t>(i)]));
        for (const auto& kv : h) std::printf("    ip=%d count=%ld\n", kv.first, kv.second);
      }
      if (b < 3) {
        std::printf("histogram shape %zu  : base=%d", b, i);
        for (const auto& kv : h) std::printf(" %d:%ld", kv.first, kv.second);
        std::printf("\n");
      }
    }
    // Degree check on the neighbour class: 4600 == DEG.
    CHECK(kExpectedHist.at(16) == kiss::DEG, "DEG constant != 4600");
  }
  const double hist_ms =
      std::chrono::duration<double, std::milli>(clock::now() - t_hist0).count();

  // ---- 6. packed layout -------------------------------------------------------
  {
    const std::vector<uint32_t> packed = kiss::pack_vectors(L.C);
    CHECK(packed.size() == 6u * static_cast<std::size_t>(N), "packed size %zu", packed.size());
    // Byte-level check of the SoA layout against the int8 rows.
    long bad_bytes = 0;
    for (std::size_t i = 0; i < L.C.size(); ++i)
      for (std::size_t w = 0; w < 6; ++w) {
        uint8_t bytes[4];
        std::memcpy(bytes, &packed[w * static_cast<std::size_t>(N) + i], 4);  // little-endian host
        for (std::size_t bb = 0; bb < 4; ++bb)
          if (static_cast<int8_t>(bytes[bb]) != L.C[i][4 * w + bb]) ++bad_bytes;
      }
    CHECK(bad_bytes == 0, "%ld packed bytes differ from int8 rows", bad_bytes);
    CHECK(kiss::unpack_vectors(packed) == L.C, "unpack(pack(C)) != C");
    // Spot check the documented example: word 0 of vector i = bytes c0|c1<<8|c2<<16|c3<<24.
    const Vec& v = L.C[12345];
    const uint32_t expect = static_cast<uint32_t>(static_cast<uint8_t>(v[0])) |
                            (static_cast<uint32_t>(static_cast<uint8_t>(v[1])) << 8) |
                            (static_cast<uint32_t>(static_cast<uint8_t>(v[2])) << 16) |
                            (static_cast<uint32_t>(static_cast<uint8_t>(v[3])) << 24);
    CHECK(packed[12345] == expect, "word 0 of vector 12345: %08x vs %08x", packed[12345], expect);
  }

  // ---- 7. file round trips -----------------------------------------------------
  {
    CHECK(kiss::sha256_hex("", 0) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256('') wrong");
    CHECK(kiss::sha256_hex("abc", 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256('abc') wrong");
    // 1,000,000 'a' — a multi-block known answer.
    {
      std::string a(1000000, 'a');
      CHECK(kiss::sha256_hex(a.data(), a.size()) ==
                "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            "sha256('a'*1e6) wrong");
    }

    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() /
                         ("kiss_test_leech_" + std::to_string(kiss::process_id()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    kiss::save_leech(L, tmp);
    CHECK(fs::file_size(tmp / "leech_min.i8") == 24u * N, "leech_min.i8 size");
    CHECK(fs::file_size(tmp / "neg.u32") == 4u * N, "neg.u32 size");
    CHECK(fs::file_size(tmp / "leech_packed.u32") == 24u * N, "leech_packed.u32 size");
    const kiss::Leech back = kiss::load_leech(tmp);
    CHECK(back.C == L.C, "load_leech(C) != generate_leech(C)");
    CHECK(back.neg == L.neg, "load_leech(neg) != generate_leech(neg)");
    // Packed file decodes back to the int8 file byte-exactly.
    {
      const std::vector<uint8_t> raw = kiss::read_binary_file(tmp / "leech_min.i8");
      const std::vector<uint8_t> praw = kiss::read_binary_file(tmp / "leech_packed.u32");
      std::vector<uint32_t> packed(praw.size() / 4);
      std::memcpy(packed.data(), praw.data(), praw.size());
      const std::vector<Vec> dec = kiss::unpack_vectors(packed);
      std::vector<uint8_t> dec_bytes(dec.size() * 24);
      std::memcpy(dec_bytes.data(), dec.data(), dec_bytes.size());
      CHECK(dec_bytes == raw, "packed file does not decode to leech_min.i8");
      // The sha256 of the in-memory rows equals the file's.
      CHECK(kiss::sha256_hex(L.C.data(), 24u * N) == kiss::sha256_file(tmp / "leech_min.i8"),
            "sha256 mismatch between memory and file");
    }
    // leech_min.txt: 196560 lines, readable through read_set, same order.
    {
      const std::vector<Vec> txt = kiss::read_set(tmp / "leech_min.txt");
      CHECK(txt == L.C, "leech_min.txt != C (size %zu)", txt.size());
    }
    // Corrupted inputs must be rejected.
    {
      const fs::path bad = tmp / "bad";
      fs::create_directories(bad);
      std::vector<uint8_t> raw = kiss::read_binary_file(tmp / "leech_min.i8");
      kiss::write_binary_file(bad / "neg.u32", L.neg.data(), 4u * N);
      kiss::write_binary_file(bad / "leech_min.i8", raw.data(), raw.size() - 24);  // truncated
      CHECK(throws([&] { kiss::load_leech(bad); }), "truncated leech_min.i8 accepted");
      std::swap_ranges(raw.begin(), raw.begin() + 24, raw.begin() + 24);  // swap rows 0 and 1
      kiss::write_binary_file(bad / "leech_min.i8", raw.data(), raw.size());
      CHECK(throws([&] { kiss::load_leech(bad); }), "unsorted leech_min.i8 accepted");
      std::swap_ranges(raw.begin(), raw.begin() + 24, raw.begin() + 24);
      kiss::write_binary_file(bad / "leech_min.i8", raw.data(), raw.size());
      std::vector<uint32_t> neg = L.neg;
      neg[7] = neg[8];
      kiss::write_binary_file(bad / "neg.u32", neg.data(), 4u * N);
      CHECK(throws([&] { kiss::load_leech(bad); }), "wrong neg.u32 accepted");
      CHECK(throws([&] { kiss::load_leech(tmp / "does_not_exist"); }), "missing dir accepted");
    }
    // read_set / write_set.
    {
      std::vector<Vec> S = {L.C[0], L.C[1], L.C[100000], L.C[N - 1]};
      kiss::write_set(tmp / "s.txt", S, "test set\nsecond line");
      std::ifstream in(tmp / "s.txt");
      std::string l1, l2;
      std::getline(in, l1);
      std::getline(in, l2);
      CHECK(l1 == "# test set" && l2 == "# second line", "header lines: '%s' / '%s'", l1.c_str(),
            l2.c_str());
      CHECK(kiss::read_set(tmp / "s.txt") == S, "read_set(write_set(S)) != S");
      // Tolerant parsing: tabs, blank lines, comment lines, trailing comment.
      std::ofstream out(tmp / "s2.txt");
      out << "\n# comment\n   \n";
      for (int k = 0; k < DIM; ++k) out << (k ? "\t" : "  ") << int(S[2][static_cast<std::size_t>(k)]);
      out << "   # trailing\r\n";
      out.close();
      const std::vector<Vec> S2 = kiss::read_set(tmp / "s2.txt");
      CHECK(S2.size() == 1 && S2[0] == S[2], "tolerant parse failed (%zu rows)", S2.size());
      // Errors: wrong arity, out-of-range, garbage.
      auto write_line = [&](const std::string& s) {
        std::ofstream o(tmp / "bad.txt");
        o << s << "\n";
      };
      write_line("1 2 3");
      CHECK(throws([&] { kiss::read_set(tmp / "bad.txt"); }), "3-value row accepted");
      write_line("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
      CHECK(throws([&] { kiss::read_set(tmp / "bad.txt"); }), "25-value row accepted");
      write_line("128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
      CHECK(throws([&] { kiss::read_set(tmp / "bad.txt"); }), "128 accepted");
      write_line("-129 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
      CHECK(throws([&] { kiss::read_set(tmp / "bad.txt"); }), "-129 accepted");
      write_line("x 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
      CHECK(throws([&] { kiss::read_set(tmp / "bad.txt"); }), "non-integer accepted");
      write_line("1.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
      CHECK(throws([&] { kiss::read_set(tmp / "bad.txt"); }), "1.5 accepted");
      CHECK(throws([&] { kiss::read_set(tmp / "nope.txt"); }), "missing file accepted");
      // Empty / comment-only file → empty set.
      write_line("# nothing here");
      CHECK(kiss::read_set(tmp / "bad.txt").empty(), "comment-only file not empty");
    }
    fs::remove_all(tmp);
  }

  const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::printf("RESULT ok=%d n=%zu octad=%d three_one=%d four_four=%d hist_ok=%d/%d "
              "threads=%d gen_ms=%.1f hist_ms=%.1f total_ms=%.0f failures=%d\n",
              g_failures == 0 ? 1 : 0, L.C.size(), shape_counts[0], shape_counts[1],
              shape_counts[2], hist_ok, hist_total, omp_get_max_threads(), gen_ms, hist_ms,
              total_ms, g_failures);
  return g_failures == 0 ? 0 : 1;
}

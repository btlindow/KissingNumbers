// verify_s — C++ half of the certificate chain (docs/design.md §2.4, README §5).
//
//   verify_s <S.txt> [--data DIR] [--generate]
//
// Loads C from <DIR>/leech_min.i8 + neg.u32 (default DIR = data; falls back to
// generate_leech() with a note if the files are missing, --generate forces
// that), reads the set, runs verify_independent (norm 32, membership,
// distinct, Gram off-diagonal ≤ 8) and, on success, reports whether the set is
// antipodal, its Gram histogram, and its tightness profile (number of free
// vertices = 0 ⇔ S is a maximal independent set).
//
// Last line: `RESULT ok=1 size=<n> antipodal=<0|1> ...` (exit 0) or
// `RESULT ok=0 size=<n> reason="..."` (exit 1).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include <omp.h>

#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/types.h"
#include "kiss/verify.h"

namespace {

void usage() {
  std::fprintf(stderr, "usage: verify_s <S.txt> [--data DIR] [--generate]\n");
}

std::string quote(std::string s) {
  for (char& c : s)
    if (c == '"') c = '\'';
  return "\"" + s + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();

  std::string file, data_dir = "data";
  bool force_generate = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (a == "--generate") {
      force_generate = true;
    } else if (a == "-h" || a == "--help") {
      usage();
      return 1;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage();
      return 1;
    } else if (file.empty()) {
      file = a;
    } else {
      usage();
      return 1;
    }
  }
  if (file.empty()) {
    usage();
    std::printf("RESULT ok=0 size=0 reason=\"usage\"\n");
    return 1;
  }

  kiss::Leech L;
  std::string source;
  if (!force_generate) {
    try {
      L = kiss::load_leech(data_dir);
      source = "load_leech(" + data_dir + ")";
    } catch (const std::exception& e) {
      std::printf("note: load_leech(%s) failed (%s); using generate_leech()\n", data_dir.c_str(),
                  e.what());
    }
  }
  if (L.C.empty()) {
    L = kiss::generate_leech();
    source = "generate_leech()";
  }
  const double load_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

  std::vector<kiss::Vec> S;
  std::vector<long> lines;
  try {
    S = kiss::read_set(file);
    lines = kiss::set_line_numbers(file);
  } catch (const std::exception& e) {
    std::printf("parse error: %s\n", e.what());
    std::printf("RESULT ok=0 size=0 file=%s reason=%s\n", quote(file).c_str(),
                quote(std::string("parse: ") + e.what()).c_str());
    return 1;
  }
  if (lines.size() != S.size()) lines.clear();  // defensive: fall back to row numbers

  const kiss::VerifyResult r = kiss::verify_independent(L, S, lines);
  std::printf("file      : %s\n", file.c_str());
  std::printf("C source  : %s (%.0f ms)\n", source.c_str(), load_ms);
  std::printf("rows      : %zu\n", r.size);
  if (!r.ok) {
    std::printf("FAIL      : %s\n", r.message.c_str());
    std::printf("RESULT ok=0 size=%zu file=%s reason=%s\n", r.size, quote(file).c_str(),
                quote(r.message).c_str());
    return 1;
  }

  const bool antipodal = kiss::is_antipodal(S);
  const std::array<long, 8> gh = kiss::gram_histogram(S);
  const int vals[7] = {-32, -16, -8, 0, 8, 16, 32};
  std::string gram;
  int max_offdiag = -32;
  for (std::size_t c = 0; c < 7; ++c) {
    if (gh[c] == 0) continue;
    if (!gram.empty()) gram += ',';
    gram += std::to_string(vals[c]) + ":" + std::to_string(gh[c]);
    max_offdiag = vals[c];
  }
  if (S.size() < 2) max_offdiag = -32;

  // Tightness profile: free vertices (tight == 0 outside S) and histogram.
  const std::vector<uint32_t> idx = kiss::set_indices(L, S);
  const auto t1 = clock::now();
  const std::vector<uint16_t> tight = kiss::tightness_cpu(L, idx);
  const double tight_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
  std::vector<uint8_t> in_s(L.C.size(), 0);
  for (uint32_t i : idx) in_s[i] = 1;
  std::map<int, long> hist;
  long free_count = 0;
  for (std::size_t v = 0; v < tight.size(); ++v) {
    if (in_s[v]) continue;
    ++hist[tight[v]];
    if (tight[v] == 0) ++free_count;
  }
  std::string th;
  for (const auto& kv : hist) {
    if (!th.empty()) th += ',';
    th += std::to_string(kv.first) + ":" + std::to_string(kv.second);
  }

  std::printf("antipodal : %d\n", antipodal ? 1 : 0);
  std::printf("gram      : %s (unordered pairs by inner product)\n", gram.c_str());
  std::printf("tightness : free=%ld (vertices outside S with no 60-degree neighbour in S) "
              "histogram(outside S)=%s  [%.1f ms, %d threads]\n",
              free_count, th.c_str(), tight_ms, omp_get_max_threads());
  const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::printf("RESULT ok=1 size=%zu antipodal=%d max_offdiag=%d gram=%s free=%ld maximal=%d "
              "tight_hist=%s file=%s ms=%.0f\n",
              r.size, antipodal ? 1 : 0, max_offdiag, gram.c_str(), free_count,
              free_count == 0 ? 1 : 0, th.c_str(), quote(file).c_str(), total_ms);
  return 0;
}

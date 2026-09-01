// gen_leech: write the canonical Leech minimal-vector data files (docs/design.md §2.2).
//
//   gen_leech [data_dir]        (default: data)
//
// Writes leech_min.i8, neg.u32, leech_packed.u32, leech_min.txt and
// checksums.sha256 (sha256sum-compatible, relative names) into data_dir,
// re-loads the binary files and checks they equal the in-memory generator.
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "kiss/io.h"
#include "kiss/leech.h"

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
  try {
    const auto t0 = clock::now();
    const kiss::Leech L = kiss::generate_leech();
    const auto t1 = clock::now();
    kiss::save_leech(L, data_dir);
    const auto t2 = clock::now();

    int shape_counts[3] = {0, 0, 0};
    for (const kiss::Vec& v : L.C) {
      const int s = kiss::leech_shape(v);
      if (s < 0) throw std::runtime_error("vector of unknown shape");
      ++shape_counts[s];
    }

    const char* names[4] = {"leech_min.i8", "neg.u32", "leech_packed.u32", "leech_min.txt"};
    std::string sums;
    std::string sha_i8;
    for (int k = 0; k < 4; ++k) {
      const std::string h = kiss::sha256_file(data_dir / names[k]);
      if (k == 0) sha_i8 = h;
      sums += h + "  " + names[k] + "\n";
      std::printf("%s  %s\n", h.c_str(), names[k]);
    }
    kiss::write_binary_file(data_dir / "checksums.sha256", sums.data(), sums.size());

    const kiss::Leech back = kiss::load_leech(data_dir);
    const bool roundtrip = back.C == L.C && back.neg == L.neg;
    const auto t3 = clock::now();
    const auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("RESULT ok=%d n=%zu shape_octad=%d shape_31=%d shape_44=%d roundtrip=%d "
                "sha256_i8=%s dir=%s gen_ms=%.1f write_ms=%.1f verify_ms=%.1f\n",
                roundtrip ? 1 : 0, L.C.size(), shape_counts[0], shape_counts[1], shape_counts[2],
                roundtrip ? 1 : 0, sha_i8.c_str(), data_dir.string().c_str(), ms(t0, t1),
                ms(t1, t2), ms(t2, t3));
    return roundtrip ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\nRESULT ok=0 dir=%s\n", e.what(), data_dir.string().c_str());
    return 1;
  }
}

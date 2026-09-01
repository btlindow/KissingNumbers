// build_adj: build the conflict-graph adjacency table on the GPU (docs/design.md §2.2, T1.6).
//
//   build_adj [data_dir] [chunk_rows]      (default: data, 4096)
//
// Reads <data_dir>/leech_min.i8 + neg.u32 (falls back to generate_leech() if
// absent), streams <data_dir>/adj.u32 (196560 x 4600 uint32, 3,616,704,000
// bytes) in row chunks, then writes <data_dir>/adj.sha256 in sha256sum format.
// Last line: RESULT ok=1 rows=196560 deg=4600 bytes=... sha256=... build_s=...
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

#include "adjacency_cuda.h"
#include "kiss/adjacency.h"
#include "kiss/io.h"
#include "kiss/leech.h"

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
  const uint32_t chunk_rows = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10))
                                       : kiss::cuda::ADJ_DEFAULT_CHUNK_ROWS;
  const auto out_file = data_dir / "adj.u32";
  const auto sha_file = data_dir / "adj.sha256";
  try {
    if (!kiss::cuda::cuda_available()) {
      std::printf("error: no CUDA device\nRESULT ok=0 dir=%s\n", data_dir.string().c_str());
      return 1;
    }
    const auto t0 = clock::now();
    kiss::Leech L;
    if (std::filesystem::exists(data_dir / "leech_min.i8") && std::filesystem::exists(data_dir / "neg.u32")) {
      L = kiss::load_leech(data_dir);
      std::printf("leech           : loaded from %s\n", data_dir.string().c_str());
    } else {
      L = kiss::generate_leech();
      std::printf("leech           : generated in memory (%s has no leech_min.i8/neg.u32)\n",
                  data_dir.string().c_str());
    }
    std::filesystem::create_directories(data_dir);

    std::size_t fr = 0, to = 0;
    kiss::cuda::device_mem_info(&fr, &to);
    std::printf("device memory   : free %.2f / total %.2f GiB before build\n",
                static_cast<double>(fr) / (1 << 30), static_cast<double>(to) / (1 << 30));

    const kiss::cuda::BuildAdjacencyStats st = kiss::cuda::build_adjacency_file(L, out_file, chunk_rows);
    std::printf("build           : chunk_rows=%u upload_s=%.3f build_s=%.3f (kernel+copy+fwrite)\n",
                st.chunk_rows, st.upload_s, st.build_s);

    const auto t1 = clock::now();
    const std::string sha = kiss::sha256_file(out_file);
    const double sha_s = std::chrono::duration<double>(clock::now() - t1).count();
    const std::string line = sha + "  adj.u32\n";
    kiss::write_binary_file(sha_file, line.data(), line.size());
    std::printf("%s  adj.u32   (sha256 in %.1f s)\n", sha.c_str(), sha_s);

    // Cheap structural sanity: mmap it back and check the file size + first/last rows.
    const kiss::Adjacency adj(out_file);
    bool ok = std::filesystem::file_size(out_file) == kiss::Adjacency::EXPECTED_BYTES;
    for (uint32_t v : {0u, static_cast<uint32_t>(kiss::N - 1)}) {
      const uint32_t* r = adj.row(v);
      for (int k = 0; k < kiss::DEG; ++k) {
        if (r[k] >= static_cast<uint32_t>(kiss::N) || (k > 0 && r[k] <= r[k - 1])) ok = false;
        if (kiss::dot(L.C[v], L.C[r[k]]) != 16) ok = false;
      }
    }
    const double total_s = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("RESULT ok=%d rows=%d deg=%d bytes=%zu sha256=%s build_s=%.3f chunk_rows=%u sha_s=%.1f "
                "total_s=%.1f file=%s\n",
                ok ? 1 : 0, kiss::N, kiss::DEG, st.bytes, sha.c_str(), st.build_s, st.chunk_rows, sha_s,
                total_s, out_file.string().c_str());
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\nRESULT ok=0 dir=%s\n", e.what(), data_dir.string().c_str());
    return 1;
  }
}

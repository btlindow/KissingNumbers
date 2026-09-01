// T1.5 — upload / free of the packed minimal vectors (docs/design.md §2.3).
#include "kiss_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace kiss::cuda {

DeviceLeech upload_leech(const Leech& L) {
  if (L.C.size() != static_cast<std::size_t>(N) || L.neg.size() != static_cast<std::size_t>(N))
    throw std::runtime_error("upload_leech: Leech has wrong size (expected N=" +
                             std::to_string(N) + " rows)");
  const std::vector<uint32_t> packed = pack_vectors(L.C);  // [6][N] SoA
  const std::size_t packed_bytes = packed.size() * sizeof(uint32_t);
  const std::size_t neg_bytes = L.neg.size() * sizeof(uint32_t);

  uint32_t* d_packed = nullptr;
  uint32_t* d_neg = nullptr;
  try {
    KISS_CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
    KISS_CUDA_CHECK(cudaMalloc(&d_neg, neg_bytes));
    KISS_CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice));
    KISS_CUDA_CHECK(cudaMemcpy(d_neg, L.neg.data(), neg_bytes, cudaMemcpyHostToDevice));
  } catch (...) {
    cudaFree(d_packed);
    cudaFree(d_neg);
    throw;
  }
  DeviceLeech d;
  d.packed = d_packed;
  d.neg = d_neg;
  return d;
}

void free_leech(DeviceLeech& d) {
  // cudaFree(nullptr) is a no-op; errors are deliberately ignored (destructor path).
  cudaFree(const_cast<uint32_t*>(d.packed));
  cudaFree(const_cast<uint32_t*>(d.neg));
  d.packed = nullptr;
  d.neg = nullptr;
}

}  // namespace kiss::cuda

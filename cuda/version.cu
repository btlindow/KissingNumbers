// T0.1 placeholder: trivial exported functions so kiss_cuda has an object.
#include "kiss/version.h"

#include <cuda_runtime.h>

namespace kiss::cuda {

int device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

std::string device_name(int dev) {
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return "";
  return p.name;
}

}  // namespace kiss::cuda

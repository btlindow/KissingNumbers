// kiss_info: print library version and the CUDA device name. T0.1 placeholder
// tool that also proves the tools/ → kiss + kiss_cuda link path works.
#include "kiss/version.h"

#include <cstdio>

int main() {
  const int ndev = kiss::cuda::device_count();
  const std::string name = ndev > 0 ? kiss::cuda::device_name(0) : "none";
  std::printf("kiss version : %s\n", kiss::version().c_str());
  std::printf("cuda devices : %d\n", ndev);
  std::printf("device 0     : %s\n", name.c_str());
  std::printf("RESULT ok=1 version=\"%s\" git=%s devices=%d device=\"%s\"\n",
              kiss::version().c_str(), kiss::git_hash().c_str(), ndev, name.c_str());
  return 0;
}

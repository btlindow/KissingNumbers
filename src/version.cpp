#include "kiss/version.h"
#include "kiss/version_info.h"

namespace kiss {

std::string version() {
  return std::string(KISS_VERSION_STRING) + " (" + KISS_GIT_HASH + ", " + KISS_BUILD_TYPE +
         ", cuda " + KISS_CUDA_VERSION + " sm_" + KISS_CUDA_ARCHS + ")";
}

std::string git_hash() { return KISS_GIT_HASH; }

}  // namespace kiss

// T0.1 placeholder header. Real API headers (types.h, golay.h, leech.h, ...)
// are owned by T1.2 — see docs/design.md §2.3.
#pragma once
#include <string>

namespace kiss {

// "0.1.0 (<git hash>, <build type>)"
std::string version();

// Short git hash the binary was built from ("unknown" outside a repo).
std::string git_hash();

namespace cuda {
// Number of CUDA devices visible (0 if none / no driver).
int device_count();
// Name of device `dev`, or "" if unavailable.
std::string device_name(int dev = 0);
}  // namespace cuda

}  // namespace kiss

// T0.1 GPU smoke test: query device 0, run a trivial kernel, check the result.
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) if no CUDA device, 1 on failure.
// Last line: RESULT ok=<0|1> device="..." cc=sm_XX mem_gb=... sms=... (docs/design.md §2.4)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kSkip = 77;

__global__ void saxpy_i32(int n, const int* __restrict__ x, int* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 3 * x[i] + y[i];
}

int fail(const char* what, cudaError_t e) {
  std::fprintf(stderr, "smoke_gpu: %s failed: %s\n", what, cudaGetErrorString(e));
  std::printf("RESULT ok=0 stage=%s\n", what);
  return 1;
}

}  // namespace

int main() {
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || ndev == 0) {
    std::printf("smoke_gpu: no CUDA device (%s) — skipping\n", cudaGetErrorString(e));
    std::printf("RESULT ok=1 skipped=1 devices=0\n");
    return kSkip;
  }
  if (e != cudaSuccess) return fail("cudaGetDeviceCount", e);

  cudaDeviceProp p{};
  if ((e = cudaGetDeviceProperties(&p, 0)) != cudaSuccess) return fail("cudaGetDeviceProperties", e);
  int drv = 0, rt = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rt);

  const double mem_gb = static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
  std::printf("device 0        : %s\n", p.name);
  std::printf("compute cap     : sm_%d%d\n", p.major, p.minor);
  std::printf("total memory    : %.2f GiB (%zu bytes)\n", mem_gb, static_cast<size_t>(p.totalGlobalMem));
  std::printf("SMs             : %d\n", p.multiProcessorCount);
  std::printf("driver/runtime  : %d / %d\n", drv, rt);

  // Trivial kernel: y = 3x + y on n ints, verified on the host.
  const int n = 1 << 20;
  std::vector<int> hx(n), hy(n);
  for (int i = 0; i < n; ++i) { hx[i] = i; hy[i] = 2 * i; }
  int *dx = nullptr, *dy = nullptr;
  if ((e = cudaMalloc(&dx, n * sizeof(int))) != cudaSuccess) return fail("cudaMalloc", e);
  if ((e = cudaMalloc(&dy, n * sizeof(int))) != cudaSuccess) return fail("cudaMalloc", e);
  if ((e = cudaMemcpy(dx, hx.data(), n * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) return fail("cudaMemcpy", e);
  if ((e = cudaMemcpy(dy, hy.data(), n * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) return fail("cudaMemcpy", e);

  saxpy_i32<<<(n + 255) / 256, 256>>>(n, dx, dy);
  if ((e = cudaGetLastError()) != cudaSuccess) return fail("kernel launch", e);
  if ((e = cudaDeviceSynchronize()) != cudaSuccess) return fail("cudaDeviceSynchronize", e);
  if ((e = cudaMemcpy(hy.data(), dy, n * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) return fail("cudaMemcpy", e);
  cudaFree(dx);
  cudaFree(dy);

  long long bad = 0;
  for (int i = 0; i < n; ++i) bad += (hy[i] != 5 * i);
  std::printf("kernel check    : %d elements, %lld mismatches\n", n, bad);

  const bool ok = (bad == 0) && (p.major == 8 && p.minor == 6);
  if (!(p.major == 8 && p.minor == 6))
    std::fprintf(stderr, "smoke_gpu: expected sm_86 (docs/design.md §1), got sm_%d%d\n", p.major, p.minor);

  std::printf("RESULT ok=%d device=\"%s\" cc=sm_%d%d mem_gb=%.2f sms=%d n=%d mismatches=%lld driver=%d runtime=%d\n",
              ok ? 1 : 0, p.name, p.major, p.minor, mem_gb, p.multiProcessorCount, n, bad, drv, rt);
  return ok ? 0 : 1;
}

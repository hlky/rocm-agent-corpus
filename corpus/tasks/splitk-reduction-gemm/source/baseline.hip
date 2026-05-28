#include <hip/hip_runtime.h>

// Template-only scaffold for splitk-reduction-gemm.
// Baseline strategy: non-split GEMM or library-selected cuBLASLt/Composable Kernel
// algorithm with split_k recorded as one.

namespace {

__global__ void splitk_reduction_gemm_baseline_kernel() {
  // TODO: implement or call a simple non-split GEMM baseline for education.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  splitk_reduction_gemm_baseline_kernel<<<1, 1, 0, stream>>>();
}

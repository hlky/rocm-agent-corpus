#include <hip/hip_runtime.h>

// Template-only scaffold for implicit-gemm-convolution.
// Baseline strategy: call cuDNN, CUTLASS implicit GEMM, TensorRT, or framework
// convolution for the same descriptor and timing boundary.

namespace {

__global__ void implicit_gemm_convolution_baseline_kernel() {
  // TODO: replace with library baseline wrapper or reference HIP baseline.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  implicit_gemm_convolution_baseline_kernel<<<1, 1, 0, stream>>>();
}

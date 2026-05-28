#include <hip/hip_runtime.h>

// Template-only scaffold for depthwise-separable-convolution.
// Baseline strategy: run depthwise and optional pointwise stages separately
// through MIOpen, MIGraphX, framework kernels, or simple HIP reference code.

namespace {

__global__ void depthwise_separable_convolution_baseline_kernel() {
  // TODO: implement separate-stage depthwise/separable convolution baseline.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  depthwise_separable_convolution_baseline_kernel<<<1, 1, 0, stream>>>();
}

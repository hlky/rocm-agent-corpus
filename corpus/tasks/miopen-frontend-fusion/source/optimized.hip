#include <hip/hip_runtime.h>

// Template-only custom HIP candidate for miopen-frontend-fusion.
// Replace this placeholder with the fixed-shape fusion path being compared
// against MIOpen frontend, Composable Kernel, and MIGraphX baselines.

namespace {

__global__ void fused_epilogue_placeholder(const float* x,
                                           const float* bias,
                                           float* y,
                                           int elements,
                                           int channels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  float v = x[i] + bias[i % channels];
  y[i] = v > 0.0f ? v : 0.0f;
}

}  // namespace

extern "C" hipError_t launch_optimized(const float* x,
                                        const float* bias,
                                        float* y,
                                        int elements,
                                        int channels,
                                        hipStream_t stream) {
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  fused_epilogue_placeholder<<<grid, block, 0, stream>>>(x, bias, y, elements, channels);
  return hipGetLastError();
}

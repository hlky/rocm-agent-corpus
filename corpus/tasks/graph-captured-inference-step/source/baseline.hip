#include <hip/hip_runtime.h>

// Template-only ordinary-launch inference step.
// Replace these kernels with the fixed inference bucket being studied.

namespace {

__global__ void affine_step(const float* x, const float* scale, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * scale[i] + 0.1f;
}

__global__ void activation_step(float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] > 0.0f ? y[i] : 0.0f;
}

}  // namespace

extern "C" hipError_t launch_baseline(const float* x,
                                       const float* scale,
                                       float* y,
                                       int n,
                                       hipStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  affine_step<<<grid, block, 0, stream>>>(x, scale, y, n);
  activation_step<<<grid, block, 0, stream>>>(y, n);
  return hipGetLastError();
}

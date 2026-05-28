#include <hip/hip_runtime.h>

__global__ void tiny_axpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

extern "C" hipError_t launch_baseline(float* y,
                                       const float* x,
                                       int n,
                                       int iterations,
                                       hipStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  for (int it = 0; it < iterations; ++it) {
    tiny_axpy<<<grid, block, 0, stream>>>(y, x, 1.001f, n);
  }
  return hipGetLastError();
}

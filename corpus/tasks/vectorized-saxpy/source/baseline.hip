#include <hip/hip_runtime.h>

__global__ void saxpy_scalar(float alpha,
                             const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ output,
                             int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = alpha * x[i] + y[i];
  }
}

extern "C" void launch_baseline(float alpha,
                                const float* x,
                                const float* y,
                                float* output,
                                int n,
                                hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  saxpy_scalar<<<grid, kBlockSize, 0, stream>>>(alpha, x, y, output, n);
}


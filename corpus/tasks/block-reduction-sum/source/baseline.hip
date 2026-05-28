#include <hip/hip_runtime.h>

__global__ void sum_atomic_per_element(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(output, input[i]);
  }
}

extern "C" void launch_baseline(const float* input,
                                float* output,
                                int n,
                                hipStream_t stream) {
  constexpr int kBlockSize = 256;
  hipMemsetAsync(output, 0, sizeof(float), stream);
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  sum_atomic_per_element<<<grid, kBlockSize, 0, stream>>>(input, output, n);
}


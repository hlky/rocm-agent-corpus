#include <hip/hip_runtime.h>

__global__ void fused_relu_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int elements,
                                  float alpha) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const float scaled = alpha * x[idx];
  y[idx] = scaled > 0.0f ? scaled : 0.0f;
}

extern "C" void launch_fused_relu(const float* x,
                                   float* y,
                                   int elements,
                                   float alpha,
                                   hipStream_t stream) {
  const int block = 256;
  const int grid = (elements + block - 1) / block;
  fused_relu_kernel<<<grid, block, 0, stream>>>(x, y, elements, alpha);
}

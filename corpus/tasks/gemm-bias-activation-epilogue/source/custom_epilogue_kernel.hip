#include <hip/hip_runtime.h>

namespace {

__device__ float relu(float value) {
  return value > 0.0f ? value : 0.0f;
}

__global__ void gemm_bias_relu_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      const float* __restrict__ bias,
                                      float* __restrict__ d,
                                      int m,
                                      int n,
                                      int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += a[row * k + kk] * b[kk * n + col];
  }

  d[row * n + col] = relu(acc + bias[col]);
}

}  // namespace

extern "C" void launch_custom_epilogue(const float* a,
                                        const float* b,
                                        const float* bias,
                                        float* d,
                                        int m,
                                        int n,
                                        int k,
                                        hipStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  gemm_bias_relu_kernel<<<grid, block, 0, stream>>>(a, b, bias, d, m, n, k);
}

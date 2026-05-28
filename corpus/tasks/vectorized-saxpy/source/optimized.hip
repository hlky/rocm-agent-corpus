#include <hip/hip_runtime.h>

__global__ void saxpy_float4(float alpha,
                             const float4* __restrict__ x,
                             const float4* __restrict__ y,
                             float4* __restrict__ output,
                             int n4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 xv = x[i];
    const float4 yv = y[i];
    output[i] = make_float4(alpha * xv.x + yv.x,
                            alpha * xv.y + yv.y,
                            alpha * xv.z + yv.z,
                            alpha * xv.w + yv.w);
  }
}

__global__ void saxpy_tail(float alpha,
                           const float* __restrict__ x,
                           const float* __restrict__ y,
                           float* __restrict__ output,
                           int start,
                           int n) {
  const int i = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = alpha * x[i] + y[i];
  }
}

extern "C" void launch_optimized(float alpha,
                                 const float* x,
                                 const float* y,
                                 float* output,
                                 int n,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const int n4 = n / 4;
  if (n4 > 0) {
    const int grid4 = (n4 + kBlockSize - 1) / kBlockSize;
    saxpy_float4<<<grid4, kBlockSize, 0, stream>>>(
        alpha,
        reinterpret_cast<const float4*>(x),
        reinterpret_cast<const float4*>(y),
        reinterpret_cast<float4*>(output),
        n4);
  }

  const int tail_start = n4 * 4;
  if (tail_start < n) {
    saxpy_tail<<<1, kBlockSize, 0, stream>>>(alpha, x, y, output, tail_start, n);
  }
}


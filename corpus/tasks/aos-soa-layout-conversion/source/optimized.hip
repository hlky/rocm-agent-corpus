#include <hip/hip_runtime.h>

__global__ void aos_to_soa_float4_kernel(const float* __restrict__ aos,
                                         float* __restrict__ x,
                                         float* __restrict__ y,
                                         float* __restrict__ z,
                                         float* __restrict__ w,
                                         int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const float4 value = reinterpret_cast<const float4*>(aos)[i];
  x[i] = value.x;
  y[i] = value.y;
  z[i] = value.z;
  w[i] = value.w;
}

extern "C" void launch_optimized(const float* aos,
                                 float* x,
                                 float* y,
                                 float* z,
                                 float* w,
                                 int n,
                                 hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  aos_to_soa_float4_kernel<<<grid, block, 0, stream>>>(aos, x, y, z, w, n);
}

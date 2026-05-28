#include <hip/hip_runtime.h>

__global__ void aos_to_soa_scalar_kernel(const float* __restrict__ aos,
                                         float* __restrict__ x,
                                         float* __restrict__ y,
                                         float* __restrict__ z,
                                         float* __restrict__ w,
                                         int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const int base = 4 * i;
  x[i] = aos[base + 0];
  y[i] = aos[base + 1];
  z[i] = aos[base + 2];
  w[i] = aos[base + 3];
}

extern "C" void launch_baseline(const float* aos,
                                float* x,
                                float* y,
                                float* z,
                                float* w,
                                int n,
                                hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  aos_to_soa_scalar_kernel<<<grid, block, 0, stream>>>(aos, x, y, z, w, n);
}

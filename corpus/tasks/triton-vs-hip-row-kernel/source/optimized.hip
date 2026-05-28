#include <hip/hip_runtime.h>

__global__ void hip_row_sum(const float* x, float* y, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += x[row * cols + col];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[row] = partial[0];
}

extern "C" void launch_optimized(const float* x, float* y, int rows, int cols, hipStream_t stream) {
  hip_row_sum<<<rows, 256, 0, stream>>>(x, y, rows, cols);
}

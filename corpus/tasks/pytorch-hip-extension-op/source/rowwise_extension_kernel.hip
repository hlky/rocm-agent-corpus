#include <hip/hip_runtime.h>

namespace {

__global__ void rowwise_sum_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int rows,
                                   int cols,
                                   int stride_row,
                                   int stride_col) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += x[row * stride_row + col * stride_col];
  }

  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    y[row] = partial[0];
  }
}

}  // namespace

extern "C" void launch_rowwise_sum_extension(const float* x,
                                              float* y,
                                              int rows,
                                              int cols,
                                              int stride_row,
                                              int stride_col,
                                              hipStream_t stream) {
  rowwise_sum_kernel<<<rows, 256, 0, stream>>>(x, y, rows, cols, stride_row, stride_col);
}

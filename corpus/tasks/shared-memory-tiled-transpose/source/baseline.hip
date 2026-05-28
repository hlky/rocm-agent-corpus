#include <hip/hip_runtime.h>

__global__ void transpose_naive(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows,
                                int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    output[col * rows + row] = input[row * cols + col];
  }
}

extern "C" void launch_baseline(const float* input,
                                float* output,
                                int rows,
                                int cols,
                                hipStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);
  transpose_naive<<<grid, block, 0, stream>>>(input, output, rows, cols);
}


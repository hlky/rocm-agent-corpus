#include <hip/hip_runtime.h>

__global__ void direct_copy(const float* input, float* output, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) output[row * cols + col] = input[row * cols + col];
}

extern "C" void launch_baseline(const float* input, float* output, int rows, int cols, hipStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  direct_copy<<<grid, block, 0, stream>>>(input, output, rows, cols);
}

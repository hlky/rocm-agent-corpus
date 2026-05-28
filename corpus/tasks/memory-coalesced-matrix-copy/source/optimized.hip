#include <hip/hip_runtime.h>

// Optimized: threadIdx.x moves across columns. For row-major storage this makes
// adjacent lanes access adjacent elements in memory.
__global__ void matrix_copy_coalesced_threads(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int rows,
                                              int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    const int offset = row * cols + col;
    output[offset] = input[offset];
  }
}

extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int rows,
                                 int cols,
                                 hipStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);
  matrix_copy_coalesced_threads<<<grid, block, 0, stream>>>(input, output, rows, cols);
}


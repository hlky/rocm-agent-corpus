#include <hip/hip_runtime.h>

// Baseline: threadIdx.x moves across rows. For row-major storage this makes
// adjacent lanes access elements separated by cols, which is usually uncoalesced.
__global__ void matrix_copy_strided_threads(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int rows,
                                            int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    const int offset = row * cols + col;
    output[offset] = input[offset];
  }
}

extern "C" void launch_baseline(const float* input,
                                float* output,
                                int rows,
                                int cols,
                                hipStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x,
            (cols + block.y - 1) / block.y);
  matrix_copy_strided_threads<<<grid, block, 0, stream>>>(input, output, rows, cols);
}


#include <hip/hip_runtime.h>

__global__ void staged_tile_copy(const float* input, float* output, int rows, int cols) {
  __shared__ float tile[8][32];
  int col = blockIdx.x * 32 + threadIdx.x;
  int row = blockIdx.y * 8 + threadIdx.y;
  if (row < rows && col < cols) {
    tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
  }
  __syncthreads();
  // TODO(gfx90a+): replace this load/store staging with global-to-LDS staging commit/wait groups.
  if (row < rows && col < cols) {
    output[row * cols + col] = tile[threadIdx.y][threadIdx.x];
  }
}

extern "C" void launch_optimized(const float* input, float* output, int rows, int cols, hipStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((cols + 31) / 32, (rows + 7) / 8);
  staged_tile_copy<<<grid, block, 0, stream>>>(input, output, rows, cols);
}

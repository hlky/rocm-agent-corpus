#include <hip/hip_runtime.h>

namespace {
constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;
}

__global__ void transpose_tiled(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows,
                                int cols) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (x < cols && y + j < rows) {
      tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * kTileDim + threadIdx.x;
  y = blockIdx.x * kTileDim + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (x < rows && y + j < cols) {
      output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int rows,
                                 int cols,
                                 hipStream_t stream) {
  dim3 block(kTileDim, kBlockRows);
  dim3 grid((cols + kTileDim - 1) / kTileDim,
            (rows + kTileDim - 1) / kTileDim);
  transpose_tiled<<<grid, block, 0, stream>>>(input, output, rows, cols);
}


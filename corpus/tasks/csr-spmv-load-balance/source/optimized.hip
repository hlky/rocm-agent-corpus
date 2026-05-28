#include <hip/hip_runtime.h>

namespace {

constexpr int kBlockSize = 128;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void csr_spmv_one_warp_per_row(const int* __restrict__ row_offsets,
                                          const int* __restrict__ column_indices,
                                          const float* __restrict__ values,
                                          const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int rows) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp_id >= rows) {
    return;
  }

  float local = 0.0f;
  const int begin = row_offsets[warp_id];
  const int end = row_offsets[warp_id + 1];
  for (int offset = begin + lane; offset < end; offset += kWarpSize) {
    local += values[offset] * x[column_indices[offset]];
  }

  const float sum = warp_sum(local);
  if (lane == 0) {
    y[warp_id] = sum;
  }
}

}  // namespace

extern "C" void launch_optimized(const int* row_offsets,
                                 const int* column_indices,
                                 const float* values,
                                 const float* x,
                                 float* y,
                                 int rows,
                                 hipStream_t stream) {
  const int warps_per_block = kBlockSize / kWarpSize;
  const int grid = (rows + warps_per_block - 1) / warps_per_block;
  csr_spmv_one_warp_per_row<<<grid, kBlockSize, 0, stream>>>(
      row_offsets, column_indices, values, x, y, rows);
}

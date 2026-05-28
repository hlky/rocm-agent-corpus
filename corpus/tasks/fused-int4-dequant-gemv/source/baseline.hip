#include <hip/hip_runtime.h>

#include <stddef.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ int unpack_u4(uint8_t packed, int col) {
  return (col & 1) == 0 ? static_cast<int>(packed & 0x0f)
                        : static_cast<int>((packed >> 4) & 0x0f);
}

}  // namespace

__global__ void int4_dequant_gemv_one_thread_per_row(
    const uint8_t* __restrict__ packed_weights,
    const float* __restrict__ x,
    const float* __restrict__ scales,
    const uint8_t* __restrict__ zero_points,
    float* __restrict__ y,
    int rows,
    int cols,
    int group_size) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int packed_cols = (cols + 1) / 2;
  const int groups_per_row = (cols + group_size - 1) / group_size;
  const uint8_t* row_weights = packed_weights + static_cast<size_t>(row) * packed_cols;
  const float* row_scales = scales + static_cast<size_t>(row) * groups_per_row;
  const uint8_t* row_zero_points = zero_points + static_cast<size_t>(row) * groups_per_row;

  float acc = 0.0f;
  for (int col = 0; col < cols; ++col) {
    const uint8_t packed = row_weights[col >> 1];
    const int q = unpack_u4(packed, col);
    const int group = col / group_size;
    const int zp = static_cast<int>(row_zero_points[group] & 0x0f);
    const float weight = static_cast<float>(q - zp) * row_scales[group];
    acc += weight * x[col];
  }

  y[row] = acc;
}

extern "C" void launch_baseline(const uint8_t* packed_weights,
                                const float* x,
                                const float* scales,
                                const uint8_t* zero_points,
                                float* y,
                                int rows,
                                int cols,
                                int group_size,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (rows + kBlockSize - 1) / kBlockSize;
  int4_dequant_gemv_one_thread_per_row<<<grid, kBlockSize, 0, stream>>>(
      packed_weights, x, scales, zero_points, y, rows, cols, group_size);
}

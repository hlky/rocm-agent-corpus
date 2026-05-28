#include <hip/hip_runtime.h>

#include <stddef.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ int low_u4(uint8_t packed) {
  return static_cast<int>(packed & 0x0f);
}

__device__ __forceinline__ int high_u4(uint8_t packed) {
  return static_cast<int>((packed >> 4) & 0x0f);
}

}  // namespace

__global__ void int4_dequant_gemv_one_cta_per_row(
    const uint8_t* __restrict__ packed_weights,
    const float* __restrict__ x,
    const float* __restrict__ scales,
    const uint8_t* __restrict__ zero_points,
    float* __restrict__ y,
    int rows,
    int cols,
    int group_size) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float partials[];
  const int tid = threadIdx.x;
  const int packed_cols = (cols + 1) / 2;
  const int groups_per_row = (cols + group_size - 1) / group_size;
  const uint8_t* row_weights = packed_weights + static_cast<size_t>(row) * packed_cols;
  const float* row_scales = scales + static_cast<size_t>(row) * groups_per_row;
  const uint8_t* row_zero_points = zero_points + static_cast<size_t>(row) * groups_per_row;

  float local = 0.0f;
  for (int packed_col = tid; packed_col < packed_cols; packed_col += blockDim.x) {
    const uint8_t packed = row_weights[packed_col];
    const int col0 = packed_col << 1;
    const int group0 = col0 / group_size;
    const int zp0 = static_cast<int>(row_zero_points[group0] & 0x0f);
    const float scale0 = row_scales[group0];
    const float weight0 = static_cast<float>(low_u4(packed) - zp0) * scale0;
    local += weight0 * x[col0];

    const int col1 = col0 + 1;
    if (col1 < cols) {
      const int group1 = col1 / group_size;
      const int zp1 = static_cast<int>(row_zero_points[group1] & 0x0f);
      const float scale1 = row_scales[group1];
      const float weight1 = static_cast<float>(high_u4(packed) - zp1) * scale1;
      local += weight1 * x[col1];
    }
  }

  partials[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partials[tid] += partials[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[row] = partials[0];
  }
}

extern "C" void launch_optimized(const uint8_t* packed_weights,
                                 const float* x,
                                 const float* scales,
                                 const uint8_t* zero_points,
                                 float* y,
                                 int rows,
                                 int cols,
                                 int group_size,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const size_t shared_bytes = kBlockSize * sizeof(float);
  int4_dequant_gemv_one_cta_per_row<<<rows, kBlockSize, shared_bytes, stream>>>(
      packed_weights, x, scales, zero_points, y, rows, cols, group_size);
}

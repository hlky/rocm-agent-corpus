#include <hip/hip_runtime.h>

namespace {

__global__ void csr_spmv_one_thread_per_row(const int* __restrict__ row_offsets,
                                            const int* __restrict__ column_indices,
                                            const float* __restrict__ values,
                                            const float* __restrict__ x,
                                            float* __restrict__ y,
                                            int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  float sum = 0.0f;
  for (int offset = row_offsets[row]; offset < row_offsets[row + 1]; ++offset) {
    sum += values[offset] * x[column_indices[offset]];
  }
  y[row] = sum;
}

}  // namespace

extern "C" void launch_baseline(const int* row_offsets,
                                const int* column_indices,
                                const float* values,
                                const float* x,
                                float* y,
                                int rows,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (rows + kBlockSize - 1) / kBlockSize;
  csr_spmv_one_thread_per_row<<<grid, kBlockSize, 0, stream>>>(
      row_offsets, column_indices, values, x, y, rows);
}

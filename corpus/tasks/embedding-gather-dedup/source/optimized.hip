#include <hip/hip_runtime.h>

#include <stddef.h>

namespace {

__global__ void gather_unique_rows(const float* __restrict__ table,
                                   const int* __restrict__ unique_ids,
                                   float* __restrict__ unique_rows,
                                   int unique_count,
                                   int dim) {
  const size_t total = static_cast<size_t>(unique_count) * dim;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(linear % dim);
    const int unique_index = static_cast<int>(linear / dim);
    const int row = unique_ids[unique_index];
    unique_rows[linear] = table[static_cast<size_t>(row) * dim + d];
  }
}

__global__ void scatter_inverse_rows(const float* __restrict__ unique_rows,
                                     const int* __restrict__ inverse,
                                     float* __restrict__ output,
                                     int requests,
                                     int dim) {
  const size_t total = static_cast<size_t>(requests) * dim;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(linear % dim);
    const int request = static_cast<int>(linear / dim);
    const int unique_index = inverse[request];
    output[linear] = unique_rows[static_cast<size_t>(unique_index) * dim + d];
  }
}

}  // namespace

extern "C" void launch_optimized(const float* table,
                                 const int* unique_ids,
                                 const int* inverse,
                                 float* unique_rows,
                                 float* output,
                                 int requests,
                                 int unique_count,
                                 int dim,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const size_t unique_total = static_cast<size_t>(unique_count) * dim;
  const size_t request_total = static_cast<size_t>(requests) * dim;
  const int unique_grid = static_cast<int>((unique_total + kBlockSize - 1) / kBlockSize);
  const int scatter_grid = static_cast<int>((request_total + kBlockSize - 1) / kBlockSize);
  gather_unique_rows<<<unique_grid, kBlockSize, 0, stream>>>(
      table, unique_ids, unique_rows, unique_count, dim);
  scatter_inverse_rows<<<scatter_grid, kBlockSize, 0, stream>>>(
      unique_rows, inverse, output, requests, dim);
}

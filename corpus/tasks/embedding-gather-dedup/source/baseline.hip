#include <hip/hip_runtime.h>

#include <stddef.h>

namespace {

__global__ void embedding_direct_gather(const float* __restrict__ table,
                                        const int* __restrict__ ids,
                                        float* __restrict__ output,
                                        int requests,
                                        int dim) {
  const size_t total = static_cast<size_t>(requests) * dim;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(linear % dim);
    const int request = static_cast<int>(linear / dim);
    const int row = ids[request];
    output[linear] = table[static_cast<size_t>(row) * dim + d];
  }
}

}  // namespace

extern "C" void launch_baseline(const float* table,
                                const int* ids,
                                float* output,
                                int requests,
                                int dim,
                                hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const size_t total = static_cast<size_t>(requests) * dim;
  const int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  embedding_direct_gather<<<grid, kBlockSize, 0, stream>>>(
      table, ids, output, requests, dim);
}

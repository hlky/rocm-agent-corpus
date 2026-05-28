#include <hip/hip_runtime.h>

namespace {

__global__ void compact_frontier_atomic(const int* __restrict__ flags,
                                        const int* __restrict__ vertices,
                                        int* __restrict__ frontier,
                                        int* __restrict__ frontier_count,
                                        int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || flags[i] == 0) {
    return;
  }

  const int out = atomicAdd(frontier_count, 1);
  frontier[out] = vertices ? vertices[i] : i;
}

}  // namespace

extern "C" void launch_baseline(const int* flags,
                                const int* vertices,
                                int* frontier,
                                int* frontier_count,
                                int n,
                                hipStream_t stream) {
  constexpr int kBlockSize = 256;
  hipMemsetAsync(frontier_count, 0, sizeof(int), stream);
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  compact_frontier_atomic<<<grid, kBlockSize, 0, stream>>>(
      flags, vertices, frontier, frontier_count, n);
}

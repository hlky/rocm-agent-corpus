#include <hip/hip_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void compact_frontier_block_scan(const int* __restrict__ flags,
                                            const int* __restrict__ vertices,
                                            int* __restrict__ frontier,
                                            int* __restrict__ frontier_count,
                                            int n) {
  __shared__ int scan[kBlockSize];
  __shared__ int block_base;

  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  const int active = (i < n && flags[i] != 0) ? 1 : 0;
  scan[tid] = active;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    const int addend = tid >= offset ? scan[tid - offset] : 0;
    __syncthreads();
    scan[tid] += addend;
    __syncthreads();
  }

  const int block_count = scan[blockDim.x - 1];
  if (tid == 0) {
    block_base = block_count > 0 ? atomicAdd(frontier_count, block_count) : 0;
  }
  __syncthreads();

  if (active) {
    const int local_rank = scan[tid] - 1;
    frontier[block_base + local_rank] = vertices ? vertices[i] : i;
  }
}

}  // namespace

extern "C" void launch_optimized(const int* flags,
                                 const int* vertices,
                                 int* frontier,
                                 int* frontier_count,
                                 int n,
                                 hipStream_t stream) {
  hipMemsetAsync(frontier_count, 0, sizeof(int), stream);
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  compact_frontier_block_scan<<<grid, kBlockSize, 0, stream>>>(
      flags, vertices, frontier, frontier_count, n);
}

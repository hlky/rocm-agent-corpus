#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kBins = 256;
constexpr int kBlockSize = 256;
constexpr int kItemsPerThreadForLaunch = 8;
constexpr int kMaxBlocks = 4096;

__global__ void histogram_block_private(const std::uint32_t* __restrict__ input,
                                        std::uint32_t* __restrict__ histogram,
                                        int n) {
  __shared__ std::uint32_t block_histogram[kBins];

  for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
    block_histogram[bin] = 0;
  }
  __syncthreads();

  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = start; i < n; i += stride) {
    atomicAdd(&block_histogram[input[i] & (kBins - 1)], 1u);
  }
  __syncthreads();

  for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
    const std::uint32_t count = block_histogram[bin];
    if (count != 0) {
      atomicAdd(&histogram[bin], count);
    }
  }
}

}  // namespace

extern "C" void launch_optimized(const std::uint32_t* input,
                                 std::uint32_t* histogram,
                                 int n,
                                 hipStream_t stream) {
  hipMemsetAsync(histogram, 0, kBins * sizeof(std::uint32_t), stream);
  int grid = (n + kBlockSize * kItemsPerThreadForLaunch - 1) /
             (kBlockSize * kItemsPerThreadForLaunch);
  if (grid < 1) {
    grid = 1;
  }
  if (grid > kMaxBlocks) {
    grid = kMaxBlocks;
  }
  histogram_block_private<<<grid, kBlockSize, 0, stream>>>(input, histogram, n);
}

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kBins = 256;
constexpr int kBlockSize = 256;

__global__ void histogram_global_atomic(const std::uint32_t* __restrict__ input,
                                        std::uint32_t* __restrict__ histogram,
                                        int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&histogram[input[i] & (kBins - 1)], 1u);
  }
}

}  // namespace

extern "C" void launch_baseline(const std::uint32_t* input,
                                std::uint32_t* histogram,
                                int n,
                                hipStream_t stream) {
  hipMemsetAsync(histogram, 0, kBins * sizeof(std::uint32_t), stream);
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  histogram_global_atomic<<<grid, kBlockSize, 0, stream>>>(input, histogram, n);
}

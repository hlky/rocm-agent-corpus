#include <hip/hip_runtime.h>

#include <cstddef>

namespace {

__global__ void compact_atomic_kernel(const int* __restrict__ input,
                                      const unsigned char* __restrict__ flags,
                                      int* __restrict__ output,
                                      int* __restrict__ selected_count,
                                      int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || flags[i] == 0) {
    return;
  }

  const int position = atomicAdd(selected_count, 1);
  output[position] = input[i];
}

}  // namespace

extern "C" size_t select_filter_compact_temp_bytes(const int*,
                                                   const unsigned char*,
                                                   int*,
                                                   int*,
                                                   int,
                                                   hipStream_t) {
  return 0;
}

extern "C" void launch_optimized(const int* input,
                                 const unsigned char* flags,
                                 int* output,
                                 int* selected_count,
                                 int n,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  hipMemsetAsync(selected_count, 0, sizeof(int), stream);
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  compact_atomic_kernel<<<grid, kBlockSize, 0, stream>>>(
      input, flags, output, selected_count, n);
}

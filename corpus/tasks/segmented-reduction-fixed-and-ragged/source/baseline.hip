#include <hip/hip_runtime.h>

namespace {

__global__ void segmented_sum_one_thread_per_segment(const float* __restrict__ input,
                                                     const int* __restrict__ offsets,
                                                     float* __restrict__ output,
                                                     int segments,
                                                     int fixed_segment_size,
                                                     int mode) {
  const int segment = blockIdx.x * blockDim.x + threadIdx.x;
  if (segment >= segments) {
    return;
  }

  int begin = segment * fixed_segment_size;
  int end = begin + fixed_segment_size;
  if (mode != 0) {
    begin = offsets[segment];
    end = offsets[segment + 1];
  }

  float sum = 0.0f;
  for (int i = begin; i < end; ++i) {
    sum += input[i];
  }
  output[segment] = sum;
}

}  // namespace

extern "C" void launch_baseline(const float* input,
                                const int* offsets,
                                float* output,
                                int segments,
                                int fixed_segment_size,
                                int mode,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (segments + kBlockSize - 1) / kBlockSize;
  segmented_sum_one_thread_per_segment<<<grid, kBlockSize, 0, stream>>>(
      input, offsets, output, segments, fixed_segment_size, mode);
}

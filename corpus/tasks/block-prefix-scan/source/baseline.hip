#include <hip/hip_runtime.h>

__global__ void fixed_segment_scan_one_thread_per_segment(
    const float* __restrict__ input,
    float* __restrict__ output,
    int segments,
    int segment_size) {
  const int segment = blockIdx.x * blockDim.x + threadIdx.x;
  if (segment >= segments) {
    return;
  }

  const int base = segment * segment_size;
  float running = 0.0f;
  for (int i = 0; i < segment_size; ++i) {
    running += input[base + i];
    output[base + i] = running;
  }
}

extern "C" void launch_baseline(const float* input,
                                float* output,
                                int segments,
                                int segment_size,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (segments + kBlockSize - 1) / kBlockSize;
  fixed_segment_scan_one_thread_per_segment<<<grid, kBlockSize, 0, stream>>>(
      input, output, segments, segment_size);
}

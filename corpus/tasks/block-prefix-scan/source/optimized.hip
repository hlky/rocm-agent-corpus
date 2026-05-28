#include <hip/hip_runtime.h>

namespace {

int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

}  // namespace

__global__ void fixed_segment_scan_shared_block(const float* __restrict__ input,
                                                float* __restrict__ output,
                                                int segments,
                                                int segment_size) {
  extern __shared__ float shared[];
  const int segment = blockIdx.x;
  const int tid = threadIdx.x;
  if (segment >= segments) {
    return;
  }

  const int base = segment * segment_size;
  shared[tid] = tid < segment_size ? input[base + tid] : 0.0f;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float addend = 0.0f;
    if (tid >= offset) {
      addend = shared[tid - offset];
    }
    __syncthreads();
    if (tid < segment_size) {
      shared[tid] += addend;
    }
    __syncthreads();
  }

  if (tid < segment_size) {
    output[base + tid] = shared[tid];
  }
}

extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int segments,
                                 int segment_size,
                                 hipStream_t stream) {
  int block_size = next_power_of_two(segment_size);
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 1024) {
    block_size = 1024;
  }
  fixed_segment_scan_shared_block<<<segments, block_size, block_size * sizeof(float), stream>>>(
      input, output, segments, segment_size);
}

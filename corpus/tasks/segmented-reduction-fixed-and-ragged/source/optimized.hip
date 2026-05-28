#include <hip/hip_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void segmented_sum_one_block_per_segment(const float* __restrict__ input,
                                                    const int* __restrict__ offsets,
                                                    float* __restrict__ output,
                                                    int segments,
                                                    int fixed_segment_size,
                                                    int mode) {
  extern __shared__ float shared[];
  const int segment = blockIdx.x;
  const int tid = threadIdx.x;
  if (segment >= segments) {
    return;
  }

  int begin = segment * fixed_segment_size;
  int end = begin + fixed_segment_size;
  if (mode != 0) {
    begin = offsets[segment];
    end = offsets[segment + 1];
  }

  float local = 0.0f;
  for (int i = begin + tid; i < end; i += blockDim.x) {
    local += input[i];
  }
  shared[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[segment] = shared[0];
  }
}

}  // namespace

extern "C" void launch_optimized(const float* input,
                                 const int* offsets,
                                 float* output,
                                 int segments,
                                 int fixed_segment_size,
                                 int mode,
                                 hipStream_t stream) {
  segmented_sum_one_block_per_segment<<<segments, kBlockSize,
                                        kBlockSize * sizeof(float), stream>>>(
      input, offsets, output, segments, fixed_segment_size, mode);
}

#include <hip/hip_runtime.h>

#include <math.h>

namespace {

constexpr int kModeLayerNorm = 1;

__device__ float block_sum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

__global__ void norm_one_block_per_row(const float* __restrict__ input,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols,
                                       float epsilon,
                                       int mode) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int base = row * cols;
  if (mode == kModeLayerNorm) {
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
      local_sum += input[base + col];
    }
    const float mean = block_sum(local_sum, shared) / static_cast<float>(cols);

    float local_sum_squares = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
      const float centered = input[base + col] - mean;
      local_sum_squares += centered * centered;
    }
    const float variance = block_sum(local_sum_squares, shared) / static_cast<float>(cols);
    const float inv_std = rsqrtf(variance + epsilon);

    for (int col = tid; col < cols; col += blockDim.x) {
      const float normalized = (input[base + col] - mean) * inv_std;
      output[base + col] = normalized * gamma[col] + beta[col];
    }
    return;
  }

  float local_sum_squares = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = input[base + col];
    local_sum_squares += value * value;
  }
  const float mean_square = block_sum(local_sum_squares, shared) / static_cast<float>(cols);
  const float inv_rms = rsqrtf(mean_square + epsilon);

  for (int col = tid; col < cols; col += blockDim.x) {
    output[base + col] = input[base + col] * inv_rms * gamma[col];
  }
}

}  // namespace

extern "C" void launch_optimized(const float* input,
                                 const float* gamma,
                                 const float* beta,
                                 float* output,
                                 int rows,
                                 int cols,
                                 float epsilon,
                                 int mode,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  norm_one_block_per_row<<<rows, kBlockSize, kBlockSize * sizeof(float), stream>>>(
      input, gamma, beta, output, rows, cols, epsilon, mode);
}

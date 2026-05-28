#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>

__global__ void softmax_one_block_per_row(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int rows,
                                          int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int base = row * cols;
  float local_max = -FLT_MAX;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, input[base + col]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = expf(input[base + col] - row_max);
    output[base + col] = value;
    local_sum += value;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / shared[0];

  for (int col = tid; col < cols; col += blockDim.x) {
    output[base + col] *= inv_sum;
  }
}

extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int rows,
                                 int cols,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  softmax_one_block_per_row<<<rows, kBlockSize, kBlockSize * sizeof(float), stream>>>(
      input, output, rows, cols);
}


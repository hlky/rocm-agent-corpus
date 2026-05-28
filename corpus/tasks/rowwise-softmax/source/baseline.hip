#include <hip/hip_runtime.h>

#include <math.h>

__global__ void softmax_one_thread_per_row(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int rows,
                                           int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int base = row * cols;
  float max_value = input[base];
  for (int col = 1; col < cols; ++col) {
    max_value = fmaxf(max_value, input[base + col]);
  }

  float sum = 0.0f;
  for (int col = 0; col < cols; ++col) {
    const float value = expf(input[base + col] - max_value);
    output[base + col] = value;
    sum += value;
  }

  const float inv_sum = 1.0f / sum;
  for (int col = 0; col < cols; ++col) {
    output[base + col] *= inv_sum;
  }
}

extern "C" void launch_baseline(const float* input,
                                float* output,
                                int rows,
                                int cols,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (rows + kBlockSize - 1) / kBlockSize;
  softmax_one_thread_per_row<<<grid, kBlockSize, 0, stream>>>(input, output, rows, cols);
}


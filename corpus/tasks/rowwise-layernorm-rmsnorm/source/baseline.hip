#include <hip/hip_runtime.h>

#include <math.h>

namespace {

constexpr int kModeLayerNorm = 1;

__global__ void norm_one_thread_per_row(const float* __restrict__ input,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ output,
                                        int rows,
                                        int cols,
                                        float epsilon,
                                        int mode) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int base = row * cols;
  if (mode == kModeLayerNorm) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += input[base + col];
    }
    const float mean = sum / static_cast<float>(cols);

    float sum_squares = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float centered = input[base + col] - mean;
      sum_squares += centered * centered;
    }
    const float inv_std = rsqrtf(sum_squares / static_cast<float>(cols) + epsilon);

    for (int col = 0; col < cols; ++col) {
      const float normalized = (input[base + col] - mean) * inv_std;
      output[base + col] = normalized * gamma[col] + beta[col];
    }
    return;
  }

  float sum_squares = 0.0f;
  for (int col = 0; col < cols; ++col) {
    const float value = input[base + col];
    sum_squares += value * value;
  }
  const float inv_rms = rsqrtf(sum_squares / static_cast<float>(cols) + epsilon);

  for (int col = 0; col < cols; ++col) {
    output[base + col] = input[base + col] * inv_rms * gamma[col];
  }
}

}  // namespace

extern "C" void launch_baseline(const float* input,
                                const float* gamma,
                                const float* beta,
                                float* output,
                                int rows,
                                int cols,
                                float epsilon,
                                int mode,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (rows + kBlockSize - 1) / kBlockSize;
  norm_one_thread_per_row<<<grid, kBlockSize, 0, stream>>>(
      input, gamma, beta, output, rows, cols, epsilon, mode);
}

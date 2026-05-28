#include <hip/hip_runtime.h>

#include <math.h>

namespace {

constexpr int kModeLayerNorm = 1;

__global__ void norm_backward_atomic_kernel(const float* __restrict__ x,
                                            const float* __restrict__ dy,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ saved_mean,
                                            const float* __restrict__ saved_inv,
                                            float* __restrict__ dx,
                                            float* __restrict__ dgamma,
                                            float* __restrict__ dbeta,
                                            int rows,
                                            int cols,
                                            int mode) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int base = row * cols;
  const float inv = saved_inv[row];

  if (mode == kModeLayerNorm) {
    const float mean = saved_mean[row];
    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_xhat = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float xhat = (x[base + col] - mean) * inv;
      const float dy_gamma = dy[base + col] * gamma[col];
      sum_dy_gamma += dy_gamma;
      sum_dy_gamma_xhat += dy_gamma * xhat;
    }

    const float inv_cols = 1.0f / static_cast<float>(cols);
    const float mean_dy_gamma = sum_dy_gamma * inv_cols;
    const float mean_dy_gamma_xhat = sum_dy_gamma_xhat * inv_cols;
    for (int col = 0; col < cols; ++col) {
      const float dy_value = dy[base + col];
      const float xhat = (x[base + col] - mean) * inv;
      const float dy_gamma = dy_value * gamma[col];
      dx[base + col] = (dy_gamma - mean_dy_gamma - xhat * mean_dy_gamma_xhat) * inv;
      atomicAdd(dgamma + col, dy_value * xhat);
      atomicAdd(dbeta + col, dy_value);
    }
    return;
  }

  float dot = 0.0f;
  for (int col = 0; col < cols; ++col) {
    dot += dy[base + col] * gamma[col] * x[base + col];
  }

  const float scale = inv * inv * inv * dot / static_cast<float>(cols);
  for (int col = 0; col < cols; ++col) {
    const float x_value = x[base + col];
    const float dy_value = dy[base + col];
    dx[base + col] = dy_value * gamma[col] * inv - x_value * scale;
    atomicAdd(dgamma + col, dy_value * x_value * inv);
  }
}

}  // namespace

extern "C" void launch_baseline(const float* x,
                                const float* dy,
                                const float* gamma,
                                const float* saved_mean,
                                const float* saved_inv,
                                float* dx,
                                float* dgamma,
                                float* dbeta,
                                int rows,
                                int cols,
                                int mode,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const int grid = (rows + kBlockSize - 1) / kBlockSize;
  norm_backward_atomic_kernel<<<grid, kBlockSize, 0, stream>>>(
      x, dy, gamma, saved_mean, saved_inv, dx, dgamma, dbeta, rows, cols, mode);
}

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
  // Keep later reductions from reusing shared[0] before every lane has read it.
  const float result = shared[0];
  __syncthreads();
  return result;
}

__global__ void norm_backward_dx_kernel(const float* __restrict__ x,
                                        const float* __restrict__ dy,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ saved_mean,
                                        const float* __restrict__ saved_inv,
                                        float* __restrict__ dx,
                                        int rows,
                                        int cols,
                                        int mode) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int base = row * cols;
  const float inv = saved_inv[row];

  if (mode == kModeLayerNorm) {
    const float mean = saved_mean[row];
    float local_dy_gamma = 0.0f;
    float local_dy_gamma_xhat = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
      const float xhat = (x[base + col] - mean) * inv;
      const float dy_gamma = dy[base + col] * gamma[col];
      local_dy_gamma += dy_gamma;
      local_dy_gamma_xhat += dy_gamma * xhat;
    }

    const float sum_dy_gamma = block_sum(local_dy_gamma, shared);
    const float sum_dy_gamma_xhat = block_sum(local_dy_gamma_xhat, shared);
    const float inv_cols = 1.0f / static_cast<float>(cols);
    const float mean_dy_gamma = sum_dy_gamma * inv_cols;
    const float mean_dy_gamma_xhat = sum_dy_gamma_xhat * inv_cols;
    for (int col = tid; col < cols; col += blockDim.x) {
      const float xhat = (x[base + col] - mean) * inv;
      const float dy_gamma = dy[base + col] * gamma[col];
      dx[base + col] = (dy_gamma - mean_dy_gamma - xhat * mean_dy_gamma_xhat) * inv;
    }
    return;
  }

  float local_dot = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_dot += dy[base + col] * gamma[col] * x[base + col];
  }

  const float dot = block_sum(local_dot, shared);
  const float scale = inv * inv * inv * dot / static_cast<float>(cols);
  for (int col = tid; col < cols; col += blockDim.x) {
    const float x_value = x[base + col];
    dx[base + col] = dy[base + col] * gamma[col] * inv - x_value * scale;
  }
}

__global__ void norm_backward_param_kernel(const float* __restrict__ x,
                                           const float* __restrict__ dy,
                                           const float* __restrict__ saved_mean,
                                           const float* __restrict__ saved_inv,
                                           float* __restrict__ dgamma,
                                           float* __restrict__ dbeta,
                                           int rows,
                                           int cols,
                                           int mode) {
  extern __shared__ float shared[];
  float* shared_dgamma = shared;
  float* shared_dbeta = shared + blockDim.x;

  const int col = blockIdx.x;
  const int tid = threadIdx.x;
  if (col >= cols) {
    return;
  }

  float local_dgamma = 0.0f;
  float local_dbeta = 0.0f;
  for (int row = tid; row < rows; row += blockDim.x) {
    const int offset = row * cols + col;
    const float dy_value = dy[offset];
    const float inv = saved_inv[row];
    const float normalized =
        mode == kModeLayerNorm ? (x[offset] - saved_mean[row]) * inv : x[offset] * inv;
    local_dgamma += dy_value * normalized;
    if (mode == kModeLayerNorm) {
      local_dbeta += dy_value;
    }
  }

  shared_dgamma[tid] = local_dgamma;
  shared_dbeta[tid] = local_dbeta;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_dgamma[tid] += shared_dgamma[tid + stride];
      shared_dbeta[tid] += shared_dbeta[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    dgamma[col] = shared_dgamma[0];
    dbeta[col] = mode == kModeLayerNorm ? shared_dbeta[0] : 0.0f;
  }
}

}  // namespace

extern "C" void launch_optimized(const float* x,
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
  constexpr int kBlockSize = 256;
  norm_backward_dx_kernel<<<rows, kBlockSize, kBlockSize * sizeof(float), stream>>>(
      x, dy, gamma, saved_mean, saved_inv, dx, rows, cols, mode);
  norm_backward_param_kernel<<<cols, kBlockSize, 2 * kBlockSize * sizeof(float), stream>>>(
      x, dy, saved_mean, saved_inv, dgamma, dbeta, rows, cols, mode);
}

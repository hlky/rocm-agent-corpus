#include <hip/hip_runtime.h>

#include <math.h>

namespace {

__global__ void adamw_moment_stage(const float* __restrict__ grads,
                                   float* __restrict__ exp_avg,
                                   float* __restrict__ exp_avg_sq,
                                   int n,
                                   float beta1,
                                   float beta2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const float g = grads[i];
  exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
  exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
}

__global__ void adamw_update_stage(const float* __restrict__ exp_avg,
                                   const float* __restrict__ exp_avg_sq,
                                   float* __restrict__ update_buffer,
                                   int n,
                                   float eps,
                                   float bias_correction1,
                                   float bias_correction2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const float m_hat = exp_avg[i] / bias_correction1;
  const float v_hat = exp_avg_sq[i] / bias_correction2;
  update_buffer[i] = m_hat / (sqrtf(v_hat) + eps);
}

__global__ void adamw_apply_stage(float* __restrict__ params,
                                  const float* __restrict__ update_buffer,
                                  int n,
                                  float lr,
                                  float weight_decay) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  params[i] -= lr * (update_buffer[i] + weight_decay * params[i]);
}

}  // namespace

extern "C" void launch_baseline(float* params,
                                const float* grads,
                                float* exp_avg,
                                float* exp_avg_sq,
                                float* update_buffer,
                                int n,
                                float lr,
                                float beta1,
                                float beta2,
                                float eps,
                                float weight_decay,
                                float bias_correction1,
                                float bias_correction2,
                                hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  adamw_moment_stage<<<grid, kBlockSize, 0, stream>>>(grads, exp_avg, exp_avg_sq, n, beta1, beta2);
  adamw_update_stage<<<grid, kBlockSize, 0, stream>>>(
      exp_avg, exp_avg_sq, update_buffer, n, eps, bias_correction1, bias_correction2);
  adamw_apply_stage<<<grid, kBlockSize, 0, stream>>>(params, update_buffer, n, lr, weight_decay);
}

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float adamw_one(float param,
                                           float grad,
                                           float& m,
                                           float& v,
                                           float lr,
                                           float beta1,
                                           float beta2,
                                           float eps,
                                           float weight_decay,
                                           float bias_correction1,
                                           float bias_correction2) {
  m = beta1 * m + (1.0f - beta1) * grad;
  v = beta2 * v + (1.0f - beta2) * grad * grad;
  const float m_hat = m / bias_correction1;
  const float v_hat = v / bias_correction2;
  const float update = m_hat / (sqrtf(v_hat) + eps);
  return param - lr * (update + weight_decay * param);
}

__global__ void adamw_fused_scalar(float* __restrict__ params,
                                   const float* __restrict__ grads,
                                   float* __restrict__ exp_avg,
                                   float* __restrict__ exp_avg_sq,
                                   int n,
                                   float lr,
                                   float beta1,
                                   float beta2,
                                   float eps,
                                   float weight_decay,
                                   float bias_correction1,
                                   float bias_correction2) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float m = exp_avg[i];
    float v = exp_avg_sq[i];
    params[i] = adamw_one(params[i], grads[i], m, v, lr, beta1, beta2, eps, weight_decay,
                          bias_correction1, bias_correction2);
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
  }
}

__global__ void adamw_fused_float4(float4* __restrict__ params4,
                                   const float4* __restrict__ grads4,
                                   float4* __restrict__ exp_avg4,
                                   float4* __restrict__ exp_avg_sq4,
                                   int n4,
                                   float lr,
                                   float beta1,
                                   float beta2,
                                   float eps,
                                   float weight_decay,
                                   float bias_correction1,
                                   float bias_correction2) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4 p = params4[i];
    const float4 g = grads4[i];
    float4 m = exp_avg4[i];
    float4 v = exp_avg_sq4[i];

    p.x = adamw_one(p.x, g.x, m.x, v.x, lr, beta1, beta2, eps, weight_decay,
                    bias_correction1, bias_correction2);
    p.y = adamw_one(p.y, g.y, m.y, v.y, lr, beta1, beta2, eps, weight_decay,
                    bias_correction1, bias_correction2);
    p.z = adamw_one(p.z, g.z, m.z, v.z, lr, beta1, beta2, eps, weight_decay,
                    bias_correction1, bias_correction2);
    p.w = adamw_one(p.w, g.w, m.w, v.w, lr, beta1, beta2, eps, weight_decay,
                    bias_correction1, bias_correction2);

    params4[i] = p;
    exp_avg4[i] = m;
    exp_avg_sq4[i] = v;
  }
}

bool aligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x0f) == 0;
}

}  // namespace

extern "C" void launch_optimized(float* params,
                                 const float* grads,
                                 float* exp_avg,
                                 float* exp_avg_sq,
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
  if ((n % 4) == 0 && aligned16(params) && aligned16(grads) &&
      aligned16(exp_avg) && aligned16(exp_avg_sq)) {
    const int n4 = n / 4;
    const int grid = (n4 + kBlockSize - 1) / kBlockSize;
    adamw_fused_float4<<<grid, kBlockSize, 0, stream>>>(
        reinterpret_cast<float4*>(params),
        reinterpret_cast<const float4*>(grads),
        reinterpret_cast<float4*>(exp_avg),
        reinterpret_cast<float4*>(exp_avg_sq),
        n4, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
    return;
  }

  const int grid = (n + kBlockSize - 1) / kBlockSize;
  adamw_fused_scalar<<<grid, kBlockSize, 0, stream>>>(
      params, grads, exp_avg, exp_avg_sq, n, lr, beta1, beta2, eps,
      weight_decay, bias_correction1, bias_correction2);
}

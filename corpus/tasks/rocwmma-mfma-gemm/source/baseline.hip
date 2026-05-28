#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

__global__ void hgemm_scalar_kernel(const half* __restrict__ a,
                                    const half* __restrict__ b,
                                    float* __restrict__ c,
                                    int m,
                                    int n,
                                    int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    const float a_value = __half2float(a[row * k + kk]);
    const float b_value = __half2float(b[col * k + kk]);
    acc += a_value * b_value;
  }

  c[row * n + col] = acc;
}

}  // namespace

extern "C" void launch_baseline(const half* a,
                                const half* b,
                                float* c,
                                int m,
                                int n,
                                int k,
                                hipStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  hgemm_scalar_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

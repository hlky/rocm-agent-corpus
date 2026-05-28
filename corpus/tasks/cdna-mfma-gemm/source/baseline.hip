#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void scalar_hgemm(const half* a, const half* b, float* c, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  c[row * n + col] = acc;
}

extern "C" void launch_baseline(const half* a, const half* b, float* c, int m, int n, int k, hipStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (m + 15) / 16);
  scalar_hgemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

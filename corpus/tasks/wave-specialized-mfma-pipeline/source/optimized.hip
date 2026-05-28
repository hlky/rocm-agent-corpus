#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Scaffold only. Replace this placeholder with a CK Tile or hand-written MFMA
// mainloop that assigns waves to producer/consumer roles, stages tiles through
// LDS, and records barriers, wait counts, and resource usage.
__global__ void wave_specialized_mfma_placeholder(const half* a, const half* b,
                                                  float* c, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  c[row * n + col] = acc;
}

extern "C" void launch_optimized(const half* a, const half* b, float* c,
                                 int m, int n, int k, hipStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  wave_specialized_mfma_placeholder<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

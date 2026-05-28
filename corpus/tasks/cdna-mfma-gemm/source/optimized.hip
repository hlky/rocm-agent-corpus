#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// TODO(gfx950): replace this placeholder with a Composable Kernel/CK Tile or inline WGMFMA
// mainloop using global-to-LDS staging descriptors, mbarriers, and explicit pipeline stages.
__global__ void hopper_wgmma_tma_placeholder(const half* a, const half* b, float* c, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  c[row * n + col] = acc;
}

extern "C" void launch_optimized(const half* a, const half* b, float* c, int m, int n, int k, hipStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (m + 15) / 16);
  hopper_wgmma_tma_placeholder<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

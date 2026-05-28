#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void scalar_tile_gemm(const half* a, const half* b, float* c, int tiles) {
  int tile = blockIdx.x;
  int lane = threadIdx.x;
  if (tile >= tiles || lane >= 32) return;
  int row = lane / 8;
  int col = lane % 8;
  float acc = 0.0f;
  for (int k = 0; k < 16; ++k) {
    acc += __half2float(a[tile * 16 * 16 + row * 16 + k]) * __half2float(b[tile * 16 * 8 + k * 8 + col]);
  }
  c[tile * 16 * 8 + row * 8 + col] = acc;
}

extern "C" void launch_baseline(const half* a, const half* b, float* c, int tiles, hipStream_t stream) {
  scalar_tile_gemm<<<tiles, 32, 0, stream>>>(a, b, c, tiles);
}

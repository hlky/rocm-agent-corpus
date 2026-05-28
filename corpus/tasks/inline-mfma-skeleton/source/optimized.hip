#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void mma_sync_placeholder(const half* a, const half* b, float* c, int tiles) {
  int tile = blockIdx.x;
  int lane = threadIdx.x;
  if (tile >= tiles || lane >= 32) return;
  // TODO: load A/B fragments into registers and issue mfma.aligned.* LLVM IR / AMD GCN ISA.
  int row = lane / 8;
  int col = lane % 8;
  float acc = 0.0f;
  for (int k = 0; k < 16; ++k) {
    acc += __half2float(a[tile * 16 * 16 + row * 16 + k]) * __half2float(b[tile * 16 * 8 + k * 8 + col]);
  }
  c[tile * 16 * 8 + row * 8 + col] = acc;
}

extern "C" void launch_optimized(const half* a, const half* b, float* c, int tiles, hipStream_t stream) {
  mma_sync_placeholder<<<tiles, 32, 0, stream>>>(a, b, c, tiles);
}

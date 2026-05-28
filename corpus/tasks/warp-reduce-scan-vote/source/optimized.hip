#include <hip/hip_runtime.h>

__global__ void shuffle_warp_group_sum_scan_vote(const float* input, float* sums, float* scans, unsigned* masks, int groups) {
  int gid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (gid >= groups) return;
  unsigned active = 0xffffffffu;
  float v = input[gid * 32 + lane];
  unsigned positive = __ballot_sync(active, v > 0.0f);
  float prefix = v;
  for (int offset = 1; offset < 32; offset <<= 1) {
    float other = __shfl_up_sync(active, prefix, offset);
    if (lane >= offset) prefix += other;
  }
  scans[gid * 32 + lane] = prefix;
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(active, v, offset);
  }
  if (lane == 0) {
    sums[gid] = v;
    masks[gid] = positive;
  }
}

extern "C" void launch_optimized(const float* input, float* sums, float* scans, unsigned* masks, int groups, hipStream_t stream) {
  int warps_per_block = 8;
  shuffle_warp_group_sum_scan_vote<<<(groups + warps_per_block - 1) / warps_per_block, warps_per_block * 32, 0, stream>>>(input, sums, scans, masks, groups);
}

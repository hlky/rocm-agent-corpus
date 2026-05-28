#include <hip/hip_runtime.h>

__global__ void shared_warp_group_sum_scan_vote(const float* input, float* sums, float* scans, unsigned* masks, int groups) {
  __shared__ float tile[256];
  int gid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (gid >= groups) return;
  float v = input[gid * 32 + lane];
  tile[threadIdx.x] = v;
  __syncthreads();
  if (lane == 0) {
    float sum = 0.0f;
    unsigned mask = 0;
    int base = (threadIdx.x / 32) * 32;
    for (int i = 0; i < 32; ++i) {
      float x = tile[base + i];
      sum += x;
      scans[gid * 32 + i] = sum;
      if (x > 0.0f) mask |= (1u << i);
    }
    sums[gid] = sum;
    masks[gid] = mask;
  }
}

extern "C" void launch_baseline(const float* input, float* sums, float* scans, unsigned* masks, int groups, hipStream_t stream) {
  int warps_per_block = 8;
  shared_warp_group_sum_scan_vote<<<(groups + warps_per_block - 1) / warps_per_block, warps_per_block * 32, 0, stream>>>(input, sums, scans, masks, groups);
}

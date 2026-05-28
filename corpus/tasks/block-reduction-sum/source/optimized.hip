#include <hip/hip_runtime.h>

__global__ void sum_block_reduce(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int n) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int global = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float value = 0.0f;
  if (global < n) {
    value += input[global];
  }
  const int second = global + blockDim.x;
  if (second < n) {
    value += input[second];
  }
  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile float* warp = shared;
    warp[tid] += warp[tid + 32];
    warp[tid] += warp[tid + 16];
    warp[tid] += warp[tid + 8];
    warp[tid] += warp[tid + 4];
    warp[tid] += warp[tid + 2];
    warp[tid] += warp[tid + 1];
  }

  if (tid == 0) {
    atomicAdd(output, shared[0]);
  }
}

extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int n,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  hipMemsetAsync(output, 0, sizeof(float), stream);
  const int grid = (n + kBlockSize * 2 - 1) / (kBlockSize * 2);
  sum_block_reduce<<<grid, kBlockSize, kBlockSize * sizeof(float), stream>>>(input, output, n);
}


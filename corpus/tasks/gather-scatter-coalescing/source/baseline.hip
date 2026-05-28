#include <hip/hip_runtime.h>

__global__ void gather_scatter_scalar_kernel(const float* __restrict__ input,
                                             const int* __restrict__ indices,
                                             float* __restrict__ gathered,
                                             float* __restrict__ scattered,
                                             int n,
                                             int coalesced_mode) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const int src = coalesced_mode ? i : indices[i];
  gathered[i] = input[src];

  const int dst = coalesced_mode ? i : indices[i];
  scattered[dst] = input[i];
}

extern "C" void launch_baseline(const float* input,
                                const int* indices,
                                float* gathered,
                                float* scattered,
                                int n,
                                int coalesced_mode,
                                hipStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  gather_scatter_scalar_kernel<<<grid, block, 0, stream>>>(
      input, indices, gathered, scattered, n, coalesced_mode);
}

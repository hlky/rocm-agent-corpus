#include <hip/hip_runtime.h>

__global__ void gather_scatter_irregular_fallback_kernel(const float* __restrict__ input,
                                                         const int* __restrict__ indices,
                                                         float* __restrict__ gathered,
                                                         float* __restrict__ scattered,
                                                         int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const int mapped = indices[i];
  gathered[i] = input[mapped];
  scattered[mapped] = input[i];
}

__global__ void gather_scatter_coalesced_float4_kernel(const float* __restrict__ input,
                                                       float* __restrict__ gathered,
                                                       float* __restrict__ scattered,
                                                       int n) {
  const int vector_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = vector_i * 4;

  if (base + 3 < n) {
    const float4 value = reinterpret_cast<const float4*>(input)[vector_i];
    reinterpret_cast<float4*>(gathered)[vector_i] = value;
    reinterpret_cast<float4*>(scattered)[vector_i] = value;
    return;
  }

  for (int i = base; i < n; ++i) {
    gathered[i] = input[i];
    scattered[i] = input[i];
  }
}

extern "C" void launch_optimized(const float* input,
                                 const int* indices,
                                 float* gathered,
                                 float* scattered,
                                 int n,
                                 int coalesced_mode,
                                 hipStream_t stream) {
  const int block = 256;
  if (coalesced_mode) {
    const int vector_count = (n + 3) / 4;
    const int grid = (vector_count + block - 1) / block;
    gather_scatter_coalesced_float4_kernel<<<grid, block, 0, stream>>>(
        input, gathered, scattered, n);
  } else {
    const int grid = (n + block - 1) / block;
    gather_scatter_irregular_fallback_kernel<<<grid, block, 0, stream>>>(
        input, indices, gathered, scattered, n);
  }
}

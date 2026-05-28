#include <hip/hip_runtime.h>

__global__ void scale_kernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

extern "C" hipError_t launch_baseline(const float* h_in,
                                       float* h_out,
                                       float* d_in,
                                       float* d_out,
                                       int n,
                                       hipStream_t stream) {
  hipMemcpyAsync(d_in, h_in, sizeof(float) * n, hipMemcpyHostToDevice, stream);
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  scale_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
  hipMemcpyAsync(h_out, d_out, sizeof(float) * n, hipMemcpyDeviceToHost, stream);
  return hipGetLastError();
}

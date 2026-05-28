#include <hip/hip_runtime.h>

__global__ void scale_kernel_streamed(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

extern "C" hipError_t launch_optimized(const float* h_in,
                                        float* h_out,
                                        float* d_in,
                                        float* d_out,
                                        int n,
                                        int chunk_elements,
                                        hipStream_t* streams,
                                        int stream_count) {
  dim3 block(256);
  for (int offset = 0; offset < n; offset += chunk_elements) {
    int chunk = (chunk_elements < n - offset) ? chunk_elements : (n - offset);
    int s = (offset / chunk_elements) % stream_count;
    float* din = d_in + offset;
    float* dout = d_out + offset;
    hipMemcpyAsync(din, h_in + offset, sizeof(float) * chunk, hipMemcpyHostToDevice, streams[s]);
    dim3 grid((chunk + block.x - 1) / block.x);
    scale_kernel_streamed<<<grid, block, 0, streams[s]>>>(din, dout, chunk);
    hipMemcpyAsync(h_out + offset, dout, sizeof(float) * chunk, hipMemcpyDeviceToHost, streams[s]);
  }
  return hipGetLastError();
}

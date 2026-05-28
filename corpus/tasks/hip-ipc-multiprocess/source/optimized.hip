#include <hip/hip_runtime.h>

// Template-only optimized side of hip-ipc-multiprocess.
// Replace with the consumer-side kernel that operates directly on the opened
// IPC buffer after peer access and event synchronization are configured.

namespace {

__global__ void consume_shared_buffer(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f;
}

}  // namespace

extern "C" hipError_t launch_optimized(const float* shared_in,
                                        float* out,
                                        int n,
                                        hipStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  consume_shared_buffer<<<grid, block, 0, stream>>>(shared_in, out, n);
  return hipGetLastError();
}

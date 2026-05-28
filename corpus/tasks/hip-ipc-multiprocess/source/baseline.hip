#include <hip/hip_runtime.h>

// Template-only baseline for hip-ipc-multiprocess.
// A complete harness should split producer and consumer into separate
// processes, export hipIpcMemHandle_t/hipIpcEventHandle_t from the producer,
// and time setup separately from steady-state device work.

namespace {

__global__ void fill_buffer(float* data, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = value;
}

}  // namespace

extern "C" hipError_t launch_baseline(float* data, int n, hipStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  fill_buffer<<<grid, block, 0, stream>>>(data, n, 1.0f);
  return hipGetLastError();
}

#include <hip/hip_runtime.h>

// Template-only graph-captured inference step.
// A production harness should instantiate once, replay many times, and keep
// capture/instantiate/update/replay timings separate.

namespace {

__global__ void affine_step_graph(const float* x, const float* scale, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * scale[i] + 0.1f;
}

__global__ void activation_step_graph(float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] > 0.0f ? y[i] : 0.0f;
}

}  // namespace

extern "C" hipError_t launch_optimized(const float* x,
                                        const float* scale,
                                        float* y,
                                        int n,
                                        hipStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  affine_step_graph<<<grid, block, 0, stream>>>(x, scale, y, n);
  activation_step_graph<<<grid, block, 0, stream>>>(y, n);
  hipStreamEndCapture(stream, &graph);
  hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  hipGraphLaunch(exec, stream);
  hipError_t status = hipGetLastError();
  hipGraphExecDestroy(exec);
  hipGraphDestroy(graph);
  return status;
}

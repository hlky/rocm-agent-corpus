#include <hip/hip_runtime.h>

__global__ void tiny_axpy_graph(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

extern "C" hipError_t launch_optimized(float* y,
                                        const float* x,
                                        int n,
                                        int iterations,
                                        hipStream_t stream) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  for (int it = 0; it < iterations; ++it) {
    tiny_axpy_graph<<<grid, block, 0, stream>>>(y, x, 1.001f, n);
  }
  hipStreamEndCapture(stream, &graph);
  hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  hipGraphLaunch(exec, stream);
  hipError_t status = hipGetLastError();
  hipGraphExecDestroy(exec);
  hipGraphDestroy(graph);
  return status;
}

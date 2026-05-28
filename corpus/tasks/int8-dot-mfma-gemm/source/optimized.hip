#include <hip/hip_runtime.h>

// Template-only scaffold for int8-dot-mfma-gemm.
// Summary: Compare scalar/dequantized INT8 matmul, DP4A-style custom kernels, and library INT8 Matrix Core paths with explicit scale and output contracts.
// Optimized strategy: Use packed int8 vector loads, DP4A-style accumulation for fallback, or map the task to library/Composable Kernel IMFMA tiles for Matrix Core comparison.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void int8_dp4a_imma_gemm_optimized_kernel() {
  // TODO: implement the optimized HIP path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  int8_dp4a_imma_gemm_optimized_kernel<<<1, 1, 0, stream>>>();
}

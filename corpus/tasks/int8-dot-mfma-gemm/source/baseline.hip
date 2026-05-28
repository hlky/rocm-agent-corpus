#include <hip/hip_runtime.h>

// Template-only scaffold for int8-dot-mfma-gemm.
// Summary: Compare scalar/dequantized INT8 matmul, DP4A-style custom kernels, and library INT8 Matrix Core paths with explicit scale and output contracts.
// Baseline strategy: Materialize dequantized operands or use a scalar int8 dot-product kernel with int32 accumulation for oracle comparison.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void int8_dp4a_imma_gemm_baseline_kernel() {
  // TODO: implement the baseline CUDA path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  int8_dp4a_imma_gemm_baseline_kernel<<<1, 1, 0, stream>>>();
}

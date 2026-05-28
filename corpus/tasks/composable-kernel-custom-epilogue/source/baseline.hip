#include <hip/hip_runtime.h>

// Template-only scaffold for composable-kernel-custom-epilogue.
// Summary: Create a task boundary for custom Composable Kernel epilogue visitors that fuse bias, activation, scaling, auxiliary outputs, or metadata writes after GEMM.
// Baseline strategy: Use hipBLASLt or Composable Kernel built-in linear-combination epilogue, then run unsupported postprocessing in a separate kernel.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void composable-kernel_custom_epilogue_visitor_baseline_kernel() {
  // TODO: implement the baseline CUDA path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  composable-kernel_custom_epilogue_visitor_baseline_kernel<<<1, 1, 0, stream>>>();
}

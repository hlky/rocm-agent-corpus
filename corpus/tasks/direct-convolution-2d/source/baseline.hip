#include <hip/hip_runtime.h>

// Template-only scaffold for direct-convolution-2d.
// Summary: Teach direct convolution indexing, boundary handling, shared input tile reuse, and MIOpen/Composable Kernel baseline comparison.
// Baseline strategy: One output element per thread loops over channels and filter taps with direct global-memory loads.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void direct_convolution_2d_baseline_kernel() {
  // TODO: implement the baseline CUDA path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  direct_convolution_2d_baseline_kernel<<<1, 1, 0, stream>>>();
}

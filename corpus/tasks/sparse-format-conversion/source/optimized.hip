#include <hip/hip_runtime.h>

// Template-only scaffold for sparse-format-conversion.
// Summary: Convert between COO, CSR, CSC, ELL, and BSR-like sparse formats while exposing sorting, prefix, and metadata costs.
// Optimized strategy: Fuse histogram/count, prefix, and scatter steps for known formats, block sizes, and sortedness constraints.

namespace {

__global__ void sparse_format_conversion_optimized_kernel() {
  // TODO: replace with the optimized implementation before measuring.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  sparse_format_conversion_optimized_kernel<<<1, 1, 0, stream>>>();
}

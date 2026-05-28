#include <hip/hip_runtime.h>

// Template-only scaffold for sparse-format-conversion.
// Summary: Convert between COO, CSR, CSC, ELL, and BSR-like sparse formats while exposing sorting, prefix, and metadata costs.
// Baseline strategy: Use library conversion routines or a staged sort/scan/scatter pipeline with explicit temporary storage.

namespace {

__global__ void sparse_format_conversion_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  sparse_format_conversion_baseline_kernel<<<1, 1, 0, stream>>>();
}

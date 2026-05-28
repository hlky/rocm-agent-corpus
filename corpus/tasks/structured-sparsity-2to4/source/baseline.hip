#include <hip/hip_runtime.h>

// Template-only scaffold for structured-sparsity-2to4.
// Summary: Capture the 2:4 structured sparsity path, including metadata packing, pruning layout, sparse Matrix Core eligibility, and dense fallback comparison.
// Baseline strategy: Run dense GEMM or materialize sparse weights into a library-supported format and compare with a correctness oracle.

namespace {

__global__ void structured_sparsity_2to4_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  structured_sparsity_2to4_baseline_kernel<<<1, 1, 0, stream>>>();
}

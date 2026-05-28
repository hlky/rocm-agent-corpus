#include <hip/hip_runtime.h>

// Template-only scaffold for spgemm-merge-hash.
// Summary: Compare sparse matrix multiply strategies based on row-wise merge or hash accumulation against cuSPARSE SpGEMM-style planning boundaries.
// Baseline strategy: Use a simple row-wise expansion with global or temporary accumulation and a CPU/cuSPARSE oracle for final CSR output.

namespace {

__global__ void spgemm_merge_hash_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  spgemm_merge_hash_baseline_kernel<<<1, 1, 0, stream>>>();
}

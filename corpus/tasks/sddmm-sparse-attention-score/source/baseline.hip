#include <hip/hip_runtime.h>

// Template-only scaffold for sddmm-sparse-attention-score.
// Summary: Compute sampled dense-dense products only at sparse edge or block-mask locations, with attention-score metadata and coalescing contracts visible.
// Baseline strategy: One thread per nonzero computes a dot product using the sparse coordinates and dense matrices.

namespace {

__global__ void sddmm_sparse_attention_score_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  sddmm_sparse_attention_score_baseline_kernel<<<1, 1, 0, stream>>>();
}

#include <hip/hip_runtime.h>

// Template-only scaffold for sddmm-sparse-attention-score.
// Summary: Compute sampled dense-dense products only at sparse edge or block-mask locations, with attention-score metadata and coalescing contracts visible.
// Optimized strategy: Tile dense feature vectors for blocks of edges, reuse Q/K fragments, and specialize for block-sparse attention masks.

namespace {

__global__ void sddmm_sparse_attention_score_optimized_kernel() {
  // TODO: replace with the optimized implementation before measuring.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  sddmm_sparse_attention_score_optimized_kernel<<<1, 1, 0, stream>>>();
}

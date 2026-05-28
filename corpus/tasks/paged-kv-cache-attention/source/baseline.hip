#include <hip/hip_runtime.h>

// Template-only scaffold for paged-kv-cache-attention.
// Summary: Compute single-token or small-query decode attention over a paged KV cache while preserving explicit page-table, block-size, and cache-layout contracts.
// Baseline strategy: Materialize or gather contiguous KV for each sequence, then run a simple decode attention kernel or framework reference.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void paged_kv_cache_attention_baseline_kernel() {
  // TODO: implement the baseline CUDA path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  paged_kv_cache_attention_baseline_kernel<<<1, 1, 0, stream>>>();
}

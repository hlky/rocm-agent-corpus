#include <hip/hip_runtime.h>

// Template-only scaffold for gqa-mqa-decode-attention.
// Summary: Specialize decode attention for grouped-query and multi-query attention where many query heads share fewer KV heads.
// Baseline strategy: Expand or remap KV heads to query-head layout and reuse a generic decode attention boundary.

namespace {

__global__ void gqa_mqa_decode_attention_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  gqa_mqa_decode_attention_baseline_kernel<<<1, 1, 0, stream>>>();
}

#include <hip/hip_runtime.h>

// Template-only scaffold for gqa-mqa-decode-attention.
// Summary: Specialize decode attention for grouped-query and multi-query attention where many query heads share fewer KV heads.
// Optimized strategy: Map query head groups directly to shared KV heads, reduce repeated cache reads, and specialize for fixed grouping factors.

namespace {

__global__ void gqa_mqa_decode_attention_optimized_kernel() {
  // TODO: replace with the optimized implementation before measuring.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  gqa_mqa_decode_attention_optimized_kernel<<<1, 1, 0, stream>>>();
}

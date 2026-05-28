#include <hip/hip_runtime.h>

// Template-only scaffold for varlen-packed-attention.
// Summary: Handle packed variable-length Q/K/V batches without padding all sequences to the maximum length.
// Optimized strategy: Use sequence-offset metadata to map CTAs only to live tiles and avoid work on padding while preserving online-softmax correctness.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void varlen_packed_attention_optimized_kernel() {
  // TODO: implement the optimized HIP path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  varlen_packed_attention_optimized_kernel<<<1, 1, 0, stream>>>();
}

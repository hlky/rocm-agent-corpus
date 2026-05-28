#include <hip/hip_runtime.h>

// Template-only scaffold for varlen-packed-attention.
// Summary: Handle packed variable-length Q/K/V batches without padding all sequences to the maximum length.
// Baseline strategy: Pad variable-length sequences to a dense batch and run the existing online attention seed or framework SDPA reference.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void varlen_packed_attention_baseline_kernel() {
  // TODO: implement the baseline HIP path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  varlen_packed_attention_baseline_kernel<<<1, 1, 0, stream>>>();
}

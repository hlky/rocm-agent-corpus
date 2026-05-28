#include <hip/hip_runtime.h>

// Template-only scaffold for radix-sort-key-value.
// Summary: Cover key/value radix sort as a hipCUB/rocThrust competitor and as a reusable primitive for routing, sparse metadata, and sampling.
// Baseline strategy: Use hipCUB DeviceRadixSort or rocThrust sort_by_key with explicit temp-storage and allocation timing policy.
// Replace the placeholder kernel with the operation-specific implementation
// before recording correctness or timing evidence.

namespace {

__global__ void radix_sort_key_value_baseline_kernel() {
  // TODO: implement the baseline HIP path and preserve this task's contract.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  radix_sort_key_value_baseline_kernel<<<1, 1, 0, stream>>>();
}

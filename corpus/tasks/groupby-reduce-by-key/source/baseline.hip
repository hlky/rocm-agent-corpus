#include <hip/hip_runtime.h>

// Template-only scaffold for groupby-reduce-by-key.
// Summary: Group sorted or unsorted key/value pairs and reduce values per key while exposing skew, preprocessing, and temporary-storage costs.
// Baseline strategy: Use rocThrust/hipCUB reduce_by_key on sorted input or radix-sort keys before grouped reduction for unsorted input.

namespace {

__global__ void groupby_reduce_by_key_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  groupby_reduce_by_key_baseline_kernel<<<1, 1, 0, stream>>>();
}

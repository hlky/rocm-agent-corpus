#include <hip/hip_runtime.h>

// Template-only scaffold for wide-k-topk-selection.
// Baseline strategy: use the strongest same-contract library path. For sorted
// output, start with CUB DeviceRadixSort or Thrust sort-by-key. For unsorted
// membership, start with CUB DeviceTopK. For framework parity, measure
// PyTorch torch.topk at the same timing boundary.

namespace {

__global__ void wide_k_topk_selection_baseline_kernel() {
  // TODO: implement or invoke the selected library baseline from a harness.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  wide_k_topk_selection_baseline_kernel<<<1, 1, 0, stream>>>();
}

#include <hip/hip_runtime.h>

// Template-only scaffold for reduce-scatter-overlap.
// Summary: Overlap chunked local compute with RCCL reduce-scatter to model tensor/sequence-parallel training communication boundaries.
// Baseline strategy: Run local compute over the full buffer, then RCCL ReduceScatter on a communication stream after compute completes.

namespace {

__global__ void reduce_scatter_overlap_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  reduce_scatter_overlap_baseline_kernel<<<1, 1, 0, stream>>>();
}

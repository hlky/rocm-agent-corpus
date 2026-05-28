#include <hip/hip_runtime.h>

// Template-only scaffold for allgather-overlap.
// Summary: Overlap all-gather of remote shards with local compute over already available tensor-parallel blocks.
// Baseline strategy: Run RCCL AllGather to materialize the full tensor, then launch the consumer compute kernel.

namespace {

__global__ void allgather_overlap_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  allgather_overlap_baseline_kernel<<<1, 1, 0, stream>>>();
}

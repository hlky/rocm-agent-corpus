#include <hip/hip_runtime.h>

// Template-only scaffold for allgather-overlap.
// Summary: Overlap all-gather of remote shards with local compute over already available tensor-parallel blocks.
// Optimized strategy: Pipeline shard arrival with consumer compute using stream events and partitioned output buffers.

namespace {

__global__ void allgather_overlap_optimized_kernel() {
  // TODO: replace with the optimized implementation before measuring.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  allgather_overlap_optimized_kernel<<<1, 1, 0, stream>>>();
}

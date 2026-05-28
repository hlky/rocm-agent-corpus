#include <hip/hip_runtime.h>

// Template-only scaffold for reduce-scatter-overlap.
// Summary: Overlap chunked local compute with RCCL reduce-scatter to model tensor/sequence-parallel training communication boundaries.
// Optimized strategy: Chunk work and issue ReduceScatter as soon as each chunk is ready while later chunks continue computing.

namespace {

__global__ void reduce_scatter_overlap_optimized_kernel() {
  // TODO: replace with the optimized implementation before measuring.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  reduce_scatter_overlap_optimized_kernel<<<1, 1, 0, stream>>>();
}

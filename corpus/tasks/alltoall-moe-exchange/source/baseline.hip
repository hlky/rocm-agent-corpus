#include <hip/hip_runtime.h>

// Template-only scaffold for alltoall-moe-exchange.
// Summary: Model MoE token permutation, rank-to-rank exchange, and unpermutation with RCCL/rocSHMEM comparison boundaries.
// Baseline strategy: Build send counts, permute tokens into rank-major buffers, run RCCL send/recv or all-to-all equivalent, then unpermute after expert compute.

namespace {

__global__ void alltoall_moe_exchange_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  alltoall_moe_exchange_baseline_kernel<<<1, 1, 0, stream>>>();
}

#include <hip/hip_runtime.h>

// Template-only scaffold for block-scaled-fp4-gemm.
// Summary: Define the low-precision block-scaled FP4/NVFP4 GEMM task boundary, including packing, scale metadata, accumulator, and architecture feature checks.
// Optimized strategy: Use block-scale metadata in the matmul pipeline, avoid materialized dequantization, and specialize for supported Matrix Core paths.

namespace {

__global__ void block_scaled_fp4_gemm_optimized_kernel() {
  // TODO: replace with the optimized implementation before measuring.
}

}  // namespace

extern "C" void launch_optimized(hipStream_t stream) {
  block_scaled_fp4_gemm_optimized_kernel<<<1, 1, 0, stream>>>();
}

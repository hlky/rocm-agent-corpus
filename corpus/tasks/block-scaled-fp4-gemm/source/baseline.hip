#include <hip/hip_runtime.h>

// Template-only scaffold for block-scaled-fp4-gemm.
// Summary: Define the low-precision block-scaled FP4/NVFP4 GEMM task boundary, including packing, scale metadata, accumulator, and architecture feature checks.
// Baseline strategy: Materialize dequantized FP16/BF16 operands or call the closest Transformer Engine on ROCm/Composable Kernel/hipBLASLt path when available.

namespace {

__global__ void block_scaled_fp4_gemm_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  block_scaled_fp4_gemm_baseline_kernel<<<1, 1, 0, stream>>>();
}

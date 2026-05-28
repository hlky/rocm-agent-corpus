// Template-only baseline sketch.
//
// Fill this file with a real hipBLASLt Matmul runner and, when available, a
// Composable Kernel runner using the same row-major contract as custom_epilogue_kernel.hip.
// Required metadata:
// - hipBLASLt compute type, scale type, math mode, workspace bytes.
// - Matrix layout descriptors and row-major-to-column-major mapping.
// - Epilogue attributes, especially bias pointer and ReLU/GELU mode.
// - Heuristic count, selected algorithm, and whether each algorithm was timed.
// - Composable Kernel CTA tile, warp tile, instruction tile, stages, scheduler, and
//   epilogue operator.

#include <hip/hip_runtime.h>

extern "C" void launch_library_epilogue_baseline(const float*,
                                                  const float*,
                                                  const float*,
                                                  float*,
                                                  int,
                                                  int,
                                                  int,
                                                  hipStream_t) {
  // Intentionally empty: this is a compile-oriented placeholder until the
  // benchmark links against hipBLASLt and/or Composable Kernel.
}

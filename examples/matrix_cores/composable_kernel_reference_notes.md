# Composable Kernel and CK Tile Reference Notes

Composable Kernel and CK Tile are the strongest local reference material for serious Tensor
Core kernels. Use them as competitors, correctness references, and parts bins.
Using Composable Kernel still counts as custom-kernel work when the agent records and
modifies the specialization surface.

## What To Record

- Composable Kernel commit or submodule state.
- Example ancestor under `third_party/composable-kernel/examples`.
- Target GFX and `Composable Kernel_NVCC_ARCHS`.
- Element, accumulator, compute, scale, and output types.
- Layouts and leading dimensions.
- CTA tile, warp tile, instruction tile, and cluster shape.
- Pipeline stages and copy atom.
- Shared-memory layout, padding, and swizzle.
- Scheduler or persistent strategy.
- Epilogue operator, visitor, vector width, and broadcast axes.
- Workspace size and initialization cost.

## Where To Look

- `third_party/composable-kernel/examples`: GEMM, fused epilogue, grouped GEMM, CDNA3,
  and CDNA4/RDNA4 examples.
- `third_party/composable-kernel/media/docs/cpp/efficient_gemm.md`: mainloop and
  hierarchy concepts.
- `third_party/composable-kernel/media/docs/cpp/gemm_api_3x.md`: Composable Kernel 3.x GEMM model.
- `third_party/composable-kernel/media/docs/cpp/ck_tile/`: CK Tile layout, tensors, copy atoms,
  and MFMA atoms.
- `third_party/cuda-samples/cpp/3_CUDA_Features/`: rocWMMA teaching kernels for
  FP16, BF16, TF32, INT8, and FP64 where available.

## FP8 Boundary

For FP8 or block-scaled low precision, start with library baselines and Composable Kernel
examples before writing raw HIP. Record the FP8 format, scale layout, amax
policy, accumulator type, output type, and any fast-accumulation mode. Do not
mix FP8 and FP16 records unless the task is explicitly comparing numerical
contracts.

## Epilogue Boundary

A custom Matrix Core mainloop that only beats plain hipBLAS/rocBLAS is not enough when
hipBLASLt supports the same epilogue. Compare against:

- hipBLASLt with the closest supported epilogue.
- Composable Kernel with a custom epilogue visitor or output operator.
- Framework or MIGraphX plugin paths when deployment overhead matters.

If Composable Kernel wins, keep the record. The useful result is the boundary: which tile,
schedule, or epilogue surface was hard to beat, and which narrower custom path
remains plausible.

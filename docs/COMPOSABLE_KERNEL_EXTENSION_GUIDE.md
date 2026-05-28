# Composable Kernel and CK Tile Extension Guide

Composable Kernel is the path when an agent needs custom high-performance GEMM-like
kernels that libraries cannot express directly.

## Mental Model

Modern Composable Kernel kernels are assembled from:

- Problem shape: M, N, K, batch/group dimensions.
- Data types: element, accumulator, compute, scale types.
- Layouts: row-major, column-major, interleaved, tensor layouts.
- Tile shapes: threadblock/CTA, warp, instruction/MFMA atom.
- Mainloop: global-to-shared pipeline, async copy/global-to-LDS staging, MFMA scheduling.
- Epilogue: accumulator transformation, bias, activation, stores.
- Swizzle/scheduler: how CTAs map to output tiles.

## When to Reach for Composable Kernel

- GEMM with a custom epilogue not supported by hipBLASLt.
- Fused scale/bias/activation/dropout/residual paths.
- Research kernels for new layouts or datatypes.
- Architecture-specific Matrix Core experiments.
- Grouped GEMM or shape-specialized kernels.

## Custom Epilogues

Common epilogue needs:

- Linear combination: `D = alpha * accum + beta * C`.
- Bias: add per-row, per-column, or per-element bias.
- Activation: ReLU, GELU, SiLU, clamp.
- Residual: combine with another tensor.
- Quantization: scale, clamp, cast to int8/fp8-like output.
- Auxiliary outputs: store pre-activation or amax.

Agent workflow:

1. Find the closest Composable Kernel example under `third_party/composable-kernel/examples`.
2. Identify the epilogue type and output operator.
3. Change one axis at a time: datatype, layout, tile shape, epilogue.
4. Add a small host reference and benchmark before tuning.
5. Record architecture, `Composable Kernel_NVCC_ARCHS`, tile shape, stages, and schedule.

## CK Tile Concepts

CK Tile expresses layouts and tiled MFMA/copy operations as composable algebraic
objects. Agents should look for:

- Shape: logical tensor dimensions.
- Stride: memory layout mapping.
- Layout composition: tiling and swizzling.
- Copy atom: how data moves.
- MFMA atom: hardware instruction abstraction.
- Tiled copy/MFMA: how atoms are repeated across a tile.

## Tuning Axes

- CTA tile shape.
- Warp tile shape.
- Instruction tile/MFMA atom.
- Pipeline stages.
- Cluster shape on CDNA3+.
- Shared-memory layout and swizzle.
- Epilogue vector length.
- Split-K or stream-K strategy.
- Persistent scheduling.

## Failure Modes

- Correct output for one layout but wrong leading dimensions generally.
- Epilogue broadcasts along the wrong axis.
- Workspace or alignment assumptions are hidden in the benchmark.
- Register pressure makes a larger tile slower.
- Shared-memory use reduces occupancy enough to lose.
- Comparing against hipBLAS/rocBLAS with different math mode or TF32 setting.

## Example Locations

- `third_party/composable-kernel/examples`
- `third_party/composable-kernel/media/docs`
- `examples/composable-kernel/custom_epilogue_skeleton.hip`

## Corpus Tasks to Add

- Composable Kernel GEMM versus hipBLAS/rocBLAS baseline.
- Composable Kernel GEMM + custom bias + activation epilogue.
- Composable Kernel grouped GEMM for varied M/N/K.
- Composable Kernel int8 or FP8 quantized output epilogue.
- CK Tile educational tiled copy and tiled MFMA examples.


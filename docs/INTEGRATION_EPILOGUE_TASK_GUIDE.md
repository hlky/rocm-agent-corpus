# Integration and Epilogue Task Guide

This guide covers the template-only promotion path for three integration tasks:

- `gemm-bias-activation-epilogue`
- `migraphx-custom-op-fused-op`
- `pytorch-hip-extension-op`

These tasks are not measured claims. Label them `template-only` until a harness
builds, runs correctness checks, and records HIP-event timing with hardware and
build metadata. Say `timing-only` for future HIP-event results without rocprofiler/rocprof
counter evidence.

## GEMM Epilogue Fusion

Treat hipBLASLt and Composable Kernel as mandatory competitors for GEMM with bias and
activation epilogues. A custom HIP kernel can only claim a useful win after the
task records:

- row-major or column-major layout mapping
- math mode and compute type
- workspace bytes
- hipBLASLt heuristic count and selected algorithm
- Composable Kernel tile shape, stage count, scheduler, and epilogue operator
- exact bias broadcast axis and activation semantics

Useful custom hypotheses include fixed shapes, narrow layouts, cheaper launch
boundaries, or unsupported epilogues that would otherwise require materialized
intermediates.

## MIGraphX Plugin Fused Op

MIGraphX plugin tasks should document the whole plugin lifecycle, not just the
HIP kernel:

- shape inference
- datatype and format support
- workspace size
- serialization fields and versioning
- plugin creator fields and namespace
- enqueue behavior on the MIGraphX-provided stream
- engine-build flags, precision, profile, and MIGraphX version

Do not call a plugin faster than MIGraphX unless engine build, runtime shape,
precision, and timing boundary are identical to the baseline being compared.

## Framework Extension Op

Framework extensions are integration tasks as much as kernel tasks. Record:

- framework version
- dtype, device, sizes, strides, and contiguity assumptions
- current-stream source and synchronization boundary
- whether Python dispatch, extension launch, compiler warmup, or graph capture
  is included in timing
- eager, compiler-generated, Triton, and custom HIP baselines where practical

Reject unsupported tensors early or implement stride-aware indexing explicitly.
OneFlow source examples are useful for framework-internal HIP operator anatomy;
PyTorch extension examples are useful for fast external integration.

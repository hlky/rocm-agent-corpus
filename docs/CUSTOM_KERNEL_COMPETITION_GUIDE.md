# Custom Kernel Competition Guide

This corpus exists to help agents write custom HIP kernels that can beat,
match, specialize beyond, or intelligently extend vendor libraries.

That goal is intentionally ambitious. hipBLAS/rocBLAS, hipBLASLt, hipCUB, Composable Kernel, MIOpen,
MIGraphX, vLLM on ROCm, RCCL, and Triton are extremely strong. The corpus should
teach agents how to compete honestly anyway.

## Core Stance

Vendor libraries are:

- Baselines to beat.
- Correctness oracles.
- Architecture references.
- Sources of layout, datatype, and API constraints.
- Examples of what mature engineering looks like.
- Extension surfaces when direct HIP is not the best integration point.

Vendor libraries are not:

- A reason to stop thinking.
- A substitute for measuring the target workload.
- Proof that no narrower custom kernel can win.
- A replacement for understanding memory movement, scheduling, numerics, and
  architecture-specific features.

## What Counts as Winning

A custom kernel does not need to beat a vendor library for every shape and dtype.
It can win by being narrower.

Valid wins:

- Faster for one production shape or a small shape family.
- Faster after fusing two or more operations that the library runs separately.
- Lower latency because it avoids framework dispatch, allocation, or launch
  overhead.
- Lower memory traffic because it avoids materialized intermediates.
- Better for a known layout, alignment, mask, sparsity pattern, or batch shape.
- Equivalent speed with simpler integration or less workspace.
- Better numerical behavior for the required tolerance.
- Better total serving metric such as time-to-first-token, inter-token latency,
  or memory footprint, even if one inner primitive is slower.

Valid losses:

- The custom kernel is slower, but the record explains why.
- The custom kernel wins only for tiny shapes and loses for large shapes.
- The attempted trick, such as vectorized loads, is neutral or slower.
- The vendor library exposes a tactic or extension point that should be used as
  the next custom path.

## Competition Loop

1. Define the scoped workload.
   - Shapes, dtype, layout, strides, alignment, masks, sparsity, aliasing.
   - Target GPU and exact gfx target.
   - Correctness tolerance and numerical contract.
   - Whether timing includes framework dispatch, allocation, graph capture, or
     engine build.

2. Establish the strongest baseline.
   - hipBLAS/rocBLAS/hipBLASLt for GEMM-like work.
   - hipCUB/rocPRIM/hipCUB/rocThrust for reductions, scans, sorts, selections.
   - MIOpen/frontend for neural network primitives.
   - MIGraphX/vLLM on ROCm for deployment inference.
   - Triton or framework compiler output for Python-authored ML kernels.
   - Composable Kernel for custom Matrix Core kernels and epilogue experiments.

3. Identify what the baseline must handle that your kernel can ignore.
   - Arbitrary shapes.
   - Arbitrary strides or layouts.
   - Many dtypes.
   - Dynamic shapes.
   - Full generality around alpha/beta, transposes, epilogues, masks, or
     workspaces.
   - Multi-architecture portability.

4. Pick the attack surface.
   - Fusion.
   - Shape specialization.
   - Layout specialization.
   - Vectorization and alignment.
   - Shared-memory tiling.
   - Warp/block-level reductions.
   - Persistent scheduling.
   - Matrix Core MFMA, MFMA, global-to-LDS staging, or `global-to-LDS staging`.
   - Custom epilogue.
   - Runtime integration: MIGraphX plugin, PyTorch extension, OneFlow op,
     HIP Graph, or Triton kernel.

5. Measure against the baseline.
   - Same inputs.
   - Same output contract.
   - Same dtype and math mode.
   - Same warmup and measurement discipline.
   - Median and spread, not best-of timing.
   - Hardware and compile flags recorded.

6. Curate the result.
   - Win, loss, or neutral.
   - Why the result happened.
   - Which narrower or broader shape should be tried next.
   - Which profiler counters would confirm the hypothesis when available.

## Library-Specific Competition Notes

## hipBLAS/rocBLAS and hipBLASLt

Baseline role:

- Correctness and performance bar for GEMM and batched GEMM.
- Reference for math mode, leading dimensions, alpha/beta semantics, and
  workspace-aware algorithm choice.

Ways to compete:

- Fuse bias, activation, residual, quantization, or layout transform.
- Specialize fixed M/N/K and batch shapes.
- Remove framework overhead for tiny repeated GEMMs.
- Use Composable Kernel/CK Tile for custom tile shapes, Matrix Core paths, or epilogues.
- Compare TF32, FP16, BF16, FP8, INT8, and accumulator choices explicitly.

Required metadata:

- Transpose flags.
- Leading dimensions and layouts.
- Math mode and compute type.
- Workspace size.
- hipBLASLt algorithm or heuristic result where available.

## Composable Kernel and CK Tile

Baseline role:

- Reference implementation family for custom GEMM, Matrix Core use, tile
  scheduling, and epilogue engineering.

Ways to compete or extend:

- Custom epilogue visitor.
- Fixed-shape tile tuning.
- Grouped GEMM.
- Fused dequantization or quantization.
- Architecture-specific CDNA3/CDNA4/RDNA4 paths.
- Specialized layouts that are awkward for hipBLASLt.

Agent warning:

Using Composable Kernel is still custom-kernel work when the agent selects or modifies
tile shapes, layouts, epilogues, schedulers, datatypes, or architecture paths.
Record what changed and what library baseline it competes with.

## hipCUB, rocPRIM/hipCUB/rocThrust, and rocThrust

Baseline role:

- Generic high-performance primitives for reductions, scans, sorts, selection,
  and block/warp collectives.

Ways to compete:

- Fuse transform plus reduction into one pass.
- Exploit fixed segment sizes.
- Use warp-level specialization for small reductions.
- Privatize histograms or atomics differently for known distributions.
- Use hipCUB block primitives inside a larger custom kernel.

Agent warning:

Do not write raw shuffle trees for production unless they beat hipCUB or are needed
for education. But do write them in the corpus when they teach a concept.

## MIGraphX and vLLM on ROCm

Baseline role:

- Deployment-level inference bar, including engine build, tactic selection,
  plugins, precision, KV cache, batching, and serving metrics.

Ways to compete or extend:

- MIGraphX plugin for an unsupported fused op.
- Custom HIP kernel inside plugin `enqueue`.
- Better narrow-shape kernel than a generic plugin.
- Lower end-to-end latency through fusion or memory reuse.
- vLLM on ROCm plugin or kernel modification for attention, KV cache, or
  quantized inference.

Required metadata:

- MIGraphX version.
- Engine build flags.
- Dynamic shape profiles.
- Precision and calibration/quantization path.
- Timing cache.
- Runtime batch and sequence settings.

## Triton and Framework Kernels

Baseline role:

- Fast iteration baseline for Python-authored kernels.
- Common compiler-generated competitor for ML workloads.

Ways to compete:

- HIP C++ kernel with lower overhead or better architecture-specific code.
- Triton kernel with tuned meta-parameters.
- PyTorch/OneFlow extension that fuses framework-level operations.
- HIP Graph capture around repeated framework execution.

Required metadata:

- Compile time versus runtime.
- Meta-parameters.
- Framework version.
- Synchronization and dispatch boundary.
- Generated LLVM IR / AMD GCN ISA when making architecture claims.

## Anti-Patterns

- "hipBLAS/rocBLAS exists, so no custom GEMM examples are needed."
- "hipCUB exists, so reductions are solved."
- "MIGraphX supports this model, so no plugin study is needed."
- "Composable Kernel is too complex, so ignore it."
- "Triton is close enough, so HIP C++ is unnecessary."
- "The custom kernel lost, so the example is useless."

The corpus should preserve failed attempts when they teach why the vendor
baseline is strong or which specialization is still plausible.


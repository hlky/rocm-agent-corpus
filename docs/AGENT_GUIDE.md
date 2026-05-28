# Agent Guide

This guide describes how a HIP/ROCm optimization agent should use the corpus.

## Mission

Start from the goal of writing a custom HIP kernel that can beat, match, or
narrowly specialize beyond the strongest available library path for the exact
workload. Treat hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, hipCUB, MIGraphX, Triton, MIOpen, and
framework kernels as competitors to study, baselines to measure, correctness
oracles to trust, and extension surfaces to mine for ideas.

Do not stop at "use the library" unless measured evidence shows the custom path
cannot win for the stated shape, dtype, layout, tolerance, and GPU. When a
library wins, record why and name the next specialization that might close the
gap.

## Operating Loop

1. Establish the contract.
   - Identify input shapes, layout, dtype, aliasing, tolerances, and target GPU.
   - Decide whether the kernel is memory-bound, compute-bound, launch-bound, or
     dominated by synchronization/atomics.

2. Build a correct baseline.
   - Keep a simple reference path.
   - Add the strongest relevant library or framework path as a competitor when
     one exists.
   - Use deterministic seeds and awkward shapes.
   - Fail closed on out-of-bounds, aliasing, or tolerance issues.

3. Study the competitor.
   - Use vendor libraries as references for algorithm choice, tiling, data
     movement, epilogue fusion, numerics, and architecture assumptions.
   - Identify what generality the library keeps that this workload can drop:
     dynamic shapes, arbitrary strides, broad dtype support, unfused epilogues,
     conservative tolerances, runtime dispatch, or portability across GFXs.
   - If using Composable Kernel, hipCUB, MIGraphX, Triton, or hipBLASLt directly, document the
     customization surface and how a custom kernel would extend or beat it.

4. Form one optimization hypothesis at a time.
   - State the bottleneck.
   - State the code change.
   - State the expected timing and profiler signal.

5. Measure.
   - Use HIP events for kernel timing.
   - Use repeated iterations and report median plus spread.
   - When rocprofiler/rocprof counters are available, collect the narrowest useful
     section first.

6. Curate.
   - Record what improved, what did not, and what may regress.
   - Mark records as `timing-only` when profiler counters are unavailable.
   - Never claim counter evidence that is not present in the artifact.

## Retrieval Hints

Prefer tasks by bottleneck:

- Global-memory inefficiency: `coalescing`, `vectorized-load`, `alignment`.
- Strided stores: `shared-memory`, `tiling`, `bank-conflict`.
- Atomics: `reduction`, `atomic-contention`, `block-reduction`.
- Per-row operations: `softmax`, `row-reduction`, `numerical-stability`.
- Occupancy problems: `register-pressure`, `shared-memory`, `launch-config`.

Prefer records by evidence quality:

1. `counter-backed-measured`
2. `timing-only-measured`
3. `correctness-only`
4. `template-only`

## Response Discipline

When proposing an optimization, an agent should include:

- The expected bottleneck.
- The exact code change.
- The target shapes and GPU.
- The library or framework competitor and what generality the custom kernel
  drops.
- Correctness risks.
- Measurement plan.
- Evidence status after running.

Good answer shape:

```text
Hypothesis: stores are strided, so memory sectors per request should be high.
Change: stage a tile in shared memory and write rows contiguously.
Risk: shared-memory bank conflicts; pad the second dimension by 1.
Evidence: ROCm timing-only rerun pending; attach HIP-event median and spread after running on AMD hardware.
Counter status: rocprofiler/rocprof counters not collected yet.
```

## Library Use

Library docs in this corpus are self-contained maps for understanding and
extending competitors, not permission to abandon the custom path:

- Use hipBLAS/rocBLAS/hipBLASLt to set GEMM and epilogue baselines, then look for fixed
  shapes, fused work, layout constraints, or precision choices that a custom
  kernel can exploit.
- Use Composable Kernel to inspect Matrix Core pipelines, CK Tile layouts, tile shapes, and
  epilogue structure; copy ideas only after reducing them to the workload.
- Use hipCUB/rocPRIM/hipCUB/rocThrust for block and warp primitives, reductions, scans, and iterator
  patterns that can live inside a custom kernel.
- Use MIGraphX and vLLM on ROCm to understand inference engine behavior, plugin
  boundaries, fusion opportunities, KV-cache layouts, and deployment
  constraints.
- Use Triton as a competitor and rapid kernel sketchpad; port or
  out-specialize it in HIP C++ when control over layout, instructions, or
  integration matters.

## Common Agent Mistakes

- Optimizing before verifying correctness.
- Reporting best-of timing instead of median and spread.
- Comparing kernels with different math or tolerance.
- Overfitting to powers-of-two shapes.
- Accepting a library result as unbeatable without measuring the scoped custom
  opportunity.
- Using occupancy as a goal instead of a diagnostic.
- Adding shared memory without proving reuse or coalescing benefit.
- Ignoring setup costs for tiny kernels.
- Claiming Nsight evidence when only HIP-event timing exists.

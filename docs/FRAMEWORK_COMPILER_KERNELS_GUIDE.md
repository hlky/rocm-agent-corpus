# Framework Compiler Generated Kernels Guide

Framework compilers are first-class competitors for custom HIP kernels. nvFuser,
TorchInductor/Triton, XLA/Pallas, OneFlow fusion paths, and similar systems can
remove launches, specialize shapes, and emit strong GPU code. Use them as
baselines and generated-code references, not as a reason to stop.

## Competitor Roles

- nvFuser: fusion compiler for PyTorch/JIT-style tensor expressions; useful for
  elementwise, normalization, reductions, and pointwise-heavy chains.
- TorchInductor/Triton: common PyTorch 2.x compiler path; often emits Triton
  kernels and can outperform naive CUDA extensions by removing dispatch and
  materialized intermediates.
- XLA and Pallas: JAX/TensorFlow compiler paths; strong for whole-graph shape
  specialization, fusion, and SPMD lowering.
- OneFlow: framework operator/fusion/runtime path with distributed placement
  concerns.
- Triton handwritten kernels: both a fast custom-kernel route and the likely
  output language for some framework compiler paths.

## Extraction Checklist

Before writing HIP C++, extract the generated-kernel contract:

1. Shapes and guards: static sizes, dynamic guards, buckets, symbolic dims.
2. Layouts: contiguous assumptions, strides, channels-last, packed formats,
   alignment, vector width.
3. Fusion group: producer ops, consumer ops, reductions, epilogues, broadcasts,
   masks, and materialized intermediates.
4. Numerics: dtype promotion, accumulator type, fast math, TF32 permission,
   reductions order, tolerance.
5. Runtime boundary: compile time, warmup, graph capture, dispatch overhead,
   memory allocation, synchronization.
6. Generated code: Triton source, LLVM IR / AMD GCN ISA, AMD GCN ISA, launch parameters, register count,
   shared memory, spills when available.
7. Fallbacks: unsupported dynamic shape, unsupported dtype/layout, graph break,
   custom call, library call, or eager fallback.

Only claim architecture behavior when backed by generated LLVM IR / AMD GCN ISA or profiler
counters. Otherwise use `timing-only`.

## Ways Custom CUDA Can Compete

- Accept fewer shapes or one production bucket while the compiler keeps guards.
- Fuse across a graph break, custom op, layout conversion, or library boundary.
- Use architecture-specific features the compiler does not emit for this case:
  `global-to-LDS staging`, global-to-LDS staging, WGMFMA, inline MFMA, warp-specialized reductions, persistent
  CTAs, or custom memory ordering.
- Reduce register pressure or shared-memory traffic after inspecting generated
  code.
- Avoid compiler warmup for short-lived jobs or serving cold paths.
- Implement a custom epilogue or packed low-bit layout the compiler materializes
  through intermediates.

## When the Compiler Probably Wins

Record a negative example instead of forcing a rewrite when:

- The generated fusion already removes all large intermediates.
- The bottleneck is memory bandwidth and the generated kernel coalesces well.
- Dynamic shape guards are cheap compared with kernel work.
- The compiler emits a tuned Triton kernel close to the roofline for the target
  contract.
- A custom HIP extension loses after framework dispatch and synchronization are
  included.

Useful losses should name the compiler path, generated-kernel boundary, and the
next narrower specialization worth trying.

## Baseline Ladder

For framework workloads, compare in this order when feasible:

1. Eager/framework native implementation.
2. Framework compiler output, such as nvFuser, Inductor/Triton, XLA/Pallas, or
   OneFlow fusion.
3. Handwritten Triton or library baseline.
4. CUDA extension or standalone HIP kernel.
5. Integration path: MIGraphX plugin, OneFlow op, PyTorch extension, HIP Graph,
   or custom call.

The winner depends on the timing boundary. Kernel-only CUDA may win while
end-to-end framework timing loses; keep both numbers and label them clearly.

## Corpus Record Fields

```text
Framework/compiler:
Framework version and flags:
GPU/CUDA/compiler:
Graph/function:
Shapes/dtypes/layouts:
Fusion group:
Generated code artifact:
Baseline timing boundary:
Custom kernel boundary:
Evidence label:
Result:
Negative-example reason, if any:
Next narrower specialization:
```


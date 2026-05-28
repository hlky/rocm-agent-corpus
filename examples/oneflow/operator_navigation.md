# OneFlow Operator Navigation

Use `third_party/oneflow` as a framework-internal reference for HIP operator
integration, distributed placement, and runtime scheduling. The goal is not to
copy OneFlow blindly; it is to learn the framework contract a custom HIP op
must satisfy and where a narrower kernel can compete.

## What to Extract

Start from the user-visible operator and trace inward:

1. Op schema and registration: attributes, tensor arity, dtype/device rules.
2. Shape inference: static dimensions, symbolic dimensions, broadcasting, and
   error paths.
3. SBP/distributed placement: split, broadcast, partial-sum behavior, and any
   required collectives or layout conversions.
4. Kernel dispatch: CPU versus HIP registration, dtype specialization,
   architecture guards, and stream access.
5. HIP implementation: grid/block policy, vectorization, temporary storage,
   atomics, cooperative groups, library calls, and launch count.
6. Autograd path: backward op registration and saved tensors.
7. Fusion/scheduling: whether the framework can fuse neighboring ops or avoid
   materialized intermediates.

Record exact files and symbols when producing corpus notes. If the upstream
code changed, keep the commit or submodule state attached to the claim.

## Custom Kernel Questions

Ask these before proposing a replacement:

- Which shapes, dtypes, layouts, and distributed placements are actually needed?
- Does OneFlow support arbitrary strides/layouts that the custom kernel can
  reject?
- Does SBP introduce communication or copies that dominate kernel time?
- Is the current HIP op a library wrapper, a generated kernel, or handwritten
  HIP?
- Can the win come from fusion, fixed segment sizes, known alignment, or fewer
  launches rather than a faster standalone primitive?
- Does the timing include Python/framework dispatch, graph build, memory
  planning, or synchronization?

## Negative Examples Worth Keeping

Preserve losing attempts when they explain the boundary:

- A custom op is faster kernel-only but slower once framework dispatch and
  layout conversion are included.
- OneFlow fusion removes the intermediate the custom op expected to save.
- SBP communication dominates the local HIP kernel improvement.
- A generic OneFlow operator wins because it uses a stronger library tactic.
- A narrow kernel wins for one placement but fails for split/broadcast variants.

Label HIP-event-only results as `timing-only`. Label losing but valid attempts
as `negative example`.

## Corpus Note Template

```text
Operator:
Upstream files/symbols:
Submodule commit:
Shapes/dtypes/layouts:
SBP/placement:
Baseline path:
Custom boundary:
Fusion boundary:
Timing boundary:
Evidence label:
Result and next specialization:
```

## Useful Submodule Areas

- Operator schema and registration paths.
- HIP kernel registration and implementation paths.
- SBP and distributed placement code.
- Runtime scheduling, stream, and memory-planning code.
- Fusion/compiler integration paths.

# Memory Movement Kernel Guide

This guide tracks template-only HIP C++ tasks for memory-movement kernels that
agents can compile, validate, and later benchmark. These are scaffolds, not
measured wins. Do not claim profiler-counter evidence until rocprofiler/rocprof artifacts
exist; HIP-event-only measurements must be labeled timing-only.

## Tasks

| Task | Focus | Baseline | Optimized scaffold | Status |
| --- | --- | --- | --- | --- |
| `gather-scatter-coalescing` | Irregular index traffic versus coalesced identity traffic | Scalar gather/scatter through an index map | `float4` coalesced path for identity mode plus irregular fallback | template-only |
| `shared-halo-stencil-2d` | Reusing neighbor cells in a 2D stencil | Direct global-memory five-point stencil | Shared-memory tile with one-cell halo | template-only |
| `aos-soa-layout-conversion` | Layout conversion and vectorized tuple loads | Scalar deinterleave from AoS to SoA | One `float4` load per tuple with coalesced SoA stores | template-only |

## Benchmarking Discipline

- Build each harness twice, once with `VARIANT_BASELINE` and once with
  `VARIANT_OPTIMIZED`, linking the matching task source file.
- Preserve the mode/shape in result names, especially irregular versus coalesced
  gather/scatter runs.
- Record GPU, driver, ROCm toolkit, compiler, arch flags, warmup iterations, and
  measurement iterations before making any measured claim.
- Use rocprofiler/rocprof memory metrics when available for sector/request and shared
  memory conflict claims. If counters are unavailable, say timing-only.
- Keep library and framework paths as baselines or references, not escape
  hatches. For these scoped tasks, custom kernels can win by exploiting fixed
  layout, known alignment, identity maps, halo reuse, or reduced generality.

## Compile Sketches

```bash
hipcc -O3 -std=c++17 -DVARIANT_BASELINE harnesses/gather_scatter_benchmark.hip corpus/tasks/gather-scatter-coalescing/source/baseline.hip -o gather_baseline
hipcc -O3 -std=c++17 -DVARIANT_OPTIMIZED harnesses/gather_scatter_benchmark.hip corpus/tasks/gather-scatter-coalescing/source/optimized.hip -o gather_optimized

hipcc -O3 -std=c++17 -DVARIANT_BASELINE harnesses/stencil2d_benchmark.hip corpus/tasks/shared-halo-stencil-2d/source/baseline.hip -o stencil_baseline
hipcc -O3 -std=c++17 -DVARIANT_OPTIMIZED harnesses/stencil2d_benchmark.hip corpus/tasks/shared-halo-stencil-2d/source/optimized.hip -o stencil_optimized

hipcc -O3 -std=c++17 -DVARIANT_BASELINE harnesses/aos_soa_benchmark.hip corpus/tasks/aos-soa-layout-conversion/source/baseline.hip -o aos_soa_baseline
hipcc -O3 -std=c++17 -DVARIANT_OPTIMIZED harnesses/aos_soa_benchmark.hip corpus/tasks/aos-soa-layout-conversion/source/optimized.hip -o aos_soa_optimized
```

## Next Measurements

Start with correctness and timing-only runs across recommended shapes in each
`task.json`. For gather/scatter, capture both irregular and coalesced modes
because they teach different boundaries: an optimized identity-map path can be
fast, while arbitrary permutations usually need either preprocessing,
locality-aware data layout, or acceptance that the index stream dominates.

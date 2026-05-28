# System And Low-Level Kernel Guide

This track covers optimization surfaces where the fastest answer is often not a
single arithmetic kernel: launch overhead, stream overlap, runtime
specialization, persistent scheduling, warp collectives, async copy, and
architecture-specific Matrix Core paths.

## Evidence Rules

- Mark scaffolds and unrun examples as `template-only`.
- Say `timing-only` for HIP-event or wall-clock measurements without profiler
  counters.
- Keep graph instantiation, runtime compilation, MIGraphX engine build, and
  steady-state kernel replay as separate timing regions.
- Attach GPU, driver, toolkit, compiler, GFX flags, and library versions to every
  measured claim.
- Do not claim WGMFMA, global-to-LDS staging, `global-to-LDS staging`, or inline `mfma` benefits until the
  source path and target GFX are explicit.

## Task Boundaries

HIP Graphs are for repeated launch-bound workflows. Compare ordinary launches,
graph capture/instantiation cost, and graph replay separately.

Stream overlap tasks should record pinned host memory, chunk size, stream count,
copy engine capability, and whether timing covers H2D, kernel, and D2H.

hipRTC/nvJitLink specialization tasks should separate compile latency from cache
hit execution. Cache keys need dtype, shape, architecture, and options.

Persistent work queues are scheduling experiments. Keep negative examples where
uniform work loses to queue atomics.

Warp reduce/scan/vote tasks are low-level education and fusion building blocks.
Compare against hipCUB/rocPRIM/hipCUB/rocThrust before making production claims.

`global-to-LDS staging`, WGMFMA, global-to-LDS staging, and inline MFMA tasks are architecture-specific. Start
from the CUDA Programming Guide, LLVM IR / AMD GCN ISA ISA, and Composable Kernel examples, then record the
exact instruction shape, pipeline stages, and fallback path.

MIGraphX and Triton tasks are competitor and extension surfaces. A library path
can remain faster, but the corpus should record how an agent could specialize,
fuse, or replace the relevant boundary next.

# HIP/ROCm Agent Corpus Instructions

This repository is an internal HIP/ROCm optimization reference, guide, and corpus for agents.

## Start Here

1. Read `docs/RETRIEVAL_MAP.md`.
2. Read the relevant local skill under `.codex/skills/`.
3. Check `docs/RESULTS_SUMMARY.md` for measured examples.
4. Read `docs/CUSTOM_KERNEL_COMPETITION_GUIDE.md` before accepting a library result as unbeatable.
5. Use `third_party/` submodules for real upstream examples after pinning commits and checking licenses.

## Evidence Discipline

- Say `timing-only` when results are HIP-event timings without rocprofiler/rocprof counters.
- Say `negative example` when the attempted optimization did not improve speed.
- Do not invent profiler-counter evidence.
- Keep hardware, ROCm version, compiler flags, and gfx target metadata attached to every measured claim.
- Treat CUDA-origin results from the source corpus as historical context only, not ROCm evidence.

## Custom Kernel Discipline

- The corpus goal is to help agents write custom HIP kernels that beat, match, or narrowly specialize beyond hipBLASLt/rocBLAS, Composable Kernel, rocPRIM/hipCUB/rocThrust, MIOpen, MIGraphX, Triton, and framework-generated kernels.
- Vendor libraries are baselines, reference implementations, correctness oracles, profiling targets, and extension surfaces.
- A custom kernel can win by exploiting fixed shapes, fusion, layout knowledge, relaxed generality, lower launch overhead, custom epilogues, special numerics, architecture-specific instructions, or avoiding framework/runtime overhead.
- If a library remains faster, record why.

## Validation

```bash
python tools/validate_corpus.py
python tools/summarize_results.py
```

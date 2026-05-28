---
name: rocm-corpus-navigator
description: Navigate this HIP/ROCm Agent Corpus to find relevant tasks, guides, submodules, examples, and evidence records.
---

# ROCm Corpus Navigator

Use this skill when working inside this repository and the task is to answer,
implement, or evaluate a HIP/ROCm optimization.

## Workflow

1. Read `docs/RETRIEVAL_MAP.md`.
2. Read `docs/CUSTOM_KERNEL_COMPETITION_GUIDE.md`.
3. Classify the problem using `docs/CASE_CATALOG.md` and check
   `docs/CASE_COVERAGE_PLAN.md` for the current coverage status. If the
   problem is not in the v1 tables, inspect `Extended Case Families` and
   `data/index/case_catalog.json` before inventing a new category.
4. For promoted v2 cases, read `docs/V2_EXTENSION_PROMOTION_WAVE.md` and
   `data/index/v2_extension_wave.json`.
5. Find measured examples under `corpus/tasks`.
6. Prefer `data/records/*` entries with measured evidence.
7. Read the relevant track guide: memory movement, sparse/irregular,
   system/low-level, GEMM, attention, selection/sampling, quantization, Tensor
   Core, framework, or MIGraphX.
8. Inspect upstream code through `third_party/*` when library behavior matters.
9. Report evidence level explicitly: counter-backed, timing-only, negative,
   template-only, or planned.

## Do Not

- Claim rocprofiler/rocprof counter evidence unless a profile artifact contains it.
- Treat a timing-only record as architecture-general truth.
- Treat a vendor library as the final answer when the user is asking for custom
  kernel optimization.
- Skip library baseline checks for GEMM, reductions, scans, sorts, inference,
  FFT, sparse, or solver work. The baseline is the opponent and oracle.
- Mark a `v2-proposed` extension case as complete. Promote it only after a
  task, harness plan, library baseline list, and evidence target exist.

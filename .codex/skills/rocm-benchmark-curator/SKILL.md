---
name: rocm-benchmark-curator
description: Add ROCm timing, correctness, profiler, and curation records without mixing evidence levels.
---

# HIP/ROCm Benchmark Curator

Use this skill when adding or reviewing measured HIP/ROCm records.

## Workflow

1. Read `docs/BENCHMARKING_GUIDE.md` and `docs/INGESTION.md`.
2. Run the task-specific benchmark script.
3. Store result JSONs under `corpus/tasks/<task>/results/`.
4. Store environment JSON under `data/hardware/`.
5. Add a JSONL record under `data/records/`.
6. Update `task.json` artifacts, verification, benchmarking, and curation status.
7. Run `python tools/validate_corpus.py`.

## Evidence Rules

- Use `timing-only-measured` when only HIP-event timings are available.
- Use `timing-only-measured-negative` for neutral or slower attempted
  optimizations.
- Use `profile-attempted-blocked` when rocprofiler/rocprof fails with a permissions
  error such as `ERR_NVGPUCTRPERM`.


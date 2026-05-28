# Ingestion Guide

## 1. Add a Source

Add source metadata under `data/sources/*.json`. Prefer stable URLs and pinned
commits for repositories.

Required fields:

- `id`
- `name`
- `url`
- `kind`
- `license`
- `use_policy`
- `priority`
- `topics`
- `observed_at`

Use `use_policy` to separate metadata-only sources from code that can be copied
or transformed under its upstream license.

## 2. Add a Task

Create a directory under `corpus/tasks/<task-id>/` with:

```text
task.json
source/baseline.hip
source/optimized.hip
```

Optional but recommended:

```text
harness/
profiles/
results/
notes.md
```

Use `python tools/new_task.py ...` for a starter layout.

## 3. Benchmark

For HIP C++ kernels, prefer a harness that reports:

- warmup iterations
- measured iterations
- median, p10, p90, min, and max latency
- input shapes
- tolerance and error summary
- GPU and toolkit metadata

Store results as JSON in the task's `results/` directory.

For the seed matrix tasks, use:

```powershell
python tools/run_matrix_task.py <task-id> baseline --rows 4096 --cols 4096 --write-result
python tools/run_matrix_task.py <task-id> optimized --rows 4096 --cols 4096 --write-result
```

## 4. Profile

rocprofiler/rocprof should be run with enough metrics to support the optimization
claim. Examples:

```powershell
rocprof --set full --print-details=all --target-processes all .\benchmark.exe
rocprof --section MemoryWorkloadAnalysis_Tables --print-details=all .\benchmark.exe
```

When saving profiler output, keep raw `.rocprof-rep` files out of git by default and
store a text or JSON summary in `profiles/`.

## 5. Curate

Each measured optimization needs a short curation note:

- Why the baseline is slow.
- What changed.
- Which profiler counters support the claim.
- Which shapes or architectures were tested.
- Which cases may regress.

## 6. Validate

Run:

```powershell
python tools/validate_corpus.py
```

Validation checks metadata shape and verifies that declared artifacts exist.

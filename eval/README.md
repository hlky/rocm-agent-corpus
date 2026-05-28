# Eval Harness

This directory contains the scaffold for evaluating custom HIP kernel
submissions against vendor and framework baselines.

Status: scaffolded manifests only. The planned commands are documented, but no
executable harness implementation is present yet.

## Layout

| Path | Purpose |
| --- | --- |
| `tasks/` | Task contracts, shape sets, baseline definitions, and task manifests. |
| `submissions/` | Submitted custom kernels and submission metadata. |
| `reports/` | Attempt archives, comparisons, and generated summaries. |
| `harness/` | Planned compile, test, benchmark, compare, archive, and report tooling. |

## Operating Loop

1. Select a task from `eval/tasks/`.
2. Add or point to a custom-kernel submission under `eval/submissions/`.
3. Compile the submission for the target architecture.
4. Run correctness checks against the task oracle.
5. Benchmark the submission and the strongest applicable baseline.
6. Classify the result as win, match, specialized-win, loss, neutral,
   negative-example, correctness-fail, compile-fail, or measurement-invalid.
7. Archive the attempt under `eval/reports/`.

See `docs/AGENT_EVAL_HARNESS.md` for the staged implementation plan.

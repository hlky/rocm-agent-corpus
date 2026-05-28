# Eval Harness Tooling

This directory is reserved for the executable harness implementation.

Initial status: manifest-only. Commands below are planned, not implemented.

## Planned Commands

```bash
python eval/harness/compile.py --task <task_id> --submission <submission_dir>
python eval/harness/test.py --task <task_id> --submission <submission_dir>
python eval/harness/baseline.py --task <task_id> --baseline <baseline_id>
python eval/harness/benchmark.py --task <task_id> --submission <submission_dir> --baseline <baseline_id>
python eval/harness/compare.py --attempt <attempt_json> --baseline <baseline_json>
python eval/harness/archive_attempt.py --attempt <attempt_json> --output eval/reports/
python eval/harness/report.py --all
```

## Implementation Notes

- Emit JSON for every step so attempts can be archived even after failures.
- Keep compile, correctness, benchmark, comparison, and archive steps separable.
- Make vendor baselines explicit and reproducible.
- Preserve evidence labels. Do not upgrade timing-only results to
  counter-backed results without profiler counters.

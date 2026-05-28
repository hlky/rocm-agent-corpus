# Eval Tasks

Task definitions live here. A task describes the operation, input/output
contract, visible and hidden shape policy, correctness tolerance, allowed
shortcuts, baseline competitors, and expected optimization surfaces.

Initial status: manifest-only.

Future task directories should use:

```text
eval/tasks/<task_id>/
  task.json
  README.md
  visible_shapes.json
  baselines.json
  oracle/
```

Every task should make clear how a custom kernel could beat, match, or
specialize beyond a vendor or framework baseline.

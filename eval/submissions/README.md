# Eval Submissions

Submission records live here. A submission is a custom HIP implementation for
one eval task, plus metadata describing the optimization hypothesis and known
constraints.

Initial status: manifest-only.

Future submission directories should use:

```text
eval/submissions/<submission_id>/
  submission.json
  src/
  build/
  notes.md
```

Submissions should state which generality they intentionally drop. Examples:
fixed shape, fixed dtype, contiguous layout, no arbitrary stride support,
architecture-specific instructions, or fused epilogue semantics.

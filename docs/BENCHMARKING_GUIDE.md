# Benchmarking Guide

The corpus should be useful to agents because measurements are reproducible and
clearly scoped.

## Required Timing Fields

Every result JSON should include:

- `task_id`
- `variant`
- `operation`
- shape fields such as `n`, `rows`, `cols`
- `warmup_iters`
- `measure_iters`
- `median_ms`
- `p10_ms`
- `p90_ms`
- `min_ms`
- `max_ms`
- correctness summary
- build command
- hardware record id or file

## Timing Rules

- Use HIP events around the measured GPU work.
- Warm up before measuring.
- Synchronize on the stop event, not the whole device unless necessary.
- Keep allocation and host-to-device setup outside kernel timing unless the task
  explicitly includes end-to-end cost.
- Include required reset kernels or memset operations when they are part of the
  algorithm contract.
- Report medians. Do not use only the fastest run.

## Correctness Rules

- Test non-power-of-two sizes where possible.
- State tolerance and numerical risks.
- For reductions, account for valid floating-point reorder differences.
- For softmax, check row sums and max absolute error against a stable reference.
- Use guard conditions for boundary tiles.

## Profiler Evidence Levels

`counter-backed-measured`:
rocprofiler/rocprof counters are available and the record links to summary artifacts.

`timing-only`:
HIP-event timing and correctness are available, but profiler counters are not.
This is still useful, but claims must stay tied to timing.

`profile-attempted-blocked`:
Profiler execution was attempted and blocked by host permissions, missing
counter support, unavailable profiler tools, or runtime/container restrictions.

`unmeasured`:
The example is a template or static code comparison only.

## Shape Sweeps

Each task should eventually include:

- One large power-of-two shape.
- One awkward non-power-of-two shape.
- One small shape where launch overhead may dominate.
- One shape near an alignment or tile boundary.

## Hardware Notes

Always capture:

```bash
rocm-smi
hipcc --version
rocprof --version
```

Use `tools/collect_env.py` to store this in `data/hardware/`.


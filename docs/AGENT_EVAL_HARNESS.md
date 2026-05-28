# Agent Eval Harness

This document describes the staged evaluation harness for agents that write
custom HIP kernels. The harness is intended to compile, test, benchmark,
compare, classify, and archive kernel attempts against strong vendor and
framework baselines.

The goal is not to tell agents to delegate to hipBLAS/rocBLAS, Composable Kernel, hipCUB, MIGraphX,
Triton, or framework-generated kernels. The goal is to make those systems the
competition: baselines, correctness oracles, profiling targets, reference
surfaces, and extension surfaces.

## Harness Mission

The harness should answer five questions for every submitted kernel:

1. Does it compile for the requested ROCm GFX target and HIP toolchain?
2. Does it pass correctness checks across visible and hidden shapes?
3. Is it faster, slower, or neutral against the relevant baseline?
4. Which generality did the custom kernel drop to get its result?
5. Is the evidence timing-only, counter-backed, blocked, or template-only?

Every result should preserve the task, submission, hardware, build flags,
library baseline, correctness tolerance, timing method, and classification.

## Directory Layout

```text
eval/
  README.md
  tasks/
    README.md
    task_manifest.json
  submissions/
    README.md
    submission_manifest.json
  reports/
    README.md
    report_manifest.json
  harness/
    README.md
    harness_manifest.json
data/index/eval_harness.json
```

The manifests are intentionally small and stable. Future stages can add Python
or CMake implementation files under `eval/harness/` without changing the
top-level contract.

## Staged Build-Out

### Stage 0: Scaffold

Status: scaffolded.

Create documentation, manifests, directory structure, result taxonomy, and
planned commands. No benchmarks are run at this stage.

Outputs:

- `docs/AGENT_EVAL_HARNESS.md`
- `eval/README.md`
- `eval/*/*_manifest.json`
- `data/index/eval_harness.json`

### Stage 1: Compile And Correctness Loop

Status: planned.

Implement a local harness that accepts a task id and submission directory,
builds the submitted HIP/C++ code, runs visible correctness tests, and emits a
machine-readable attempt record.

Planned commands:

```bash
python eval/harness/compile.py --task <task_id> --submission <submission_dir>
python eval/harness/test.py --task <task_id> --submission <submission_dir>
```

Required classification states:

- `compile-pass`
- `compile-fail`
- `correctness-pass`
- `correctness-fail`
- `unsupported-configuration`

### Stage 2: Vendor Baseline Comparator

Status: planned.

Add task-specific baseline runners for hipBLAS/rocBLAS/hipBLASLt, Composable Kernel, hipCUB/rocPRIM/hipCUB/rocThrust,
MIGraphX, Triton, MIOpen, or framework-generated code. A task may have multiple
baselines when more than one competitor is relevant.

Planned command:

```bash
python eval/harness/baseline.py --task <task_id> --baseline <baseline_id>
```

Baseline rules:

- Treat libraries as competitors and oracles, not as escape hatches.
- Store the library version, exact API path, flags, math mode, dtype, layout,
  shape set, and timing boundary.
- If a baseline has broader generality than the submitted kernel, document the
  dropped generality so specialized custom wins are interpretable.

### Stage 3: Benchmark Sweep

Status: planned.

Run timed sweeps across visible, hidden, and stress shapes. Store per-shape
statistics and aggregate result classifications.

Planned command:

```bash
python eval/harness/benchmark.py --task <task_id> --submission <submission_dir> --baseline <baseline_id>
```

Default evidence label for HIP-event timing without rocprofiler/rocprof counters:

- `timing-only-measured`

If rocprofiler/rocprof counters are attempted but blocked by permissions:

- `profile-attempted-blocked`

Do not invent counter evidence.

### Stage 4: Comparison And Classification

Status: planned.

Compare the submission against the strongest applicable baseline. Classify the
attempt by correctness and speedup.

Planned command:

```bash
python eval/harness/compare.py --attempt <attempt_json> --baseline <baseline_json>
```

Recommended result classes:

- `win`: correct and faster than the best applicable baseline.
- `match`: correct and within the configured equivalence band.
- `specialized-win`: correct and faster after intentionally dropping
  documented generality.
- `loss`: correct but slower than the baseline.
- `neutral`: correct and not materially different.
- `negative-example`: valid attempted optimization that regressed or did not
  improve.
- `correctness-fail`: compiled but produced invalid output.
- `compile-fail`: did not build.
- `measurement-invalid`: timing setup is invalid or incomplete.

### Stage 5: Attempt Archive

Status: planned.

Archive every meaningful attempt, including losing attempts, so agents can
learn where custom kernels do and do not beat the competition.

Planned command:

```bash
python eval/harness/archive_attempt.py --attempt <attempt_json> --output eval/reports/
```

Archive records should include:

- task id and task version
- submitted files and hashes
- build command and compiler output summary
- GPU, driver, ROCm toolkit, and architecture target
- correctness test summary
- baseline id and baseline metadata
- timing results and evidence label
- result classification
- notes explaining the optimization hypothesis
- next recommended attack surface

### Stage 6: Remote GPU Runner

Status: planned.

Add a remote runner for benchmark jobs on disposable GPU machines. This stage
should support cheap GPUs for compile/test loops and selected stronger GPUs for
architecture-specific comparisons. Long-running benchmarks should remain
explicitly requested.

Planned command:

```bash
python eval/harness/remote_run.py --task <task_id> --submission <submission_dir> --gpu-class <class>
```

Remote records must preserve the same metadata as local records.

### Stage 7: Regression And Leaderboards

Status: planned.

Produce reports that show per-task winners, losses, regressions, and
architecture-specific boundaries.

Planned commands:

```bash
python eval/harness/report.py --task <task_id>
python eval/harness/report.py --all
```

Reports should avoid one-dimensional leaderboards. A slower kernel can still be
valuable if it teaches a boundary, exposes a correctness trap, or shows why a
vendor library is hard to beat.

## Task Contract

Each eval task should define:

- `task_id`
- `task_version`
- operation semantics
- input and output tensors
- dtype, layout, alignment, and stride rules
- visible shapes
- hidden shape policy
- tolerance policy
- allowed headers and libraries
- disallowed shortcuts
- baseline competitors
- expected optimization surfaces
- architecture notes

Tasks should include both general tasks and specialized tasks. Specialized
tasks should explicitly state which generality may be dropped.

## Submission Contract

Each submission should include:

- source files
- optional build metadata
- declared target architectures
- optimization hypothesis
- known limitations
- expected task id and version

The harness should treat undeclared assumptions as suspicious but not
automatically invalid unless they break the task contract.

## Baseline Contract

A baseline should be strong enough to be meaningful. For example:

- GEMM: hipBLASLt and/or Composable Kernel, with appropriate math mode and epilogue.
- Reduction/scan/sort: hipCUB or rocPRIM/hipCUB/rocThrust primitives.
- Inference plugin task: MIGraphX or vLLM on ROCm plugin/runtime path.
- Attention: FlashAttention, MIOpen attention, vLLM on ROCm, or Triton where
  relevant.
- Framework op: PyTorch, OneFlow, Triton, or generated fused kernel competitor.

The harness should record when a baseline is unavailable instead of silently
substituting a weak baseline.

## Result Record Sketch

```json
{
  "attempt_id": "example-task__submission-name__20260520T120000Z",
  "task_id": "example-task",
  "task_version": "0.1.0",
  "submission_id": "submission-name",
  "hardware_id": "runpod-a4000-example",
  "toolchain": {
    "rocm": "version-required",
    "driver_runtime": "version-required",
    "compiler": "hipcc",
    "arch": "gfx1030"
  },
  "correctness": {
    "status": "correctness-pass",
    "max_abs_error": 0.0,
    "max_rel_error": 0.0
  },
  "timing": {
    "evidence_label": "timing-only-measured",
    "submission_ms": 0.0,
    "baseline_ms": 0.0,
    "speedup_vs_baseline": 1.0
  },
  "classification": "match",
  "notes": "Template record only; no measured claim."
}
```

## Evidence Discipline

- Use `timing-only-measured` for HIP-event timings without profiler counters.
- Use `counter-backed-measured` only when profiler counters are present.
- Use `profile-attempted-blocked` when rocprofiler/rocprof was attempted but blocked.
- Use `template-only` for scaffold records that contain no measured result.
- Use `negative-example` for correct attempts that are slower or neutral but
  educational.

## First Implementation Targets

The first executable harness should focus on tasks where custom kernels can
plausibly win through specialization:

- small and fixed-shape GEMM variants
- rowwise softmax and masked softmax
- RMSNorm and LayerNorm
- reductions and segmented reductions
- top-k and argmax
- quantize/dequantize and fused epilogues
- KV-cache update/read kernels
- simple attention tiles

These targets should be paired with strong library baselines so wins and losses
are meaningful.

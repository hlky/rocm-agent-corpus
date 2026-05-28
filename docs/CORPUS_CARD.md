# Corpus Card

## Purpose

HIP/ROCm Agent Corpus is intended to support agents that generate, repair, explain,
and optimize HIP kernels. The corpus emphasizes grounded optimization: a model
should learn to connect code changes to correctness tests, profiler counters,
hardware constraints, compiler behavior, and benchmark evidence.

## Composition

The scaffold contains four data classes:

- Source metadata: official docs, open-source repositories, lectures, benchmark
  suites, and papers.
- Tasks: problem statements with baseline and optimized artifacts.
- Optimization records: before/after changes, expected or measured effects, and
  reasoning.
- Profiles and environment records: hardware, driver, ROCm toolkit, compiler
  flags, benchmark settings, and profiler summaries.

## Intended Uses

- Retrieval for CUDA coding agents.
- Supervised examples of optimization reasoning.
- Evaluation tasks for code generation and repair.
- Regression tests for benchmark harnesses and verification logic.
- Analysis of failed or misleading optimization attempts.

## Out-of-Scope Uses

- Treating unmeasured examples as performance ground truth.
- Training on copied third-party documentation without license review.
- Comparing GPUs without normalized hardware and driver metadata.
- Using single-shape speedups as evidence of general kernel quality.

## Collection Process

Every measured item should include:

- Target GPU, driver, ROCm toolkit, compiler, OS, and clock/power state when
  available.
- Full build command and launch configuration.
- Correctness test settings, seeds, tolerances, and input shapes.
- Baseline and optimized timing distributions, not just a best run.
- rocprofiler/rocprof metrics or a clear reason profiling is unavailable.
- A curation note explaining why the optimization is valid.

## Licensing

This repository's original scaffold and seed examples are original project
content. External source manifests record upstream licenses and retrieval
policies. Before importing substantial upstream code or generated derivatives,
pin the source revision and preserve license notices.

## Known Limitations

The seed tasks are not benchmarked in this workspace. They are examples of the
record format and should not be used as measured performance data until they are
run and profiled on a CUDA-capable system.


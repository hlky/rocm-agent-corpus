# Multi-GPU Task Guide

This guide covers template-only multi-GPU task seeds. They are intended to
preserve optimization hypotheses even when the current machine has no RCCL or
rocSHMEM-capable multi-GPU setup.

## Evidence Policy

- Say `template-only` for these scaffolds until a multi-GPU harness runs.
- Say `timing-only` for HIP-event timings without Nsight Systems, Nsight
  Compute, or transport-level evidence.
- Say `negative example` when chunking, GPU-initiated enqueue, or another
  attempted optimization loses.
- Do not claim communication overlap from source structure alone. Attach a
  timeline, event timing breakdown, or profiler-counter evidence before making
  an overlap claim.

## RCCL AllReduce Overlap

Use `corpus/tasks/rccl-overlap-allreduce` to compare a phase-separated local
compute plus allreduce baseline with a chunked pipeline. RCCL remains the
collective baseline and correctness oracle. The custom-kernel opportunity is in
the surrounding schedule: fixed chunks, stream dependencies, fused local
compute, reduced framework launch overhead, or shape-specific communication
granularity.

Future records should include rank count, bytes per rank, chunk size, stream
count, RCCL version, topology, relevant RCCL environment variables, GPU model,
driver, ROCm toolkit, build flags, timer type, warmups, iterations, and
correctness tolerance.

## rocSHMEM Queue

Use `corpus/tasks/rocshmem-queue` to compare a host-mediated queue with a
GPU-initiated one-sided enqueue. rocSHMEM is the communication primitive and
reference implementation surface; the custom work is queue layout, ordering,
overflow handling, batching, and the kernel that decides when remote work is
published.

Future records should include PE count, transport, queue capacity, entry size,
rocSHMEM version, symmetric allocation details, memory-ordering policy, timer
type, warmups, iterations, and whether the measurement includes queue draining.

## Validation Without Hardware

It is acceptable for these sources to be guarded by `ENABLE_RCCL_TEMPLATE` or
`ENABLE_rocSHMEM_TEMPLATE`. A no-hardware validation pass should still run:

```bash
python tools/validate_corpus.py
python -m json.tool data/index/multigpu_task_track.json > NUL
```

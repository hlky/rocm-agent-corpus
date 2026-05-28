# HIP/ROCm Agent Corpus

A structured corpus for training, evaluating, and debugging agents that optimize HIP kernels and ROCm library paths.

Each corpus item should connect a problem statement, baseline code, optimized code, correctness checks, benchmark results, hardware metadata, profiler evidence, and the reasoning that explains why a change helped. This repository was ported from the CUDA agent corpus, but CUDA-origin timing artifacts were intentionally removed. Re-run tasks on ROCm hardware before making performance claims.

## Quick Start

```powershell
python tools/validate_corpus.py
python tools/collect_env.py
```

Create a new task skeleton:

```powershell
python tools/new_task.py my-task "My HIP/ROCm optimization task" --domain memory --tags coalescing,tiling
```

On a ROCm machine, benchmark a seed matrix task:

```powershell
python tools/run_matrix_task.py memory-coalesced-matrix-copy baseline --rows 4096 --cols 4096 --arch gfx942 --write-result
python tools/run_matrix_task.py memory-coalesced-matrix-copy optimized --rows 4096 --cols 4096 --arch gfx942 --write-result
```

## Repository Layout

```text
architecture/             GFX/CDNA/RDNA-specific custom-kernel labs.
corpus/tasks/             Curated HIP optimization tasks and source artifacts.
data/index/               Machine-readable navigation and case maps.
data/records/             ROCm measurement records; currently empty pending reruns.
data/sources/             ROCm source manifests and retrieval pointers.
docs/                     Agent guides, track guides, benchmarking policy, and roadmaps.
examples/                 HIP, rocPRIM, hipBLASLt, rocWMMA, MIGraphX, and Triton sketches.
eval/                     Agent submission and evaluation harness scaffold.
harnesses/                Shared HIP benchmark harnesses.
schemas/                  JSON Schemas for corpus files.
third_party/              ROCm-oriented submodule candidates; not initialized by default.
tools/                    Validation, task scaffolding, benchmarking, and environment capture.
```

## Main Guides

- `docs/RETRIEVAL_MAP.md`: first stop for navigation.
- `docs/BENCHMARKING_GUIDE.md`: correctness, timing, and provenance rules.
- `docs/PYTORCH_BASELINE_CHALLENGE.md`: PyTorch-public-API challenge scaffold for ROCm.
- `docs/ROAD_TO_SOTA_TOPK.md`: Top-K roadmap, baselines, and evidence checklist.
- `docs/CUDA_ROCM_PARITY_NOTES_20260528.md`: first parity pass against the CUDA corpus.
- `docs/CUDA_ROCM_TASK_PARITY_MAP.md`: task-by-task CUDA to ROCm equivalence map.
- `docs/ROCM_PARITY_SCORECARD_20260528.md`: current task, harness, architecture, and evidence parity status.
- `docs/ARCHITECTURE_LABS.md`: ROCm `gfx` target labs and evidence gates.
- `tools/check_cuda_rocm_task_parity.py`: validates CUDA task coverage against ROCm equivalents.
- `docs/HIPBLAS_ROCBLAS_GUIDE.md`: hipBLAS/rocBLAS/hipBLASLt baselines.
- `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md`: Composable Kernel and CK Tile paths.
- `docs/ROCPRIM_HIPCUB_ROCTHRUST_GUIDE.md`: rocPRIM, hipCUB, and rocThrust primitives.
- `docs/MATRIX_CORE_ROCWMMA_GUIDE.md`: rocWMMA, MFMA, and Matrix Core work.
- `docs/MIGRAPHX_INFERENCE_GUIDE.md`: MIGraphX and inference integration.
- `docs/GPU_GFX_ARCHITECTURE_GUIDE.md`: GFX target and architecture-specific tuning.

## Seed Coverage

The corpus keeps the CUDA corpus task coverage but ports source artifacts and metadata to HIP/ROCm. Seed tasks include memory coalescing, tiled transpose, SAXPY, reductions, softmax, RMSNorm, scans, histograms, small GEMM, attention, top-k sampling, INT4 dequant GEMV, and rocWMMA MFMA GEMM.

Expansion scaffolds cover sparse/irregular workloads, HIP graphs, hipRTC specialization, persistent work queues, LDS-staged MFMA pipelines, inline MFMA, MIGraphX, PyTorch HIP extensions, RCCL overlap, rocSHMEM queues, real-model PyTorch baseline challenges, and wide-K Top-K experiments.

## Evidence Policy

- `timing-only` means HIP-event or framework timing without profiler counters.
- `counter-backed-measured` requires attached rocprofiler/rocprof counter artifacts.
- CUDA-origin measurements from the source corpus are not ROCm evidence.
- Library baselines are opponents and correctness oracles, not escape hatches.
- If a custom kernel loses to hipBLASLt, Composable Kernel, rocPRIM, MIOpen, MIGraphX, Triton, or framework code, record why; losing attempts are useful corpus data.

## Repo-Local Agent Skills

This repo includes local skill files under `.codex/skills/`:

- `rocm-corpus-navigator`
- `rocm-custom-kernel-competitor`
- `rocm-library-optimizer`
- `rocm-benchmark-curator`
- `rocm-architecture-specialist`

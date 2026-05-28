# CUDA to ROCm Parity Notes - 2026-05-28

This note records the first parity pass against `H:/cuda-agent-corpus`.

## Added ROCm Parity Areas

- PyTorch-public-API challenge docs, indexes, result schema, and manifest.
- Real-model problem-size and API shape-matrix references for challenge tasks.
- Top-K roadmap docs, implementation survey, baseline harness blueprint, wide-k
  matrix, and wild-idea backlog.
- Wave 3 ROCm task scaffolds for layout transforms, packed transcode, HIP memory
  pools, convolution variants, normalization backward, multi-tensor AdamW,
  Split-K GEMM, wide-K Top-K, HIP IPC, captured inference replay, MIOpen fusion,
  and vLLM on ROCm plugin boundaries.
- HIP ports of PyTorch/Triton baseline runners and selected harnesses, including
  memory-pool, normalization-backward, and hipCUB reduce/scan/histogram
  baseline harnesses.

## Evidence Boundary

No CUDA-origin result artifacts were copied. The added tasks are
`template-only` or harness-ready until ROCm runs attach hardware, ROCm version,
compiler flags, gfx target, correctness, timing, and profiler metadata.

When using the ported Top-K and PyTorch material, treat CUDA-specific source
paths as algorithm and contract references only. They are not ROCm performance
evidence.

## Follow-Up Parity Work

- Run `memory-pool-allocator` and `normalization-backward` on AMD hardware.
- Add same-hardware PyTorch, Triton, hipCUB, FlashInfer, vLLM on ROCm, and
  framework baseline records for Top-K and normalization tasks.
- Audit older docs that still mention CUDA-origin seed timings and update them
  only after ROCm records exist.

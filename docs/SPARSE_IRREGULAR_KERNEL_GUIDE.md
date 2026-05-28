# Sparse and Irregular Kernel Guide

This track covers reductions, graph frontiers, CSR sparse linear algebra,
embedding gathers, and block-sparse attention. The common issue is not only
arithmetic throughput; it is uneven work, indirect memory access, duplicate
work, and whether a custom kernel can exploit a narrower sparsity contract than
a general library primitive.

## Evidence Policy

- New tasks in this guide are `template-only` until a harness, correctness
  oracle, hardware metadata, and HIP-event timings are attached.
- Say `timing-only` for HIP-event results without Nsight counters.
- Say `negative example` when a custom load-balance, dedupe, or sparse traversal
  attempt loses to the baseline.
- Do not claim cuSPARSE, hipCUB, FlashAttention, Composable Kernel, vLLM, or FlashInfer wins
  from an isolated seed kernel without matching the input contract and timing
  boundary.

## Baseline Roles

- hipCUB/rocPRIM/hipCUB/rocThrust: segmented reductions, scans, select/compaction, sort, unique, and
  block or warp collectives.
- CUDA library samples: sparse library API patterns and correctness contracts
  for CSR-style operations.
- FlashAttention, Composable Kernel, vLLM, and FlashInfer: attention tiling, serving
  layouts, paged/block metadata, and stronger block-sparse competitors.

## Custom Kernel Angles

- Drop arbitrary-shape generality when segment size, row length distribution, or
  block sparsity pattern is fixed.
- Fuse predicate generation, scan, scatter, or epilogue work so the custom path
  moves less memory than a standalone library primitive chain.
- Use warp-per-row, CTA-per-row, or hybrid policies based on row length
  histograms rather than average nonzeros alone.
- Exploit duplicate ids only when the cost of sorting, uniquing, inverse mapping,
  and scatter is included or explicitly excluded.
- Preserve losing attempts when they show that a library primitive is already
  strong for a density, distribution, or stability contract.

## Seed Tasks

- `segmented-reduction-fixed-and-ragged`: fixed and offsets-based segmented sum.
- `csr-spmv-load-balance`: one-thread-per-row CSR SpMV versus warp-per-row seed.
- `embedding-gather-dedup`: direct gather versus pre-sorted unique gather and
  inverse scatter scaffold.
- `frontier-compaction-bfs`: global atomic compaction versus block scan/select.
- `block-sparse-attention-forward`: dense block-mask scan versus listed
  block-sparse online softmax skeleton.

## Metadata to Record Before Measuring

- GPU name, gfx target, driver, ROCm toolkit, compiler, build command,
  and architecture flags.
- Shape, dtype, layout, sparsity density, row or segment length histogram, and
  duplicate-id ratio when relevant.
- Library baseline version, temporary storage size, launch count, and whether
  allocation or preprocessing is included.
- Correctness oracle, tolerance, random seed, and output ordering contract.
- Profiler status. If counters are unavailable, keep the claim `timing-only`.

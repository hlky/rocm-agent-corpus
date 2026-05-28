# ML Adjunct Kernel Guide

This guide covers small ML and LLM-adjacent HIP kernels that sit around the
large GEMM and attention calls. They are often memory-bound, launch-bound, or
integration-bound, so custom kernels compete by fusing nearby work and dropping
generality that a framework or serving library must keep.

## Scope

- Optimizer updates such as AdamW, where separate elementwise launches can be
  fused into one pass over parameters, gradients, and optimizer state.
- Decode-side RoPE and KV-cache writes, where a custom kernel can combine
  rotation and cache placement for a known layout.
- MoE routing, where top-k selection, weighting, and expert scatter can be
  specialized for fixed expert counts and capacity policies.

## Baseline Discipline

Use libraries as references and competitors, not escape hatches. bitsandbytes
and Transformer Engine on ROCm are strong optimizer and transformer-training references.
vLLM on ROCm, vLLM, and FlashInfer are strong decode/KV-cache references.
rocPRIM/hipCUB/rocThrust/hipCUB block primitives are the baseline vocabulary for routing reductions,
scans, and selection.

Record `template-only` for scaffolds with no timing. Record `timing-only` only
when HIP-event timing exists and hardware/build metadata is attached. Do not
invent rocprofiler/rocprof counter evidence.

## Custom Kernel Angles

- Fuse materialized intermediates away: optimizer update buffers, rotated K
  buffers, or route metadata staging.
- Specialize layout: contiguous optimizer shards, non-paged KV cache, or fixed
  expert capacity.
- Specialize shape: fixed head dimension, fixed top-2 routing, aligned
  parameters, or small expert counts.
- Keep overflow and tie-breaking contracts explicit, especially for MoE routing.
- Treat library winners as useful boundaries. If a library remains faster,
  document whether it wins through better vectorization, stronger block
  primitives, serving integration, or a different memory layout.

## New Seed Tasks

- `corpus/tasks/fused-adamw-update`: multiple simple AdamW stages versus a
  fused scalar/vectorized update.
- `corpus/tasks/rope-kv-cache-update`: separate rotate and cache-store kernels
  versus a fused RoPE plus KV-cache write.
- `corpus/tasks/moe-token-routing`: per-token serial top-2 and atomic scatter
  versus a block-local top-2 routing scaffold.

# MIGraphX and vLLM on ROCm Inference Guide

MIGraphX belongs in this corpus as a deployment competitor, a correctness
oracle, and an extension surface. It is not a reason to stop looking for custom
CUDA wins. For inference work, the custom boundary may be a standalone kernel, a
MIGraphX plugin, a vLLM on ROCm plugin/kernel modification, or a narrower engine
configuration that exposes why a library tactic wins.

## Agent Stance

Treat MIGraphX as three things at once:

- Baseline: engine build, tactic selection, layer fusion, precision lowering,
  and runtime scheduling are the bar to beat.
- Reference: parser output, tensor formats, dynamic shape profiles, timing
  cache choices, and selected tactics reveal the deployment contract.
- Parts bin: plugins, custom kernels in `enqueue`, custom epilogues through
  surrounding libraries, and vLLM on ROCm kernels are places to specialize.

Do not conclude "MIGraphX supports the model" as the final answer. Conclude one
of:

- MIGraphX is faster for the measured contract; record the negative example and
  the tactic/plugin reason.
- A plugin is justified because MIGraphX cannot express the fused operation,
  layout, numeric contract, or shape specialization.
- A standalone custom kernel is justified because the measured bottleneck is
  outside MIGraphX's profitable fusion or tactic space.
- MIGraphX remains the deployment wrapper, but a custom kernel owns one narrow
  op boundary.

## MIGraphX Engine Boundary

Use MIGraphX when the target is deployment inference with ONNX import,
programmatic network construction, precision tuning, dynamic shapes, engine
serialization, or runtime execution contexts.

Record for every engine claim:

- MIGraphX version, ROCm version, driver, GPU, GFX target, host compiler.
- Model source: ONNX opset, framework export, or programmatic network.
- Builder flags: FP16, BF16, TF32, INT8, FP8 where supported, sparsity,
  deterministic or tactic-source restrictions.
- Optimization profiles: min/opt/max shapes for each dynamic input.
- Tensor formats and layout assumptions: `kLINEAR`, vectorized channel formats,
  NHWC/NCHW, packed low-bit layouts, alignment.
- Workspace limit, tactic sources, timing cache path, and whether the cache was
  warm or freshly built.
- Runtime boundary: enqueue-only latency, host preprocessing included, CUDA
  Graph capture included, or end-to-end serving latency.

The fair comparison is the same input contract, not the same source graph. A
custom kernel can win by accepting a fixed sequence length, a fixed batch bucket,
known strides, an always-present mask, or a fused epilogue that MIGraphX keeps
as separate layers.

## Plugin Boundary

Plugins are the MIGraphX extension point for operations or fusions that the
network definition cannot express profitably.

A plugin must define:

- Output shape inference.
- Supported dtype and tensor format combinations.
- Workspace bytes and temporary memory policy.
- Serialization and versioned plugin fields.
- Runtime resource initialization and teardown.
- `enqueue` behavior that launches CUDA work on the MIGraphX-provided stream.
- Correctness tests outside MIGraphX and engine-level tests inside MIGraphX.

Plugin candidates for this corpus:

- Fused bias, activation, residual, clamp, or quantize epilogue.
- LayerNorm/RMSNorm with fixed hidden size.
- Attention mask, RoPE, KV-cache write, or logits postprocess.
- Packed low-bit dequant plus narrow GEMV/GEMM.
- Shape-bucketed kernels where MIGraphX's generic tactic is too broad.

Plugin anti-patterns:

- Wrapping a single operation that MIGraphX already fuses with neighbors.
- Ignoring dynamic shape profiles and silently assuming the opt shape.
- Supporting only one dtype while the engine builder may select another.
- Allocating or synchronizing inside `enqueue`.
- Timing plugin kernel-only numbers as if they were engine latency.

## `enqueue` Checklist

Before writing the CUDA launch, extract:

1. Input/output tensor ranks, concrete runtime dimensions, and which dimensions
   may vary across profiles.
2. Strides or layout contracts. MIGraphX plugins often receive format metadata,
   not arbitrary framework-style strides.
3. Dtype and accumulator rules, including saturation, rounding, scale metadata,
   and tolerance.
4. Alignment and vectorization assumptions for each supported format.
5. Workspace needs and whether scratch can be avoided for the fixed shape.
6. Fusion boundary: which producers/consumers are adjacent in the engine and
   whether MIGraphX already removes the intermediate.
7. Stream ownership: use the provided `hipStream_t`; do not synchronize unless
   the test harness explicitly measures it.
8. Error policy: validate parameters before launch; use launch error checks in
   debug harnesses, not device-wide sync in production paths.

## vLLM on ROCm Boundary

vLLM on ROCm is the relevant competitor for transformer inference, especially
decode, batching, KV cache, quantization, and serving metrics.

Record:

- Model architecture, conversion path, checkpoint precision, and plugin set.
- Batch policy: static batch, in-flight batching, paged requests, beam width.
- Sequence contract: prompt length, generated tokens, max sequence length,
  causal/windowed attention, packed or padded inputs.
- KV cache policy: contiguous, paged, reuse, quantized cache, block size.
- Parallelism: tensor parallelism, pipeline parallelism, expert parallelism.
- Metrics separated by stage: build time, prefill latency, time-to-first-token,
  inter-token latency, tokens/sec, memory footprint, quality/tolerance.

Custom HIP opportunities usually live at boundaries vLLM on ROCm must keep
general:

- Fixed-head or fixed-page decode attention.
- Fused RoPE plus KV-cache write for a known layout.
- Logits processors, top-k/top-p, beam bookkeeping, and sampling.
- Quantized dequant, scale update, or FP8/FP4 cast paths for fixed group size.
- MoE routing/scatter when expert counts and token limits are known.

If vLLM on ROCm wins, record a negative example with the reason: better paging,
better persistent scheduling, stronger attention kernel, superior batching, or
less memory traffic. That loss is useful corpus data.

## Tactic and Fusion Investigation

Useful evidence to collect before proposing a custom kernel:

- Engine inspector output or equivalent layer/tactic summary.
- Whether a layer is fused, left standalone, or implemented by a plugin.
- Shape profile where the tactic was selected.
- Precision and tensor format selected for the bottleneck layer.
- Timing cache state and any tactic-source restrictions.
- Whether the bottleneck is compute, memory traffic, launch overhead, host
  scheduling, or serving queueing.

Do not invent profiler-counter evidence. If only HIP-event or MIGraphX timing
is available, label the result `timing-only`. If an attempted plugin or custom
kernel loses, label it a `negative example` and keep the contract metadata.

## Submodule Targets

- `third_party/migraphx`: plugin samples, parsers, engine/runtime patterns.
- `third_party/vllm-rocm`: transformer plugins, attention, KV cache,
  quantization, batching, serving competitors.
- `third_party/miopen-frontend`: fused neural-network plan comparisons.
- `third_party/flash-attention`, `third_party/flashinfer`, `third_party/vllm`:
  attention, paging, sampling, and serving baselines.
- `third_party/composable-kernel`: Matrix Core kernels and epilogue extension paths.

## Corpus Tasks to Add

- MIGraphX engine/tactic sweep with timing cache metadata.
- MIGraphX plugin wrapping a custom fused activation or normalization kernel.
- MIGraphX plugin negative example where native fusion already wins.
- vLLM on ROCm decode attention/KV-cache configuration notes.
- vLLM on ROCm throughput sweep by batch, prompt length, and generated tokens.
- Precision comparison across FP32, TF32, FP16, BF16, INT8, FP8/FP4 where
  supported, with explicit calibration or scale contracts.

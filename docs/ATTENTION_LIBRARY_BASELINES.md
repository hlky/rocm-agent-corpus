# Attention Library Baselines

This file defines library-baseline guidance for the
`online-attention-forward` task. No ROCm timing artifact is currently checked in;
the default seed shape remains a rerun target for
`batch=1,heads=4,query=128,key=128,head_dim=64,causal=1`. Do not treat
CUDA-origin timings as a FlashAttention, MIOpen, Triton, vLLM on ROCm, vLLM, or
FlashInfer win claim.

The goal is to make agents compare attention kernels honestly while still
looking for custom kernels that beat, match, specialize beyond, or extend the
library path.

## Library Stance

Libraries are competitors, correctness oracles, profiling targets, and source
references. They are not final answers for this corpus.

Use a library result to answer:

- What performance bar must the custom kernel beat?
- Which mask, dtype, layout, cache, and timing contract did the library solve?
- Which generality can a custom kernel drop?
- Which fusion, shape, query-position, KV-cache, or deployment boundary remains
  available for a narrower kernel?
- If the library wins, what boundary did it teach?

Do not stop at "use FlashAttention" or "use vLLM." Record the winning library,
why it won, and the next narrower custom hypothesis.

## Required Match Contract

Every baseline for `online-attention-forward` must state the following before
timings are compared:

- Operation: `O = softmax(scale * Q * K^T + mask) * V`.
- Task mode: prefill, decode, extend, or mixed prefill/decode.
- Shape: batch size, query heads, KV heads, query length, key length, head
  dimension, and value dimension.
- Dtype: Q, K, V, accumulator, output, and any KV-cache dtype.
- Layout: Q, K, V, output, and KV cache. The seed layout is contiguous
  `[batch, heads, sequence, head_dim]`.
- Scale: the seed uses `1 / sqrt(head_dim)`.
- Mask semantics: causal or noncausal, padding/ragged masks, sliding window,
  ALiBi/bias, attention sinks, and any all-masked-row behavior.
- Dropout: the measured seed has no dropout. Use `dropout=0` or
  `dropout_p=0.0` for inference comparisons.
- Query-position convention: the seed uses suffix-aligned causal positions:
  `absolute_query = query_index + max(0, key_length - query_length)`, visible
  when `key_index <= absolute_query`. Thus `query_length=1,key_length=N`
  models latest-token decode and can see the whole KV cache.
- Timing boundary: whether timing includes only the attention kernel, framework
  dispatch, graph capture, JIT/compile, engine build, plan creation, layout
  conversion, QKV packing, KV-cache update, scheduler work, host/device copies,
  or output validation.
- Evidence label: say `timing-only` for HIP-event timings without Nsight
  counters. Do not invent Matrix Core, occupancy, memory-transaction, or cache
  evidence.

If any item does not match, either adapt the inputs and include the adaptation
inside the timing boundary, or label the result as a non-equivalent baseline.

## Overall Ladder

Use this sequence before claiming that a custom attention kernel wins:

1. CPU stable softmax oracle with the exact mask and query-position convention.
2. Educational CUDA baseline from
   `corpus/tasks/online-attention-forward/source/baseline.hip`.
3. Online-tiled custom seed from
   `corpus/tasks/online-attention-forward/source/optimized.hip`.
4. Materialized GEMM plus softmax plus GEMM path.
5. Triton or framework SDPA with the backend and dispatch boundary recorded.
6. FlashAttention or MIOpen SDPA for dense prefill/fused attention.
7. vLLM on ROCm, vLLM, FlashInfer, or FlashAttention KV-cache paths for decode,
   paged KV, and serving-shaped batches.
8. Narrow custom HIP kernel or plugin that exploits a constraint above
   baselines must keep general.

## FlashAttention Ladder

Primary references:

- `third_party/flash-attention`
- `docs/ATTENTION_KERNEL_GUIDE.md`
- `docs/MIOPEN_ATTENTION_GUIDE.md`

Use FlashAttention first for dense prefill and fused attention workloads. Use
its KV-cache entrypoints separately for decode or extend workloads.

Concrete ladder:

1. `flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale,
   causal=causal)` for separate Q, K, V tensors.
2. `flash_attn_qkvpacked_func` only when the production workload already stores
   packed QKV, or when QKV packing cost is included in the baseline timing.
3. `flash_attn_varlen_func` when padding or ragged sequence lengths are part of
   the task contract.
4. `flash_attn_with_kvcache` for decode or extend, especially when the baseline
   updates K/V cache and attends in one kernel.
5. FlashAttention-3 or architecture-specific paths only when the target GPU and
   dtype support that path; record the version and commit.

Must match:

- Layout: FlashAttention commonly uses `[batch, sequence, heads, head_dim]`;
  the seed uses `[batch, heads, sequence, head_dim]`. Include transpose or
  packing cost unless the upstream workload naturally produces the library
  layout.
- Causal mask: require bottom-right or suffix-aligned behavior when
  `query_length != key_length`. That matches the seed convention and avoids
  fake decode differences.
- Dtype: compare FP32 seed results only against FP32-equivalent library results,
  or explicitly label an FP16/BF16/FP8 baseline as a changed numeric contract.
- Dropout: use `dropout_p=0.0`.
- Biases: disable ALiBi, local windows, rotary application, soft caps, and
  attention sinks unless the custom kernel implements the same feature.

Custom attack surface after FlashAttention:

- Fixed `batch,heads,query,key,head_dim` with no ragged or dropout support.
- Fused scale, mask, rotary, bias, or output-side work that would otherwise be
  extra launches.
- Decode-only latency when the KV layout and query position are fixed.
- A plugin or extension path that embeds the kernel where FlashAttention would
  need layout conversion.

If FlashAttention wins, record whether the reason is Matrix Core mainloops,
better QK/PV tiling, split-K/V decode, pipeline depth, or less register/shared
memory pressure. Without profiler artifacts, keep this as a timing-only
inference, not counter evidence.

## MIOpen SDPA Ladder

Primary references:

- `third_party/miopen-frontend`
- `docs/MIOPEN_ATTENTION_GUIDE.md`
- `docs/LIBRARY_GUIDE.md`

Use MIOpen for SDPA when the operation can be expressed as a MIOpen graph
or PyTorch-integrated MIOpen SDPA op.

Concrete ladder:

1. Build a MIOpen SDPA forward graph with Q, K, V, scale, mask, and
   output descriptors matching the task.
2. Generate operation plans and filter by dtype, compute type, workspace,
   determinism, dropout, mask behavior, and target GPU support.
3. Autotune or benchmark the viable engine configs, then record the selected
   engine, workspace bytes, and plan filters.
4. Compare against the custom HIP seed with the same device-resident inputs.
5. If no equivalent MIOpen plan exists for the GPU, dtype, mask, or layout,
   record `baseline-unavailable` instead of substituting a weaker path silently.

Must match:

- Dtype and accumulation policy, especially FP16/BF16/FP8 versus FP32.
- Tensor dimensions and strides. If MIOpen wants a different descriptor order,
  record whether the tensors are a view, a transpose, or a materialized copy.
- Causal mask and query-position semantics for `query_length != key_length`.
- Dropout disabled for the seed comparison.
- Timing boundary: plan creation and autotuning are setup costs; measure them
  separately from steady-state graph execution unless the production workload
  rebuilds plans at runtime.

Custom attack surface after MIOpen:

- A fixed-shape direct HIP path on GPUs or dtypes where no MIOpen engine is
  available.
- A custom graph/plugin boundary that fuses unsupported mask, bias, KV-cache, or
  output work.
- A narrower kernel that avoids workspace or descriptor-general layout support.

## Triton And Framework SDPA Ladder

Primary references:

- `third_party/triton/python/tutorials/06-fused-attention.py`
- `docs/TRITON_KERNEL_GUIDE.md`
- `docs/FRAMEWORK_EXTENSION_GUIDE.md`

Use this ladder to compare against compiler-generated and Python-authored
attention paths before moving to lower-level HIP C++.

Concrete ladder:

1. Framework eager SDPA, such as `scaled_dot_product_attention`, with backend
   selection recorded. Force or log whether the backend is flash, memory
   efficient, MIOpen, math, or another generated path.
2. Framework compiled SDPA, such as `torch.compile`, with compile time separated
   from steady-state runtime.
3. Triton fused attention from the local tutorial or a shape-specialized Triton
   kernel. Record block sizes, number of warps, stages, persistent or
   warp-specialized mode, and generated-code inspection if used.
4. CUDA graph captured framework or Triton path, if the production workload
   captures repeated iterations.
5. Custom HIP C++ only after the framework/Triton baseline and its dispatch
   boundary are clear.

Must match:

- Tensor layout and strides. PyTorch SDPA accepts `[batch, heads, query, dim]`
  style tensors; Triton tutorials may use different ordering.
- `is_causal`, explicit attention masks, scale, dropout, training/eval mode,
  and deterministic settings.
- Dtype and backend math mode, including TF32 allowances for FP32 framework
  paths.
- Timing boundary: exclude first-call JIT/compile when reporting steady-state
  kernel runtime, but record it as setup overhead. Include Python/framework
  dispatch when comparing end-to-end application latency.

Custom attack surface after Triton/framework:

- Lower dispatch overhead for small fixed shapes.
- Architecture-specific instructions or scheduling not exposed by the compiler.
- Fusion with surrounding CUDA work that would cross framework op boundaries.
- MIGraphX plugin or C++ extension integration where Python JIT is unsuitable.

## vLLM on ROCm, vLLM, And FlashInfer Decode Ladder

Primary references:

- `third_party/vllm-rocm`
- `third_party/vllm`
- `third_party/flashinfer`
- `docs/MIGRAPHX_INFERENCE_GUIDE.md`

Serving libraries are strongest for decode, paged KV, mixed batches, and
end-to-end inference metrics. They are not interchangeable with dense prefill
baselines unless the contract is reshaped to match.

Concrete ladder:

1. Isolated contiguous decode baseline for
   `query_length=1,key_length=N,causal=1`, matching the seed's latest-token
   query convention.
2. Paged-KV decode baseline with page or block size, block table, last-page
   length, KV layout, and cache dtype recorded.
3. Batched decode with realistic request composition: batch size, context length
   distribution, query heads, KV heads, grouped-query or multi-query mapping,
   and scheduler assumptions.
4. Mixed prefill/decode or extend benchmark when the production path interleaves
   prompt processing and generation.
5. End-to-end serving metric: time-to-first-token, inter-token latency,
   tokens/sec, memory footprint, and quality/tolerance checks.

vLLM on ROCm baseline checklist:

- Record engine build flags, vLLM on ROCm version, attention backend/plugin,
  paged KV settings, max tokens in paged KV cache, batch and beam settings,
  precision, quantization, and timing cache.
- Separate engine build time from runtime.
- Compare custom kernels through a plugin or kernel replacement boundary when
  possible, not only as a detached microkernel.

vLLM baseline checklist:

- Record backend selection, block size, KV-cache dtype, scheduler/batch
  composition, prefix-cache settings, and whether the benchmark is prefill,
  decode, extend, or mixed.
- Use vLLM attention benchmarks for isolated attention and serving benchmarks
  for end-to-end claims.

FlashInfer baseline checklist:

- Record wrapper type, such as `BatchDecodeWithPagedKVCacheWrapper` or
  `BatchPrefillWithPagedKVCacheWrapper`.
- Record `kv_layout`, page block size, workspace bytes, `plan` inputs,
  `use_tensor_cores`, Q dtype, KV dtype, number of query heads, number of KV
  heads, and context lengths.
- Separate planning/setup from `run` unless the workload replans at runtime.

Must match:

- Decode query position: the seed aligns `query_length=1` to the newest cache
  position. A top-left causal convention is not equivalent.
- KV layout and paging: contiguous B,H,S,D seed tensors are not the same memory
  problem as paged KV. Include conversion/cache-build cost when converting.
- Dtype and cache quantization: BF16, FP8, INT8, or FP4 KV cache is a changed
  numeric and bandwidth contract unless the custom kernel uses the same.
- Timing boundary: isolated attention kernel timings do not prove serving wins.
  Serving claims need scheduler, cache, batching, and token metric boundaries.

Custom attack surface after serving libraries:

- Fixed page size and fixed head dimension.
- Single-request or narrow-batch latency where general schedulers carry overhead.
- Custom cache layout, grouped-query mapping, rotary/bias fusion, or attention
  variant not covered by the serving stack.
- vLLM on ROCm plugin or vLLM/FlashInfer kernel replacement with the same
  batching boundary.

## GEMM Plus Softmax Plus GEMM Ladder

Primary references:

- `docs/GEMM_COMPETITION_TRACK.md`
- `docs/HIPBLAS_ROCBLAS_GUIDE.md`
- `docs/ROCPRIM_HIPCUB_ROCTHRUST_GUIDE.md`

This path materializes the attention matrix. It is usually not the strongest
long-sequence prefill implementation, but it is a valuable correctness oracle,
debug baseline, and explicit memory-traffic floor.

Concrete ladder:

1. Batched or strided QK GEMM:
   `scores[b,h,q,k] = scale * Q[b,h,q,d] * K[b,h,k,d]^T`.
2. Mask and softmax kernel over each `[b,h,q,:]` row. The mask must implement
   the seed suffix-aligned causal rule, or the result is not equivalent.
3. Batched or strided PV GEMM:
   `output[b,h,q,d] = P[b,h,q,k] * V[b,h,k,d]`.
4. Optional CUDA graph capture of the three-kernel sequence for repeated fixed
   shapes.
5. Optional hipBLASLt algorithm search for QK and PV separately, with workspace,
   compute type, and selected algorithm recorded.

Must match:

- The scale can be folded into QK GEMM alpha or applied in the softmax kernel,
  but record where it happens.
- The materialized score/probability dtype must be recorded. FP32 materialized
  scores are not the same as FP16/BF16 score storage.
- Include the score/probability allocation, initialization, and memory traffic
  if the production path would pay it.
- Count launches: QK GEMM, mask/softmax, PV GEMM, and any layout conversion or
  graph replay boundary.

Custom attack surface after materialized GEMM:

- Online softmax avoids the score/probability matrix.
- Fused mask, softmax, and PV reduce memory traffic and launches.
- Fixed shapes can keep row statistics and output accumulators in registers or
  shared memory.
- Matrix Core QK/PV tiles can be integrated without materializing `P`.

## Record Template

For each library result, attach this metadata to the measured record or a
baseline sidecar:

- Baseline id and version or commit.
- Entrypoint, backend, engine, plugin, or wrapper name.
- Hardware, driver, ROCm toolkit, library versions, compiler, and arch flags.
- Shape, dtype, layout, scale, mask, dropout, prefill/decode classification, and
  query-position convention.
- Warmup, measured iterations, timer type, synchronization points, and timing
  boundary.
- Setup costs measured separately: compile, autotune, plan, engine build, graph
  capture, packing, transpose, cache construction, and validation.
- Correctness oracle, tolerances, seed, max absolute error, max relative error,
  and all-masked-row behavior when applicable.
- Evidence label: `timing-only`, `counter-backed-measured`,
  `profile-attempted-blocked`, or `negative example`.
- Win/loss notes and the next custom-kernel hypothesis.

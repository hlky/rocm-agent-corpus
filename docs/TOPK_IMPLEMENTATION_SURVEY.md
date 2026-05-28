# Top-K Implementation Survey

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This survey maps how major CUDA and inference stacks implement Top-K-like
operations, then translates those implementations into custom-kernel attack
surfaces. It is a companion to `docs/ROAD_TO_SOTA_TOPK.md` and should be read
before claiming that a custom Top-K kernel beats PyTorch, hipCUB, vLLM on ROCm,
FlashInfer, vLLM, OneFlow, Transformer Engine, or Composable Kernel-derived paths.

Evidence status: source-inspection only. No new GPU timings or rocprofiler/rocprof
counters are added by this document. Existing corpus Top-K timing remains
`timing-only`.

Verified on 2026-05-21 against local submodules plus upstream PyTorch source
and public docs. Re-check before treating any item as the current latest.

## Quick Contract Map

| Stack | Main Top-K Surface | Default Contract | Core Algorithm Shape | Best Custom Opening |
| --- | --- | --- | --- | --- |
| PyTorch/ATen | `torch.topk` | `sorted=True`; tied indices are not stable | kth-value radix select, gather, optional sort; full-sort fallback on some ROCm paths | Fixed shape, known layout, no framework allocation, fused consumer |
| hipCUB/rocPRIM | `hipcub::DeviceTopK` | Unsorted; determinism not guaranteed | AIR radix Top-K with candidate buffers and filter passes | Sorted/tie-specific custom path, or fused downstream use |
| CUDA-origin MIGraphX-LLM contrast | sampling kernels, `topkLastDim` | Serving-dependent | two-stage block-reduce sampling; AIR/radix last-dim Top-K; small MoE reduce | Use as algorithm contrast only; prefer vLLM on ROCm/FlashInfer for ROCm baselines |
| FlashInfer | `flashinfer.top_k`, sampling APIs | `top_k` unsorted by default; sampling is statistical | radix Top-K, cluster path, FilteredTopK, rejection sampling | Match exact narrow contract, avoid sort/materialization, exploit graph/static shapes |
| vLLM | sampler dispatch, Triton, persistent Top-K | Runtime-selected | FlashInfer/PyTorch/Triton Qrita-style masking; persistent sparse-attention indexer | Beat the actual selected path for fixed mode, not the name "vLLM" |
| OneFlow | framework `top_k` op | sorted indices | heap path for small k, full sort for larger k or single instance | Simple baseline for heap-vs-sort crossover and framework-op cost |
| Transformer Engine | fused MoE router Top-K | routing map/probs, not generic Top-K | warp-per-token shared-memory Top-K, naive below k=16, radix above | Fuse score function, bias, grouped routing, normalization |
| Composable Kernel | CDNA3 GEMM TopK+Softmax epilogue | fused Top-K softmax, k=2/4 optimized | CDNA3 epilogue visitor tracks top-k values and logsumexp | Move Top-K before materialized logits or into GEMM epilogue |

Do not compare these rows without normalizing the contract. "Top-K" can mean
sorted values, unsorted membership, threshold mask, sampled token, MoE routing
map, page-table-transformed sparse-attention indices, or a fused epilogue that
never materializes candidates.

## PyTorch / ATen

Primary references:

- Upstream source:
  `https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cpp`
- Upstream source:
  `https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu`
- Upstream source:
  `https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SortingRadixSelect.cuh`
- Public docs:
  `https://docs.pytorch.org/docs/2.12/generated/torch.topk.html`

Observed contract:

- `torch.topk(input, k, dim=None, largest=True, sorted=True)` returns values
  and indices.
- The docs say `sorted=True` makes returned elements sorted, but tied indices
  are not guaranteed stable and may vary across invocations.
- `sorted=False` changes only output order, not the selected membership
  contract.

Implementation notes:

- CUDA dispatch in `TensorTopK.cpp` calls `launch_gather_topk_kernel`, then
  sorts only the selected `k` results if the user requested sorted output.
- For sorted selected results, ATen uses `sortKeyValueInplace` for small enough
  slices, otherwise allocates temporary sorted values/indices, calls CUDA sort
  on the selected result, gathers original indices, and copies back.
- Current CUDA `TensorTopK.cu` single-block path uses `radixSelect` to find the
  kth value, then gathers values strictly better than the kth value and fills
  remaining slots with values equal to the kth boundary.
- The gather stage uses prefix scans or warp compaction depending on platform.
  This is a selection-plus-gather design; it does not inherently produce sorted
  output.
- The multi-block path builds per-block radix digit counts, scans counts by
  slice, tracks kth counts, then gathers top-k values. This helps large slices
  and few slices.
- The source makes the input contiguous before kernel launch. If the custom
  kernel accepts the original stride/layout directly, record that as an omitted
  copy or as extra custom functionality.
- A ROCm-specific path can use full sort for certain large 1D cases and
  WarpMergeSort for small slices. Do not generalize those choices to CUDA
  without checking the compiled backend.

How to use it in this corpus:

- Use PyTorch as a framework oracle and practical baseline, not as the final
  opponent for serving decode.
- Record framework dispatch, allocation, `.contiguous()` copy, output
  allocation, and whether sorted output is requested.
- If the custom contract has lower-token-id tie breaks, repair or separately
  validate PyTorch tied rows because PyTorch does not promise that policy.

Custom attack surfaces:

- Fixed `k`, fixed row length, contiguous row-major logits.
- `sorted=False` when the downstream consumer only needs membership.
- Fused top-k plus softmax/sample that avoids materializing values and indices.
- Preallocated outputs and scratch, HIP Graph replay, or direct extension
  launch to avoid repeated framework overhead.
- Deterministic tie policy that avoids ATen's post-selection generic sort.

## hipCUB / rocPRIM/hipCUB/rocThrust

Primary local references:

- `third_party/cccl/cub/cub/device/device_topk.cuh`
- `third_party/cccl/cub/cub/device/dispatch/dispatch_topk.cuh`
- `third_party/cccl/cub/cub/agent/agent_topk.cuh`
- `third_party/cccl/cub/cub/block/block_topk.cuh`
- `third_party/cccl/cub/cub/block/specializations/block_topk_air.cuh`
- `third_party/cccl/cub/cub/device/dispatch/tuning/tuning_topk.cuh`
- `third_party/cccl/cub/test/catch2_test_device_topk_api.cu`
- `third_party/cccl/cub/test/catch2_test_device_topk_common.cuh`
- `third_party/cccl/cub/test/catch2_test_device_topk_pairs.cu`
- `third_party/cccl/cub/test/catch2_test_device_segmented_topk_keys.cu`
- `third_party/cccl/cub/cub/device/device_radix_sort.cuh`
- `third_party/cccl/cub/cub/device/device_select.cuh`

Public docs:

- `https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceTopK.html`

Observed contract:

- `DeviceTopK` finds largest or smallest K keys or key-value pairs.
- The current API only supports unsorted output and non-guaranteed
  determinism. The caller must explicitly request
  `cuda::execution::determinism::not_guaranteed` and
  `cuda::execution::output_ordering::unsorted`.
- Equal elements across the kth boundary can produce any valid subset of tied
  elements. This is not a stable lower-index tie policy.

Implementation notes:

- The algorithm is AIR TopK-style radix selection. It filters candidates over
  radix passes rather than sorting the entire input.
- Primitive keys are transformed into radix-orderable unsigned representations.
  Max selection can invert bits; decomposed custom keys use field-level radix
  traits.
- The default tuning uses 8 bits/pass for 1-byte keys and 11 bits/pass for
  2/4/8-byte keys, with 512-thread default policies and vectorized loads where
  the policy allows.
- Dispatch allocates temporary candidate buffers from hipCUB temp storage. The
  candidate capacity is roughly input-size dependent, with ping-pong key buffers
  and optional index buffers.
- Pass 0 is histogram-only over the full input. Later passes combine filtering
  and histogramming. The final pass filters candidates to outputs.
- `agent_topk` tracks remaining k, candidate length, kth-key prefix, filter
  counts, and front/back output counters. Equal-to-kth candidates are written
  from the back, which explains arbitrary tied-boundary subsets.
- `block_topk_air` exposes a block-level specialization: radix selection first,
  partition/scatter later. Selection itself does not move all data.
- Tests sort hipCUB outputs before comparison or check unordered membership. That
  is the right validation model for hipCUB `DeviceTopK`.
- `DeviceRadixSort` is the sorted/stable contrast point. Use it for full-sort
  oracle or high-k regimes.
- `DeviceSelect` preserves selected input order and is a threshold compaction
  part, not a Top-K primitive.

How to use it in this corpus:

- hipCUB `DeviceTopK` is the first serious baseline for unsorted Top-K
  membership.
- It is a `contract-mismatch` for sorted deterministic Top-K unless a
  deterministic repair or post-sort is included and timed.
- Always record temp-storage query, temp bytes, allocation/reuse policy, and
  whether result validation checks order or membership.

Custom attack surfaces:

- Sorted deterministic output without a separate full sort.
- Lower-token-id or stable tied-boundary repair.
- Tiny fixed `k` where hipCUB's generic AIR machinery does too much.
- Fused downstream stages that do not need hipCUB's materialized key/value output.
- Segmented or ragged specialization where public `DeviceTopK` is not the exact
  interface.

## CUDA-Origin MIGraphX-LLM Contrast

CUDA-origin references from the source corpus:

- `third_party/tensorrt-llm/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu`
- `third_party/tensorrt-llm/cpp/tensorrt_llm/kernels/samplingTopKKernels.h`
- `third_party/tensorrt-llm/cpp/tensorrt_llm/kernels/samplingTopPKernels.cu`
- `third_party/tensorrt-llm/cpp/tensorrt_llm/kernels/samplingAirTopPKernels.cu`
- `third_party/tensorrt-llm/cpp/tensorrt_llm/kernels/topkLastDim.cu`
- `third_party/tensorrt-llm/triton_kernels/topk_details/_topk_forward.py`
- `third_party/tensorrt-llm/docs/source/features/sampling.md`

Observed contract:

- MIGraphX-LLM Top-K is usually serving sampling, not standalone tensor
  selection.
- The current public sampling docs describe Torch Sampler as default for
  non-beam sampling, TRTLLM Sampler as deprecated, and Top-K/Top-P behavior in
  terms of `SamplingParams`.
- Top-P-after-Top-K rescales probabilities selected by Top-K before Top-P.
- The docs do not guarantee a particular treatment of tied probabilities.
- The Torch Sampler leverages FlashInfer optimized sampling kernels and
  sorting-free implementations when possible.

Implementation notes:

- `samplingTopKKernels.cu` implements a two-stage Top-K sampling pipeline.
- Stage 1 scans logits/probabilities by row/token, uses
  `hipcub::BlockReduce<TopK_2<T>>`, repeatedly extracts maxima, writes temp ids
  and values, and poisons selected temp logits to `-MAX_T_VAL`.
- Stage 2 merges the per-block candidates, exponentiates if inputs are logits
  rather than probabilities, computes sums/CDFs/logprobs as requested, and uses
  rocRAND/CURAND state to sample.
- `TOP_K_MAX` is 1024. `invokeBatchTopKSampling` chooses block sizes for
  `k <= 16`, `k <= 32`, `k <= 64`, and `k <= 1024`.
- Workspace includes full temp logits plus temp Top-K ids/values. Treat
  workspace as part of the comparison unless it is explicitly reused outside
  the timing boundary.
- `topkLastDim.cu` contains a specialized AIR TopK implementation. It has a
  small candidate path for `inputLength <= 128 && k <= 8` using a MoE-style
  warp `reduceTopK`, and a stable radix path for larger cases.
- The stable radix path can use hipCUB segmented stable sort when sorted output is
  required.
- Classic Top-P can sort probabilities with hipCUB segmented radix sort, then
  sample from sorted probabilities.
- AIR Top-P uses histogram/radix thresholding and has deterministic variants
  with constraints.
- MIGraphX-LLM Triton Top-K packs value and index into integer keys, uses
  `tl.topk`, then bitonic merge state.

How to use it in this corpus:

- Treat this section as CUDA-origin algorithm context only. For ROCm baselines,
  prefer actual vLLM on ROCm, FlashInfer, PyTorch, hipCUB/rocPRIM, or Triton
  paths measured in the target environment.
- Record finished-state handling, end ids, skip-decode flags, batch slots,
  tokens-per-step, top-k/top-p/min-p/beam policy, logits processors, return
  logprobs, rocRAND/CURAND state, and workspace.
- If a custom kernel omits serving generality, that can be a valid
  `custom-specialized-win`, but the omitted features must be listed.

Custom attack surfaces:

- Fixed `k <= 8` or `k <= 16` without dynamic Top-K arrays.
- No finished-state or per-request heterogeneity.
- Precomputed uniforms instead of rocRAND/CURAND state.
- Sampled-token-only output instead of temp ids, temp values, logprobs, and
  return-all-selected-tokens.
- Fused logits processors before Top-K insertion.
- Static batch/vocab so workspace and launch configuration are compiled or
  cached.

## FlashInfer

Primary local references:

- `third_party/flashinfer/flashinfer/topk.py`
- `third_party/flashinfer/flashinfer/sampling.py`
- `third_party/flashinfer/include/flashinfer/topk.cuh`
- `third_party/flashinfer/include/flashinfer/topk_common.cuh`
- `third_party/flashinfer/include/flashinfer/fast_topk_clusters_exact.cuh`
- `third_party/flashinfer/include/flashinfer/sampling.cuh`
- `third_party/flashinfer/include/flashinfer/air_top_p.cuh`
- `third_party/flashinfer/csrc/topk.cu`
- `third_party/flashinfer/csrc/sampling.cu`
- `third_party/flashinfer/benchmarks/bench_topk.py`
- `third_party/flashinfer/benchmarks/bench_sampling.py`
- `third_party/flashinfer/tests/utils/test_topk.py`

Public docs:

- `https://docs.flashinfer.ai/api/topk.html`
- `https://docs.flashinfer.ai/generated/flashinfer.top_k.html`
- `https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_sampling_from_probs.html`
- `https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html`

Observed contract:

- `flashinfer.top_k` is designed as a `torch.topk` replacement for large
  tensors and large vocabularies, but unlike PyTorch it defaults to unsorted
  output for performance.
- `deterministic=False` is the faster default. Tie-break modes for smaller or
  larger index force deterministic mode.
- The docs say the radix Top-K path is O(n) in vocabulary size and that
  `torch.topk` may be faster for small vocabularies below about 1000.
- Sampling APIs often promise statistical equivalence, not identical sampled
  token ids for a given external uniform stream.

Implementation notes:

- `top_k` chooses a cluster exact path when gfx target major is 10,
  deterministic mode is off, and graph-safety constraints allow it.
- Cluster Top-K allocates per-row overflow buffers, chooses more clusters for
  lower batch sizes, and falls back to one cluster for smaller model lengths.
- The standard radix path caches a 1 MB `row_states_buffer`, allocates output
  values, and calls `radix_topk`.
- Deterministic sorted output can be handled on device for `k <= 2048`; other
  sorted requests may sort the selected values with `torch.sort`.
- `FilteredTopK` needs 128 KB dynamic shared memory. This is an architecture
  and occupancy constraint, not just an API flag.
- Page-table and ragged transform APIs fuse Top-K selection with index
  transforms for sparse attention.
- Sampling APIs include Top-K sampling from probabilities, Top-K plus Top-P
  sampling from logits/probabilities, renormalization helpers, seed/offset
  controls, deterministic mode, and HIP Graph-friendly tensor seed/offset
  forms.
- FlashInfer tests compare by kth-value threshold when ties can differ, then
  separately test sorted compatibility.

How to use it in this corpus:

- Treat FlashInfer as a primary large-vocab inference baseline.
- Separate `flashinfer.top_k` selection from `flashinfer.sampling.*` APIs.
- Record whether the API consumes logits or probabilities. If it consumes
  probabilities, include softmax timing unless the custom kernel starts from
  probabilities too.
- Record deterministic mode, tie break, graph-safe path, environment override,
  cached row-state buffer, dynamic shared memory, sorted postprocessing, seed,
  offset, and whether RNG work is timed.

Custom attack surfaces:

- Small-vocab or tiny-k cases where FlashInfer itself notes PyTorch may win.
- Sorted deterministic output beyond the device-sorted sweet spot.
- Fixed shape with no need for row-state cache, graph-safety fallback, or
  Python API dispatch.
- Fused output transform or sampled-token-only path that avoids materializing
  full candidate lists.
- Architecture-specific kernels for GPUs without 128 KB shared memory support
  or with different GFXEM carveout behavior.

## vLLM

Primary local references:

- `third_party/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `third_party/vllm/vllm/v1/sample/ops/topk_topp_triton.py`
- `third_party/vllm/csrc/sampler.cu`
- `third_party/vllm/csrc/topk.cu`
- `third_party/vllm/csrc/persistent_topk.cuh`
- `third_party/vllm/benchmarks/benchmark_topk_topp.py`

Public source/doc references:

- `https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_triton.py`
- `https://github.com/vllm-project/vllm/blob/main/csrc/persistent_topk.cuh`

Observed contract:

- "vLLM Top-K" is not one implementation. The runtime may choose FlashInfer,
  Triton, PyTorch-native, ROCm AITER, XPU, or custom HIP paths depending on
  platform, feature flags, batch size, generators, and requested outputs.
- Processed logits/logprobs and per-request generators can force fallback away
  from FlashInfer.
- The sparse-attention `persistent_topk` path is not the normal token sampler.
  It is a DeepSeek/sparse-attention ragged indexer with `k` in
  `{512, 1024, 2048}` and FP32 logits.

Implementation notes:

- `TopKTopPSampler` uses FlashInfer on supported CUDA hardware when the mode
  allows it. It calls `logits.contiguous()` before the FlashInfer path.
- PyTorch fallback sorts logits when Top-P is involved. For Top-K-only and CPU
  sync allowed, it can avoid full vocab sort by using `torch.topk` values as a
  threshold and masking below that threshold.
- FlashInfer-backed sampling calls deterministic FlashInfer sampling APIs for
  Top-K, Top-P, or combined Top-K+Top-P.
- The Triton path is Qrita-inspired pivot masking, not materialized sorted
  Top-K. It computes statistics, chooses outlier pivots, performs ternary or
  quaternary pivot searches, handles duplicate boundary logits, and then masks
  logits for sampling.
- `apply_top_k_top_p_triton` applies Top-K before Top-P.
- `csrc/sampler.cu` contains C++ sampler Top-K paths with shared histograms,
  hipCUB scan/sort, insertion sort, and decode choices by vocab width.
- `persistent_topk.cuh` routes by sequence length: direct fill when
  `seq_len <= TopK`, 2048-bin histogram, 256-bin histogram, or a large-row
  multi-CTA radix path. It embeds a FlashInfer FilteredTopK fallback.
- `csrc/topk.cu` queries GFX count and max dynamic shared memory, can choose
  FilteredTopK when `num_rows > 32` and 128 KB shared memory is available, and
  otherwise launches persistent kernels. It zeroes row state with
  `cudaMemsetAsync`; timings should say whether the memset is included.

How to use it in this corpus:

- Always record the actual selected vLLM path. A result against PyTorch
  fallback is not a result against FlashInfer, and vice versa.
- Record environment variables, gfx target support, logprob mode,
  generator use, contiguous-copy cost, logits processors, backend workspace, and
  whether the boundary is isolated sampling or full decode.
- Treat Triton pivot masking as a filtering/sampling baseline, not a sorted
  values+indices Top-K baseline unless the exact outputs are materialized and
  checked.

Custom attack surfaces:

- Fixed serving mode where vLLM dispatch overhead or fallback behavior is known.
- No per-request generators or processed-logprobs output.
- Shape-static logits where a HIP extension can avoid Python/Triton launch and
  cached-buffer overhead.
- Materialized sorted Top-K when vLLM's path only masks logits, or sampled-only
  output when sorted materialization is not needed.
- Ragged indexer variants for k=512/1024/2048 where custom kernels can exploit
  length buckets or page-table locality.

## OneFlow

Primary local references:

- `source:oneflow/oneflow/user/kernels/top_k_kernel.cu`
- `source:oneflow/oneflow/core/functional/functional_api.yaml`
- `source:oneflow/oneflow/core/functional/impl/array_functor.cpp`

Observed contract:

- OneFlow exposes a `topk` framework API with `largest` and `sorted`
  arguments.
- The HIP kernel returns top-k indices, with values gathered later in the
  framework path.

Implementation notes:

- For `k > 30` or `instance_num == 1`, OneFlow initializes indices and uses a
  hipCUB-based descending sort path over all pairs, then copies the first `k`
  indices per instance.
- For `k <= 30` and multiple instances, it launches `HeapTopKKernel`.
- `HeapTopKKernel` gives each thread a heap over a strided subset, then
  bitonic-sorts all per-thread heap entries in shared memory and writes the
  first `k` indices.
- Heap size is rounded up to a power of two, and the number of heaps is bounded
  by shared memory and max threads per block.

How to use it in this corpus:

- Use OneFlow as a compact framework-kernel reference for the heap-vs-full-sort
  crossover.
- It is a useful negative-example generator: the threshold `k > 30` is not a
  universal CUDA truth, but it is a concrete production heuristic to beat.

Custom attack surfaces:

- Replace per-thread heaps with register lists for tiny fixed `k`.
- Avoid bitonic sorting all heap entries when only unsorted membership is
  needed.
- Tune the heap/list/sort boundary per GPU and row length.
- Fuse value gather, routing, or sampling after index selection.

## Transformer Engine Fused Router

Primary local references:

- `source:transformer-engine/transformer_engine/pytorch/router.py`
- `source:transformer-engine/transformer_engine/common/fused_router/fused_topk_with_score_function.cu`
- `source:transformer-engine/transformer_engine/common/fused_router/fused_score_for_moe_aux_loss.cu`
- `source:transformer-engine/transformer_engine/common/include/transformer_engine/fused_router.h`
- `source:transformer-engine/transformer_engine/common/util/topk.cu`

Observed contract:

- This is MoE routing, not generic Top-K values/indices.
- Inputs are token-by-expert logits plus optional expert bias and grouped
  routing metadata.
- Outputs are probabilities and routing maps, with forward/backward support.

Implementation notes:

- One warp handles one token and owns shared-memory buffers for scores,
  Top-K scores, Top-K indices, optional group scores, and masked scores.
- It can apply score functions before or after Top-K: softmax, sigmoid, or
  sqrt-softplus-style processing.
- It supports grouped Top-K: choose groups first, mask other experts, then Top-K
  inside the selected groups.
- Forward dispatch uses a naive Top-K function below `topk < 16`; for larger k
  it switches to a radix Top-K function because radix selection is O(E) in
  number of experts while naive selection starts to dominate.
- Backward masks gradients for unselected experts and applies the backward path
  for the score function around the Top-K boundary.

How to use it in this corpus:

- Treat this as the strongest local reference for `moe-token-routing`, not as a
  row-major vocabulary Top-K baseline.
- Record score function, grouped Top-K policy, expert bias, scaling factor,
  routing map format, backward requirements, and shared-memory size.

Custom attack surfaces:

- Fixed expert count and fixed top-k, usually k=1/2/4/8.
- Fused bias, score transform, Top-K, normalization, and routing-map writes.
- A forward-only inference router can omit backward buffers and save memory.
- If grouped Top-K is disabled, remove group buffers and branch paths.

## Composable Kernel Top-K Softmax Epilogue

Primary local references:

- `third_party/cutlass/examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu`
- `third_party/cutlass/include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp`
- `third_party/cutlass/include/cutlass/epilogue/fusion/operations.hpp`
- `third_party/cutlass/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp`

Observed contract:

- This is not a standalone Top-K API. It fuses GEMM, Top-K, and softmax in an
  CDNA3 epilogue visitor.
- The example fuses over the N dimension and assumes static Top-K.
- K=2 and K=4 are performance-optimized and enabled by default. Other K values
  can be enabled by removing an assertion, but the example warns about serious
  performance implications.
- The example notes that repeated top-k values are problematic because the
  fused path tracks values, not unique value/index pairs.

Implementation notes:

- The epilogue visitor keeps a per-fragment sorted top-k array.
- It reduces Top-K values across lanes using shuffle-xor or shuffle-down
  reductions.
- After reduction, it keeps only the minimum of the Top-K set and logsumexp of
  the Top-K values. Non-Top-K outputs are masked, and selected outputs get
  softmax values.
- The main opportunity is moving Top-K into the producer boundary so logits do
  not need to be fully materialized and reread.

How to use it in this corpus:

- Treat this as an epilogue-fusion reference for LM-head, MoE, or retrieval
  score producers.
- Do not use it as a generic replacement for `torch.topk`; it has narrower
  assumptions and CDNA3 epilogue constraints.

Custom attack surfaces:

- Fuse Top-K into matmul or score-generation epilogues.
- Track value-index pairs to repair duplicate-value correctness.
- Generate separate kernels for k=1/2/4/8 instead of generic Top-K.
- Compare materialized GEMM plus Top-K versus fused epilogue at the same output
  boundary.

## Cross-Stack Lessons

### Separate Selection, Masking, Sampling, and Routing

Selection returns values or indices. Masking writes `-inf` or a boolean mask.
Sampling returns token ids and may never materialize Top-K candidates. Routing
writes expert maps and weights. The fastest implementation for one contract can
be invalid for another.

### Sortedness Is Expensive and Often Optional

hipCUB and FlashInfer both expose unsorted fast paths. PyTorch defaults to sorted
output and sorts after selection when needed. A custom kernel should carry two
variants when possible:

- `unsorted-membership`: fastest candidate set for downstream sampling or
  thresholding.
- `sorted-deterministic`: values and indices in order with explicit tie policy.

### Ties Are a Correctness Boundary

Use pair comparisons `(value, index)` when the corpus contract demands lower or
higher token ids. Libraries often document tied-boundary instability or leave
it path-dependent. Add adversarial rows with all equal logits and repeated
values around kth.

### Workspace Is Part of the Algorithm

Record all of these:

- hipCUB temp-storage bytes and whether query/allocation is timed.
- vLLM on ROCm full temp logits, temp Top-K ids/values, and rocRAND/CURAND state.
- FlashInfer row-state cache, overflow buffers, 128 KB shared-memory path, and
  sorted `torch.sort` fallback.
- vLLM cached buffers, contiguous copies, row-state `cudaMemsetAsync`, and
  workspace tensors.
- Framework output allocations and copies.

### K Regime Determines the Kernel Family

- `k=1`: value-index reduction, often no Top-K machinery needed.
- `k=2..8`: register list or fixed compare network, warp/CTA merge.
- `k=16..32`: network or two-level merge starts competing with heap/list.
- `k=64..256`: heap/list, two-pass candidates, or radix threshold.
- `k>=512`: radix, histogram, threshold partition, or full sort.
- High `k/n`: full sort may be the honest winner.

### Architecture Matters

FilteredTopK and large shared-memory designs require architecture-specific
checks. CDNA3/CDNA4/RDNA4 features matter only if the source or AMD GCN ISA uses them.
Record `sm`, dynamic shared memory, amdclang++ resource/ISA registers, spills, occupancy estimate,
and whether the baseline selected a different backend on that GPU.

## Baseline Checklist for New Top-K Work

1. Freeze the exact contract:
   values, indices, mask, sampled id, routing map, sortedness, tie policy,
   NaN policy, RNG, and output materialization.
2. Choose matching baselines:
   PyTorch for oracle, hipCUB `DeviceTopK` for unsorted membership, hipCUB radix sort
   for full sort, FlashInfer for inference selection/sampling, vLLM on ROCm or
   vLLM for serving boundaries, OneFlow/Transformer Engine/Composable Kernel for
   framework, MoE, or epilogue-specific variants.
3. Record the actual path selected:
   PyTorch selection plus sort, vLLM FlashInfer path, vLLM Triton path,
   vLLM on ROCm Torch Sampler, direct kernel, etc.
4. Time the same boundary:
   no allocation in one path and allocation in another unless that is the
   deployment reality.
5. Validate adversarial rows:
   all-equal, boundary ties, masks, NaNs, infinities, short rows, long rows,
   `k=1`, `k=n`, and non-power-of-two vocab.
6. Label evidence honestly:
   `timing-only` without rocprofiler/rocprof counters, `negative example` for correct
   losses, and `contract-mismatch` for sortedness/ties/RNG/materialization
   differences.

## Next Corpus Expansions

- Add same-hardware PyTorch, hipCUB `DeviceTopK`, hipCUB radix sort, FlashInfer,
  vLLM on ROCm, and vLLM baseline harnesses for the wide-k matrix.
- Add a standalone hipCUB `DeviceTopK` example that records the required
  execution environment and validates unordered membership.
- Add PyTorch extension harnesses that separate framework dispatch from HIP
  kernel time.
- Add a FlashInfer baseline script that toggles `sorted`, `deterministic`,
  tie-break, graph-safe mode, and small/large vocab shapes.
- Add a vLLM path-probe script that prints whether FlashInfer, Triton, PyTorch,
  or a custom HIP path was selected.
- Add a vLLM on ROCm direct-kernel or minimal-sampler reproduction with
  workspace accounting.
- Add MoE routing and Composable Kernel epilogue tasks so Top-K fusion beyond sampling is
  represented.

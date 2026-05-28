# Top-K Baseline Harness Blueprint

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This blueprint turns the Top-K roadmap and implementation survey into a
measurement plan an agent can execute. It does not add new measured evidence by
itself. Use it when building the next Top-K benchmark scripts, especially for
`corpus/tasks/wide-k-topk-selection` and `corpus/tasks/block-topk-sampling`.

Evidence status: `template-only`. All future timings from this blueprint must
say `timing-only` unless rocprofiler/rocprof counters are attached.

## Goal

Build one harness family that can compare:

- custom HIP Top-K kernels,
- PyTorch/ATen `torch.topk`,
- hipCUB `DeviceTopK`,
- hipCUB `DeviceRadixSort`,
- FlashInfer `top_k` and sampling APIs,
- vLLM's actual selected sampler path,
- vLLM on ROCm sampling or direct Top-K kernels,
- optional OneFlow, Transformer Engine, and Composable Kernel specializations.

The harness must not hide contract differences. Every result must say whether
it measured selection, masking, sampling, routing, ragged indexing, or epilogue
fusion.

## Harness Layers

Use three layers, not one giant script.

| Layer | Purpose | Recommended Language | Timing Boundary |
| --- | --- | --- | --- |
| `oracle` | CPU correctness and adversarial cases | C++ or Python | not performance |
| `selection_baselines` | PyTorch, hipCUB TopK, hipCUB sort, custom selection | C++ plus Python wrapper | HIP events around device work |
| `serving_baselines` | FlashInfer, vLLM, vLLM on ROCm sampling paths | Python and optional C++ | API call or direct kernel, clearly labeled |

Keep selection and serving results separate. A fast vLLM masking path is not a
sorted values/indices Top-K result unless it actually returns and validates
those outputs.

## Result Families

### Selection Result

Use this for `torch.topk`, hipCUB `DeviceTopK`, hipCUB radix sort, custom HIP, and
OneFlow-like framework Top-K.

Required output fields:

- `values[rows, k]` when materialized.
- `indices[rows, k]`.
- `membership_hash` for unsorted membership checks.
- `sortedness`: `sorted`, `unsorted`, or `mask-only`.
- `tie_policy`: lower index, higher index, stable input order, unspecified, or
  not guaranteed.
- `nan_policy`.
- `workspace_bytes`.
- `launch_count`.

Correctness modes:

- `exact_sorted`: values and indices match the CPU stable reference.
- `exact_unsorted_membership`: same selected set, order ignored.
- `threshold_equivalent`: all values are above kth threshold plus a valid
  boundary subset. Use only when ties are explicitly permitted.
- `contract_mismatch`: useful timing, but not equivalent.

### Sampling Result

Use this for Top-K sampling, Top-P, Top-K+Top-P, FlashInfer sampling, vLLM
samplers, and vLLM on ROCm sampling.

Required output fields:

- `sampled_ids[rows]`.
- optional `selected_indices[rows, k]`.
- optional `selected_values[rows, k]`.
- optional `selected_probs[rows, k]`.
- `rng_source`: precomputed uniforms, Philox seed/offset, rocRAND/CURAND state,
  PyTorch generator, or library internal.
- `sample_match_policy`: exact uniform replay, exact seed replay, or
  statistical equivalence.
- `filter_order`: top-k first, top-p first, joint, or not applicable.
- `probability_input`: logits or probabilities.

Correctness modes:

- `exact_uniform_replay`: same token id for supplied uniforms.
- `exact_seed_replay`: same token id for a documented seed/offset policy.
- `distributional`: statistical test passes, exact token ids may differ.
- `contract_mismatch`: RNG or filter policy differs.

### Routing / Epilogue Result

Use this for Transformer Engine MoE routing or Composable Kernel TopK+Softmax epilogues.

Required output fields:

- `route_indices`, `routing_map`, or epilogue output tensor.
- score function and grouped-routing policy.
- whether backward is part of the contract.
- whether logits or scores were ever materialized.
- producer operation shape, such as GEMM M/N/K.

## First Benchmark Matrix

The first wide-k run should use fixed deterministic input data and these rows:

| Shape ID | Rows | N / Vocab | K | Sorted | Purpose |
| --- | ---: | ---: | ---: | --- | --- |
| `k1_llm` | 4096 | 32000 | 1 | false | argmax/greedy |
| `k4_seed` | 1024 | 32768 | 4 | true | current seed parity |
| `k8_llm` | 1024 | 32768 | 8 | true | tiny fixed-k |
| `k16_small` | 512 | 32768 | 16 | true | network/list boundary |
| `k32_small` | 256 | 32768 | 32 | true | small-k merge boundary |
| `k64_medium` | 256 | 65536 | 64 | true | heap/list/radix boundary |
| `k128_medium` | 128 | 65536 | 128 | true | medium-k |
| `k256_large` | 64 | 131072 | 256 | true | large-k candidate buffers |
| `k512_large` | 32 | 131072 | 512 | true | radix/full-sort boundary |
| `k1024_large` | 16 | 131072 | 1024 | true | large-k |
| `k8192_near_full` | 16 | 32768 | 8192 | true | full-sort crossover |
| `awkward_vocab` | 128 | 1009 | 32 | true | non-power-of-two |
| `tiny_rows` | 65536 | 512 | 4 | true | many short rows |
| `few_long_rows` | 8 | 262144 | 32 | true | multi-CTA pressure |
| `ragged_zipf` | 4096 | 1..65536 | 1..128 | mixed | segmented scheduler |

Do not average these into one score. Each shape is a different regime.

## Adversarial Correctness Pack

Every backend should run the same adversarial pack:

- all logits equal,
- two distinct values tied across kth,
- ascending row,
- descending row,
- max at column 0,
- max at last column,
- repeated top value more than k times,
- non-power-of-two row length,
- `k=1`,
- `k=n`,
- all masked,
- one unmasked,
- NaN present,
- `+Inf` and `-Inf`,
- BF16/FP16 close values that round to the same value,
- ragged length 0, 1, `k-1`, `k`, and long.

If a library does not define one of these policies, record the result as
`contract_mismatch` or exclude that case from same-contract timing.

## Backend Recipes

### CPU Oracle

Implement one stable CPU reference:

- compare `(value, -index)` for lower-index ties when largest=true,
- explicit NaN policy,
- stable full sort for sorted output,
- set-membership validation for unsorted output,
- separate sampling reference that consumes precomputed uniforms.

The CPU oracle is not a performance baseline.

### PyTorch

Recipe:

- Generate `torch.Tensor` inputs on CUDA.
- Warm allocator and optionally preallocate output tensors with `out=`.
- Run `torch.topk(logits, k, dim=-1, largest=True, sorted=sorted)`.
- For sampling, run softmax over selected values and either consume a
  precomputed uniform with a custom CDF expression or use `torch.multinomial`
  and record the RNG contract.

Record:

- PyTorch version and CUDA version.
- Whether `out=` buffers are used.
- Whether `torch.cuda.synchronize()` wraps timing.
- Whether `.contiguous()` was needed.
- Whether framework dispatch and allocation are included.
- Tie instability warning for tied rows.

### hipCUB DeviceTopK

Recipe:

- Use `MaxPairs` for `(value, index)` outputs.
- Pass `cuda::execution::determinism::not_guaranteed` and
  `cuda::execution::output_ordering::unsorted`.
- Query temp storage once, allocate once, warm up, then time only execution for
  steady-state kernel timings.
- Validate unordered membership, not sorted order.

Record:

- temp storage bytes,
- whether query/allocation is timed,
- key type and value type,
- decomposer if custom key is used,
- output ordering `unsorted`,
- determinism `not_guaranteed`.

Use hipCUB `DeviceRadixSort::SortPairsDescending` separately for sorted full-sort
contracts.

### FlashInfer

Recipe:

- Test `flashinfer.top_k(logits, k, sorted=..., deterministic=...)`.
- Sweep `tie_break=NONE`, lower-index, and higher-index modes where supported.
- Test small vocab, standard LLM vocab, and large vocab separately.
- For sampling-facing contracts, test `top_k_sampling_from_probs` and
  `top_k_top_p_sampling_from_logits`.

Record:

- FlashInfer version/commit,
- selected algorithm if exposed or controlled by environment,
- sorted and deterministic flags,
- tie-break mode,
- graph-safe mode,
- row-state cache behavior,
- dynamic shared-memory requirement,
- seed/offset and whether sampling is exact or distributional,
- whether a `torch.sort` fallback ran for sorted output.

### vLLM

Recipe:

- Add a path-probe wrapper that logs the selected sampler backend before
  timing.
- Test `VLLM_USE_FLASHINFER_SAMPLER=1`, `0`, and default when feasible.
- Test modes that force fallback: per-request generators, processed logits, and
  processed logprobs.
- Time isolated sampler calls separately from end-to-end decode.

Record:

- vLLM version/commit,
- selected path: FlashInfer, Triton, PyTorch, AITER, XPU, or custom HIP,
- whether `logits.contiguous()` was timed,
- whether logits processors ran,
- generator mode,
- logprobs mode,
- batch size threshold that selected Triton or PyTorch,
- cached-buffer allocation.

Do not write "vLLM baseline" without the actual path.

### vLLM on ROCm

Recipe:

- Prefer a direct minimal invocation only after matching the sampled-output
  contract. Otherwise use vLLM on ROCm as serving-boundary evidence.
- For Top-K sampling, record whether the path is Torch Sampler, TRTLLM Sampler,
  or direct `invokeBatchTopKSampling`.
- For Top-P or Top-K+Top-P, record filter order and sampler backend.

Record:

- vLLM on ROCm version/commit,
- `maxTopK`,
- top-p/min-p/temperature,
- workspace bytes from helper,
- full temp logits bytes,
- temp ids/values bytes,
- rocRAND/CURAND state,
- finished-state, end-id, skip-decode, batch-slot, and tokens-per-step flags,
- return-logprobs or return-all-selected-token modes.

### OneFlow

Recipe:

- Use only when studying framework-op integration or heap-vs-sort crossover.
- Compare `k <= 30` heap path and larger-k full-sort path if the local build can
  expose them.

Record:

- whether heap path or full-sort path ran,
- shared memory size for heap path,
- hipCUB sort temp storage for full-sort path,
- whether values are gathered in the timed boundary.

### Transformer Engine

Recipe:

- Use for `moe-token-routing`, not generic vocabulary Top-K.
- Compare fixed expert counts and `topk=1/2/4/8/16`.
- Separate forward-only inference from forward+backward training.

Record:

- score function,
- grouped Top-K policy,
- expert bias,
- routing map layout,
- shared-memory size,
- topk function mode: naive or radix,
- backward included or excluded.

### Composable Kernel

Recipe:

- Use for producer-fused GEMM/score Top-K.
- Compare materialized GEMM plus Top-K against Composable Kernel TopK+Softmax epilogue.
- Start with static `k=2` and `k=4` on CDNA3-compatible hardware.

Record:

- GFX target,
- GEMM M/N/K,
- epilogue tile and Top-K static value,
- duplicate-value policy,
- whether logits/scores are materialized.

## Timing Boundaries

Each result must choose one:

- `kernel_only`: HIP events around one custom or library kernel.
- `steady_state_api`: API call with preallocated reusable workspace.
- `api_with_allocation`: includes output/workspace allocation.
- `graph_replay`: HIP Graph replay only.
- `framework_call`: includes Python or framework dispatch.
- `serving_step`: includes logits processors, sampler, scheduler-visible work,
  and framework runtime boundaries.

Never compare different boundaries as a win/loss without labeling it
`contract_mismatch` or `comparison_incomplete`.

## Result JSON Fields

Minimum fields for Top-K baseline result artifacts:

```json
{
  "schema_version": "0.1.0",
  "task_id": "wide-k-topk-selection",
  "shape_id": "k32_small",
  "backend": "cub-device-topk",
  "backend_version_or_commit": "local-submodule-commit",
  "actual_backend_path": "hipcub::DeviceTopK::MaxPairs",
  "contract_family": "selection",
  "timing_boundary": "steady_state_api",
  "evidence_label": "timing-only",
  "rows": 256,
  "n": 32768,
  "k": 32,
  "dtype": "fp32",
  "sortedness": "unsorted",
  "tie_policy": "not_guaranteed",
  "correctness_mode": "exact_unsorted_membership",
  "median_ms": 0.0,
  "p10_ms": 0.0,
  "p90_ms": 0.0,
  "workspace_bytes": 0,
  "allocation_timed": false,
  "launch_count": 0,
  "gpu_name": "",
  "gfx_target": "",
  "driver_version": "",
  "rocm_or_cuda_runtime_version": "",
  "notes": "template example"
}
```

Use `schemas/topk_baseline_result.schema.json` for validation once result files
are added.

## Implementation Order

1. Add CPU oracle and adversarial pack.
2. Add PyTorch sorted and unsorted timing. No ROCm PyTorch Top-K baseline is
   attached yet; record exact-contract, raw unsorted, and tie-unstable variants
   separately.
3. Add hipCUB `DeviceTopK` unsorted membership.
4. Add hipCUB `DeviceRadixSort` sorted full-sort oracle.
5. Add custom `k=1/2/4/8` variants.
6. Add FlashInfer `top_k` and sampling APIs.
7. Add vLLM path probe and isolated sampler timing.
8. Add vLLM on ROCm only when the same serving contract can be invoked.
9. Add OneFlow, Transformer Engine, and Composable Kernel only for their specialized
   contracts.

## Graduation Criteria

A Top-K regime graduates from `template-only` when it has:

- CPU correctness for the adversarial pack,
- at least one same-hardware PyTorch result,
- the strongest applicable hipCUB or full-sort result,
- FlashInfer/vLLM/vLLM on ROCm result when sampling-facing,
- custom result or explicit library-win/negative example,
- complete timing-boundary and workspace metadata,
- `timing-only` or `counter-backed` evidence label.

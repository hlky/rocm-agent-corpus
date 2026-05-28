# Selection and Sampling Library Baselines

This guide defines the library-baseline comparison surface for
`block-topk-sampling`. It is guidance only until a GPU run attaches measured
records. Treat every result from these recipes as `template-only` until the
exact contract, hardware, build metadata, and timing boundary are recorded.

The seed task currently implements top-k temperature sampling:

- Input logits are row-major `[rows, vocab_size]` FP32.
- `k <= 8`.
- Top-k output is sorted by descending logit.
- Ties are resolved by lower token id.
- Sampling uses `softmax(top_k_logits / temperature)`.
- Uniform random numbers are precomputed and supplied as input.
- It is not a top-p implementation.

Any library comparison that changes one of those points is a different
contract, not a custom-kernel win or loss.

## Baseline Ladder

Use the baselines in this order. Stop only after the strongest applicable
baseline for the same contract is measured or marked unavailable.

| Baseline | Role | Expected Comparison |
| --- | --- | --- |
| hipCUB `DeviceTopK` | Unsorted top-k selection baseline | Strong for top-k membership, not equivalent to sorted output or deterministic ties |
| hipCUB `DeviceRadixSort` | Full sorted oracle and stress baseline | Strong correctness oracle for sorted order, often too general for small fixed `k` |
| hipCUB `DeviceSelect` | Predicate compaction building block | Useful for threshold/top-p sketches, not a standalone top-k sampler |
| vLLM on ROCm top-k/top-p kernels | Serving-stack baseline and edge-case reference | Strong when sampling features, finished states, runtime args, and workspace match |
| vLLM sampler paths | Serving baseline with framework, Triton, and FlashInfer dispatch | Fair only when logits processors, generators, and logprob return modes match |
| FlashInfer sampling/top-k/AIR top-p | Optimized inference sampling baseline | Strong for sampling from probabilities/logits and sorting-free top-p/top-k paths |
| PyTorch `torch.topk` plus `torch.multinomial` | Framework baseline and oracle | Easy to reproduce but may include framework dispatch, allocation, and sync costs |

## Fairness Metadata

Every measured comparison must record these fields. Missing metadata downgrades
the result to `template-only` or `comparison-incomplete`.

- Baseline name, version, upstream commit, and local path.
- GPU name, gfx target, driver, ROCm toolkit, compiler, and build flags.
- Rows, vocab size, `k`, temperature, dtype, layout, strides, and alignment.
- Whether the baseline is top-k, top-p, top-k plus top-p, min-p, greedy, or beam.
- Whether top-k output is sorted, unsorted, or not materialized.
- Tie policy: lower token id, higher token id, first observed, stable input
  order, unspecified, or not guaranteed.
- NaN and mask policy.
- RNG policy: precomputed uniforms, library-generated RNG, PyTorch generator,
  CURAND state, seed/offset arrays, deterministic mode, or not included.
- Whether RNG generation time is included in timing.
- Whether logits processors are included: temperature, repetition/presence/
  frequency penalties, bad-word masks, grammar/guided decoding masks, stop/EOS
  logic, no-repeat n-gram, logit bias, or custom request processors.
- Whether softmax, probability renormalization, and output probability/logprob
  materialization are included.
- Workspace bytes, workspace query, allocation, pooling, reuse, and whether
  allocation is timed.
- Number of kernel launches, framework dispatch boundary, stream, graph capture
  status, and synchronization points.
- Correctness oracle, tolerances, and whether sampled ids are expected to match
  exactly or only statistically.
- Evidence label: `timing-only` when HIP-event timings lack profiler counters;
  `negative example` when an attempted optimization does not improve speed.

## Top-K Versus Top-P

Top-k and top-p are not interchangeable.

Top-k keeps candidates by rank. For the seed contract, the top-k list must be
sorted by descending logit, and lower token id wins ties. Probabilities are
renormalized only across those `k` candidates.

Top-p keeps the smallest probability-ordered prefix whose cumulative mass
reaches `p`, after probabilities are formed. It requires a sorted or
equivalent thresholding policy over probabilities, and it has boundary-token
semantics that can differ across implementations. Combining top-k and top-p
requires recording filter order. vLLM on ROCm documents top-p-after-top-k
behavior in its sampler path, where probabilities selected by top-k are
rescaled before top-p is applied.

Do not claim top-p equivalence from `block-topk-sampling` unless a separate
top-p contract and oracle are added.

## hipCUB Baselines

### `hipcub::DeviceTopK`

Local references:

- `third_party/rocm-libraries/cub/cub/device/device_topk.hpp`
- `third_party/rocm-libraries/cub/test/catch2_test_device_topk_api.hip`

Use `DeviceTopK::MaxPairs` or `MaxKeys` as the first hipCUB selection baseline
when the contract accepts unordered top-k output and non-guaranteed tie
determinism. The upstream API currently requires acknowledging
`cuda::execution::determinism::not_guaranteed` and
`cuda::execution::output_ordering::unsorted`.

For the seed task, that means `DeviceTopK` is not contract-equivalent by
itself, because the seed writes sorted top-k candidates and has deterministic
lower-token-id ties. Valid comparison choices:

- Compare only top-k set membership and classify as `contract-mismatch` for
  sortedness/ties.
- Add a post-sort or deterministic tie repair step and time it.
- Change the custom contract to unsorted top-k only and record that as a
  separate benchmark.

Record temporary-storage query and allocation separately. Time the query only
if the deployment pays it in steady state.

### `hipcub::DeviceRadixSort`

Local reference:

- `third_party/rocm-libraries/cub/cub/device/device_radix_sort.hpp`

Use `DeviceRadixSort::SortPairsDescending` on `(logit_key, token_id)` pairs
when a fully sorted oracle is needed. It is stable, supports ascending and
descending order, and can provide deterministic sorted output when keys encode
the intended tie policy.

For lower-token-id ties with descending logits, encode a composite key or use a
post-processing rule that sorts equal logits by ascending token id. A raw sort
by logit alone is not enough to match the seed contract. This baseline is often
expected to be a `library-win` for full sort contracts and a `negative example`
for small fixed `k` when it does unnecessary work.

Record whether double-buffer or non-double-buffer APIs are used and the
workspace factor. The source documents variants using approximately `N` or
`2N` auxiliary storage depending on API shape.

### `hipcub::DeviceSelect`

Local reference:

- `third_party/rocm-libraries/cub/cub/device/device_select.hpp`

Use `DeviceSelect::If`, `Flagged`, or `FlaggedIf` as a threshold-compaction
building block, not as a standalone top-k sampler. It preserves relative input
order among selected items. That is useful for:

- Compaction after a threshold found by a separate selection pass.
- Top-p candidate prefix experiments after a cutoff is known.
- Masked or filtered vocabulary studies.

It does not solve rank ordering, top-k membership by itself, softmax
renormalization, or sampling. If a custom kernel beats a pipeline that uses
`DeviceSelect` plus other kernels, record launch count and threshold-compute
cost explicitly.

## vLLM on ROCm Baselines

Local references:

- `third_party/vllm/cpp/migraphx_llm/kernels/samplingTopKKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/samplingTopPKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/samplingAirTopPKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/topkLastDim.hip`
- `third_party/vllm/docs/source/features/sampling.md`

vLLM on ROCm is a serving baseline, not just an isolated kernel collection.
The top-k sampling path includes staged top-k selection, sampling, runtime
arguments, finished-state handling, optional return of selected tokens/logprobs,
and CURAND state use. The top-p paths include both classic top-p sampling and
AIR top-p workspace-driven thresholding. `topkLastDim.hip` is also relevant for
AIR TopK-style last-dimension selection and MoE/router-like top-k.

Fair comparison requirements:

- Record sampler backend and vLLM on ROCm version or commit.
- Record whether Torch Sampler, TRTLLM Sampler, or a direct kernel invocation is
  used.
- Record top-k, top-p, temperature, min-p, beam, and greedy policy.
- Record whether finished sequences, end ids, skip-decode flags, batch slots,
  and tokens-per-step are active.
- Record whether logits processors and return-logits/logprobs features are
  active.
- Record CURAND state, seed handling, and whether RNG is timed.
- Record workspace sizes from the top-k, top-p, or AIR top-p helpers and
  whether allocation is reused.

Expected classifications:

- `library-win` when vLLM on ROCm matches the serving contract and is faster.
- `custom-specialized-win` only when the custom path drops documented serving
  generality, preserves correctness, and beats the matching timed boundary.
- `contract-mismatch` when comparing seed top-k-only sampling to top-p, beam,
  or return-logprob behavior.

## vLLM Baselines

Local references:

- `third_party/vllm/csrc/topk.hip`
- `third_party/vllm/csrc/persistent_topk.hpp`
- `third_party/vllm/csrc/sampler.hip`
- `third_party/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `third_party/vllm/vllm/v1/sample/ops/topk_topp_triton.py`
- `third_party/vllm/vllm/v1/sample/logits_processor/builtin.py`
- `third_party/vllm/benchmarks/benchmark_topk_topp.py`

vLLM chooses among PyTorch-native, Triton, FlashInfer, ROCm, XPU, and custom
CUDA paths depending on platform, batch size, generator use, logprob mode, and
feature request. The Python sampler path may fall back when per-request
generators or processed logits/logprobs are requested. It can call FlashInfer
for CUDA top-k/top-p sampling when supported.

Fair comparison requirements:

- Record the actual path selected, not just "vLLM".
- Record whether logits are contiguous and whether a `.contiguous()` copy is
  timed.
- Record whether per-request generators force fallback.
- Record whether processed logits/logprobs are returned.
- Record whether logits processors from `vllm/v1/sample/logits_processor` are
  included before sampling.
- Record whether the comparison is isolated sampler time or end-to-end decode
  step time.

Expected classifications:

- `library-win` or `serving-win` when vLLM's selected path is faster at the same
  end-to-end boundary.
- `custom-specialized-win` when a narrow custom HIP kernel beats the actual
  selected vLLM path for a fixed shape and the omitted vLLM generality is
  documented.
- `comparison-incomplete` when the selected path is unknown.

## FlashInfer Baselines

Local references:

- `third_party/flashinfer/include/flashinfer/topk.hpp`
- `third_party/flashinfer/include/flashinfer/sampling.hpp`
- `third_party/flashinfer/include/flashinfer/air_top_p.hpp`
- `third_party/flashinfer/csrc/topk.hip`
- `third_party/flashinfer/csrc/sampling.hip`
- `third_party/flashinfer/benchmarks/bench_topk.py`
- `third_party/flashinfer/benchmarks/bench_sampling.py`

FlashInfer is a primary inference sampling baseline. Its sampling header exposes
sampling from logits/probabilities, top-k sampling from probabilities, top-p
sampling from probabilities, top-k plus top-p sampling, min-p sampling, and
renormalization helpers. Its AIR top-p header adapts the vLLM on ROCm AIR Top-P
algorithm as a standalone path.

Fair comparison requirements:

- Record whether the API consumes logits or probabilities.
- If the API consumes probabilities, include or exclude softmax consistently.
- Record deterministic mode, seed array/value, offset array/value, and whether
  RNG work is timed.
- Record whether top-k/top-p results are statistically equivalent rather than
  token-id identical for a fixed uniform stream.
- Record whether output probabilities, logprobs, or selected candidate lists are
  materialized.
- Record dynamic shared memory and external workspace where used.

Expected classifications:

- `library-win` when FlashInfer matches the sampling contract and is faster.
- `near-match` when token ids differ but the documented contract is statistical
  equivalence and distribution tests pass.
- `contract-mismatch` when the seed requires exact precomputed-uniform samples
  but the FlashInfer path generates or consumes RNG differently.

## Framework Baseline

Use PyTorch as the easy framework baseline:

```python
values, indices = torch.topk(logits, k, dim=-1, largest=True, sorted=True)
probs = torch.softmax(values / temperature, dim=-1)
sample_offsets = torch.multinomial(probs, num_samples=1)
sampled = indices.gather(1, sample_offsets)
```

Fair comparison requirements:

- Record PyTorch version, ROCm version, dtype, device, stream, and determinism
  flags.
- Record whether `torch.topk(sorted=True)` or `sorted=False` is used.
- Record that PyTorch tie behavior may not match lower-token-id ties unless
  repaired.
- Record whether `torch.multinomial` RNG generation and any CPU-GPU
  synchronization are included.
- Record allocator behavior and whether output tensors are preallocated.
- Record whether framework dispatch is part of the timed boundary.

This baseline is valid as an oracle and a practical reference. It is usually not
the strongest serving baseline when vLLM on ROCm, vLLM, or FlashInfer matches
the same contract.

## Result Classifications

Use these labels for baseline records:

| Classification | Meaning |
| --- | --- |
| `custom-win` | Custom kernel is correct and faster for the same contract. |
| `custom-specialized-win` | Custom kernel wins by exploiting documented fixed shape, fusion, layout, RNG, or omitted generality. |
| `near-match` | Performance is close enough to be useful, with no correctness gap. |
| `library-win` | Best matching library or serving path is faster. |
| `serving-win` | End-to-end serving path wins even if an isolated custom kernel is faster. |
| `negative-example` | Attempt is correct but slower, and the reason is useful corpus data. |
| `contract-mismatch` | Sortedness, ties, RNG, top-k/top-p, processors, or output materialization differ. |
| `comparison-incomplete` | Baseline path, workspace, RNG, processor, or launch boundary is missing. |
| `correctness-fail` | Output contract is violated. |
| `compile-fail` | Baseline or custom path does not build. |
| `template-only` | Guidance or scaffold only; no GPU measurement is attached. |

When no profiler counters are collected, say `timing-only`. Do not infer memory
bandwidth, occupancy, cache hit rate, or warp efficiency from timings alone.

# Road to SOTA Top-K

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This guide is for a project trying to build a custom HIP Top-K kernel that can
compete with strong library and serving-stack implementations. It is not a
claim that this corpus already has a SOTA Top-K kernel.

Current corpus anchor:

- `corpus/tasks/block-topk-sampling`
- `harnesses/topk_sampling_benchmark.hip`
- `docs/SELECTION_SAMPLING_KERNEL_GUIDE.md`
- `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`
- `docs/TOPK_IMPLEMENTATION_SURVEY.md`
- `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md`
- `data/index/selection_sampling_track.json`
- `data/index/topk_implementation_survey.json`
- `data/index/topk_baseline_harness_blueprint.json`
- `data/index/topk_wide_k_matrix.json`
- `data/index/topk_wild_ideas.json`
- `corpus/tasks/wide-k-topk-selection`
- `docs/TOPK_WILD_IDEAS_LAB.md`
- ROCm evidence status: no timing records are checked in yet. Run the HIP
  harness and same-contract PyTorch, hipCUB, vLLM on ROCm, FlashInfer, and
  OneFlow baselines on AMD hardware before making custom-win claims.

For the sibling project building Top-K now, the first route is:

1. Freeze the exact contract from `SOTA Target`.
2. Read `docs/TOPK_IMPLEMENTATION_SURVEY.md` and identify whether the target
   is selection, masking, sampling, routing, ragged indexing, or producer
   epilogue fusion.
3. Read `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md` before writing benchmark
   code or result JSON.
4. Extend the harness metadata in `Stage 0`.
5. Run same-hardware PyTorch, hipCUB, FlashInfer, and any relevant serving-stack
   baselines in `Stage 1`.
6. Use `corpus/tasks/wide-k-topk-selection` for any `k > 8` claim.
7. Specialize `k=1/2/4/8` in `Stage 2`, then cover
   `k=16/32/64/128/256/512/1024` before declaring the family broad.
8. Read `docs/TOPK_WILD_IDEAS_LAB.md` when the straightforward ladder stalls.
9. Only then chase fused sampling, ragged/segmented Top-K, and architecture
   sweeps.

## SOTA Target

For Top-K, "SOTA" must be scoped. A generic `torch.topk` replacement for every
dtype, shape, stride, sortedness, and tie policy is a different target from a
decode-time `top-k + softmax + sample` kernel for one vocab size.

Before optimizing, freeze:

- Operation: argmax, Top-K values, Top-K pairs, Top-K sampling, Top-P sampling,
  beam step, MoE expert routing, or segmented/ragged Top-K.
- Shape: rows, vocab or segment length, `k`, beam width, max ragged length, and
  whether dimensions are fixed at compile time.
- Layout: row-major contiguous logits, strided rows, padded vocab, packed
  segments, or page-table/ragged indirection.
- Dtype: FP32, FP16, BF16, FP8 score storage, INT logits after dequant, and
  accumulator/compare type.
- Output: values, indices, probabilities, logprobs, sampled ids, parent beam
  ids, routing weights, or only an in-register consumer.
- Ordering: sorted Top-K, unsorted Top-K, stable order, or only membership.
- Ties: lower token id, higher token id, first observed, stable input order, or
  unspecified.
- Exceptional values: NaN, +/-Inf, masked logits, all-masked rows, and
  duplicate equal scores.
- Sampling: temperature, Top-K before Top-P or after Top-P, RNG source,
  precomputed uniforms versus generated RNG, deterministic mode, and whether
  RNG time is included.
- Timing boundary: isolated kernel, full logits processor pipeline, framework
  dispatch, allocation, HIP Graph replay, or end-to-end decode step.

The most plausible custom wins come from fixed shape, small fixed `k`, fusion,
known layout, weaker output materialization, or lower launch overhead.

## Regime Map

| Regime | Common Shapes | First Custom Candidate | Strong Baseline |
| --- | --- | --- | --- |
| Argmax / `k=1` | Many rows, vocab 1K-256K | Warp or block pair reduction | hipCUB reduce/argmax, framework argmax |
| Tiny `k <= 8` | LLM sampling, MoE top-2/top-4, beam candidates | Per-thread register list plus warp/CTA merge | hipCUB `DeviceTopK`, PyTorch `topk`, FlashInfer/vLLM on ROCm sampling |
| Small `k <= 32` | Routing, reranking, decode sampling | Bitonic or compare-exchange network over candidates | hipCUB TopK or radix sort with post-filter |
| Medium `k` 32-256 | Reranking, retrieval, sparse attention blocks | Two-pass block candidates, heap/list hybrid, radix select | hipCUB TopK, hipCUB radix sort, PyTorch/ATen |
| Large `k` or `k/V` high | `k` hundreds to thousands | Radix sort/select, threshold partition, maybe full sort | hipCUB radix sort, rocThrust sort, PyTorch `topk` |
| Tiny vocab | Vocab/segment <= warp or CTA tile | Warp-per-row or CTA-per-row full network | Framework topk, hipCUB block primitives |
| Large vocab | 32K, 50K, 128K, 256K+ | Multi-CTA row, two-pass candidate staging, persistent rows | FlashInfer, vLLM on ROCm/vLLM, hipCUB |
| Many short rows | Small segments, large row count | Warp-per-row, multiple rows per CTA | hipCUB segmented primitives, framework batched topk |
| Few long rows | Batch 1-16 with large vocab | Multi-CTA per row, persistent CTA pool | Serving-stack sampler, hipCUB radix/select |
| Ragged / segmented | Request-specific vocab masks or page tables | Segmented Top-K with offset table and length guards | FlashInfer transforms, framework sort/topk |
| Top-K + softmax/sample | Decode token selection | Fused select, exp, sum, CDF, sample | vLLM on ROCm, vLLM, FlashInfer |
| Top-K + Top-P | Sampling with nucleus policy | Fused Top-K then Top-P prefix, or histogram/AIR-style path | vLLM on ROCm AIR Top-P, FlashInfer |

Treat these as starting guesses. The winning row/CTA mapping changes quickly
with `k`, vocab size, output contract, and GFX.

## Wide-K Coverage Matrix

The corpus tracks wide-k work in `data/index/topk_wide_k_matrix.json` and
`corpus/tasks/wide-k-topk-selection`. A Top-K implementation should not be
called broad until it has at least one correctness-checked and same-hardware
baseline result in each relevant regime.

| K Regime | Representative K | Primary Custom Family | Strongest First Baseline | Evidence Goal |
| --- | ---: | --- | --- | --- |
| Argmax | 1 | value-index reduction | PyTorch argmax/topk, hipCUB reduction | tie/mask/NaN-correct greedy path |
| Tiny K | 2, 4, 8 | register list, fixed network, warp/CTA merge | current seed, hipCUB DeviceTopK, FlashInfer sampling | specialize without register spills |
| Small K | 16, 32 | bitonic/top-k network, two-level merge | PyTorch topk, hipCUB DeviceTopK, hipCUB RadixSort | sorted versus unsorted boundary |
| Medium K | 64, 128, 256 | heap/list hybrid, two-pass candidates, radix-select | PyTorch topk, hipCUB DeviceTopK/RadixSort | list/network versus radix decision |
| Large K | 512, 1024, 4096 | radix-select, threshold partition, bucket/pivot search | hipCUB RadixSort, PyTorch topk | when full sort starts winning |
| Near-Full K | `k/n >= 0.25` | full sort or top-n-minus-bottom | hipCUB RadixSort, rocThrust, PyTorch sort/topk | classify as sort-dominated when true |
| Ragged K | 1..128 per segment | length-bucketed segmented scheduler | padded PyTorch, hipCUB segmented parts, FlashInfer transforms | avoid invalid padded/row-major comparisons |

The first benchmark batch should include:

- `k=1`: rows=4096, n=32000, sorted=false.
- `k=4` and `k=8`: rows=1024, n=32768, sorted=true.
- `k=16` and `k=32`: rows=512/256, n=32768, sorted=true.
- `k=64` and `k=128`: rows=256/128, n=65536, sorted=true.
- `k=256`, `k=512`, and `k=1024`: rows=64/32/16, n=131072,
  sorted=true.
- One high-ratio case such as rows=16, n=32768, k=8192.
- One ragged case with length buckets instead of padding-only measurement.

Use separate result classifications for each regime. A `k=4` win is not a
`k=128` claim, and a `k=1024` loss to full sort is useful evidence rather than
a failure of the overall track.

## Wild Ideas Mode

`docs/TOPK_WILD_IDEAS_LAB.md` and `data/index/topk_wild_ideas.json` collect
speculative Top-K optimization hypotheses. Use them when the normal algorithm
ladder has produced honest baselines and the project needs fresh directions.

The first hypotheses to test are:

- `multi-pivot-fence` for `k=64..512`.
- `unsorted-first-late-sort` for `k=16..64`.
- `complement-topk` for high `k/n`.
- `flashtopk-epilogue` if the project can move Top-K into the LM-head boundary.
- `adversarial-autotuner` before believing any speedup.

Every wild idea must carry an exactness class:

- `exact`: same Top-K result contract.
- `exact-after-repair`: fast first pass plus proof/repair.
- `contract-changing`: useful only when the caller accepts a different
  operation, such as exact categorical sampling instead of materialized Top-K.

Do not let a clever idea outrun the evidence labels. A wild idea that loses
cleanly is still high-value corpus material.

## Baseline Ladder

Measure the strongest same-contract baseline before calling a custom result a
win.

1. CPU or simple CUDA oracle
   - Defines ties, NaNs, masks, sortedness, and sampled-token behavior.
   - Not a performance target.

2. PyTorch/ATen
   - `torch.topk(sorted=True/False)` is the fastest practical oracle to wire up.
   - Pair with `torch.softmax`, `torch.multinomial`, or processor kernels only
     if the custom timing includes the same logical work.
   - Record allocator behavior, dispatch boundary, determinism flags, and any
     `.contiguous()` copy.

3. rocThrust
   - Useful for sort/partial-sort prototypes and iterator tricks.
   - Usually a weak production baseline if a pipeline launches several kernels
     or hides temporary allocation.

4. hipCUB / rocPRIM/hipCUB/rocThrust
   - `hipcub::DeviceTopK` is the first device-level Top-K baseline when unsorted
     output and its determinism contract are acceptable.
   - `hipcub::DeviceRadixSort` is the full-sort oracle and high-`k` baseline.
   - `hipcub::DeviceSelect` and scans are parts for threshold or Top-P sketches,
     not standalone Top-K.
   - `hipcub::WarpReduce`, `BlockReduce`, `BlockLoad`, and `BlockScan` are also
     parts bins inside a fused custom kernel.

5. vLLM on ROCm
   - Baseline for serving Top-K/Top-P sampling, beam helpers, finished-state
     handling, runtime arguments, and rocRAND/CURAND state.
   - Compare isolated kernels only when the vLLM on ROCm path is invoked at the
     same boundary and feature set.

6. vLLM
   - Baseline for end-to-end serving behavior and the actual sampler path chosen
     at runtime: PyTorch, Triton, FlashInfer, or custom HIP.
   - Record whether per-request generators, processed logits, logprobs, or
     feature flags force fallback.

7. FlashInfer
   - Primary inference sampling and Top-K/Top-P baseline.
   - Important for logits/probabilities contract, deterministic mode, tie-break
     modes, ragged/page-table transforms, and multi-CTA or filtered Top-K paths.

8. OneFlow / framework extension path
   - Relevant when the target is a framework custom op, stream/stride boundary,
     or dispatch-cost comparison rather than an isolated kernel.
   - Use OneFlow/PyTorch extension anatomy to measure whether custom HIP wins
     after framework integration costs are included.

## Current External Snapshot

Verified on 2026-05-21. Re-check before treating any item as "latest."

| Reference | Why It Matters |
| --- | --- |
| `docs/TOPK_IMPLEMENTATION_SURVEY.md` | Local source-inspection survey of PyTorch/ATen, hipCUB/rocPRIM, vLLM on ROCm, FlashInfer, vLLM, OneFlow, Transformer Engine, and Composable Kernel Top-K-like implementations. Read it before picking baselines. |
| `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md` | Practical harness blueprint for benchmark shape IDs, timing boundaries, correctness modes, backend recipes, and result JSON fields. |
| [CUB DeviceTopK](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceTopK.html) | CUDA-origin API reference for a CUB/hipCUB-style unordered Top-K contract. Verify current ROCm support before treating it as a ROCm baseline. |
| [PyTorch CUDA Top-K dispatch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cpp) | Shows that ATen gathers Top-K first and sorts selected results only when requested; framework dispatch, allocations, and tie instability must be recorded. |
| [PyTorch CUDA Top-K kernels](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu) | Source for radix kth-value selection, single-block and multi-block gather paths, and backend-specific choices. |
| [FlashInfer sampling API](https://docs.flashinfer.ai/api/sampling.html) | Primary inference baseline for sampling from logits/probabilities, Top-K, Top-P, Top-K+Top-P, min-p, renormalization, and speculative sampling helpers. |
| [FlashInfer Top-K API](https://docs.flashinfer.ai/generated/flashinfer.top_k.html) | Documents radix Top-K, unsorted fast default, deterministic mode, tie-break modes, sorted fallback behavior, and small-vocab caveats. |
| [FlashInfer `top_k_top_p_sampling_from_logits`](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html) | Fused logits-to-sample path with explicit `filter_apply_order`, deterministic mode, seed/offset handling, and HIP Graph compatibility notes. |
| [vLLM ROCm support](https://docs.vllm.ai/en/latest/getting_started/installation/amd.html) | Serving-stack setup and backend-selection context for ROCm runs. Measure the actual sampler path selected in your environment. |
| [vLLM Top-K/Top-P Triton path](https://docs.vllm.ai/en/v0.20.0/api/vllm/v1/sample/ops/topk_topp_triton/) | Current public vLLM doc path that references a Qrita-inspired Top-K/Top-P implementation. Measure the actual selected backend in your run. |
| [vLLM Top-K/Top-P Triton source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_triton.py) | Qrita-inspired pivot masking path with duplicate-boundary handling; it is not the same contract as sorted materialized Top-K. |
| [PyTorch `torch.topk`](https://docs.pytorch.org/docs/2.12/generated/torch.topk.html) | Convenient framework oracle/baseline; useful but not usually the strongest serving baseline. |
| [Qrita](https://arxiv.org/abs/2602.01518) | 2026 pivot/truncation Top-K and Top-P paper reporting up to 2x throughput versus vLLM, SGLang, and FlashInfer baselines with deterministic output. Treat as a research target to reproduce. |
| [FlashSampling](https://arxiv.org/abs/2603.15854) | 2026 exact sampling work emphasizing memory-efficient decode-regime sampling and fused matmul-sample directions. Relevant if the project can change the logits materialization boundary. |
| [RTop-K](https://arxiv.org/abs/2409.00822) | Row-wise GPU Top-K paper that motivates pivot/search alternatives beyond sort and bitonic networks. |
| [RadiK](https://arxiv.org/abs/2501.14336) | Radix Top-K selection work for larger `k`, input length, and batch-size scalability. |
| [FAISS GPU k-selection paper](https://arxiv.org/abs/1702.08734) | Classic register-heavy GPU k-selection design reference for nearest-neighbor workloads; not an LLM sampler baseline but still valuable for small-k thinking. |

Relevant local references:

- `docs/TOPK_IMPLEMENTATION_SURVEY.md`
- `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md`
- `data/index/topk_implementation_survey.json`
- `data/index/topk_baseline_harness_blueprint.json`
- `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`
- `docs/ROCPRIM_COMPETITION_TRACK.md`
- `third_party/rocm-libraries` for rocPRIM, hipCUB, and rocThrust primitives
- `third_party/vllm/csrc/topk.cu`
- `third_party/vllm/csrc/persistent_topk.cuh`
- `third_party/vllm/csrc/sampler.cu`
- `third_party/flashinfer/include/flashinfer/topk.cuh`
- `third_party/flashinfer/include/flashinfer/sampling.cuh`
- `third_party/flashinfer/include/flashinfer/air_top_p.cuh`
- `third_party/composable-kernel` for producer-side fusion and epilogue references
- CUDA-origin TensorRT-LLM, CUB/CCCL, OneFlow, and Transformer Engine source
  paths in `docs/TOPK_IMPLEMENTATION_SURVEY.md` as algorithm contrast only

## Algorithm Ladder

Climb the ladder one rung at a time. Keep losing attempts when they teach the
boundary.

### 0. Full Sort Baseline

Sort the entire row or segment, then slice `k`.

Use it for:

- Correctness oracle.
- Sorted-output contract.
- High `k` or high `k/V`.
- Stressing tie and NaN policies.

Expected weakness: it does unnecessary work for small fixed `k`.

### 1. Serial Row Selection

One thread scans one row and maintains a local sorted list.

Use it for:

- Tiny rows.
- Debug oracle.
- Launch-bound measurements.

Expected weakness: no row-level parallelism. The CUDA-origin corpus seed showed
this losing badly for rows=1024, vocab=32768, k=4; rerun on AMD hardware before
using that as ROCm evidence.

### 2. Per-Thread Register List Plus Warp Merge

Each lane scans a strided slice and keeps `k` local candidates. Warp shuffles or
hipCUB `WarpReduce` merge candidates.

Use it for:

- `k <= 8`.
- Vocab up to a few thousand per row.
- Many rows where warp-per-row provides enough parallelism.
- Argmax, top-2 routing, and tiny beam expansions.

Watch:

- Register pressure from `k` values and `k` indices.
- Branchy insertion logic.
- Tie policy in lane merges.
- Occupancy loss when unrolling too aggressively.

### 3. CTA-Per-Row Candidate Merge

Each CTA scans a row, writes thread-local candidates to shared memory, and
reduces/merges them.

Use it for:

- Vocab 16K-128K.
- `k <= 8` to `k <= 32`.
- Top-K plus softmax/sample in one launch.

Watch:

- Shared memory footprint: `block_threads * k * (value + index)`.
- Merge complexity: binary tree of `2k` candidate inserts can dominate for
  larger `k`.
- Bank conflicts and dynamic shared-memory limits.
- Block size sensitivity across `gfx90a`, `gfx1030`, and `gfx1100`.

The current seed is here: one CTA per row, 256 threads, `k <= 8`, shared-memory
candidate merge, then one thread writes Top-K and samples.

### 4. Bitonic / Top-K Networks

Use fixed compare-exchange networks over candidate tiles instead of repeated
insertion.

Use it for:

- Fixed `k` and fixed tile sizes.
- Sorted output.
- Warp-level or CTA-level candidate merge.

Watch:

- Networks can do more compares than a data-dependent insertion list for very
  small `k`.
- The best network differs for `k=4`, `k=8`, `k=16`, and `k=32`.
- Deterministic tie breaks need pair compares, not value-only compares.

### 5. Heap / Selection Hybrid

Maintain a bounded min-heap, sorted list, or loser tree per thread/warp/block.

Use it for:

- Medium `k`.
- Larger candidate streams where full sorting is still wasteful.
- Cases where sorted final output is required.

Watch:

- Heap updates are branchy and pointer-like.
- Register heaps spill quickly.
- Shared-memory heaps need careful synchronization.

### 6. Radix Select / Threshold Partition

Find a score threshold by radix passes or histogram bins, compact candidates
above the threshold, then repair boundary ties/order.

Use it for:

- Large `k`.
- Large vocab.
- Top-P or thresholded workflows.
- BF16/FP16/FP32 scores that can be transformed into sortable keys.

Watch:

- Float key transforms must preserve desired order.
- Boundary buckets can overflow candidate buffers.
- Ties around threshold are correctness traps.
- Extra passes over logits may lose to a simpler one-pass kernel for tiny `k`.

### 7. Two-Pass Block Candidates

First pass emits per-block Top-K candidates for each row. Second pass merges
the much smaller candidate set.

Use it for:

- Very large vocab or few long rows.
- Multi-CTA per row.
- Occupancy recovery when one CTA per row is not enough.

Watch:

- Temporary candidate buffer size.
- Extra global writes and reads.
- Launch overhead unless graph-captured or fused into a serving step.
- Deterministic merge order across CTAs.

### 8. Persistent / Streaming Top-K

Persistent CTAs pull rows or row tiles from a queue, keep request metadata hot,
and handle small batches without paying one launch per tiny operation.

Use it for:

- Decode batches with variable active requests.
- Few rows with large vocab.
- Serving systems where scheduler overhead matters.

Watch:

- Fairness and queue-depth effects.
- Cooperative launch requirements.
- Deadlock-free work queues.
- Integration with HIP Graphs and stream priorities.

### 9. Fused Sampling Kernel

Fuse logits processors, Top-K, temperature scaling, softmax over selected
candidates, CDF construction, and sampling.

Use it for:

- LLM decode where materialized Top-K values are not needed by later code.
- Per-request policies grouped by common parameters.
- Fixed vocab and fixed `k`.

Watch:

- `expf` cost and approximation choices.
- RNG state and deterministic replay.
- Whether probabilities/logprobs must be returned.
- Exact filter order for Top-K plus Top-P.
- Statistical equivalence versus exact sampled-token equivalence.

## GPU-Specific Considerations

### GFX Targets

- `gfx90a`: A100-class CDNA2/RDNA2. Use as an architecture-distinct data-center
  baseline. `cp.async` exists, but pure Top-K usually streams each logit once,
  so async shared-memory staging is not automatically useful.
- `gfx1030`: RDNA2-class consumer/pro target. Re-tune block size and shared-memory
  use; do not import A100 occupancy assumptions.
- `gfx1100`: RDNA3 inference target. Compare MIGraphX/vLLM/FlashInfer paths because
  serving-stack tactics can be strong.
- `gfx942`: portable CDNA3. Keep it separate from `gfx942` records.
- `gfx942`: CDNA3-specific TMA/WGMMA suffix target. TMA is more relevant for
  regular multidimensional tiles than for a single streaming logits row, but it
  may matter for fused kernels that load structured metadata, page tables, or
  reused blocks.
- `gfx950` / CDNA4/RDNA4: verify exact toolkit flags and library support before
  making suffix-specific claims. Re-run baselines after CUDA, MIGraphX,
  FlashInfer, vLLM, hipCUB, or compiler upgrades.

### Warp and Block Mechanics

- Use pair comparisons `(value, index)` everywhere ties matter.
- Prefer warp shuffles or hipCUB warp/block collectives before inventing fragile
  reductions unless the custom tree is the experiment.
- Consider warp-per-row for small vocab and many rows.
- Consider CTA-per-row for LLM vocab and tiny `k`.
- Consider multi-CTA rows when `rows` is small and vocab is large.
- Avoid global atomics for allocating result slots unless output density is
  provably tiny. The corpus has a select/compact negative example where global
  atomics lose as density rises.
- Use scans for compaction or Top-P prefix workflows when the selected set is
  variable-sized.

### Memory and Occupancy

- Pure Top-K is usually bandwidth dominated plus compare dominated, not data
  reuse dominated.
- Coalesced row-major reads matter more than shared-memory staging when each
  logit is consumed once.
- Candidate storage controls occupancy: local arrays, pair structs, unrolled
  networks, and output buffers raise registers quickly.
- Shared memory scales with `threads * k * sizeof(pair)`. That blocks larger
  `k` before arithmetic does.
- Always record amdclang++ resource/ISA registers, spills, static shared memory, dynamic shared
  memory, block size, and occupancy estimate.
- Vectorized loads need alignment and enough contiguous work; do not assume
  wider loads are faster without measurement.

### Determinism

Deterministic Top-K is a contract, not a free property.

Record:

- Tie rule.
- Whether unsorted output is acceptable.
- Whether equal values can return any valid top-k set.
- Whether sampled ids must match a fixed uniform stream or only pass
  distribution tests.
- Whether multi-CTA merge order can change boundary ties.
- Whether NaNs are ignored, treated as `-inf`, propagated, or unspecified.

## Benchmark Protocol

Use the corpus timing rules:

- HIP events around the measured GPU work.
- Warmup before timing.
- Median, p10, p90, min, and max.
- Allocation outside timing unless allocation is part of the target boundary.
- Same stream, same inputs, same output contract, same build flags.
- Say `timing-only` when there are no rocprofiler/rocprof counters.
- Say `negative example` when a correct attempted optimization does not improve
  speed.
- Keep hardware and build metadata attached to every measured claim.

Minimum shape matrix:

| Purpose | Rows | Vocab / Segment | K | Notes |
| --- | ---: | ---: | ---: | --- |
| Corpus seed | 1024 | 32768 | 4 | Repeat for same-hardware library baselines |
| Greedy/argmax | 4096 | 32000 | 1 | Warp and CTA variants |
| Large LLM vocab | 256 | 128256 | 8 | Current task recommended shape |
| Awkward vocab | 128 | 1009 | 4 | Non-power-of-two path |
| Tiny row | 65536 | 64-512 | 1-8 | Warp-per-row, multiple rows per CTA |
| Few long rows | 1-32 | 128K-256K | 4-32 | Multi-CTA and persistent variants |
| Medium K | 256-4096 | 32K-128K | 32-128 | Heap/network/radix boundary |
| Large K | 16-1024 | 32K-128K | 512+ | Radix/full-sort boundary |
| Ragged | variable | 1-128K | 1-32 | Offset table and length guards |
| Top-K+Top-P | 1-4096 | 32K-128K | 1-64 | Filter order and probability tests |

Report:

- Rows, vocab or segment lengths, `k`, dtype, temperature, Top-P threshold, and
  mask density.
- Sortedness, tie policy, NaN policy, RNG policy, and output materialization.
- Baseline name, version or commit, selected backend, workspace bytes, and
  launch count.
- Build command, `hipcc` version, driver, ROCm runtime, `-gencode` flags, amdclang++ resource/ISA
  resource output, and GPU gfx target.
- Correctness mismatches, maximum value/probability error, sampled-token
  mismatch or statistical test result.
- Whether timing includes logits processors, softmax, RNG, allocation, graph
  replay, framework dispatch, or host synchronization.

## Correctness and Adversarial Cases

Top-K bugs often pass random tests. Add adversarial rows:

- All equal logits.
- Repeated ties across the `k` boundary.
- Ascending, descending, and nearly sorted logits.
- Max values at the first and last columns.
- Non-power-of-two vocab and rows.
- `k=1`, `k=2`, `k=8`, `k=blockDim`, and `k` near vocab.
- All masked, one unmasked, and mask-density sweeps.
- `+Inf`, `-Inf`, NaN, denormals, and extreme negative logits.
- Temperature near 0, 1, and high values.
- Uniforms exactly 0, exactly 1, and exactly on CDF boundaries.
- Top-P thresholds where the cumulative sum lands exactly on `p`.
- BF16/FP16 cases with close scores that round to the same value.
- Ragged segments with length 0, 1, `k`, `k-1`, and very long length.
- Multi-CTA rows where boundary candidates come from different CTAs.

For sampling, decide whether correctness means exact token equality for a fixed
uniform stream or distributional equivalence across many samples. Do not mix
the two in one result.

## Staged Experiments

### Stage 0: Contract and Harness

- Extend the current harness to support sorted and unsorted Top-K modes.
- Add optional masks, NaNs, `+Inf`, `-Inf`, and ragged offsets.
- Add PyTorch/ATen and hipCUB correctness-oracle paths.
- Record amdclang++ resource/ISA output for the existing seed.

Exit criteria:

- CPU, framework, and custom outputs agree for deterministic contracts.
- Result JSON includes full contract metadata.

### Stage 1: Same-Hardware Baselines

- Measure PyTorch `torch.topk` plus optional softmax/sample.
- Measure hipCUB `DeviceTopK` for unsorted membership.
- Measure hipCUB `DeviceRadixSort` for sorted oracle.
- Measure FlashInfer Top-K/sampling where the contract matches.
- Try vLLM on ROCm and vLLM only when the actual serving boundary can be made
  explicit.

Exit criteria:

- The current HIP seed is classified against at least hipCUB and one inference
  baseline on AMD hardware, or marked `comparison-incomplete` with blockers.

### Stage 2: Tiny-K Kernel Family

- Specialize `k=1`, `k=2`, `k=4`, and `k=8` at compile time.
- Compare insertion list, compare-exchange network, and hipCUB collectives inside
  a custom kernel.
- Test warp-per-row, CTA-per-row, and multiple rows per CTA.
- Keep sorted and unsorted variants separate.

Exit criteria:

- Best tiny-`k` variant is selected per vocab range and GFX.
- Any losing variant records register/shared-memory reason when known.

### Stage 3: Large-Vocab and Few-Row Family

- Add multi-CTA-per-row first pass with per-tile candidates.
- Add second-pass merge kernel and graph-captured two-pass timing.
- Test persistent CTA scheduling for small active row counts.
- Sweep vocab 32K, 50K, 128K, and 256K.

Exit criteria:

- Identify the row count and vocab size where one CTA per row stops scaling.

### Stage 4: Medium/Large-K Family

- Implement heap/list hybrid for `k=32..256`.
- Implement radix-select or histogram threshold candidate generation.
- Compare against hipCUB radix sort and PyTorch `topk(sorted=True/False)`.
- Track candidate-buffer overflow and threshold repair costs.
- Extend the sweep through `k=512`, `k=1024`, and one high `k/n` case to find
  the full-sort crossover.

Exit criteria:

- A decision boundary for network/list versus radix/full sort is measured.
- The wide-k task has one recorded result for each covered k regime, even if
  the result is a library win or negative custom example.

### Stage 5: Fused Sampling

- Fuse temperature, Top-K, softmax over selected candidates, CDF, and sample.
- Add optional output modes: sampled id only, Top-K ids, Top-K values,
  probabilities, and logprobs.
- Add Top-K then Top-P path with explicit filter order.
- Add RNG policies: precomputed uniforms, rocRAND/CURAND state, seed/offset arrays,
  deterministic replay.

Exit criteria:

- Isolated kernel and serving-boundary timings are recorded separately.
- Sampling correctness mode is explicit.

### Stage 6: Segmented and Routing Variants

- Add segmented Top-K with offsets and lengths.
- Add MoE top-2/top-4 routing variant with bias, score function, and weight
  normalization.
- Add page-table transform style output for sparse attention or retrieval.

Exit criteria:

- The guide can distinguish row-major vocab Top-K from ragged/segmented Top-K
  without reusing invalid baselines.

### Stage 7: Architecture Sweep

- Run an available `gfx1030` or newer AMD target first, then keep architecture
  claims scoped to that gfx target.
- Add `gfx90a`, `gfx1100`, `gfx942` or `gfx942`, and CDNA4/RDNA4 `gfx950` or
  available target as hardware allows.
- Rebuild with exact AMD GCN ISA targets and record LLVM IR / AMD GCN ISA fallback policy.
- For CDNA3/CDNA4/RDNA4, do not claim TMA/WGMMA relevance unless AMD GCN ISA or source
  actually uses those features.

Exit criteria:

- Each GFX has best parameters, amdclang++ resource/ISA resources, and same-contract baseline
  timings.

### Stage 8: Counter-Backed Evidence

- Run rocprofiler/rocprof where counter permissions allow it.
- Collect memory throughput, achieved occupancy, register spills, branch
  efficiency, shared-memory conflicts, and instruction mix.
- If counters are blocked, store `profile-attempted-blocked` and keep timing
  claims labeled `timing-only`.

Exit criteria:

- Timing-only hypotheses are either counter-backed or clearly marked as pending.

### Stage 9: Wild-Idea Lab

- Pick one idea from `data/index/topk_wild_ideas.json`.
- Add `idea_id`, `contract_class`, `k_regime`, and `repair_rate` fields to the
  result artifacts.
- Run one friendly, one random, and one adversarial shape before optimizing.
- Compare against the strongest matching library baseline.
- Preserve the result as `hypothesis-supported`, `negative-example`,
  `contract-changing`, or `comparison-incomplete`.

Exit criteria:

- The idea either graduates into a task/harness variant or is documented as a
  negative example with the reason it failed.

## Decision Rules

- If `k <= 8`, start with compile-time specialized register lists and
  warp/CTA merges.
- If sorted output is not needed, measure an unsorted path and hipCUB `DeviceTopK`
  before paying sort cost.
- If Top-K feeds sampling immediately, avoid materializing full candidate data
  unless downstream code requires it.
- If batch rows are many and vocab is moderate, prefer one warp or one CTA per
  row over multi-CTA rows.
- If batch rows are few and vocab is huge, test multi-CTA and persistent
  variants early.
- If `k` is large, stop trying to stretch tiny-`k` insertion code; move to
  radix/select/full-sort baselines.
- If the custom kernel loses to a serving stack, record which generality the
  serving stack handled and which narrower fused path remains plausible.
- If an optimization is correct but slower, preserve it as a negative example.

## Claim Checklist

A Top-K result is ready for the corpus only when it says:

- Exact operation and timing boundary.
- Shape, dtype, layout, alignment, and output contract.
- Sortedness, tie, mask, NaN, and RNG policy.
- Hardware, driver, ROCm toolkit, compiler, and `hipcc` flags.
- Baseline names, versions/commits, selected backend, and workspace policy.
- Correctness oracle and adversarial cases.
- Median/p10/p90/min/max timing.
- Evidence label: `timing-only`, `counter-backed-measured`,
  `profile-attempted-blocked`, or `negative example`.
- Classification: `custom-win`, `custom-specialized-win`, `near-match`,
  `library-win`, `serving-win`, `contract-mismatch`,
  `comparison-incomplete`, `correctness-fail`, or `compile-fail`.

The north star is not "rewrite `topk` because libraries exist." It is to find
the exact narrow contract where a custom HIP kernel can reduce work, fuse
passes, avoid overhead, or teach why the library remains the right baseline.

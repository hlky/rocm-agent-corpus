# Top-K Wild Ideas Lab

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This is the deliberately speculative companion to `docs/ROAD_TO_SOTA_TOPK.md`.
It is the "idiot-savant mode" shelf: odd, sharp hypotheses that might uncover a
custom Top-K win, plus the benchmark gates that keep the corpus honest.

These are not claims. Treat every item as `hypothesis-only` until it has:

- an exact contract,
- a correctness oracle,
- same-hardware PyTorch/hipCUB/FlashInfer or serving-stack baselines,
- k-regime coverage from `data/index/topk_wide_k_matrix.json`,
- timing labels, and profiler labels when counters exist.

## Grounding Snapshot

Verified on 2026-05-20:

- hipCUB `DeviceTopK` is AIR TopK-based, linear-work, and returns unordered output
  with non-guaranteed determinism for tied boundary cases.
- Qrita uses sigma-truncation and quaternary pivot search to reduce Top-K and
  Top-P work while preserving deterministic output.
- FlashSampling shows a stronger boundary shift for sampling: avoid
  materializing logits, fuse sampling into the LM-head matmul, and keep only
  tile/group winners on chip.
- vLLM has a Qrita-inspired Top-K/Top-P Triton path. Always record the actual
  selected backend.
- vLLM on ROCm sampling currently matters as a serving-stack comparison, not
  just as isolated kernels.
- Split-bucket partition work is a useful reminder that bucket rules, local
  per-bucket selection, and merge policy can be treated as one execution model.
- Cross-layer sparse-attention work such as Kascade/IndexCache suggests a
  different axis: reuse Top-K indices across layers when the model tolerates or
  validates that reuse.

Useful links live in `docs/ROAD_TO_SOTA_TOPK.md` and
`data/sources/core_resources.json`.

Additional wild-idea references:

| Reference | Use In This Lab |
| --- | --- |
| [Split-bucket partition](https://link.springer.com/article/10.1007/s11227-024-06031-x) | Bucket/select/merge execution models and fused bucket experiments. |
| [Foundations of Top-k Decoding](https://arxiv.org/abs/2505.19371) | Decoding-policy semantics; keep policy changes separate from exact Top-K selection. |
| [Probability simplex decoding](https://huggingface.co/papers/2602.18292) | Simplex/projection view for contract-changing decoder experiments. |
| [Kascade](https://arxiv.org/abs/2512.16391) | Top-K index reuse in sparse attention with quality labels. |
| [IndexCache](https://arxiv.org/abs/2603.12201) | Cross-layer Top-K index reuse and trace-driven validation plans. |

## How to Read This

Every idea has three labels:

- `exact`: can preserve the same Top-K contract.
- `exact-after-repair`: first pass may be approximate or lossy, but a repair
  pass proves the final answer.
- `contract-changing`: only valid when the caller accepts a different operation,
  such as exact categorical sampling without materialized Top-K.

The right response to a weird idea is not belief. It is a tiny harness, one
adversarial test, one same-hardware baseline, and a saved negative example if
it loses.

## Hypothesis Catalog

| Idea | Regime | Contract | Why It Might Work | First Experiment |
| --- | --- | --- | --- | --- |
| Multi-pivot fence | `k=64..4096` | exact-after-repair | Count several thresholds per pass to shrink candidates faster than binary pivot search | Four pivots over FP32 logits, then repair ties around kth |
| Exponent-first radix | `k=64..4096` | exact-after-repair | FP logits often separate by exponent/sign before mantissa work | Count high-order sortable-key bins, compact one bin, repair mantissa |
| Complement Top-K | high `k/n` | exact | When `k` is near `n`, selecting bottom `n-k` can be cheaper than selecting top `k` | Compare `k=8192,n=32768` Top-K vs bottom-24576 complement variants |
| Unsorted-first late-sort | `k=2..256` | exact when sorted repair included | Many consumers need membership or sampling, not globally sorted candidates | Time unsorted candidate path plus optional final k-sort |
| FlashTopK epilogue | decode LM-head | contract-changing or exact Top-K if values emitted | Avoid logits write/read by emitting per-tile candidates from GEMM epilogue | Composable Kernel/hipBLASLt epilogue sketch that writes tile candidates only |
| Temporal candidate cache | decode loops | exact-after-repair | Consecutive decode steps often reuse hot vocab regions or pivots | Reuse prior-step block thresholds, then full validation/repair |
| Hot-block vocab layout | fixed model serving | exact if permutation tracked | Put frequent/high-variance vocab blocks together to improve early candidate discovery | Offline vocab permutation plus inverse-index output repair |
| K-regime JIT ladder | all | exact | Runtime dispatch among specialized kernels beats one generic kernel | hipRTC/cache variants for k=1,2,4,8,16,32,64,128,256 |
| Boundary-tie microkernel | all sorted deterministic | exact | Handle common no-tie rows fast; invoke tie repair only for rare boundary duplicates | Fast path ignores equal-boundary repair, records rows needing repair |
| Ragged length buckets | ragged k | exact | Scheduling by segment length avoids padding and load imbalance | Bucket segments by length, run warp/CTA/multi-CTA variants per bucket |
| Score-bin bitset select | `k=16..256` | exact-after-repair | Bitsets/ballots can compact threshold candidates with less shared-memory state | Per-tile score bins plus warp ballot candidate emit |
| Two-pass top-k + CDF | sampling | exact for top-k sampling if threshold repair exact | Avoid materializing sorted k values when only a sampled id is needed | Find kth threshold, sum selected exp values, sample in second pass |
| Confidence-gated argmax | greedy or relaxed sampling | contract-changing | Very peaked rows may not need broad Top-K work | Only valid for greedy/approx modes; compare quality and exactness separately |
| Grouped tensor-parallel merge | multi-GPU/vocab shards | exact | Keep local shard candidates, merge only small candidate lists across ranks | RCCL allgather of per-shard top-k candidates plus deterministic merge |
| Adversarial autotuner | all | exact testing tool | Generate rows that break each trick before benchmarks look good | Synthesize ties, NaNs, high-k/n, flat logits, and ragged extremes |
| Low-precision prefilter | `k=16..1024` | exact-after-repair | Use half/int8/fp8 shadow scores to prune, then reload exact scores for repair | Keep top-M by approximate score, prove/repair against exact FP32 |
| Block upper-bound pruning | decode LM-head or cached blocks | exact-after-repair | Per-vocab-block bounds can skip blocks whose max possible score cannot beat kth | Store block max/norm bounds, scan only live blocks, validate skipped blocks |
| Max-K superset cache | mixed per-row k | exact | Compute `Kmax` once for rows with many k values, then slice smaller k outputs | Group rows by `Kmax`, compare with separate per-k kernels |
| Policy bucket scheduler | serving batch | exact | Group rows by k, sortedness, temperature, and output mode before launching kernels | Compare grouped dispatch against one generic mixed-policy kernel |
| Deterministic packed key | sorted deterministic | exact | Pack score bits and inverted token id into sortable 64-bit keys for library or custom radix paths | Validate ties/NaNs and measure radix-sort overhead |
| In-warp reservoir duel | tiny/small k | exact-after-repair | Lanes keep local reservoirs, duel pairwise, and emit only contested candidates | Compare against insertion lists and bitonic networks for k=4..32 |
| Loser-tree block merge | medium k | exact | Merge block candidates with a tournament/loser tree instead of repeated full k-list insertion | Second-pass merge benchmark for k=64..256 |
| Persistent Top-K service | small batch decode | exact | Keep resident CTAs and request metadata alive to avoid launch/scheduler overhead | Queue-driven decode microbench with graph and non-graph baselines |
| Cluster DGFX candidate ring | CDNA3/CDNA4/RDNA4 | exact | Use cluster/distributed shared memory to merge candidates across CTAs before global writes | CDNA3-only prototype for few rows, huge vocab |
| cp.async score conveyor | large contiguous rows | exact | Async copy score tiles into shared memory to overlap load and candidate merge | Prove whether staging helps or just adds shared-memory traffic |
| Token-class split lanes | fixed model serving | exact-after-repair | Route frequent, control, numeric, and rare token classes through different candidates and repair globally | Per-class top-m plus global merge with exact validation |
| Gumbel race sampler | sampling | contract-changing | Sample via `argmax(logit + gumbel)` or top-k Gumbel candidates instead of materialized probabilities | Statistical tests against exact categorical/top-k sampling |
| Sparsemax/simplex decoder | decoding policy | contract-changing | Treat Top-K/Top-P/min-p as projections or thresholded simplex operations with one generic threshold kernel | Implement threshold solver and classify as policy experiment |
| Draft-token candidate reuse | speculative decoding | exact-after-repair | Draft model candidates predict verifier top-k regions for accepted/rejected spans | Trace hit-rate benchmark with full verifier repair |
| Model-trace offline tuner | all serving | testing tool | Use real logits traces to choose per-regime kernels, pivots, and buckets instead of synthetic-only tuning | Train no model; just mine traces for dispatch rules and holdout results |
| Split-bucket one-iteration kernel | `k=64..4096` | exact-after-repair | Bucket, locally select, and merge in one fused execution step | Implement one SBP-style iteration and compare to multi-kernel bucket/select |
| Cross-layer Top-K index reuse | sparse attention | contract-changing or exact-after-repair | Reuse anchor-layer Top-K indices in nearby layers or validate them cheaply | Measure reuse hit rate, quality drift, and repair cost on sparse attention traces |

## Idea Details

### Multi-Pivot Fence

Hypothesis: for medium and large `k`, count several candidate thresholds in one
pass rather than iterating one pivot at a time. For each row, estimate a score
range, choose four or eight pivots, count elements above each pivot, then
narrow the interval that contains the kth score.

Why it might work:

- It reduces global passes for large vocab rows.
- It maps to warp/block counters and avoids maintaining huge candidate arrays.
- It aligns with Qrita-style multi-pivot thinking while leaving room for CUDA
  C++ specialization.

Why it might fail:

- Counts are cheap, but extra passes over logits are not.
- Flat distributions and duplicate boundary values require repair.
- Pivot choice can be bad for skewed or quantized logits.

Benchmark gate:

- Compare against hipCUB `DeviceTopK`, hipCUB `DeviceRadixSort`, PyTorch `topk`, and
  a one-pivot threshold implementation for `k=64,128,256,512,1024`.

### Exponent-First Radix

Hypothesis: for FP32/BF16/FP16 scores, perform a cheap high-bit pass over
sortable transformed keys, then run exact selection only inside the winning
exponent/mantissa bucket.

Why it might work:

- Many LLM logits have a small high-score region.
- High-order key bins can cheaply separate obvious losers.
- Boundary repair can preserve exactness.

Why it might fail:

- Temperature, penalties, or quantized scores may flatten bins.
- If most candidates land in one high bucket, the pass is wasted.
- NaNs, signed zero, and tie policy need explicit key transforms.

Benchmark gate:

- Track candidate count after each bin pass and classify rows where the pass
  fails to shrink the search space.

### Complement Top-K

Hypothesis: when `k/n` is high, compute the bottom `n-k` and derive the top-k
complement, or stop pretending selection beats full sort.

Why it might work:

- A `k=8192,n=32768` sorted Top-K writes a large output anyway.
- If `n-k` is small, bottom selection may have smaller candidate state.
- It provides a clean crossover test against full radix sort.

Why it might fail:

- Producing sorted top-k after complement may still require sorting most of the
  row.
- Memory movement can dominate.

Benchmark gate:

- Include `k/n` values `0.1`, `0.25`, `0.5`, and `0.9`; record when full sort
  wins and mark it as a library-win if appropriate.

### Unsorted-First Late-Sort

Hypothesis: many Top-K consumers only require candidate membership or sampling
over the selected set. Return unsorted candidates first; sort only in a repair
or output-materialization phase when the contract demands it.

Why it might work:

- hipCUB `DeviceTopK` is already an unsorted baseline; matching that contract can
  avoid waste.
- Top-K sampling can often compute softmax and CDF over unordered candidates.
- Sorted final output cost is only paid for callers that actually need it.

Why it might fail:

- Existing framework APIs often promise sorted outputs by default.
- Deterministic ties become harder if the unsorted candidate set is unstable.

Benchmark gate:

- Every result must say `sorted=false`, `sorted=true-with-repair`, or
  `contract-mismatch`.

### FlashTopK Epilogue

Hypothesis: Top-K after an LM-head GEMM is often a memory-traffic problem, not
a selection problem. Let each GEMM tile keep local candidates and write only
per-tile candidate lists instead of materialized logits.

Why it might work:

- It follows the FlashSampling lesson: move the boundary before logits hit HBM.
- Candidate output is tiny compared with `[batch, vocab]` logits.
- It can be exact if the hierarchical merge sees every tile winner list.

Why it might fail:

- hipBLASLt epilogues may not expose enough custom state for true candidate
  lists; Composable Kernel may be the practical route.
- Register pressure in GEMM epilogue may reduce Tensor Core throughput.
- Top-K plus logprob output may still need values for many tokens.

Benchmark gate:

- Compare against materialized LM-head GEMM plus Top-K, not just isolated
  Top-K. Keep a separate isolated-kernel result for diagnosis.

### Temporal Candidate Cache

Hypothesis: in decode, prior-step winners, high-score vocab blocks, or pivot
thresholds are useful hints for the next step. Reuse hints, then validate or
repair exactly.

Why it might work:

- Request-local distributions often have repeated frequent tokens or stable
  high-logit regions.
- A hint can seed pivots or block scheduling before scanning the full row.

Why it might fail:

- Exact Top-K still requires proving no unseen block beats the candidates.
- Cache misses may add overhead and branch divergence.
- Model behavior can change after punctuation, tool calls, or code tokens.

Benchmark gate:

- Report hint hit rate, repair rate, and worst-case fallback cost. Never claim
  exactness without the validation pass.

### Hot-Block Vocab Layout

Hypothesis: physically or logically reorder vocabulary into hot/high-variance
blocks for a fixed model, then run block-first Top-K while mapping indices back
to original token ids.

Why it might work:

- Hot blocks may produce candidates earlier, improving early pruning and cache
  locality.
- It is model-specific specialization, which is exactly where custom kernels
  can beat generic libraries.

Why it might fail:

- Reordering the LM-head weight matrix can break integration or quantization
  layout.
- Full exact Top-K still needs scanning cold blocks unless a certified bound
  exists.

Benchmark gate:

- Measure with and without physical weight permutation. Include inverse-index
  repair in the timing boundary.

### K-Regime JIT Ladder

Hypothesis: one generic Top-K kernel loses because `k` is a compile-time
behavior knob. Generate specialized kernels for common `k` values and dispatch
from a tiny runtime table.

Why it might work:

- Unrolled candidate lists and compare networks are much cleaner for fixed
  `k`.
- Register counts and shared-memory layout can be tuned per regime.
- Serving stacks often use a small set of policy values.

Why it might fail:

- Compile/cache overhead and code size can become real.
- Dynamic per-request `k` diversity may defeat specialization.

Benchmark gate:

- Record compile time, cache hit rate, binary size, selected variant, amdclang++ resource/ISA
  registers, and fallback behavior for unseen `k`.

### Boundary-Tie Microkernel

Hypothesis: most random rows do not have many equal scores at the kth boundary.
Use a fast path that finds candidate scores, then launch a tiny repair only for
rows where boundary ties can affect deterministic indices.

Why it might work:

- Deterministic lower-token-id ties are expensive to preserve everywhere.
- The repair rate may be near zero for FP32 logits.

Why it might fail:

- Quantized logits, masked logits, and all-equal adversarial rows can make
  boundary ties common.
- Extra launches may dominate unless graph-captured or fused.

Benchmark gate:

- Include all-equal and quantized/tied rows. Report repair fraction and fast
  path invalidation rate.

### Ragged Length Buckets

Hypothesis: ragged Top-K needs a scheduler, not just a kernel. Bucket segments
by length and run a warp, CTA, or multi-CTA candidate depending on the segment.

Why it might work:

- Padding to max length can be catastrophically unfair.
- Short segments want warp-per-row; long segments want multi-CTA work.

Why it might fail:

- Bucketing, prefix sums, and output scatter can erase the win.
- Serving integrations may already have page-table layouts that constrain the
  scheduler.

Benchmark gate:

- Compare against padded framework Top-K and a hipCUB/segmented pipeline. Record
  bucketing overhead separately.

### Score-Bin Bitset Select

Hypothesis: for medium `k`, a row can be summarized into score bins, then
candidate positions above the winning bin can be emitted with warp ballots and
local prefix counts.

Why it might work:

- It trades large sorted candidate arrays for compact counters and bitsets.
- It may be strong for fixed vocab and bounded score ranges.

Why it might fail:

- Bin collisions near the kth boundary require exact repair.
- The bitset itself can become expensive for huge vocab.

Benchmark gate:

- Record bin occupancy, candidate overflow, and repair size. Compare against
  radix-select and hipCUB radix sort.

### Two-Pass Top-K + CDF

Hypothesis: if the output is one sampled id from Top-K, a sorted top-k list is
unnecessary. First find an exact kth threshold; second pass computes the
selected exponential sum and samples from the selected set.

Why it might work:

- Avoids writing k values and sorting candidates when the consumer only needs
  one id.
- Keeps the CDF construction tied to selection instead of a separate materialized
  Top-K output.

Why it might fail:

- Two full passes over logits can lose for tiny `k`.
- Boundary ties and exact uniform mapping need careful policy.

Benchmark gate:

- Compare against fused `block-topk-sampling`, FlashInfer sampling, and a
  materialized Top-K plus softmax path.

### Confidence-Gated Argmax

Hypothesis: for relaxed or greedy modes, rows with a huge max margin can skip
Top-K machinery and use argmax or a smaller candidate set.

Why it might work:

- Many production requests use greedy or near-greedy settings.
- A cheap max/second-max pass can classify easy rows.

Why it might fail:

- This is not exact Top-K unless the contract changes.
- Quality and sampling distribution can drift.

Benchmark gate:

- Mark as `contract-changing`. Use only for greedy, approximate, or quality-
  evaluated serving experiments.

### Grouped Tensor-Parallel Merge

Hypothesis: when vocab is sharded across GPUs, each rank should produce local
top-k candidates and exchange only candidates, not full logits.

Why it might work:

- Communication volume becomes `O(world_size * k)`, not `O(vocab)`.
- Deterministic merge is small and easy to test.

Why it might fail:

- Local top-k must include enough candidates for global ties and policies.
- RCCL latency may dominate for tiny batches.

Benchmark gate:

- Record local candidate count, allgather bytes, merge time, and tie repair.

### Adversarial Autotuner

Hypothesis: the best way to find real ideas is to generate rows that make each
candidate fail, then autotune against both random and adversarial suites.

Why it might work:

- It prevents overfitting to friendly logits.
- It exposes whether a technique is exact, repairable, or merely lucky.

Why it might fail:

- Adversarial suites can overweight rare cases and pick slow conservative
  kernels.

Benchmark gate:

- Keep random, model-like, and adversarial suites separate. Select production
  variants with a holdout set.

### Low-Precision Prefilter

Hypothesis: approximate scores can cheaply shrink the candidate set, as long as
the final answer is repaired with exact scores.

Variants:

- Load FP16/BF16 scores first, then reload FP32 only near the boundary.
- Keep a quantized shadow logits buffer for repeated serving policies.
- Use INT8/FP8 dequant bounds to eliminate obvious losers before exact repair.

Why it might work:

- Candidate filtering is often memory-bandwidth bound.
- Approximate scores can be vectorized more aggressively.
- Exact repair protects the final contract.

Why it might fail:

- Close scores near the kth boundary can force large repair sets.
- Maintaining a shadow buffer can cost more than it saves.
- Quantization and tie policies can create hard adversarial cases.

Benchmark gate:

- Report prefilter candidate count, repair count, exact reload bytes, and rows
  where approximate pruning failed.

### Block Upper-Bound Pruning

Hypothesis: if each vocab block has a certified upper bound on its possible
score for the current query/hidden state, blocks below the current kth score
can be skipped exactly.

Possible bounds:

- Stored max logit from a previous exact pass, only valid with validation.
- Weight-block norm times hidden-state norm for LM-head GEMM.
- Quantization block scale and maximum code value.
- Per-block max after logits processors when processors are block-local.

Why it might work:

- It moves Top-K toward branch-and-bound maximum inner-product search.
- It can turn full-vocab scans into live-block scans for peaked workloads.

Why it might fail:

- Loose bounds skip nothing.
- Computing bounds can be as expensive as computing logits.
- Non-local processors and masks can invalidate precomputed metadata.

Benchmark gate:

- Record skipped-block fraction, false-live fraction, bound-compute time, and
  exact validation cost.

### Max-K Superset Cache

Hypothesis: in a mixed-policy serving batch, compute the largest requested `k`
once per row or policy group, then slice smaller `k` outputs.

Why it might work:

- Avoids launching separate kernels for `k=4`, `k=8`, `k=16`, etc.
- Produces reusable candidate lists for logprob and sampling features.

Why it might fail:

- `Kmax` can be much larger than most rows need.
- Sorted output for `Kmax` may overpay for rows needing only a sampled id.

Benchmark gate:

- Record policy distribution, wasted candidates, and latency versus per-k
  specialized dispatch.

### Policy Bucket Scheduler

Hypothesis: the fastest Top-K kernel is not one kernel. It is a scheduler that
groups rows by policy and launches a specialized kernel per group.

Bucket keys:

- `k`,
- sorted versus unsorted output,
- dtype,
- vocab length,
- temperature/top-p/min-p mode,
- output materialization: ids only, values, probabilities, logprobs.

Why it might work:

- Specialization recovers compile-time constants and simpler memory layouts.
- It avoids punishing the whole batch for one expensive row policy.

Why it might fail:

- Bucketing and scatter/gather overhead can dominate small batches.
- Too many buckets create too many launches unless graph-captured.

Benchmark gate:

- Report bucket count, rows per bucket, launch count, bucket overhead, and
  graph replay compatibility.

### Deterministic Packed Key

Hypothesis: deterministic sorted Top-K can be expressed as sorting/selecting a
single packed integer key, simplifying tie handling for library and custom
radix paths.

Sketch:

- Transform float score to monotonic sortable bits.
- Pack score bits with a token-id field that encodes the tie rule.
- For lower-token-id ties under descending scores, invert or bias the token id
  so ordinary integer comparison matches the contract.

Why it might work:

- It turns value/index pair compares into scalar key compares.
- hipCUB radix sort can become a stronger deterministic oracle.

Why it might fail:

- Full 64-bit keys increase bandwidth.
- NaN, signed zero, and infinities need explicit canonicalization.
- Token ids may not fit if too many score bits are retained.

Benchmark gate:

- Include all-equal, NaN, and signed-zero tests before timing. Record key width
  and packed-key bandwidth.

### In-Warp Reservoir Duel

Hypothesis: instead of each lane holding a sorted list, lanes hold a tiny
reservoir and periodically duel with neighboring lanes, dropping losers early.

Why it might work:

- Reduces per-thread insertion pressure.
- Keeps comparisons in warp registers with shuffle exchange.
- May be strong for `k=4..32`.

Why it might fail:

- Duel scheduling can miss candidates unless repair is exact.
- Extra shuffle traffic may lose to a boring insertion list.

Benchmark gate:

- Compare against insertion list, bitonic network, and hipCUB warp/block
  collectives with identical tie policy.

### Loser-Tree Block Merge

Hypothesis: medium-k second-pass merging should use a tournament/loser tree
over sorted block candidate lists instead of repeatedly inserting into one
global k-list.

Why it might work:

- Each merge step touches only the next winner from a block list.
- It separates first-pass candidate generation from final sorted output.

Why it might fail:

- Candidate lists from first pass may not be sorted cheaply.
- Shared-memory tree state may be bulky for many blocks.

Benchmark gate:

- Run as a second-pass-only harness with synthetic sorted candidate lists and
  with real first-pass outputs.

### Persistent Top-K Service

Hypothesis: for small decode batches, a GPU-resident service loop can own
request metadata and avoid relaunching tiny selection kernels.

Why it might work:

- Top-K can be launch-bound for batch 1..16.
- Persistent CTAs can keep policy metadata and workspace hot.

Why it might fail:

- Persistent kernels complicate stream semantics and integration.
- Queue management can cost more than kernel launches.
- Fairness and cancellation become correctness concerns.

Benchmark gate:

- Compare ordinary launch, HIP Graph replay, and persistent service on the
same request trace.

### Cluster DGFX Candidate Ring

Hypothesis: on CDNA3/CDNA4/RDNA4, CTAs in a cluster can merge candidates through
distributed shared memory before writing one compact global candidate list.

Why it might work:

- Few-row, huge-vocab Top-K needs multi-CTA rows.
- Cluster-local merge can avoid global temporary traffic.

Why it might fail:

- Cluster launch constraints reduce occupancy.
- DGFX/barrier overhead may exceed global-memory savings.
- It is architecture-specific and must not be applied to CDNA2/RDNA2/RDNA3 records.

Benchmark gate:

- Mark records with exact `gfx942`, `gfx950`, or `gfx1200` target and compare
  against non-cluster multi-CTA merge.

### cp.async Score Conveyor

Hypothesis: for large contiguous rows, use `cp.async` to stage score tiles while
the previous tile is merged into candidates.

Why it might work:

- It may hide global-load latency for expensive medium-k merges.

Why it might fail:

- Pure Top-K has little data reuse, so shared-memory staging may add traffic.
- Candidate merge may not be compute-heavy enough to hide copies.

Benchmark gate:

- Treat this as likely-negative until measured. Record shared-memory bandwidth,
  occupancy loss, and whether amdclang++ resource/ISA emits the intended async-copy path.

### Token-Class Split Lanes

Hypothesis: for fixed-model serving, split tokens into classes with different
score distributions, run cheap class-local Top-M, then repair with a global
merge.

Classes might include:

- control/special tokens,
- whitespace/punctuation,
- numeric/code tokens,
- frequent natural-language tokens,
- rare tail tokens.

Why it might work:

- Some classes are tiny and high impact.
- Class-specific thresholds may find candidates earlier.

Why it might fail:

- Class metadata and global repair may dominate.
- Distribution shifts across languages, code, or domains can break the
  heuristic.

Benchmark gate:

- Use real traces and report per-class candidate hit rate. Always run global
  exact repair before claiming exactness.

### Gumbel Race Sampler

Hypothesis: if the real operation is sampling, not materialized Top-K, Gumbel
race tricks can replace probability normalization or candidate sorting.

Why it might work:

- Sampling from categorical distributions can be expressed as an argmax over
  perturbed logits.
- It may avoid explicit softmax and CDF for some policies.

Why it might fail:

- Top-K-filtered sampling is not the same as full-vocabulary Gumbel sampling.
- RNG cost and statistical testing become central.

Benchmark gate:

- Mark as `contract-changing` unless it exactly matches the required sampling
  distribution. Use distribution tests, not token equality.

### Sparsemax / Simplex Decoder

Hypothesis: Top-K, Top-P, min-p, sparsemax-like truncation, and some constrained
decoding policies can be treated as thresholded simplex/projection operations,
sharing one threshold kernel.

Why it might work:

- The expensive part becomes finding a threshold and selected set.
- A single threshold solver may cover several sampling policies.

Why it might fail:

- It changes the decoding policy unless the caller explicitly asks for it.
- Numerical stability and probability semantics are easy to muddle.

Benchmark gate:

- Classify as `contract-changing`. Keep quality/perplexity or task metrics
  separate from exact Top-K timing.

### Draft-Token Candidate Reuse

Hypothesis: speculative decoding draft tokens can predict useful candidate
regions for the verifier model, especially across accepted spans.

Why it might work:

- Accepted draft spans often imply related high-probability verifier tokens.
- Candidate hints can seed pivots or hot blocks.

Why it might fail:

- Rejected spans may be exactly where hints are misleading.
- Exact verifier Top-K still needs full validation.

Benchmark gate:

- Record accepted/rejected span hit rates and compare against no-hint verifier
  Top-K with full repair.

### Model-Trace Offline Tuner

Hypothesis: synthetic logits are not enough. Mine real logits traces to choose
dispatch rules, pivots, bucket sizes, and wild-idea eligibility.

Why it might work:

- The score distribution determines whether threshold tricks work.
- Serving policies often repeat a small set of `k`, top-p, output, and dtype
  combinations.

Why it might fail:

- Trace-specific tuning can overfit.
- Privacy and storage constraints may limit trace availability.

Benchmark gate:

- Use train/holdout trace splits. Never select a production rule only from the
  same trace used for reporting.

### Split-Bucket One-Iteration Kernel

Hypothesis: a bucket-partition Top-K pass should not be a loose pipeline of
histogram, compact, select, and merge kernels. A single iteration can bucket a
tile, run local selection, and emit merge-ready candidates while the data is
still hot.

Why it might work:

- It removes global memory round trips between bucket and select phases.
- It gives a unified execution shape for radix buckets, pivot buckets, and
  score-range buckets.
- It can stop early when the bucket containing the kth element is known.

Why it might fail:

- One fused kernel may be too complex and register-heavy.
- Bucket imbalance can leave most CTAs idle.
- Boundary repair and global merge may still dominate.

Benchmark gate:

- Compare against separate bucket/count/compact/select kernels and hipCUB radix
  sort for `k=128,256,512,1024`.

### Cross-Layer Top-K Index Reuse

Hypothesis: for sparse attention or routing-style Top-K, exact indices computed
in anchor layers may be good hints for nearby layers. Reuse them where the
caller accepts approximation, or validate and repair them for exactness.

Why it might work:

- Sparse attention indexers can become a significant fraction of long-context
  inference cost.
- Nearby layers often attend to overlapping token sets.
- Reused indices are tiny compared with recomputing all scores.

Why it might fail:

- It is not exact unless validated against the current layer's scores.
- Quality drift can hide in long-context or reasoning workloads.
- Reuse decisions are model- and layer-specific.

Benchmark gate:

- Record anchor interval, reuse hit rate, quality metric, exact repair cost,
  and whether the result is approximate or exact-after-repair.

## Idea Selection Grid

| Situation | Try First | Avoid First |
| --- | --- | --- |
| `k=1` greedy decode | deterministic packed key, confidence-gated argmax if policy allows | medium-k heaps |
| `k=4..8` with sampling | unsorted-first late-sort, two-pass Top-K plus CDF, FlashTopK epilogue | radix/full sort |
| `k=16..64` sorted output | in-warp reservoir duel, unsorted-first late-sort, loser-tree merge | high-ratio complement path |
| `k=128..1024` | multi-pivot fence, exponent-first radix, score-bin bitset select | tiny-k insertion lists |
| high `k/n` | complement Top-K, full-sort crossover study | pretending selection must win |
| ragged segments | ragged length buckets, policy bucket scheduler | padded-only conclusions |
| decode LM-head bottleneck | FlashTopK epilogue, block upper-bound pruning, model-trace tuner | isolated Top-K-only claims |
| CDNA3/CDNA4/RDNA4 few-row huge-vocab | cluster DGFX candidate ring, persistent service | CDNA2/RDNA2-tuned CTA assumptions |
| sparse attention Top-K indices | cross-layer Top-K index reuse, model-trace tuner | claiming exact attention without repair |

## First Lab Batch

1. Add result metadata fields for `idea_id`, `contract_class`, `k_regime`, and
   `repair_rate`.
2. Implement a toy multi-pivot fence for `k=64,128,256`.
3. Implement unsorted-first plus late-sort for `k=16,32`.
4. Implement complement Top-K for high `k/n`.
5. Add low-precision prefilter with exact FP32 repair.
6. Add adversarial row generation before any timing runs.
7. Add one trace-driven experiment if real logits traces are available.
8. For sparse attention Top-K, add a separate trace with anchor/reuse layers and
   classify it as approximate, exact-after-repair, or quality-only.

The best outcome is not necessarily a win. The best outcome is a clean map of
which ideas deserve CUDA engineering time and which ones should become negative
examples.

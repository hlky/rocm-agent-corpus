# rocPRIM/hipCUB/rocThrust Competition Track

This track teaches agents to compete with hipCUB, rocPRIM/hipCUB/rocThrust, and rocThrust for
reductions, scans, histograms, top-k, and related data-movement primitives.
The intended answer is not "use hipCUB and stop." The intended answer is:

1. Understand the strongest rocPRIM/hipCUB/rocThrust baseline.
2. Use it as a correctness oracle and performance bar.
3. Identify which generality the target workload can discard.
4. Build a custom kernel, or a larger fused kernel using hipCUB warp/block
   primitives, that can beat, match, or narrowly specialize beyond the baseline.
5. Record wins and losses honestly.

The rocPRIM/hipCUB/rocThrust family is a very strong competitor. A custom kernel that loses is still
useful corpus data when it explains what hipCUB's generic implementation is doing
well and what narrower specialization should be tried next.

## Scope

Use this guide for:

- Device-wide reductions and transform reductions.
- Prefix scans, segmented scans, row scans, and stream compaction.
- Warp and block reductions/scans embedded inside custom kernels.
- Histograms, privatized counters, and atomic aggregation.
- Top-k, argmax, selection, and small-k ranking.
- rocThrust prototypes that should become custom HIP kernels.
- Fused ML primitives that contain reductions or scans: softmax, layer norm,
  RMSNorm, routing, sampling, filtering, and loss reductions.

Do not use this guide as a replacement for measuring. hipCUB often wins broad
single-primitive workloads. Custom kernels most often win by being narrower or
by doing more work per byte read.

## Library Roles

Treat hipCUB, rocPRIM/hipCUB/rocThrust, and rocThrust as:

- Baselines to beat.
- Correctness oracles.
- API references for edge cases, offsets, temporary storage, streams, and
  determinism.
- Parts bins for `hipcub::Warp*` and `hipcub::Block*` collectives inside custom
  kernels.
- Architecture-policy examples for tile sizing and dispatch.

Do not treat them as:

- Proof that reductions, scans, or histograms are solved for every workload.
- A reason to skip fusion.
- A replacement for shape-specific kernels.
- Evidence that a kernel cannot win for tiny, fixed, masked, sparse, or
  fused workloads.

## Where to Look

Primary upstream paths:

- `third_party/rocm-libraries/cub/cub/device/device_reduce.hpp`
- `third_party/rocm-libraries/cub/cub/device/device_scan.hpp`
- `third_party/rocm-libraries/cub/cub/device/device_histogram.hpp`
- `third_party/rocm-libraries/cub/cub/device/device_select.hpp`
- `third_party/rocm-libraries/cub/cub/device/device_topk.hpp`
- `third_party/rocm-libraries/cub/cub/block/block_reduce.hpp`
- `third_party/rocm-libraries/cub/cub/block/block_scan.hpp`
- `third_party/rocm-libraries/cub/cub/warp/warp_reduce.hpp`
- `third_party/rocm-libraries/cub/cub/warp/warp_scan.hpp`
- `third_party/rocm-libraries/cub/test/`

Corpus paths for this track:

- `docs/ROCPRIM_COMPETITION_TRACK.md`
- `data/index/rocprim_competition_track.json`
- `examples/rocprim/`

## Competition Loop

1. Define the primitive and the surrounding operation.
   - Is it just a sum, or `transform -> reduce -> epilogue`?
   - Is the scan global, per row, per segment, or per warp?
   - Does the histogram have few bins, many bins, skewed bins, or fixed bins?
   - Is top-k sorted or unsorted? Are indices required?

2. Establish the rocPRIM/hipCUB/rocThrust baseline.
   - `hipcub::DeviceReduce::*` for device reductions.
   - `hipcub::DeviceScan::*` for prefix scans.
   - `hipcub::DeviceHistogram::*` for histograms.
   - `hipcub::DeviceSelect::*`, `hipcub::DevicePartition::*`, and
     `hipcub::DeviceTopK::*` for filtering and ranking.
   - rocThrust for rapid prototype baselines and iterator-heavy compositions.

3. Capture baseline metadata.
   - Exact API and overload.
   - Input/output dtype and offset type.
   - Number of items, segment count, segment length distribution, row length.
   - Temporary storage size and allocation policy.
   - Stream, warmup count, timing boundary, and whether allocations are timed.
   - Determinism or floating-point associativity requirements.

4. Identify discarded generality.
   - Fixed row length or segment size.
   - Known alignment and contiguous layout.
   - One dtype or accumulator type.
   - Commutative operation only.
   - Known mask density or value distribution.
   - Small `k`, fixed bin count, or bounded value range.
   - Fused transform, predicate, activation, normalization, or writeback.

5. Pick the attack surface.
   - Fuse producer and consumer work around the reduction or scan.
   - Use one CTA per row or per segment for fixed shapes.
   - Use one warp per small row.
   - Use register-only or shared-memory staging for small reductions.
   - Privatize histogram bins per warp/block before global atomics.
   - Replace multi-kernel rocThrust chains with one kernel.
   - Use hipCUB block/warp primitives inside a larger custom kernel.
   - Remove temporary allocation overhead for tiny repeated workloads.

6. Measure and classify.
   - `win`: custom is faster for the declared contract.
   - `match`: custom is close enough and offers integration/fusion value.
   - `loss`: rocPRIM/hipCUB/rocThrust is faster; record why.
   - `negative example`: the attempted optimization did not help.
   - `template-only`: scaffold exists, no timing yet.

## DeviceReduce

### Baseline APIs

Use these as the first competitor:

- `hipcub::DeviceReduce::Sum`
- `hipcub::DeviceReduce::Min`
- `hipcub::DeviceReduce::Max`
- `hipcub::DeviceReduce::ArgMin`
- `hipcub::DeviceReduce::ArgMax`
- `hipcub::DeviceReduce::Reduce`
- `hipcub::DeviceReduce::TransformReduce` when available in the bundled rocPRIM/hipCUB/rocThrust

If `TransformReduce` is not available in the target Toolkit/rocPRIM/hipCUB/rocThrust, use a
transform iterator plus `DeviceReduce`, or a rocThrust transform-reduce prototype.

### Custom Kernel Opportunities

Custom reductions can win when:

- The transform is fused into the load path.
- The reduction is per row or per segment with fixed size.
- The result feeds a second operation that can be fused before writing memory.
- The reduction is tiny and launch/allocation overhead dominates.
- The input layout is aligned and vectorizable.
- The operation is not a generic associative reduction but a small custom state:
  for example `(sum, sumsq)`, `(max, index)`, `(max, exp_sum)`, or multiple
  statistics per row.

Common custom patterns:

- One block per row with `hipcub::BlockReduce`.
- One warp per row with `hipcub::WarpReduce`.
- Grid-stride reduction into one partial per block plus a second kernel.
- Single-pass atomic accumulation for simple integer counters or small outputs.
- Persistent-CTA loop for many small independent reductions.

### Losing Examples To Preserve

Preserve losses like these:

- A generic hand-written tree reduction slower than `DeviceReduce::Sum`.
- Vectorized loads that slow down due to alignment guards or register pressure.
- Atomic-only global accumulation that collapses under contention.
- Two-pass custom reduction that loses because hipCUB already has better tiling.
- A custom argmax that is numerically or tie-break inconsistent with the oracle.

Loss records should include the next narrower hypothesis: fixed row size,
fused transform, warp-only path, or reduced output state.

## DeviceScan

### Baseline APIs

Use these as the first competitor:

- `hipcub::DeviceScan::InclusiveSum`
- `hipcub::DeviceScan::ExclusiveSum`
- `hipcub::DeviceScan::InclusiveScan`
- `hipcub::DeviceScan::ExclusiveScan`

For segmented or keyed scans, compare against the strongest available path:
hipCUB primitives where applicable, rocThrust scan-by-key prototypes, or a custom
baseline made from per-segment `BlockScan` kernels. Record which path was used.

### Custom Kernel Opportunities

Custom scans can win when:

- Segments are fixed-size rows and each row fits in one warp or one CTA.
- The scan is fused with predicate generation, compaction, normalization, or
  scatter.
- The workload has many tiny independent scans where global `DeviceScan`
  overhead is too high.
- The output only needs the final prefix value, not all prefix elements.
- The scan operation is paired with a local transform or mask.
- The target only needs inclusive or exclusive semantics for one dtype.

Common custom patterns:

- One warp per segment with `hipcub::WarpScan`.
- One CTA per segment with `hipcub::BlockScan`.
- Two-level scan: per-block local scan plus scanned block totals.
- Fused scan + writeback for compaction or row prefix sums.
- Cooperative groups or warp shuffle scan for educational variants.

### Scan Correctness Traps

Agents frequently get these wrong:

- Inclusive versus exclusive output.
- Initial value handling.
- Carry propagation between CTAs.
- Segment boundary handling.
- Floating-point non-associativity and deterministic comparison tolerances.
- In-place scans where input and output ranges overlap unexpectedly.
- Offset type overflow for large item counts.

## Block and Warp Primitives

hipCUB block and warp collectives are not the enemy. They are reusable components
for custom kernels.

Use:

- `hipcub::BlockReduce<T, BLOCK_THREADS>`
- `hipcub::BlockScan<T, BLOCK_THREADS>`
- `hipcub::WarpReduce<T>`
- `hipcub::WarpScan<T>`
- `hipcub::BlockLoad` and `hipcub::BlockStore` when load/store shape matters

The custom part is the surrounding kernel:

- Fused load transform.
- Multiple statistics in registers.
- Shape-specific row/segment mapping.
- Fused epilogue.
- Avoiding global temporaries.
- Avoiding additional launches.

Record whether the custom kernel uses hipCUB internals. A kernel that uses
`BlockReduce` inside a fused layer norm is still a custom-kernel competitor.

## Temporary Storage

Device-level hipCUB algorithms usually follow a two-call pattern:

1. Call with `d_temp_storage = nullptr` to query `temp_storage_bytes`.
2. Allocate or reuse workspace.
3. Call again with the workspace on the target stream.

For fair comparison:

- Do not time one-time workspace query unless the deployment truly pays it.
- Record workspace bytes.
- Record whether workspace allocation is cached, pooled, or timed.
- For repeated tiny workloads, measure both preallocated and allocation-included
  boundaries; allocation overhead can dominate the decision.
- If a custom kernel needs no workspace, record that as part of the result.

## Fusion Patterns

The most important rocPRIM/hipCUB/rocThrust competition theme is fusion. A generic hipCUB primitive
must read and write the primitive boundary. A custom kernel can often eliminate
that boundary.

High-value fusions:

- `transform -> reduce`: loss, norm, weighted sum, masked sum.
- `reduce -> normalize`: softmax, RMSNorm, layer norm.
- `reduce(max) -> reduce(sum(exp)) -> write`: online softmax variants.
- `predicate -> scan -> scatter`: compaction and filtering.
- `histogram -> normalize`: probability bins or routing counts.
- `top-k -> sample`: logits filtering and decoding.
- `argmax -> gather`: classification or token selection.

When comparing to rocPRIM/hipCUB/rocThrust, include the full pipeline time. Do not compare a fused
custom kernel only to the inner hipCUB primitive if the hipCUB path still needs extra
kernels around it.

## Fixed Shape Wins

hipCUB is designed for broad shape coverage. Custom kernels can drop that
generality.

Good fixed-shape targets:

- Rows of 32, 64, 128, 256, 512, 1024, or 2048 elements.
- Small batch counts repeated many times.
- Known contiguous row-major layout.
- Known alignment for `float2`, `float4`, `half2`, or vectorized integer loads.
- Fixed number of histogram bins.
- Fixed `k` for top-k, especially `k <= 8`.
- Fixed segment size for routing, beam search, or token filtering.

Record the shape contract in the task. A kernel that only wins for
`rows=4096, cols=128, dtype=float` is still valuable when the contract is clear.

## Histograms

### Baseline APIs

Use:

- `hipcub::DeviceHistogram::HistogramEven`
- `hipcub::DeviceHistogram::HistogramRange`
- `hipcub::DeviceHistogram::MultiHistogramEven`
- `hipcub::DeviceHistogram::MultiHistogramRange`

Also compare against rocThrust or custom prototypes for unusual value transforms,
weighted bins, or per-row histograms.

### Custom Kernel Opportunities

Custom histograms can win when:

- Bin count is small and fixed.
- Input distribution is known and skew can be handled.
- Histograms are per row, per block, or per expert.
- Counters can be privatized in shared memory and reduced later.
- Values are already in a compact integer domain.
- The histogram is fused with quantization, routing, sampling, or normalization.

Common custom patterns:

- Per-warp privatized bins followed by block reduction.
- Per-block shared-memory bins followed by global atomics.
- Two-phase histogram for high contention.
- Warp-aggregated atomics for repeated keys.
- Vectorized loads for packed small integer samples.

Histogram losses are common when:

- Global atomics collide heavily.
- Shared-memory atomics bank-conflict or serialize.
- The input distribution changes from uniform to skewed.
- The custom kernel ignores counter overflow.
- The hipCUB baseline amortizes setup better than expected.

## Top-K and Selection

### Baseline APIs

Use:

- `hipcub::DeviceTopK::MaxKeys`
- `hipcub::DeviceTopK::MinKeys`
- `hipcub::DeviceTopK::MaxPairs`
- `hipcub::DeviceTopK::MinPairs`
- `hipcub::DeviceSelect::If`
- `hipcub::DeviceSelect::Flagged`
- `hipcub::DevicePartition::*`
- `hipcub::DeviceRadixSort::*` when sorted top-k is required
- rocThrust sort or partial workflows for prototype correctness

Check whether the hipCUB top-k result is sorted or unsorted for the chosen API and
record that contract. If the downstream consumer does not require sorted
results, an unsorted top-k baseline may be stronger.

### Custom Kernel Opportunities

Custom top-k can win when:

- `k` is tiny and fixed.
- The candidate set is one row, one beam, or one token slice.
- Top-k is fused with masking, bias, temperature, repetition penalty, or
  sampling.
- Only max/argmax is needed.
- Ties have a simple target-specific rule.
- The output can remain in registers/shared memory for the next step.

Common custom patterns:

- Per-thread local top-k followed by warp merge.
- Bitonic or insertion-network top-k for very small fixed `k`.
- One CTA per row with shared-memory candidate buffers.
- Top-k plus prefix sum for nucleus or threshold sampling.
- Argmax with pair state `(value, index)` using `BlockReduce`.

Top-k losses are especially useful. Record whether the loss came from too much
register pressure, bad merge logic, extra sorting work, or a stronger unsorted
hipCUB baseline than expected.

## rocThrust as Prototype, Not Final Answer

rocThrust is useful for:

- Writing a quick correctness baseline.
- Expressing iterator transforms.
- Prototyping filtering, transform-reduce, sort, or scan-by-key flows.

rocThrust can lose production comparisons because:

- Chained algorithms launch multiple kernels.
- Temporary allocations are hidden.
- Generic iterator machinery may obscure memory traffic.
- It may not fuse the exact producer/consumer boundary needed.

When a rocThrust prototype is slower, translate the operation into:

- One fused HIP kernel.
- A hipCUB device primitive plus a custom epilogue.
- A custom kernel using hipCUB block/warp primitives.

## Required Task Metadata

Every rocPRIM/hipCUB/rocThrust competition task should record:

- Primitive family: reduce, scan, histogram, select, top-k.
- Baseline API and exact overload.
- Custom strategy.
- Input dtype, output dtype, accumulator dtype.
- Number of items or row/segment shape.
- Segment length distribution if variable.
- Alignment and layout assumptions.
- Temporary storage bytes for hipCUB.
- Whether workspace allocation is timed.
- Number of kernel launches in baseline and custom path.
- Correctness oracle and tolerance.
- Evidence label.
- Hardware, ROCm toolkit, compiler flags, and GFX target when measured.

## Planned Corpus Tasks

The machine-readable task list lives in
`data/index/rocprim_competition_track.json`. Planned high-priority tasks include:

- Fused transform reduce versus `DeviceReduce::TransformReduce`.
- Fixed row/block reduction versus `DeviceReduce::Sum`.
- Warp-per-row reduction versus hipCUB for small rows.
- Fixed row inclusive scan using `BlockScan`.
- Predicate + scan + scatter compaction versus `DeviceSelect::If`.
- Small fixed-bin histogram with privatized bins.
- Tiny fixed-k top-k with fused mask and bias.
- Argmax pair reduction with deterministic tie handling.
- Negative examples for vectorization, atomics, and generic shuffle trees.

## Acceptance Criteria for New Examples

A useful rocPRIM/hipCUB/rocThrust competition example should answer:

- What exact hipCUB/rocThrust baseline is being challenged?
- What generality is the custom kernel discarding?
- Which data movement is eliminated?
- How many launches are avoided?
- What correctness edge cases are dangerous?
- What would make the attempt a loss?
- What profiler counters would be useful later, even if unavailable now?

If no measurements exist yet, label the example `template-only`.


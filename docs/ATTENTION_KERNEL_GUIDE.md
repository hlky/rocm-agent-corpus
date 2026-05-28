# Attention Kernel Guide

This track is for agents writing custom HIP attention kernels, not for stopping
at "use FlashAttention." FlashAttention, MIOpen, vLLM on ROCm, vLLM,
FlashInfer, Triton, and Composable Kernel are the performance bar, correctness references,
and extension surfaces. A custom kernel still matters when the workload is
narrower than those libraries: fixed shape, fixed layout, fused pre/post work,
decode-only latency, unusual masks, paged KV assumptions, or deployment plugin
boundaries.

Current task:

- `corpus/tasks/online-attention-forward/`
- `harnesses/attention_forward_benchmark.hip`
- `data/index/attention_kernel_track.json`

The default seed shape is a ROCm timing target; no result files are currently
checked in. New shapes remain `template-only` until result files are attached. If an attempted
optimization is slower or neutral, keep it as a `negative example` with
hardware and build metadata.

## Problem Contract

The seed task computes scaled dot-product attention forward:

```text
scores[q, k] = scale * dot(Q[q, :], K[k, :])
P[q, :] = softmax(scores[q, :] + mask[q, :])
O[q, :] = P[q, :] * V[:, :]
```

The starting contract is FP32 input, FP32 accumulation, contiguous
`[batch, heads, sequence, head_dim]` layout, `head_dim == value_dim`, optional
causal masking, no dropout, no ragged sequences, and no paged KV cache. That is
deliberately narrow so the custom-kernel mechanics are visible.

## Online Softmax

Materializing `QK^T`, applying softmax, and then multiplying by `V` writes and
reads an attention matrix of size `query_length * key_length`. FlashAttention
style kernels avoid that by streaming over key/value tiles and maintaining
online softmax statistics for each query row.

For one query row, keep:

- `m`: running maximum score
- `l`: running sum of exponentials under the current max
- `o`: running unnormalized output vector

For a new key tile with scores `s_j` and values `v_j`:

```text
m_new = max(m, max_j(s_j))
alpha = exp(m - m_new)
p_j = exp(s_j - m_new)
l_new = alpha * l + sum_j(p_j)
o_new = alpha * o + sum_j(p_j * v_j)
```

After all key tiles:

```text
O = o / l
```

This recurrence is the core lesson. It preserves stable softmax behavior while
letting the kernel keep only tile-local scores and values instead of a full
attention matrix.

## QK And V Tiling

The seed optimized kernel assigns one CTA to one `(batch, head, query)` row.
Each CTA loads a small K/V tile into shared memory, computes QK scores for that
tile, updates online softmax state, and accumulates the corresponding V
contribution.

Important choices to record:

- Key tile size and head dimension.
- Whether Q is read from global memory, cached in registers, or staged.
- Shared-memory bytes for K, V, scores, reductions, and output accumulators.
- CTA, warp, and thread ownership of scores and output dimensions.
- Boundary handling for partial tiles and masked keys.

Future optimized paths should replace scalar dot loops with Matrix Core QK and
PV tiles where the dtype contract allows it. Composable Kernel/CK Tile and FlashAttention are
excellent references for tile shapes and pipeline structure, but a copied
library path is not the final answer unless the record explains the
customization surface and next narrower custom hypothesis.

## Shared Memory And Register Pressure

Attention is easy to make correct and easy to make slow. The output accumulator
for one query row is `head_dim` floats, and keeping too much per thread can
spill registers. Moving accumulators to shared memory can reduce register
pressure but add shared-memory traffic. Larger K/V tiles improve data reuse but
raise shared-memory use and may lower occupancy.

Measure or record:

- Registers per thread and spills when profiler data is available.
- Dynamic shared-memory bytes per CTA.
- Occupancy limits from registers, shared memory, and CTA size.
- Whether a larger tile is a win or a negative example.
- Whether the result is `timing-only` or counter-backed.

Do not invent profiler-counter evidence. HIP-event timings alone are useful,
but claims about memory transactions, occupancy, or tensor-pipe utilization need
actual profiler artifacts.

## Causal Masking

Causal masking changes both correctness and performance. In prefill,
`query_length == key_length` and query `q` can see keys `0..q`. In decode,
`query_length` is often `1` while `key_length` is the full context length; the
single query should usually be aligned to the newest KV position, so it can see
the whole cache.

The seed harness uses:

```text
absolute_query = query_index + max(0, key_length - query_length)
visible if key_index <= absolute_query
```

That makes `query=1,key=N,causal=1` model latest-token decode. Any benchmark
record must state its mask semantics. A mismatch here can create fake wins or
fake correctness failures.

Optimization surfaces:

- Skip fully masked key tiles in causal prefill.
- Avoid branch-heavy per-score masking when tile bounds prove visibility.
- Add explicit padding masks or sequence lengths only after the causal contract
  is correct.
- Treat sparse/block masks as a separate task with metadata for mask layout.

## Prefill Versus Decode

Prefill and decode are different competitions.

Prefill:

- Many query rows are active.
- QK and PV are GEMM-like.
- Matrix Cores, tiled mainloops, and FlashAttention-style IO savings dominate.
- Full attention fusion matters more than isolated QK speed.

Decode:

- `query_length` is usually one or a few.
- KV cache reads dominate latency.
- Paged KV layout, batch composition, and scheduler behavior matter.
- vLLM, vLLM on ROCm, and FlashInfer are especially important references.

A decode-specialized custom kernel may win by fixing head dimension, KV layout,
page size, batch shape, or causal position. But an isolated kernel win is not an
end-to-end serving win unless scheduler, cache, and batching boundaries are
included in the metric.

## Baseline Ladder

Use this ladder before claiming an attention result:

1. CPU stable reference for correctness.
2. Naive custom HIP baseline for indexing and mask semantics.
3. Online-softmax custom HIP baseline.
4. GEMM plus softmax plus GEMM path where it matches the contract.
5. FlashAttention or MIOpen for prefill/fused attention.
6. Triton or framework-generated attention for compiler baselines.
7. vLLM on ROCm, vLLM, or FlashInfer for decode, paged KV, and serving context.
8. Narrow custom kernel or plugin that exploits a constraint the above must keep
   general.

When a library remains faster, record why. Common reasons are Matrix Core
mainloops, better pipelining, better split work, paged-cache planning, and lower
register pressure. That is useful corpus data, not a failure.

## Benchmark Commands

Example compile commands for an gfx1030 machine:

```powershell
hipcc -O3 -std=c++17 -arch=gfx1030 -DVARIANT_BASELINE `
  harnesses/attention_forward_benchmark.hip `
  corpus/tasks/online-attention-forward/source/baseline.hip `
  -o attention_baseline.exe

hipcc -O3 -std=c++17 -arch=gfx1030 -DVARIANT_OPTIMIZED `
  harnesses/attention_forward_benchmark.hip `
  corpus/tasks/online-attention-forward/source/optimized.hip `
  -o attention_optimized.exe
```

Example runs:

```powershell
.\attention_baseline.exe 1 4 128 128 64 1 5 30
.\attention_optimized.exe 1 4 128 128 64 1 5 30
.\attention_optimized.exe 1 32 1 2048 128 1 5 30
```

The harness prints JSON with shape, HIP-event timing fields, logical throughput,
CPU-reference error, device name, and gfx target. Store measured results
under `corpus/tasks/online-attention-forward/results/` and attach hardware and
build metadata before updating the task curation status.

## Metadata Checklist

Every measured record should include:

- Batch, heads, query length, key length, head dimension, value dimension.
- Layout and strides for Q, K, V, output, and KV cache if present.
- Dtype for Q, K, V, accumulator, and output.
- Scale, causal flag, mask type, dropout, and query-position convention.
- Prefill or decode classification.
- Tile sizes, shared-memory bytes, registers per thread where available.
- Baselines and versions: FlashAttention, MIOpen, Triton, vLLM on ROCm, vLLM,
  FlashInfer, framework, or custom-only.
- Warmup, measured iterations, timer type, synchronization points.
- Correctness reference, tolerance, seed, and max errors.
- Evidence label: `template-only`, `timing-only`, counter-backed, or blocked
  profiler attempt.
- Win/loss notes and the next custom hypothesis.

## Next Custom Hypotheses

Good follow-up kernels for this track:

- Warp-specialized decode for fixed `head_dim=128`.
- Matrix Core prefill with online softmax and PV accumulation.
- Double-buffered K/V tiles with `global-to-LDS staging` on CDNA2/CDNA3 paths.
- Causal tile skipping for long prefill.
- Paged-KV decode specialized to one page size.
- Fused rotary embedding, ALiBi/bias, or output projection-adjacent epilogue.
- Grouped-query or multi-query attention with fixed head sharing.

Keep the scope narrow, the evidence honest, and the library comparison close to
the exact contract being claimed.

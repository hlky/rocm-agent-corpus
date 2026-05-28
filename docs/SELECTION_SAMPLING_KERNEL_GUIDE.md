# Selection and Sampling Kernel Guide

This track covers custom HIP kernels for choosing tokens, beams, experts, or
other row-wise candidates from score vectors. Vendor and serving-stack kernels
are the baseline, oracle, and design reference; the custom path is justified
only when the workload is narrower or more fused than the generic path.

Current seed task:

- `corpus/tasks/block-topk-sampling`
- `harnesses/topk_sampling_benchmark.hip`
- Evidence status: no ROCm result files are currently attached. New shapes
  remain `template-only` until result files are attached.

## Workload Boundaries

Selection tasks differ mainly by how much ordering and probability mass they
must preserve.

| Workload | Output Contract | Common Baseline | Custom-Kernel Opening |
| --- | --- | --- | --- |
| Argmax or argmin | One value-index pair per row | hipCUB ArgMax, framework argmax | Fuse mask, bias, penalty, or stop-token logic into the reduction |
| Top-k selection | `k` best candidates, sorted or unsorted | hipCUB DeviceTopK, radix sort, framework topk | Fixed small `k`, fixed vocab, row-major logits, no full sort |
| Beam-search step | Best `(beam, token)` expansions plus parent ids | vLLM on ROCm/vLLM beam helpers | Fuse score update, length penalty, EOS handling, parent gather |
| Top-k sampling | Sample from softmax over top-k candidates | vLLM on ROCm, vLLM, FlashInfer | Fuse processors plus top-k plus temperature softmax |
| Top-p sampling | Smallest prefix whose probability mass reaches `p` | vLLM on ROCm AIR Top-P, FlashInfer | Approximate or histogram path for fixed vocab and serving policy |
| Logits processors | Mask, bias, penalties, bad words, grammar constraints | Serving stack processor pipeline | Fuse processors before selection to avoid extra passes over vocab |

Be explicit about sortedness. A top-k kernel that returns unsorted candidates is
not equivalent to a sorted framework `topk` call unless the downstream sampler
accepts unordered candidates.

## Argmax and Small Top-K

Argmax is a pair reduction over `(value, index)`. The comparison policy is part
of correctness:

- Tie policy: lower token id, higher token id, or first observed.
- NaN policy: ignored, propagated, or treated as `-inf`.
- Mask policy: masked tokens are skipped or assigned `-inf`.
- Index width: `int32` is enough for common vocab sizes, but record the choice.

For `k <= 8`, a common custom shape is:

1. Each thread scans a strided slice of the row and keeps a small sorted register
   list.
2. A warp or CTA reduction merges lists.
3. One lane or thread writes the final candidate list and any sampling outputs.

For larger `k`, consider a bitonic merge network, radix-select style partition,
or heap/register-list hybrid. A full sort is usually a warning sign unless the
downstream contract truly requires a globally sorted output.

## Sampling Boundaries

Sampling kernels are easy to benchmark unfairly because the logical operation
often spans several kernels in a serving stack.

Record whether timing includes:

- Logits processors such as repetition penalty, presence/frequency penalty,
  no-repeat n-gram, bad-word masks, guided decoding masks, or custom request
  processors.
- Temperature scaling.
- Top-k filtering.
- Top-p or min-p filtering.
- Softmax or probability renormalization.
- RNG generation or only use of precomputed uniform random values.
- Beam parent selection, gather, and finished-sequence handling.

Temperature changes the score scale before probabilities are formed:

```text
prob_i = exp((logit_i - max_logit) / temperature) / sum_j exp(...)
```

Top-k limits the candidate set by rank. Top-p limits it by cumulative
probability after sorting by probability. Combining top-k and top-p means the
implementation must define which filter runs first and how probabilities are
renormalized. The seed task implements top-k plus temperature sampling with
precomputed uniforms; it deliberately does not claim to implement top-p.

## Warp and Block Strategies

Useful implementation patterns:

- Warp argmax: one value-index pair per lane, reduce with shuffle operations or
  `hipcub::WarpReduce`.
- Block top-k: per-thread fixed candidate list, then shared-memory or hipCUB
  `BlockReduce` merge.
- Bitonic candidate merge: store `2k` values per merge group, sort or partially
  sort, keep `k`.
- Heap/register-list hybrid: maintain a small min-heap or sorted list per
  thread for `k` larger than a few entries.
- Histogram or AIR-style top-p: approximate the distribution into bins, find a
  cutoff, then renormalize selected probabilities.
- Persistent decode helper: keep per-request metadata hot when launch overhead
  or small batches dominate.

The seed `optimized.hip` uses one CTA per row, local sorted lists, and a
shared-memory reduction over candidate lists. It is a starting point, not an
end-state. Future variants should test warp-per-row for small vocabularies,
multi-CTA rows for very large vocabularies, and hipCUB block collectives inside
larger fused kernels.

## Library and Serving References

Use these local upstream paths as references before claiming a custom win:

rocPRIM/hipCUB/rocThrust/hipCUB:

- `third_party/rocm-libraries/cub/cub/device/device_topk.hpp`
- `third_party/rocm-libraries/cub/cub/device/device_select.hpp`
- `third_party/rocm-libraries/cub/cub/device/device_radix_sort.hpp`
- `third_party/rocm-libraries/cub/cub/block/block_reduce.hpp`
- `third_party/rocm-libraries/cub/cub/warp/warp_reduce.hpp`
- `third_party/rocm-libraries/cub/test/catch2_test_device_topk_api.hip`

vLLM on ROCm:

- `third_party/vllm/cpp/migraphx_llm/kernels/samplingTopKKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/samplingTopPKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/samplingAirTopPKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/beamSearchKernels.hip`
- `third_party/vllm/cpp/migraphx_llm/kernels/topkLastDim.hip`
- `third_party/vllm/cpp/migraphx_llm/thop/fusedTopkSoftmax.cpp`
- `third_party/vllm/cpp/migraphx_llm/thop/IndexerTopKOp.cpp`
- `third_party/vllm/docs/source/features/sampling.md`

vLLM:

- `third_party/vllm/csrc/topk.hip`
- `third_party/vllm/csrc/persistent_topk.hpp`
- `third_party/vllm/csrc/sampler.hip`
- `third_party/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `third_party/vllm/vllm/v1/sample/ops/topk_topp_triton.py`
- `third_party/vllm/vllm/v1/sample/logits_processor/builtin.py`
- `third_party/vllm/vllm/entrypoints/generate/beam_search/utils.py`
- `third_party/vllm/benchmarks/benchmark_topk_topp.py`

FlashInfer:

- `third_party/flashinfer/include/flashinfer/topk.hpp`
- `third_party/flashinfer/include/flashinfer/sampling.hpp`
- `third_party/flashinfer/include/flashinfer/air_top_p.hpp`
- `third_party/flashinfer/csrc/topk.hip`
- `third_party/flashinfer/csrc/sampling.hip`
- `third_party/flashinfer/benchmarks/bench_topk.py`
- `third_party/flashinfer/benchmarks/bench_sampling.py`

These projects are not escape hatches. They define the bar, the edge cases, and
often the next extension surface.

## Where Custom Kernels Can Win

Custom selection and sampling kernels can plausibly win when they drop
generality or fuse work:

- Fixed `k` such as 1, 2, 4, or 8.
- Fixed or bounded vocab size.
- Known contiguous row-major logits.
- Per-request sampling parameters already grouped by policy.
- Logits processors fused into the selection pass.
- Top-k plus softmax plus sampling in one launch.
- Beam score update plus top-k plus parent-id materialization in one launch.
- MoE expert top-k fused with bias, grouping, sigmoid/softmax, and routing
  metadata writes.
- Decode batches small enough that launch count and framework overhead matter.
- A serving integration can reuse workspace or keep metadata resident.

Custom kernels are likely to lose when they reimplement a generic full sort, use
large register lists that spill, ignore tie or NaN semantics, or compare against
a weak baseline while vLLM on ROCm, vLLM, FlashInfer, hipCUB, or a framework fused
path already covers the exact contract.

## Benchmarking the Seed Task

Build baseline:

```powershell
hipcc -O3 -std=c++17 -DVARIANT_BASELINE `
  harnesses/topk_sampling_benchmark.hip `
  corpus/tasks/block-topk-sampling/source/baseline.hip `
  -o topk_sampling_baseline.exe
```

Build optimized:

```powershell
hipcc -O3 -std=c++17 -DVARIANT_OPTIMIZED `
  harnesses/topk_sampling_benchmark.hip `
  corpus/tasks/block-topk-sampling/source/optimized.hip `
  -o topk_sampling_optimized.exe
```

Run examples:

```powershell
.\topk_sampling_baseline.exe 1024 32768 4 0.8 5 30
.\topk_sampling_optimized.exe 1024 32768 4 0.8 5 30
```

The harness reports JSON with median, p10, p90, min, max, logical throughput,
top-k mismatches, sample mismatches, and probability error against a CPU
reference. Do not record a win unless both variants use the same inputs,
stream-timing boundary, sampling contract, and build flags.

## Records to Keep

For every measured result, attach:

- GPU name, gfx target, driver, ROCm toolkit, and compiler version.
- Build command and architecture flags.
- Rows, vocab size, `k`, temperature, and whether RNG generation was included.
- Sorted or unsorted top-k contract.
- Tie, NaN, and mask policy.
- Baseline name and version or upstream commit.
- Workspace allocation policy.
- Evidence label: `timing-only`, `counter-backed-measured`, or
  `negative example`.

Losing attempts are valuable. Keep negative examples for register spilling,
sorted-output overhead, weak occupancy on tiny batches, top-p cutoff mismatch,
and serving-stack baselines that remain faster after all launch boundaries are
made fair.

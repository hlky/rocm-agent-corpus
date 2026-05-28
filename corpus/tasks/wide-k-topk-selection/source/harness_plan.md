# Harness Plan

- Operation: `wide-k-topk-selection`
- Main benchmark blueprint: `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md`
- Contract: select top-k values and indices per row or ragged segment, with
  sorted and unsorted modes recorded separately.
- Baselines:
  - CPU stable sorted reference for deterministic correctness.
  - PyTorch `torch.topk(sorted=True/False)` for framework parity.
  - hipCUB `DeviceTopK` for unsorted membership where its determinism contract is
    acceptable.
  - hipCUB `DeviceRadixSort` or rocThrust sort for sorted and high-k contracts.
  - FlashInfer, vLLM, or vLLM on ROCm only when the operation is inside a
    sampling-facing contract.
- Custom candidates:
  - `k=1`: value-index reduction.
  - `k=2..8`: compile-time register list or fixed compare-exchange network.
  - `k=16..32`: warp/CTA candidate network with sorted and unsorted variants.
  - `k=64..256`: heap/list hybrid or two-pass block candidates.
  - `k=512..1024+`: radix-select, threshold partition, or full-sort boundary.
  - `k/n` high: compare against full sort before adding complex selection code.
- Required correctness cases:
  - all equal scores,
  - ties across the kth boundary,
  - ascending and descending rows,
  - NaN, +Inf, -Inf policies,
  - non-power-of-two row lengths,
  - `k=1`, `k=2`, `k=8`, `k=32`, `k=128`, `k=1024`, and `k` near `n`,
  - ragged segments with length 0, 1, `k-1`, `k`, and very long lengths.
- Recommended benchmark grid:
  - rows=4096 n=32000 k=1 sorted=false
  - rows=1024 n=32768 k=4/8 sorted=true
  - rows=512 n=32768 k=16/32 sorted=true
  - rows=256 n=65536 k=64 sorted=true
  - rows=128 n=65536 k=128 sorted=true
  - rows=64 n=131072 k=256 sorted=true
  - rows=32 n=131072 k=512 sorted=true
  - rows=16 n=131072 k=1024 sorted=true
  - rows=16 n=32768 k=8192 sorted=true
  - ragged rows=4096 n_min=1 n_max=65536 k=1..128
- Required metrics:
  - median, p10, p90, min, max latency,
  - rows, n, k, k/n,
  - sortedness, tie, NaN, and mask policies,
  - temp storage bytes and allocation timing policy,
  - kernel launch count,
  - amdclang++ resource/ISA register/shared-memory/spill summary,
  - baseline version or commit,
  - correctness hash and mismatch counts.
- Result schema:
  - Use `schemas/topk_baseline_result.schema.json` for baseline result JSON
    artifacts once measurements are added.

Do not classify a result as a custom win until the matching k regime has a
same-hardware library baseline and the output contract matches.

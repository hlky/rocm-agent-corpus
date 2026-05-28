# Harness Plan

- Operation: radix-sort-key-value
- Baseline: Use hipCUB DeviceRadixSort or rocThrust sort_by_key with explicit temp-storage and allocation timing policy.
- Optimized candidate: Specialize radix passes for fixed bit width, local histograms, and fused downstream consume where full generic sort is unnecessary.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: n=1048576 uint32 keys/value32; n=16777216 uint32 keys/value32; rows=4096 cols=256 segmented-ish fixed ranges
- Required metrics: median latency; num_items; key bits; value bytes; temp storage bytes; allocation included; stability contract; correctness hash

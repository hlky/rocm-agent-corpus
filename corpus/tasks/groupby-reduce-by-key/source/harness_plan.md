# Harness Plan

- Operation: groupby-reduce-by-key
- Baseline: Use rocThrust/hipCUB reduce_by_key on sorted input or radix-sort keys before grouped reduction for unsorted input.
- Optimized candidate: Specialize for known key ranges or skew using block-local aggregation, privatized bins, or fixed-key shared reductions.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: n=1048576 groups=1024 uniform; n=1048576 groups=1024 Zipf; n=16777216 groups=65536 sorted; n=1048576 small fixed key range
- Required metrics: median latency; num_items; num_groups; sorted_input; group-size histogram; temp storage bytes; preprocessing included; correctness max error

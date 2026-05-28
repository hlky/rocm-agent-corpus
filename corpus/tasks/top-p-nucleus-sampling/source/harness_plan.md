# Harness Plan

- Operation: top-p-nucleus-sampling
- Baseline: Apply softmax, sort probabilities with hipCUB/rocThrust or framework ops, scan cumulative probability, then sample from the selected nucleus.
- Optimized candidate: Fuse logits transforms, partial selection, cumulative probability, and sampling for fixed vocab and threshold regimes.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: batch=1 vocab=32000 top_p=0.9; batch=32 vocab=32000 top_p=0.95; batch=128 vocab=128000 top_p=0.9
- Required metrics: median latency; batch; vocab; top_p; top_k cap; RNG seed; selected count distribution; baseline versions; correctness/statistical checks

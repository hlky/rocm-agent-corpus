# Harness Plan

- Operation: varlen-packed-attention
- Baseline: Pad variable-length sequences to a dense batch and run the existing online attention seed or framework SDPA reference.
- Optimized candidate: Use sequence-offset metadata to map CTAs only to live tiles and avoid work on padding while preserving online-softmax correctness.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: batch=128 avg_q=64 max_q=256 head_dim=64; batch=32 avg_q=512 max_q=2048 head_dim=128; decode-like batch=256 q=1 variable_k<=8192
- Required metrics: median latency; non-padding token count; padding ratio; max sequence length; head_dim; library baseline version; correctness max error

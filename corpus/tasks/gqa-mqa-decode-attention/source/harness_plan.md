# Harness Plan

- Operation: gqa-mqa-decode-attention
- Baseline: Expand or remap KV heads to query-head layout and reuse a generic decode attention boundary.
- Optimized candidate: Map query head groups directly to shared KV heads, reduce repeated cache reads, and specialize for fixed grouping factors.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: tokens=1 q_heads=32 kv_heads=8 head_dim=128; tokens=16 q_heads=40 kv_heads=8 head_dim=128; tokens=64 q_heads=64 kv_heads=1 head_dim=128
- Required metrics: median decode latency; query_heads; kv_heads; head_dim; tokens; cache layout; KV bytes read estimate; correctness max error

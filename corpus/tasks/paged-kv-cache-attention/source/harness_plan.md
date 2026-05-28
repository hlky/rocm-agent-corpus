# Harness Plan

- Operation: paged-kv-cache-decode-attention
- Baseline: Materialize or gather contiguous KV for each sequence, then run a simple decode attention kernel or framework reference.
- Optimized candidate: Read paged KV blocks directly inside the attention loop, fuse page-table traversal with online softmax, and specialize for decode batch/head layout.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: tokens=1 heads=32 kv_heads=8 head_dim=128 pages=4096; tokens=16 heads=32 kv_heads=8 head_dim=128 variable lengths; tokens=64 heads=40 kv_heads=8 head_dim=128 pages=8192
- Required metrics: median decode latency; tokens per step; page size; KV dtype; page-table bytes; cache hit/coalescing notes; correctness max error; hardware metadata

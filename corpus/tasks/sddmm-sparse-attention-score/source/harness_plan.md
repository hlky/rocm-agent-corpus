# Harness Plan

- Operation: sddmm-sparse-attention-score
- Baseline: One thread per nonzero computes a dot product using the sparse coordinates and dense matrices.
- Optimized candidate: Tile dense feature vectors for blocks of edges, reuse Q/K fragments, and specialize for block-sparse attention masks.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: rows=4096 cols=4096 nnz_per_row=16 head_dim=64; block_sparse seq=4096 block=64 density=0.1; power-law graph nnz=1048576 head_dim=128
- Required metrics: median latency; nnz; density; format; head_dim; block size; edge ordering; effective FLOP/s; correctness max error

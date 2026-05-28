# Harness Plan

- Operation: spgemm-merge-hash
- Baseline: Use a simple row-wise expansion with global or temporary accumulation and a CPU/rocSPARSE oracle for final CSR output.
- Optimized candidate: Use row-length-aware merge or hash accumulators with hipCUB primitives for scans/sorts where they reduce irregular overhead.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: uniform rows nnz=16; power-law rows nnz avg=32; blocky sparse M=N=K=4096 density=0.01
- Required metrics: symbolic time; numeric time; M; N; K; nnzA; nnzB; nnzC; row-length histogram; workspace bytes; correctness max error

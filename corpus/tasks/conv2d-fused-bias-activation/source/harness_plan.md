# Harness Plan

- Operation: conv2d-fused-bias-activation
- Baseline: Run convolution, bias add, and activation as separate stages with materialized intermediate output.
- Optimized candidate: Fuse bias and activation into the convolution writeback or library/plugin epilogue boundary.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: N=1 C=64 H=W=56 K=64 R=S=3 relu; N=32 C=128 H=W=28 K=256 R=S=3 silu; N=1 C=256 H=W=14 K=256 R=S=1 gelu
- Required metrics: end-to-end latency; intermediate bytes; activation kind; MIOpen/MIGraphX plan; workspace bytes; correctness max error

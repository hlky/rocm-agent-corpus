# Harness Plan

- Operation: block-scaled-fp4-gemm
- Baseline: Materialize dequantized FP16/BF16 operands or call the closest Transformer Engine on ROCm/Composable Kernel/hipBLASLt path when available.
- Optimized candidate: Use block-scale metadata in the matmul pipeline, avoid materialized dequantization, and specialize for supported Matrix Core paths.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: M=128 N=4096 K=4096; M=4096 N=4096 K=4096; M=4096 N=11008 K=4096
- Required metrics: median latency; M; N; K; FP4 format; scale block shape; accumulator dtype; arch flags; library algorithm; correctness tolerance

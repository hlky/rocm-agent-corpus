# Harness Plan

- Operation: int8-dot-mfma-gemm
- Baseline: Materialize dequantized operands or use a scalar int8 dot-product kernel with int32 accumulation for oracle comparison.
- Optimized candidate: Use packed int8 vector loads, DP4A-style accumulation for fallback, or map the task to library/Composable Kernel IMFMA tiles for Matrix Core comparison.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: M=1 N=4096 K=4096; M=128 N=4096 K=4096; M=4096 N=4096 K=4096
- Required metrics: median latency; M; N; K; layout; accumulator dtype; scale policy; effective TOPS; library algorithm; correctness max error

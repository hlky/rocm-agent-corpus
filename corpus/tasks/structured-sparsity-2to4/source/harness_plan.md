# Harness Plan

- Operation: structured-sparsity-2to4
- Baseline: Run dense GEMM or materialize sparse weights into a library-supported format and compare with a correctness oracle.
- Optimized candidate: Pack 2:4 metadata and use sparse Matrix Core-compatible tiles or a custom metadata-aware fallback for unsupported shapes.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: M=128 N=4096 K=4096 fp16; M=4096 N=4096 K=4096 fp16; M=1 N=4096 K=4096 int8/fp16 mixed
- Required metrics: median latency; sparsity validity; metadata bytes; M; N; K; dtype; dense baseline; sparse library path; correctness max error

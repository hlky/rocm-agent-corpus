# Harness Plan

- Operation: splitk-reduction-gemm
- Baselines: hipBLASLt heuristic-selected GEMM, Composable Kernel Split-K kernels, Triton matmul with comparable split/reduction settings.
- Correctness: compare `C` against hipBLAS/rocBLAS/hipBLASLt or CPU FP32 accumulation; include `split_k=1` as a control.
- Optimized candidate: K-partitioned Tensor Core GEMM with explicit partial-output reduction using atomics, workspace reduction, or epilogue reduction.
- Recommended shapes: `M=128 N=4096 K=16384`; `M=256 N=256 K=65536`; `M=4096 N=4096 K=4096`; `M=64 N=8192 K=8192`.
- Required metrics: latency, effective TFLOP/s, split factor, reduction policy, workspace bytes, math mode, library algorithm, hardware/build metadata, correctness max error, evidence label.
- Evidence discipline: separate GEMM time, reduction time, and end-to-end time.

# Harness Plan

- Operation: composable-kernel-custom-epilogue
- Baseline: Use hipBLASLt or Composable Kernel built-in linear-combination epilogue, then run unsupported postprocessing in a separate kernel.
- Optimized candidate: Implement the unsupported postprocessing as a Composable Kernel visitor/epilogue so GEMM writeback performs the extra work once.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: M=N=K=4096 fp16; M=128 N=4096 K=4096 fp16; M=4096 N=11008 K=4096 fp16
- Required metrics: GEMM latency; epilogue latency delta; extra launches avoided; tile shape; register/shared memory pressure; workspace bytes; correctness max error

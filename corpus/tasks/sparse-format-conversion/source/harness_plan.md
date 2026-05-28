# Harness Plan

- Operation: sparse-format-conversion
- Baseline: Use library conversion routines or a staged sort/scan/scatter pipeline with explicit temporary storage.
- Optimized candidate: Fuse histogram/count, prefix, and scatter steps for known formats, block sizes, and sortedness constraints.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: COO->CSR nnz=1048576; CSR->COO nnz=16777216; CSR->BSR block=16 density=0.05
- Required metrics: median latency; source format; destination format; nnz; shape; index dtype; sort included; temp storage bytes; correctness hash

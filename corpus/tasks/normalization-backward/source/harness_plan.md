# Harness Plan

- Operation: normalization-backward
- Baselines: PyTorch autograd/eager, Transformer Engine normalization backward, Triton normalization kernels where available.
- Correctness: compare `dx`, `dgamma`, and `dbeta` against a PyTorch reference for LayerNorm and RMSNorm; include saved-statistic and recompute variants.
- Optimized candidate: fixed-hidden-size CUDA backward with row reductions plus staged `dgamma/dbeta` reductions.
- Recommended shapes: `rows=8192 cols=768 layernorm`; `rows=4096 cols=4096 rmsnorm`; `rows=32768 cols=1024 layernorm`; `rows=2048 cols=8192 rmsnorm`.
- Required metrics: median latency, dtype, mode, reduction policy, baseline versions, hardware/build metadata, correctness max error, evidence label.
- Evidence discipline: mark early results as `timing-only` unless rocprofiler/rocprof counter artifacts are attached.

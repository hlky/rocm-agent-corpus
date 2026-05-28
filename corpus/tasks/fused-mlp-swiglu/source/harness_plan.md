# Harness Plan

- Operation: fused-mlp-swiglu
- Baseline: Run two projection GEMMs, materialize gate/up activations, apply SwiGLU in a separate kernel, then run the down projection.
- Optimized candidate: Fuse bias, activation, gate multiply, scaling, and optional quantization into the closest legal GEMM epilogue or a shape-specialized custom boundary.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: tokens=1 hidden=4096 inter=11008; tokens=128 hidden=4096 inter=11008; tokens=4096 hidden=8192 inter=28672
- Required metrics: end-to-end MLP latency; intermediate bytes; launch count; GEMM library algorithm; epilogue type; correctness tolerance; hardware metadata

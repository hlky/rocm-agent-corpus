# Harness Plan

- Operation: cudnn-frontend-fused-subgraph
- Baseline: MIOpen operation graph with plan build time, execution plan metadata, workspace bytes, dtype, layout, strides, and plan-cache state recorded.
- Library baselines: MIOpen, MIGraphX engine/plugin path, Composable Kernel custom epilogue where GEMM-like, and framework/compiler output when relevant.
- Optimized candidate: fixed-shape custom HIP or Composable Kernel kernel that drops dynamic layout/general dtype support and fuses the materialized intermediate away.
- Correctness: compare MIOpen, optimized CUDA, and CPU/framework oracle on deterministic tensors before timing.
- Evidence discipline: template-only until measured; say timing-only for HIP-event timings without rocprofiler/rocprof counters; keep negative examples if MIOpen or MIGraphX wins.
- Recommended shapes: N=32,H=56,W=56,C=64 conv+bias+relu; B=16,S=2048,H=4096 layernorm+activation; B=1,S=128,H=8192 inference epilogue bucket.
- Required metrics: median latency; plan build time; workspace bytes; cudnn engine id; dtype and layout; library baseline versions; correctness max error; hardware metadata.

# Harness Plan

- Operation: fp8-cast-scale-amax
- Baseline: Run separate scale/cast and amax-reduction passes or call framework/Transformer Engine on ROCm reference utilities.
- Optimized candidate: Fuse scale, cast, saturation, and block or tensor amax collection in one pass with vectorized loads and deterministic metadata writes.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: n=1048576 fp16->fp8; rows=4096 cols=4096; rows=16384 cols=11008
- Required metrics: median latency; num elements; FP8 format; scale dtype; amax reduction policy; saturation count; bandwidth; correctness tolerance

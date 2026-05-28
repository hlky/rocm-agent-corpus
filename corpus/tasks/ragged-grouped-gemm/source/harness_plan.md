# Harness Plan

- Operation: ragged-grouped-gemm
- Baseline: Loop over groups and call a library GEMM per shape or pad groups into a single dense batch.
- Optimized candidate: Use a persistent or grouped scheduler that maps CTAs across shape buckets and avoids padding when raggedness is high.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: experts=8 tokens_per_expert varied 0..256 hidden=4096 inter=11008; groups=128 small M varied N=K=256; groups=16 mixed M,N,K
- Required metrics: median latency; group count; shape histogram; padding bytes; library calls; scheduler policy; effective FLOP/s; correctness max error

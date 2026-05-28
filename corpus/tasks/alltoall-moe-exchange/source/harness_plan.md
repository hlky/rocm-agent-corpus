# Harness Plan

- Operation: alltoall-moe-exchange
- Baseline: Build send counts, permute tokens into rank-major buffers, run RCCL send/recv or all-to-all equivalent, then unpermute after expert compute.
- Optimized candidate: Fuse routing metadata construction with packing, overlap exchange with local expert work, or use rocSHMEM for GPU-initiated transfers.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: ranks=2 tokens=4096 hidden=4096 experts=8; ranks=4 tokens=8192 hidden=4096 experts=32 skewed; ranks=8 tokens=16384 hidden=8192 experts=64
- Required metrics: end-to-end exchange latency; rank count; tokens; hidden; experts; skew; bytes sent matrix; timeline evidence; correctness hash

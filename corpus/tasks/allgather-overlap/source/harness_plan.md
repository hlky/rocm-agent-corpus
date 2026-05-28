# Harness Plan

- Operation: allgather-overlap
- Baseline: Run RCCL AllGather to materialize the full tensor, then launch the consumer compute kernel.
- Optimized candidate: Pipeline shard arrival with consumer compute using stream events and partitioned output buffers.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: ranks=2 hidden=4096 tokens=4096; ranks=4 hidden=8192 tokens=2048; ranks=8 hidden=12288 tokens=1024
- Required metrics: end-to-end latency; rank count; bytes per shard; pipeline depth; topology; stream count; timeline evidence; correctness max error

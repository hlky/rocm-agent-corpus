# Harness Plan

- Operation: reduce-scatter-overlap
- Baseline: Run local compute over the full buffer, then RCCL ReduceScatter on a communication stream after compute completes.
- Optimized candidate: Chunk work and issue ReduceScatter as soon as each chunk is ready while later chunks continue computing.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: ranks=2 bytes=256MiB chunks=8; ranks=4 bytes=1GiB chunks=16; ranks=8 bytes=4GiB chunks=32
- Required metrics: end-to-end latency; rank count; bytes per rank; chunk size; topology; transport; overlap percentage; timeline evidence; correctness max error

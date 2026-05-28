# Harness Plan

- Operation: framework-compiler-generated-kernel
- Baseline: Run eager PyTorch and torch.compile/Inductor, collect generated kernel/source metadata, and time with synchronized HIP events or benchmark utilities.
- Optimized candidate: Replace the generated kernel boundary with a handwritten HIP extension specialized for the same static shape and fusion contract.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: rowwise norm + activation rows=4096 cols=4096; masked elementwise fusion n=16777216; small dynamic-shape row op batch=1..64
- Required metrics: eager latency; compiled latency; custom extension latency; compile time; dispatch included; generated kernel hash; correctness tolerance

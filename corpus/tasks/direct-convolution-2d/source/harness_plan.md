# Harness Plan

- Operation: direct-convolution-2d
- Baseline: One output element per thread loops over channels and filter taps with direct global-memory loads.
- Optimized candidate: Tile spatial neighborhoods in shared memory and specialize for fixed filter size, layout, and channel pack.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: N=1 C=64 H=W=56 K=64 R=S=3; N=32 C=64 H=W=56 K=128 R=S=3; N=1 C=3 H=W=224 K=64 R=S=7
- Required metrics: median latency; layout; filter shape; stride/padding; effective FLOP/s; DRAM bytes estimate; MIOpen algorithm; correctness max error

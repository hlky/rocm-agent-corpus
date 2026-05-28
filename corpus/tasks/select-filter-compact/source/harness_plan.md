# Harness Plan

- Operation: select-filter-compact
- Baseline: Generate flags or predicates, run hipCUB DeviceSelect or rocThrust copy_if, and record temporary storage and allocation policy.
- Optimized candidate: Fuse predicate generation, block-local prefix counts, one global reservation per block, and scatter for known mask-density regimes.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: n=1048576 density=0.01; n=1048576 density=0.5; n=16777216 density=0.9; rows=4096 items=256 per-row compact
- Required metrics: median latency; num_items; mask density; stable ordering; num_selected; temp storage bytes; launch count; correctness hash

# Harness Plan

- Operation: unique-run-length-encode
- Baseline: Use hipCUB DeviceRunLengthEncode or rocThrust unique/reduce_by_key and record temporary storage.
- Optimized candidate: Fuse boundary detection, prefix counts, and unique/count writes for fixed dtype and known run-length distributions.
- Before timing, add a CPU or strongest-library oracle and run edge-case correctness checks.
- Before win/loss claims, record same-hardware library baselines and evidence labels.
- Recommended shapes: n=1048576 avg_run=1; n=1048576 avg_run=8; n=16777216 skewed runs
- Required metrics: median latency; num_items; num_runs; run-length histogram; temp storage bytes; output sizing policy; correctness hash

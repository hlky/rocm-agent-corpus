# Harness Plan

- Operation: graph-captured-inference-step
- Baseline: ordinary stream launches for the same fixed inference bucket, with no allocations in the timed region.
- Library/runtime baselines: HIP Graph replay, MIGraphX engine execution with graph capture when applicable, framework graph/capture path if the workload starts in PyTorch or cuda-python.
- Optimized candidate: preallocate fixed input/output/workspace buffers, capture the inference step once, instantiate the graph, and replay for steady-state latency.
- Correctness: run ordinary-launch and graph-replay paths on identical buffers and compare outputs before timing.
- Evidence discipline: template-only until measured; report graph capture, instantiation, update, and replay timings separately; say timing-only unless profiler artifacts exist.
- Recommended shapes: batch=1 hidden=4096 tokens=1 decode bucket; batch=8 hidden=4096 tokens=16 microbatch; batch=32 hidden=8192 tokens=1 serving bucket.
- Required metrics: ordinary launch latency; graph capture time; graph instantiate time; graph replay latency; parameter update cost; fixed buffer sizes; timer_type; hardware metadata.

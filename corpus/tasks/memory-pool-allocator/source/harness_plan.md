# Harness Plan

- Operation: memory-pool-allocator
- Baseline: Per-iteration `hipMalloc`/`hipFree` plus a sentinel touch kernel.
- Optimized candidate: `hipMallocAsync`/`hipFreeAsync` on the default stream-ordered memory pool plus the same sentinel touch kernel.
- Oracle: Device checksum from sentinel writes across allocated ranges, stream synchronization checks, and optional rocm-sanitizer runs.
- Recommended shapes: allocation_bytes=4096 streams=1 allocations_per_stream=32; allocation_bytes=65536 streams=1 allocations_per_stream=16; allocation_bytes=1048576 streams=4 allocations_per_stream=4
- Required metrics: median allocation latency; median end-to-end iteration latency; pool warmup policy; peak live bytes; reserved bytes; reuse hit rate; stream count; synchronization boundary
- Evidence notes: The current harness reports host steady-clock time around each allocation/touch/free iteration plus stream synchronization. Treat it as timing-only API/workload latency, not pure kernel time.

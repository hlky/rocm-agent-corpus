# Harness Plan

- Operation: hip-ipc-multiprocess-shared-buffer
- Baseline: ordinary per-process allocation and explicit copy or same-process producer/consumer path for correctness.
- Library/runtime baselines: HIP IPC/runtime samples, ROCm runtime peer access APIs, HIP events, and RCCL/rocSHMEM only as broader multi-process communication references.
- Optimized candidate: consumer opens producer memory and event handles, optionally enables peer access, and works directly on the shared device buffer.
- Correctness: producer fills deterministic data, exports memory/event handles, consumer waits on the event, runs a checksum or transform kernel, and validates expected output.
- Evidence discipline: template-only until measured; separate handle setup from steady-state latency; say timing-only for HIP-event timings without counters.
- Recommended shapes: bytes=4MiB same-GPU two-process; bytes=64MiB same-node P2P two-GPU; bytes=256MiB producer-consumer ring.
- Required metrics: handle export/import time; steady-state latency; effective bandwidth; device ordinals; peer access enabled; synchronization method; timer_type; hardware metadata.

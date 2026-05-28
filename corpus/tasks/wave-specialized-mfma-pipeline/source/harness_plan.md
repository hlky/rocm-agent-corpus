# Harness Plan

- Baselines: scalar HIP reference for correctness, hipBLASLt, and Composable Kernel/CK Tile for performance.
- Correctness: compare against CPU FP32 accumulation over rounded FP16 inputs and record max absolute and relative error.
- Build matrix: `gfx942` and any `gfx950` target supported by the local ROCm toolkit, kept as separate records.
- Evidence: record `hipcc` command line, ROCm version, library versions, VGPR/SGPR/LDS metadata, pipeline role counts, stage count, and disassembly evidence for MFMA/LDS claims.
- Labels: use `template-only` until measured, `timing-only` for HIP-event timing without profiler counters, and `counter-backed-measured` only with attached rocprofiler/rocprof counter artifacts.

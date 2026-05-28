# Harness Plan

- Baselines: scalar HIP reference, hipBLASLt, rocBLAS where applicable, and Composable Kernel/CK Tile.
- Correctness: CPU FP32 accumulation from the same FP16 inputs; record tolerance, bad-count, and shape.
- Build matrix: `gfx90a`, `gfx942`, and `gfx950` only when the local toolkit supports each target.
- Required metadata: tile shape, stage count, LDS bytes, VGPRs/SGPRs, occupancy estimate, library algorithm or CK collective, and exact hipcc flags.
- Evidence: do not claim an LDS-staged or MFMA-specific win without disassembly or profiler evidence that supports the mechanism.

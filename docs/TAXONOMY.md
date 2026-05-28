# Optimization Taxonomy

Use these tags consistently so retrieval can find the right examples.

## Memory Access

- `coalescing`: consecutive lanes access consecutive memory locations.
- `strided-access`: memory pattern causes extra sectors or transactions.
- `shared-memory`: explicit use of block-local shared memory.
- `bank-conflict`: multiple threads contend for the same shared-memory bank.
- `tiling`: data is staged or partitioned to increase reuse.
- `vectorized-load`: use of wider loads or stores such as `float4`.
- `alignment`: pointer or shape constraints enabling efficient memory access.
- `register-pressure`: many live values limit occupancy or cause spills.
- `spill-management`: register pressure, local memory traffic, and occupancy
  tradeoffs.

## Execution

- `occupancy`: active waves or blocks per GFX target.
- `ilp`: instruction-level parallelism.
- `wave-divergence`: lanes in a wave follow different control paths.
- `wave-specialization`: different waves perform distinct pipeline roles.
- `launch-config`: block size, grid size, dynamic shared memory, launch bounds.
- `persistent-kernel`: kernel keeps work resident to reduce launch overhead or
  improve scheduling.

## Math and Libraries

- `reduction`: sum, max, min, or other associative reductions.
- `scan`: prefix operations.
- `gemm`: matrix multiplication.
- `matrix-core`: MFMA, rocWMMA, or library Matrix Core paths.
- `cub`: hipCUB primitives.
- `composable-kernel`: Composable Kernel or CK Tile implementation patterns.
- `rocthrust`: rocThrust algorithms.

## Correctness Risks

- `race-condition`: missing synchronization or conflicting writes.
- `numerical-drift`: floating-point differences beyond tolerance.
- `shape-overfit`: speedup works for one shape but fails elsewhere.
- `benchmark-artifact`: timing change caused by measurement error.
- `undefined-behavior`: out-of-bounds, invalid aliasing, or invalid casts.

## Useful rocprofiler/rocprof Signals

- Memory coalescing: sectors per request, memory throughput, load/store
  efficiency.
- Occupancy: achieved occupancy, theoretical occupancy, active waves per GFX.
- Register pressure: registers per thread, local memory traffic, spill loads and
  stores.
- Shared memory: bank conflicts, shared load/store throughput.
- Divergence: branch efficiency and predicated-off thread counts.
- Compute saturation: GFX throughput, issue slot utilization, tensor pipe usage.


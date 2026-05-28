# Optimization Playbook

This playbook is the corpus-level checklist for HIP/ROCm optimization agents.

## Memory Coalescing

Symptoms:

- Low bandwidth for simple copy-like kernels.
- Adjacent lanes access memory with large stride.

Changes:

- Map `threadIdx.x` to the contiguous dimension.
- Use structure-of-arrays instead of array-of-structures when lanes need one
  field at a time.
- Prefer aligned vectorized loads only when alignment and tails are handled.

Risks:

- Vectorized casts require alignment.
- Reindexing can accidentally transpose or alias data.

## Shared-Memory Tiling

Symptoms:

- One side of a transpose or stencil is coalesced and the other is strided.
- Global memory values are reused by neighboring threads.

Changes:

- Stage tiles in shared memory.
- Pad shared tiles, commonly `[tile][tile + 1]`, to avoid bank conflicts.
- Keep boundary checks explicit.

Risks:

- Shared memory can reduce occupancy.
- A tile that is only used once may not pay for synchronization.

## Reductions

Symptoms:

- Atomic contention on one or a few memory locations.
- Serial per-row or per-block accumulation.

Changes:

- Reduce within a thread block first, then use one atomic per block.
- Use warp-level primitives or hipCUB when production quality is needed.
- Accumulate in a wider type when numerical error matters.

Risks:

- Floating-point reorder changes results.
- Non-associative operations need stronger correctness criteria.

## Row-Wise Softmax

Symptoms:

- One thread processes a whole row.
- Multiple kernel launches read and write the same matrix repeatedly.

Changes:

- Use one block per row for moderate column counts.
- Reduce max first for numerical stability.
- Reduce sum of exponentials, then normalize.

Risks:

- Very wide rows need multi-block reductions.
- Fast math may change tolerances.
- Storing intermediate exponentials trades memory bandwidth for compute.

## Occupancy and Register Pressure

Symptoms:

- Low active warps with high register or shared-memory usage.
- Local memory traffic from spills.

Changes:

- Tune block size.
- Reduce live ranges.
- Consider launch bounds only after measuring.
- On ROCm, inspect VGPR pressure, occupancy, and spill code before changing
  launch bounds or forcing narrower live ranges.

Risks:

- Higher occupancy can reduce locality or ILP.
- Lower register count can increase instruction count.

## Libraries as Rivals and References

Before writing custom kernels, inspect the strongest relevant library path:

- hipCUB/rocPRIM/hipCUB/rocThrust for reductions, scans, sorts, selections.
- hipBLAS/rocBLAS/hipBLASLt/Composable Kernel for GEMM-like work.
- MIOpen for neural network primitives.
- rocSPARSE/rocSOLVER/rocFFT for domain kernels.

Do this to establish the bar, not to end the investigation. Custom kernels are
most likely to win when they exploit assumptions a general library cannot:

- Fixed or narrow shape families.
- Fusion across library boundaries.
- Known layout, alignment, or sparsity.
- Custom epilogues or side outputs.
- Lower dispatch, allocation, or framework overhead.
- Architecture-specific instructions or scheduling.
- Relaxed precision or special numerical constraints.

When the custom path loses, keep the record. Boundary cases teach agents where
the vendor implementation is still the right benchmark to study.

# GEMM Examples

This folder contains starter files for the GEMM competition track. They are
scaffolds, not benchmark results.

Use them with:

- `docs/GEMM_COMPETITION_TRACK.md`
- `data/index/gemm_competition_track.json`
- `docs/HIPBLAS_ROCBLAS_GUIDE.md`
- `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md`

## Files

- `fixed_shape_sgemm_skeleton.hip`: small fixed-shape custom GEMM skeleton with a
  fused bias/ReLU-style epilogue hook.
- `hipblaslt_baseline_skeleton.hip`: hipBLASLt baseline skeleton showing the
  descriptors and metadata an agent should fill in before timing.

## Agent Use

1. Pick a task id from `data/index/gemm_competition_track.json`.
2. Write the exact GEMM contract: shape, dtype, layout, alpha/beta, epilogue,
   tolerance, and timing boundary.
3. Add a host or simple device reference.
4. Implement the custom path.
5. Add hipBLAS/rocBLAS/hipBLASLt/Composable Kernel baselines that match the same contract.
6. Record wins and losses. Mark HIP-event-only measurements as `timing-only`.

Do not treat these skeletons as optimized kernels. Their purpose is to keep
future agent work honest and navigable.


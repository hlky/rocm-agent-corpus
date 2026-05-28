# Matrix Core Examples

These examples support `docs/MATRIX_CORE_ROCWMMA_GUIDE.md` and
`data/index/matrix_core_track.json`. They are reference sketches for future
agents, not measured results.

## Files

- `rocwmma_fragment_notes.hip`: a compact rocWMMA fragment skeleton that shows the
  safe way to apply uniform accumulator operations without assuming fragment
  element coordinates.
- `inline_mma_sync_skeleton.hip`: an annotated placeholder for the inline
  `mfma` path. Fill in register packing only after choosing an exact GFX and
  LLVM IR / AMD GCN ISA instruction shape.
- `composable-kernel_ck_tile_reference_notes.md`: a checklist for using Composable Kernel/CK Tile as
  reference material, competitor, and extension surface.

## Benchmark Seed

Use the benchmarkable task under:

```text
corpus/tasks/rocwmma-mfma-gemm/
```

Build the seed with:

```powershell
hipcc -std=c++17 -O3 -arch=gfx90a -DVARIANT_OPTIMIZED `
  harnesses/rocwmma_gemm_benchmark.hip `
  corpus/tasks/rocwmma-mfma-gemm/source/optimized.hip `
  -o build/rocwmma_optimized.exe
```

Do not label a result as measured until the harness has run on a named GPU.

## Agent Checklist

1. State the exact dtype, accumulator, layout, leading dimensions, epilogue, and
   tolerance.
2. Run the scalar baseline and rocWMMA path against the same CPU reference.
3. Add hipBLASLt and Composable Kernel baselines before claiming a custom win.
4. Keep TF32, FP16, BF16, FP8, INT8, and sub-byte tasks separate unless the
   task explicitly studies a math-mode tradeoff.
5. Record `timing-only` when HIP events are the only evidence.

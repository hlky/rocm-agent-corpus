# rocWMMA Matrix Core GEMM Notes

This task is a narrow seed for custom Matrix Core kernels. It is meant to be
small enough to read and benchmark, then strong enough to serve as the first
step toward inline MFMA, shared-memory staging, and Composable Kernel/CK Tile variants.

## Contract

```text
C[M, N] = A[M, K] * B[K, N]
```

- A is FP16 row-major.
- B is FP16 column-major.
- C is FP32 row-major.
- Accumulation is FP32.
- No epilogue is fused in the seed source.

The column-major B operand keeps the first rocWMMA kernel close to the CUDA Samples
teaching pattern. A production row-major GEMM should document its wrapper,
transpose, or shared-memory staging strategy before comparing against libraries.

## Build

```powershell
hipcc -std=c++17 -O3 -arch=gfx90a -DVARIANT_BASELINE `
  harnesses/rocwmma_gemm_benchmark.hip `
  corpus/tasks/rocwmma-mfma-gemm/source/baseline.hip `
  -o build/rocwmma_baseline.exe

hipcc -std=c++17 -O3 -arch=gfx90a -DVARIANT_OPTIMIZED `
  harnesses/rocwmma_gemm_benchmark.hip `
  corpus/tasks/rocwmma-mfma-gemm/source/optimized.hip `
  -o build/rocwmma_optimized.exe
```

Replace `gfx90a` with the exact benchmark target and record the full command in
any result JSON.

## Run

```powershell
.\build\rocwmma_baseline.exe 256 256 256 10 50
.\build\rocwmma_optimized.exe 256 256 256 10 50
```

Arguments are:

```text
benchmark [M] [N] [K] [warmup_iters] [measure_iters] [abs_tol] [rel_tol]
```

The optimized launcher uses rocWMMA only for dimensions divisible by 16. Ragged
shapes fall back to a scalar HIP kernel so correctness runs remain safe while
future agents add masked tile staging.

## Next Experiments

- Add shared-memory staging and a padded/skewed layout.
- Increase output tiles per CTA to improve occupancy.
- Add a fused bias or activation epilogue and compare against hipBLASLt epilogue
  support.
- Replace rocWMMA with inline `mfma` for a fixed GFX target and record register
  fragment layout.
- Rebuild the same contract with Composable Kernel/CK Tile and record CTA tile, warp tile,
  instruction tile, stages, schedule, swizzle, and epilogue.
- Add separate BF16, TF32, and FP8 tasks instead of mixing math contracts.

# hipBLAS/rocBLAS and hipBLASLt Guide

## Use hipBLAS/rocBLAS When

- The operation is standard BLAS.
- Layouts are conventional enough to express with transpose flags and leading
  dimensions.
- You need reliable peak-ish GEMM performance quickly.

Common calls:

- `hipblasSgemm`, `hipblasGemmEx`
- `hipblasGemmStridedBatchedEx`
- `hipblasLtMatmul` for newer descriptor-driven matmul

## Use hipBLASLt When

- You need matmul epilogues such as bias or activation.
- You need algorithm selection and workspace tuning.
- You need more control over matrix layouts.
- You want to evaluate TF32, FP16, BF16, FP8, or int8 paths.

## hipBLASLt Concepts

Descriptors:

- `hipblasLtMatmulDesc_t`: operation, compute type, epilogue, pointer mode.
- `hipblasLtMatrixLayout_t`: matrix dtype, rows, cols, leading dimension, batch.
- `hipblasLtMatmulPreference_t`: workspace and implementation preferences.

Algorithm search:

- Use `hipblasLtMatmulAlgoGetHeuristic`.
- Run more than one heuristic result when performance matters.
- Record workspace size and selected algorithm id in benchmark metadata.

Epilogues:

- Bias.
- ReLU.
- GELU in supported versions/configurations.
- Auxiliary output paths for backward-like workflows.

Agent checks:

- Are dimensions expressed in column-major hipBLAS/rocBLAS terms or row-major wrappers?
- Are leading dimensions correct?
- Is TF32 enabled or disabled intentionally?
- Is workspace included in reproducibility metadata?
- Is the epilogue mathematically equivalent to the custom kernel?

## Example Locations

- `third_party/rocblas-examples`
- `examples/hipblaslt/matmul_bias_relu_skeleton.hip`
- `harnesses/hipblaslt_hgemm_benchmark.hip`

## Corpus Baseline Runner

The repo includes a first library-baseline runner for the
`rocwmma-mfma-gemm` seed:

```powershell
python tools/run_library_baseline.py rocwmma-mfma-gemm:hipblaslt-hgemm --m 256 --n 256 --k 256 --arch gfx1030 --write-result
```

This benchmark keeps the seed contract:

- `A` is FP16 row-major `[M,K]`.
- `B` is FP16 column-major `[K,N]`.
- `C` is FP32 row-major `[M,N]`.
- Accumulation is FP32 with no epilogue.

The hipBLASLt descriptor maps the task as `C^T[N,M] = B^T[N,K] * A^T[K,M]` in
default column-major layouts. Record that mapping in any result so row-major
wrapper mistakes do not look like performance claims.

No ROCm hipBLASLt result artifact is currently checked in. The CUDA-origin
corpus had a library timing seed for this contract, but it was intentionally not
copied; rerun the custom rocWMMA and hipBLASLt paths on the same AMD GPU before
making any custom-vs-library claim.

## Corpus Tasks to Add

- GEMM baseline with naive custom kernel versus hipBLAS/rocBLAS.
- GEMM + bias + ReLU custom fusion versus hipBLASLt epilogue.
- Strided batched GEMM versus looped GEMM launches.
- Small GEMM batch where launch overhead and grouped shapes matter.
- TF32 on/off correctness/performance comparison.

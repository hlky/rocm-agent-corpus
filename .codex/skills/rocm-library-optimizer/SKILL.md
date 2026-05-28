---
name: rocm-library-optimizer
description: Use ROCm libraries as baselines, references, and extension surfaces for optimization work.
---

# HIP/ROCm Library Optimizer

Use this skill for GEMM, reductions, scans, sorts, softmax-like ML kernels,
sparse operations, FFTs, inference engines, and other HIP/ROCm library-adjacent
work. The goal is not to stop at the library; it is to understand what must be
beaten or customized.

## Workflow

1. Read `docs/CUSTOM_KERNEL_COMPETITION_GUIDE.md`.
2. Read `docs/LIBRARY_GUIDE.md` to identify the strongest library baseline and
   extension surface.
3. For GEMM or matmul fusion, read `docs/HIPBLAS_ROCBLAS_GUIDE.md`.
4. For custom GEMM fusion or research kernels, read
   `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md`.
5. For attention, sampling, or low-bit work, read the relevant self-contained
   baseline ladder:
   `docs/ATTENTION_LIBRARY_BASELINES.md`,
   `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`, or
   `docs/QUANTIZATION_LIBRARY_BASELINES.md`.
6. Inspect examples under `examples/` and upstream submodules under
   `third_party/`.
7. State the assumptions the custom kernel can exploit that the generic library
   must preserve.
8. Benchmark against the best available library baseline.
9. If the library wins, record the loss and the next narrower specialization to
   try.

---
name: rocm-custom-kernel-competitor
description: Build and evaluate custom HIP kernels against ROCm library and framework baselines.
---

# HIP/ROCm Custom Kernel Competitor

Use this skill when the task is to write, improve, or evaluate a custom HIP
kernel against hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, hiphipCUB/rocPRIM/hiphipCUB/rocThrust, MIGraphX, Triton, MIOpen, or
framework-generated code.

## Workflow

1. Read `docs/CUSTOM_KERNEL_COMPETITION_GUIDE.md`.
2. Read `docs/CASE_COVERAGE_PLAN.md` and the relevant track or library-baseline
   guide so the task is attached to the corpus map.
3. Define the scoped workload: shapes, dtype, layout, strides, tolerance, GPU,
   and timing boundary.
4. Establish the strongest vendor/framework baseline and a simple correctness
   oracle.
5. Identify which generality the custom kernel can drop.
6. Pick one attack surface: fusion, fixed shape, layout, memory traffic,
   epilogue, launch overhead, architecture-specific feature, or integration
   boundary.
7. Implement and measure one hypothesis at a time.
8. Record wins, losses, and neutral results as corpus data.

## Rules

- Do not answer with "use the library" as the final move when the request is
  custom-kernel optimization.
- Do use libraries to define the performance bar and correctness contract.
- Treat losing custom attempts as useful if they teach why the baseline is
  strong.
- Keep evidence labels honest: timing-only, counter-backed, negative, or
  template-only.

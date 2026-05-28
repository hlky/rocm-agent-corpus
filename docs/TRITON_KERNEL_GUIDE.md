# Triton Kernel Guide

Triton is not HIP C++, but it belongs in this corpus because it is a common and
productive route for GPU kernel optimization, especially in ML workloads.

## Use Triton When

- The kernel is tensor-program-like and shape-specialized.
- Python integration and iteration speed matter.
- You need a strong custom-kernel baseline before writing HIP C++.
- You want portability across AMD/ROCm and other backends, subject to compiler
  support.

## Compare Against HIP C++ When

- You need low-level control over `global-to-LDS staging`, global-to-LDS staging, WGMFMA, inline LLVM IR / AMD GCN ISA, or exact
  memory ordering.
- Register pressure or instruction scheduling must be inspected at AMD GCN ISA level.
- The kernel is used inside MIGraphX/Composable Kernel/plugin infrastructure.
- You need maximal architecture-specific tuning.

## Corpus Tasks to Add

- Triton SAXPY versus CUDA SAXPY.
- Triton rowwise softmax versus CUDA rowwise softmax.
- Triton matmul versus hipBLAS/rocBLAS/Composable Kernel.
- Triton layernorm and RMSNorm.
- Triton fused attention educational kernel.

## Agent Checklist

- Record block sizes, num warps, num stages, and constexpr meta-parameters.
- Separate compile time from runtime.
- Include warmup iterations.
- Compare against framework compiler output where relevant.
- Inspect generated LLVM IR / AMD GCN ISA for architecture-sensitive claims.

Submodule:

- `third_party/triton`


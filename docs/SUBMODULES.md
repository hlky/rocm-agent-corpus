# Submodules

Third-party repositories are tracked as ROCm-oriented submodule candidates so agents can inspect real upstream code without copying large projects into corpus records.

The populated CUDA `third_party/` checkout from the source corpus was not copied. Initialize only the submodules needed for a task, pin commits, and preserve upstream license metadata before importing code.

## Current Submodules

| Path | Upstream |
| --- | --- |
| `third_party/rocm` | `https://github.com/ROCm/ROCm.git` |
| `third_party/rocm-examples` | `https://github.com/ROCm/rocm-examples.git` |
| `third_party/rocm-libraries` | `https://github.com/ROCm/rocm-libraries.git` |
| `third_party/composable-kernel` | `https://github.com/ROCm/composable_kernel.git` |
| `third_party/rocblas-examples` | `https://github.com/ROCm/rocBLAS-Examples.git` |
| `third_party/miopen` | `https://github.com/ROCm/MIOpen.git` |
| `third_party/rccl` | `https://github.com/ROCm/rccl.git` |
| `third_party/rocshmem` | `https://github.com/ROCm/rocSHMEM.git` |
| `third_party/migraphx` | `https://github.com/ROCm/AMDMIGraphX.git` |
| `third_party/triton` | `https://github.com/triton-lang/triton.git` |
| `third_party/vllm` | `https://github.com/vllm-project/vllm.git` |
| `third_party/flash-attention` | `https://github.com/Dao-AILab/flash-attention.git` |
| `third_party/flashinfer` | `https://github.com/flashinfer-ai/flashinfer.git` |
| `third_party/bitsandbytes` | `https://github.com/bitsandbytes-foundation/bitsandbytes.git` |
| `third_party/pytorch` | `https://github.com/pytorch/pytorch.git` |
| `third_party/kernelbench` | `https://github.com/ScalingIntelligence/KernelBench.git` |
| `third_party/gpu-mode-lectures` | `https://github.com/gpu-mode/lectures.git` |
| `third_party/gpu-mode-reference-kernels` | `https://github.com/gpu-mode/reference-kernels.git` |

## Clone

```bash
git clone --recurse-submodules <repo>
```

If already cloned:

```bash
git submodule update --init --recursive
```

## Agent Rule

When using a submodule as evidence, record the submodule path, commit SHA, specific file path, copied or adapted snippet, and license cleanup required before public release.

# CUDA Vendor Library Reference Guide

This is not a "delegate to the library and stop" guide. The corpus goal is to
help agents write custom kernels that can beat, match, specialize, or extend
vendor libraries for a scoped workload.

Use libraries as:

- Baselines to beat.
- Correctness oracles.
- Sources of API and layout constraints.
- Profiling targets that reveal what strong implementations do.
- Extension surfaces, such as Composable Kernel epilogues, hipBLASLt epilogues, hipCUB block
  primitives, MIGraphX plugins, or Triton kernels.

The library path is still valuable when it wins. But a winning library result
should become training data: what assumptions made it win, what fixed-shape or
fused custom kernel might beat it, and what architecture features it likely uses.

## Competition Tree

1. Dense matrix multiply or linear algebra:
   - Establish hipBLAS/rocBLAS/hipBLASLt as the baseline and correctness oracle.
   - Inspect hipBLASLt descriptors, math mode, epilogue support, workspace, and
     selected algorithm behavior.
   - Build custom candidates through Composable Kernel/CK Tile, direct HIP, or Triton when
     fixed shapes, fusion, custom layout, special numerics, or reduced overhead
     may beat the general library path.

2. Reductions, scans, selections, sorts:
   - Use rocPRIM/hipCUB/rocThrust/hipCUB as the baseline and as internal building blocks.
   - Write custom kernels when the operation can be fused, shape-specialized,
     privatized differently, or simplified relative to the generic primitive.
   - Compare against both hipCUB and a deliberately simple educational baseline.

3. FFT, sparse, solvers, images, compression:
   - Use cuFFT, cuSPARSE, cuSOLVER, NPP, nvJPEG, nvCOMP, or library samples as
     reference implementations and performance baselines.
   - Custom work should focus on fusion around the library call, restricted
     formats, fixed sizes, or eliminating data motion.

4. Neural network primitives:
   - Compare against framework kernels, MIOpen/frontend, hipBLASLt, Composable Kernel,
     MIGraphX, vLLM on ROCm, and Triton where applicable.
   - Custom kernels should target fusion, fixed shapes, memory-layout knowledge,
     specialized masks, quantization paths, or lower dispatch overhead.

5. Multi-GPU:
   - Use RCCL/rocSHMEM as communication baselines and integration targets.
   - Custom kernels compete by overlapping communication, changing partitioning,
     reducing bytes moved, or fusing communication-adjacent work.

## Library Map

| Library | Use When | Corpus Paths |
| --- | --- | --- |
| hipBLAS/rocBLAS | Standard BLAS, GEMM, batched GEMM | `docs/HIPBLAS_ROCBLAS_GUIDE.md`, `third_party/rocm-examples` |
| hipBLASLt | Matmul with descriptors, epilogues, alg search | `docs/HIPBLAS_ROCBLAS_GUIDE.md`, `harnesses/hipblaslt_hgemm_benchmark.hip`, `examples/hipblaslt` |
| Composable Kernel/CK Tile | Custom GEMM, Matrix Core kernels, custom epilogues | `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md`, `third_party/composable-kernel` |
| hipCUB | Device/block/warp primitives | `docs/ROCPRIM_HIPCUB_ROCTHRUST_GUIDE.md`, `docs/ROCPRIM_COMPETITION_TRACK.md`, `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`, `third_party/rocm-libraries`, `examples/rocprim` |
| rocThrust | STL-like GPU algorithms | `third_party/rocm-libraries`, `examples/rocthrust` |
| cuFFT | FFTs and convolution via FFT | `third_party/rocm-examples` |
| cuSPARSE | Sparse matrix kernels | `docs/SPARSE_IRREGULAR_KERNEL_GUIDE.md`, `corpus/tasks/csr-spmv-load-balance`, `third_party/rocm-examples` |
| cuSOLVER | Dense/sparse factorization and solves | `third_party/rocm-examples` |
| MIOpen | Neural network primitives | external docs and framework integrations |
| RCCL | Multi-GPU collectives | `docs/MULTIGPU_GUIDE.md`, `third_party/rccl` |
| rocSHMEM | GPU-side communication | `docs/MULTIGPU_GUIDE.md`, `third_party/rocshmem` |
| FlashAttention | Attention optimization reference | `docs/ATTENTION_LIBRARY_BASELINES.md`, `docs/MIOPEN_ATTENTION_GUIDE.md`, `third_party/flash-attention` |
| MIGraphX | Inference engine build/runtime/plugin deployment | `docs/MIGRAPHX_INFERENCE_GUIDE.md`, `third_party/migraphx` |
| vLLM on ROCm | LLM inference, plugins, KV cache, serving | `docs/MIGRAPHX_INFERENCE_GUIDE.md`, `docs/ATTENTION_LIBRARY_BASELINES.md`, `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`, `docs/QUANTIZATION_LIBRARY_BASELINES.md`, `third_party/vllm-rocm` |
| OneFlow | Framework HIP operator/runtime patterns | `docs/FRAMEWORK_EXTENSION_GUIDE.md`, `third_party/oneflow` |
| Triton | Python-authored GPU kernels and compiler path | `docs/TRITON_KERNEL_GUIDE.md`, `third_party/triton` |
| vLLM | Paged attention, sampling, serving scheduler | `docs/ATTENTION_LIBRARY_BASELINES.md`, `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`, `third_party/vllm` |
| FlashInfer | Decode/prefill attention, paged KV, sampling | `docs/ATTENTION_LIBRARY_BASELINES.md`, `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`, `third_party/flashinfer` |
| bitsandbytes | Low-bit matmul/GEMV and optimizers | `docs/QUANTIZATION_LIBRARY_BASELINES.md`, `third_party/bitsandbytes` |
| Transformer Engine on ROCm | FP8, MXFP8, NVFP4, transformer kernels | `docs/QUANTIZATION_LIBRARY_BASELINES.md`, `third_party/transformer-engine` |

## Agent Rule

When a library is relevant, an agent should write down:

- Which library path is the baseline.
- Which assumptions the library must support that the custom kernel can drop.
- Which fusion, layout, datatype, architecture, or launch behavior the custom
  kernel exploits.
- What correctness and performance bar the custom kernel must beat.
- If the library wins, what was learned and which narrower workload might still
  be beatable.

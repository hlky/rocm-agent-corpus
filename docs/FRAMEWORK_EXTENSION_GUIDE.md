# Framework Extension Guide

Agents often need to optimize kernels inside a framework rather than as isolated
HIP C++ programs. This guide maps those paths.

## PyTorch Extensions

Use when:

- A Python model needs a custom HIP op.
- You need quick integration with tensors, autograd, or benchmark scripts.
- A Triton kernel is insufficient or not portable enough.

Key pieces:

- `torch.utils.cpp_extension`
- C++ binding layer.
- HIP `.hip` implementation.
- Autograd wrapper when backward is needed.
- Shape, dtype, device checks.

Corpus tasks to add:

- PyTorch extension for SAXPY.
- PyTorch extension for rowwise softmax.
- Autograd extension for layernorm.
- Benchmark against PyTorch eager and TorchInductor/Triton.

## OneFlow

OneFlow is useful as a framework-internal reference because it is designed
around distributed training and contains HIP operator/runtime patterns.

Look for:

- HIP op kernels.
- SBP/distributed abstractions.
- Actor/runtime scheduling.
- Memory planning and tensor layout decisions.
- Framework-level fusion or placement decisions.

Submodule:

- `source:oneflow`

Corpus tasks to add:

- OneFlow operator anatomy notes.
- OneFlow HIP kernel case study.
- Distributed tensor placement example.
- Comparison of framework-level fusion versus hand-written HIP extension.

## TensorFlow and XLA

Planned references:

- TensorFlow custom ops.
- XLA custom calls.
- HIP graph capture interactions.
- Shape specialization and compiler-generated kernels.

## JAX and Pallas

Planned references:

- Pallas GPU kernels.
- XLA lowering.
- Shape-polymorphic benchmark caveats.

## Agent Rule

When optimizing in a framework, record:

- Framework version.
- ROCm/HIP and compiler versions.
- Tensor shapes, strides, dtypes, layouts.
- Whether timing includes framework dispatch and synchronization.
- Whether graph compilation or warmup is included.
- Baselines: eager, compiler-generated, Triton, library, and custom HIP where
  possible.


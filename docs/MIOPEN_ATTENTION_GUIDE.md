# MIOpen, Attention, and ML Kernel Guide

This guide covers neural-network primitives where agents often overreach with
custom kernels.

## MIOpen

Use MIOpen when:

- The operation is a supported neural-network primitive.
- You need graph-style fusion.
- You need operation plans and engine configuration.
- You are implementing attention or convolution variants that MIOpen can express.

Agent workflow:

1. Inspect `third_party/miopen-frontend`.
2. Build the operation graph first.
3. Generate engine configs.
4. Filter configs for workspace, determinism, numeric behavior, and hardware.
5. Benchmark selected plans against framework and custom baselines.

## Attention

High-performance attention is an IO-aware algorithm problem, not just a softmax
problem.

Core ideas:

- Tile Q, K, V to reduce HBM traffic.
- Keep partial statistics for online softmax.
- Avoid materializing the full attention matrix when possible.
- Use Matrix Cores for QK and PV matmuls.
- Handle masks, causal bounds, dropout, and variable sequence lengths explicitly.

Reference paths:

- `third_party/flash-attention`
- `third_party/miopen-frontend`
- `third_party/composable-kernel`
- `third_party/gpu-mode-reference-kernels`

## ML Kernel Families to Cover

- Softmax.
- LayerNorm and RMSNorm.
- GELU, SiLU, and activation fusion.
- Attention forward and backward.
- Optimizer update kernels.
- Quantization/dequantization.
- MoE routing and grouped GEMM.
- Embedding lookup and gradient accumulation.

## Agent Warnings

- Do not compare attention kernels without matching mask, scale, dtype,
  dropout, causal behavior, and accumulation precision.
- Backward kernels have different bottlenecks and correctness risks.
- Small sequence lengths may be launch-bound.
- Very long sequence lengths may need split-K/V or paged attention strategies.


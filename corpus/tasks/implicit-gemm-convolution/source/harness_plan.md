# Harness Plan

- Operation: implicit-gemm-convolution
- Baseline: MIOpen/backend convolution plan, Composable Kernel implicit-GEMM convolution, MIGraphX inference tactic, or framework convolution with matching layout and dtype.
- Optimized candidate: Fixed-shape implicit-GEMM HIP or Composable Kernel-derived kernel that specializes iterator math, tile shape, layout, padding, and epilogue.
- Oracle: MIOpen, Composable Kernel, or CPU convolution reference across layout, padding, stride, and dilation cases.
- Recommended shapes: N=32 H=W=56 C=64 K=128 R=S=3 stride=1 pad=1 NHWC fp16; N=1 H=W=224 C=3 K=64 R=S=7 stride=2 pad=3 NCHW fp32; N=16 H=W=28 C=128 K=256 R=S=1 stride=1 NHWC bf16
- Required metrics: median end-to-end latency; math mode; layout; workspace bytes; algorithm or tactic; Matrix Core eligibility; correctness max error
- Evidence notes: Include build flags, GPU, MIOpen/Composable Kernel/MIGraphX versions, and whether timing includes layout transforms or framework dispatch.

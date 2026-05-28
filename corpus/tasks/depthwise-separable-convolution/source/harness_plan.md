# Harness Plan

- Operation: depthwise-separable-convolution
- Baseline: MIOpen grouped/depthwise convolution, MIGraphX tactic, framework kernel, or separate depthwise plus pointwise HIP stages with materialized intermediate output.
- Optimized candidate: Channel-specialized depthwise kernel, NHWC vectorized channel-pack kernel, or fused depthwise/bias/activation/pointwise path.
- Oracle: MIOpen, MIGraphX, framework, or CPU convolution reference for depthwise-only and depthwise-plus-pointwise modes.
- Recommended shapes: N=1 H=W=112 C=32 R=S=3 stride=1 NHWC fp16 depthwise-only; N=32 H=W=56 C=64 R=S=3 stride=1 NHWC fp16 depthwise-plus-pointwise K=128; N=8 H=W=28 C=256 R=S=5 stride=1 NCHW fp32 depthwise-only
- Required metrics: median end-to-end latency; layout; channel multiplier; fusion boundary; workspace bytes; library tactic or algorithm; correctness max error
- Evidence notes: State whether the measured boundary includes pointwise convolution and activation. Keep HIP-event-only records labeled timing-only.

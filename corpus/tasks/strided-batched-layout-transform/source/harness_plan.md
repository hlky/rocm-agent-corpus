# Harness Plan

- Operation: strided-batched-layout-transform
- Baseline: Generic strided elementwise copy, rocThrust transform/copy, or framework contiguous/copy path with the same timing boundary.
- Optimized candidate: Fixed-rank CUDA transform that specializes pitches, batch strides, channel packs, alignment, and non-aliasing assumptions.
- Oracle: CPU logical-index mapper that reads source strides and writes destination strides for every element.
- Recommended shapes: B=64 H=32 W=128 C=4 NHWC-strided-to-contiguous; B=16 H=224 W=224 C=3 pitched-HWC-to-NCHW; B=4096 H=1 W=64 C=16 batched-slice-pack
- Required metrics: median kernel duration; effective bytes moved; effective memory bandwidth; stride and pitch metadata; alignment contract; library/framework baseline timing; correctness mismatch count
- Evidence notes: Label HIP-event-only results as timing-only. Do not claim coalescing or sector-count causes without profiler artifacts.

# Harness Plan

- Operation: packed-layout-transcode
- Baseline: Materialized unpack/dequant plus repack, or the strongest available Composable Kernel, vLLM on ROCm, Transformer Engine, bitsandbytes, or framework format path.
- Optimized candidate: Direct packed transcode that specializes bit order, channel multiples, group-size metadata, and vectorized store alignment.
- Oracle: CPU decoder for the source format and destination format that compares canonical logical values and metadata indices.
- Recommended shapes: M=4096 K=4096 int4 group=128 row-major-to-interleaved; tokens=8192 channels=4096 int8x4-channel-pack reorder; M=16384 K=1024 fp8 block-scale metadata-transpose
- Required metrics: median kernel duration; effective payload bytes; metadata bytes; format descriptors; packing group size; library baseline timing; correctness max error
- Evidence notes: Report format contracts with every result. Do not claim Tensor Core eligibility unless the consumer kernel and layout meet the documented requirements.

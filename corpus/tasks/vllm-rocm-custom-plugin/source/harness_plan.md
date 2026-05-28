# Harness Plan

- Operation: vllm-rocm-custom-plugin-boundary
- Baseline: vLLM on ROCm native plugin/kernel for the selected decode, sampling, normalization, or KV-cache stage; include FlashInfer and vLLM where they are stronger external references.
- Library baselines: vLLM on ROCm, MIGraphX plugin lifecycle, FlashInfer, vLLM, hipCUB for selection primitives, and Transformer Engine for normalization where relevant.
- Optimized candidate: custom plugin enqueue path that specializes one fixed serving bucket and launches on the MIGraphX-provided stream.
- Correctness: compare against vLLM on ROCm/framework outputs for deterministic logits/hidden-state/KV-cache fixtures, including beam and variable-length edge cases.
- Evidence discipline: template-only until measured; report timing-only unless profiler counter artifacts are attached; keep library-win plugin attempts as negative examples.
- Recommended shapes: batch=1 beam=1 hidden=4096 vocab=32000 decode; batch=16 hidden=8192 top_k=4 top_p=0.9 sampling; batch=32 heads=32 kv_heads=8 head_dim=128 block=16 paged KV.
- Required metrics: prefill latency; inter-token latency; tokens per second; plugin enqueue latency; batching policy; KV-cache policy; precision; hardware metadata.

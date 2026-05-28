# Harness Plan

- Operation: multi-tensor-adamw
- Baselines: PyTorch RDNA3mW foreach, PyTorch fused RDNA3mW when available, Apex-style fused RDNA3m, bitsandbytes optimizer paths.
- Correctness: compare all parameters and optimizer states against a PyTorch/CPU reference after one step and repeated steps.
- Optimized candidate: one launch over a tensor list with chunk scheduling, vectorized aligned spans, scalar tails, and recorded tensor-size histogram.
- Recommended shapes: `1024 tensors x 4096 elements fp32`; `128 tensors mixed 1K..1M elements`; transformer block shard histogram; many tiny tensors totaling `262144` elements.
- Required metrics: optimizer step latency, tensor count, total elements, dtype/state dtype, alignment, baseline versions, hardware/build metadata, correctness max error, evidence label.
- Evidence discipline: include framework dispatch boundary explicitly; use `timing-only` unless profiler counters are attached.

# Wave 3 Tracking

This wave expands the corpus from measured seed kernels into higher-value custom
kernel domains. The intent is still competition and specialization, not
automatic delegation to libraries.

## ROCm Timing Status

The source CUDA corpus had an A4000 timing pass for several wave 3 scaffolds.
Those result artifacts were not copied into this ROCm corpus. The rows below
are ROCm follow-up targets and remain unmeasured until rerun on AMD hardware.

| Track | Owned Paths | Done Means |
| --- | --- | --- |
| Attention / FlashAttention-style forward | `docs/ATTENTION_KERNEL_GUIDE.md`, `corpus/tasks/online-attention-forward/`, `harnesses/attention_forward_benchmark.hip`, `data/index/attention_kernel_track.json` | ROCm timing-only rerun needed |
| Selection / sampling | `docs/SELECTION_SAMPLING_KERNEL_GUIDE.md`, `corpus/tasks/block-topk-sampling/`, `harnesses/topk_sampling_benchmark.hip`, `data/index/selection_sampling_track.json` | ROCm timing-only rerun needed |
| Quantization / fused dequant | `docs/QUANTIZATION_KERNEL_GUIDE.md`, `corpus/tasks/fused-int4-dequant-gemv/`, `harnesses/int4_dequant_gemv_benchmark.hip`, `data/index/quantization_kernel_track.json` | ROCm timing-only rerun needed |
| Matrix Core / rocWMMA | `docs/MATRIX_CORE_ROCWMMA_GUIDE.md`, `corpus/tasks/rocwmma-mfma-gemm/`, `harnesses/rocwmma_gemm_benchmark.hip`, `examples/matrix_cores/`, `data/index/matrix_core_track.json` | ROCm timing-only rerun needed |

## Acceptance Notes

- Each task must describe custom-kernel win conditions and library comparison
  points separately.
- Each task should include enough harness metadata for `tools/run_hip_task.py`
  integration in a later pass.
- Unmeasured scaffolds should be marked as such. Do not imply profiler evidence
  without a future `rocprof` run on a counter-enabled machine.
- Negative results are first-class records. A slower seed is still useful when
  the curation note explains why.

## Follow-Up GPU Work

- Add library baselines for each accepted task.
- Sweep additional shapes on Runpod or another cheap GPU.
- Capture rocprofiler/rocprof counters later on a machine with GPU counter access.
- Add library baselines where practical: Composable Kernel/CK Tile for GEMM, hipCUB/rocPRIM/hipCUB/rocThrust for
  selection and scans, FlashAttention/FlashInfer/vLLM/vLLM on ROCm for attention
  and decoding, bitsandbytes/Transformer Engine on ROCm/vLLM on ROCm for quantization.

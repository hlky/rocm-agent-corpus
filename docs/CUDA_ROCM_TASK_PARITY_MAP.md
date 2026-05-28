# CUDA to ROCm Task Parity Map

This map keeps ROCm coverage comparable to `H:/cuda-agent-corpus` without
treating CUDA-origin measurements as ROCm evidence.

## Policy

- Identical task IDs mean the HIP/ROCm corpus keeps the same problem contract.
- Renamed task IDs mean the CUDA feature is hardware-, runtime-, or
  library-specific and has a ROCm-native equivalent.
- CUDA-origin results are historical context only. ROCm records must attach AMD
  hardware, ROCm version, `hipcc` flags, `gfx` target, correctness, timing, and
  evidence label.

## Validation

Run the parity checker after adding, renaming, or removing tasks:

```bash
python tools/check_cuda_rocm_task_parity.py --source H:/cuda-agent-corpus
```

The checker verifies that every CUDA task directory has a same-ID ROCm scaffold
or a renamed equivalent listed below, and that ROCm has no unmapped extra task
directory.

## Renamed Equivalents

| CUDA task | ROCm task | Reason |
| --- | --- | --- |
| `compute-sanitizer-racecheck` | `rocm-sanitizer-racecheck` | ROCm sanitizer/debugger workflow |
| `cp-async-tiled-copy` | `lds-tiled-copy` | ROCm LDS staging scaffold |
| `cuda-graphs-launch-overhead` | `hip-graphs-launch-overhead` | HIP Graph runtime |
| `cuda-ipc-multiprocess` | `hip-ipc-multiprocess` | HIP IPC runtime |
| `cudnn-frontend-fusion` | `miopen-frontend-fusion` | MIOpen graph/fusion path |
| `cutlass-custom-epilogue-visitor` | `composable-kernel-custom-epilogue` | Composable Kernel epilogue path |
| `hopper-wgmma-tma-gemm` | `cdna-mfma-gemm` | CDNA MFMA/LDS GEMM scaffold |
| `inline-mma-sync-skeleton` | `inline-mfma-skeleton` | AMD MFMA instruction path |
| `int8-dp4a-imma-gemm` | `int8-dot-mfma-gemm` | ROCm INT8 dot/MFMA ladder |
| `nccl-overlap-allreduce` | `rccl-overlap-allreduce` | RCCL collective path |
| `nsight-counter-roofline` | `rocprof-counter-roofline` | rocprofiler/rocprof evidence path |
| `nvrtc-specialized-kernel-cache` | `hiprtc-specialized-kernel-cache` | hipRTC specialization path |
| `nvshmem-queue` | `rocshmem-queue` | rocSHMEM one-sided communication |
| `pytorch-oneflow-extension-op` | `pytorch-hip-extension-op` | PyTorch HIP extension boundary |
| `sass-diff-regression` | `amdisa-diff-regression` | AMD GCN ISA/codegen diffing |
| `sm80-sm86-ampere-split` | `gfx90a-gfx942-cdna-split` | ROCm GFX generation split |
| `sm90a-hopper-wgmma-tma` | `gfx942-cdna3-mfma-lab` | CDNA3 MFMA/LDS lab |
| `sm100-sm120-blackwell-path` | `gfx950-gfx1200-rocm-portability` | CDNA4/RDNA4 portability path |
| `tensorrt-engine-tactic-sweep` | `migraphx-engine-tuning-sweep` | MIGraphX engine/tactic path |
| `tensorrt-llm-custom-plugin` | `vllm-rocm-custom-plugin` | vLLM on ROCm plugin/kernel path |
| `tensorrt-plugin-fused-op` | `migraphx-custom-op-fused-op` | MIGraphX custom op/plugin path |
| `tma-multicast-gemm` | `global-to-lds-mfma-gemm` | LDS-staged MFMA GEMM boundary |
| `triton-vs-cuda-row-kernel` | `triton-vs-hip-row-kernel` | HIP row-kernel comparison |
| `warp-specialized-wgmma` | `wave-specialized-mfma-pipeline` | ROCm wave-role MFMA pipeline |
| `wmma-tensorcore-gemm` | `rocwmma-mfma-gemm` | rocWMMA/MFMA Matrix Core seed |

All other CUDA task IDs currently have same-ID ROCm task scaffolds.

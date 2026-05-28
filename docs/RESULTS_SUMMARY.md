# Results Summary

This corpus was created from the CUDA agent corpus on 2026-05-20. The
CUDA-origin timing result files and JSONL records were removed during the ROCm
port so agents do not confuse NVIDIA measurements with HIP/ROCm evidence.

## Checked-In ROCm Measurements

These are HIP-event `timing-only` records on AMD Radeon RX 9070 XT (`gfx1201`)
with ROCm/PyTorch from `H:\dinoml_v2\.venv\rocm`. No rocprofiler/rocprof
counters are attached yet.

| Task | Shape | Baseline ms | Optimized ms | Speedup | Record |
| --- | --- | ---: | ---: | ---: | --- |
| `memory-coalesced-matrix-copy` | 4096x4096 fp32 copy | 3.282050 | 0.235300 | 13.948364x | `data/records/memory_coalesced_matrix_copy_gfx1201_20260528.jsonl` |
| `block-reduction-sum` | 16777216 fp32 elements | 36.872601 | 2.191500 | 16.825280x | `data/records/block_reduction_sum_gfx1201_20260528.jsonl` |
| `rowwise-softmax` | 4096x1024 fp32 rows | 0.570600 | 0.071700 | 7.958159x | `data/records/rowwise_softmax_gfx1201_20260528.jsonl` |

## First Measurement Targets

- `shared-memory-tiled-transpose` with `tools/run_matrix_task.py`.
- `vectorized-saxpy`, `block-topk-sampling`, and `fused-int4-dequant-gemv` with `tools/run_hip_task.py`.
- `rocwmma-mfma-gemm` with both `tools/run_hip_task.py` and `tools/run_library_baseline.py` for hipBLASLt comparison.

Record HIP-event timing as `timing-only`. Promote a result to `counter-backed-measured` only after attaching rocprofiler/rocprof counter artifacts plus hardware and build metadata.

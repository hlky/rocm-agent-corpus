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
| `shared-memory-tiled-transpose` | 4096x4096 fp32 transpose | 14.533300 | 0.658300 | 22.077017x | `data/records/shared_memory_tiled_transpose_gfx1201_20260528.jsonl` |
| `block-reduction-sum` | 16777216 fp32 elements | 36.872601 | 2.191500 | 16.825280x | `data/records/block_reduction_sum_gfx1201_20260528.jsonl` |
| `rowwise-softmax` | 4096x1024 fp32 rows | 0.570600 | 0.071700 | 7.958159x | `data/records/rowwise_softmax_gfx1201_20260528.jsonl` |
| `block-topk-sampling` | 1024x32768 k=4 temperature=0.8 | 8.291950 | 0.590200 | 14.049390x | `data/records/block_topk_sampling_gfx1201_20260528.jsonl` |
| `vectorized-saxpy` | 16777216 fp32 elements | 0.337750 | 0.333600 | 1.012440x | `data/records/vectorized_saxpy_gfx1201_20260528.jsonl` |
| `fused-int4-dequant-gemv` | 4096x4096 group_size=128 | 0.477550 | 0.087400 | 5.463959x | `data/records/fused_int4_dequant_gemv_gfx1201_20260528.jsonl` |

## Shape Sweep Measurements

| Task | Shape | Baseline ms | Optimized ms | Speedup | Record |
| --- | --- | ---: | ---: | ---: | --- |
| `vectorized-saxpy` | 16777219 fp32 elements | 0.360000 | 0.340200 | 1.058201x | `data/records/vectorized_saxpy_tail_gfx1201_20260528.jsonl` |
| `fused-int4-dequant-gemv` | 4096x4097 group_size=128 | 0.584200 | 0.091850 | 6.360370x | `data/records/fused_int4_dequant_gemv_odd_cols_gfx1201_20260528.jsonl` |
| `shared-memory-tiled-transpose` | 1009x2039 fp32 transpose | 0.051850 | 0.006700 | 7.738806x | `data/records/shared_memory_tiled_transpose_awkward_gfx1201_20260528.jsonl` |

## Partial Or Blocked Runs

- `rocwmma-mfma-gemm`: scalar HIP baseline passed for 256x256x256 on gfx1201,
  but the optimized rocWMMA path is `build-blocked` in this venv because
  `rocwmma/rocwmma.hpp` is unavailable. No Matrix Core speedup is claimed.
- `rocwmma-mfma-gemm:hipblaslt-hgemm`: the hipBLASLt library baseline is
  `build-blocked` in this venv because `hipblasLt.h` is unavailable.

## First Measurement Targets

- `rocwmma-mfma-gemm` with both `tools/run_hip_task.py` and `tools/run_library_baseline.py` for hipBLASLt comparison.

The INT4 GEMV result is a seed-baseline comparison only; attach a matching
quantized library, inference-engine, Triton, or framework baseline before making
broader INT4 performance claims.

Record HIP-event timing as `timing-only`. Promote a result to `counter-backed-measured` only after attaching rocprofiler/rocprof counter artifacts plus hardware and build metadata.

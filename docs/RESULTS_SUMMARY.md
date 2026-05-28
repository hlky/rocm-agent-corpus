# Results Summary

No ROCm measurements are checked in yet.

This corpus was created from the CUDA agent corpus on 2026-05-20. The CUDA-origin timing result files and JSONL records were removed during the ROCm port so agents do not confuse NVIDIA measurements with HIP/ROCm evidence.

## First Measurement Targets

- `memory-coalesced-matrix-copy` and `shared-memory-tiled-transpose` with `tools/run_matrix_task.py`.
- `vectorized-saxpy`, `block-reduction-sum`, `rowwise-softmax`, `block-topk-sampling`, and `fused-int4-dequant-gemv` with `tools/run_hip_task.py`.
- `rocwmma-mfma-gemm` with both `tools/run_hip_task.py` and `tools/run_library_baseline.py` for hipBLASLt comparison.

Record HIP-event timing as `timing-only`. Promote a result to `counter-backed-measured` only after attaching rocprofiler/rocprof counter artifacts plus hardware and build metadata.

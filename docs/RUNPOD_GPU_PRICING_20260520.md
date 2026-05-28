# Runpod GPU Pricing Snapshot - 2026-05-20

This file is retained only as historical context from the CUDA source corpus.
The original snapshot listed NVIDIA GPUs and CUDA compute capabilities, so it is
not a valid ROCm hardware selection guide.

## ROCm Use Policy

- Refresh Runpod availability and prices before creating a pod.
- Prefer AMD GPUs with a ROCm-supported `gfx` target reported by `rocminfo`.
- Record the exact GPU model, ROCm version, driver/runtime metadata, and `gfx`
  target in every measurement.
- Do not treat CUDA compute capability, NVIDIA GPU names, or source-corpus
  Runpod timings as ROCm evidence.

## Refresh Checklist

1. Query the current Runpod catalog.
2. Filter for AMD GPUs with ROCm support for the required task.
3. Verify the pod image includes `hipcc`, `rocminfo`, `rocm-smi`, and
   rocprofiler/rocprof tooling.
4. Run `python tools/collect_env.py` before benchmarking.
5. Save the refreshed price/availability note as a new dated file rather than
   editing this historical snapshot.

For operational setup, read `docs/RUNPOD_GPU_PASS.md`.

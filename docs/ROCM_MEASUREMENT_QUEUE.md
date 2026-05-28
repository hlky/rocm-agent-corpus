# ROCm Measurement Queue

This queue is the first evidence work needed after task parity. In this Windows
workspace, `H:\dinoml_v2\.venv\rocm` provides `hipcc` and PyTorch ROCm for
timing-only runs on AMD Radeon RX 9070 XT (`gfx1201`). `rocprof`, `rocprofv3`,
`rocminfo`, and `rocm-smi` are still unavailable here, so profiler-counter
evidence is not claimed from this environment.

## Environment Capture

Run this first on the AMD/ROCm host and keep the output with the result files:

```bash
python tools/collect_env.py
python tools/inspect_rocm_arch.py
```

Record the exact GPU name, reported `gfx` target, ROCm version, `hipcc` version,
driver/runtime details, and compile flags for every result.

The current Windows venv hardware record is
`data/hardware/rx9070xt_gfx1201_rocm721_windows_20260528.json`.

## First Timing-Only Runs

These commands produce useful HIP-event `timing-only` records when run with
`--write-result` on ROCm hardware. Replace `<gfx-target>` with the exact target
reported by the environment and attach a matching hardware record id when
available.

```bash
python tools/run_hip_task.py memory-coalesced-matrix-copy baseline --arch <gfx-target> --write-result
python tools/run_hip_task.py memory-coalesced-matrix-copy optimized --arch <gfx-target> --write-result

python tools/run_hip_task.py block-reduction-sum baseline --arch <gfx-target> --write-result
python tools/run_hip_task.py block-reduction-sum optimized --arch <gfx-target> --write-result

python tools/run_hip_task.py rowwise-softmax baseline --arch <gfx-target> --write-result
python tools/run_hip_task.py rowwise-softmax optimized --arch <gfx-target> --write-result

python tools/run_hip_task.py block-topk-sampling baseline --arch <gfx-target> --write-result
python tools/run_hip_task.py block-topk-sampling optimized --arch <gfx-target> --write-result

python tools/run_hip_task.py rocwmma-mfma-gemm baseline --arch <gfx-target> --write-result
python tools/run_hip_task.py rocwmma-mfma-gemm optimized --arch <gfx-target> --write-result
```

Completed first timing-only gfx1201 records:

- `memory-coalesced-matrix-copy`: 13.948364x optimized over baseline for
  4096x4096 fp32 copy.
- `block-reduction-sum`: 16.825280x optimized over baseline for a
  16777216-element fp32 sum.
- `rowwise-softmax`: 7.958159x optimized over baseline for 4096x1024 fp32
  softmax.
- `block-topk-sampling`: 14.049390x optimized over baseline for 1024x32768
  k=4 temperature=0.8 top-k sampling.

Blocked in the current Windows venv:

- `rocwmma-mfma-gemm`: scalar HIP baseline passed for 256x256x256, but the
  optimized rocWMMA path failed to build because `rocwmma/rocwmma.hpp` was not
  available. Install or expose rocWMMA headers/libraries before claiming custom
  Matrix Core timing.

## First Library Baseline

Run the library baseline for the same GEMM shape before claiming a custom
rocWMMA/MFMA win:

```bash
python tools/run_library_baseline.py rocwmma-mfma-gemm:hipblaslt-hgemm --arch <gfx-target> --write-result
```

## First Counter-Backed Promotion

After a timing-only result passes correctness, rerun one seed task under
rocprofiler/rocprof and attach the profiler summary artifact. The first
counter-backed promotion should be a simple memory or reduction task before a
more complex attention or GEMM claim.

Recommended order:

1. `memory-coalesced-matrix-copy`
2. `block-reduction-sum`
3. `rowwise-softmax`
4. `rocwmma-mfma-gemm`

If profiler collection fails, record `profile-attempted-blocked` with the exact
tool command, error text, ROCm version, and host permissions context.

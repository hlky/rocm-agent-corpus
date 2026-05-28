# ROCm Measurement Queue

This queue is the first evidence work needed after task parity. It is blocked in
this Windows workspace because `hipcc`, `rocminfo`, and `rocm-smi` are not
installed, so no ROCm timing or profiler evidence is claimed here.

## Environment Capture

Run this first on the AMD/ROCm host and keep the output with the result files:

```bash
python tools/collect_env.py
python tools/inspect_rocm_arch.py
```

Record the exact GPU name, reported `gfx` target, ROCm version, `hipcc` version,
driver/runtime details, and compile flags for every result.

## First Timing-Only Runs

These commands produce useful HIP-event `timing-only` records when run with
`--write-result` on ROCm hardware:

```bash
python tools/run_hip_task.py memory-coalesced-matrix-copy baseline --arch gfx942 --write-result
python tools/run_hip_task.py memory-coalesced-matrix-copy optimized --arch gfx942 --write-result

python tools/run_hip_task.py block-reduction-sum baseline --arch gfx942 --write-result
python tools/run_hip_task.py block-reduction-sum optimized --arch gfx942 --write-result

python tools/run_hip_task.py rowwise-softmax baseline --arch gfx942 --write-result
python tools/run_hip_task.py rowwise-softmax optimized --arch gfx942 --write-result

python tools/run_hip_task.py block-topk-sampling baseline --arch gfx942 --write-result
python tools/run_hip_task.py block-topk-sampling optimized --arch gfx942 --write-result

python tools/run_hip_task.py rocwmma-mfma-gemm baseline --arch gfx942 --write-result
python tools/run_hip_task.py rocwmma-mfma-gemm optimized --arch gfx942 --write-result
```

Replace `gfx942` with the exact target reported by `rocminfo`. Do not infer the
target from the marketing GPU name.

## First Library Baseline

Run the library baseline for the same GEMM shape before claiming a custom
rocWMMA/MFMA win:

```bash
python tools/run_library_baseline.py rocwmma-mfma-gemm:hipblaslt-hgemm --arch gfx942 --write-result
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

# ROCm Parity Scorecard - 2026-05-28

This scorecard defines what "as good as the CUDA corpus" currently means for
the ROCm corpus, without treating CUDA-origin measurements as ROCm evidence.

## Current Status

| Area | Status | Check |
| --- | --- | --- |
| Task coverage | parity achieved | `python tools/check_cuda_rocm_task_parity.py --source H:/cuda-agent-corpus` reports 85 CUDA tasks, 85 ROCm tasks, and 25 renamed equivalents |
| Repository publication | done | public GitHub repo, default branch `main`, README, description, Apache-2.0 license |
| Architecture guidance | ROCm-native scaffolded | `docs/GPU_GFX_ARCHITECTURE_GUIDE.md`, `docs/ARCHITECTURE_LABS.md`, and `architecture/gfx*/README.md` use `gfx` targets, `hipcc`, rocprofiler/rocprof, and AMD GCN ISA evidence rules |
| Runtime/library equivalents | scaffolded | HIP Graphs, hipRTC, RCCL, rocSHMEM, MIOpen, MIGraphX, hipBLASLt/rocBLAS, Composable Kernel, rocPRIM/hipCUB/rocThrust |
| Harness equivalents | scaffolded | HIP harnesses exist for the CUDA seed harness families, but most have not been run on AMD hardware in this repo |
| Evidence parity | started | `tools/summarize_results.py` reports four `timing-only` ROCm optimization records; counter-backed evidence is still pending |

## Gate Commands

Run these before declaring a parity pass ready:

```bash
python tools/check_cuda_rocm_task_parity.py --source H:/cuda-agent-corpus
python tools/validate_corpus.py
python tools/summarize_results.py
```

Use `timing-only` for HIP-event timing without rocprofiler/rocprof counters, and
use `negative example` when a custom HIP optimization loses.

## Next Highest-Value Parity Work

1. Install or expose rocWMMA, hipBLASLt, and Composable Kernel dependencies,
   then rerun `rocwmma-mfma-gemm` optimized and library baselines.
   Add more same-hardware ROCm records for additional shape sweeps.
   `memory-coalesced-matrix-copy`, `block-reduction-sum`,
   `rowwise-softmax`, and `block-topk-sampling` now have first
   `timing-only` gfx1201 records.
2. Add library baseline records for the same shapes:
   hipCUB/rocPRIM, hipBLASLt/rocBLAS, Composable Kernel, MIOpen, and Triton
   where applicable.
3. Promote at least one counter-backed record with rocprofiler/rocprof artifacts.
4. Keep CUDA-origin Top-K, PyTorch, and architecture notes as algorithm context
   only until ROCm runs attach AMD hardware, ROCm version, compiler flags, gfx
   target, timing, correctness, and evidence labels.

See `docs/ROCM_MEASUREMENT_QUEUE.md` for the first command queue to run on an
AMD/ROCm host.

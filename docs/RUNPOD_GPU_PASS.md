# Runpod GPU Pass

Use this when the corpus needs real ROCm measurements and the local machine does
not have the ROCm toolkit, rocprofiler/rocprof, or usable GPU access.

## When to Use It

Use a remote GPU only after the local corpus structure validates and there is a
benchmark harness to run. A cheap pod is enough for seed tasks, but compilation
still benefits from a reasonable CPU allocation.

Good first targets:

- Low-cost RTX 30/40-series or L4-class GPUs when available below the requested
  budget.
- At least 4 vCPUs if compiling Composable Kernel-like kernels.
- 20 GB workspace volume for this scaffold; use more for cloned benchmark repos.

## Workflow

1. Query current Runpod prices and choose a stocked Secure Cloud GPU.
2. Provision a pod with the `runpod-codex-remote` skill.
3. Copy or clone this project into `/workspace/rocm-agent-corpus`.
4. Verify ROCm tools:

```bash
hipcc --version
rocprof --version
rocm-smi
```

5. Run seed benchmarks:

```bash
python tools/validate_corpus.py
python tools/run_matrix_task.py memory-coalesced-matrix-copy baseline --rows 4096 --cols 4096 --write-result
python tools/run_matrix_task.py memory-coalesced-matrix-copy optimized --rows 4096 --cols 4096 --write-result
python tools/run_matrix_task.py shared-memory-tiled-transpose baseline --rows 4096 --cols 4096 --write-result
python tools/run_matrix_task.py shared-memory-tiled-transpose optimized --rows 4096 --cols 4096 --write-result
```

6. Add profiler summaries:

```bash
rocprof --section MemoryWorkloadAnalysis_Tables --print-details=all \
  ./out/memory-coalesced-matrix-copy/baseline/benchmark 4096 4096 5 20
```

Save concise text summaries under each task's `profiles/` directory. Raw
`.rocprof-rep` files are ignored by default because they are often large.

## Budget Discipline

Keep the first pass short:

- Build once per variant.
- Use one large square shape and one awkward rectangular shape.
- Capture `rocprof` only after correctness and timing pass.
- Stop the pod immediately after copying results back.


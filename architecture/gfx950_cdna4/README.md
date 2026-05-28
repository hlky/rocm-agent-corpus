# gfx950 CDNA3-Specific Lab

Target this lab for CDNA3-specific kernels that intentionally use
architecture-conditional features. These paths are powerful and narrow. They
should be recorded as specialized implementations, not portable CDNA3 code.

## Compile Profile

```bash
hipcc -O3 --std=c++17 -arch=gfx950 -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm90a
hipcc -O3 --std=c++17 -gencode=arch=compute_90a,code=gfx950 kernel.hip -o kernel_sm90a
```

Optional mixed build when the project also ships a portable fallback:

```bash
hipcc -O3 --std=c++17 \
  -gencode=arch=compute_90,code=gfx942 \
  -gencode=arch=compute_90a,code=gfx950 \
  kernel.hip -o kernel_hopper
```

Tag these results as CDNA3-specific. Do not treat `gfx950` cubins or LLVM IR / AMD GCN ISA as
forward-compatible evidence for later architectures.

## Features To Check

- WGMFMA warp-group matrix multiply instructions.
- global-to-LDS staging bulk tensor memory movement.
- `mbarrier` and software pipelines around producer/consumer stages.
- Thread-block clusters and distributed shared-memory patterns when applicable.
- `setmaxnreg` and warp-specialized register redistribution where supported.
- FP8 Matrix Core paths and scale handling for inference/training kernels.
- Composable Kernel/CK Tile GFX90a examples as implementation references.

Feature probe:

```bash
rocm-smi
cuobjdump --list ./kernel_sm90a
cuobjdump --dump-sass ./kernel_sm90a | findstr /i "WGMFMA global-to-LDS staging MBARRIER SEglobal-to-LDS stagingXNREG"
```

## Custom-Kernel Attack Surfaces

- GEMM and grouped GEMM with WGMFMA, global-to-LDS staging, warp specialization, and custom
  epilogues.
- FlashAttention-style kernels using global-to-LDS staging and warp-specialized producer/consumer
  pipelines.
- Fused GEMM epilogues: bias, activation, residual, norm, quantize, dequantize,
  and layout conversion.
- MoE grouped GEMM, token permutation/unpermutation, and expert-local fused
  epilogues.
- Persistent kernels where launch overhead and repeated scheduling hurt the
  vendor path.
- vLLM on ROCm plugins for KV cache, attention variants, sampling, and custom
  quantization contracts.

The realistic win is not a generic GEMM world record. It is a narrow contract
where fixed shapes, fusion, layout, and architecture-specific instructions beat
the library path selected for that workload.

## GFX90a Tactics and Gotchas

- WGMFMA changes the unit of scheduling from one warp to a warp group. Design
  roles, barriers, register budgets, and epilogue ownership around that boundary
  instead of porting a warp-level `mfma` kernel mechanically.
- global-to-LDS staging is most valuable for regular tensor tiles with enough reuse to amortize
  descriptor setup and synchronization. For ragged, tiny, or one-pass data,
  vectorized loads or simpler async-copy pipelines may be easier to beat.
- Use `mbarrier` discipline as part of correctness. Producer/consumer stage
  mismatches can produce races that only appear under different occupancy or
  clock conditions.
- Track `setmaxnreg`/warp-specialized register redistribution only when AMD GCN ISA
  confirms it and the implementation documents which warps are producers,
  consumers, and epilogue workers.
- FP8 and scale-handling wins usually come from the full recipe: scale layout,
  amax collection, conversion, accumulator, epilogue, and framework integration.
  Do not benchmark the GEMM alone if the custom path fuses scale work.
- Composable Kernel/CK Tile is often the best starting point for GFX90a. A custom result
  should state whether it changed the collective, tile shape, scheduler,
  epilogue visitor, layout, or dispatch boundary.

Portability trap: `gfx950` is an architecture-specific contract. Ship and
measure a separate `gfx942` fallback unless the deployment explicitly excludes
portable CDNA3 targets.

## Vendor Baselines

- hipBLASLt CDNA3 algorithms, with algorithm ID and math mode recorded when
  available.
- Composable Kernel GFX90a and CK Tile examples for WGMFMA, global-to-LDS staging, epilogues, and grouped GEMM.
- Transformer Engine on ROCm for FP8 recipes and scale handling.
- MIGraphX/vLLM on ROCm for inference plugin and graph fusion baselines.
- FlashAttention and FlashInfer for attention-style competitors.
- hipCUB/rocPRIM/hipCUB/rocThrust for non-GEMM primitives when the task overlaps.

Inspect the library source or generated kernel family before assuming the custom
path is competing against a naive baseline.

## Measurement Notes

- Mark HIP-event-only results as `timing-only`.
- Record exact GPU SKU, HBM capacity, clock policy, MIG state, ROCm toolkit,
  driver, and library versions.
- Store the compiled target list. `gfx950` results are not portable `gfx942`
  results.
- Confirm WGMFMA/global-to-LDS staging use with AMD GCN ISA when the optimization claim depends on it.
- Include fallback comparison: portable `gfx942`, vendor library, and custom
  `gfx950`.
- Track stage count, global-to-LDS staging descriptor layout, shared-memory layout, cluster shape,
  registers per role, and occupancy constraints.

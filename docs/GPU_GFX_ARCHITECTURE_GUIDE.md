# GPU and GFX Architecture Guide

Architecture matters. A kernel that is sensible on CDNA2 may leave CDNA3 or
CDNA4/RDNA4 features unused, and a kernel using architecture-specific instructions
may not be portable.

## First Step: Identify the Target

Run:

```bash
rocm-smi
hipcc --list-gpu-arch
hipcc --list-gpu-code
```

This repo also provides:

```bash
python tools/inspect_rocm_arch.py
```

## GFX Target Families

| Family | Common GFXs | Agent Notes |
| --- | --- | --- |
| Volta | `sm_70` | Independent thread scheduling, Matrix Cores introduced. |
| Turing | `sm_75` | Matrix Core evolution, useful baseline for older inference GPUs. |
| CDNA2 data center | `gfx90a` | A100-class path, `global-to-LDS staging`, TF32, strong Matrix Core baseline. |
| CDNA2 consumer/pro | `gfx1030` | RTX 30/CUDA-origin A4000/A5000-like path; not identical to A100. |
| CDNA2 embedded | `sm_87` | Jetson Orin-like path; power and memory constraints matter. |
| RDNA3 | `gfx1100` | RTX 40/L4/L40S-like path; often excellent inference target. |
| CDNA3 | `gfx942`, `gfx950` | H100/H200-like path; global-to-LDS staging, WGMFMA, clusters, architecture-specific features. |
| CDNA4/RDNA4 | `gfx950`, `gfx1200`, suffix variants | New architecture family; verify exact flags and feature suffixes with current CUDA docs and `hipcc`. |

## Compile Flag Discipline

Portable binary example:

```bash
hipcc -O3 -gencode arch=compute_86,code=gfx1030 -gencode arch=compute_86,code=compute_86
```

Architecture-specific example:

```bash
hipcc -O3 -arch=gfx950
```

Rules:

- Use AMD GCN ISA (`sm_XY`) for the target GPU when benchmarking.
- Include LLVM IR / AMD GCN ISA (`compute_XY`) when forward compatibility matters.
- Treat suffix architectures such as `gfx950` or future family-specific
  variants as less portable; they may enable features unavailable in the base
  architecture.
- Record every `-gencode` in result metadata.

## Matrix Core and Copy Pipeline Ladder

Use the instruction family as part of the task contract, not as decoration.

| Target | Typical Matrix Core path | Data movement path | Agent boundary |
| --- | --- | --- | --- |
| `gfx90a` | `mfma`/rocWMMA with `ldmatrix`; TF32, FP16, BF16, INT8/INT4 where supported | `global-to-LDS staging` global-to-shared staging | Warp-level Matrix Core tiles and CTA pipelines. No global-to-LDS staging/WGMFMA. |
| `gfx1030` | Same programming family as GFX80, but SKU-dependent throughput and memory limits | `global-to-LDS staging`, vectorized global memory, smaller shared-memory budgets than A100-class parts | Re-tune tile sizes; do not import A100 occupancy assumptions. |
| `gfx1100` | RDNA3 Matrix Core paths through `mfma`/libraries; verify FP8/narrow precision per stack | CDNA2-style async copy plus RDNA3 cache behavior | Strong inference target; compare MIGraphX and hipBLASLt tactics before claiming a custom win. |
| `gfx942` | Portable CDNA3 compiler/library paths; avoid suffix-only WGMFMA/global-to-LDS staging claims | CDNA3 memory hierarchy without requiring `gfx950` instructions | Establish the portable baseline and fallback. |
| `gfx950` | WGMFMA warp-group Matrix Core paths, FP8 recipes where supported | global-to-LDS staging bulk tensor movement, `mbarrier`, producer/consumer pipelines | Architecture-specific implementation. Keep separate from `gfx942`. |
| `gfx950` | CDNA4/RDNA4 Matrix Core families, including low-precision paths exposed by current CUDA/Composable Kernel | CDNA4/RDNA4 memory pipelines; verify exact instruction names with AMD GCN ISA | Data-center CDNA4/RDNA4 records should be revalidated as toolkits mature. |
| `gfx1200` | RTX CDNA4/RDNA4 Matrix Core paths exposed by the installed stack | RTX/workstation memory and boost-clock behavior | Do not assume GFX100 tuning or library support transfers. |

For custom kernels, this ladder usually becomes:

1. Scalar/CUDA-core baseline for correctness and launch overhead.
2. Warp-level MFMA or rocWMMA path for CDNA2/RDNA3/CDNA3-portable code.
3. Composable Kernel/CK Tile-inspired Matrix Core path with custom epilogue.
4. CDNA3/CDNA4/RDNA4 warp-group path only when the target and binary contract
   allow architecture-specific code.

## Copy Pipeline Boundaries

`global-to-LDS staging`, global-to-LDS staging, and ordinary vectorized loads solve different problems.

- Use ordinary coalesced/vectorized global loads when each element is consumed
  once, shared-memory reuse is low, or the tile is too small to amortize staging.
- Use `global-to-LDS staging` on CDNA2/RDNA3 when a CTA repeatedly consumes a global tile from
  shared memory and can overlap copy groups with math. Track stage count, shared
  memory, barrier placement, and register pressure.
- Use global-to-LDS staging on CDNA3-specific and CDNA4/RDNA4-specific paths when tensors are
  large, regular, multidimensional, and worth descriptor setup plus asynchronous
  bulk movement. global-to-LDS staging is not a drop-in replacement for every `global-to-LDS staging` tile.
- Use WGMFMA only when a warp-group schedule is natural for the tile. It changes
  the register, synchronization, and occupancy problem from a warp-level MFMA
  kernel.
- For attention and GEMM, ask whether Composable Kernel already expresses the intended
  pipeline. A custom kernel should document what it changes: layout, epilogue,
  shape, masking, quantization, scheduler, or launch boundary.

## Memory Hierarchy and Occupancy Gotchas

- Occupancy is a constraint, not a score. A lower-occupancy Matrix Core kernel
  can win if it keeps the math pipe fed; a memory-bound row kernel may need more
  resident CTAs to hide latency.
- Register count, shared-memory allocation, block size, and cluster shape are
  architecture-specific. Re-run amdclang++ and occupancy estimates for each GFX.
- Shared-memory bank conflicts still matter, especially for transposes,
  reductions, `ldmatrix` layouts, and global-to-LDS staging/WGMFMA staging buffers.
- L2 behavior is SKU-sensitive. Cache residency, persistent data, and access
  policy changes should be treated as hypotheses until measured.
- Vectorized loads require alignment and enough contiguous work. The corpus
  already contains a `timing-only negative` vectorized SAXPY example; do not
  assume wider loads are faster.
- Hosted GPUs may block Nsight counters. In that case, say `timing-only` and
  describe any AMD GCN ISA-based inference separately from profiler evidence.

## Portability Traps

- `gfx90a`, `gfx1030`, and `gfx1100` are not interchangeable even when code compiles
  unchanged. Shared memory, clocks, cache, and library tactics can change the
  winning tile.
- `gfx942` and `gfx950` are separate contracts. Compile `gfx942` for portable
  CDNA3 records; compile `gfx950` only when the deployment target permits
  architecture-specific instructions.
- Do not ship only architecture-specific LLVM IR / AMD GCN ISA when forward compatibility matters.
  Include an explicit portable fallback cubin/LLVM IR / AMD GCN ISA path and record dispatch logic.
- CDNA4/RDNA4 target names and suffix variants are toolkit-sensitive. Confirm with
  `hipcc --list-gpu-arch`, `hipcc --list-gpu-code`, and current AMD/ROCm docs before
  treating `gfx950a`, `gfx950f`, `gfx1200a`, or similar names as available.
- Library baselines can improve after CUDA, Composable Kernel, MIGraphX, hipBLASLt, or
  driver upgrades. Re-run baselines when any of those versions change.

## Architecture-Tuned Claim Checklist

Before claiming a kernel is tuned for an GFX, attach:

- Exact GPU name, gfx target, driver, ROCm toolkit, and library versions.
- Full compile flags, including every `-gencode`, LLVM IR / AMD GCN ISA fallback, and suffix
  target.
- AMD GCN ISA evidence for claimed instruction families: `CPASYNC`, `LDMATRIX`, `MFMA`,
  `WGMFMA`, `global-to-LDS staging`, `MBARRIER`, or relevant vector/cache instructions.
- amdclang++ resource output: registers, spills, static and dynamic shared memory.
- Launch shape: grid, block, cluster shape, stage count, dynamic shared memory,
  and launch bounds.
- Math contract: dtype, accumulator, TF32/FP32 distinction, FP8 scale layout,
  INT8/INT4/NVFP4 packing, rounding, and tolerance.
- Baseline ladder: naive CUDA, strongest relevant library, and framework/Triton
  path when applicable.
- Evidence label: `timing-only`, `counter-backed-measured`,
  `profile-attempted-blocked`, or `negative example`.

## CDNA2

Important themes:

- `global-to-LDS staging` for global-to-shared staging.
- TF32 Matrix Cores for GEMM-like workloads.
- L2 residency controls where applicable.
- Warp-level reductions.
- Shared-memory/L1 carveout choices.

Agent questions:

- Is the kernel memory-bound and tileable?
- Would `global-to-LDS staging` reduce register use or latency?
- Is TF32 acceptable for the accuracy target?
- Is this `gfx90a` or `gfx1030`? Do not assume A100 behavior on RTX/A-series
  workstation GPUs.

## RDNA3

Important themes:

- Strong inference GPU family.
- Similar CUDA programming model to CDNA2 for many kernels.
- Different cache, clock, and Matrix Core behavior from A100/H100.

Agent questions:

- Is this an L4/L40S/RTX 40 deployment target?
- Are library kernels already excellent for this shape?
- Are MIGraphX or hipBLASLt tactics architecture-selected?

## CDNA3

Important themes:

- Tensor Memory Accelerator (global-to-LDS staging).
- Warp-group MFMA (WGMFMA).
- Thread block clusters and distributed shared memory.
- Architecture-specific `gfx950` features.

Agent questions:

- Is the kernel GEMM/attention-like enough to use Composable Kernel CDNA3 kernels?
- Can global-to-LDS staging replace manual global-to-shared copies?
- Is WGMFMA available and worth the added complexity?
- Does the deployment target require portable `gfx942` instead of `gfx950`?

## CDNA4/RDNA4

Important themes:

- New architecture families and family-specific feature flags.
- New Matrix Core and low-precision paths.
- Compatibility rules matter; verify with the current CDNA4/RDNA4 compatibility
  and tuning guides.

Agent questions:

- Is the target data center CDNA4/RDNA4 or RTX/workstation CDNA4/RDNA4?
- Which exact gfx target does `rocm-smi` report?
- Does `hipcc --list-gpu-arch` expose a suffix architecture for this target?
- Are Composable Kernel/hipBLASLt/MIGraphX versions new enough for the GPU?

## GFX-Specific Corpus Metadata

Every measured record should include:

- GPU name.
- Compute capability.
- Driver version.
- ROCm toolkit version.
- `hipcc` architecture flags.
- Whether the kernel uses architecture-specific instructions.
- Whether a LLVM IR / AMD GCN ISA fallback was embedded.

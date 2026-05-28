# GPU and GFX Architecture Guide

Architecture matters. A HIP kernel that is sensible on one AMD target can be
the wrong shape for another because wave size, LDS capacity, Matrix Core paths,
cache behavior, clocks, and library coverage all move.

## First Step: Identify the Target

Run:

```bash
rocminfo
rocm-smi
hipcc --version
hipcc --list-gpu-targets
```

This repo also provides:

```bash
python tools/inspect_rocm_arch.py
```

Record the reported `gfx` target from `rocminfo`; do not infer it from a GPU
marketing name.

## GFX Target Families

| Family | Common GFX Targets | Agent Notes |
| --- | --- | --- |
| CDNA2 | `gfx90a` | MI200-class data-center baseline for MFMA, LDS tiling, hipBLASLt, and Composable Kernel comparisons. |
| RDNA2 | `gfx1030` | Use as a memory, launch-overhead, and portability target only when the local ROCm stack supports it. |
| RDNA3 | `gfx1100` | Good fixed-shape inference and memory-bound fusion target; verify WMMA/MFMA and low-precision support per SKU. |
| CDNA3 | `gfx942` | MI300-class target for MFMA, CK Tile, attention, grouped GEMM, and inference baselines. |
| CDNA4 | `gfx950` | Data-center target; verify exact ROCm/library support and rerun baselines after upgrades. |
| RDNA4 | `gfx1200` | Workstation/deployment target; keep records separate from CDNA4. |

Use `data/index/gpu_architectures.json` and `docs/ARCHITECTURE_LABS.md` for
machine-readable and task-oriented navigation.

## Compile Flag Discipline

Compile measured kernels with an explicit offload target:

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx942 -lineinfo kernel.hip -o kernel_gfx942
```

Rules:

- Record every offload target in result metadata.
- Keep each `gfx` target as a separate compatibility contract.
- If the toolkit exposes suffix or family-specific targets, record them as
  target-specific evidence and provide a portable fallback when deployment needs
  one.
- Store `hipcc --version`, `hipcc --list-gpu-targets`, ROCm runtime/driver
  metadata, GPU model, and library versions.

## Matrix Core And LDS Ladder

Use instruction families as claims only when you have proof.

| Target | Matrix/Compute Path | Data Movement Path | Agent Boundary |
| --- | --- | --- | --- |
| `gfx90a` | MFMA through HIP, rocWMMA, hipBLASLt, or Composable Kernel | LDS tiling, barriers, wait counts, vectorized global memory | CDNA2 data-center baselines and fixed-shape specialization. |
| `gfx1030` | SKU/toolkit-dependent; verify before claiming | Vectorized memory, LDS tiling, wave-size-sensitive code | Memory-bound and launch-overhead tasks. |
| `gfx1100` | WMMA/MFMA availability must be verified | LDS/cache behavior plus inference-oriented library paths | Fixed-shape inference and fused memory-bound kernels. |
| `gfx942` | MFMA and CK Tile/CDNA3 library paths | LDS staging and HBM/cache-aware tiling | Portable CDNA3 records. |
| `gfx950` | Exact MFMA/WMMA and low-precision support depends on toolkit/library stack | LDS staging, multi-GPU topology, library maturity | CDNA4 data-center records, rerun after upgrades. |
| `gfx1200` | Exact WMMA/MFMA and low-precision support depends on SKU/toolkit | Workstation clocks, cache behavior, inference libraries | RDNA4 deployment records. |

Disassembly command:

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx942
```

Look for `v_mfma`, `v_wmma`, `ds_read`, `ds_write`, `s_waitcnt`,
`s_barrier`, `global_load`, and `global_store` only when they support a claim.

## Copy And Staging Boundaries

- Use ordinary coalesced/vectorized global loads when each element is consumed
  once or the tile is too small to amortize LDS staging.
- Use LDS staging when a workgroup repeatedly consumes a global tile and can
  overlap movement, synchronization, and compute.
- Use Composable Kernel/CK Tile as a reference before writing a raw MFMA
  pipeline. A custom kernel should state what it changes: tile shape, layout,
  scheduler, epilogue, quantization metadata, or dispatch boundary.
- Do not use NVIDIA-specific terms such as TMA, WGMMA, SASS, or cubin as ROCm
  evidence. CUDA-origin material is historical context only.

## Portability Traps

- `gfx90a`, `gfx1030`, `gfx1100`, `gfx942`, `gfx950`, and `gfx1200` are not
  interchangeable just because HIP source compiles.
- A result on one GFX target is not architecture-general truth.
- Library baselines can change after ROCm, Composable Kernel, MIGraphX,
  hipBLASLt, Triton, PyTorch, or driver upgrades.
- Consumer/workstation results need clock, power, and thermal metadata.
- HIP-event timings without profiler counters are `timing-only`.

## Architecture-Tuned Claim Checklist

Attach:

- GPU name and reported `gfx` target.
- ROCm toolkit, runtime/driver, hipcc, and library versions.
- Full compile flags and offload targets.
- Correctness contract: shape, dtype, layout, math mode, accumulator, tolerance.
- Vendor baseline: hipBLAS/rocBLAS/hipBLASLt, Composable Kernel, hipCUB,
  rocPRIM, rocThrust, MIOpen, MIGraphX, vLLM on ROCm, Triton, or framework code.
- Resource metadata: VGPRs, SGPRs, LDS bytes, spills, block size, grid size,
  wave-size assumptions, stage count, and launch bounds when available.
- Disassembly or library metadata for any instruction-family claim.
- Evidence label: `template-only`, `timing-only`, `counter-backed-measured`,
  `profile-attempted-blocked`, `negative example`, or `correctness-only`.

## Measurement Metadata

Every measured record should include:

- GPU name and `gfx` target.
- ROCm toolkit and runtime/driver version.
- `hipcc` command line and offload target.
- Library versions and selected algorithm or generated-kernel identity.
- Hardware state: clocks, power, topology for multi-GPU work.
- Timing boundary: isolated kernel, library call, framework call, graph replay,
  or end-to-end application path.

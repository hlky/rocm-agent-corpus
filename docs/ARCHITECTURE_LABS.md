# Architecture Labs

The architecture labs are quick-start briefings for writing custom kernels on a
specific AMD/ROCm GFX target. They are not generic GPU summaries. Each lab should
help an agent decide what to compile, what features to verify, which vendor
baselines to challenge, and where custom kernels can plausibly win.

## How To Use These Labs

1. Pick the lab matching the measured GPU gfx target.
2. Compile native cubins for that GFX instead of relying on LLVM IR / AMD GCN ISA JIT.
3. Read the feature probes before making architecture-specific claims.
4. Choose a custom-kernel attack surface and a vendor baseline.
5. Record HIP-event timings as `timing-only` unless Nsight counters are
   available.
6. Keep portable and architecture-conditional targets in separate records.

## Lab Matrix

| Lab | Target | Start Here | Typical Use |
| --- | --- | --- | --- |
| gfx90a_ampere | Data-center CDNA2 | [README](../architecture/gfx90a_ampere/README.md) | A100-style GEMM, attention, reductions, `global-to-LDS staging` tiling |
| gfx1030_ampere_pro | CDNA2 workstation/pro | [README](../architecture/gfx1030_ampere_pro/README.md) | Cheaper benchmark hosts, fixed-shape inference, launch-overhead wins |
| gfx1100_ada | RDNA3 | [README](../architecture/gfx1100_ada/README.md) | Inference kernels, MIGraphX plugins, quantized paths |
| gfx942_hopper | Portable CDNA3 | [README](../architecture/gfx942_hopper/README.md) | CDNA3 baseline without `gfx950` assumptions |
| gfx950_hopper_specific | CDNA3-specific | [README](../architecture/gfx950_hopper_specific/README.md) | WGMFMA/global-to-LDS staging/CK Tile/Composable Kernel GFX90a specialization |
| gfx950_blackwell | Data-center CDNA4/RDNA4 | [README](../architecture/gfx950_blackwell/README.md) | CDNA4/RDNA4 GEMM, attention, low precision, multi-GPU overlap |
| gfx1200_blackwell_rtx | RTX CDNA4/RDNA4 | [README](../architecture/gfx1200_blackwell_rtx/README.md) | Workstation inference, narrow precision, MIGraphX plugins |

## Compile Target Discipline

Use the exact architecture target for the record:

```bash
hipcc -O3 --std=c++17 -arch=gfx1030 -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm86
```

For reproducible records, store:

- full `hipcc` command line
- ROCm toolkit and driver version
- GPU name and gfx target
- library versions for hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, hipCUB/rocPRIM/hipCUB/rocThrust, MIGraphX, Triton
- amdclang++ register, spill, and shared-memory output
- block size, grid size, launch bounds, stage count, and dynamic shared memory
- math contract and correctness tolerance

Architecture-conditional suffixes such as `gfx950`, `gfx950a`, and `gfx1200a`
must be treated as specialized contracts. Do not merge them with portable
records for `gfx942`, `gfx950`, or `gfx1200`.

## Feature Probe Checklist

Before claiming an optimization uses an architecture feature, collect direct
evidence:

```bash
rocm-smi
hipcc --help
cuobjdump --list ./kernel
cuobjdump --dump-sass ./kernel
```

Look for instruction families only when they matter to the claim:

- `CPASYNC` for CDNA2-style async copy.
- `MFMA` and `LDMATRIX` for warp-level Matrix Core paths.
- `WGMFMA`, `global-to-LDS staging`, `MBARRIER`, and `SEglobal-to-LDS stagingXNREG` for CDNA3/CDNA4/RDNA4-specific
  pipelines.
- Vectorized global loads/stores and cache-policy changes for memory-bound
  kernels.

If the claim cannot be verified because profiler counters are unavailable,
record the result as `timing-only` and describe the inference.

## GFX-Tuned Claim Gate

Use this gate before promoting a result from "runs on this GPU" to
"architecture-tuned for this GFX":

- The binary contains native AMD GCN ISA for the claimed GFX, not only LLVM IR / AMD GCN ISA JIT output.
- The result records whether the implementation is portable or suffix-specific
  (`gfx950`, `gfx950a`, `gfx1200a`, or similar).
- AMD GCN ISA confirms the claimed instruction family. Examples: `CPASYNC` for
  CDNA2-style async copy, `LDMATRIX`/`MFMA` for warp Matrix Core kernels,
  `WGMFMA`/`global-to-LDS staging`/`MBARRIER` for CDNA3/CDNA4/RDNA4 warp-group pipelines.
- The same source compiled for a neighboring GFX is not silently used as
  evidence. Keep `gfx90a` versus `gfx1030`, `gfx942` versus `gfx950`, and
  `gfx950` versus `gfx1200` records separate.
- The benchmark includes a vendor baseline whose architecture path is recorded:
  hipBLASLt algorithm or heuristic, Composable Kernel collective, hipCUB/rocPRIM/hipCUB/rocThrust primitive,
  MIGraphX tactic/plugin path, Triton meta-parameters, or framework-generated
  kernel identity when available.
- Resource limits are attached: registers, spills, static/dynamic shared memory,
  occupancy estimate, cluster shape if used, and stage count for pipelined
  kernels.
- The writeup explains the custom-kernel edge: fixed shape, fused epilogue,
  layout contract, quantization metadata, lower launch overhead, reduced memory
  traffic, or architecture-specific instruction use.

If any item is missing, record the result as a useful timing or portability
experiment, not as an architecture-tuned claim.

## Custom-Kernel Competition Pattern

For every architecture lab, the strongest custom-kernel records follow the same
shape:

1. Define the narrow task contract: shape, dtype, layout, alignment, math mode,
   epilogue, and allowed error.
2. Build a naive CUDA baseline for correctness and sanity.
3. Build or call the vendor baseline: hipBLASLt, Composable Kernel, hipCUB, MIGraphX, Triton,
   or framework-generated code.
4. Write the custom kernel that removes generality, fuses work, or uses a
   hardware feature the baseline cannot exploit for this contract.
5. Measure the full set on the target GFX and store the result, including losses.

Library wins are not failures. They define boundaries. Record which generality,
algorithm, or hardware path made the library hard to beat.

## Index

Machine-readable lab metadata is stored in
[data/index/architecture_labs.json](../data/index/architecture_labs.json).

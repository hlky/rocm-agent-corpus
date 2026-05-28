# gfx942 CDNA3 Lab

Target this lab for portable CDNA3 builds that avoid architecture-conditional
suffix instructions. Use it when the kernel should run on gfx target 9.0
devices without requiring `gfx950` code paths.

## Compile Profile

```bash
hipcc -O3 --std=c++17 -arch=gfx942 -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm90
hipcc -O3 --std=c++17 -gencode=arch=compute_90,code=gfx942 kernel.hip -o kernel_sm90
```

Useful variants:

```bash
hipcc -O3 --use_fast_math -arch=gfx942 ...
hipcc -O3 -arch=gfx942 -Xamdclang++=-v,-warn-spills ...
hipcc -O3 -arch=gfx942 --resource-usage ...
```

Keep `gfx942` and `gfx950` results separate. Architecture-specific `a` targets
are not interchangeable with portable cubins.

## Features To Check

- CDNA3 memory hierarchy, occupancy behavior, and larger shared-memory options.
- Matrix Core modes available through portable compiler/library paths.
- Cluster-related APIs and barriers only if the compiled target and runtime
  support them without `gfx950` assumptions.
- Async-copy and pipeline behavior versus CDNA2, while avoiding WGMFMA/global-to-LDS staging
  claims unless the code is actually built for the `a` target.
- Compatibility behavior when embedding LLVM IR / AMD GCN ISA and cubins for mixed clusters.

Feature probe:

```bash
rocm-smi
cuobjdump --list ./kernel_sm90
cuobjdump --dump-sass ./kernel_sm90 | findstr /i "MFMA LDMATRIX"
```

## Custom-Kernel Attack Surfaces

- Portable CDNA3 specialization where the same source can also be compared
  against an `gfx950` implementation.
- Memory-bound transformer primitives: RMSNorm, LayerNorm, softmax, RoPE,
  KV-cache transforms, quant/dequant, and logits processing.
- Custom hipCUB competitors where larger shared-memory choices alter the best
  block size.
- GEMM-adjacent kernels that need CDNA3 scheduling and occupancy tuning but do
  not yet justify WGMFMA/global-to-LDS staging complexity.
- MIGraphX plugin kernels intended to remain portable across CDNA3 deployment
  environments.

Use this lab to establish the portable baseline before adding a separate
`gfx950` specialization.

## GFX90 Portable Tactics and Gotchas

- Use `gfx942` for the fallback path that should run across CDNA3-compatible
  deployments without architecture-specific suffix requirements. This is the
  right record for "portable CDNA3," not for peak WGMFMA/global-to-LDS staging specialization.
- Keep Matrix Core claims conservative. If the build is `gfx942`, do not use
  WGMFMA/global-to-LDS staging as the explanation unless the compiled binary and CUDA docs for the
  target prove those instructions are available in that contract.
- Re-tune memory-bound kernels from CDNA2. Larger or different shared-memory
  options can change the best block size for reductions, softmax, norm, and
  layout conversion, even without an GFX90a pipeline.
- Use the portable record to quantify the cost and benefit of the later GFX90a
  path: same shape, same math contract, same library versions, separate binary.
- For plugin work, decide whether deployment requires a portable cubin. A fast
  `gfx950` kernel is not a valid replacement if the engine must load on all
  CDNA3 targets in the fleet.
- Cluster APIs and cooperative launch behavior should be recorded explicitly:
  cluster shape, launch mode, shared-memory footprint, and fallback behavior.

Portability trap: combining `gfx942` and `gfx950` timings into one "CDNA3"
result hides the most important architecture boundary in this generation.

## Vendor Baselines

- hipBLAS/rocBLAS/hipBLASLt CDNA3 paths with math mode and algorithm recorded.
- Composable Kernel portable GFX90 kernels, kept distinct from GFX90a examples.
- hipCUB/rocPRIM/hipCUB/rocThrust for reductions, scans, select, sort, histogram.
- MIGraphX/vLLM on ROCm and Transformer Engine on ROCm for inference primitives.
- Triton or framework-generated CDNA3 kernels for generated-code comparison.

When a vendor path uses architecture-specific kernels internally, note that the
comparison may not be purely portable.

## Measurement Notes

- Mark HIP-event-only results as `timing-only`.
- Record whether the binary contains `gfx942`, `compute_90`, or architecture
  conditional targets.
- Do not cite WGMFMA or global-to-LDS staging as the reason for a result unless the AMD GCN ISA or build
  target proves it.
- Compare against `gfx950` only as a separate row with a separate contract.
- Capture launch bounds, registers, shared memory, block clusters if used, and
  whether the kernel needs cooperative launch semantics.

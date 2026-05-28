# Architecture Labs

The architecture labs are quick-start briefings for writing custom kernels on a
specific AMD/ROCm GFX target. They help an agent decide what to compile, what to
verify, which library baseline to challenge, and how to label evidence.

## How To Use These Labs

1. Pick the lab matching the measured GPU's reported `gfx` target.
2. Compile with an explicit `--offload-arch=<gfx>` target.
3. Record ROCm, driver/runtime, library, compiler, and GPU metadata.
4. Choose a vendor baseline before writing the custom kernel.
5. Mark HIP-event-only timing as `timing-only`.
6. Promote to `counter-backed-measured` only with attached rocprofiler/rocprof
   counter artifacts.

## Lab Matrix

| Lab | Target | Start Here | Typical Use |
| --- | --- | --- | --- |
| `gfx90a_cdna2` | CDNA2 | [README](../architecture/gfx90a_cdna2/README.md) | MI200-class GEMM, attention, reductions, LDS tiling |
| `gfx1030_rdna2` | RDNA2 | [README](../architecture/gfx1030_rdna2/README.md) | Memory-bound kernels, launch overhead, portability probes |
| `gfx1100_rdna3` | RDNA3 | [README](../architecture/gfx1100_rdna3/README.md) | Inference helpers, quantization, framework extension kernels |
| `gfx942_cdna3` | CDNA3 | [README](../architecture/gfx942_cdna3/README.md) | MI300-class GEMM, attention, grouped GEMM, framework integration |
| `gfx950_cdna4` | CDNA4 | [README](../architecture/gfx950_cdna4/README.md) | Low precision, grouped GEMM, attention, multi-GPU overlap |
| `gfx1200_rdna4` | RDNA4 | [README](../architecture/gfx1200_rdna4/README.md) | Workstation/deployment inference and fixed-shape fusion |

## Compile Target Discipline

Use explicit targets and store the command:

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx942 -lineinfo kernel.hip -o kernel_gfx942
hipcc --version
hipcc --list-gpu-targets
rocminfo
```

For reproducible records, store:

- full `hipcc` command line
- ROCm toolkit and driver/runtime version
- GPU name, `gfx` target, clocks, and power state
- library versions for hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, hipCUB,
  rocPRIM, rocThrust, MIOpen, MIGraphX, vLLM on ROCm, Triton, and framework
  baselines
- compiler resource output when available: VGPRs, SGPRs, LDS bytes, spills, and
  launch bounds
- block size, grid size, wave-size assumptions, dynamic LDS, stage count, and
  correctness tolerance

## Feature Probe Checklist

Before claiming an architecture feature, collect direct evidence:

```bash
rocminfo
hipcc --list-gpu-targets
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx942
```

Look for instruction families only when they matter to the claim:

- `v_mfma` or `v_wmma` for Matrix Core paths.
- `ds_read`, `ds_write`, `s_waitcnt`, and `s_barrier` for LDS staging and
  software pipelines.
- `global_load`, `global_store`, and vectorized variants for memory traffic
  claims.
- Library algorithm or generated-kernel identifiers when the baseline uses
  hipBLASLt, Composable Kernel, MIGraphX, Triton, or framework compiler output.

## GFX-Tuned Claim Gate

Promote a result from "runs on this GPU" to "architecture-tuned" only when:

- The binary was built for the claimed `gfx` target.
- The task records whether the implementation is portable or target-specific.
- Disassembly or library metadata supports any instruction-family claim.
- A fair vendor baseline is attached with the same shape, dtype, layout, and
  math contract.
- Resource limits are attached: VGPRs, SGPRs, LDS bytes, occupancy estimate,
  stage count, and wave-size assumptions where relevant.
- The writeup explains the custom-kernel edge: fixed shape, fusion, layout,
  reduced memory traffic, lower launch overhead, or architecture-specific
  instruction use.

If any item is missing, keep the result as useful timing, compile, or
portability evidence rather than an architecture-tuned claim.

## Custom-Kernel Competition Pattern

1. Define shape, dtype, layout, alignment, math mode, epilogue, and tolerance.
2. Build a simple HIP baseline for correctness and sanity.
3. Compare with the strongest library or framework baseline.
4. Write the custom kernel that removes generality, fuses work, or uses a
   verified hardware path.
5. Store wins, losses, and neutral results with the same evidence discipline.

Library wins are not failures. They define useful boundaries.

## Index

Machine-readable lab metadata is stored in
[data/index/architecture_labs.json](../data/index/architecture_labs.json).

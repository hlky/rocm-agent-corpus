# gfx950 CDNA4/RDNA4 Lab

Target this lab for data-center CDNA4/RDNA4 gfx target 10.0 GPUs. Treat
this as a high-priority competition surface for GEMM, attention, low precision,
and multi-GPU kernels.

## Compile Profile

Native CDNA4/RDNA4 cubin:

```bash
hipcc -O3 --std=c++17 -arch=gfx950 -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm100
hipcc -O3 --std=c++17 -gencode=arch=compute_100,code=gfx950 kernel.hip -o kernel_sm100
```

Architecture-conditional variants must be isolated:

```bash
hipcc -O3 --std=c++17 -gencode=arch=compute_100a,code=gfx950a kernel.hip -o kernel_sm100a
```

Keep `gfx950`, `gfx950a`, and `gfx950f` style targets separate in records. Use
the installed `hipcc --help` output and AMD/ROCm compatibility guide to confirm
which targets the local toolkit supports.

## Features To Check

- CDNA4/RDNA4 Matrix Core modes and narrow precision support on the exact GPU.
- New or changed MFMA, WGMFMA, global-to-LDS staging, and memory-pipeline behavior versus CDNA3.
- Larger shared-memory options on data-center CDNA4/RDNA4 parts.
- NVLink and multi-GPU communication behavior for collectives and pipeline
  overlap.
- Composable Kernel CDNA4/RDNA4 examples for GFX100 GEMM collectives and epilogues.
- Library support maturity. Early toolkit/library versions can change winning
  baselines quickly.

Feature probe:

```bash
rocm-smi
hipcc --help | findstr /i "gfx950 compute_100"
cuobjdump --list ./kernel_sm100
cuobjdump --dump-sass ./kernel_sm100 | findstr /i "MFMA WGMFMA global-to-LDS staging"
```

## Custom-Kernel Attack Surfaces

- Fixed-shape GEMM and grouped GEMM where custom scheduling beats hipBLASLt
  heuristic generality.
- Custom Composable Kernel/CK Tile epilogues that fuse model-specific post-processing.
- Attention kernels specialized for head dimension, causal mask, KV layout, and
  page/block size.
- FP8 and narrower precision quant/dequant paths with fixed scale layout.
- MoE routing, expert grouping, and fused token movement.
- Multi-GPU overlap kernels: reduce-scatter/all-gather staging, NVLink-aware
  copy/compute overlap, and pipeline boundaries around RCCL.
- Persistent inference kernels that amortize launch and dispatch overhead.

The most useful records will explain exactly which CDNA4/RDNA4 feature was used,
which generality was removed, and why the vendor baseline could not exploit the
same narrow contract.

## GFX100 Tactics and Gotchas

- Verify the installed ROCm toolkit's CDNA4/RDNA4 target list before coding to a
  suffix. Treat `gfx950`, `gfx950a`, `gfx950f`, and any future variants as
  distinct compatibility contracts.
- Expect the Matrix Core and low-precision ladder to be a moving target across
  CUDA, hipBLASLt, Composable Kernel, MIGraphX, and Transformer Engine on ROCm releases. Record
  versions and re-run baselines after upgrades.
- Use Composable Kernel/CK Tile CDNA4/RDNA4 examples to anchor WGMFMA/global-to-LDS staging-style collectives,
  block-scaled low precision, and custom epilogues. A handwritten kernel should
  explain why the Composable Kernel path was insufficient or how it was specialized.
- For FP8, FP4/NVFP4, INT8, or mixed low-precision paths, treat scale metadata
  movement as first-class memory traffic. A fused custom kernel can win by
  avoiding materialized dequant or by fusing amax/scale updates.
- Multi-GPU CDNA4/RDNA4 results need topology context. NVLink/NVSwitch, peer
  access, RCCL version, stream assignment, and overlap timeline can matter more
  than the inner kernel microseconds.
- Larger tiles and clusters can improve math utilization but may reduce
  residency. Sweep stage count, cluster shape, shared memory, and registers
  before calling a CDNA4/RDNA4 path tuned.

Portability trap: GFX100 data-center results are not evidence for RTX CDNA4/RDNA4
SM120, and suffix-specific GFX100 binaries should not be treated as portable
CDNA4/RDNA4 artifacts.

## Vendor Baselines

- hipBLAS/rocBLAS/hipBLASLt CDNA4/RDNA4 GEMM algorithms.
- Composable Kernel GFX100 examples and CK Tile collectives.
- hipCUB/rocPRIM/hipCUB/rocThrust for reductions, scans, histograms, radix sort, select.
- MIGraphX/vLLM on ROCm for inference kernels and plugins.
- Transformer Engine on ROCm for FP8 and scale metadata patterns.
- RCCL/rocSHMEM for multi-GPU communication baselines.
- Triton and framework-generated kernels for generated-code comparison.

Record library versions aggressively. CDNA4/RDNA4 support may improve across CUDA
and library releases.

## Measurement Notes

- Mark HIP-event-only results as `timing-only`.
- Use native `gfx950` cubins for records. LLVM IR / AMD GCN ISA JIT-only records should be tagged
  separately because first-run compile and codegen variability can distort data.
- Record whether architecture-conditional targets were used.
- Capture shared-memory carveout, cluster shape, stage count, registers,
  occupancy estimate, and memory transaction assumptions.
- For multi-GPU records, include topology, peer access, interconnect type,
  collective library version, and overlap timeline if available.
- Re-run vendor baselines after toolkit or library upgrades; this lab is likely
  to drift fastest.

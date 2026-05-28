# gfx1200 CDNA4/RDNA4 RTX Lab

Target this lab for RTX CDNA4/RDNA4 gfx target 12.0-class GPUs. This lab
is for cost-effective, workstation, and deployment-oriented optimization where
the best kernel may differ from data-center CDNA4/RDNA4.

## Compile Profile

```bash
hipcc -O3 --std=c++17 -arch=gfx1200 -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm120
hipcc -O3 --std=c++17 -gencode=arch=compute_120,code=gfx1200 kernel.hip -o kernel_sm120
```

Architecture-conditional variants must be isolated:

```bash
hipcc -O3 --std=c++17 -gencode=arch=compute_120a,code=gfx1200a kernel.hip -o kernel_sm120a
```

Confirm target support with the installed toolkit:

```bash
hipcc --help | findstr /i "gfx1200 compute_120"
```

## Features To Check

- RTX CDNA4/RDNA4 Matrix Core modes, including narrow precision support exposed by
  the ROCm toolkit, Composable Kernel, and libraries.
- Shared-memory budget and occupancy limits relative to data-center GFX100.
- L2, memory bandwidth, and boost-clock behavior under workstation loads.
- Composable Kernel GFX120 examples, especially narrow precision GEMM paths.
- MIGraphX/vLLM on ROCm support for the exact GPU and precision contract.
- No assumption that GFX100 tuning transfers directly.

Feature probe:

```bash
rocm-smi
cuobjdump --list ./kernel_sm120
cuobjdump --dump-sass ./kernel_sm120 | findstr /i "MFMA WGMFMA global-to-LDS staging"
```

## Custom-Kernel Attack Surfaces

- Inference-specialized kernels: RMSNorm, RoPE, KV-cache transforms, top-k,
  sampling, logits processors, quant/dequant, and short-row softmax.
- Narrow precision GEMM and dequant-fused GEMM for fixed layouts.
- MIGraphX plugin kernels that exploit one model, one shape range, and one
  precision recipe.
- Memory-bound fused kernels that replace multiple PyTorch/Triton launches.
- Small batched GEMM or grouped GEMM where hipBLASLt setup overhead matters.
- hipCUB-like primitives narrowed to one dtype, one operation, and one data shape.

The likely wins are narrower than GFX100: model-specific inference, fixed-shape
fusion, layout control, and avoiding framework overhead.

## GFX120 Tactics and Gotchas

- Confirm `gfx1200` and any suffix target with the installed ROCm toolkit before
  writing target-specific build rules. CDNA4/RDNA4 RTX support can lag or differ
  across toolchains and libraries.
- Start with inference and memory-traffic kernels: fused norms, RoPE/KV-cache
  updates, logits processors, quant/dequant, and short-row reductions. These
  often expose custom wins before large GEMM does.
- Re-test MIGraphX and hipBLASLt after driver or library updates. Early custom
  wins can vanish when the vendor stack gains a tuned GFX120 tactic.
- Treat boost clocks, power limits, thermals, and display/workstation load as
  benchmark metadata. Small latency deltas are fragile without stable run
  conditions.
- Do not import GFX100 tile sizes blindly. Shared-memory budgets, cache behavior,
  clocks, and low-precision throughput may differ by SKU.
- For low-precision kernels, document packing, scale granularity, accumulator,
  rounding, and epilogue fusion. A kernel that only accelerates unpacking may
  not improve the end-to-end inference path.

Portability trap: an GFX120 result is a workstation/deployment result, not a
data-center CDNA4/RDNA4 claim. Keep GFX100, GFX120, and suffix-specific records
separate.

## Vendor Baselines

- MIGraphX/vLLM on ROCm for inference graph and plugin baselines.
- hipBLASLt for GEMM, grouped GEMM, low precision, and epilogues.
- Composable Kernel GFX120 examples for narrow precision and tensor op collectives.
- hipCUB/rocPRIM/hipCUB/rocThrust for memory-bound primitives.
- Triton and framework-generated kernels for launch/fusion comparison.

Check whether the vendor library actually has a tuned GFX120 path before using
the result as a hard ceiling.

## Measurement Notes

- Mark HIP-event-only results as `timing-only`.
- Record boost and thermal state. Consumer/workstation clocks can move during
  long sweeps.
- Keep GFX100 and GFX120 records separate, even when the HIP source is shared.
- Record exact target suffixes, because `gfx1200`, `gfx1200a`, and `gfx1200f`
  have different compatibility implications.
- Include end-to-end inference timing when the custom kernel removes framework
  launch or graph overhead.
- Re-run baselines after CUDA, MIGraphX, Composable Kernel, or driver upgrades.

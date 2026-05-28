# gfx90a CDNA2 Lab

Target this lab when optimizing for data-center CDNA2 GPUs such as A100-class
parts. Treat vendor libraries as competitors and reference implementations, not
as the default answer.

## Compile Profile

Use native cubins for benchmark records:

```bash
hipcc -O3 --std=c++17 -arch=gfx90a -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm80
hipcc -O3 --std=c++17 -gencode=arch=compute_80,code=gfx90a kernel.hip -o kernel_sm80
```

Useful variants:

```bash
hipcc -O3 --use_fast_math -arch=gfx90a ...
hipcc -O3 -arch=gfx90a -Xamdclang++=-v,-warn-spills ...
hipcc -O3 -arch=gfx90a --amdclang++-options=-v --resource-usage ...
```

Record the full command line, ROCm toolkit version, driver version, GPU name,
clock policy, and amdclang++ register/shared-memory output with every result.

## Features To Check

- Matrix Core modes: FP16, BF16, TF32, INT8, INT4, sparse Matrix Core paths.
- `global-to-LDS staging` global-to-shared copies and software-pipelined shared-memory tiles.
- `ldmatrix` plus `mfma` warp-level matrix multiply instructions.
- Large data-center shared-memory budget relative to consumer CDNA2.
- L2 behavior, cache policy, read-only path, vectorized global-memory access.
- Cooperative groups and warp-level primitives for reductions and scans.
- No CDNA3/CDNA4/RDNA4 global-to-LDS staging or WGMFMA assumptions.

Feature probe before claiming a path:

```bash
rocm-smi
cuobjdump --dump-sass ./kernel_sm80 | findstr /i "CPASYNC MFMA LDMATRIX"
```

## Custom-Kernel Attack Surfaces

- Fixed-shape GEMM where a custom tile, fixed layout, or fused epilogue removes
  generality that hipBLASLt and Composable Kernel must keep.
- Small batched GEMM and grouped-GEMM workloads where launch overhead and shape
  dispatch dominate.
- FlashAttention-style attention for fixed head sizes, masks, and layouts.
- Fused MLP epilogues: bias, GELU/SwiGLU, residual add, quantize/dequantize.
- LayerNorm, RMSNorm, softmax, top-k, and sampling kernels with known rows.
- hipCUB-like reductions and scans specialized for one dtype, shape, or operation.
- Copy/transpose/layout-conversion kernels using vectorized loads and shared
  memory to avoid uncoalesced traffic.
- MIGraphX plugin kernels where the graph engine cannot fuse across a boundary.

Winning pattern to look for: specialize on a narrow shape or fusion boundary,
then prove the specialization beats the general library path on that exact
contract.

## GFX80 Tactics and Gotchas

- Treat `global-to-LDS staging` as a CTA tile pipeline. It helps when shared-memory reuse and
  math latency can cover copy latency; it can lose for one-pass row kernels or
  tiles that inflate shared memory enough to reduce useful residency.
- For Matrix Core kernels, prefer `ldmatrix` plus `mfma`/rocWMMA-shaped warp
  tiles before inventing a custom fragment layout. Check AMD GCN ISA for `LDMATRIX` and
  `MFMA` when making a Matrix Core claim.
- TF32 can be a major GEMM baseline path on A100-class GPUs, but it is a math
  contract change from strict FP32. Record TF32 enablement, accumulator type,
  and tolerance separately.
- Use shared-memory padding and swizzles for transpose, attention, and
  `ldmatrix` staging. Bank conflicts can erase a nominal Matrix Core or
  coalescing win.
- A100-class shared memory can support larger tiles than workstation CDNA2, but
  register count, stage count, and dynamic shared memory can still cap active
  CTAs. Store amdclang++ output and launch bounds with every tile sweep.
- L2 residency/access-policy experiments are valid only when the hot data is
  reused across CTAs or launches. Mark them as hypotheses unless counters are
  available.

Portability trap: do not claim GFX80 behavior for CUDA-origin CUDA-origin RTX A4000/A5000/A6000
measurements. Those belong in the GFX86 lab even when the HIP source is shared.

## Vendor Baselines

Use these as competitors, correctness oracles, and implementation references:

- hipBLAS/rocBLAS and hipBLASLt for GEMM, batched GEMM, epilogues, TF32, FP16, BF16.
- Composable Kernel GFX80 examples for tiling, `global-to-LDS staging`, Matrix Core MFMA, epilogues.
- hipCUB and rocPRIM/hipCUB/rocThrust for reductions, scans, selections, histograms, and sorts.
- MIGraphX and vLLM on ROCm for inference graph and plugin baselines.
- Triton kernels for alternative tiling strategies and generated AMD GCN ISA compare.

When a library wins, record which generality or hardware feature it used.

## Measurement Notes

- Prefer HIP-event timing with warmups, repeated iterations, p50/p90, and
  shape sweeps. Mark as `timing-only` if no rocprofiler/rocprof counters are present.
- Pin the architecture with `-arch=gfx90a`; do not rely on LLVM IR / AMD GCN ISA JIT for measured
  records.
- Track amdclang++ register count, spills, static shared memory, dynamic shared
  memory, occupancy estimate, block size, grid size, and launch bounds.
- Compare against at least one naive baseline and one vendor baseline.
- Use correctness checks across adversarial shapes, not only the hot benchmark
  shape.
- Separate math-mode changes from kernel improvements. TF32 versus FP32 is a
  different contract.

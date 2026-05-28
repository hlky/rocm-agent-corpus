# gfx1030 CDNA2 Pro Lab

Target this lab for CDNA2 professional and workstation GPUs such as CUDA-origin CUDA-origin RTX A4000,
A5000, A6000, A10-class, and related gfx1030 parts. These GPUs are often cheaper
benchmark hosts, but their memory system, clocks, and shared-memory budget do
not behave like A100.

## Compile Profile

```bash
hipcc -O3 --std=c++17 -arch=gfx1030 -lineinfo -Xamdclang++=-v kernel.hip -o kernel_sm86
hipcc -O3 --std=c++17 -gencode=arch=compute_86,code=gfx1030 kernel.hip -o kernel_sm86
```

Useful variants:

```bash
hipcc -O3 --use_fast_math -arch=gfx1030 ...
hipcc -O3 -arch=gfx1030 -Xamdclang++=-v,-warn-spills ...
hipcc -O3 -arch=gfx1030 --resource-usage ...
```

Avoid assuming A100 tuning transfers directly. Re-measure tile sizes, shared
memory carveout, occupancy, and vector width.

## Features To Check

- Matrix Core modes: FP16, TF32, INT8, INT4, sparse Matrix Core paths.
- BF16 support and performance must be probed per SKU and toolkit path.
- `global-to-LDS staging`, `ldmatrix`, and `mfma` are relevant, but useful tile sizes may
  differ from gfx90a.
- Shared-memory capacity is smaller than A100-class gfx90a; large tiles can lose
  occupancy or fail launch.
- Thermal boost and workstation driver settings can add variance.
- No CDNA3/CDNA4/RDNA4 global-to-LDS staging or WGMFMA assumptions.

Feature probe:

```bash
rocm-smi --query-gpu=name,compute_cap,clocks.sm,clocks.mem,power.limit --format=csv
cuobjdump --dump-sass ./kernel_sm86 | findstr /i "CPASYNC MFMA LDMATRIX"
```

## Custom-Kernel Attack Surfaces

- Kernels that are launch-overhead limited: small reductions, small GEMMs,
  tensor transforms, norm kernels, and short-row softmax.
- Fixed-shape inference kernels that can beat generalized framework dispatch.
- Fused elementwise chains where memory traffic dominates and Matrix Cores are
  not the limiting resource.
- Layout conversion, transpose, pack/unpack, and quant/dequant kernels with
  tight vectorization and alignment contracts.
- Specialized hipCUB competitors for reductions and scans with known operation,
  dtype, and block size.
- Composable Kernel-inspired GEMM microkernels for small or unusual shapes where library
  heuristic selection is conservative.

The most common win is removing runtime generality: fixed dimensions, fixed
alignment, one dtype, one epilogue, and one launch shape.

## GFX86 Tactics and Gotchas

- Start from GFX80 source patterns, then shrink or retune tiles. Shared-memory
  capacity, clocks, and memory bandwidth differ enough that A100 tile shapes can
  lose occupancy or become latency-bound here.
- `global-to-LDS staging` remains useful for tiled GEMM, attention, convolution-like stencils,
  and transpose/layout kernels, but use a two- or three-stage sweep instead of
  assuming the deepest pipeline wins.
- Matrix Core paths should be verified with `LDMATRIX` and `MFMA` in AMD GCN ISA. Probe
  BF16 and narrow-precision performance per SKU and library stack before making
  a precision-specific claim.
- Workstation/pro CDNA2 is a good place to find launch-overhead wins: fused
  RMSNorm/RoPE/KV writes, small reductions, and fixed-shape dequant kernels can
  beat framework dispatch even when a larger library primitive would win at
  scale.
- Watch register pressure from aggressive unrolling. On smaller tiles, spills
  often cost more than the instruction overhead they were meant to remove.
- Thermal and boost behavior can dominate small differences. Use medians,
  repeated sweeps, and clock/power notes before promoting a narrow win.

Portability trap: `gfx1030` evidence does not prove A100 behavior. Re-run
libraries and custom kernels on `gfx90a` before generalizing to data-center
CDNA2.

## Vendor Baselines

- hipBLAS/rocBLAS/hipBLASLt for GEMM and batched GEMM.
- Composable Kernel GFX80/SM86 kernels for tiled GEMM patterns and epilogues.
- hipCUB/rocPRIM/hipCUB/rocThrust for reductions, scans, histograms, select, sort.
- MIGraphX for inference fusion and plugin comparison.
- Triton for generated code comparison, especially for compact kernels.

Document the exact library version and chosen algorithm if the API exposes it.

## Measurement Notes

- Mark HIP-event-only results as `timing-only`.
- Watch boost-clock drift. Prefer repeated runs, medians, and stable power state
  notes.
- Use `-arch=gfx1030` for records from this lab. Do not store gfx90a binaries as
  gfx1030 evidence.
- Record amdclang++ registers, spills, shared memory, block size, vector width,
  warmup count, iteration count, and correctness tolerance.
- Include a shape sweep around the target shape. gfx1030 wins often disappear
  when rows, columns, or batch counts move slightly.

# Quantization Kernel Guide

This guide defines the quantization and fused dequantization track for the ROCm
agent corpus. The goal is to help agents write custom kernels that can beat,
match, or narrowly specialize beyond generic quantized inference and training
paths from hipBLASLt, Composable Kernel, bitsandbytes, Transformer Engine on ROCm, vLLM on ROCm,
MIGraphX plugins, Triton, vLLM, FlashInfer, and framework-generated kernels.

Vendor libraries are still the performance bar and correctness reference. They
are not the stopping point. A custom quantization kernel wins when it exploits a
fixed shape, layout, scale format, fused operation, or deployment boundary that
the generic path must keep configurable.

## Evidence Policy

- Use `template-only` for scaffolds that compile or describe a benchmark but
  have not been measured.
- Say `timing-only` for HIP-event timing results without rocprofiler/rocprof
  counters.
- Say `negative example` when a packed, vectorized, fused, or specialized path
  does not improve speed.
- Do not invent memory-transaction, cache, Matrix Core, occupancy, or pipe
  utilization evidence. Record those only after profiler counters are collected.
- Attach GPU, gfx target, ROCm toolkit, driver, compiler, build command,
  library versions, shape, dtype, layout, scale layout, zero-point layout,
  group size, warmup count, measurement count, and correctness tolerance to
  every measured claim.

## Format Concepts

### INT8

INT8 paths usually store signed or unsigned 8-bit values plus scale metadata.
Common contracts are:

- Symmetric: `real = int8_value * scale`.
- Asymmetric: `real = (uint8_value - zero_point) * scale`.
- Accumulate in INT32 or FP32, then scale, clamp, bias, activate, or cast in the
  epilogue.

INT8 custom kernels can compete when the task has a fixed layout, a fixed
per-channel or per-group scale scheme, a fused epilogue, or a small-batch serving
boundary where descriptor and framework overhead matter.

### INT4

INT4 paths store two 4-bit codes per byte. The first corpus task uses unsigned
nibbles:

- Low nibble stores even column `k`.
- High nibble stores odd column `k + 1`.
- `real_weight = (q4 - zero_point) * scale`.
- `scale` and `zero_point` are per output row and per K-group.

Other real systems may use signed int4, NF4-like codebooks, or different nibble
order. Do not assume the packing order; record it in the task metadata and test
odd K, non-square shapes, and group tails.

### FP8

FP8 kernels commonly use compact floating formats with separate scaling
metadata, amax history, and FP16/BF16/FP32 accumulation choices. E4M3-style and
E5M2-style contracts differ in range, precision, saturation, NaN, and infinity
handling. Treat each as a separate numerical contract and compare only against a
library path configured for the same scale and cast policy.

Transformer Engine on ROCm is the primary local reference for transformer FP8 scale,
amax, and framework integration patterns. Composable Kernel and hipBLASLt are relevant for
FP8 GEMM mainloops and epilogues when the current local submodule version
supports the target architecture and datatype.

### NVFP4-Style And Block-Scaled 4-Bit Formats

For this corpus, "nvfp4-style" means a 4-bit floating or minifloat code paired
with block-scale metadata. Exact encodings, supported instructions, and library
APIs are architecture and release dependent, so records must point to the local
Transformer Engine on ROCm, Composable Kernel, vLLM on ROCm, or official documentation version
used for the experiment.

Agent checklist:

- Identify the 4-bit code format and scale dtype.
- Record block size, scale layout, and any second-level scale.
- Record accumulator dtype and output dtype.
- Keep dequantization fused unless the benchmark is explicitly studying a
  materialized dequant baseline.

## Scale And Zero-Point Layouts

Scale granularity controls both numerical quality and memory traffic.

- Per-tensor scale is simple but often too coarse.
- Per-output-channel scale is common for weight-only quantized linear layers.
- Per-group scale splits K into fixed groups such as 32, 64, 128, or 256
  elements.
- Per-token or per-row activation scale is common for dynamic activation
  quantization.
- Block-scaled formats may have scale blocks that are independent of logical
  GEMM tile boundaries.

Zero points may be absent, scalar, per-channel, or per-group. A custom kernel
can win when the group size and metadata layout are fixed enough to remove
generic indexing and branch paths.

Correctness traps:

- Group tails when `K % group_size != 0`.
- Zero-point dtype and signedness.
- Nibble order inside packed int4 bytes.
- Scale broadcast axis.
- Accumulation order and tolerance.
- Bias, residual, activation, and requantization semantics.

## Packed Loads

Packed low-bit kernels trade arithmetic for memory bandwidth. The win depends on
whether unpacking and scale loads are cheaper than reading wider dequantized
weights.

Implementation notes:

- Load multiple packed bytes per thread when alignment permits, then unpack in
  registers.
- Keep the packed layout coalesced along K for GEMV and GEMM mainloops.
- Reuse scale and zero-point metadata across all values in a group.
- Avoid materializing dequantized weights to global memory unless it is the
  measured baseline.
- Test odd K and group-tail shapes to catch off-by-one unpack bugs.
- Record whether packed loads are byte, 32-bit, 64-bit, 128-bit, or vectorized.

The seed optimized kernel in
`corpus/tasks/fused-int4-dequant-gemv/source/optimized.hip` reads one packed byte
for two weights, dequantizes both in registers, multiplies by the activation
vector, and reduces inside one CTA per output row.

## Fused Dequantization

The core optimization is to fuse:

1. Packed load.
2. Unpack.
3. Scale and zero-point correction.
4. Multiply or attention score contribution.
5. Reduction, epilogue, or output cast.

Do not compare a fused custom kernel only against a weak baseline that first
dequantizes all weights to FP16/FP32 and then calls GEMM. Include the strongest
available direct quantized library or plugin path when one matches the contract.
If the custom kernel only beats the materialized-dequant baseline, record that
as a partial win, not a general library win.

## Integration Targets

### GEMV

GEMV is important for decode-heavy LLM inference, reranking, embeddings, and
small-batch projection work. Custom kernels can win when:

- Batch size is one or small.
- The K dimension and group size are fixed.
- Weight-only quantization avoids reading FP16/FP32 weights.
- Bias, activation, residual add, or output quantization can be fused.
- The library path has setup, descriptor, or dispatch overhead that dominates.

The first task in this track is `fused-int4-dequant-gemv`.

### GEMM

GEMM is the main path for prefill, training, and batched inference. Baselines
must include hipBLASLt, Composable Kernel, vLLM on ROCm, Triton, or framework-generated
paths where relevant. A custom kernel can compete by fixing tile shapes, scale
granularity, output type, epilogue, or architecture path. Composable Kernel can also be a
custom extension surface when the agent changes layout, tile shape, epilogue,
or datatype policy.

### Attention

Quantization appears in QKV projections, KV cache storage, attention score
computation, and value accumulation. For attention tasks, compare against
FlashAttention, FlashInfer, vLLM, vLLM on ROCm, MIOpen, or Triton when
they match the deployment scope. Custom wins often come from fusing dequantized
KV loads with online softmax, causal masking, paged-cache indexing, or one-query
decode scheduling.

## Reference And Competitor Map

- `third_party/bitsandbytes`: practical low-bit quantization and optimizer
  kernels; useful for int8/int4 packing, dequantization, and inference/training
  integration ideas.
- `source:transformer-engine`: FP8 and transformer-layer scale/amax
  reference; use for FP8 and block-scaled transformer paths.
- `third_party/composable-kernel`: Matrix Core GEMM, CK Tile layouts, int8/FP8 paths, custom
  epilogues, and low-bit extension surfaces.
- `third_party/vllm`: LLM inference plugins, quantized linear layers,
  KV cache, batching, and serving-level baselines.
- `third_party/migraphx`: plugin boundary, engine precision, dynamic shape, and
  deployment baseline.
- `third_party/triton`: fast sketch kernels and compiler-generated competitors.
- `third_party/vllm` and `third_party/flashinfer`: serving and attention
  integration references.

## Baseline Ladder

1. CPU reference.
   - Defines packing, scale, zero point, accumulation, and tolerance.

2. Materialized dequant baseline.
   - Dequantize to FP16/FP32, then run GEMV/GEMM/attention.
   - Useful but usually not the strongest performance baseline.

3. Naive fused custom HIP.
   - One thread per row or one output per thread.
   - Teaches indexing, nibbles, group metadata, and numerical contract.

4. Cooperative fused custom HIP.
   - One warp or one CTA per row/tile.
   - Reuses packed loads, scales, and reductions.

5. Matrix Core or architecture-specific custom path.
   - Composable Kernel, CK Tile, rocWMMA, MFMA, MFMA, global-to-LDS staging, or native low-bit instructions where
     applicable.

6. Vendor and framework competitors.
   - hipBLASLt, Composable Kernel examples, MIGraphX, vLLM on ROCm, bitsandbytes,
     Transformer Engine on ROCm, Triton, vLLM, FlashInfer, or framework compiler output.

## Custom-Kernel Win Conditions

A custom quantization kernel can credibly win when it drops generality that the
library must preserve:

- Fixed `M`, `N`, `K`, batch size, head dimension, group size, or page size.
- Known packed layout, alignment, and nibble order.
- Known scale and zero-point layout.
- Weight-only quantization with reused weight metadata.
- Decode GEMV where generic GEMM scheduling underutilizes the GPU.
- Fusion with bias, activation, residual, rotary transform, mask, softmax,
  requantization, or output layout conversion.
- Avoiding a materialized dequantized tensor.
- Avoiding framework dispatch, descriptor setup, allocation, or extra launches.
- MIGraphX plugin or vLLM on ROCm kernel boundary that removes graph breaks.
- Architecture-specific instructions or memory movement patterns not available
  through the generic path.

Keep losing cases. A custom int4 kernel that loses to vLLM on ROCm or Composable Kernel
because unpacking overhead dominates is valuable when it records the shape,
layout, and next plausible specialization.

## Current Scaffold

Task:

- `corpus/tasks/fused-int4-dequant-gemv/task.json`

Sources:

- `corpus/tasks/fused-int4-dequant-gemv/source/baseline.hip`
- `corpus/tasks/fused-int4-dequant-gemv/source/optimized.hip`

Harness:

- `harnesses/int4_dequant_gemv_benchmark.hip`

Example build commands:

```bash
hipcc -O3 -std=c++17 -arch=gfx90a -DVARIANT_BASELINE ^
  corpus/tasks/fused-int4-dequant-gemv/source/baseline.hip ^
  harnesses/int4_dequant_gemv_benchmark.hip ^
  -o build/int4_dequant_gemv_baseline

hipcc -O3 -std=c++17 -arch=gfx90a -DVARIANT_OPTIMIZED ^
  corpus/tasks/fused-int4-dequant-gemv/source/optimized.hip ^
  harnesses/int4_dequant_gemv_benchmark.hip ^
  -o build/int4_dequant_gemv_optimized
```

Default problem shape:

- `rows=4096`
- `cols=4096`
- `group_size=128`
- packed row-major int4 weights with two values per byte
- per-row/per-group FP32 scales
- per-row/per-group uint4 zero points stored in bytes
- FP32 activation vector and FP32 output

The default seed shape is a ROCm timing target; no result files are currently
checked in. Future measurements should mark HIP-event-only results as
`timing-only` and attach the full hardware and build metadata.

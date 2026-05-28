# Matrix Core and rocWMMA Guide

This guide is the entry point for the Matrix Core / rocWMMA / MFMA track. The goal
is to help agents write custom HIP kernels that can compete with, specialize
beyond, or extend hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, CK Tile, Triton, MIGraphX,
vLLM on ROCm, vLLM, FlashInfer, and framework-generated kernels.

The default seed shape is a ROCm timing target. Do not describe any shape as
measured until the harness has been built and run on a named AMD GPU with
hardware, toolkit, build flags, timing, and correctness metadata attached.

## Seed Workload

The seed task is `corpus/tasks/rocwmma-mfma-gemm/`.

Contract:

```text
C[M, N] = A[M, K] * B[K, N]
```

- `A`: FP16, row-major contiguous, leading dimension `K`.
- `B`: FP16, column-major contiguous, leading dimension `K`.
- `C`: FP32, row-major contiguous, leading dimension `N`.
- Accumulation: FP32.
- Epilogue: none in the benchmark seed.
- Optimized path: rocWMMA 16x16x16 tiles when `M`, `N`, and `K` are multiples of
  16; scalar fallback otherwise.

The column-major `B` layout is intentional. It matches the classic
`matrix_a row_major` plus `matrix_b col_major` rocWMMA teaching path and makes the
fragment layout visible. A production row-major wrapper can either transpose the
logical operand mapping, stage into shared memory, use a different instruction
layout, or move to Composable Kernel/CK Tile where the layout algebra is explicit.

## Baseline Ladder

Use the same problem contract at every rung.

1. CPU reference.
   The harness computes a host FP32 reference from the exact FP16 inputs copied
   to the GPU. This is the correctness oracle, not a speed baseline.

2. Naive HIP baseline.
   One thread computes one output element using scalar half-to-float conversion
   and FP32 accumulation. This teaches indexing, layout, and tolerance.

3. rocWMMA custom kernel.
   One warp computes one 16x16 output tile with `rocwmma::load_matrix_sync`,
   `mma_sync`, and `store_matrix_sync`. This is intentionally simple: no
   multi-stage pipeline, no LDS software pipeline, no shared-memory swizzle.

4. Inline `mfma`.
   Use handwritten LLVM IR / AMD GCN ISA only when the instruction shape, register layout,
   operand layout, and target GFX are part of the lesson. Inline MFMA is where
   agents learn what rocWMMA hid, but it is also where portability mistakes become
   easy.

5. Composable Kernel or CK Tile.
   Treat Composable Kernel and CK Tile as reference material, competitors, and extension
   surfaces. A Composable Kernel-derived kernel counts as custom work when tile shapes,
   schedules, layouts, data types, or epilogues are selected or modified and
   recorded.

6. hipBLAS/rocBLAS, hipBLASLt, Triton, MIGraphX, vLLM on ROCm, vLLM, and FlashInfer.
   These are strong baselines and correctness references. They are not a reason
   to stop. If they win, record why and state the next narrower custom path.

The first hipBLASLt runnable baseline is
`harnesses/hipblaslt_hgemm_benchmark.hip`, exposed through
`tools/run_library_baseline.py`. It measures the same FP16-input,
FP32-accumulate, FP32-output contract as `rocwmma-mfma-gemm`. No ROCm
hipBLASLt result artifact is currently checked in; run the custom and library
paths on the same AMD GPU before making a speedup claim.

## rocWMMA Mental Model

rocWMMA is a warp-level API. A warp cooperatively owns fragments of A, B, and C.
The thread-to-element mapping inside a fragment is deliberately opaque.

Core operations:

- `rocwmma::fragment`: declares a matrix operand or accumulator fragment.
- `rocwmma::fill_fragment`: initializes an accumulator fragment.
- `rocwmma::load_matrix_sync`: warp-synchronous load from memory into a fragment.
- `rocwmma::mma_sync`: warp-synchronous matrix multiply accumulate.
- `rocwmma::store_matrix_sync`: warp-synchronous store from an accumulator
  fragment.

Practical rules:

- Every warp lane must participate in the same rocWMMA operation.
- Pointers and leading dimensions must satisfy the API alignment requirements
  for the selected datatype and layout.
- Fragment element order is not a portable memory layout.
- Do not pass fragments between translation units compiled for different GFX
  targets.
- Keep the first kernel tile-aligned, then add guarded edge handling as a
  separate optimization.

## Fragment Layouts

Fragments are not arrays with a stable row/column order. The rocWMMA API exposes a
`num_elements` member and an `x[]` storage array, but that storage is useful
only for elementwise operations that apply identically to all accumulator
elements, such as multiplying by `alpha` or clamping every value.

If an epilogue needs coordinates, prefer one of these paths:

- Store the accumulator fragment to shared memory or global memory with
  `store_matrix_sync`, then run coordinate-aware code.
- Use Composable Kernel/CK Tile epilogue machinery where the accumulator tile layout is
  represented explicitly.
- Move to inline MFMA only after documenting the register fragment layout for the
  exact instruction and architecture.

Common layout mistakes:

- Treating row-major `B[K, N]` as compatible with `matrix_b col_major` without
  changing the address formula.
- Passing `N` as the leading dimension for a column-major B operand when the
  leading dimension should be `K`.
- Square-only testing that hides a swapped leading dimension.
- Assuming a fragment index maps to the same row and column on Volta, CDNA2,
  CDNA3, and CDNA4/RDNA4.

## Dtype Boundaries

Matrix Core comparisons are only fair when the math contract is explicit.

FP16:

- Good first rocWMMA target.
- Typical contract is FP16 inputs with FP32 accumulation and FP32 or FP16 output.
- Compare against hipBLASLt and Composable Kernel with the same accumulator and output
  type.

BF16:

- Treat as a CDNA2-and-newer Matrix Core path only when the target GPU and ROCm
  toolkit prove support.
- Use a separate task or compile-time path because tolerances and conversion
  behavior differ from FP16.
- Composable Kernel and hipBLASLt are usually better starting references than handwritten
  rocWMMA for production BF16 kernels.

TF32:

- TF32 is not strict FP32.
- A scalar FP32 custom kernel compared against a library TF32 Matrix Core GEMM
  is not a same-contract comparison.
- Record hipBLAS/rocBLAS/hipBLASLt math mode, ROCm version, and whether TF32 is allowed.

FP8 and lower precision:

- Do not assume the simple rocWMMA API is the right interface.
- Use hipBLASLt, Composable Kernel/CK Tile, Transformer Engine on ROCm, vLLM on ROCm, vLLM, or
  FlashInfer as baselines and reference implementations.
- Record FP8 format, scale layout, amax behavior, accumulator type, output type,
  and any fast-accumulation mode.
- CDNA3, RDNA3, and CDNA4/RDNA4 paths can differ materially. Verify exact GFX flags,
  toolkit support, and library versions before writing architecture claims.

INT8 and sub-byte:

- Treat quantization, scale/dequant, clamp, and packing as part of the kernel
  contract.
- Custom wins often come from fixed scale layout or fused epilogues, not plain
  GEMM throughput.

## rocWMMA Versus `mfma`

rocWMMA is portable enough for teaching and first custom kernels. Inline
`mfma` is useful when the kernel needs instruction shapes or register
control not exposed by rocWMMA.

Inline MFMA work must record:

- LLVM IR / AMD GCN ISA instruction shape, such as `m16n8k16`.
- Operand layouts, such as `row.col`.
- Input, accumulator, and output types.
- Register packing and lane ownership.
- How operands are loaded and how LDS/shared memory is arranged.
- Target architecture and compile flags.

Example instruction family to study for CDNA2-style FP16:

```text
mfma.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Do not copy an inline LLVM IR / AMD GCN ISA snippet into a production claim without a correctness
reference and a library baseline. The instruction can be correct while the tile
mapping, leading dimensions, or epilogue are wrong.

## Epilogues

Matrix Core mainloops are only part of the contest. A custom kernel may win by
fusing work after the multiply:

- Bias add.
- ReLU, GELU, SiLU, clamp, or scale.
- Residual add.
- Dequantization or quantization.
- Layout conversion.
- Auxiliary output such as pre-activation or amax.

Comparison rules:

- Plain hipBLAS/rocBLAS plus separate kernels is not the strongest baseline when
  hipBLASLt supports the same epilogue.
- Composable Kernel custom epilogues are reference material and a credible custom path.
- Record broadcast axis, vector width, output alignment, and whether the
  epilogue raises register pressure enough to slow the mainloop.

## Occupancy and Scheduling

The seed rocWMMA kernel uses one warp per 16x16 tile. That is deliberately simple
and often not enough to beat libraries.

Tune these axes one at a time:

- Warps per CTA and output tiles per CTA.
- CTA tile shape.
- Warp tile shape.
- K tile and loop unrolling.
- Registers per thread.
- Shared memory per CTA.
- Occupancy after dynamic shared memory and register allocation.
- Split-K, stream-K, persistent scheduling, or grouped scheduling.

Occupancy is not a performance claim by itself. Record it as context, then
measure median latency and throughput. If rocprofiler/rocprof counters are not
available, say `timing-only` and avoid counter-backed explanations.

## Shared Memory and Swizzling

The seed optimized kernel loads directly from global memory with rocWMMA. The next
serious step is usually staged shared memory.

Shared-memory checklist:

- Keep rocWMMA loads aligned.
- Add padding or swizzling to avoid bank conflicts.
- Verify the row/column mapping after any skew.
- Use vectorized global loads only when alignment is guaranteed.
- For CDNA-style kernels, consider LDS software pipelining after the non-pipelined shared
  path is correct.
- For CDNA3 and newer, consider Composable Kernel/CK Tile examples before
  hand-rolling barrier and wait-count logic.

Swizzling is architecture-sensitive. Keep a simple unswizzled version as a
negative or reference example when a swizzle does not improve timing.

## Architecture Gotchas

RDNA2:

- Treat Matrix Core support as SKU/toolkit-dependent and verify before claiming it.
- Use ROCm examples and Composable Kernel examples as references before adding inline MFMA.

CDNA2:

- Keep `gfx90a` records separate from RDNA targets.
- TF32 is a separate math contract.
- LDS staging can matter for shared-memory pipelines.
- BF16 support should be checked against the exact GPU and toolkit.

RDNA3:

- Often strong for inference workloads.
- Do not assume all CDNA2 tuning choices transfer unchanged.
- Compare against MIGraphX, hipBLASLt, Composable Kernel, and framework-generated kernels
  for deployment-shaped problems.

CDNA3:

- MFMA, LDS staging, CK Tile layouts, and HBM/cache behavior change the design space.
- Keep `gfx942` records separate from later CDNA4 or suffix-specific targets.
- Composable Kernel/CK Tile examples are the recommended starting point.

CDNA4/RDNA4:

- Verify exact gfx target and ROCm toolkit support with local tools and
  current AMD/ROCm documentation.
- Low-precision Matrix Core paths may use new instruction families and scaling
  modes.
- Record whether a kernel is portable LLVM IR / AMD GCN ISA, base GFX AMD GCN ISA, or suffix-specific
  AMD GCN ISA.

## Benchmark Protocol

The scaffold harness is `harnesses/rocwmma_gemm_benchmark.hip`.

Example build commands:

```powershell
hipcc -std=c++17 -O3 --offload-arch=gfx90a -DVARIANT_BASELINE `
  harnesses/rocwmma_gemm_benchmark.hip `
  corpus/tasks/rocwmma-mfma-gemm/source/baseline.hip `
  -o build/rocwmma_baseline.exe

hipcc -std=c++17 -O3 --offload-arch=gfx90a -DVARIANT_OPTIMIZED `
  harnesses/rocwmma_gemm_benchmark.hip `
  corpus/tasks/rocwmma-mfma-gemm/source/optimized.hip `
  -o build/rocwmma_optimized.exe
```

Example run shape:

```powershell
.\build\rocwmma_baseline.exe 256 256 256 10 50
.\build\rocwmma_optimized.exe 256 256 256 10 50
```

Every measured record should include:

- GPU name, gfx target, driver, ROCm toolkit, and compiler version.
- `--offload-arch` flags.
- M, N, K, layouts, leading dimensions, dtype, accumulator type, and output
  type.
- Warmup and measured iterations.
- Median, p10, p90, min, and max latency.
- Correctness tolerance, max absolute error, max relative error, and bad count.
- Baselines run and their versions.
- hipBLASLt algorithm and workspace metadata when used.
- Composable Kernel commit, tile shape, stages, scheduler, and epilogue when used.
- Evidence label: `timing-only`, `counter-backed-measured`,
  `profile-attempted-blocked`, or `template-only`.

## Track Files

- `data/index/matrix_core_track.json`: machine-readable track plan.
- `corpus/tasks/rocwmma-mfma-gemm/`: benchmarkable seed task.
- `harnesses/rocwmma_gemm_benchmark.hip`: CPU reference, HIP-event timing, and JSON
  output harness.
- `harnesses/hipblaslt_hgemm_benchmark.hip`: hipBLASLt competitor harness for the
  rocWMMA seed contract.
- `tools/run_library_baseline.py`: build/run entrypoint for library baselines.
- `examples/matrix_cores/`: notes and skeletons for rocWMMA, inline MFMA, and
  Composable Kernel/CK Tile follow-up work.

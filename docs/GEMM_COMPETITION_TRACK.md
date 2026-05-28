# GEMM Competition Track

This track teaches agents to compete with GEMM libraries honestly. The target is
not "call hipBLAS/rocBLAS and stop." The target is to understand what hipBLAS/rocBLAS, hipBLASLt,
Composable Kernel, Triton, and framework compilers are doing well enough to beat, match,
specialize beyond, or extend them for a scoped workload.

GEMM is the spine of many HIP/ROCm optimization problems: MLP projections,
attention score/value products, convolution lowering, MoE experts, LoRA/adapters,
quantized inference, batched small linear algebra, and fused epilogues. The
corpus should preserve both wins and losses, because the boundary where a custom
kernel loses to a library is also useful training data.

## Mission

For every GEMM task, produce an evidence trail that lets an agent answer:

- What exact problem is being solved?
- What is the strongest known baseline for that exact problem?
- What generality does the baseline carry that the custom kernel can ignore?
- What custom attack surface is being tested?
- Did the custom path win, lose, or tie?
- Why did that happen, and what would be tried next?

Vendor libraries in this track are:

- Competitors to beat.
- Correctness oracles.
- Metadata references for math modes, layouts, leading dimensions, epilogues,
  workspaces, and algorithm choices.
- Extension surfaces when direct HIP is not the best integration point.

They are not escape hatches. A library result can be the best measured result,
but the record should still explain the next plausible custom path.

## Result Classes

- `custom-win`: custom HIP is faster for the declared scope.
- `custom-specialized-win`: custom HIP wins only for fixed shapes, layouts, or
  fused work.
- `near-match`: custom HIP is close enough to teach a useful implementation
  pattern.
- `library-win`: library remains faster; record the suspected reason.
- `negative-example`: attempted optimization is neutral or slower.
- `correctness-fail`: output is wrong or numerics do not satisfy the contract.
- `compile-fail`: the implementation does not build on the target environment.
- `timing-only`: timing uses HIP events without Nsight counter evidence.

## Baseline Ladder

Use this ladder to avoid comparing an advanced custom kernel only against an
easy baseline.

1. Host or simple device reference.
   - Purpose: correctness oracle, not performance.
   - Keep dtype conversion and accumulation semantics explicit.

2. Naive custom kernel.
   - One output element per thread.
   - Global-memory loads for every multiply.
   - Useful for teaching indexing, layout, alpha/beta, and epilogue semantics.

3. Shared-memory tiled custom kernel.
   - Cooperative CTA tiles of A and B.
   - Coalesced loads, synchronization, register accumulation.
   - Teaches tile shape, bank conflicts, occupancy, and edge handling.

4. Matrix Core custom kernel.
   - rocWMMA, inline MFMA, CK Tile, or handwritten `mma`/`wgmma` paths where
     applicable.
   - Teaches fragment layout, accumulator type, alignment, and architecture
     specificity.

5. hipBLAS/rocBLAS baseline.
   - Standard GEMM, strided batched GEMM, and `hipblasGemmEx`.
   - Record transposes, leading dimensions, compute type, math mode, and TF32
     policy.

6. hipBLASLt baseline.
   - Descriptor-driven matmul, algorithm search, workspace limits, epilogues,
     and layout control.
   - Record returned algorithm metadata and how many candidates were tested.

7. Composable Kernel baseline or extension.
   - Use the closest Composable Kernel example for datatype, architecture, and epilogue.
   - Record tile shape, stages, schedule, swizzle, cluster shape where relevant,
     and Composable Kernel commit.

8. Triton or framework compiler baseline where relevant.
   - Useful for Python-oriented workloads and generated-kernel comparison.
   - Record meta-parameters, compile boundary, dispatch overhead, and graph
     capture status.

The custom kernel does not need to climb every rung for every task, but the
record should explain which rungs were skipped and why.

## Shape Taxonomy

GEMM performance claims without shape taxonomy are usually misleading. Each task
should declare which family it belongs to.

### Tiny Fixed GEMM

Examples:

- `M,N,K <= 32`
- repeated inner loops in simulation, graphics, geometry, routing, or small
  linear layers

Custom win hypotheses:

- Library launch and descriptor overhead dominate.
- Shape constants enable full unrolling.
- Inputs fit in registers or one CTA.
- Fusion removes a second launch.

Common losses:

- Kernel launch overhead still dominates unless many independent matrices are
  batched into one launch.
- Per-thread scalar work underutilizes the GFX.
- Register-only implementations lose coalescing or occupancy.

### Small Batched GEMM

Examples:

- Many 16x16, 32x32, 64x64 matrices.
- Strided batched linear algebra.
- Attention heads with small per-head dimensions.

Custom win hypotheses:

- One CTA or warp per matrix avoids generic scheduler overhead.
- Known batch stride and alignment simplify indexing.
- Fusing bias, scale, activation, or residual amortizes launch overhead.

Baselines:

- `hipblasGemmStridedBatchedEx`
- hipBLASLt batched matmul
- Composable Kernel batched/grouped GEMM

### Medium Square GEMM

Examples:

- 128x128x128 through 2048x2048x2048.

Custom win hypotheses:

- Mostly educational unless using Matrix Cores well.
- Narrow layouts or fused epilogues can still win.
- Fixed dimensions may let a custom tile beat a general heuristic for one GPU.

Common losses:

- hipBLASLt/Composable Kernel Matrix Core kernels are extremely strong.
- Shared-memory FP32 kernels usually lose to Matrix Core baselines when the
  dtype contract allows Matrix Cores.

### Large Square GEMM

Examples:

- 4096x4096x4096 and larger.

Custom win hypotheses:

- Split-K, stream-K, persistent scheduling, or architecture-specific mainloops.
- Custom epilogue avoids large extra memory passes.

Common losses:

- Library kernels have mature scheduling and memory pipelines.
- Poor split-K reduction or epilogue stores erase any mainloop gains.

### Tall-Skinny and Skinny-Wide GEMM

Examples:

- `M >> N`, `N >> M`, or very small K.
- Projection tails, reductions expressed as GEMM, ranking/scoring paths.

Custom win hypotheses:

- Library tiles waste work on skinny dimensions.
- Warp-specialized or CTA-specialized kernels reduce idle lanes.
- Layout-aware vector loads beat generic transpose handling.

Baselines:

- hipBLASLt with multiple algorithms and workspace limits.
- Composable Kernel kernels tuned for tall/skinny shapes.

### Attention Score and Value GEMM

Examples:

- `Q * K^T` with scale/mask/softmax nearby.
- `P * V` where P is the attention probability matrix.

Custom win hypotheses:

- Full attention fusion avoids materializing intermediate matrices.
- Known head dimension, sequence length, causal mask, or block mask narrows the
  problem beyond GEMM.
- FlashAttention-style online softmax changes the memory traffic model.

Baselines:

- hipBLAS/rocBLAS/hipBLASLt GEMM plus separate softmax/value steps.
- FlashAttention or framework attention kernels when the task expands beyond
  pure GEMM.

### Grouped GEMM and MoE

Examples:

- Different expert shapes per token group.
- Many small or medium GEMMs with irregular batch counts.

Custom win hypotheses:

- Group metadata is known or can be pre-sorted.
- Persistent scheduling and work queues reduce load imbalance.
- Fusion with token permutation/unpermutation avoids memory traffic.

Baselines:

- Composable Kernel grouped GEMM.
- hipBLASLt grouped or batched paths where available.
- Framework/MoE library kernels.

### Quantized GEMM

Examples:

- int8, int4, FP8, block-scaled formats.
- Fused dequantization, matrix multiply, bias, activation, requantization.

Custom win hypotheses:

- Scale layout and group size are fixed.
- Dequantization can be fused into the mainloop or epilogue.
- Output type and clamp policy are narrower than library options.

Baselines:

- hipBLASLt quantized matmul where supported.
- Composable Kernel int8/FP8 examples.
- MIGraphX engine/plugin baselines for inference deployment.

## Layout and Contract Checklist

Every GEMM task should specify:

- Logical equation: `D = epilogue(alpha * op(A) * op(B) + beta * C)`.
- Matrix order used by the custom kernel: row-major, column-major, or tensor
  layout.
- Library mapping: whether row-major wrappers invert operand order for hipBLAS/rocBLAS.
- `M`, `N`, `K`, batch count, group count, and split-K policy.
- Transpose/conjugate flags.
- Leading dimensions and batch strides.
- Alignment assumptions for A, B, C, D, and auxiliary tensors.
- Contiguity and permitted padding.
- Aliasing rules between C and D.
- Alpha and beta semantics.
- Accumulator type and output type.
- Math mode: FP32, TF32, FP16 accumulation, BF16, FP8, int8, or mixed.
- Denormal, saturation, clamp, rounding, NaN, and infinity behavior when
  relevant.
- Correctness tolerance per dtype and shape.

## Metadata Requirements

Attach this metadata to measured records:

- Task id and source file.
- Result class.
- GPU name, gfx target, GFX count, clocks if known, memory size.
- Driver, ROCm toolkit, and relevant library versions.
- Build command, compiler version, `-arch`/`-code`, and important macros.
- Problem shape, layout, strides, dtype, compute type, and epilogue.
- Baselines run and their versions.
- hipBLAS/rocBLAS math mode or hipBLASLt compute type.
- hipBLASLt workspace cap, algorithm count queried, algorithm chosen, and
  heuristic result metadata when available.
- Composable Kernel commit, example ancestor, tile shape, stages, scheduler, and arch.
- Triton meta-parameters and compile boundary when used.
- Warmup count, measured iterations, timer type, synchronization points, and
  whether HIP Graph capture was used.
- Correctness method, tolerance, seed, and reference implementation.
- Profiler status: `timing-only`, `rocprof-blocked`, or counter set name.
- Notes explaining the win/loss hypothesis.

Do not invent profiler counter evidence. If only HIP-event timings are
available, mark the record `timing-only`.

## Comparison Rules

- Compare the same math contract. TF32 versus FP32 is a different task unless
  the task explicitly studies that tradeoff.
- Compare the same epilogue. A custom fused bias+ReLU kernel should compare
  against hipBLASLt bias+ReLU where supported, not plain hipBLAS/rocBLAS alone.
- Separate engine build, descriptor creation, allocation, and steady-state
  timing unless the workload needs end-to-end latency.
- For tiny GEMMs, measure both one-off launch latency and many-matrix batched
  throughput.
- For dynamic-shape inference, record whether algorithm selection or engine
  building is included.
- Avoid best-of timing. Store median and spread when the harness supports it.
- Keep losing attempts. They prevent future agents from rediscovering dead
  ends.

## Custom Kernel Attack Surfaces

### Shape Specialization

Replace runtime dimensions with compile-time constants. This enables unrolling,
fixed CTA shapes, fixed epilogue vector width, and simpler boundary handling.

Good targets:

- Tiny fixed GEMM.
- Fixed MLP projection sizes.
- Fixed attention head dimensions.
- Known batch count or group sizes.

Risk:

- Over-specialization creates too many kernels.
- Instruction cache pressure or compile time can become part of the real cost.

### Layout Specialization

Exploit known row-major, column-major, interleaved, padded, or transposed
layouts.

Good targets:

- Avoiding explicit layout transform before or after GEMM.
- Fusing transpose into loads/stores.
- Fixed leading dimensions with alignment guarantees.

Risk:

- A library with the correct layout descriptor may already do this well.
- Wrong leading dimensions can pass square tests and fail rectangular tests.

### Fusion and Epilogues

Fuse post-GEMM work:

- Bias.
- ReLU, GELU, SiLU, clamp.
- Residual add.
- Scale and quantize.
- Dequantize operands.
- Store auxiliary output such as pre-activation or amax.
- Layout conversion on output.

Baselines:

- hipBLASLt epilogue if supported.
- Composable Kernel custom epilogue.
- MIGraphX plugin for deployment contexts.

Risk:

- Epilogue register pressure can slow the mainloop.
- Unsupported broadcasts or alignment assumptions can hide correctness bugs.

### Matrix Core Mainloops

Use rocWMMA, MFMA, WGMFMA, Composable Kernel, or CK Tile to target Matrix Cores.

Good targets:

- FP16, BF16, TF32, FP8, int8.
- Shapes aligned to instruction tile requirements.
- Large enough K to amortize setup.

Risk:

- Fragment layouts are easy to misuse.
- Accumulator and conversion choices can change numerical results.
- A scalar or shared-memory FP32 kernel should not be compared to Matrix Core
  library results unless the math contract allows the difference.

### Scheduling

Tune how work maps to GFXs:

- CTA tile shape.
- Warp tile shape.
- Split-K or stream-K.
- Persistent CTAs.
- Grouped work queues.
- CDNA3/CDNA4/RDNA4 clusters and global-to-LDS staging/WGMFMA where applicable.

Risk:

- The best schedule is architecture-specific.
- More parallelism can create expensive reductions.
- Larger tiles can lose to register pressure or occupancy limits.

## hipBLAS/rocBLAS and hipBLASLt Track Notes

For hipBLAS/rocBLAS:

- Use it as the plain GEMM bar.
- Record row-major wrapper choices.
- Record TF32 policy explicitly.
- Use strided batched APIs for repeated same-shape GEMM.

For hipBLASLt:

- Treat descriptors as part of the task definition.
- Query multiple algorithms when possible.
- Vary workspace caps intentionally.
- Record epilogue support and whether it matches the custom epilogue.
- Keep descriptor creation outside steady-state timing unless end-to-end latency
  is the target.

hipBLASLt is often the strongest baseline for fused GEMM epilogues. If a custom
kernel beats plain hipBLAS/rocBLAS but loses to hipBLASLt, record that as a useful
boundary, not as a failed corpus entry.

## Composable Kernel and CK Tile Track Notes

Composable Kernel is both a competitor and a parts bin. A Composable Kernel-based kernel counts as
custom competition when the agent makes and records meaningful choices:

- Datatype and accumulator policy.
- Layouts and alignment.
- CTA, warp, and instruction tile shapes.
- Pipeline stages and copy atom.
- Swizzle, scheduler, or persistent strategy.
- Custom epilogue visitor or output operator.
- Architecture path such as CDNA2 `global-to-LDS staging`, CDNA3 global-to-LDS staging/WGMFMA, or future
  CDNA4/RDNA4-specific paths.

Recommended workflow:

1. Reproduce a nearby Composable Kernel example unchanged.
2. Replace only the problem shape and dtype.
3. Add the host reference and metadata capture.
4. Change the epilogue.
5. Tune tile shape and schedule.
6. Compare against hipBLASLt with the closest equivalent epilogue.
7. Record what remains library-general and what is custom-specialized.

## Planned Task Families

The machine-readable index for this section is
`data/index/gemm_competition_track.json`.

Minimum task families:

- `sgemm-naive-shared`: FP32 naive versus shared-memory tiled GEMM.
- `sgemm-fixed-small`: tiny fixed-shape GEMM where launch overhead and
  specialization matter.
- `tf32-policy-study`: hipBLAS/rocBLAS/hipBLASLt TF32 on/off and custom FP32 contract.
- `hgemm-tensor-core`: FP16/BF16 Matrix Core custom path versus hipBLASLt and
  Composable Kernel.
- `gemm-bias-relu-epilogue`: custom fused epilogue versus hipBLASLt epilogue.
- `gemm-custom-unsupported-epilogue`: Composable Kernel/custom HIP path for epilogues
  hipBLASLt cannot express.
- `strided-batched-small`: same-shape batched GEMM.
- `grouped-gemm-moe`: variable-shape grouped GEMM for MoE.
- `attention-qk-score`: GEMM as part of attention score computation.
- `quantized-gemm-dequant-epilogue`: int8/FP8/block-scaled path with fused
  scale/dequant/quant behavior.

## Losing Examples to Preserve

Keep examples like these with clear labels:

- Shared-memory FP32 kernel loses to hipBLASLt TF32 because the contracts differ.
- Naive vectorized loads are slower due to misalignment or register pressure.
- A custom bias+ReLU kernel beats plain hipBLAS/rocBLAS plus separate ReLU but loses to
  hipBLASLt epilogue.
- A tiny GEMM kernel wins for one matrix but loses when batched poorly.
- A Composable Kernel tile shape improves one square size but loses on tall-skinny shapes.
- Split-K improves GFX occupancy but the reduction overhead dominates.
- Matrix Core implementation is fast but fails tolerance because accumulation or
  rounding semantics differ.

## Agent Workflow

1. Read the task record in `data/index/gemm_competition_track.json`.
2. Write down the exact GEMM contract before coding.
3. Implement or select the reference result.
4. Implement the simplest custom kernel.
5. Add the strongest relevant library baseline.
6. Add the targeted custom optimization.
7. Measure with the same inputs and the same synchronization discipline.
8. Classify the result.
9. Record the next hypothesis, even for losses.

An agent should prefer a narrow, correctly measured custom result over a broad
claim. GEMM libraries are hard to beat; the corpus wins by making the contest
precise.


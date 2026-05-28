# Quantization Library Baselines

This guide records the library and framework baselines that must be considered
before claiming a custom win for `fused-int4-dequant-gemv`.

The goal is not to make vendor libraries an escape hatch. The goal is to make
the comparison honest enough that a custom HIP kernel can still win by
specializing the exact shape, layout, scale metadata, output contract, or
deployment boundary.

## Seed Contract

The current task contract is:

| Field | Required value |
| --- | --- |
| Operation | `y[row] = sum_k x[k] * ((q4(row,k) - zp(row,group(k))) * scale(row,group(k)))` |
| Weights | `uint8_t packed_weights[rows, ceil(cols / 2)]`, row-major |
| Packing order | even `k` in low nibble, odd `k + 1` in high nibble |
| Quantized value | unsigned 4-bit integer, `q4 in [0, 15]` |
| Group index | `group(k) = k / group_size` |
| Group count | `groups_per_row = ceil(cols / group_size)` |
| Scale layout | FP32, row-major `[rows, groups_per_row]` |
| Zero-point layout | one byte per `[row, group]`; semantic value is `zero_points[index] & 0x0f` |
| Dequant contract | affine unsigned int4, `(q4 - zero_point) * scale` |
| Activation | FP32 vector `[cols]` |
| Accumulator | FP32 in the CUDA seed, FP64 in the CPU oracle before final FP32 cast |
| Output | FP32 vector `[rows]` |
| Seed tolerance | `max_abs_error <= 5.0e-3` or `max_rel_error <= 5.0e-4` |

Do not silently change any of these fields when comparing against a library.
If a baseline uses NF4, FP4, signed INT4, INT8, FP8, block-scaled FP4, packed
zero-points, per-channel scale, per-token activation scale, FP16 output, BF16
output, or a different nibble order, record it as a different contract.

## Fairness Gate

A direct library comparison is fair only when all of these match:

- Shape: same `rows`, `cols`, `group_size`, and group-tail handling.
- Packing: same byte address, nibble order, and odd-`cols` high-nibble policy.
- Scale metadata: same dtype, row/group indexing, broadcast axis, and group
  tail behavior.
- Zero-point metadata: same semantic value, storage layout, signedness, and
  masking rule.
- Math: same dequant formula, accumulator dtype, output dtype, bias/activation
  epilogue, and rounding/cast policy.
- Tolerance: same correctness oracle and thresholds, or a separately documented
  tolerance for a deliberately different dtype contract.
- Timing boundary: same input residency, allocation policy, synchronization,
  warmup, measured iterations, and whether framework dispatch, graph capture,
  engine build, conversion, or dequantization is included.

If any item differs, use a partial or not-applicable label. A custom kernel that
beats only a materialized dequant path is useful corpus data, but it is not a
general library win.

## Result Labels

Use these labels in notes and result metadata:

| Label | Meaning |
| --- | --- |
| `custom-win-fair` | Custom kernel beats the closest direct library path under the same contract. |
| `custom-specialized-win-fair` | Custom kernel beats a fair library path for a narrower fixed shape, layout, epilogue, or timing boundary. |
| `near-match-fair` | Custom and library timings are within measurement noise under the same contract. |
| `library-win-fair` | Library path wins under the same contract. Keep the result and record why. |
| `partial-win-materialized-dequant-only` | Custom beats dequantize-to-FP16/FP32 plus GEMV/GEMM, but no direct low-bit competitor was measured. |
| `partial-win-format-conversion` | Custom beats a baseline that required changing quantization format, packing, or metadata layout. |
| `partial-win-framework-overhead-only` | Custom wins only when framework dispatch, allocation, conversion, or extra launches are included. |
| `library-win-different-contract` | Library is faster, but the format or numerical contract differs, so it is not the exact seed bar. |
| `not-applicable` | The library path cannot express the seed contract without material conversion or custom code. |
| `negative-example` | Attempted baseline or custom path is slower, unsupported, or numerically mismatched, and the reason is recorded. |
| `correctness-fail` | Timing is invalid because output did not satisfy the declared tolerance. |

## Baseline Order

For `fused-int4-dequant-gemv`, compare in this order when a GPU is available:

1. CPU oracle with the seed contract.
2. Materialized dequant baseline.
3. Existing custom seed kernels.
4. bitsandbytes when its 4-bit contract is relevant.
5. Composable Kernel/CK Tile low-bit or block-scaled GEMV/GEMM paths.
6. hipBLASLt where the dtype and scale contract is expressible.
7. MIGraphX or vLLM on ROCm plugin/engine path for deployment inference.
8. Transformer Engine on ROCm for FP8, MXFP8, or NVFP4/block-scaled references.
9. Triton and framework-generated kernels.

The strongest applicable direct low-bit path is the fair comparison. Earlier
baselines are still useful for teaching and regression testing.

## Materialized Dequant Baseline

This baseline dequantizes `packed_weights` into a temporary dense matrix, then
calls GEMV/GEMM through hipBLAS/rocBLAS, hipBLASLt, PyTorch, or another framework.

Required contract:

- Read `packed_weights[row * packed_cols + (k >> 1)]`.
- Extract `q4` from low nibble for even `k`, high nibble for odd `k`.
- Read `scale = scales[row * groups_per_row + k / group_size]`.
- Read `zp = zero_points[row * groups_per_row + k / group_size] & 0x0f`.
- Compute `(q4 - zp) * scale` into FP32 or the explicitly chosen temporary dtype.
- Preserve row-major logical weights unless the downstream library call records
  a transpose or leading-dimension conversion.

Record:

- Temporary dtype: FP32 or FP16/BF16.
- Temporary layout and leading dimensions.
- Temporary bytes.
- Dequant kernel timing, GEMV/GEMM timing, and total timing.
- Whether allocation and conversion are inside the timed region.
- GEMV/GEMM backend and version.
- Same seed tolerance for FP32 temporary; separate tolerance for FP16/BF16
  temporary.

Use `partial-win-materialized-dequant-only` when a custom fused kernel beats
this path but no matching direct low-bit library path has been measured.

## bitsandbytes

bitsandbytes is a practical low-bit reference, especially for `Linear4bit`,
`matmul_4bit`, and `gemv_4bit`. Its common 4-bit contracts are FP4 and NF4
codebook quantization with `absmax` block metadata. That is not the same as the
seed affine unsigned INT4 contract unless an experiment explicitly proves the
values, block metadata, output dtype, and tolerance match.

Use bitsandbytes in two modes:

- Direct competitor for a bitsandbytes-compatible task, such as NF4/FP4
  `Linear4bit` or `gemv_4bit`.
- Format-family reference for the seed task, with `partial-win-format-conversion`
  or `not-applicable` if the affine zero-point contract cannot be represented.

Record:

- API: `Linear4bit`, `bnb.matmul_4bit`, `F.gemv_4bit`, or explicit
  dequantization.
- `quant_type`: `fp4` or `nf4`; do not call this seed INT4 unless the codebook
  and affine mapping are documented.
- `blocksize` and whether `cols % blocksize == 0`.
- `absmax` shape and dtype.
- `code` / codebook identity.
- `compress_statistics` and nested quantization state.
- Activation dtype and output dtype. `gemv_4bit` returns the activation dtype.
- Weight orientation. The fast path may call `gemv_4bit(A, B.t(), ...)`.
- Bias handling and whether bias timing is included.
- Fallback behavior for unsupported shape or dtype.
- Correctness tolerance used for the bitsandbytes numerical contract.

For the current seed, bitsandbytes is fair only if the benchmark either
implements the exact affine uint4 row/group contract through a documented path,
or changes the task contract to NF4/FP4 and records that change. Otherwise, keep
it as a strong partial/reference baseline rather than a fair library bar.

Local references:

- `third_party/bitsandbytes/agents/api_surface.md`
- `third_party/bitsandbytes/bitsandbytes/backends/cuda/ops.py`
- `third_party/bitsandbytes/bitsandbytes/autograd/_functions.py`

## Composable Kernel And CK Tile Low-Bit Paths

Composable Kernel/CK Tile is both a competitor and an extension surface. It is the right
place to study int4/uint4 MFMA layouts, FP8 kernels, block-scaled FP4/NVFP4, and
custom epilogues. It is also custom-kernel work when an agent changes tile
shape, layout, iterator, scale loading, epilogue, or scheduler.

Potential baselines and references:

- INT4/UINT4 Matrix Core GEMM paths in Composable Kernel/CK Tile.
- `examples/91_fp4_gemv/91_fp4_gemv.hip` for block-scaled FP4 GEMV.
- `examples/94_ada_fp8_blockwise/ada_fp8_blockwise.hip` for RDNA3 FP8 blockwise
  GEMM.
- `examples/87_blackwell_geforce_gemm_blockwise/` for CDNA4/RDNA4 FP8 blockwise
  and groupwise GEMM.
- `examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.hip`
  for NVFP4 block-scaled GEMM.

Record:

- Composable Kernel commit and example ancestor.
- Architecture target, such as `gfx90a`, `gfx1100`, `gfx950`, `gfx950`, or
  `gfx1200`.
- Operand types: `int4b_t`, `uint4b_t`, FP8 type, FP4/NVFP4 type, or block
  scaled wrapper.
- Logical layouts and internal tensor layouts.
- Packing order and whether it matches the seed byte/nibble order.
- Scale factor dtype, vector size, block shape, preshuffled layout, and
  broadcast axis.
- Zero-point support. If absent, record `zero_point_layout = none`.
- Accumulator dtype and epilogue compute dtype.
- Output dtype, output layout, and epilogue.
- CTA/warp/instruction tile shapes, stages, scheduler, workspace, and alignment.
- Whether the path is a direct library call, a modified Composable Kernel custom kernel,
  or a reference implementation.

For the current affine uint4 GEMV seed, a stock block-scaled FP4 path is not a
fair direct baseline because the numerical format and scale layout differ. It is
still an important competitor for future FP4/block-scaled tasks and an extension
surface for a custom row/group affine dequant epilogue.

## hipBLASLt Where Applicable

Use hipBLASLt for matmul baselines when the low-bit or dequantized contract can
be expressed through hipBLASLt descriptors. For the current packed affine INT4
GEMV seed, hipBLASLt is normally applicable as:

- Materialized FP32/FP16/BF16 dequant plus GEMV/GEMM.
- INT8, FP8, or block-scaled matmul baseline when the task contract is changed
  to a supported hipBLASLt datatype and scale layout.

Do not claim a fair hipBLASLt INT4 comparison if the library path cannot consume
the packed seed weights and per-row/per-group zero-points directly.

Record:

- hipBLAS/rocBLAS/hipBLASLt version, ROCm toolkit, driver, and GPU.
- `hipblasLtMatmulDesc_t` compute type and scale type.
- Matrix layouts, leading dimensions, transposes, batch count, and order.
- Alpha/beta, pointer mode, math mode, and TF32 policy.
- Workspace cap, heuristic count, selected algorithm, and timing cache if used.
- Epilogue attributes and whether they match the custom epilogue.
- Direct quantized path versus materialized dequant path.
- Conversion and allocation timing boundary.

Use `partial-win-materialized-dequant-only` for wins against dequant plus
hipBLASLt. Use `custom-win-fair` only when the hipBLASLt path matches the declared
dtype, scale, zero-point, accumulator, output, and epilogue contract.

## MIGraphX And vLLM on ROCm

MIGraphX and vLLM on ROCm are deployment baselines. They answer a different
question from an isolated kernel benchmark: can a model engine, plugin, tactic,
or serving runtime beat the custom path once batching, KV cache, engine fusion,
and runtime overhead are included?

Baseline types:

- MIGraphX engine using supported quantized layers or explicit dequant plus
  matmul layers.
- MIGraphX plugin that wraps a custom fused dequant GEMV kernel.
- vLLM on ROCm checkpoint and engine using quantization such as `int4_awq`,
  `int4_wo`, `w4a8_awq`, `fp8`, or `nvfp4` where supported by the local
  version and GPU.
- vLLM on ROCm plugin or kernel path for weight-only linear, MoE, attention, or
  KV-cache quantization.

Record:

- MIGraphX and vLLM on ROCm versions and repository commits.
- Model or synthetic layer, checkpoint format, and conversion command.
- Quantization algorithm, group size, AWQ/GPTQ/weight-only flags,
  calibration data, and whether zero-points exist.
- Weight packing order, scale layout, zero-point layout, activation scale
  layout, accumulator dtype, and output dtype.
- Engine build command, precision flags, dynamic profiles, timing cache,
  workspace, plugin flags, and selected tactic if available.
- Runtime batch size, sequence lengths, tokens, scheduler policy, KV-cache
  dtype/layout, and whether in-flight batching is active.
- Whether timing is isolated plugin `enqueue`, engine layer timing, engine
  runtime, or end-to-end serving.

For the seed task, vLLM on ROCm is fair only when the engine or plugin execk_tiles
the same affine uint4 row/group contract and reports an equivalent FP32 output
or explicitly declared output contract. End-to-end serving wins are valuable,
but label them as deployment wins instead of isolated kernel wins.

Local references:

- `third_party/migraphx`
- `third_party/vllm/examples/quantization/README.md`
- `third_party/vllm/cpp/micro_benchmarks/`
- `third_party/vllm/cpp/migraphx_llm/plugins/`

## Transformer Engine on ROCm FP8 And Block-Scaled References

Transformer Engine on ROCm is not a direct affine INT4 GEMV baseline for the seed task.
It is the primary local reference for transformer FP8, MXFP8, NVFP4, scale/amax
state, block-scaled linear layers, and framework integration.

Use it when the task is changed to:

- FP8 transformer linear.
- FP8 per-tensor, per-channel, per-token, or block-scaled GEMM.
- MXFP8 or NVFP4 block-scaled linear on supported CDNA4/RDNA4 paths.
- Training or inference paths where scale update, amax history, and recipe
  semantics are part of correctness.

Record:

- Transformer Engine on ROCm commit and framework binding.
- Recipe name and format: FP8 E4M3/E5M2/HYBRID, MXFP8, NVFP4, or block-scaled
  FP8.
- Scale dtype, scale layout, amax history, update policy, and delayed/current
  scaling behavior.
- Activation and weight quantizers, including rowwise/columnwise usage.
- Accumulator dtype, output dtype, bias dtype, and any cast restrictions.
- Hardware availability checks and reason when a recipe is skipped.
- Whether timing includes framework autocast, quantizer update, cached FP8
  weights, or only the GEMM kernel.

For `fused-int4-dequant-gemv`, label Transformer Engine on ROCm as `not-applicable` or
`partial-win-format-conversion` unless the experiment is explicitly reframed as
an FP8 or block-scaled task.

Local references:

- `source:transformer-engine/README.rst`
- `source:transformer-engine/benchmarks/linear/benchmark_linear.py`
- `source:transformer-engine/benchmarks/linear/benchmark_grouped_linear.py`

## Triton And Framework Baselines

Triton and framework-generated kernels are useful because they can implement the
same seed contract quickly and can expose whether HIP C++ is winning on kernel
quality or only on integration overhead.

Fair Triton baseline:

- Implement the same byte/nibble unpack.
- Use the same row/group scale and zero-point indexing.
- Accumulate and output under the same dtype contract.
- Use the same shape and tolerance.
- Record `BLOCK_*`, `num_warps`, `num_stages`, vectorization, masks, and whether
  `tl.dot`, scalar multiply-add, or block-scaled dot instructions are used.

Framework baselines:

- PyTorch eager materialized dequant plus `torch.matmul`.
- `torch.compile` or Inductor output for the materialized or fused expression.
- Framework extension or custom op path.
- bitsandbytes or Transformer Engine on ROCm through their framework modules.

Record:

- Framework, Triton, compiler, and ROCm versions.
- Dispatch boundary: eager, compiled, HIP Graph, extension op, or raw kernel.
- Compilation time versus runtime.
- Allocation and conversion timing boundary.
- Generated LLVM IR / AMD GCN ISA only when inspected; do not invent compiler evidence.

Use `custom-win-fair` for a HIP C++ win over a Triton kernel only when the
Triton kernel implements the same contract. Use
`partial-win-framework-overhead-only` when HIP C++ wins because the framework
path includes dispatch, allocation, or materialized intermediates that are not
part of the isolated kernel boundary.

Local references:

- `third_party/triton/python/tutorials/03-matrix-multiplication.py`
- `third_party/triton/python/tutorials/10-block-scaled-matmul.py`
- `docs/TRITON_KERNEL_GUIDE.md`
- `docs/FRAMEWORK_EXTENSION_GUIDE.md`

## Required Record Template

Every measured baseline record should include:

```json
{
  "task_id": "fused-int4-dequant-gemv",
  "baseline_id": "materialized-dequant | bitsandbytes | composable-kernel-ck_tile | hipblaslt | migraphx | vllm-rocm | transformer-engine | triton | framework-generated",
  "result_label": "partial-win-materialized-dequant-only",
  "evidence_label": "timing-only",
  "hardware": {
    "gpu_name": "",
    "gfx_target": "",
    "driver_version": "",
    "rocm_version": ""
  },
  "shape": {
    "rows": 4096,
    "cols": 4096,
    "group_size": 128
  },
  "contract": {
    "packing_order": "even k low nibble, odd k high nibble",
    "scale_layout": "fp32 row-major [rows, ceil(cols / group_size)]",
    "zero_point_layout": "uint8 storage, low nibble semantic value per [row, group]",
    "accumulator_dtype": "fp32",
    "output_dtype": "fp32",
    "tolerance": {
      "max_abs_error": 0.005,
      "max_rel_error": 0.0005
    }
  },
  "timing_boundary": {
    "timer_type": "cuda-event",
    "warmup_iterations": 10,
    "measured_iterations": 50,
    "allocation_included": false,
    "conversion_included": false,
    "dispatch_included": false
  },
  "notes": "Say timing-only when no profiler counters are attached."
}
```

No result should claim profiler-counter evidence unless rocprofiler/rocprof artifacts are
attached. No result should claim a fair low-bit library win or loss until the
contract fields above are filled in.

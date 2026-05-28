# PyTorch Baseline HIP/ROCm Challenge

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This challenge turns public PyTorch APIs and real model problem sizes into
HIP/ROCm optimization tasks. PyTorch is the baseline and oracle. A custom kernel is
successful when it preserves the public contract and improves a clearly timed
boundary on the same hardware.

The challenge is intentionally shaped like KernelBench and GPU Mode reference
kernels, but with this corpus' evidence discipline and model-real shapes.

## Baseline Ladder

| Baseline ID | Meaning | Required metadata |
| --- | --- | --- |
| `pytorch-eager-public-api` | Direct PyTorch function/module implementation of the task contract | PyTorch version, ROCm/runtime version, dtype, layout, stream, timing boundary |
| `pytorch-compile-default` | Same PyTorch reference wrapped with `torch.compile` using default settings | compile mode, warmup policy, graph breaks, generated backend if known |
| `pytorch-compile-specialized` | `torch.compile` with task-approved static shapes or mode flags | all specialization assumptions and compile overhead policy |
| `pytorch-library-backed` | Public PyTorch op that dispatches to hipBLAS/rocBLAS, MIOpen, rocFFT, rocSPARSE, or another backend | actual backend path if known, library versions when observable |
| `strongest-external-competitor` | Optional non-PyTorch baseline such as hipCUB, Composable Kernel, hipBLASLt, MIGraphX, vLLM, FlashInfer | exact API, workspace, allocation, launch count, and contract differences |

The primary challenge score is against `pytorch-eager-public-api` unless the
task explicitly declares `pytorch-compile-default` as the primary baseline.
External competitors are recorded to prevent weak wins from looking stronger
than they are.

Current ROCm status: no PyTorch challenge result artifacts are checked in yet.
The CUDA-origin source corpus had seed timings for `rowwise-softmax`; treat
those as historical context only and rerun on AMD hardware before scoring a
ROCm challenge result.

## Task Levels

| Level | Contract | Example |
| --- | --- | --- |
| L0 single public op | Replace one PyTorch API call | `torch.topk`, `torch.sum`, `torch.nn.functional.rms_norm` |
| L1 shape-specialized public op | Same semantics, fixed shape/dtype/layout subset | rowwise softmax for `[M,4096]`, top-k for `V=151936,k=20` |
| L2 public subgraph | Replace a sequence of PyTorch ops with one or more custom kernels | residual + RMSNorm, router softmax + top-k + normalize |
| L3 model block | Replace a model-real block with a HIP extension path | VAE conv block, transformer attention block, MoE routing block |
| L4 serving step | Replace a repeated inference step | decode attention + KV update + logits Top-K |

Higher levels may drop more generality, but the dropped generality must be
declared and scored as a specialization rather than hidden behavior.

## Real-Shape Admission Gate

Every public PyTorch API contract used by a task in this challenge must cite at
least one row from `docs/PYTORCH_API_REAL_WORLD_SHAPE_MATRIX.md` through a
`real_shape_matrix_refs` field. The broad public API catalog is only a candidate
map. If an API is `catalog-only` or `blocked-until-sourced` in the matrix, gather
a real model size before turning it into a scored task.

## Result Classification

| Class | Rule |
| --- | --- |
| `correctness-fail` | Output, shape, dtype, exception, or semantics do not match the task contract |
| `compile-fail` | Submission cannot build/load for the target environment |
| `measurement-invalid` | Timing includes forbidden work, misses synchronization, or lacks required baseline metadata |
| `loss` | Correct but materially slower than the primary PyTorch baseline |
| `neutral` | Correct and within the task equivalence band, usually +/- 5 percent |
| `match` | Correct and performance-equivalent while reducing some other cost such as memory or launch count |
| `win` | Correct and faster than the primary PyTorch baseline without undeclared specialization |
| `specialized-win` | Correct and faster because fixed shapes, layout, dtype, or narrower semantics were declared |
| `negative-example` | Correct but slower/neutral and educational enough to archive |

## Scoring

Correctness gates performance. A failed correctness check receives score 0 for
that shape.

Per-shape score is 100 points:

| Component | Points | Rule |
| --- | ---: | --- |
| Correctness | 40 | Shape, dtype, values, edge cases, and task-specific semantics pass |
| Performance | 45 | Based on `speedup = pytorch_baseline_ms / candidate_ms` |
| Evidence | 10 | Hardware/toolchain/timing metadata, warmups, p10/median/p90, launch/allocation flags |
| Generality discipline | 5 | Declared dropped generality, no hidden shortcuts, clear unsupported cases |

Default performance points:

| Speedup vs primary baseline | Performance points |
| ---: | ---: |
| `< 0.75x` | 0 |
| `0.75x` to `< 0.95x` | 5 |
| `0.95x` to `< 1.05x` | 10 |
| `1.05x` to `< 1.25x` | 20 |
| `1.25x` to `< 1.50x` | 30 |
| `1.50x` to `< 2.00x` | 38 |
| `>= 2.00x` | 45 |

Task authors may set a different `target_speedup` band when PyTorch already
dispatches to a highly tuned library. For example, beating `torch.mm` backed by
hipBLAS/rocBLAS by 1.10x on a narrow fixed shape can be a strong specialized result,
while beating a chain of five pointwise ops by only 1.10x may be weak.

Aggregate suite score:

```text
visible_score = mean(per_shape_score for visible shapes)
hidden_score = mean(per_shape_score for hidden shapes)
suite_score = 0.60 * hidden_score + 0.40 * visible_score
```

When hidden shapes are not available yet, report `visible-only` and do not rank
the task as complete.

## Correctness Policies

| Family | Default policy |
| --- | --- |
| Pointwise | exact for integer/bool, tolerance for floating point, broadcast semantics preserved |
| Reductions | tolerance with fp32 accumulation expectation when PyTorch uses it; NaN behavior specified |
| Arg reductions and Top-K | exact indices/values under declared tie policy; sortedness and stable/unstable behavior specified |
| Sampling | exact seed replay only when RNG contract is supplied; otherwise distributional tests plus probability mass checks |
| Scatter/atomic updates | duplicate-index policy and tolerated non-associativity specified |
| Attention | compare output tensor against PyTorch SDPA/eager path under dtype tolerance; masks and causal flags explicit |
| GEMM/conv | tolerance by dtype and accumulation mode; epilogue math order documented |
| Sparse | layout invariants, coalesced state, and compressed index semantics checked |

## Timing Boundary

Every result must name one timing boundary:

- `kernel_only`: one or more raw HIP kernels, no API dispatch or allocation.
- `steady_state_api`: PyTorch/custom extension call after warmup and allocation.
- `api_with_allocation`: includes workspace/output allocation by design.
- `graph_replay`: HIP graph replay only; graph capture cost excluded.
- `framework_call`: public PyTorch call including framework dispatch.
- `serving_step`: full decode/sampling step with declared state updates.

Record whether output allocation, workspace query/allocation, `contiguous`
copies, random generation, and host-device synchronization are included.

## Challenge Tracks

| Track | PyTorch API contract | Real shape matrix refs | First tasks | Admission |
| --- | --- | --- | --- | --- |
| `torch_pointwise_fusion` | pointwise APIs, `where`, activations | `pt_pointwise_cfg_scheduler`, `pt_layout_latent_pack_concat` | fused scale/shift/clip, residual activation | `challenge-admitted` |
| `torch_reduction_norm` | `sum`, `amax`, `layer_norm`, `rms_norm`, `softmax` | `pt_reduction_norm_rows`, `pt_softmax_attention_rows` | RMSNorm, masked softmax, logsumexp | `challenge-admitted` |
| `torch_selection_sampling` | `topk`, `sort`, `multinomial` | `pt_selection_sampling_rows` | Top-K, Top-P, router top-k | `challenge-admitted` |
| `torch_gather_scatter_moe` | `gather`, `index_select`, `scatter_add`, `nonzero` | `pt_gather_scatter_compaction`, `pt_selection_sampling_rows` | dedup gather, grouped scatter-add | `challenge-admitted` |
| `torch_dense_linear` | `matmul`, `addmm`, `linear`, `bmm` | `pt_dense_linear_gemm_rows` | fixed-shape GEMM/GEMV, custom epilogue | `challenge-admitted` |
| `torch_attention` | `scaled_dot_product_attention`, MHA modules | `pt_softmax_attention_rows`, `pt_dense_linear_gemm_rows` | GQA decode, joint attention, fixed noncausal image attention | `challenge-admitted` |
| `torch_position_scan_rope` | `cumsum`, `arange`, `sin`, `cos`, RoPE/position helpers | `pt_position_scan_rope_rows` | Allegro 3D RoPE apply, Qwen3-VL `cu_seqlens`/M-RoPE, DETR 2D sine position scan | `challenge-admitted` |
| `torch_conv_codec` | conv/pool/interpolate/unfold/fold | `pt_conv_resample_codec_rows`, `pt_pointwise_cfg_scheduler` | Conv2d/3d islands, patchify/unpatchify | `challenge-admitted` |
| `torch_sparse_irregular` | sparse APIs, `segment_reduce`, block sparse candidates | `pt_sparse_block_extension` | segmented reduce, format conversion, sparse softmax | `candidate-with-shape` |
| `torch_optimizer_foreach` | `torch.optim`, `_foreach_*` | none yet | multi-tensor AdamW, norm clipping | `blocked-until-sourced` |

## Minimum Result Fields

Use `schemas/pytorch_baseline_challenge_result.schema.json` for machine-readable
records. A valid record needs:

- task and shape id
- public PyTorch reference expression/module
- baseline id and candidate id
- correctness status, tolerance, max abs/rel error
- candidate and baseline timing statistics
- speedup and score
- timing boundary and evidence label
- hardware, driver/runtime, ROCm, PyTorch, compiler/build metadata
- declared dropped generality
- classification and next attack surface

## First Implementation Stages

1. Define reference PyTorch modules for 10 visible shapes from
   `docs/REAL_MODEL_PROBLEM_SIZES.md`.
2. Add a runner that measures PyTorch eager with HIP events and reports
   `timing-only-measured`.
3. Add `ModelNew`/extension loading similar to KernelBench for custom HIP
   submissions.
4. Add hidden shape variants per suite.
5. Add `torch.compile` baseline timing.
6. Add external library competitors for categories where PyTorch eager is too
   weak or already library-backed.
7. Archive losing attempts as negative examples.

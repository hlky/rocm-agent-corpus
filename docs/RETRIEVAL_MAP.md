# Retrieval Map

This is the first file an agent should read after `README.md`.

## Fast Path by Intent

Question: "How should I optimize this kernel?"

Read:

1. `docs/AGENT_GUIDE.md`
2. `docs/OPTIMIZATION_PLAYBOOK.md`
3. Relevant task under `corpus/tasks/`
4. Strongest relevant competitor guide or upstream submodule under
   `third_party/`

Question: "Should I write a custom kernel?"

Read:

1. `docs/AGENT_GUIDE.md`
2. `docs/CUSTOM_KERNEL_COMPETITION_GUIDE.md`
3. `docs/OPTIMIZATION_PLAYBOOK.md`
4. `docs/CASE_CATALOG.md`
5. The relevant competitor guide:
   `docs/HIPBLAS_ROCBLAS_GUIDE.md`, `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md`,
   `docs/ROCPRIM_HIPCUB_ROCTHRUST_GUIDE.md`, `docs/MIGRAPHX_INFERENCE_GUIDE.md`, or
   `docs/TRITON_KERNEL_GUIDE.md`
6. `docs/LIBRARY_GUIDE.md`

Question: "Which competition track should I expand next?"

Read:

1. `docs/CASE_COVERAGE_PLAN.md`
2. `docs/CASE_CATALOG.md`, especially `Extended Case Families`
3. `data/index/case_catalog.json`
4. `docs/V2_EXTENSION_PROMOTION_WAVE.md`
5. `docs/V2_EXTENSION_WAVE3.md`
6. `data/index/v2_extension_wave.json`
7. `docs/CUDA_ROCM_TASK_PARITY_MAP.md`
8. `docs/INITIAL_TRACKING_LIST.md`
9. `docs/CORPUS_EXPANSION_STAGES.md`
10. `docs/GEMM_COMPETITION_TRACK.md`
11. `docs/ROCPRIM_COMPETITION_TRACK.md`
12. `docs/ARCHITECTURE_LABS.md`
13. `docs/AGENT_EVAL_HARNESS.md`

Question: "How do I turn a PyTorch op or subgraph into a ROCm challenge?"

Read:

1. `docs/PYTORCH_PUBLIC_API_KERNEL_CATALOG.md`
2. `docs/PYTORCH_API_REAL_WORLD_SHAPE_MATRIX.md`
3. `docs/REAL_MODEL_PROBLEM_SIZES.md`
4. `docs/PYTORCH_BASELINE_CHALLENGE.md`
5. `eval/tasks/pytorch_baseline_challenge_manifest.json`
6. `data/index/pytorch_public_api_kernel_catalog.json`
7. `data/index/pytorch_api_real_world_shape_matrix.json`
8. `data/index/real_model_problem_sizes.json`
9. `data/index/pytorch_baseline_challenge.json`
10. `schemas/pytorch_baseline_challenge_result.schema.json`

Question: "What is not covered by v1 yet?"

Read:

1. `docs/CASE_CATALOG.md`, starting at `Extended Case Families`
2. `data/index/case_catalog.json`, field `extension_families`
3. `docs/CASE_COVERAGE_PLAN.md`, section `Extension Coverage Backlog`
4. `docs/V2_EXTENSION_PROMOTION_WAVE.md` for already promoted v2 scaffolds
5. Promote one row only after choosing a task ID, library baselines, harness
   plan, and first evidence target.

Question: "How do I add a measured example?"

Read:

1. `docs/BENCHMARKING_GUIDE.md`
2. `docs/INGESTION.md`
3. `schemas/task.schema.json`
4. `schemas/optimization_record.schema.json`

Question: "Which path for inference or framework integration?"

Read:

1. `docs/PYTORCH_PUBLIC_API_KERNEL_CATALOG.md`,
   `docs/PYTORCH_API_REAL_WORLD_SHAPE_MATRIX.md`,
   `docs/REAL_MODEL_PROBLEM_SIZES.md`, and
   `docs/PYTORCH_BASELINE_CHALLENGE.md` when the starting point is a PyTorch
   public API, module, or model subgraph.
2. `docs/MIGRAPHX_INFERENCE_GUIDE.md`
3. `docs/ATTENTION_KERNEL_GUIDE.md` and
   `docs/ATTENTION_LIBRARY_BASELINES.md` for prefill/decode attention
   boundaries and library ladders.
4. `docs/SELECTION_SAMPLING_KERNEL_GUIDE.md` and
   `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md` for logits, top-k, beam, and
   sampling.
5. `docs/ROAD_TO_SOTA_TOPK.md`, `docs/TOPK_IMPLEMENTATION_SURVEY.md`,
   `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md`, and
   `docs/TOPK_WILD_IDEAS_LAB.md` for custom Top-K and sampling work.
6. `docs/QUANTIZATION_KERNEL_GUIDE.md` and
   `docs/QUANTIZATION_LIBRARY_BASELINES.md` for low-bit and fused dequant
   paths.
7. `docs/FRAMEWORK_EXTENSION_GUIDE.md`
8. `docs/FRAMEWORK_COMPILER_KERNELS_GUIDE.md`
9. `docs/TRITON_KERNEL_GUIDE.md`
10. `third_party/migraphx`, `third_party/vllm`, `third_party/pytorch`,
   `third_party/triton`
11. Return to `docs/AGENT_GUIDE.md` and identify the custom kernel boundary:
   plugin, fused op, fixed-shape kernel, extension op, or HIP replacement.

Question: "What changes for this GPU architecture?"

Read:

1. `docs/GPU_GFX_ARCHITECTURE_GUIDE.md`
2. `data/index/gpu_architectures.json`
3. `docs/RUNPOD_GPU_PRICING_20260520.md` when choosing a cheap Runpod GPU.
4. `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md`
5. Official tuning-guide source entries in `data/sources/core_resources.json`

## Task Map

| Task | Concept | Evidence |
| --- | --- | --- |
| `memory-coalesced-matrix-copy` | Row-major coalescing | template-only |
| `shared-memory-tiled-transpose` | Shared-memory tile and padding | template-only |
| `vectorized-saxpy` | Vectorized load/store caveats | template-only |
| `block-reduction-sum` | Atomic contention and block reduction | template-only |
| `rowwise-softmax` | Row reduction and stable softmax | template-only |
| `rowwise-layernorm-rmsnorm` | Cooperative normalization reductions | template-only |
| `block-prefix-scan` | Fixed-segment block scan | template-only |
| `histogram-privatized-atomics` | Shared-memory histogram privatization | template-only |
| `small-fixed-gemm` | Small batched GEMM seed | template-only |
| `online-attention-forward` | Online softmax attention tiling | template-only |
| `block-topk-sampling` | Top-k plus temperature sampling | template-only |
| `wide-k-topk-selection` | Top-k across k=1 through large/full-sort regimes | template-only |
| `fused-int4-dequant-gemv` | Packed INT4 fused dequant GEMV | template-only |
| `rocwmma-mfma-gemm` | rocWMMA Matrix Core HGEMM | template-only |
| `gather-scatter-coalescing` | Irregular versus coalesced index-map traffic | template-only |
| `shared-halo-stencil-2d` | 2D stencil with shared-memory halo reuse | template-only |
| `aos-soa-layout-conversion` | AoS to SoA vectorized layout conversion | template-only |
| `segmented-reduction-fixed-and-ragged` | Fixed and ragged segmented sums | template-only |
| `csr-spmv-load-balance` | CSR SpMV row-length load balance | template-only |
| `embedding-gather-dedup` | Embedding gather with duplicate-id reuse | template-only |
| `frontier-compaction-bfs` | Graph frontier compaction and scan/select | template-only |
| `block-sparse-attention-forward` | Block-sparse attention visible-block traversal | template-only |
| `hip-graphs-launch-overhead` | Launch overhead and graph replay | template-only |
| `streamed-copy-compute-overlap` | H2D/kernel/D2H stream overlap | template-only |
| `memory-pool-allocator` | HIP memory pool and scratch allocator behavior | template-only |
| `hiprtc-specialized-kernel-cache` | Runtime compilation and specialization cache | template-only |
| `persistent-work-queue` | Persistent CTA work queues | template-only |
| `warp-reduce-scan-vote` | Warp shuffle, scan, and vote primitives | template-only |
| `lds-tiled-copy` | LDS tiled staging | template-only |
| `cdna-mfma-gemm` | CDNA MFMA/LDS GEMM boundary | template-only |
| `inline-mfma-skeleton` | Inline LLVM IR / AMD GCN ISA MFMA skeleton | template-only |
| `migraphx-engine-tuning-sweep` | MIGraphX engine/tactic/timing-cache sweep | template-only |
| `triton-vs-hip-row-kernel` | Triton compiler baseline versus HIP row kernel | template-only |
| `fused-adamw-update` | Fused optimizer update versus staged elementwise passes | template-only |
| `multi-tensor-adamw` | Multi-tensor optimizer launch amortization | template-only |
| `rope-kv-cache-update` | Fused RoPE and KV-cache write | template-only |
| `moe-token-routing` | MoE top-2 routing and expert scatter | template-only |
| `gemm-bias-activation-epilogue` | GEMM epilogue fusion versus hipBLASLt/Composable Kernel | template-only |
| `migraphx-custom-op-fused-op` | MIGraphX plugin lifecycle plus enqueue kernel | template-only |
| `pytorch-hip-extension-op` | Framework extension stream/stride boundary | template-only |
| `rccl-overlap-allreduce` | Chunked compute plus RCCL AllReduce overlap | template-only |
| `rocshmem-queue` | GPU-initiated rocSHMEM queueing | template-only |
| `paged-kv-cache-attention` | Decode attention over paged KV cache | template-only |
| `gqa-mqa-decode-attention` | GQA/MQA grouped-head decode attention | template-only |
| `varlen-packed-attention` | Packed variable-length attention | template-only |
| `fused-mlp-swiglu` | MLP SwiGLU fusion boundary | template-only |
| `direct-convolution-2d` | Direct 2D convolution microkernel | template-only |
| `conv2d-fused-bias-activation` | Conv2D bias/activation fusion | template-only |
| `implicit-gemm-convolution` | MIOpen/Composable Kernel implicit-GEMM convolution boundary | template-only |
| `depthwise-separable-convolution` | Depthwise/separable convolution fusion | template-only |
| `radix-sort-key-value` | hipCUB/rocThrust key-value radix sort competitor | template-only |
| `top-p-nucleus-sampling` | Top-p logits sampling | template-only |
| `fp8-cast-scale-amax` | FP8 cast, scale, and amax fusion | template-only |
| `block-scaled-fp4-gemm` | Block-scaled FP4/NVFP4 GEMM boundary | template-only |
| `int8-dot-mfma-gemm` | INT8 DP4A/IMFMA GEMM ladder | template-only |
| `structured-sparsity-2to4` | 2:4 structured sparsity metadata and GEMM | template-only |
| `ragged-grouped-gemm` | Ragged/grouped GEMM scheduling | template-only |
| `splitk-reduction-gemm` | Split-K GEMM partial reductions | template-only |
| `composable-kernel-custom-epilogue` | Composable Kernel custom epilogue visitor | template-only |
| `pytorch-inductor-generated-kernel` | Framework compiler generated-kernel baseline | template-only |
| `rocprof-counter-roofline` | rocprofiler/rocprof counter and roofline evidence scaffold | template-only |
| `select-filter-compact` | hipCUB/rocThrust select and stream compaction | template-only |
| `groupby-reduce-by-key` | Grouped aggregation and reduce-by-key | template-only |
| `unique-run-length-encode` | Unique values and run-length counts | template-only |
| `sddmm-sparse-attention-score` | Sparse sampled dense-dense score construction | template-only |
| `spgemm-merge-hash` | Sparse matrix multiply merge/hash strategies | template-only |
| `sparse-format-conversion` | COO/CSR/CSC/ELL/BSR conversion costs | template-only |
| `reduce-scatter-overlap` | Chunked compute plus RCCL ReduceScatter overlap | template-only |
| `allgather-overlap` | Pipelined RCCL AllGather and consumer compute | template-only |
| `alltoall-moe-exchange` | MoE token exchange across ranks | template-only |
| `hip-ipc-multiprocess` | HIP IPC multiprocess shared-buffer workflow | template-only |
| `graph-captured-inference-step` | HIP Graph captured inference/decode step | template-only |
| `miopen-frontend-fusion` | MIOpen frontend operation graph fusion | template-only |
| `vllm-rocm-custom-plugin` | vLLM on ROCm plugin/kernel boundary | template-only |
| `normalization-backward` | LayerNorm/RMSNorm backward reductions | template-only |
| `strided-batched-layout-transform` | Strided and batched layout transforms | template-only |
| `packed-layout-transcode` | Packed low-bit/layout transcode | template-only |
| `rocm-sanitizer-racecheck` | ROCm sanitizer/debugger correctness workflow | template-only |
| `amdisa-diff-regression` | LLVM IR / AMD GCN ISA/codegen regression evidence | template-only |
| `autotune-parameter-sweep` | Reproducible autotuning sweep | template-only |
| `gfx90a-gfx942-cdna-split` | CDNA2 gfx90a versus CDNA3 gfx942 lab | template-only |
| `gfx942-cdna3-mfma-lab` | CDNA3 gfx942 MFMA/LDS lab | template-only |
| `wave-specialized-mfma-pipeline` | Producer/consumer wave-role MFMA pipeline scaffold | template-only |
| `global-to-lds-mfma-gemm` | LDS-staged MFMA GEMM boundary | template-only |
| `gfx950-gfx1200-rocm-portability` | CDNA4 versus RDNA4 ROCm portability path | template-only |

## Submodule Map

| Path | Use |
| --- | --- |
| `third_party/composable-kernel` | GEMM/Matrix Core competitor internals, CK Tile layouts, tile pipelines, custom epilogues |
| `third_party/rocm-libraries` | hipCUB, rocThrust, libcudacxx competitors and reusable block/warp primitives |
| `third_party/rocm` | ROCm umbrella repository and release/source navigation |
| `third_party/rocm-examples` | Library baselines, correctness oracles, and API examples for comparison |
| `third_party/rocblas-examples` | rocBLAS and hipBLAS usage examples |
| `third_party/kernelbench` | LLM CUDA generation benchmark and task format |
| `third_party/gpu-mode-lectures` | Teaching material and notebooks |
| `third_party/gpu-mode-reference-kernels` | Competitive/reference kernel tasks |
| `third_party/miopen` | MIOpen kernels, graph/fusion references, and convolution/attention baselines |
| `third_party/rccl` | Multi-GPU collective communication |
| `third_party/rocshmem` | GPU-initiated one-sided multi-GPU communication |
| `third_party/flash-attention` | IO-aware attention kernels and online softmax reference |
| `third_party/migraphx` | MIGraphX plugin boundaries, parsers, samples, engine/runtime competitor patterns |
| `third_party/triton` | Python-authored GPU kernel competitors, sketches, and compiler lowering examples |
| `third_party/vllm` | LLM serving, paged attention, KV cache, scheduling competitors |
| `third_party/flashinfer` | Inference attention, sampling, paged KV, decode/prefill references |
| `third_party/bitsandbytes` | Quantization, low-bit kernels, optimizer references |
| `third_party/pytorch` | Fusion compiler and generated-kernel competitor references |

## Competitor Guides

Use these docs to understand the library path before trying to beat or extend
it. The output should be a custom-kernel hypothesis, a measured boundary, or a
documented extension plan.

| Competitor | Read | Look For |
| --- | --- | --- |
| hipBLAS/rocBLAS/hipBLASLt | `docs/HIPBLAS_ROCBLAS_GUIDE.md` | GEMM baseline, epilogue fusion, layout and precision assumptions |
| Composable Kernel | `docs/COMPOSABLE_KERNEL_EXTENSION_GUIDE.md` | Tile shapes, Matrix Core pipelines, CK Tile layouts, custom epilogues |
| hipCUB/rocThrust/rocPRIM/hipCUB/rocThrust | `docs/ROCPRIM_HIPCUB_ROCTHRUST_GUIDE.md` | Reduction/scan primitives, iterator tricks, reusable warp/block pieces |
| MIGraphX/vLLM on ROCm | `docs/MIGRAPHX_INFERENCE_GUIDE.md` | Plugin boundaries, fusion opportunities, KV-cache and batching assumptions |
| Triton | `docs/TRITON_KERNEL_GUIDE.md` | Fast sketch kernels, compiler-generated competitors, HIP port targets |
| Frameworks | `docs/FRAMEWORK_EXTENSION_GUIDE.md` | Custom op boundaries, dispatch cost, shape/layout specialization points |
| FlashAttention/MIOpen/vLLM/FlashInfer | `docs/ATTENTION_KERNEL_GUIDE.md` | Online softmax, QK/V tiling, decode/prefill boundaries |
| vLLM on ROCm/vLLM/FlashInfer/hipCUB | `docs/SELECTION_SAMPLING_KERNEL_GUIDE.md` | Top-k, top-p, logits processors, sampling and beam helpers |
| bitsandbytes/Transformer Engine on ROCm/Composable Kernel | `docs/QUANTIZATION_KERNEL_GUIDE.md` | Packed low-bit formats, scale metadata, fused dequant |
| bitsandbytes/Composable Kernel/hipBLASLt/MIGraphX/Triton | `docs/QUANTIZATION_LIBRARY_BASELINES.md` | Fair low-bit baseline contracts and materialized-dequant boundaries |
| FlashAttention/MIOpen/Triton/vLLM on ROCm/vLLM/FlashInfer | `docs/ATTENTION_LIBRARY_BASELINES.md` | Attention library ladders, decode/paged-KV contracts, and timing boundaries |
| hipCUB/vLLM on ROCm/vLLM/FlashInfer/PyTorch | `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md`, `docs/TOPK_IMPLEMENTATION_SURVEY.md`, `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md` | Top-k/top-p/sampling baseline contracts, implementation paths, workspace notes, and result classifications |
| Matrix Core paths | `docs/MATRIX_CORE_ROCWMMA_GUIDE.md` | rocWMMA, MFMA, Composable Kernel/CK Tile, epilogues, architecture boundaries |

## Track Guides

| Track | Read | Output |
| --- | --- | --- |
| GEMM | `docs/GEMM_COMPETITION_TRACK.md` | hipBLAS/rocBLAS/hipBLASLt/Composable Kernel/Triton/custom comparison tasks |
| rocPRIM/hipCUB/rocThrust | `docs/ROCPRIM_COMPETITION_TRACK.md` | hipCUB/rocThrust baselines plus custom reductions/scans/histograms |
| Attention | `docs/ATTENTION_KERNEL_GUIDE.md` | FlashAttention-style custom attention and decode/prefill tasks |
| Selection/Sampling | `docs/SELECTION_SAMPLING_KERNEL_GUIDE.md`, `docs/ROAD_TO_SOTA_TOPK.md`, `docs/TOPK_IMPLEMENTATION_SURVEY.md`, `docs/TOPK_BASELINE_HARNESS_BLUEPRINT.md`, `docs/TOPK_WILD_IDEAS_LAB.md`, `data/index/topk_implementation_survey.json`, `data/index/topk_baseline_harness_blueprint.json`, `data/index/topk_wide_k_matrix.json` | Custom top-k, top-p, logits, beam, and sampling tasks |
| Quantization | `docs/QUANTIZATION_KERNEL_GUIDE.md` | Low-bit packed loads, scale layouts, fused dequant tasks |
| Matrix Core/rocWMMA | `docs/MATRIX_CORE_ROCWMMA_GUIDE.md` | rocWMMA, inline MFMA, Composable Kernel/CK Tile, and epilogue tasks |
| Memory Movement | `docs/MEMORY_MOVEMENT_KERNEL_GUIDE.md` | Gather/scatter, stencil, AoS/SoA, coalescing, and halo tasks |
| Sparse/Irregular | `docs/SPARSE_IRREGULAR_KERNEL_GUIDE.md` | Segmented reductions, CSR SpMV, embeddings, graph frontiers, sparse attention |
| System/Low-Level | `docs/SYSTEM_LOW_LEVEL_KERNEL_GUIDE.md` | Graphs, overlap, hipRTC, persistent queues, wave primitives, LDS staging, MFMA, MIGraphX/Triton |
| ML Adjunct | `docs/ML_ADJUNCT_KERNEL_GUIDE.md` | Optimizer updates, RoPE/KV-cache updates, and MoE routing |
| Integration/Epilogue | `docs/INTEGRATION_EPILOGUE_TASK_GUIDE.md` | GEMM epilogues, MIGraphX plugins, and framework extension operators |
| Multi-GPU Tasks | `docs/MULTIGPU_TASK_GUIDE.md` | RCCL overlap and rocSHMEM queue templates |
| Architecture | `docs/ARCHITECTURE_LABS.md` | GFX-specific compile flags, feature probes, and measured records |
| Eval Harness | `docs/AGENT_EVAL_HARNESS.md` | Agent submission compile/test/benchmark/classification loop |

## Evidence Labels

- `counter-backed-measured`: timing plus attached rocprofiler/rocprof counter artifacts.
- `timing-only`: HIP-event or framework timing and correctness, no counters.
- `negative example`: a valid optimization attempt that did not improve performance.
- `profile-attempted-blocked`: profiler was attempted but counters were blocked.
- `template-only`: code and metadata only.

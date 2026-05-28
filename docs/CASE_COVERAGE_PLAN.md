# Case Coverage Plan

This file is the working contract for completing the v1 rows in
`docs/CASE_CATALOG.md`. The extended case families in that file are a v2
backlog until promoted here with concrete tasks and baseline plans. Every
contract row should end with:

- at least one correctness-tested task,
- at least one measured timing record,
- the strongest relevant library baseline,
- at least one custom-kernel win/loss hypothesis,
- explicit evidence labels and hardware scope.

Library baselines are competitors, oracles, and extension surfaces. They are
not final answers unless the record explains why the library wins and what
narrower custom path remains plausible.

## Coverage Matrix

| Catalog Case | Status | Primary Task / Artifact | Required Library Baselines | Next Step |
| --- | --- | --- | --- | --- |
| Copy, map, SAXPY | scaffolded | `memory-coalesced-matrix-copy`, `vectorized-saxpy` | rocThrust transform, simple HIP reference | Add rocThrust transform timing and more alignment sweeps |
| Transpose | measured | `shared-memory-tiled-transpose` | Composable Kernel layout examples where GEMM-adjacent | Add rectangular and non-multiple tile sweeps |
| Gather/scatter | scaffolded | `gather-scatter-coalescing` | hipCUB select/partition, rocThrust gather/scatter | Run irregular/coalesced timing and add hipCUB/rocThrust comparison |
| Stencil | scaffolded | `shared-halo-stencil-2d` | ROCm examples, MIOpen/MIVisionX where applicable | Run halo/shared-memory timing and add boundary sweeps |
| AoS to SoA | scaffolded | `aos-soa-layout-conversion` | rocThrust transform, hipCUB block load/store | Run vectorized layout-conversion timing and alignment sweeps |
| Sum/min/max | measured | `block-reduction-sum` | hipCUB DeviceReduce | Add hipCUB DeviceReduce baseline record |
| Segmented reduction | scaffolded | `segmented-reduction-fixed-and-ragged` | hipCUB DeviceSegmentedReduce, rocThrust reduce_by_key | Add harness, hipCUB baseline, fixed/ragged timing |
| Scan/prefix sum | measured | `block-prefix-scan` | hipCUB DeviceScan, hipCUB BlockScan | Add hipCUB DeviceScan and BlockScan baseline records |
| Histogram | measured | `histogram-privatized-atomics` | hipCUB DeviceHistogram where applicable | Add uniform/skewed/single-bin hipCUB comparison |
| Top-k | measured seed + library recipes | `block-topk-sampling`, `docs/SELECTION_SAMPLING_LIBRARY_BASELINES.md` | hipCUB DeviceTopK, radix sort, vLLM on ROCm, vLLM, FlashInfer | Add hipCUB/FlashInfer/vLLM on ROCm timing where feasible |
| GEMM | guide + measured seeds + hipBLASLt baseline | `small-fixed-gemm`, `rocwmma-mfma-gemm`, `harnesses/hipblaslt_hgemm_benchmark.hip` | hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, Triton | Rerun custom rocWMMA and hipBLASLt on same GPU; add Composable Kernel/Triton |
| Batched GEMM | measured negative seed | `small-fixed-gemm` | hipBLAS/rocBLAS strided batched, hipBLASLt, Composable Kernel grouped/batched | Add library baselines and batch-size sweeps |
| GEMM + bias/activation | scaffolded | `gemm-bias-activation-epilogue` | hipBLASLt epilogues, Composable Kernel epilogues, Triton | Add benchmark harness and hipBLASLt/Composable Kernel epilogue timing |
| Custom GEMM | guide + measured seed | `rocwmma-mfma-gemm` | Composable Kernel/CK Tile, hipBLASLt | Add Composable Kernel/CK Tile baseline recipe and task metadata |
| Quantized GEMM | measured seed + library recipes | `fused-int4-dequant-gemv`, `docs/QUANTIZATION_LIBRARY_BASELINES.md` | bitsandbytes, Composable Kernel, hipBLASLt, vLLM on ROCm, Triton | Add materialized dequant and library baseline timing records |
| Softmax | measured | `rowwise-softmax` | framework/Triton/hipCUB block primitives where applicable | Add Triton/framework baseline recipe |
| LayerNorm/RMSNorm | measured | `rowwise-layernorm-rmsnorm` | framework, Triton, Transformer Engine on ROCm, vLLM on ROCm | Add fused residual and half/bfloat16 variants |
| Attention | measured seed + library recipes | `online-attention-forward`, `docs/ATTENTION_LIBRARY_BASELINES.md` | FlashAttention, MIOpen, Triton, vLLM on ROCm, vLLM, FlashInfer | Add measured library baselines and decode/paged KV tasks |
| Embedding ops | scaffolded | `embedding-gather-dedup` | framework embedding, hipCUB sort/select for dedupe | Add harness and duplicate-ratio timing sweeps |
| Optimizers | scaffolded | `fused-adamw-update` | fused framework optimizer, bitsandbytes, Transformer Engine on ROCm where applicable | Add optimizer harness and framework/library timing |
| SpMV/SpMM | scaffolded | `csr-spmv-load-balance`, `docs/SPARSE_IRREGULAR_KERNEL_GUIDE.md` | rocSPARSE, Composable Kernel sparse where applicable | Add rocSPARSE baseline and row-distribution timing |
| Graph traversal | scaffolded | `frontier-compaction-bfs` | hipCUB select/scan, Gunrock-like references | Add harness and hipCUB select/scan baseline |
| Sparse attention | scaffolded | `block-sparse-attention-forward` | FlashAttention block sparse, Composable Kernel sparse, Triton | Add block-mask attention harness and library recipes |
| Launch overhead | scaffolded | `hip-graphs-launch-overhead` | HIP Graphs, stream capture examples | Add measured ordinary-launch versus graph-replay record |
| Overlap copy/compute | scaffolded | `streamed-copy-compute-overlap` | HIP streams/events, async memcpy samples | Add pinned-memory overlap timing |
| Multi-GPU collectives | scaffolded | `rccl-overlap-allreduce`, `rocshmem-queue` | RCCL, rocSHMEM | Measure when multi-GPU hardware exists |
| Runtime compilation | scaffolded | `hiprtc-specialized-kernel-cache` | hipRTC | Add compile/cache-hit timing records |
| Persistent services | scaffolded | `persistent-work-queue` | HIP cooperative groups where available, runtime queues | Add fairness/queue-depth timing sweeps |
| Warp primitives | scaffolded | `warp-reduce-scan-vote` | HIP C++ intrinsics, hipCUB WarpReduce/WarpScan | Add hipCUB WarpReduce/WarpScan baseline and harness |
| LDS staging | scaffolded | `lds-tiled-copy`, `global-to-lds-mfma-gemm` | ROCm examples, Composable Kernel/CK Tile mainloops | Run gfx90a/gfx942 LDS-staging task where supported |
| Matrix Core MFMA | measured seed + hipBLASLt baseline | `rocwmma-mfma-gemm` | hipBLASLt, Composable Kernel/CK Tile, Triton | Add same-hardware custom/library timing and Composable Kernel baseline |
| MFMA/LDS pipelines | scaffolded | `cdna-mfma-gemm`, `wave-specialized-mfma-pipeline`, `global-to-lds-mfma-gemm` | Composable Kernel/CK Tile CDNA examples | Compile/test on CDNA; keep gfx targets separate |
| Inline LLVM IR / AMD GCN ISA | scaffolded | `inline-mfma-skeleton` | LLVM IR, AMD GCN ISA inspection | Add compile-only validation and AMD GCN ISA inspection notes |
| MIGraphX engine build | scaffolded | `migraphx-engine-tuning-sweep` | MIGraphX builder/runtime | Add MIGraphX install/run recipe and timing-cache artifacts |
| MIGraphX plugin | scaffolded | `migraphx-custom-op-fused-op` | MIGraphX plugin API | Add compile/test recipe and engine timing |
| LLM inference | scaffolded adjunct tasks | `rope-kv-cache-update`, `moe-token-routing`, `block-topk-sampling`, `online-attention-forward` | vLLM on ROCm, vLLM, FlashInfer | Add decode/KV/MoE measured baselines |
| Framework custom op | scaffolded | `pytorch-hip-extension-op` | PyTorch extension, OneFlow op | Add extension build and framework baseline timing |
| Triton kernel | scaffolded | `triton-vs-hip-row-kernel` | Triton, framework compiler | Add paired Triton/ROCm benchmark record |
| CDNA2 tuning | guide + scaffolded tasks | architecture labs | Composable Kernel, hipBLASLt | Add gfx90a timing and LDS-staging tasks |
| RDNA3 inference | guide + pricing | architecture labs | MIGraphX, hipBLASLt, Composable Kernel | Add gfx1100 timing pass |
| CDNA3 tuning | guide + pricing | architecture labs | Composable Kernel/CK Tile, hipBLASLt, MIGraphX | Add gfx942 timing pass for rocWMMA/attention |
| CDNA4/RDNA4 tuning | guide + pricing | architecture labs | current ROCm libraries, Composable Kernel, MIGraphX | Add gfx950/gfx1200 compile/timing pass when available |

## Library Coverage Targets

| Library / Ecosystem | Must Cover | Current Status | Next Concrete Step |
| --- | --- | --- | --- |
| HIP Runtime / Driver | streams, events, graphs, memory, cooperative groups, hipRTC | scaffolded | Measure launch/graph/runtime compilation tasks |
| rocprofiler/rocprof / Systems | counters, timelines, permission failures | partial | Add counter-enabled machine runbook and sample profile artifacts |
| hipBLAS/rocBLAS | GEMM, strided batched, GemmEx, math modes | partial | Add GEMM library baseline harnesses |
| hipBLASLt | algorithms, layouts, epilogues, workspace | HGEMM baseline measured + epilogue scaffolded | Rerun same-hardware custom/library comparisons and add epilogue timing |
| Composable Kernel / CK Tile | GEMM, epilogues, grouped GEMM, global-to-LDS staging/MFMA | partial | Add Composable Kernel baseline recipe per GEMM task |
| rocPRIM/hipCUB/rocThrust / hipCUB / rocThrust | reduce, scan, histogram, select, top-k, sort | partial | Add hipCUB baseline harnesses for measured rocPRIM/hipCUB/rocThrust tasks |
| MIOpen | SDPA, operation graphs, plans, fusions | baseline recipe added | Add measured attention baseline recipe |
| MIGraphX | engine build, plugins, precision/tactics | partial | Promote plugin skeleton and engine task |
| vLLM on ROCm | attention, KV cache, RMSNorm, sampling, quantization | partial | Add decode/sampling/KV baseline recipes |
| Triton | compiler baseline and HIP port target | partial | Add paired Triton/HIP tasks |
| OneFlow / PyTorch extension | framework custom op boundaries | partial | Add extension task with stream/stride checks |
| vLLM / FlashInfer | paged attention, sampling, serving decode | baseline recipes added | Add decode and top-k measured records |
| bitsandbytes | low-bit quantization and optimizer kernels | baseline recipe added | Add int4 GEMV and optimizer task timing |
| Transformer Engine on ROCm | FP8, normalization, transformer kernels | partial | Add FP8/block-scaled task |
| RCCL / rocSHMEM | multi-GPU collectives and one-sided comms | scaffolded | Measure templates pending multi-GPU hardware |
| rocSPARSE | SpMV/SpMM | scaffolded | Add rocSPARSE measured baseline |
| hipRTC / hipRTC cache/link path | runtime specialization | scaffolded | Add compile/cache-hit timing record |

## Remaining Work After Current Scaffold Wave

Every v1 row in `docs/CASE_CATALOG.md` now has at least a measured seed, guide,
or template task. Remaining v1 work is completion quality rather than
first-contact coverage:

- add measured records for template-only tasks,
- add strongest library baselines for every measured custom seed,
- rerun custom and library paths on the same hardware before win/loss claims,
- add rocprofiler/rocprof evidence where counter/timeline access is
  available,
- promote architecture guide rows into measured GFX-specific passes.

## Extension Coverage Backlog

The v2 extension rows in `docs/CASE_CATALOG.md` are deliberately not counted as
complete-v1 obligations yet. Promote a row from `v2-proposed` to `scaffolded`
when it has a task ID, baseline library list, harness plan, and first evidence
target.

| Extension Family | Status | Candidate First Tasks | Required Library Baselines | First Artifact |
| --- | --- | --- | --- | --- |
| Memory, layout, and data plumbing | v2-proposed | `strided-batched-layout-transform`, `packed-layout-transcode`, `memory-pool-allocator` | rocThrust, hipCUB, HIP memory pools, framework tensor copies | layout/allocator task manifests |
| Convolution, signal, and classical linear algebra | partially scaffolded | `direct-convolution-2d`, `conv2d-fused-bias-activation`, `implicit-gemm-convolution`, `small-gemv-token-batch` | MIOpen, Composable Kernel conv, MIGraphX, rocFFT, hipBLAS/rocBLAS/rocSOLVER | `direct-convolution-2d`, `conv2d-fused-bias-activation` |
| Sorting, grouping, and selection | partially scaffolded | `radix-sort-key-value`, `select-filter-compact`, `groupby-reduce-by-key`, `unique-run-length-encode`, `top-p-nucleus-sampling` | hipCUB DeviceRadixSort, hipCUB DeviceSelect, hipCUB DeviceRunLengthEncode, rocThrust, vLLM on ROCm, vLLM, FlashInfer | radix/sampling/select/grouping scaffolds |
| Transformer and ML fusion | partially scaffolded | `paged-kv-cache-attention`, `varlen-packed-attention`, `gqa-mqa-decode-attention`, `fused-mlp-swiglu`, `normalization-backward` | FlashAttention, MIOpen, vLLM on ROCm, vLLM, FlashInfer, Transformer Engine on ROCm, Triton | `paged-kv-cache-attention`, `varlen-packed-attention`, `gqa-mqa-decode-attention`, `fused-mlp-swiglu` |
| Quantization and numeric formats | partially scaffolded | `fp8-cast-scale-amax`, `block-scaled-fp4-gemm`, `int8-dot-mfma-gemm`, `int4-groupwise-gemv` | Transformer Engine on ROCm, hipBLASLt, Composable Kernel, MIGraphX, bitsandbytes | `fp8-cast-scale-amax`, `block-scaled-fp4-gemm`, `int8-dot-mfma-gemm` |
| Sparse, structured sparsity, and graph work | partially scaffolded | `sddmm-sparse-attention-score`, `spgemm-merge-hash`, `sparse-format-conversion`, `structured-sparsity-2to4`, `gnn-neighbor-aggregation` | rocSPARSE, Composable Kernel sparse, MIGraphX, graph references | SDDMM/SpGEMM/format/2:4 sparse scaffolds |
| Matrix Core, Composable Kernel, and CK Tile internals | partially scaffolded | `composable-kernel-custom-epilogue`, `ragged-grouped-gemm`, `splitk-reduction-gemm`, `wave-specialized-mfma-pipeline`, `global-to-lds-mfma-gemm` | Composable Kernel/CK Tile, hipBLASLt, Triton | CK epilogue, grouped GEMM, wave pipeline, and LDS-staged GEMM scaffolds |
| Inference, framework, and compiler integration | partially scaffolded | `miopen-frontend-fusion`, `graph-captured-inference-step`, `vllm-rocm-custom-plugin`, `pytorch-inductor-generated-kernel` | MIOpen, HIP Graphs, MIGraphX, vLLM on ROCm, PyTorch Inductor, OneFlow | `pytorch-inductor-generated-kernel` |
| Multi-GPU and runtime systems | partially scaffolded | `reduce-scatter-overlap`, `allgather-overlap`, `alltoall-moe-exchange`, `hip-ipc-multiprocess` | RCCL, rocSHMEM, HIP IPC/P2P, framework distributed | collective overlap, IPC, and MoE exchange scaffolds |
| Tooling, verification, and agent evaluation | partially scaffolded | `rocprof-counter-roofline`, `rocm-sanitizer-racecheck`, `amdisa-diff-regression`, `autotune-parameter-sweep` | rocprofiler/rocprof, ROCm sanitizer/debugger, llvm-objdump/roc-objdump, Composable Kernel profiler | profiling, sanitizer, AMD GCN ISA, and autotune scaffolds |
| Architecture-specific extension paths | partially scaffolded | `gfx90a-gfx942-cdna-split`, `gfx942-cdna3-mfma-lab`, `gfx950-gfx1200-rocm-portability` | AMD tuning guides, Composable Kernel, hipBLASLt, MIGraphX | GFX split, MFMA/LDS, and CDNA4/RDNA4 portability scaffolds |

The promoted waves are summarized in
`docs/V2_EXTENSION_PROMOTION_WAVE.md`. These tasks raise corpus task count but
remain `template-only` until correctness and timing artifacts are added.

## Completion Rule

A catalog row can be marked `complete-v1` when it has:

1. A task with baseline, optimized/custom, harness, and CPU or library oracle.
2. At least one timing-only or counter-backed record on named hardware.
3. A library baseline record or a justified "not applicable" note.
4. A negative or library-win record when the custom path does not win.
5. Retrieval links from `docs/RETRIEVAL_MAP.md` and relevant repo-local skills.

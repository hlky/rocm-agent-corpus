# HIP/ROCm Optimization Case Catalog

The corpus should cover each v1 row in the main tables with at least one
correctness-tested task, one measured record, and one vendor-baseline comparison
when applicable. The extended case families later in this file are a v2 backlog
for promotion into that same standard. The point is to help agents compete with
or extend the vendor path, not to stop at it.

## Memory Movement

| Case | Vendor Baseline / Reference | Custom Kernel Topics | Current Coverage |
| --- | --- | --- | --- |
| Copy, map, SAXPY | custom or rocThrust transform | coalescing, vectorized loads, alignment | `memory-coalesced-matrix-copy`, `vectorized-saxpy` |
| Transpose | custom, Composable Kernel layouts for tiled GEMM-adjacent cases | shared tiles, bank conflicts | `shared-memory-tiled-transpose` |
| Gather/scatter | hipCUB select/partition when possible | irregular access, cache behavior, coalescing loss | `gather-scatter-coalescing` scaffold |
| Stencil | custom | halo loading, shared tiles, boundary handling | `shared-halo-stencil-2d` scaffold |
| AoS to SoA | custom or hipCUB transforms | layout conversion, vectorized loads | `aos-soa-layout-conversion` scaffold |

## Reductions and Prefix Work

| Case | Vendor Baseline / Reference | Custom Kernel Topics | Current Coverage |
| --- | --- | --- | --- |
| Sum/min/max | hipCUB DeviceReduce | warp/block reductions, atomics | `block-reduction-sum` |
| Segmented reduction | hipCUB DeviceSegmentedReduce | segment balance, memory layout | `segmented-reduction-fixed-and-ragged` scaffold |
| Scan/prefix sum | hipCUB DeviceScan | upsweep/downsweep, block scan | `block-prefix-scan` |
| Histogram | hipCUB histogram where applicable | privatization, shared atomics, merge | `histogram-privatized-atomics` |
| Top-k | hipCUB/rocThrust sort/select or custom | heap/register selection, warp cooperation | `block-topk-sampling` scaffold |

## Dense Linear Algebra

| Case | Vendor Baseline / Reference | Extension Points | Current Coverage |
| --- | --- | --- | --- |
| GEMM | hipBLAS/rocBLAS/hipBLASLt | transpose flags, leading dimensions, TF32, tensor ops | guide, measured seeds, hipBLASLt baseline |
| Batched GEMM | hipBLAS/rocBLAS StridedBatched or hipBLASLt | batch stride, grouped shapes | `small-fixed-gemm` measured negative seed |
| GEMM + bias/activation | hipBLASLt or Composable Kernel | epilogues, visitors, workspace, alg selection | guide, examples, `gemm-bias-activation-epilogue` scaffold |
| Custom GEMM | Composable Kernel/CK Tile | tile shapes, mainloop, epilogue, swizzle | guide, submodule, `rocwmma-mfma-gemm` |
| Quantized GEMM | hipBLASLt/Composable Kernel | scales, zero points, int8/fp8/nvfp4 | `fused-int4-dequant-gemv` scaffold |

## ML Kernels

| Case | Vendor Baseline / Reference | Custom Kernel Topics | Current Coverage |
| --- | --- | --- | --- |
| Softmax | framework/library unless fused | row reductions, stability, vectorization | `rowwise-softmax` |
| LayerNorm/RMSNorm | framework, custom for fusion | row reductions, variance, precision | `rowwise-layernorm-rmsnorm` |
| Attention | MIOpen/FlashAttention/Composable Kernel-like | tiling, online softmax, Matrix Cores | `online-attention-forward` measured seed |
| Embedding ops | custom | irregular memory, atomics, dedupe | `embedding-gather-dedup` scaffold |
| Optimizers | fused framework kernels | vectorization, mixed precision | `fused-adamw-update` scaffold |

## Sparse and Irregular

| Case | Vendor Baseline / Reference | Custom Kernel Topics | Current Coverage |
| --- | --- | --- | --- |
| SpMV/SpMM | rocSPARSE | format choice, load balance | `csr-spmv-load-balance` scaffold and guide |
| Graph traversal | custom/Gunrock-like libraries | frontier compaction, atomics | `frontier-compaction-bfs` scaffold |
| Sparse attention | custom/Composable Kernel variants | block sparsity, metadata layout | `block-sparse-attention-forward` scaffold |

## HIP/ROCm Runtime Systems

| Case | Vendor Baseline / Reference | Custom Topics | Current Coverage |
| --- | --- | --- | --- |
| Launch overhead | HIP Graphs | capture, graph update, memory reuse | guide |
| Overlap copy/compute | streams + async copies | pinned memory, stream ordering | `streamed-copy-compute-overlap` scaffold |
| Multi-GPU collectives | RCCL/rocSHMEM | topology, overlap, partitioning | `rccl-overlap-allreduce`, `rocshmem-queue` scaffolds |
| Runtime compilation | hipRTC | specialization, cache keys | `hiprtc-specialized-kernel-cache` scaffold |
| Persistent services | custom | queues, occupancy, fairness | `persistent-work-queue` scaffold |

## Low-Level Extensions

| Case | Vendor Baseline / Reference | Custom Topics | Current Coverage |
| --- | --- | --- | --- |
| Warp primitives | HIP C++/hipCUB | shuffle, ballot, cooperative groups | `warp-reduce-scan-vote` scaffold |
| LDS staging | HIP C++/AMD GCN ISA/Composable Kernel | LDS tiles, barriers, wait counts, pipeline stages | `lds-tiled-copy`, `global-to-lds-mfma-gemm` scaffolds |
| Matrix Core MFMA | Composable Kernel/CK Tile | MFMA atoms, layouts, fragments | `rocwmma-mfma-gemm` measured seed and hipBLASLt baseline |
| MFMA/LDS pipelines | Composable Kernel/CK Tile on CDNA | LDS staging, barriers, wave roles, MFMA | `cdna-mfma-gemm`, `wave-specialized-mfma-pipeline` scaffolds |
| Inline LLVM IR / AMD GCN ISA | last resort | constraints, portability, AMD GCN ISA inspection | `inline-mfma-skeleton` scaffold |

## Inference and Framework Integration

| Case | Vendor Baseline / Reference | Custom Topics | Current Coverage |
| --- | --- | --- | --- |
| MIGraphX engine build | MIGraphX | dynamic profiles, timing cache, precision, tactic selection | guide and `migraphx-engine-tuning-sweep` scaffold |
| MIGraphX plugin | MIGraphX plugin API | shape inference, formats, enqueue, serialization | guide, skeleton, `migraphx-custom-op-fused-op` scaffold |
| LLM inference | vLLM on ROCm | KV cache, batching, quantization, plugins, serving metrics | guide plus RoPE/KV, MoE, top-k, attention scaffolds |
| Framework custom op | PyTorch/OneFlow extension path | tensor checks, autograd, stream semantics | guide, skeleton, `pytorch-hip-extension-op` scaffold |
| Triton kernel | Triton | meta-parameters, compiler output, framework integration | guide, skeleton, `triton-vs-hip-row-kernel` scaffold |

## Architecture-Specific Paths

| Case | Vendor Baseline / Reference | Custom Topics | Current Coverage |
| --- | --- | --- | --- |
| CDNA2 tuning | Composable Kernel/hipBLASLt | MFMA, LDS staging, gfx90a | guide and index |
| RDNA3 inference | MIGraphX/hipBLASLt | tactic selection, gfx1100 compile flags | guide and index |
| CDNA3 tuning | Composable Kernel/CK Tile | MFMA, LDS staging, gfx942 portability | guide and index |
| CDNA4/RDNA4 tuning | current ROCm libraries, Composable Kernel, MIGraphX | gfx950/gfx1200, family-specific target checks | guide and index |

## Extended Case Families

The rows below are a v2 backlog. They are intentionally broader than the first
coverage contract above. Treat them as `v2-proposed` until a row is promoted
into `docs/CASE_COVERAGE_PLAN.md` with a concrete task, harness, library
baseline, and evidence target.

### Memory, Layout, and Data Plumbing

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `strided-batched-layout-transform` | Strided and batched layout transforms | rocThrust, hipCUB block load/store, framework tensor copies | stride math, vectorization, alignment, non-contiguous tensors | layout-conversion task with stride sweep |
| `packed-layout-transcode` | Packed format transcode | Composable Kernel layouts, vLLM on ROCm, Transformer Engine on ROCm | interleaving, swizzles, scale metadata, vectorized stores | packed int4/fp8 format task |
| `async-p2p-copy-pack` | Pack, copy, and unpack across peers | HIP P2P, RCCL point-to-point | pack kernels, stream ordering, interconnect boundaries | P2P staging task |
| `memory-pool-allocator` | Device memory pool and scratch allocator behavior | HIP memory pools, framework caching allocators | allocation overhead, reuse, fragmentation, graph capture | allocator microbenchmark task |
| `image-plane-format-convert` | Image/video plane conversion | NPP, nvJPEG, framework preprocessors | planar/interleaved formats, color conversion, fused normalization | image preprocessing task |
| `nvcomp-streaming-decode` | Compression/decompression adjacent kernels | nvCOMP, CUDA samples | block framing, decode-plus-transform fusion, bandwidth limits | compression pipeline task |

### Convolution, Signal, and Classical Linear Algebra

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `direct-convolution-2d` | Direct 2D convolution | MIOpen, Composable Kernel conv, framework conv | shared tiles, constant filters, small kernels | `direct-convolution-2d` scaffold |
| `conv2d-fused-bias-activation` | Fused inference convolution | MIOpen, MIGraphX, Composable Kernel conv epilogues | conv epilogues, bias/activation fusion, layout constraints | `conv2d-fused-bias-activation` scaffold |
| `depthwise-separable-convolution` | Depthwise and separable convolution | MIOpen, MIGraphX, framework kernels | channel grouping, vectorized channel packs, fusion | depthwise conv task |
| `implicit-gemm-convolution` | Implicit-GEMM convolution | MIOpen, Composable Kernel implicit GEMM | iterator math, layout transforms, Matrix Core eligibility | implicit-GEMM conv task |
| `winograd-convolution-tile` | Winograd convolution tile | MIOpen, MIGraphX | transform overhead, numerical error, fixed filters | Winograd task |
| `fft-convolution-fusion` | FFT-based convolution and correlation | rocFFT, MIOpen where applicable | library-call fusion, pointwise multiply, batched plans | rocFFT-adjacent task |
| `small-gemv-token-batch` | Small GEMV and token-batch matvec | hipBLAS/rocBLAS, hipBLASLt, bitsandbytes | persistent weights, dequant, wave-level dot products | GEMV decode task |
| `triangular-solve-small-batch` | Small batched triangular solve | hipBLAS/rocBLAS, rocSOLVER | dependency scheduling, register tiling, fixed-size batches | solver-adjacent task |

### Sorting, Grouping, and Selection

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `radix-sort-key-value` | Key/value radix sort | hipCUB DeviceRadixSort, rocThrust sort | fixed key width, local histograms, temporary storage | `radix-sort-key-value` scaffold |
| `select-filter-compact` | Predicate select, filter, and compact | hipCUB DeviceSelect, rocThrust copy_if | mask generation, prefix counts, fused transforms | `select-filter-compact` scaffold |
| `unique-run-length-encode` | Unique and run-length encode | hipCUB DeviceRunLengthEncode, rocThrust unique | sorted inputs, segment boundaries, output sizing | `unique-run-length-encode` scaffold |
| `groupby-reduce-by-key` | Group-by and reduce-by-key | hipCUB/rocThrust reduce_by_key | skewed groups, atomics versus scans, shared aggregation | `groupby-reduce-by-key` scaffold |
| `top-p-nucleus-sampling` | Top-p and nucleus sampling | vLLM on ROCm, vLLM, FlashInfer, hipCUB sort/select | partial sort, prefix probability, RNG, rejection paths | `top-p-nucleus-sampling` scaffold |
| `deterministic-reduction` | Deterministic reductions | hipCUB, framework deterministic modes | fixed trees, reproducibility, compensated sums | determinism task |

### Transformer and ML Fusion

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `fused-residual-dropout-norm` | Residual, dropout, and norm fusion | framework kernels, Transformer Engine on ROCm, vLLM on ROCm | RNG state, vectorized rows, mixed precision | fused norm task |
| `normalization-backward` | LayerNorm/RMSNorm backward | PyTorch, Transformer Engine on ROCm, Triton, Apex-like references | dgamma/dbeta reductions, saved statistics, mixed precision | norm backward task |
| `fused-mlp-swiglu` | MLP activation and SwiGLU/GeGLU fusion | hipBLASLt, Composable Kernel, vLLM on ROCm, Triton | epilogues, gating, activation math, intermediate avoidance | `fused-mlp-swiglu` scaffold |
| `cross-entropy-loss` | Cross entropy and label smoothing | framework kernels, MIOpen where applicable | max/sum reductions, sparse targets, gradient fusion | loss task |
| `rotary-alibi-positioning` | RoPE, ALiBi, and positional transforms | vLLM on ROCm, vLLM, FlashInfer | sin/cos layout, vectorized complex multiply, cache writes | RoPE/ALiBi task |
| `paged-kv-cache-attention` | Paged KV-cache read/write and decode | vLLM, FlashInfer, vLLM on ROCm | page tables, coalescing, fragmentation, batch scheduling | `paged-kv-cache-attention` scaffold |
| `varlen-packed-attention` | Variable-length packed attention | FlashAttention varlen, MIOpen SDPA, FlashInfer, Triton | prefix offsets, ragged batches, mask elimination, scheduler metadata | `varlen-packed-attention` scaffold |
| `gqa-mqa-decode-attention` | GQA/MQA decode attention | vLLM on ROCm, FlashInfer, vLLM, FlashAttention where applicable | KV head sharing, grouped heads, cache reads, small-query batching | `gqa-mqa-decode-attention` scaffold |
| `beam-search-step` | Beam search step and logits processors | vLLM on ROCm, vLLM, framework generation | top-k/top-p composition, penalties, parent tracking | beam step task |
| `masked-logits-softmax-fusion` | Mask, scale, bias, softmax, and logits fusion | PyTorch, Triton, vLLM on ROCm, MIOpen SDPA-adjacent paths | mask application, stable row reductions, log-softmax, logits processors | masked logits task |
| `activation-gated-fusion` | Activation and gating fusion | hipBLASLt epilogues, Composable Kernel epilogues, Triton | custom epilogues, approximations, auxiliary outputs | activation epilogue task |
| `ragged-sequence-pack-unpack` | Ragged sequence pack/unpack | framework nested tensors, hipCUB select/scan | offsets, padding removal, variable lengths | ragged tensor task |
| `multi-tensor-adamw` | Multi-tensor optimizer update | PyTorch foreach/fused optimizers, Apex-style fused Adam, bitsandbytes | tensor-list batching, mixed precision states, launch amortization | multi-tensor optimizer task |

### Quantization, Precision, and Numeric Format Work

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `fp8-cast-scale-amax` | FP8 cast, scale, and amax update | Transformer Engine on ROCm, hipBLASLt, Composable Kernel | amax reductions, delayed scaling, saturation | `fp8-cast-scale-amax` scaffold |
| `block-scaled-fp4-gemm` | Block-scaled FP4/NVFP4 matmul path | Transformer Engine on ROCm, Composable Kernel, MIGraphX | block scales, packing, Matrix Core eligibility | `block-scaled-fp4-gemm` scaffold |
| `int8-dot-mfma-gemm` | INT8 GEMM/GEMV with DP4A or IMFMA paths | hipBLASLt INT8, Composable Kernel INT8, MIGraphX, bitsandbytes | packing, signedness, scale placement, Matrix Core versus DP4A fallback | `int8-dot-mfma-gemm` scaffold |
| `int4-groupwise-gemv` | Groupwise INT4 GEMV and decode | bitsandbytes, vLLM on ROCm, Composable Kernel | packed nibbles, group scales, zero points | INT4 GEMV task |
| `quantize-dequantize-reorder` | Quantize/dequantize plus reorder | framework quantization, MIGraphX | scale layout, transpose fusion, saturation | Q/DQ reorder task |
| `stochastic-rounding-kernel` | Stochastic rounding and noise kernels | framework/TE paths | RNG quality, vectorized conversion, reproducibility | stochastic rounding task |
| `mixed-precision-accumulation` | Mixed-precision accumulation contracts | hipBLASLt, Composable Kernel, framework kernels | error budgets, accumulator type, K-splitting | numeric contract task |

### Sparse, Structured Sparsity, and Graph Work

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `sddmm-sparse-attention-score` | SDDMM and sparse score construction | rocSPARSE, Composable Kernel sparse, graph libraries | metadata traversal, coalesced edge blocks, reuse | `sddmm-sparse-attention-score` scaffold |
| `spgemm-merge-hash` | Sparse matrix multiply | rocSPARSE SpGEMM | merge/hash strategies, row imbalance, memory planning | `spgemm-merge-hash` scaffold |
| `sparse-format-conversion` | COO/CSR/CSC/ELL/BSR conversion | rocSPARSE, hipCUB sort/scan | prefix counts, sorting, block metadata | `sparse-format-conversion` scaffold |
| `bsr-block-sparse-matmul` | Block-sparse matmul | rocSPARSE BSR, Composable Kernel sparse, Triton | block masks, Matrix Core block size, scheduler | BSR matmul task |
| `structured-sparsity-2to4` | 2:4 structured sparsity | Composable Kernel sparse, MIGraphX, dense GEMM | metadata packing, pruning layout, sparse MFMA | `structured-sparsity-2to4` scaffold |
| `gnn-neighbor-aggregation` | GNN neighbor aggregation | rocSPARSE, graph references, framework ops | frontier grouping, atomics, load balance | GNN aggregation task |

### Matrix Core, Composable Kernel, and CK Tile Internals

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `composable-kernel-custom-epilogue` | Composable Kernel custom epilogue visitor | Composable Kernel, hipBLASLt epilogues | visitor trees, auxiliary tensors, activation/scale fusion | `composable-kernel-custom-epilogue` scaffold |
| `ck_tile-layout-algebra-task` | CK Tile layout algebra | Composable Kernel/CK Tile examples | layout composition, tilers, tensor views, swizzles | CK Tile layout task |
| `persistent-gemm-scheduler` | Persistent GEMM scheduling | Composable Kernel persistent kernels, hipBLASLt | work queues, CTA residency, split tiles | persistent GEMM task |
| `splitk-reduction-gemm` | Split-K GEMM reductions | hipBLASLt, Composable Kernel | partial accumulation, reduction epilogue, atomics | split-K task |
| `ragged-grouped-gemm` | Grouped and ragged GEMM | hipBLASLt grouped matmul where available, Composable Kernel grouped GEMM, Triton grouped matmul | shape bucketing, pointer arrays, scheduling, padding avoidance | `ragged-grouped-gemm` scaffold |
| `grouped-gemm-moe` | Grouped GEMM for MoE | hipBLASLt grouped paths, Composable Kernel grouped GEMM, vLLM on ROCm | shape bucketing, scheduler, expert imbalance | grouped GEMM task |
| `wave-specialized-mfma-pipeline` | Wave-specialized MFMA pipeline | Composable Kernel/CK Tile CDNA kernels | producer/consumer waves, barriers, register pressure | `wave-specialized-mfma-pipeline` scaffold |
| `global-to-lds-mfma-gemm` | LDS-staged MFMA GEMM | Composable Kernel/CK Tile CDNA examples | tile staging, wait counts, LDS layout | `global-to-lds-mfma-gemm` scaffold |

### Inference, Framework, and Compiler Integration

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `miopen-frontend-fusion` | MIOpen operation graph fusion | MIOpen | graph construction, plan selection, fused pointwise ops | MIOpen task |
| `graph-captured-inference-step` | HIP Graph captured inference/decode step | HIP Graphs, MIGraphX runtime, PyTorch HIP graphs | fixed buffers, replay updates, launch amortization, graph-safe allocators | captured inference task |
| `migraphx-dynamic-plugin` | MIGraphX dynamic-shape plugin | MIGraphX plugin API | shape expressions, format support, serialization | dynamic plugin task |
| `vllm-rocm-custom-plugin` | vLLM on ROCm custom plugin | vLLM on ROCm plugins | decode metadata, KV cache, quantization, engine build | vLLM on ROCm plugin task |
| `pytorch-inductor-generated-kernel` | PyTorch Inductor generated kernel baseline | PyTorch compiler stack, Triton | generated code inspection, HIP replacement, fusion limits | `pytorch-inductor-generated-kernel` scaffold |
| `oneflow-user-op` | OneFlow user op and runtime integration | OneFlow operator stack | stream semantics, SBP/distribution, tensor layout | OneFlow op task |
| `hip-python-driver-launch` | HIP Python driver launch path | rocm-hip-python, Numba where relevant | launch overhead, module loading, interop harnesses | Python driver task |
| `onnx-custom-op-bridge` | ONNX/runtime custom-op bridge | MIGraphX parser, ONNX Runtime where applicable | shape/type contracts, plugin registration, deployment | ONNX bridge task |

### Multi-GPU, Partitioning, and Runtime Systems

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `reduce-scatter-overlap` | Reduce-scatter overlap | RCCL, framework distributed | chunking, fusion, stream priorities, dependency graphs | `reduce-scatter-overlap` scaffold |
| `allgather-overlap` | All-gather overlap | RCCL, framework distributed | staging, prefetch, tensor parallel layouts | `allgather-overlap` scaffold |
| `alltoall-moe-exchange` | All-to-all MoE token exchange | RCCL, rocSHMEM, Megatron-like references | token permutation, routing metadata, overlap | `alltoall-moe-exchange` scaffold |
| `p2p-staging` | Peer-access staging | HIP P2P, RCCL | peer access, topology, copy engines, staging buffers | P2P topology task |
| `gpudirect-rdma-boundary` | GPUDirect RDMA boundary | RCCL, UCX-like stacks | registration, pinned buffers, network overlap | RDMA boundary guide/task |
| `hip-ipc-multiprocess` | HIP IPC and multiprocess sharing | HIP IPC samples, framework serving | handle lifetime, synchronization, memory ownership | HIP IPC task |
| `partition-contention` | Multi-tenant contention and isolation | ROCm deployment docs | occupancy caps, scheduling, benchmark hygiene | isolation benchmark task |

### Tooling, Verification, and Agent Evaluation

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `rocprof-counter-roofline` | rocprofiler/rocprof counter and roofline artifact | rocprofiler/rocprof | occupancy, memory throughput, tensor utilization, stall reasons | `rocprof-counter-roofline` scaffold |
| `rocm-sanitizer-racecheck` | Sanitizer-driven correctness | rocm-sanitizer | racecheck, initcheck, synccheck, CI gating | `rocm-sanitizer-racecheck` scaffold |
| `amdisa-diff-regression` | AMD GCN ISA/LLVM IR inspection and regression | llvm-objdump/roc-objdump, amdclang++ logs | instruction mix, register drift, spills, codegen changes | `amdisa-diff-regression` scaffold |
| `autotune-parameter-sweep` | Autotuning search harness | Composable Kernel profiler, Triton autotune | tile search, warmup discipline, overfit prevention | `autotune-parameter-sweep` scaffold |
| `ulp-stability-harness` | ULP and numerical stability harness | library oracle, CPU high precision | tolerance envelopes, adversarial inputs, reductions | numeric QA task |
| `thermal-clock-benchmark` | Thermal, clock, and power benchmark hygiene | rocm-smi, DCGM where available | persistence mode, clocks, throttling labels | benchmark hygiene task |
| `occupancy-register-pressure` | Occupancy and register-pressure study | HIP occupancy API, rocprofiler/rocprof | launch bounds, spills, shared memory tradeoffs | occupancy task |

### Architecture-Specific Extension Paths

| Case ID | Case | Library Baselines / References | Custom Kernel Topics | Promotion Artifact |
| --- | --- | --- | --- | --- |
| `gfx1030-rdna2-portability` | RDNA2 fallback path | ROCm examples, MIGraphX, hipBLASLt where supported | memory traffic, wave-size assumptions, inference fallback | gfx1030 lab |
| `gfx90a-gfx942-cdna-split` | CDNA2 versus CDNA3 split | hipBLASLt, Composable Kernel, MIGraphX | `gfx90a` versus `gfx942`, MFMA, cache behavior, LDS staging | `gfx90a-gfx942-cdna-split` scaffold |
| `gfx1100-rdna3-inference` | RDNA3 inference path | MIGraphX, hipBLASLt, FlashInfer | tactic choices, L2 behavior, decode kernels | gfx1100 lab |
| `gfx942-cdna3-mfma-lab` | CDNA3 architecture-specific path | Composable Kernel/CK Tile, hipBLASLt, MIGraphX | `gfx942` MFMA, LDS staging, CK Tile | `gfx942-cdna3-mfma-lab` scaffold |
| `gfx950-gfx1200-rocm-portability` | CDNA4/RDNA4 family path | ROCm GPU support, Composable Kernel, hipBLASLt, MIGraphX | exact gfx targets, low-precision paths, compatibility checks | `gfx950-gfx1200-rocm-portability` scaffold |
| `domain-dynamic-programming` | Dynamic programming kernels | domain libraries where applicable | recurrence tiling, wave cooperation, architecture fallback | dynamic-programming task |
| `workgroup-scheduling-queue` | Workgroup scheduling and persistent queues | HIP cooperative patterns, Composable Kernel examples | scheduling, queue layout, barriers | workqueue task |

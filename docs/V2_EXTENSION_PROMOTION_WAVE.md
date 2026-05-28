# V2 Extension Promotion Wave

These waves promote the highest-leverage `v2-proposed` rows from
`docs/CASE_CATALOG.md` into template-only corpus tasks. These tasks do not claim
correctness, timing, or profiler evidence yet. They give agents concrete task
IDs, artifact slots, baseline libraries, recommended shapes, and first harness
plans.

## Wave 1 Promoted Tasks

| Task | Family | Primary Library Baselines | First Evidence Target |
| --- | --- | --- | --- |
| `paged-kv-cache-attention` | Transformer and ML fusion | vLLM on ROCm, vLLM, FlashInfer, FlashAttention | decode latency with page-table metadata |
| `gqa-mqa-decode-attention` | Transformer and ML fusion | vLLM on ROCm, vLLM, FlashInfer, FlashAttention | grouped-head decode timing |
| `varlen-packed-attention` | Transformer and ML fusion | FlashAttention varlen, MIOpen, FlashInfer, Triton | packed versus padded attention timing |
| `fused-mlp-swiglu` | Transformer and ML fusion | hipBLASLt, Composable Kernel, vLLM on ROCm, Transformer Engine on ROCm, Triton | end-to-end MLP launch and intermediate-byte count |
| `direct-convolution-2d` | Convolution and signal | MIOpen, Composable Kernel conv | direct tiled convolution versus MIOpen |
| `conv2d-fused-bias-activation` | Convolution and signal | MIOpen, MIGraphX, Composable Kernel conv epilogues | fused epilogue versus staged post-op |
| `radix-sort-key-value` | Sorting and grouping | hipCUB DeviceRadixSort, rocThrust sort | temp-storage-aware sort timing |
| `top-p-nucleus-sampling` | Sorting and grouping | vLLM on ROCm, vLLM, FlashInfer, hipCUB/rocThrust, PyTorch | logits-to-token sampling latency |
| `fp8-cast-scale-amax` | Quantization and numeric formats | Transformer Engine on ROCm, Composable Kernel, hipBLASLt | fused cast/scale/amax bandwidth |
| `block-scaled-fp4-gemm` | Quantization and numeric formats | Transformer Engine on ROCm, Composable Kernel, hipBLASLt, MIGraphX | format-aware low-precision GEMM contract |
| `int8-dot-mfma-gemm` | Quantization and numeric formats | hipBLASLt, Composable Kernel, MIGraphX, bitsandbytes | DP4A fallback versus library INT8 path |
| `structured-sparsity-2to4` | Sparse and structured sparsity | cuSPARSELt when added, Composable Kernel sparse, MIGraphX, dense GEMM | 2:4 metadata and sparse/dense comparison |
| `ragged-grouped-gemm` | Matrix Core and Composable Kernel internals | hipBLASLt, Composable Kernel grouped GEMM, Triton, vLLM on ROCm | grouped scheduler versus looped GEMMs |
| `composable-kernel-custom-epilogue` | Matrix Core and Composable Kernel internals | Composable Kernel, hipBLASLt epilogues | custom visitor versus staged postprocessing |
| `pytorch-inductor-generated-kernel` | Framework and compiler integration | PyTorch eager, torch.compile/Inductor, nvFuser, Triton | generated-kernel versus handwritten HIP boundary |
| `rocprof-counter-roofline` | Tooling and evidence | rocprofiler/rocprof, Nsight Systems | counter-backed artifact template or blocked-counter label |

## Wave 2 Promoted Tasks

| Task | Family | Primary Library Baselines | First Evidence Target |
| --- | --- | --- | --- |
| `select-filter-compact` | Sorting and grouping | hipCUB DeviceSelect, hipCUB DevicePartition, rocThrust copy_if | density-aware compact timing |
| `groupby-reduce-by-key` | Sorting and grouping | hipCUB/rocPRIM/hipCUB/rocThrust reduce-by-key, rocThrust reduce_by_key, sort plus segmented reduce | skew-aware grouped aggregation timing |
| `unique-run-length-encode` | Sorting and grouping | hipCUB DeviceRunLengthEncode, rocThrust unique/reduce_by_key | run-length distribution timing |
| `sddmm-sparse-attention-score` | Sparse and structured sparsity | cuSPARSE where applicable, Composable Kernel sparse, dense masked attention | sampled score construction timing |
| `spgemm-merge-hash` | Sparse and structured sparsity | cuSPARSE SpGEMM, custom merge/hash paths | symbolic plus numeric SpGEMM timing |
| `sparse-format-conversion` | Sparse and structured sparsity | cuSPARSE conversion APIs, hipCUB/rocThrust sort/scan | COO/CSR/BSR conversion timing |
| `reduce-scatter-overlap` | Multi-GPU and runtime systems | RCCL ReduceScatter, framework distributed collectives | phase-separated versus overlapped timeline |
| `allgather-overlap` | Multi-GPU and runtime systems | RCCL AllGather, framework distributed collectives | pipelined shard-consume timeline |
| `alltoall-moe-exchange` | Multi-GPU and runtime systems | RCCL send/recv or all-to-all equivalent, rocSHMEM, Megatron-like references | MoE exchange with permutation costs |
| `rocm-sanitizer-racecheck` | Tooling and evidence | rocm-sanitizer racecheck, initcheck, synccheck | sanitizer log artifact contract |
| `amdisa-diff-regression` | Tooling and evidence | amdclang++ logs, cuobjdump, nvdisasm, rocprofiler/rocprof summaries | register/spill/instruction diff artifact |
| `autotune-parameter-sweep` | Tooling and evidence | rocprofiler-sdk examples or Google Benchmark, Composable Kernel profiler, Triton autotune | reproducible sweep plus holdout result |
| `gfx90a-gfx942-cdna-split` | Architecture-specific extension paths | CDNA2 tuning guide, hipBLASLt, Composable Kernel, MIGraphX | same-task gfx90a versus gfx1030 comparison |
| `gfx942-cdna3-mfma-lab` | Architecture-specific extension paths | CDNA3 tuning guide, Composable Kernel/CK Tile, hipBLASLt, MIGraphX | gfx942 versus gfx950 WGMFMA/global-to-LDS staging boundary |

## Promotion Rules

- Keep these tasks `template-only` until the baseline, optimized candidate, and
  oracle are implemented.
- Record the strongest relevant library path before claiming a custom win.
- Keep HIP-event timing as `timing-only`; require attached Nsight reports for
  `counter-backed-measured`.
- Attach hardware, driver, ROCm toolkit, compile flags, library versions,
  shapes, and timer boundaries to every future result.

## Next Promotion Candidates

The next v2 rows to promote should cover `memory-pool-allocator`,
`strided-batched-layout-transform`, `depthwise-separable-convolution`,
`implicit-gemm-convolution`, `normalization-backward`,
`multi-tensor-adamw`, `splitk-reduction-gemm`, `warp-specialized-wgmma`,
`tma-multicast-gemm`, `miopen-frontend-fusion`,
`vllm-rocm-custom-plugin`, `cuda-ipc-multiprocess`, and
`sm100-sm120-blackwell-path`.

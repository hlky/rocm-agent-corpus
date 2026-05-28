# Roadmap

The corpus goal is to help agents write custom HIP kernels that beat, match,
or narrowly specialize beyond hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, hipCUB/rocPRIM/hipCUB/rocThrust, MIGraphX,
Triton, and framework-generated kernels. Vendor libraries remain essential, but
they are baselines, correctness oracles, profiling targets, implementation
references, and extension surfaces rather than default stopping points.

The staged operating plan lives in
[`docs/CORPUS_EXPANSION_STAGES.md`](CORPUS_EXPANSION_STAGES.md). Use that file
to decide what to build next, what evidence each task needs, and how to keep
library material self-contained while preserving the custom-kernel competition
focus.

## Current Priorities

- Stage 0: align navigation, local skills, and guide language around custom
  kernel competition rather than vendor-library delegation.
- Stage 1: expand measured microkernels with correctness tests, timing-only
  records, negative examples, and concise "why it won/lost" notes.
- Stage 2: build hipCUB/rocPRIM/hipCUB/rocThrust competitor tasks for reductions, scans, histograms,
  and transform-reduce workloads.
- Stage 3: build GEMM/hipBLAS/rocBLAS/hipBLASLt competitor tasks, especially small fixed
  shapes, epilogue fusion, batched/grouped GEMM, and layout-specialized cases.
- Stage 4: add Composable Kernel custom epilogue and CK Tile examples that agents can modify
  or mine for handwritten kernels.
- Stage 5: add MIGraphX plugin and inference-kernel tasks for fused activation,
  normalization, quant/dequant, KV-cache, top-k, and small-batch inference.
- Stage 6: add framework and Triton competitor tracks using PyTorch/OneFlow and
  Triton as references and baselines.
- Stage 7: run GPU/GFX architecture sweeps with exact hardware, compiler, and
  build metadata.
- Stage 8: upgrade timing-only records to rocprofiler/rocprof counter-backed records
  when counter access is available.
- Stage 9: turn the harness into an automated eval system for human-written and
  agent-written kernels.

## Near-Term Task Additions

- hipCUB DeviceReduce versus custom block/warp reduction.
- hipCUB DeviceScan versus fixed-shape or fused custom scan.
- rocThrust transform-reduce versus fused custom kernel.
- Prefix scan: naive, block scan, hipCUB DeviceScan, and fused consumer variants.
- LayerNorm and RMSNorm row reductions with framework, Triton, and custom HIP
  competitors.
- Rowwise softmax and log-softmax with HIP C++, Triton, and framework
  baselines.
- GEMM ladder: naive, shared-memory tiled, tensor-core custom, hipBLAS/rocBLAS,
  hipBLASLt, Composable Kernel.
- GEMM plus bias, activation, residual, scaling, and quant/dequant epilogues.
- Small fixed-shape, batched, strided-batched, and grouped GEMM.
- HIP Graph launch overhead benchmark for launch-bound custom kernels.
- global-to-LDS staging and architecture-specific tiled copy educational kernels.
- Stencil with shared-memory halo and awkward boundary cases.
- MIGraphX plugin benchmark for fused activation, LayerNorm, RMSNorm, or
  quant/dequant.
- vLLM on ROCm engine/runtime notes for small-batch and decode-heavy inference.
- Triton rowwise softmax and normalization versus HIP C++ custom kernels.
- OneFlow HIP operator anatomy case study with a custom competitor.
- Architecture-specific compile-flag and shape sweeps for available GPUs.

## Self-Contained Library Reference Tracks

Library tracks should explain how to run the strong baseline, how to inspect it,
and how to modify or beat it next.

- hipBLASLt algorithm search, workspace sweeps, layouts, and epilogues.
- Composable Kernel custom epilogue visitors, CK Tile layout algebra, tile selection, and
  architecture dispatch.
- hipCUB BlockReduce, BlockScan, warp primitives, and Device-level APIs as both
  baselines and parts inside custom kernels.
- MIOpen attention plan construction and comparison points for custom
  attention-adjacent kernels.
- FlashAttention implementation notes as a reference for IO-aware custom kernel
  design.
- MIGraphX plugin lifecycle, vLLM on ROCm KV cache, batching, quantization, and
  runtime overhead.
- OneFlow operator/runtime internals and framework dispatch costs.
- Triton compiler behavior, generated-code inspection, and HIP C++ competitor
  translation notes.
- RCCL and rocSHMEM examples for communication-bound custom-kernel contexts.

## Evidence Upgrades

- Keep every HIP-event-only result labeled as timing-only.
- Run current tasks on a machine with rocprofiler/rocprof counter access.
- Add counter-backed records for current timing-only tasks.
- Add shape sweeps for awkward sizes, small launch-bound sizes, and
  architecture-sensitive breakpoints.
- Add at least two GPU architectures for core task families.
- Track compiler flags, LLVM IR / AMD GCN ISAAS output, and LLVM IR / AMD GCN ISA snippets for low-level
  examples.
- Add architecture-specific records for gfx1030, gfx1100, gfx942/gfx950, and newer
  CDNA4/RDNA4 targets as hardware is available.

## Navigation Upgrades

- Add generated Markdown or HTML indexes.
- Add task tags to a searchable SQLite or JSON index.
- Add "choose your optimization" decision tables for reductions, scans, GEMM,
  epilogues, inference plugins, Triton competitors, and architecture sweeps.
- Add or refine local skills for corpus navigation, custom-kernel competition,
  Composable Kernel, hipBLASLt, rocPRIM/hipCUB/rocThrust/hipCUB, MIGraphX plugins, Triton translation, and
  profiler triage.

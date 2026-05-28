# V2 Extension Wave 3

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


Wave 3 promotes the next set of extended CASE_CATALOG rows into concrete
template-only tasks. The goal is to keep expanding the corpus toward custom
kernel competition against MIOpen, Composable Kernel, hipBLASLt, hipCUB, MIGraphX, vLLM on ROCm,
framework compilers, HIP Graphs, and architecture-specific paths.

## Promoted ROCm Task/Equivalent Set

CUDA-specific rows from the source corpus are mapped to ROCm equivalents when
the AMD path is an existing task or architecture lab rather than a new task
directory.

| Task | Family | Primary Baselines | First Evidence Target |
| --- | --- | --- | --- |
| `strided-batched-layout-transform` | Memory/layout | rocThrust, hipCUB BlockLoad/BlockStore, framework tensor copies | stride-aware bandwidth and alignment sweep |
| `packed-layout-transcode` | Memory/layout + quantization | Composable Kernel layouts, vLLM on ROCm, Transformer Engine | packed int4/fp8 transcode bytes and correctness |
| `memory-pool-allocator` | Runtime systems | HIP memory pools, framework caching allocators | first HIP allocator timing pass |
| `implicit-gemm-convolution` | Convolution | MIOpen, Composable Kernel convolution, MIGraphX | implicit-GEMM conv versus direct/staged baseline |
| `depthwise-separable-convolution` | Convolution | MIOpen, MIGraphX, framework kernels | depthwise fusion and channel-pack sensitivity |
| `normalization-backward` | Transformer training | PyTorch, Transformer Engine, Triton | dgamma/dbeta and dx reduction timing |
| `multi-tensor-adamw` | Optimizers | PyTorch foreach/fused, Apex-style fused Adam, bitsandbytes | tensor-list launch amortization |
| `splitk-reduction-gemm` | GEMM/Matrix Core | hipBLASLt, Composable Kernel, Triton | split-K partial reduction boundary |
| `cdna-mfma-gemm` / `gfx942-cdna3-mfma-lab` | CDNA Matrix Core | hipBLASLt, rocBLAS, Composable Kernel/CK Tile | MFMA and LDS-staging boundary |
| `wave-specialized-mfma-pipeline` | Architecture pipeline | hipBLASLt, Composable Kernel/CK Tile | producer/consumer wave-role MFMA pipeline metadata |
| `global-to-lds-mfma-gemm` | Architecture GEMM | hipBLASLt, rocBLAS, Composable Kernel/CK Tile | LDS-staged MFMA GEMM versus library baseline |
| `gfx950-gfx1200-rocm-portability` | Architecture portability | AMD ROCm GPU support, hipcc target list, libraries | CDNA4/RDNA4 compile and dispatch matrix |
| `miopen-frontend-fusion` | Inference/framework | MIOpen, MIGraphX, custom HIP | op-graph fusion versus handwritten fused op |
| `vllm-rocm-custom-plugin` | LLM inference | vLLM on ROCm, vLLM, FlashInfer | plugin/kernel boundary for decode-side fusion |
| `hip-ipc-multiprocess` | Multi-process systems | HIP IPC samples, framework serving | IPC handle setup, ownership, and sync overhead |
| `graph-captured-inference-step` | Runtime systems | HIP Graphs, MIGraphX runtime, PyTorch HIP graphs | replay latency with fixed buffers |
| `architecture/gfx950_cdna4`, `architecture/gfx1200_rdna4` | Architecture | AMD tuning guides, Composable Kernel, hipBLASLt, MIGraphX | CDNA4/RDNA4 portability checklist |

## Promotion Rules

- Keep every row `template-only` until a correctness oracle and benchmark harness
  exist.
- Separate library baselines from custom-kernel hypotheses. A task may use a
  library as a baseline, oracle, or extension point without making the corpus
  answer "just call the library."
- Require timing-only labels for HIP-event or framework-timer measurements.
- Require profile artifacts before making rocprofiler/rocprof counter claims.
- Preserve negative examples when the custom candidate loses.

## Next Evidence Push

The first runnable harnesses should prioritize:

- Continue `normalization-backward`, because the HIP harness and PyTorch/Triton
  baseline runners are scaffolded but need AMD hardware measurements.
- `implicit-gemm-convolution`, because MIOpen/Composable Kernel coverage is one of the
  biggest remaining library gaps.
- `graph-captured-inference-step`, because it connects launch overhead, MIGraphX,
  PyTorch HIP graphs, and fixed-buffer serving.

## Evidence Status

No CUDA-origin result artifacts were copied into the ROCm corpus. The newly
ported Wave 3 rows are `template-only` or harness-ready until an AMD run writes
result artifacts with hardware, ROCm version, compiler flags, gfx target, and an
explicit evidence label.

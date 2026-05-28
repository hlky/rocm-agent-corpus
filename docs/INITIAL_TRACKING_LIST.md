# Initial Tracking List

This is the preserved seed list for expanding HIP/ROCm Agent Corpus into a custom
kernel competition corpus: everything an agent needs to understand, challenge,
match, specialize beyond, or extend hipBLAS/rocBLAS, hipBLASLt, Composable Kernel, hipCUB/rocPRIM/hipCUB/rocThrust,
MIGraphX, vLLM on ROCm, Triton, and framework-generated kernels.

Status key:

- `todo`: not started.
- `scaffolded`: guide, manifest, or skeleton exists.
- `measured`: has at least one timing/correctness record.
- `counter-backed`: has rocprofiler/rocprof or equivalent counter evidence.

## 1. Competitor Baseline Tracks

| Track | Target Coverage | Status |
| --- | --- | --- |
| hipBLAS/rocBLAS/hipBLASLt | GEMM, batched GEMM, grouped GEMM, GEMM epilogues, TF32/FP16/BF16/FP8/INT8 | scaffolded |
| hipCUB/rocPRIM/hipCUB/rocThrust | reduce, scan, segmented reduce, histogram, radix sort, select/top-k | measured seeds for reduction, scan, histogram; remaining families scaffolded |
| Composable Kernel/CK Tile | custom epilogues, CK Tile layouts, grouped GEMM, CDNA3/CDNA4/RDNA4 global-to-LDS staging/MFMA paths | scaffolded |
| MIGraphX | plugins, dynamic shapes, precision, tactic selection, engine/runtime timing | scaffolded |
| vLLM on ROCm | KV cache ops, logits/top-k/sampling, quant/dequant, RMSNorm/LayerNorm, batching | scaffolded |
| Triton | HIP C++ competitor pairs for ML kernels and generated-code inspection | scaffolded |

## 2. Kernel Families

| Family | Target Coverage | Status |
| --- | --- | --- |
| LayerNorm / RMSNorm | forward, backward, framework/Triton/library/custom comparisons | ROCm rerun needed |
| Softmax family | softmax, log-softmax, masked softmax, online softmax | measured |
| Attention | naive, tiled, FlashAttention-style, prefill/decode split | ROCm rerun needed |
| Selection and sampling | top-k, argmax, sampling, beam-search helpers | ROCm rerun needed |
| Positional and KV ops | RoPE, ALiBi, KV-cache update/read, paged KV cache | todo |
| Quantization | int8, int4, fp8, nvfp4-style notes, quant/dequant fusion | ROCm rerun needed |
| MoE | routing, token permutation/unpermutation, grouped GEMM dispatch | todo |
| Stencils/convolution | stencil, convolution microkernels, im2col-free paths | todo |
| Prefix work | scan, segmented scan, histogram, warp-aggregated atomics | ROCm rerun needed |
| Small GEMM | fixed-shape GEMMs where custom kernels can beat library overhead | ROCm rerun needed; library and Matrix Core competitors still needed |

## 3. Architecture-Specific Labs

| Lab | Target Coverage | Status |
| --- | --- | --- |
| `gfx90a` CDNA2 datacenter | MFMA, LDS staging, Matrix Cores | scaffolded |
| `gfx1030` RDNA2 pro/consumer | RX 6000/MI210-adjacent portability checks | scaffolded |
| `gfx1100` RDNA3 | Radeon 7000 inference paths | scaffolded |
| `gfx942` CDNA3 portable | MI300-class CDNA3 path | scaffolded |
| `gfx950` CDNA4 datacenter | exact ROCm/library support pending hardware | scaffolded |
| `gfx1200` RDNA4 workstation | exact ROCm/library support pending hardware | scaffolded |

Each lab should include compile flags, compatibility notes, Matrix Core paths,
shared-memory limits, register pressure notes, architecture-specific features,
known library support status, and measured records per GPU.

## 4. Agent Evaluation Harness

| Component | Target Coverage | Status |
| --- | --- | --- |
| Task schema | baseline, optimized, hidden shapes, competitor baselines | scaffolded |
| Submission runner | compile, test, benchmark submitted kernels | scaffolded |
| Shape sweeps | powers of two, awkward shapes, small launch-bound cases | scaffolded |
| Competitor comparisons | library, framework, Triton, prior custom kernels | scaffolded |
| Result classifier | win, loss, neutral, correctness fail, compile fail | scaffolded |
| Attempt archive | store patch, explanation, output, metadata, failure mode | scaffolded |
| Reports | Markdown/JSON summaries for humans and agents | scaffolded |

## 5. Extra Submodules To Consider

| Repo | Why It Matters | Status |
| --- | --- | --- |
| `AMD/ROCm/TransformerEngine` | fp8/transformer kernels and quantization paths | scaffolded |
| `AMD/ROCm/Fuser` or nvFuser-adjacent repos | framework fusion competitors and generated kernels | scaffolded |
| `vllm-project/vllm` | LLM serving, paged attention, KV cache, scheduling | scaffolded |
| `flashinfer-ai/flashinfer` | inference kernels, attention, sampling, paged KV | scaffolded |
| `pytorch/pytorch` | framework HIP kernels and extension targets | tracked as source, submodule deferred |
| `bitsandbytes-foundation/bitsandbytes` | quantization kernels and optimizers | scaffolded |
| `AMD/ROCm/Megatron-LM` | transformer training/inference patterns | scaffolded |
| `ROCm/rocm-hip-python` | Python HIP integration and driver/runtime APIs | source candidate |
| `NVlabs/tiny-cuda-nn` | CUDA-first compact-kernel contrast; port before ROCm use | source candidate |
| `jax-ml/jax` | XLA/Pallas/framework compiler competitor path | tracked as source, submodule deferred |

## 6. Recommended Next Spine

Build the GEMM/hipBLASLt/Composable Kernel competition track next. It should become the
spine of the corpus because attention, MLPs, MoE, quantization, MIGraphX
plugins, and many framework kernels reduce to matmul-adjacent primitives.

Minimum next deliverables:

- hipBLASLt baseline harness with algorithm search metadata. Status: scaffolded.
- Small fixed-shape custom GEMM task. Status: scaffolded example, task pending.
- GEMM + bias + activation epilogue task. Status: planned in GEMM track.
- Composable Kernel custom epilogue example that an agent can modify. Status: scaffolded.
- Records comparing naive, custom, hipBLASLt, and Composable Kernel paths. Status: pending.

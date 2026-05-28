# PyTorch Public API Kernel Catalog

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This catalog maps PyTorch public APIs to HIP kernel families that agents can
target with custom kernels. It is not a full copy of the PyTorch reference API.
It is the CUDA-relevant slice: operations that launch kernels, dispatch to CUDA
libraries, or form common generated-kernel graphs.

This catalog is not automatic benchmark admission. A public API listed here is
only `challenge-admitted` after it appears in
`docs/PYTORCH_API_REAL_WORLD_SHAPE_MATRIX.md` with a sourced real-world problem
size. APIs without a matrix row remain `catalog-only` and should not be used as
PyTorch-baseline challenge tasks.

Observed on 2026-05-21. The PyTorch stable docs redirected to the 2.12 docs in
this environment. Primary references:

- https://docs.pytorch.org/docs/2.12/torch.html
- https://docs.pytorch.org/docs/2.12/nn.functional.html
- https://docs.pytorch.org/docs/2.12/nn.html
- https://docs.pytorch.org/docs/2.12/linalg.html
- https://docs.pytorch.org/docs/2.12/fft.html
- https://docs.pytorch.org/docs/2.12/sparse.html
- https://docs.pytorch.org/docs/2.12/special.html
- https://docs.pytorch.org/docs/2.12/masked.html
- https://docs.pytorch.org/docs/2.12/nested.html
- https://docs.pytorch.org/docs/2.12/distributions.html

## How To Use This Catalog

Treat PyTorch as the first public-contract baseline and correctness oracle.
The goal is not to say "PyTorch already does it." The goal is to define the
observable contract and then beat it by specializing shape, dtype, layout,
fusion boundary, launch pattern, allocation behavior, or architecture.

For every candidate:

1. Identify the public API contract.
2. Record dtype, layout, stride, tie, NaN, RNG, and determinism semantics.
3. Time `pytorch-eager-public-api` on the same GPU.
4. Optionally time `torch.compile` or a framework-generated path as a stronger
   PyTorch-family baseline.
5. Only then compare custom HIP, Triton, Composable Kernel, hipCUB, MIOpen, hipBLASLt, or
   MIGraphX paths.

## CUDA-Relevant API Categories

| Category | Public API examples | Kernel families | Custom attack surface |
| --- | --- | --- | --- |
| Tensor creation, fill, copy, casts | `empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `rand`, `randn`, `to`, `copy_` | Fill, RNG, cast, copy, layout conversion | Fuse fill/cast/copy into first consumer; vectorized stores; shape-specialized RNG; avoid temporary allocation |
| Shape and layout plumbing | `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `stack`, `split`, `chunk`, `squeeze`, `unsqueeze`, `tile` | Metadata-only views, materializing copies, concat/split kernels | Eliminate materialization, fuse concat/split with producer/consumer, specialize strides and alignment |
| Pointwise unary/binary/ternary | `abs`, `neg`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `add`, `mul`, `div`, `addcmul`, `clamp`, `where` | Elementwise map, broadcast map, vectorized map | Fuse long chains, use vectorized loads/stores, keep values in registers after reductions/GEMM epilogues |
| Comparison, logical, bitwise | `eq`, `ne`, `lt`, `gt`, `isnan`, `isinf`, `logical_and`, `bitwise_and` | Predicate maps, masks, bit packing | Fuse mask generation with select/scatter/reduce; bit-pack masks for dense boolean traffic |
| Reductions and statistics | `sum`, `prod`, `mean`, `amax`, `amin`, `max`, `min`, `argmax`, `argmin`, `var`, `std`, `norm`, `logsumexp`, `all`, `any`, `count_nonzero` | Warp/block reductions, multi-pass reductions, arg reductions | Fixed row width, one CTA per row, vectorized reductions, fused scale/bias/residual, custom tie handling |
| Prefix and cumulative ops | `cumsum`, `cumprod`, cumulative log/prob patterns | Block scan, device scan, segmented scan | Fixed segment scan, short-row scan in one CTA, fuse scan with compaction or sampling |
| Selection, sorting, uniqueness | `topk`, `kthvalue`, `sort`, `argsort`, `msort`, `unique`, `unique_consecutive`, `bucketize`, `searchsorted` | Top-k, partial sort, radix sort, binary search, run-length encode | Tiny-k register selection, radix select, row-specialized large-k, fused sampling, known sortedness/tie policy |
| Indexing, gather, scatter | `gather`, `take`, `take_along_dim`, `index_select`, `index_add`, `index_copy`, `index_reduce`, `scatter`, `scatter_add`, `scatter_reduce`, `masked_select`, `masked_fill`, `nonzero` | Gather/scatter, atomics, compaction, segmented updates | Coalesce known index patterns, dedupe repeated ids, privatize atomics, fuse index generation and consumer |
| Dense matmul and BLAS-like ops | `mm`, `matmul`, `bmm`, `addmm`, `baddbmm`, `chain_matmul`, `einsum`, `inner`, `outer`, `mv`, `addr` | GEMM, batched GEMM, GEMV, epilogue fusion | Fixed shapes, small batches, custom epilogues, low-bit dequant, split-K, grouped/ragged GEMM |
| Linear algebra solvers/decompositions | `torch.linalg.solve`, `inv`, `cholesky`, `qr`, `svd`, `eig`, `eigh`, `lu_factor`, `lstsq`, `matrix_exp`, `norm` | cuSOLVER/hipBLAS/rocBLAS-backed factorizations, batched small matrix kernels | Small fixed matrices, known symmetry/triangularity, batched-specialized kernels, preconditioned shortcuts; current sourced candidate is Helios fixed Cholesky noise |
| FFT and spectral | `torch.fft.fft`, `ifft`, `rfft`, `irfft`, `fftn`, `fftshift`, `stft`, `istft` | cuFFT plans, windowing, transpose, complex pointwise | Fuse windowing/postprocessing, fixed-size FFT adjacency, avoid layout transposes |
| Sparse and semi-structured sparse | sparse tensor constructors, `sparse.mm`, `sparse.addmm`, `sparse.sampled_addmm`, `sparse.softmax`, `to_sparse_semi_structured` | SpMM, SpMV, SDDMM, sparse softmax, 2:4 sparse GEMM | Known sparsity pattern, block sparse attention, format conversion, cuSPARSELt 2:4 metadata paths |
| Neural network convolution | `torch.nn.Conv1d/2d/3d`, `conv1d/2d/3d`, transposed conv | MIOpen/Composable Kernel conv, direct conv, implicit GEMM, depthwise | Fixed filters, depthwise/separable, bias/activation fusion, small spatial maps, NHWC islands |
| Pooling, resampling, vision grids | max/avg/adaptive pooling, `interpolate`, `grid_sample`, `affine_grid`, `pixel_shuffle`, `unfold`, `fold`, padding ops | Window reductions, stencil-like sampling, gather/interpolation | Fixed window, fused resize/normalize, bounds-specialized sampling, layout-aware image paths |
| Normalization and activations | `layer_norm`, `group_norm`, `batch_norm`, `rms_norm`, `relu`, `gelu`, `silu`, `softplus`, `glu`, `prelu` | Row reductions plus pointwise epilogues, activation maps | Residual+norm fusion, fp32 accumulation with half output, one-pass RMSNorm, activation inside GEMM epilogue |
| Softmax, losses, probabilities | `softmax`, `log_softmax`, `cross_entropy`, `nll_loss`, `mse_loss`, `binary_cross_entropy`, `kl_div` | Row reductions, logsumexp, gather target, reductions | Fused logits processing, target gather, label-smoothing specialization, sampling integration |
| Attention and transformer helpers | `scaled_dot_product_attention`, `MultiheadAttention`, Transformer modules | SDPA, FlashAttention-style online softmax, matmul-softmax-matmul | Fixed head dim, causal/sliding masks, GQA/MQA, paged KV cache, fused RoPE/QK norm |
| Random and sampling | `bernoulli`, `multinomial`, `normal`, `poisson`, `randperm`, distribution sampling | RNG, prefix sums, alias tables, rejection sampling, Top-K/Top-P | Seed replay, fused logits transform + select + sample, per-row RNG state, distributional correctness tests |
| Special, masked, nested, distributions | `torch.special.logit`, `torch.special.*`, `torch.masked.mean`, `torch.masked.var`, `torch.masked.*`, `torch.nested.*`, `torch.distributions.*` | Special-function pointwise, masked reductions, ragged/nested dispatch, sampling/log-prob | Fuse mask and math, specialize ragged batches, replace generic nested dispatch for known sequence buckets |
| Optimizer and foreach APIs | `torch.optim.*`, `_foreach_*` public beta APIs | Multi-tensor pointwise, reductions, norm clipping, RDNA3mW updates | Launch amortization, grouped tensor lists, fused weight decay/update, mixed precision state packing |
| Quantization and low precision | quantize/dequantize APIs, fake quant, autocast/AMP surfaces | Cast/scale/amax, pack/unpack, int8/int4 GEMM adjacency | Fused dequant with GEMV/GEMM, FP8 scale/amax fusion, block-scaled formats |
| Distributed and collective-adjacent | `torch.distributed` collectives and distributed tensor APIs | RCCL collectives, overlap, reduce-scatter/all-gather | Chunked compute/comm overlap, fused reduce+optimizer, topology-aware staging |
| Compiler and extension surfaces | `torch.compile`, `torch.library`, custom ops, `torch.utils.cpp_extension`, profiler/benchmark utils | Generated kernels, custom operator ABI, extension build, benchmarking | Use generated kernels as baselines; replace hot op/subgraph with HIP extension; preserve stream/stride contracts |

## Kernel Challenge Families From PyTorch API

The public API collapses into a smaller set of HIP/ROCm challenge families:

| Challenge family | PyTorch contract examples | Shape matrix refs | Baseline competitors beyond PyTorch | First custom-kernel focus |
| --- | --- | --- | --- | --- |
| map_fusion | `where(relu(x * scale + bias), ...)` | `pt_pointwise_cfg_scheduler` | TorchInductor/Triton | vectorized pointwise chain, no temporaries |
| row_reduction | `sum`, `amax`, `argmax`, `logsumexp` | `pt_reduction_norm_rows` | hipCUB, Triton | one-row-per-CTA, fixed width, stable reductions |
| norm_softmax | `layer_norm`, `rms_norm`, `softmax` | `pt_reduction_norm_rows`, `pt_softmax_attention_rows` | framework kernels, Triton, vLLM on ROCm | fused residual/norm or masked softmax |
| selection_sampling | `topk`, `sort`, `multinomial` | `pt_selection_sampling_rows` | hipCUB, vLLM on ROCm, vLLM, FlashInfer | tiny-k, wide-k, Top-P, sampling fusion |
| gather_scatter | `gather`, `index_select`, `scatter_add` | `pt_gather_scatter_compaction` | hipCUB/rocThrust, framework kernels | coalesced/deduped index patterns |
| dense_linear | `matmul`, `addmm`, `linear` | `pt_dense_linear_gemm_rows` | hipBLASLt, Composable Kernel, Triton | fixed shape, custom epilogue, grouped/ragged |
| convolution_codec | `conv2d`, `conv3d`, `interpolate`, VAE blocks | `pt_conv_resample_codec_rows` | MIOpen, Composable Kernel conv, MIGraphX | small fixed conv islands, layout specialization |
| attention_decode | `scaled_dot_product_attention`, MHA | `pt_softmax_attention_rows` | FlashAttention, MIOpen, vLLM on ROCm, vLLM, FlashInfer | GQA decode, paged KV, RoPE/QK norm fusion |
| position_scan_rope | `cumsum`, `arange`, `sin`, `cos`, RoPE/position helpers | `pt_position_scan_rope_rows` | hipCUB scan, TorchInductor/Triton, FlashAttention/serving prelude code | fixed segment scans, precomputed/cached position tables, fused RoPE + KV write |
| sparse_irregular | sparse APIs, `segment_reduce`, `nonzero` | `pt_sparse_block_extension` candidate | cuSPARSE, hipCUB, Composable Kernel sparse | fixed sparse formats, block sparse, compaction |
| optimizer_multi_tensor | `optim.RDNA3mW`, `_foreach_*` | blocked until sourced | framework fused optimizers, bitsandbytes | multi-tensor launch amortization |
| fft_signal | `fft`, `stft`, audio codecs | `pt_fft_stft_audio_frontend` candidate | cuFFT, torchaudio kernels | fixed windows and fused pre/post transforms |

## Public API Categories Needing Exact Semantics

Agents must not optimize these by intuition alone:

- `topk`, `sort`, `argmax`: tie ordering, sorted output, NaN behavior, index
  dtype, and stable/unstable ordering matter.
- `softmax`, `log_softmax`, reductions: accumulation dtype and numerical
  stability define correctness.
- `multinomial` and RNG APIs: exact replay may be impossible across custom
  kernels unless the task defines seed/state semantics.
- `scatter_add`, `index_add`, `index_reduce`: duplicate indices and atomic
  ordering can change floating-point results.
- sparse APIs: layout invariants, coalesced state, compressed index contracts,
  and 2:4 shape constraints are part of the public behavior.
- `torch.compile`: generated code is a baseline path, not a semantic contract by
  itself. The contract remains the original public PyTorch function/module.

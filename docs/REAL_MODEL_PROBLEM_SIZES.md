# Real Model Problem Sizes

> ROCm parity note: This file was adapted from `H:/cuda-agent-corpus` on 2026-05-28. CUDA-origin benchmark numbers, source paths, and library names are comparison context only; do not cite them as ROCm evidence. New ROCm measurements must carry AMD hardware, ROCm version, compiler flags, gfx target, and an explicit evidence label.


This document extracts ROCm-relevant shapes from the local Diffusers and
Transformers audit reports under:

- `H:/dinoml_v2_agents/agents/plans/diffusers/*/report.md`
- `H:/dinoml_v2_agents/agents/plans/transformers/*/report.md`

No benchmark result is claimed here. These are problem-size anchors for future
PyTorch-baseline challenge tasks.

For API-level admission, also read
`docs/PYTORCH_API_REAL_WORLD_SHAPE_MATRIX.md`. That file maps public PyTorch API
contracts to the real shapes below plus additional source-specific rows.

## Extraction Rule

Use these shapes as challenge inputs only after defining the exact PyTorch
reference graph and correctness policy. A report shape is evidence that the
size is model-real; it is not evidence that a custom kernel is faster.

## Diffusers And Generative Models

| Shape ID | Source | Runtime tensors | Kernel pressure |
| --- | --- | --- | --- |
| `sd15_vae_decode_512` | `diffusers/stable_diffusion_1_5/report.md:186` | latent `[B,4,64,64]` -> image `[B,3,512,512]`; prompt `[2B,77,768]` with CFG | Conv2d, GroupNorm, SiLU, nearest upsample, VAE decode |
| `sd15_unet_cfg_512` | `diffusers/stable_diffusion_1_5/report.md:151` | UNet input `[2B,4,64,64]`, channels 320/640/1280/1280, cross-attn dim 768 | Conv/resnet blocks, cross-attention, CFG concat/chunk |
| `sdxl_unet_cfg_1024` | `diffusers/stable_diffusion_xl/report.md:117` | prompt `[B,77,2048]`, latent `[B,4,128,128]`, CFG input `[2B,4,128,128]` | SDXL cross-attention, UNet conv/norm, scheduler broadcasts |
| `sdxl_vae_decode_1024` | `diffusers/stable_diffusion_xl/report.md:170` | `[B,4,128,128]` -> `[B,3,1024,1024]`, scale `0.13025` | VAE conv island, force-upcast boundary |
| `sd3_transformer_1024` | `diffusers/stable_diffusion_3/report.md:173` | latent `[B,16,128,128]`, patch size 2 -> 4096 image tokens, inner 1536 for medium | Patch embed/unpatchify, joint image+text attention, QK norm |
| `flux_transformer_1024` | `diffusers/flux/report.md:155` | VAE latent `[B,16,128,128]` packed to `[B,4096,64]`; T5 `[B,L,4096]`, often `L=512`; CLIP pooled `[B,768]` | 19 dual + 38 single transformer layers, 24 heads, head dim 128 |
| `flux_vae_decode_1024` | `diffusers/flux/report.md:164` | unpacked `[B,16,128,128]` -> image `[B,3,1024,1024]`; scale/shift `0.3611/0.1159` | VAE decode, latent shift/scale fusion |
| `cogvideox_t2v_49f_720x480` | `diffusers/cogvideox/report.md:179` | 49 frames -> 13 latent frames; latents `[B,13,16,60,90]`; joint sequence about 17,776 tokens | Video transformer attention, patch/unpatchify, Conv3d/VAE |
| `wan_i2v_81f_480x832` | `diffusers/wan/report.md:202` | latents `[B,16,21,60,104]`, patch tokens `[B,21*30*52,inner]`, decode `[B,3,81,480,832]` | NCTHW Conv3d patchify, video attention, per-channel latent mean/std |
| `wan_i2v_condition_81f` | `diffusers/wan/report.md:216` | image `[B,3,H,W]`, condition `[B,16,T_lat,H/8,W/8]`, mask `[B,4,T_lat,H/8,W/8]`, concat `[B,36,T_lat,H/8,W/8]` | Channel concat, mask construction, VAE encode, added K/V branch |
| `sana_1024_linear_attention` | `diffusers/sana/report.md:180` | prompt `[B,300,2304]`, latents `[B,32,32,32]`, 1024 image tokens | Linear attention, high channel latent, CFG batch |
| `sana_4k_linear_attention` | `diffusers/sana/report.md:192` | latents `[B,32,128,128]`, 16,384 image tokens | Long image-token attention and memory pressure |
| `stable_audio_latent_1024` | `diffusers/stable_audio/report.md:161` | latent `[B_eff,64,1024]`, self-attn over 1025 tokens, cross-attn to 130 condition tokens | 1D conv/audio codec, GQA attention, CFG/audio length handling |
| `longcat_audio_5_15_30s` | `diffusers/longcat_audio_dit/report.md:657` | latent lengths 58, 176, 351 for 5, 15, 30 seconds; channels 64; prompt `L <= 512` | Audio DiT attention, duration-specialized kernels |

## Transformers And Multimodal Models

| Shape ID | Source | Runtime tensors | Kernel pressure |
| --- | --- | --- | --- |
| `llama2_7b_prefill_decode` | `transformers/llama/report.md:77` | H=4096, layers=32, QH/KVH=32/32, D=128, V=32000, contexts 128/1024/4096 | RMSNorm, QKV GEMM, RoPE, MHA attention, LM head |
| `llama3_8b_gqa` | `transformers/llama/report.md:77` | H=4096, QH/KVH=32/8, D=128, V=128256, max pos 8192 | GQA attention, KV cache without repeat materialization, large vocab logits |
| `apertus_8b_long_context` | `transformers/apertus/report.md:79` | H=4096, layers=32, QH/KVH=32/8, V=131072, context 65536 | Long-context GQA decode, last-token LM head, large vocab selection |
| `qwen3_8b_dense` | `transformers/qwen3/report.md:115` | H=4096, layers=36, QH/KVH=32/8, D=128, V=151936, max pos 40960 | Q/K head RMSNorm, GQA, very large LM head, Top-K 20 generation |
| `qwen3_4b_2507_long` | `transformers/qwen3/report.md:115` | H=2560, QH/KVH=32/8, V=151936, max pos 262144, RoPE theta 5e6 | Long-context RoPE/cache, tied embedding variants |
| `qwen3_moe_30b_a3b` | `transformers/qwen3_moe/report.md:115` | H=2048, layers=48, QH/KVH=32/4, experts=128, top-k=8 | Router fp32 softmax `[M,128]`, Top-K 8, grouped expert GEMMs |
| `qwen3_moe_235b_a22b` | `transformers/qwen3_moe/report.md:115` | H=4096, layers=94, QH/KVH=64/4, experts=128, top-k=8 | Extreme MoE routing, grouped GEMM, scatter-add |
| `mixtral_8x7b_moe` | `transformers/mixtral/report.md:83` | H=4096, QH/KVH=32/8, D=128, experts=8/top-2, context 32768 | Router `4096->8`, top-2, per-expert gate/up/down GEMMs |
| `whisper_large_v3` | `transformers/whisper/report.md:62` | H=1280, 32 encoder + 32 decoder layers, heads=20, D=64, mel frames 3000 -> 1500 tokens | Conv1d frontend, encoder/decoder attention, decoder LM logits |
| `vit_base_patch16_224` | `transformers/vit/report.md:52` | 224x224, patch16 -> 196 patches + CLS, H=768, heads=12 | Patch Conv2d/GEMM, self-attention length 197 |
| `vit_large_patch32_384` | `transformers/vit/report.md:75` | 384x384, patch32 -> 144 patches + CLS, H=1024 | Large patch GEMM and transformer block |
| `clipseg_352_mask` | `transformers/clipseg/report.md:122` | image 352x352, 22x22 grid, 485 vision tokens, text `[B,<=77,512]`, logits `[B,352,352]` | Vision/text attention, decoder mask upsample/head |
| `detr_resnet50_800_1333` | `transformers/detr/report.md:70` | feature tokens about 25x42=1050, DC5 about 50x84=4200, object queries 100, d_model 256 | Encoder attention, decoder cross-attention, box/class heads |
| `mamba2_decode` | `transformers/mamba2/report.md:56` | H=4096, expand=2, 128 SGFX heads, vocab 32768, recurrent state `[B,num_heads,head_dim,state]` | Causal depthwise Conv1d, SGFX scan/update, no attention cache |

## Challenge Shape Suites

Use these suites to keep benchmark coverage broad:

| Suite | Shape IDs | Why it matters |
| --- | --- | --- |
| `image_vae_conv_islands` | `sd15_vae_decode_512`, `sdxl_vae_decode_1024`, `flux_vae_decode_1024` | Conv/norm/activation fusion and layout specialization in real image codecs |
| `image_transformer_denoisers` | `sd3_transformer_1024`, `flux_transformer_1024`, `sana_1024_linear_attention`, `sana_4k_linear_attention` | Non-LLM transformer attention with image-token grids and diffusion conditioning |
| `video_transformer_denoisers` | `cogvideox_t2v_49f_720x480`, `wan_i2v_81f_480x832`, `wan_i2v_condition_81f` | Huge sequence counts, NCTHW layout, Conv3d patchify, VAE temporal compression |
| `audio_generation` | `stable_audio_latent_1024`, `longcat_audio_5_15_30s` | 1D sequence kernels, audio codec convs, cross-attention with smaller condition length |
| `llm_dense_decode` | `llama2_7b_prefill_decode`, `llama3_8b_gqa`, `apertus_8b_long_context`, `qwen3_8b_dense`, `qwen3_4b_2507_long` | Prefill/decode, GQA, RoPE, KV cache, large-vocab logits and sampling |
| `moe_routing` | `qwen3_moe_30b_a3b`, `qwen3_moe_235b_a22b`, `mixtral_8x7b_moe` | Router softmax/top-k, token grouping, grouped expert GEMM, scatter-add |
| `vision_classic_transformers` | `vit_base_patch16_224`, `vit_large_patch32_384`, `clipseg_352_mask`, `detr_resnet50_800_1333` | Patch extraction, moderate attention, detection/segmentation heads |
| `position_scan_rope` | `allegro_3d_rope_grid`, `bloom_alibi_position_cumsum`, `qwen3_vl_cu_seqlens_mrope`, `mm_grounding_dino_special_logit` | Prefix scans, position tables, sine/cosine RoPE application, multimodal packed-sequence metadata |

## Additional API-Anchoring Shapes

These rows came from the same local report roots and are used by the PyTorch API
shape matrix when a public API needs a concrete size that was not part of the
first suite table.

| Shape ID | Source | Runtime tensors | Public API pressure |
| --- | --- | --- | --- |
| `aimv2_rmsnorm_l2` | `transformers/aimv2/report.md:187`, `:240`, `:354` | vision/text feature vectors `[B,projection_dim]`; RMSNorm hidden widths 768/1024; vision sequences 256/576/1024 and text `L <= 77` | `sum`, `mean`, `rsqrt`, `rms_norm`, `matmul`, SDPA |
| `rag_doc_scores_and_marginalize` | `transformers/rag/report.md:93`, `:121`, `:142` | `bmm([B,1,768] x [B,768,n_docs]) -> [B,n_docs]`; generator logits `[B*n_docs,T,V]`; `logsumexp` over docs | `bmm`, `log_softmax`, `logsumexp`, `gather`, `sum`, `masked_fill` |
| `superpoint_superglue_480x640` | `transformers/superglue/report.md:139`, `:143`, `:203` | pair input `[B,2,3,480,640]`; detector `[B*2,1,480,640]`; score `[B,65,H/8,W/8]`; padded keypoints `[B,2,K,256]` | `conv2d`, `relu`, `max_pool2d`, `softmax`, `nonzero`, `topk`, `grid_sample`, `logsumexp` |
| `ace_step_audio_dit_patchify` | `diffusers/ace_step/report.md:148`, `:201` | DiT concat `[B,T,192]`; Conv1d patchify `[B,2048,T/2]`; ConvTranspose1d unpatchify back to `[B,T,64]` | `cat`, `pad`, `transpose`, `conv1d`, transposed conv |
| `audioflamingo3_stft_pool_stitch` | `transformers/audioflamingo3/report.md:133`, `:211`, `:222` | STFT/log-mel chunks `[num_windows_total,128,3000]`; AvgPool1d `[W,1500,1280] -> [W,750,1280]`; audio rows `[sum(post_lengths),3584]` | `stft`, `avg_pool1d`, `nonzero`, masked/indexed scatter |
| `patchtsmixer_temporal_unfold` | `transformers/patchtsmixer/report.md:101`, `:157` | time series `[B,T,C] -> [B,C,P,patch_length]` by temporal unfold | `Tensor.unfold`, reshape/permute, distribution heads |
| `vit_mae_masked_reconstruction` | `transformers/vit_mae/report.md:117`, `:163`, `:214` | patch noise `[B,196]`; gathered visible patches; target `[B,L,P*P*C]`; per-patch MSE masked sum | `argsort`, `gather`, `mse_loss`, reductions |
| `bark_fine_codebook_sampling` | `transformers/bark/report.md:242`, `:307` | fine codebook logits over 1024 tokens; argmax or softmax plus multinomial | `argmax`, `softmax`, `multinomial` |
| `mra_sparse_32x32_blocks` | `transformers/mra/report.md:82`, `:118`, `:232` | fixed 32x32 sparse blocks with float tensors | sparse/custom extension candidate |
| `afmoe_tp_router_surface` | `transformers/afmoe/report.md:100`, `:179` | up to 256 experts, top-k 4, context 262144/sliding 4096, tensor-parallel plans | MoE routing, distributed/TP candidate |
| `allegro_3d_rope_grid` | `diffusers/allegro/report.md:306`, `:320`, `:376`, `:542` | default video token grid `22x45x80 = 79200`; 24 heads, head dim 96 split into temporal/height/width chunks of 32; RoPE grid `[3,1,79200]` | `cartesian_prod`, `arange`, `sin`, `cos`, gather, rotate-half RoPE |
| `bloom_alibi_position_cumsum` | `transformers/bloom/report.md:78`, `:89`, `:204` | attention masks `[B,S]`; position ids from `attention_mask.cumsum`; checkpoints span heads 16..112, hidden 1024..14336, vocab 250880 | `cumsum`, mask multiply/fill, expand/broadcast |
| `qwen3_vl_cu_seqlens_mrope` | `transformers/qwen3_vl/report.md:120`, `:187`, `:240`, `:247` | grid metadata `[num_items,3]`; flattened patches `[sum_items,grid_t*grid_h*grid_w,C*2*16*16]`; `cu_seqlens = cumsum(repeat_interleave(grid_h*grid_w, grid_t))`; text `position_ids` can be `[4,B,S]` | `repeat_interleave`, `cumsum`, `pad`, `searchsorted`, M-RoPE `sin/cos` |
| `aria_idefics_bucketized_positions` | `transformers/aria/report.md:118`, `:285`; `transformers/idefics2/report.md:169`, `:265`; `transformers/idefics3/report.md:205`, `:305` | Aria square inputs 490 and 980 expand to 128 and 256 image tokens; Idefics2 uses a 70x70 learned position grid; Idefics3 uses 26x26 | `bucketize`, `clamp`, `arange`, indexed position lookup, masked scatter adjacency |
| `shap_e_searchsorted_unique_render` | `diffusers/shap_e/report.md:241`, `:498`, `:502` | Shap-E prior latents `[B,1024,1024]`; renderer CDF/searchsorted/gather/random interpolation/sort; marching-cubes dynamic unique/gather/interpolate mesh path | `searchsorted`, `unique`, `sort`, `gather`, random interpolation |
| `mm_grounding_dino_special_logit` | `transformers/mm_grounding_dino/report.md:80`, `:143`, `:153`, `:170` | image preprocessing short edge 800/long edge 1333; decoder outputs logits `[B,900,256]` and boxes `[B,900,4]`; box/reference math uses `torch.special.logit(eps=1e-5)` and 2D cumsum sine positions | `torch.special.logit`, `cumsum`, `meshgrid`, `linspace`, `where`, `clamp` |
| `pe_audio_masked_groupnorm` | `transformers/pe_audio/report.md:91` | masked GroupNorm over grouped channel/time axes when padding exists; hidden sizes 768/1024/1792, heads 6/8/14, head dim 128 | `torch.masked.mean`, `torch.masked.var`, masked normalization |
| `helios_cholesky_noise_blocks` | `diffusers/helios/report.md:299`, `:519`, `:529` | pyramid stage block-correlated noise builds covariance `I*(1+gamma) - ones*gamma`, computes Cholesky, and applies per-block matmul for fixed patch size and `gamma` | `torch.linalg.cholesky`, small matrix factorization, block matmul |
| `bamba_mamba2_chunk_scan` | `transformers/bamba/report.md:77`, `:132`, `:160`, `:256`, `:341` | Bamba-9B Mamba layer uses `Linear(4096->16768)`, split sections `[8192,8448,128]`, depthwise Conv1d channels 8448 kernel 4, chunk size 256, recurrent state `[B,128,64,128]` | chunked selective scan, Conv1d, split, state update |

## First PyTorch References To Build

Each shape suite should start with a PyTorch module/function that is small
enough to run as an oracle:

- VAE conv island: `torch.nn.Conv2d`, `torch.nn.GroupNorm`, `torch.nn.SiLU`,
  nearest `interpolate`, residual add.
- Transformer row kernels: `torch.nn.functional.scaled_dot_product_attention`,
  `torch.nn.functional.layer_norm`, `torch.nn.functional.rms_norm`, matmul,
  `silu`, elementwise multiply.
- Top-K/logits: `torch.topk`, `torch.softmax`, `torch.multinomial`,
  `torch.argmax`, and exact rowwise comparison helpers.
- MoE router: `linear`, `softmax`, `topk`, normalize selected probabilities,
  grouped gather/scatter reference.
- Patchify/unpatchify: `conv2d`/`conv3d`, `view`, `permute`, `reshape`, and
  `einsum` where the source report uses it.

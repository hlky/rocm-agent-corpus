# gfx942 CDNA3 Lab

Use this lab for portable CDNA3 `gfx942` builds such as MI300-class targets.
Keep these records separate from any future suffix-specific or CDNA4 paths.

## Compile Profile

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx942 -lineinfo kernel.hip -o kernel_gfx942
hipcc --list-gpu-targets
```

Store ROCm version, GPU model, firmware/runtime metadata, library versions, and
compiler resource output with the benchmark record.

## Features To Verify

- MFMA Matrix Core paths and Composable Kernel/CK Tile collectives for the exact
  dtype and layout.
- LDS staging, software pipeline depth, barriers, and occupancy tradeoffs.
- Larger HBM and cache behavior for attention, KV-cache, and grouped-GEMM
  workloads.
- MIGraphX, vLLM on ROCm, Transformer Engine on ROCm, Triton, and framework
  baselines where relevant.

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx942 | grep -Ei "v_mfma|ds_|s_waitcnt|s_barrier"
```

## Custom-Kernel Attack Surfaces

- Fixed-shape GEMM, grouped GEMM, fused epilogues, and low-precision metadata
  movement.
- Attention prefill/decode, paged KV cache, RoPE, Top-K, and sampling.
- hipCUB/rocPRIM competitors for reductions, scans, select, sort, and histogram.
- MIGraphX or PyTorch extension kernels where a fixed model contract removes
  runtime generality.

## Gotchas

- Do not cite NVIDIA WGMMA/TMA concepts as ROCm evidence. Use AMD instruction
  names and disassembly when making architecture claims.
- Keep portable `gfx942` records separate from `gfx950` or future suffix
  records.
- Mark HIP-event-only results as `timing-only`.

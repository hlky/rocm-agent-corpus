# gfx1100 RDNA3 Lab

Use this lab for RDNA3 `gfx1100`-class devices. It is useful for inference,
memory-bound fused kernels, and cost-sensitive ROCm deployments.

## Compile Profile

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx1100 -lineinfo kernel.hip -o kernel_gfx1100
hipcc --list-gpu-targets
```

Record GPU model, ROCm version, driver/runtime metadata, clock policy, power
state, and exact library versions for every result.

## Features To Verify

- WMMA/MFMA availability and performance on the exact SKU and toolkit.
- LDS tiling, cache locality, vector width, and wave-size assumptions.
- MIGraphX, hipBLASLt, Composable Kernel, Triton, and framework-generated
  baselines for the same shape and math contract.
- Low-precision paths only after recording packing, scale layout, accumulator,
  and output dtype.

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx1100 | grep -Ei "v_mfma|v_wmma|ds_|s_waitcnt"
```

## Custom-Kernel Attack Surfaces

- RMSNorm, LayerNorm, RoPE, KV-cache update/read, dequantize, logits
  processing, Top-K, sampling, and short-row softmax.
- MIGraphX or PyTorch extension kernels that exploit fixed model shapes.
- Compact GEMM or grouped GEMM shapes where dispatch and heuristic overhead are
  visible.

## Gotchas

- RDNA3 is not "small CDNA"; remeasure tiles and occupancy.
- End-to-end graph timing can matter more than one inner kernel timing.
- Mark HIP-event-only measurements as `timing-only`.
